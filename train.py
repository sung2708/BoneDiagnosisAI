import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, random_split, WeightedRandomSampler
from torchvision import transforms
from transformers import AutoTokenizer
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from PIL import Image
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import warnings
from collections import defaultdict
from torch.nn.functional import cross_entropy

# Suppress warnings
warnings.filterwarnings('ignore')

# Load config
from configs.config import DATA_PATHS, MODEL_CONFIG, TRAINING_CONFIG

# Initialize tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_CONFIG['text_model'])

class BoneDiseaseDataset:
    def __init__(self, root_dir, oversample_minority=True):
        # Xác định 3 lớp bệnh cố định
        self.classes = ["U_xuong", "Viem_nhiem", "Chan_thuong"]
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        self.samples = []
        self.labels = []

        # Load data từ các thư mục con tương ứng
        for class_name in self.classes:
            class_dir = os.path.join(root_dir, class_name)
            if not os.path.exists(class_dir):
                print(f"Warning: Class directory {class_dir} not found!")
                continue

            images = [f for f in os.listdir(class_dir)
                     if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            if not images:
                print(f"Warning: No images found in {class_dir}")
                continue

            for img_name in images:
                self.samples.append({
                    'image_path': os.path.join(class_dir, img_name),
                    'label': self.class_to_idx[class_name],
                    'question': self.generate_question(class_name),
                    'answer': class_name
                })
                self.labels.append(self.class_to_idx[class_name])

        # Cân bằng dữ liệu nếu cần
        if oversample_minority and len(self.labels) > 0:
            self._balance_dataset()

        if len(self.samples) == 0:
            raise ValueError("No valid samples found in dataset directory!")

        print("\nDataset Statistics:")
        for class_name, class_idx in self.class_to_idx.items():
            count = sum(1 for label in self.labels if label == class_idx)
            print(f"{class_name}: {count} samples")

    def generate_question(self, class_name):
        """Tạo câu hỏi phù hợp với từng loại bệnh"""
        question_templates = {
            "U_xuong": ["Có dấu hiệu u xương trên ảnh không?", "Phim X-quang có khối u xương không?"],
            "Viem_nhiem": ["Có dấu hiệu viêm nhiễm xương không?", "Xương có bị nhiễm trùng không?"],
            "Chan_thuong": ["Có chấn thương xương nào không?", "Xương có bị gãy hoặc tổn thương không?"]
        }
        return np.random.choice(question_templates[class_name])

    def _balance_dataset(self):
        label_counts = np.bincount(self.labels)
        max_count = max(label_counts)

        for class_idx in range(len(self.classes)):
            if label_counts[class_idx] < max_count:
                class_samples = [s for s in self.samples if s['label'] == class_idx]
                num_to_add = max_count - label_counts[class_idx]
                if class_samples:
                    new_samples = np.random.choice(class_samples, size=num_to_add, replace=True)
                    self.samples.extend(new_samples.tolist())
                    self.labels.extend([class_idx] * num_to_add)
                    print(f"Oversampled {self.classes[class_idx]} by adding {num_to_add} samples")

        self.labels = np.array(self.labels)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        try:
            image = Image.open(sample['image_path']).convert('RGB')
        except Exception as e:
            print(f"Error loading image {sample['image_path']}: {str(e)}")
            return None

        transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.Resize((MODEL_CONFIG['image_size'],)*2),
            transforms.ToTensor(),
            transforms.Normalize(MODEL_CONFIG['image_mean'], MODEL_CONFIG['image_std'])
        ])

        image = transform(image)

        question = tokenizer(
            sample['question'],
            max_length=MODEL_CONFIG['max_len'],
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        return {
            'image': image,
            'input_ids': question['input_ids'].squeeze(0),
            'attention_mask': question['attention_mask'].squeeze(0),
            'label': torch.tensor(sample['label']),
            'answer': sample['answer'],
            'question': sample['question']
        }

def train():
    # Fix random seed để đảm bảo tái lập kết quả
    torch.manual_seed(42)
    np.random.seed(42)

    # Initialize dataset
    full_dataset = BoneDiseaseDataset(
        DATA_PATHS['image_dir'],
        oversample_minority=TRAINING_CONFIG['oversample_minority']
    )

    # Chia tập train/val theo tỉ lệ 80/20
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(
        full_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)  # Cố định random split
    )

    # Tạo sampler để cân bằng dữ liệu trong quá trình training
    train_labels = [full_dataset.samples[i]['label'] for i in train_dataset.indices]
    class_counts = np.bincount(train_labels)
    class_weights = 1. / (class_counts + 1e-6)
    samples_weights = class_weights[train_labels]

    sampler = WeightedRandomSampler(
        weights=samples_weights,
        num_samples=len(samples_weights),
        replacement=True
    )

    # DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=TRAINING_CONFIG['batch_size'],
        sampler=sampler,
        num_workers=4,
        pin_memory=True,
        drop_last=True  # Tránh batch cuối không đủ size
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=TRAINING_CONFIG['batch_size'],
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    # Initialize model
    from models.classifier import DiseaseClassifier
    model = DiseaseClassifier().to(MODEL_CONFIG['device'])

    # Optimizer với weight decay để tránh overfitting
    optimizer = optim.AdamW(
        model.parameters(),
        lr=TRAINING_CONFIG['learning_rate'],
        weight_decay=0.01
    )

    # Scheduler để điều chỉnh learning rate
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='max',
        factor=0.1,
        patience=3,
        verbose=True
    )

    # Lưu lại lịch sử training
    history = defaultdict(list)
    best_f1 = 0

    # Training loop
    for epoch in range(TRAINING_CONFIG['epochs']):
        model.train()
        epoch_loss = 0
        train_preds = []
        train_labels = []

        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{TRAINING_CONFIG['epochs']}"):
            if batch is None:
                continue

            optimizer.zero_grad()

            inputs = {
                'image': batch['image'].to(MODEL_CONFIG['device']),
                'input_ids': batch['input_ids'].to(MODEL_CONFIG['device']),
                'attention_mask': batch['attention_mask'].to(MODEL_CONFIG['device'])
            }
            labels = batch['label'].to(MODEL_CONFIG['device'])

            outputs = model(**inputs)
            loss = cross_entropy(outputs['disease_probs'], labels)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            train_preds.extend(outputs['disease_pred'].cpu().numpy())
            train_labels.extend(labels.cpu().numpy())

        # Validation
        model.eval()
        val_preds = []
        val_labels = []
        val_loss = 0

        with torch.no_grad():
            for batch in val_loader:
                if batch is None:
                    continue

                inputs = {
                    'image': batch['image'].to(MODEL_CONFIG['device']),
                    'input_ids': batch['input_ids'].to(MODEL_CONFIG['device']),
                    'attention_mask': batch['attention_mask'].to(MODEL_CONFIG['device'])
                }
                labels = batch['label'].to(MODEL_CONFIG['device'])

                outputs = model(**inputs)
                loss = cross_entropy(outputs['disease_probs'], labels)
                val_loss += loss.item()
                val_preds.extend(outputs['disease_pred'].cpu().numpy())
                val_labels.extend(labels.cpu().numpy())

        # Tính metrics
        train_acc = accuracy_score(train_labels, train_preds)
        train_f1 = f1_score(train_labels, train_preds, average='weighted')
        val_acc = accuracy_score(val_labels, val_preds)
        val_f1 = f1_score(val_labels, val_preds, average='weighted')

        # Lưu lịch sử
        history['train_loss'].append(epoch_loss/len(train_loader))
        history['val_loss'].append(val_loss/len(val_loader))
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        history['train_f1'].append(train_f1)
        history['val_f1'].append(val_f1)

        # Save best model
        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save(model.state_dict(), os.path.join(DATA_PATHS['save_dir'], 'best_model.pt'))
            print(f"\nSaved new best model with Val F1: {best_f1:.4f}")

        # Print epoch summary
        print(f"\nEpoch {epoch+1}/{TRAINING_CONFIG['epochs']}:")
        print(f"Train Loss: {history['train_loss'][-1]:.4f} | Val Loss: {history['val_loss'][-1]:.4f}")
        print(f"Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")
        print(f"Train F1: {train_f1:.4f} | Val F1: {val_f1:.4f}")

        # Điều chỉnh learning rate
        scheduler.step(val_f1)

    # Save training history
    os.makedirs(DATA_PATHS['eval_dir'], exist_ok=True)

    # Vẽ đồ thị
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train')
    plt.plot(history['val_loss'], label='Validation')
    plt.title('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history['train_f1'], label='Train')
    plt.plot(history['val_f1'], label='Validation')
    plt.title('F1 Score')
    plt.legend()

    plt.savefig(os.path.join(DATA_PATHS['eval_dir'], 'training_metrics.png'))
    plt.close()

    print("\nTraining completed!")
    print(f"Best Validation F1: {best_f1:.4f}")

if __name__ == "__main__":
    os.makedirs(DATA_PATHS['save_dir'], exist_ok=True)
    train()
