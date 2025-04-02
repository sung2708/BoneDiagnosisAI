import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from tqdm import tqdm
from PIL import Image
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from models.classifier import DiseaseClassifier
from configs.config import MODEL_CONFIG, DATA_PATHS

# Tokenizer
tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base-v2")

# Transform ảnh
image_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

class MedicalVQADataset(Dataset):
    """Dataset cho bài toán classification bệnh"""
    def __init__(self, label_path, image_dir):
        self.image_dir = image_dir
        with open(label_path) as f:
            self.data = json.load(f)

        # Ánh xạ nhãn bệnh
        self.disease_map = {
            'u_xuong': 0,
            'viem_nhiem': 1,
            'chan_thuong': 2
        }

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        # Load ảnh
        image = Image.open(os.path.join(self.image_dir, os.path.basename(item['image_path']))).convert('RGB')
        image_tensor = image_transform(image)

        # Tokenize câu hỏi
        inputs = tokenizer(
            item['question'],
            return_tensors="pt",
            padding='max_length',
            max_length=MODEL_CONFIG['max_len'],
            truncation=True
        )

        # Nhãn và confidence
        disease_label = torch.tensor(self.disease_map[item['label']])
        confidence = torch.tensor(float(item['confidence']))

        return {
            'image': image_tensor,
            'input_ids': inputs['input_ids'].squeeze(0),
            'attention_mask': inputs['attention_mask'].squeeze(0),
            'disease_label': disease_label,
            'confidence': confidence,
            'question_id': item['question_id'],
            'answer': item['answer']  # Thêm answer để tính bleu score
        }

def weighted_loss(logits, labels, confidence):
    ce_loss = nn.CrossEntropyLoss(reduction='none')(logits, labels)
    return (ce_loss * confidence).mean()

def compute_bleu_score(predictions, references):
    # Đơn giản hóa tính bleu score, có thể thay thế bằng implementation chi tiết hơn nếu cần
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    smoothie = SmoothingFunction().method4
    scores = []
    for pred, ref in zip(predictions, references):
        pred_tokens = pred.split()
        ref_tokens = ref.split()
        scores.append(sentence_bleu([ref_tokens], pred_tokens, smoothing_function=smoothie))
    return np.mean(scores)

def evaluate_model(model, dataloader, device, disease_map_inv):
    model.eval()
    all_preds = []
    all_labels = []
    all_answers = []
    all_pred_answers = []

    with torch.no_grad():
        for batch in dataloader:
            images = batch['image'].to(device)
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['disease_label'].to(device)

            outputs = model(images, input_ids, attention_mask)
            _, preds = torch.max(outputs['disease'], 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_answers.extend(batch['answer'])
            # Giả sử model có thể trả về answer dự đoán, nếu không thì sử dụng answer mặc định
            all_pred_answers.extend([disease_map_inv[pred.item()] for pred in preds])  # Đơn giản hóa

    # Tính các metrics
    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='weighted')
    bleu = compute_bleu_score(all_pred_answers, all_answers)

    results = {
        'accuracy': accuracy,
        'f1': f1,
        'bleu': bleu,
        'predictions': list(zip(all_preds, all_labels, all_answers, all_pred_answers))
    }

    return results

def train():
    # Cấu hình
    config = MODEL_CONFIG
    os.makedirs(DATA_PATHS['save_dir'], exist_ok=True)
    os.makedirs(DATA_PATHS['eval_dir'], exist_ok=True)

    # Tạo dataset trước
    full_dataset = MedicalVQADataset(
        label_path=DATA_PATHS['label_path'],
        image_dir=DATA_PATHS['image_dir']
    )

    # Sau đó mới tạo ánh xạ ngược từ index sang tên bệnh
    disease_map_inv = {v: k for k, v in full_dataset.disease_map.items()}

    # Phần còn lại của hàm train() giữ nguyên...
    train_size = int(config['train_ratio'] * len(full_dataset))
    val_size = int(config['val_ratio'] * len(full_dataset))
    test_size = len(full_dataset) - train_size - val_size

    train_dataset, val_dataset, test_dataset = random_split(
        full_dataset,
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )

    # Tạo DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    # Model
    model = DiseaseClassifier().to(config['device'])

    # Optimizer và scheduler
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config['lr'],
        weight_decay=0.01
    )
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=len(train_loader) // 3,
        num_training_steps=len(train_loader) * config['epochs']
    )

    # Training loop
    best_val_f1 = 0.0
    for epoch in range(config['epochs']):
        model.train()
        epoch_loss = 0.0

        # Training phase
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['epochs']}"):
            images = batch['image'].to(config['device'], non_blocking=True)
            input_ids = batch['input_ids'].to(config['device'], non_blocking=True)
            attention_mask = batch['attention_mask'].to(config['device'], non_blocking=True)
            labels = batch['disease_label'].to(config['device'], non_blocking=True)
            confidence = batch['confidence'].to(config['device'], non_blocking=True)

            # Forward
            outputs = model(images, input_ids, attention_mask)
            loss = weighted_loss(outputs['disease'], labels, confidence)

            # Backward
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            epoch_loss += loss.item()

        # Validation phase
        val_results = evaluate_model(model, val_loader, config['device'], disease_map_inv)
        avg_train_loss = epoch_loss / len(train_loader)

        print(f"\nEpoch {epoch+1}/{config['epochs']}:")
        print(f"Train Loss: {avg_train_loss:.4f}")
        print(f"Val Accuracy: {val_results['accuracy']:.4f}")
        print(f"Val F1 Score: {val_results['f1']:.4f}")
        print(f"Val BLEU Score: {val_results['bleu']:.4f}")

        # Lưu model tốt nhất dựa trên F1 score
        if val_results['f1'] > best_val_f1:
            best_val_f1 = val_results['f1']
            torch.save(model.state_dict(), os.path.join(DATA_PATHS['save_dir'], "best_model.pt"))
            print("Saved best model based on validation F1 score!")

    # Đánh giá trên test set
    print("\nEvaluating on test set...")
    test_results = evaluate_model(model, test_loader, config['device'], disease_map_inv)

    print("\nTest Results:")
    print(f"Accuracy: {test_results['accuracy']:.4f}")
    print(f"F1 Score: {test_results['f1']:.4f}")
    print(f"BLEU Score: {test_results['bleu']:.4f}")

    # Lưu kết quả evaluation
    with open(os.path.join(DATA_PATHS['eval_dir'], 'test_results.json'), 'w') as f:
        json.dump({
            'accuracy': test_results['accuracy'],
            'f1': test_results['f1'],
            'bleu': test_results['bleu']
        }, f, indent=4)

if __name__ == "__main__":
    train()
