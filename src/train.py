import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from tqdm import tqdm
from PIL import Image
import numpy as np
from models.classifier import DiseaseClassifier
from configs import MODEL_CONFIG, DATA_PATHS

# Tokenizer
tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base-v2")

# Transform ảnh
image_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

class MedicalVQADataset(Dataset):
    """Dataset cho VQA y tế với multi-question và confidence"""
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
            max_length=MODEL_CONFIG.max_len,
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
            'confidence': confidence
        }

def train():
    # Cấu hình
    config = MODEL_CONFIG
    os.makedirs(config.save_dir, exist_ok=True)

    # Dataset từ file JSON label
    train_dataset = MedicalVQADataset(
        label_path=DATA_PATHS['label_path'],
        image_dir=DATA_PATHS['image_dir']
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True if torch.cuda.is_available() else False
    )

    # Model
    model = DiseaseClassifier().to(config.device)

    # Optimizer và scheduler
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.lr,
        weight_decay=0.01
    )
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=len(train_loader)//3,
        num_training_steps=len(train_loader)*config.epochs
    )

    # Loss function với trọng số confidence
    def weighted_loss(logits, labels, confidence):
        ce_loss = nn.CrossEntropyLoss(reduction='none')(logits, labels)
        return (ce_loss * confidence).mean()

    # Training loop
    best_loss = float('inf')
    for epoch in range(config.epochs):
        model.train()
        epoch_loss = 0.0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.epochs}"):
            # Chuyển dữ liệu sang device
            images = batch['image'].to(config.device, non_blocking=True)
            input_ids = batch['input_ids'].to(config.device, non_blocking=True)
            attention_mask = batch['attention_mask'].to(config.device, non_blocking=True)
            labels = batch['disease_label'].to(config.device, non_blocking=True)
            confidence = batch['confidence'].to(config.device, non_blocking=True)

            # Forward
            outputs = model(images, input_ids, attention_mask)

            # Loss
            loss = weighted_loss(outputs['disease'], labels, confidence)

            # Backward
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            epoch_loss += loss.item()

        # Đánh giá và lưu model
        avg_loss = epoch_loss / len(train_loader)
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), os.path.join(config.save_dir, "best_model.pt"))

        print(f"Epoch {epoch+1}: Loss = {avg_loss:.4f} | Best Loss = {best_loss:.4f}")

if __name__ == "__main__":
    train()
