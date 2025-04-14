import torch
import torch.nn as nn
import os
import numpy as np
from torch.utils.data import DataLoader
from torch.optim import AdamW, lr_scheduler
from sklearn.metrics import accuracy_score, f1_score, classification_report
from tqdm import tqdm
from collections import defaultdict
from models.text_encoder import TextEncoder
from models.image_encoder import ImageEncoder
from models.fusion import Fusion
from models.classifier import Classifier
from Configs.config import Config

class VQASystem(nn.Module):
    def __init__(self):
        super().__init__()
        self.image_encoder = ImageEncoder()
        self.text_encoder = TextEncoder()
        self.fusion = Fusion()
        self.classifier = Classifier(Config.QUESTION_TYPES)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

    def forward(self, images, questions, question_types):
        batch_size = images.shape[0]
        num_questions = len(Config.QUESTION_TYPES)

        # Encode images (batch_size x dim)
        img_features = self.image_encoder(images)

        # Repeat image features for each question
        img_features = img_features.unsqueeze(1).repeat(1, num_questions, 1).view(-1, img_features.shape[-1])

        # Tokenize all questions
        inputs = self.text_encoder.tokenizer(
            questions,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=Config.max_text_len
        )
        input_ids = inputs['input_ids'].to(self.device)
        attention_mask = inputs['attention_mask'].to(self.device)

        # Encode text (num_questions*batch_size x dim)
        text_features = self.text_encoder(input_ids, attention_mask)

        # Fuse features
        fused_features = self.fusion(img_features, text_features)

        # Get predictions for each question type
        logits_list = self.classifier(fused_features, question_types)

        # Pad logits to maximum number of classes for batching
        max_classes = max(len(Config.QUESTION_TYPES[qt]) for qt in Config.QUESTION_TYPES)
        padded_logits = []
        for logits in logits_list:
            pad_size = max_classes - logits.shape[0]
            if pad_size > 0:
                logits = torch.cat([logits, torch.zeros(pad_size, device=logits.device)], dim=0)
            padded_logits.append(logits)

        logits = torch.stack(padded_logits).view(batch_size, num_questions, -1)

        return logits



    def load_pretrained_image_encoder(self, pretrained_path):
        """Load only the image encoder part from pretrained weights"""
        if not os.path.exists(pretrained_path):
            print(f"Warning: Pretrained weights not found at {pretrained_path}")
            return

        print(f"Loading pretrained weights from {pretrained_path}")
        checkpoint = torch.load(pretrained_path, map_location=self.device)

        # Handle different checkpoint formats
        if 'model_state_dict' in checkpoint:
            pretrained_dict = checkpoint['model_state_dict']
        else:
            pretrained_dict = checkpoint

        # Filter out unnecessary keys
        model_dict = self.image_encoder.state_dict()

        # 1. Filter out unnecessary keys
        pretrained_dict = {k: v for k, v in pretrained_dict.items()
                          if k in model_dict and model_dict[k].shape == v.shape}

        # 2. Overwrite entries in the existing state dict
        model_dict.update(pretrained_dict)

        # 3. Load the new state dict
        self.image_encoder.load_state_dict(model_dict)
        print("Successfully loaded pretrained image encoder weights")

    def train_model(self, train_loader, val_loader):
        Config.create_dirs()

        optimizer = AdamW([
            {'params': self.image_encoder.parameters(), 'lr': Config.img_lr},
            {'params': self.text_encoder.parameters(), 'lr': Config.text_lr},
            {'params': self.fusion.parameters()},
            {'params': self.classifier.parameters()}
        ], lr=Config.lr)

        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=3, factor=0.5)
        criterion = nn.CrossEntropyLoss()

        best_f1 = 0.0
        epochs_no_improve = 0

        for epoch in range(Config.num_epochs):
            # Training
            self.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0

            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{Config.num_epochs} [Train]")
            for batch in pbar:
                images, questions, question_types, labels = batch
                images = images.to(self.device)
                labels = labels.to(self.device)

                optimizer.zero_grad()

                # Forward pass
                logits = self(images, questions, question_types)

                # Reshape logits and labels for loss calculation
                logits = logits.view(-1, logits.shape[-1])
                loss = criterion(logits, labels)

                # Backward pass
                loss.backward()
                optimizer.step()

                # Metrics
                train_loss += loss.item()
                _, predicted = torch.max(logits.data, 1)
                train_total += labels.size(0)
                train_correct += (predicted == labels).sum().item()

                pbar.set_postfix({
                    'loss': train_loss/(pbar.n+1),
                    'acc': train_correct/train_total
                })

            # Validation
            val_metrics = self.evaluate(val_loader)
            val_f1 = val_metrics['macro_f1']

            # Save best model
            if val_f1 > best_f1:
                best_f1 = val_f1
                epochs_no_improve = 0
                self.save_model(os.path.join(Config.SAVE_DIR, 'best_model.pth'))
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= Config.patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break

            # Adjust learning rate
            scheduler.step(val_f1)

            print(f"Epoch {epoch+1} | "
                  f"Train Loss: {train_loss/len(train_loader):.4f} | "
                  f"Train Acc: {train_correct/train_total:.4f} | "
                  f"Val Acc: {val_metrics['accuracy']:.4f} | "
                  f"Val F1: {val_f1:.4f}")

    def evaluate(self, data_loader):
        self.eval()
        all_preds = []
        all_labels = []
        question_types = []

        with torch.no_grad():
            for batch in tqdm(data_loader, desc="Evaluating"):
                images, questions, q_types, labels = batch
                images = images.to(self.device)
                labels = labels.cpu().numpy()

                logits = self(images, questions, q_types)
                _, preds = torch.max(logits, -1)

                all_preds.extend(preds.cpu().numpy().flatten())
                all_labels.extend(labels.flatten())
                question_types.extend(q_types)

        # Calculate metrics per question type
        type_metrics = defaultdict(dict)
        for q_type in Config.QUESTION_TYPES.keys():
            idx = [i for i, t in enumerate(question_types) if t == q_type]
            if not idx:  # Skip if no samples for this question type
                continue

            type_labels = [all_labels[i] for i in idx]
            type_preds = [all_preds[i] for i in idx]
            target_names = Config.QUESTION_TYPES[q_type]

            # Ensure we only include classes that appear in the data
            present_labels = set(type_labels)
            filtered_target_names = [name for i, name in enumerate(target_names) if i in present_labels]

            type_metrics[q_type]['accuracy'] = accuracy_score(type_labels, type_preds)
            type_metrics[q_type]['f1'] = f1_score(type_labels, type_preds, average='weighted')

            # Generate classification report only for present classes
            type_metrics[q_type]['report'] = classification_report(
                type_labels,
                type_preds,
                labels=list(present_labels),
                target_names=filtered_target_names,
                zero_division=0
            )

        # Overall metrics
        accuracy = accuracy_score(all_labels, all_preds)
        macro_f1 = f1_score(all_labels, all_preds, average='macro')

        return {
            'accuracy': accuracy,
            'macro_f1': macro_f1,
            'type_metrics': dict(type_metrics),
            'report': "\n".join([f"{q_type}:\n{metrics['report']}"
                                for q_type, metrics in type_metrics.items()])
        }

    def save_model(self, path, optimizer=None, epoch=None, best_accuracy=None):
        checkpoint = {
            'model_state_dict': self.state_dict()
        }
        if optimizer is not None:
            checkpoint['optimizer_state_dict'] = optimizer.state_dict()
        if epoch is not None:
            checkpoint['epoch'] = epoch
        if best_accuracy is not None:
            checkpoint['best_accuracy'] = best_accuracy

        torch.save(checkpoint, path)


    def load_model(self, path, optimizer=None):
        checkpoint = torch.load(path, map_location=self.device)
        self.load_state_dict(checkpoint['model_state_dict'])
        if optimizer is not None and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        return checkpoint.get('epoch'), checkpoint.get('best_accuracy')

