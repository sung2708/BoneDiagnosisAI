from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
import os
import json
import re
import collections
import cv2
from PIL import Image
import torchvision.transforms as transforms
from transformers import AutoModel, AutoTokenizer
from random import choice
from torch.utils.data import Dataset, DataLoader
from ..configs import MODEL_CONFIG
from ..models.base_model import mod_model1, mod_model2, mod_model3, mod_yn_model1

# Define models
batch_size = MODEL_CONFIG.batch_size


# Initialize PhoBERT
phobert = AutoModel.from_pretrained("vinai/phobert-base-v2")
tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base-v2")

# Image and Mask transformations
image_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

mask_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

class MedicalDataset(Dataset):
    def __init__(self, image_paths, questions, answers, label_dir, image_dir):
        self.image_paths = image_paths
        self.questions = questions
        self.answers = answers
        self.label_dir = label_dir
        self.image_dir = image_dir

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Load image
        img_path = os.path.join(self.image_dir, self.image_paths[idx])
        image = Image.open(img_path).convert('RGB')

        # Load mask from JSON label
        label_path = os.path.join(self.label_dir, os.path.splitext(self.image_paths[idx])[0] + '.json')
        with open(label_path, 'r') as f:
            label_data = json.load(f)
        mask = self.generate_mask(label_data, image.size)

        # Apply transforms
        image_tensor = image_transform(image)
        mask_tensor = mask_transform(Image.fromarray(mask))

        # Tokenize question with PhoBERT
        question = self.questions[idx]
        inputs = tokenizer(question, return_tensors="pt", padding='max_length',
                         max_length=MODEL_CONFIG.max_len, truncation=True)

        # Convert answer to tensor
        answer = torch.tensor(self.answers[idx])

        return {
            'image': image_tensor,
            'mask': mask_tensor,
            'input_ids': inputs['input_ids'].squeeze(0),
            'attention_mask': inputs['attention_mask'].squeeze(0),
            'answer': answer
        }

    def generate_mask(self, label_data, img_size):
        """Generate binary mask from JSON annotations"""
        mask = np.zeros((img_size[1], img_size[0]), dtype=np.uint8)

        for shape in label_data['shapes']:
            points = np.array(shape['points'], dtype=np.int32)

            if shape['shape_type'] == 'rectangle':
                cv2.rectangle(mask, tuple(points[0]), tuple(points[1]), 1, -1)
            elif shape['shape_type'] == 'polygon':
                cv2.fillPoly(mask, [points], 1)

        return mask

def ques_standard(text):
    """Standardize Vietnamese questions"""
    temp = text.strip('?').split(' ')
    temp_list = []
    for i in range(len(temp)):
        if temp[i] != '':
            temp[i] = re.sub("[\s+\.\!\/_,$%^*(+\"\')]+|[+——()?【】“”！，。？、~@#￥%……&*（）]+ ", "", temp[i].lower())
            temp_list.append(temp[i].replace('-',' '))
    return ' '.join(temp_list)

def extract_data(file, start, end):
    """Load data from | separated file"""
    imag, ques, answ = [], [], []
    with open(file, 'r', encoding='utf-8') as f:
        lines = f.readlines()[start:end]
        for line in lines:
            parts = line.strip().split('|')
            imag.append(parts[0])
            ques.append(ques_standard(parts[-2]))
            answ.append(parts[-1])
    return imag, ques, answ

def clsf_data(start1, start2, end1, end2, mode='class'):
    """Classify data into yes/no and general answers"""
    _imag1, _ques1, _answ1 = extract_data(train_text_file, start1, end1)
    _imag2, _ques2, _answ2 = extract_data(valid_text_file, start2, end2)
    _imag, _ques, _answ = _imag1+_imag2, _ques1+_ques2, _answ1+_answ2

    yn_imag, yn_ques, yn_answ = [], [], []
    ge_imag, ge_ques, ge_answ = [], [], []

    for i in range(len(_answ)):
        if _answ[i].lower() in ['có', 'không', 'yes', 'no']:
            yn_imag.append(_imag[i])
            yn_ques.append(_ques[i])
            yn_answ.append(_answ[i])
        else:
            ge_imag.append(_imag[i])
            ge_ques.append(_ques[i])
            ge_answ.append(_answ[i])

    # Create answer dictionaries
    yn_list = sorted(list(set(yn_answ)))
    yn_dict = {yn_list[i]: i for i in range(len(yn_list))}
    yn_answ = [yn_dict[ans] for ans in yn_answ]

    ge_list = sorted(list(set(ge_answ)))
    ge_dict = {ge_list[i]: i for i in range(len(ge_list))}
    ge_answ = [ge_dict[ans] for ans in ge_answ]

    if mode == 'yn':
        return yn_imag, yn_ques, yn_answ, yn_dict
    else:
        return ge_imag, ge_ques, ge_answ, ge_dict

def data_loader(imag, ques, answ, batch_size, image_dir, label_dir):
    """Create DataLoader with image and mask support"""
    dataset = MedicalDataset(imag, ques, answ, label_dir, image_dir)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)

def train_model(model, imag_t, ques_t, answ_t, imag_v, ques_v, answ_v,
               image_dir, label_dir, log_name, threshold, yn_mode, dict_op):
    """Train model with image + mask"""
    train_loader = data_loader(imag_t, ques_t, answ_t, batch_size, image_dir, label_dir)
    valid_loader = data_loader(imag_v, ques_v, answ_v, batch_size, image_dir, label_dir)

    optimizer = optim.Adam(model.parameters(), lr=MODEL_CONFIG.lr)
    criterion = nn.CrossEntropyLoss()
    scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=3)

    best_loss = float('inf')
    for epoch in range(MODEL_CONFIG.epochs):
        # Training
        model.train()
        train_loss = 0
        for batch in tqdm(train_loader, desc=f'Epoch {epoch+1}'):
            optimizer.zero_grad()

            outputs = model(
                image=batch['image'].to(MODEL_CONFIG.device),
                mask=batch['mask'].to(MODEL_CONFIG.device),
                input_ids=batch['input_ids'].to(MODEL_CONFIG.device),
                attention_mask=batch['attention_mask'].to(MODEL_CONFIG.device)
            )

            loss = criterion(outputs, batch['answer'].to(MODEL_CONFIG.device))
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # Validation
        model.eval()
        valid_loss = 0
        with torch.no_grad():
            for batch in valid_loader:
                outputs = model(
                    image=batch['image'].to(MODEL_CONFIG.device),
                    mask=batch['mask'].to(MODEL_CONFIG.device),
                    input_ids=batch['input_ids'].to(MODEL_CONFIG.device),
                    attention_mask=batch['attention_mask'].to(MODEL_CONFIG.device)
                )
                valid_loss += criterion(outputs, batch['answer'].to(MODEL_CONFIG.device)).item()

        avg_valid_loss = valid_loss / len(valid_loader)
        scheduler.step(avg_valid_loss)

        # Save best model
        if avg_valid_loss < best_loss and avg_valid_loss < threshold:
            best_loss = avg_valid_loss
            torch.save(model.state_dict(), os.path.join(save_dir, f'{log_name}_best.pt'))

    return best_loss

# Main execution
if __name__ == "__main__":
    # Configuration
    train_text_file = "path/to/train.txt"
    valid_text_file = "path/to/valid.txt"
    test_text_file = "path/to/test.txt"
    label_dir = "path/to/labels/"
    image_dir = "path/to/images/"
    save_dir = "path/to/save/"
    os.makedirs(save_dir, exist_ok=True)

    # Load and prepare data
    mod_imag, mod_ques, mod_answ, mod_dict = clsf_data(0, 0, 2, 2)
    mod_yn_imag, mod_yn_ques, mod_yn_answ, mod_yn_dict = clsf_data(0, 0, 2, 2, mode='yn')

    # Train models
    mod_loss = train_model(
        mod_model1().to(MODEL_CONFIG.device),
        mod_imag, mod_ques, mod_answ,
        mod_imag, mod_ques, mod_answ,
        image_dir, label_dir,
        'mod_model', 2.0, False, mod_dict
    )

    mod_yn_loss = train_model(
        mod_yn_model1().to(MODEL_CONFIG.device),
        mod_yn_imag, mod_yn_ques, mod_yn_answ,
        mod_yn_imag, mod_yn_ques, mod_yn_answ,
        image_dir, label_dir,
        'mod_yn_model', 2.0, True, mod_yn_dict
    )
