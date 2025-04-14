import os
import torch
from torch.utils.data import Dataset
from torchvision.io import read_image
from PIL import Image
import numpy as np
from pathlib import Path
from Configs.config import Config
import pandas as pd
from torchvision.transforms import functional as F

class BoneDataset(Dataset):
    def __init__(self, csv_path, image_dir, transform=None, is_train=True):
        self.df = pd.read_csv(csv_path)
        self.image_dir = Path(image_dir)
        self.transform = transform
        self.is_train = is_train
        self.target_size = (224, 224)
        self._preprocess_data()

    def _preprocess_data(self):
        # Clean data
        self.df = self.df.dropna(subset=['images_path', 'Nhóm'])

        # Standardize values
        self.df['Nhóm'] = self.df['Nhóm'].str.strip()
        self.df['Bệnh'] = self.df['Bệnh'].fillna('Khác').str.strip()
        self.df['Vị trí'] = self.df['Vị trí'].fillna('Khác').str.strip()
        self.df['Phương thức chụp'] = self.df['Phương thức chụp'].fillna('Khác').str.strip()
        self.df['Tính chất u'] = self.df['Tính chất u'].fillna('Không xác định').str.strip()

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = self.image_dir / row['images_path']

        # Load image with error handling
        try:
            try:
                image = read_image(str(img_path)).float()
            except:
                image = Image.open(img_path).convert('RGB')
                image = torch.from_numpy(np.array(image)).permute(2, 0, 1).float()

            # Resize and normalize
            image = F.resize(image, self.target_size)
            image = image / 255.0

            if self.transform:
                image = self.transform(image)

        except Exception as e:
            print(f"\n⚠️ Error loading image {img_path}: {str(e)}")
            image = torch.rand(3, *self.target_size)
            print(f"⚠️ Used random image as replacement for {img_path}")

        # Generate all questions and labels for this image
        questions = []
        question_types = []
        labels = []

        for q_type in Config.QUESTION_TYPES.keys():
            try:
                question, label = self._get_question_and_label(q_type, row)
                questions.append(question)
                question_types.append(q_type)
                labels.append(label)
            except Exception as e:
                print(f"❌ Failed to process question type {q_type} for index {idx}: {e}")
                questions.append("N/A")
                question_types.append(q_type)
                labels.append(0)

        return {
            'image': image,
            'questions': questions,
            'question_types': question_types,
            'labels': torch.tensor(labels)
        }

    def _get_question_and_label(self, q_type, row):
        if q_type == 'disease_group':
            question = "Ảnh này thuộc nhóm bệnh nào?"
            label = Config.QUESTION_TYPES[q_type].index(row['Nhóm'])

        elif q_type == 'bone_tumor_type':
            question = "Bệnh nhân mắc loại u xương nào?"
            disease = row['Bệnh']
            if disease not in Config.QUESTION_TYPES[q_type]:
                disease = 'Khác'
            label = Config.QUESTION_TYPES[q_type].index(disease)

        elif q_type == 'location':
            question = "Tổn thương nằm ở vị trí nào?"
            norm_loc = self._normalize_location(row['Vị trí'])
            if norm_loc not in Config.QUESTION_TYPES[q_type]:
                norm_loc = 'Khác'
            label = Config.QUESTION_TYPES[q_type].index(norm_loc)

        elif q_type == 'imaging_modality':
            question = "Ảnh này được chụp bằng phương pháp nào?"
            modality = row['Phương thức chụp']
            if modality not in Config.QUESTION_TYPES[q_type]:
                modality = 'Khác'
            label = Config.QUESTION_TYPES[q_type].index(modality)

        elif q_type == 'tumor_nature':
            question = "Khối u này có tính chất gì?"
            nature = row['Tính chất u']
            if nature == 'Ác':
                nature = 'Ác tính'
            elif nature == 'Lành':
                nature = 'Lành tính'
            elif nature not in Config.QUESTION_TYPES[q_type]:
                nature = 'Không xác định'
            label = Config.QUESTION_TYPES[q_type].index(nature)

        else:
            raise ValueError(f"Unknown question type: {q_type}")

        # Validate label range
        num_classes = len(Config.QUESTION_TYPES[q_type])
        if label < 0 or label >= num_classes:
            raise ValueError(f"⚠️ Invalid label for {q_type}: {label} (expected 0-{num_classes - 1})")

        return question, label

    def _normalize_location(self, location):
        if pd.isna(location):
            return 'Khác'

        location = location.lower()

        if 'đùi' in location and 'trái' in location:
            return 'Xương đùi trái'
        elif 'đùi' in location and 'phải' in location:
            return 'Xương đùi phải'
        elif 'chày' in location and 'trái' in location:
            return 'Xương chày trái'
        elif 'chày' in location and 'phải' in location:
            return 'Xương chày phải'
        elif 'mác' in location and 'trái' in location:
            return 'Xương mác trái'
        elif 'mác' in location and 'phải' in location:
            return 'Xương mác phải'

        return 'Khác'
