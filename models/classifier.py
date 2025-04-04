import torch.nn as nn
from .image_encoder import EfficientImageEncoder
from .text_encoder import PhoBERTTextEncoder
from .fusion import MultiHeadedAttention
from configs.config import MODEL_CONFIG
import torch

class DiseaseClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        # Các thành phần encoder
        self.image_encoder = EfficientImageEncoder()
        self.text_encoder = PhoBERTTextEncoder()
        self.fusion = MultiHeadedAttention()

        # Các lớp phân loại
        self.disease_head = nn.Sequential(
            nn.Linear(MODEL_CONFIG['dim'], MODEL_CONFIG['dim']//2),
            nn.ReLU(),
            nn.Dropout(MODEL_CONFIG['drop']),
            nn.Linear(MODEL_CONFIG['dim']//2, len(MODEL_CONFIG['class_names'])),
            nn.Softmax(dim=1)  # Xác suất cho các lớp
        )

        self.confidence_head = nn.Sequential(
            nn.Linear(MODEL_CONFIG['dim'], 1),
            nn.Sigmoid()
        )

    def forward(self, image, input_ids, attention_mask):
        # Mã hóa các đầu vào
        img_features = self.image_encoder(image)
        text_features = self.text_encoder(input_ids, attention_mask)

        # Kết hợp thông tin
        combined = torch.cat([img_features, text_features], dim=1)
        fused = self.fusion(combined)

        # Dự đoán
        disease_probs = self.disease_head(fused[:, 0])
        confidence = self.confidence_head(fused[:, 1])

        return {
            'disease_probs': disease_probs,  # Xác suất cho từng lớp
            'disease_pred': torch.argmax(disease_probs, dim=1),  # Lớp dự đoán
            'confidence': confidence.squeeze(-1)
        }
