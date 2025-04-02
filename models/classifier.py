import torch.nn as nn
import torch
from .image_encoder import EfficientImageEncoder
from .text_encoder import PhoBERTTextEncoder
from .fusion import MultiHeadedAttention
from configs.config import MODEL_CONFIG

class DiseaseClassifier(nn.Module):
    """Model tổng hợp cho 3 loại bệnh chính với hỗ trợ confidence score"""
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
            nn.Linear(MODEL_CONFIG['dim']//2, 3)  # 3 loại bệnh
        )

        self.confidence_head = nn.Sequential(
            nn.Linear(MODEL_CONFIG['dim'], 1),
            nn.Sigmoid()
        )

        # Regularization
        self.dropout = nn.Dropout(MODEL_CONFIG['drop'])
        self.layer_norm = nn.LayerNorm(MODEL_CONFIG['dim'])

    def forward(self, image, input_ids, attention_mask):
        # Mã hóa các đầu vào
        img_features = self.image_encoder(image)
        text_features = self.text_encoder(input_ids, attention_mask)

        # Chuẩn hóa và kết hợp thông tin
        img_features = self.layer_norm(img_features)
        text_features = self.layer_norm(text_features)
        combined = torch.cat([img_features, text_features], dim=1)

        # Fusion với attention
        fused = self.fusion(combined)
        fused = self.dropout(fused)

        # Dự đoán
        disease_logits = self.disease_head(fused[:, 0])
        confidence = self.confidence_head(fused[:, 1])

        return {
            'disease': disease_logits,
            'confidence': confidence.squeeze(-1)
        }
