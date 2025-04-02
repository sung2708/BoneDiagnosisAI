import torch.nn as nn
import torch
from .image_encoder import EfficientImageEncoder
from .text_encoder import PhoBERTTextEncoder
from .fusion import MultiHeadedAttention
from ..configs import MODEL_CONFIG

class DiseaseClassifier(nn.Module):
    """Model tổng hợp cho 3 loại bệnh chính với hỗ trợ confidence score"""
    def __init__(self):
        super().__init__()
        # Các thành phần encoder
        self.image_encoder = EfficientImageEncoder()
        self.text_encoder = PhoBERTTextEncoder()
        self.fusion = MultiHeadedAttention()

        # Các lớp phân loại
        self.disease_head = nn.Linear(MODEL_CONFIG.dim, 3)  # 3 loại bệnh
        self.confidence_head = nn.Linear(MODEL_CONFIG.dim, 1)  # Dự đoán confidence

        # Dropout
        self.dropout = nn.Dropout(MODEL_CONFIG.drop)

    def forward(self, image, input_ids, attention_mask):
        # Mã hóa các đầu vào (bỏ mask nếu không dùng)
        img_features = self.image_encoder(image)
        text_features = self.text_encoder(input_ids, attention_mask)

        # Kết hợp thông tin
        combined = torch.cat([img_features, text_features], dim=1)
        fused = self.fusion(combined)

        # Dự đoán
        disease_logits = self.disease_head(self.dropout(fused[:, 0]))  # [CLS] token
        confidence = torch.sigmoid(self.confidence_head(self.dropout(fused[:, 1])))  # Confidence 0-1

        return {
            'disease': disease_logits,
            'confidence': confidence.squeeze(-1)
        }
