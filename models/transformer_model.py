import torch
import torch.nn as nn
from .image_encoder import EfficientImageEncoder
from .text_encoder import PhoBERTTextEncoder
from .fusion import MultiHeadedSelfAttention
from ..configs import MODEL_CONFIG

class DiseaseClassifier(nn.Module):
    """Model chính cho 3 loại bệnh"""
    def __init__(self, num_classes=3):
        super().__init__()
        self.image_encoder = EfficientImageEncoder()
        self.text_encoder = PhoBERTTextEncoder()
        self.fusion_attention = MultiHeadedSelfAttention()

        # Classifiers cho 3 loại bệnh
        self.disease_classifier = nn.Linear(MODEL_CONFIG.dim, num_classes)
        self.position_classifier = nn.Linear(MODEL_CONFIG.dim, 6)  # 6 vị trí

    def forward(self, image, mask, input_ids, attention_mask):
        # Encode các modality
        img_features = self.image_encoder(image, mask)
        text_features = self.text_encoder(input_ids, attention_mask)

        # Kết hợp thông tin
        combined = torch.cat([img_features, text_features], dim=1)
        fused = self.fusion_attention(combined)

        # Dự đoán
        disease_logits = self.disease_classifier(fused[:, 0])  # Lấy [CLS] token
        position_logits = self.position_classifier(fused[:, 1])  # Vị trí

        return {
            'disease': disease_logits,
            'position': position_logits
        }
