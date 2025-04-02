import torch
import torch.nn as nn
import torchvision.models as models
from ..configs import MODEL_CONFIG

class EfficientImageEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        # Backbone ResNet50
        self.cnn = models.resnet50(pretrained=True)

        # Đóng băng có chọn lọc
        for name, param in self.cnn.named_parameters():
            if 'layer4' not in name and 'fc' not in name:  # Chỉ fine-tune layer4 và fc
                param.requires_grad = False

        # Feature extractor tối ưu
        self.feature_extractor = nn.Sequential(
            *list(self.cnn.children())[:-2],  # Giữ lại đến layer4
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(2048, MODEL_CONFIG.dim)
        )

        # Layer chuẩn hóa
        self.norm = nn.LayerNorm(MODEL_CONFIG.dim)

    def forward(self, image, mask):
        # Image features
        img_features = self.feature_extractor(image)

        combined = self.norm(img_features)
        return combined.unsqueeze(1)  # [B, 1, dim]
