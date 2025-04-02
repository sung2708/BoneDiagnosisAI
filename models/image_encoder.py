import torch
import torch.nn as nn
import torchvision.models as models
from ..configs import MODEL_CONFIG

class EfficientImageEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        # Sử dụng ResNet làm backbone
        self.cnn = models.resnet50(pretrained=True)

        # Đóng băng các layer đầu
        for param in list(self.cnn.parameters())[:-4]:
            param.requires_grad = False

        # Layer cuối cùng để extract features
        self.feature_extractor = nn.Sequential(
            *list(self.cnn.children())[:-1],
            nn.Conv2d(2048, MODEL_CONFIG.dim, kernel_size=1)
        )

        # Layer xử lý mask
        self.mask_processor = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, MODEL_CONFIG.dim, kernel_size=1)
        )

    def forward(self, image, mask):
        # Extract features từ ảnh
        img_features = self.feature_extractor(image)

        # Xử lý mask
        mask_features = self.mask_processor(mask.unsqueeze(1))

        # Kết hợp features
        combined = img_features * mask_features
        return combined.flatten(2).transpose(1, 2)
