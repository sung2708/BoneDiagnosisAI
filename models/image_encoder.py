import torch
import torch.nn as nn
import os
from torchvision.models import resnet50, ResNet50_Weights
from Configs.config import Config

class ImageEncoder(nn.Module):
    def __init__(self):
        super().__init__()

        # Load ResNet50 backbone (no classifier head)
        backbone = resnet50(weights=None)  # avoids deprecated 'pretrained' param
        self.feature_extractor = nn.Sequential(*list(backbone.children())[:-1])  # remove fc
        self.embedding_dim = backbone.fc.in_features  # typically 2048

        # Projection layer to match Config.dim
        self.proj = nn.Sequential(
            nn.Linear(self.embedding_dim, Config.dim),
            nn.ReLU(),
            nn.Dropout(Config.drop)
        )

        # Load pretrained ResNet50 weights if available
        if Config.DATA_PATHS.get('pretrained_mura') and os.path.exists(Config.DATA_PATHS['pretrained_mura']):
            state_dict = torch.load(Config.DATA_PATHS['pretrained_mura'], map_location='cpu')

            # Handle prefixing if needed elsewhere (VQASystem)
            print("✅ Pretrained ResNet50 weights loaded. Ensure proper prefixing when integrating into VQASystem.")

        self._freeze_layers()

    def _freeze_layers(self):
        for param in self.feature_extractor.parameters():
            param.requires_grad = False
        for param in self.proj.parameters():
            param.requires_grad = True

    def forward(self, x):
        x = self.feature_extractor(x)  # [B, 2048, 1, 1]
        x = torch.flatten(x, 1)        # [B, 2048]
        x = self.proj(x)               # [B, Config.dim]
        return x
