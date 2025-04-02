import torch
import torch.nn as nn
import torchvision.models as models
from configs.config import MODEL_CONFIG

class EfficientImageEncoder(nn.Module):
    def __init__(self):
        super().__init__()

        # 1. Load pretrained ResNet50 with updated weights parameter
        self.cnn = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)  # Using newer weights

        # 2. More selective freezing - only unfreeze last two blocks
        for name, param in self.cnn.named_parameters():
            if not name.startswith('layer3') and not name.startswith('layer4'):
                param.requires_grad = False

        # 3. Improved feature extractor
        self.feature_extractor = nn.Sequential(
            # Remove the original fully connected layer
            *list(self.cnn.children())[:-2],
            # Better pooling strategy
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            # Add dropout for regularization
            nn.Dropout(p=MODEL_CONFIG.get('drop', 0.1)),
            # Project to desired dimension
            nn.Linear(2048, MODEL_CONFIG['dim']),
            # Add activation function
            nn.GELU()
        )

        # 4. Enhanced normalization
        self.norm = nn.LayerNorm(MODEL_CONFIG['dim'])

        # 5. Initialize weights properly
        self._init_weights()

    def _init_weights(self):
        """Initialize weights for the linear layer"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

    def forward(self, image, mask=None):  # Make mask optional
        """
        Args:
            image: input image tensor [B, 3, H, W]
            mask: optional mask tensor [B, 1, H, W]
        Returns:
            features: extracted features [B, 1, dim]
        """
        # Extract features
        img_features = self.feature_extractor(image)

        # Apply normalization
        features = self.norm(img_features)

        # Add singleton dimension
        return features.unsqueeze(1)
