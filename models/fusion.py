import torch
import torch.nn as nn
from Configs.config import Config

class Fusion(nn.Module):
    def __init__(self):
        super().__init__()
        # Attention layer, expecting a 512-dimensional input
        self.attention = nn.MultiheadAttention(
            embed_dim=Config.dim,  # Kích thước của embedding
            num_heads=Config.heads,
            batch_first=True
        )
        self.layer_norm = nn.LayerNorm(Config.dim)
        self.dropout = nn.Dropout(Config.drop)

        # Linear layer để giảm kích thước sau khi ghép nối các đặc trưng
        self.fc = nn.Linear(Config.dim * 2, Config.dim)  # Kích thước sau khi ghép nối sẽ là 1024, cần giảm lại thành 512

    def forward(self, image_features, text_features):
        # Đảm bảo đầu vào có đúng số chiều
        if len(image_features.shape) == 1:
            image_features = image_features.unsqueeze(0)
        if len(text_features.shape) == 1:
            text_features = text_features.unsqueeze(0)

        # Ghép nối các đặc trưng ảnh và văn bản
        combined = torch.cat([image_features, text_features], dim=-1)  # Kích thước sẽ là 1024

        # Sử dụng lớp Linear để giảm kích thước từ 1024 xuống 512
        combined = self.fc(combined)  # Kết quả sẽ có kích thước (batch_size, dim)

        combined = combined.unsqueeze(1)  # Thêm chiều thứ 2 cho chuỗi

        # Áp dụng cơ chế chú ý
        attn_output, _ = self.attention(combined, combined, combined)

        # Kết nối residual và chuẩn hóa lớp
        output = self.layer_norm(combined + self.dropout(attn_output))

        output = output.squeeze(1)  # Loại bỏ chiều chuỗi

        return output
