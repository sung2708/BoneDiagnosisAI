import torch
import torch.nn as nn
from transformers import AutoModel
from configs.config import MODEL_CONFIG

class PhoBERTTextEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.phobert = AutoModel.from_pretrained("vinai/phobert-base-v2")

        # Projection layer với khởi tạo tốt hơn
        self.proj = nn.Sequential(
            nn.Linear(768, MODEL_CONFIG.dim),
            nn.LayerNorm(MODEL_CONFIG.dim),
            nn.GELU()
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.phobert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        # Lấy [CLS] token
        return self.proj(outputs.last_hidden_state[:, 0]).unsqueeze(1)  # [B, 1, dim]
