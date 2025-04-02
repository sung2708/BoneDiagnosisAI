import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from configs.config import MODEL_CONFIG

class MultiHeadedAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.dim = MODEL_CONFIG.dim
        self.heads = MODEL_CONFIG.heads
        self.head_dim = self.dim // self.heads

        assert self.head_dim * self.heads == self.dim, "dim must be divisible by heads"

        self.qkv = nn.Linear(self.dim, self.dim * 3)
        self.proj = nn.Linear(self.dim, self.dim)
        self.dropout = nn.Dropout(MODEL_CONFIG.drop)

    def forward(self, x):
        B, N, _ = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, self.dim)
        return self.proj(x)

class PositionWiseFeedForward(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(MODEL_CONFIG.dim, MODEL_CONFIG.dim * 4),
            nn.GELU(),
            nn.Dropout(MODEL_CONFIG.drop),
            nn.Linear(MODEL_CONFIG.dim * 4, MODEL_CONFIG.dim)
        )

    def forward(self, x):
        return self.net(x)
