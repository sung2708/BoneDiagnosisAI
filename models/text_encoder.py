# [file name]: text_encoder.py
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
from Configs.config import Config

class TextEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        # Load PhoBERT model and tokenizer
        self.phobert = AutoModel.from_pretrained("vinai/phobert-base")
        self.tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base")

        # Projection layer to match image feature dimension
        self.proj = nn.Sequential(
            nn.Linear(768, Config.dim),
            nn.ReLU(),
            nn.Dropout(Config.drop)
        )

        # Freeze PhoBERT layers if needed
        if Config.freeze_text_encoder:
            for param in self.phobert.parameters():
                param.requires_grad = False

    def forward(self, input_ids, attention_mask):
        outputs = self.phobert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        # Use [CLS] token representation
        pooled_output = outputs.last_hidden_state[:, 0]
        return self.proj(pooled_output)
