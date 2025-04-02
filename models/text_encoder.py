import torch
import torch.nn as nn
from transformers import AutoModel
from ..configs import MODEL_CONFIG

class PhoBERTTextEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.phobert = AutoModel.from_pretrained("vinai/phobert-base")

        # Projection layer để phù hợp với dimension của model
        self.proj = nn.Linear(768, MODEL_CONFIG.dim) if 768 != MODEL_CONFIG.dim else nn.Identity()

    def forward(self, input_ids, attention_mask):
        outputs = self.phobert(input_ids=input_ids, attention_mask=attention_mask)
        return self.proj(outputs.last_hidden_state)
