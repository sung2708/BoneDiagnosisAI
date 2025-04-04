import torch.nn as nn
from transformers import AutoModel
from configs.config import MODEL_CONFIG

class PhoBERTTextEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = AutoModel.from_pretrained(MODEL_CONFIG['text_model'])
        self.proj = nn.Linear(768, MODEL_CONFIG['dim'])
        self.norm = nn.LayerNorm(MODEL_CONFIG['dim'])
        self.dropout = nn.Dropout(MODEL_CONFIG['drop'])

    def forward(self, input_ids, attention_mask):
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        # Lấy embedding của token [CLS]
        text_features = outputs.last_hidden_state[:, 0, :]
        text_features = self.proj(text_features)
        text_features = self.norm(text_features)
        text_features = self.dropout(text_features)
        return text_features.unsqueeze(1)  # Thêm dimension để phù hợp với image features
