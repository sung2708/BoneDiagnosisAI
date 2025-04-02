import torch
import torch.nn as nn
from transformers import AutoModel, RobertaConfig
from configs.config import MODEL_CONFIG

class PhoBERTTextEncoder(nn.Module):
    def __init__(self):
        super().__init__()

        # 1. Enhanced PhoBERT configuration
        config = RobertaConfig.from_pretrained(
            "vinai/phobert-base-v2",
            hidden_dropout_prob=MODEL_CONFIG.get('text_dropout', 0.1),
            attention_probs_dropout_prob=MODEL_CONFIG.get('attn_dropout', 0.1),
            pooler_type="none"  # Disable unused pooler
        )

        # 2. Load model with custom config
        self.phobert = AutoModel.from_pretrained(
            "vinai/phobert-base-v2",
            config=config,
            ignore_mismatched_sizes=True
        )

        # 3. Improved projection layer
        self.proj = nn.Sequential(
            nn.Dropout(p=MODEL_CONFIG.get('proj_dropout', 0.1)),
            nn.Linear(768, MODEL_CONFIG['dim']),
            nn.LayerNorm(MODEL_CONFIG['dim']),
            nn.GELU(),
            nn.Dropout(p=MODEL_CONFIG.get('final_dropout', 0.1))
        )

        # 4. Initialize weights properly
        self._init_weights()

    def _init_weights(self):
        """Initialize weights for linear and layer norm layers"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.LayerNorm):
                nn.init.constant_(module.weight, 1.0)
                nn.init.constant_(module.bias, 0)

    def forward(self, input_ids, attention_mask):
        """
        Args:
            input_ids: Tokenized input ids [B, seq_len]
            attention_mask: Attention mask [B, seq_len]
        Returns:
            Text features [B, 1, dim]
        """
        # Get PhoBERT outputs
        outputs = self.phobert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )

        # Extract [CLS] token and project
        cls_token = outputs.last_hidden_state[:, 0, :]  # [B, 768]
        projected = self.proj(cls_token)  # [B, dim]

        # Add sequence dimension
        return projected.unsqueeze(1)  # [B, 1, dim]
