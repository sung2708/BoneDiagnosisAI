import torch
import torch.nn as nn
from fusion import MultiHeadedSelfAttention, PositionWiseFeedForward
from ..configs import MODEL_CONFIG
from transformers import AutoModel, AutoTokenizer

dim = MODEL_CONFIG.dim
drop = MODEL_CONFIG.drop
heads = MODEL_CONFIG.heads
n_layers = MODEL_CONFIG.n_layers
vocab_size = MODEL_CONFIG.vocab_size
device = MODEL_CONFIG.device

tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base")
phobert_model = AutoModel.from_pretrained("vinai/phobert-base")

class Embeddings(nn.Module):
    def __init__(self,max_len):
        super(Embeddings, self).__init__()
        self.word_embeddings = phobert_model.embeddings.word_embeddings
        self.word_embeddings_2 = nn.Linear(128, dim, bias=False)
        self.position_embeddings = nn.Embedding(max_len, dim)
        self.type_embeddings = nn.Embedding(3, dim)
        self.LayerNorm = nn.LayerNorm(dim, eps=1e-12)
        self.dropout = nn.Dropout(drop)
        self.len = max_len
    def forward(self, input_ids, segment_ids, position_ids=None):
        input_ids = torch.cuda.LongTensor(input_ids)
        segment_ids = torch.cuda.LongTensor(segment_ids)
        if position_ids is None:
            position_ids = torch.arange(self.len, dtype=torch.long, device=device)
            position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        words_embeddings = self.word_embeddings(input_ids)
        words_embeddings = self.word_embeddings_2(words_embeddings)
        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.type_embeddings(segment_ids)
        embeddings = words_embeddings + position_embeddings + token_type_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings

class BertLayer(nn.Module):
    def __init__(self, share='all', norm='pre'):
        super(BertLayer, self).__init__()
        self.share = share
        self.norm_pos = norm
        self.norm1 = nn.LayerNorm(dim, eps=1e-12)
        self.norm2 = nn.LayerNorm(dim, eps=1e-12)
        self.drop1 = nn.Dropout(drop)
        self.drop2 = nn.Dropout(drop)
        if self.share == 'ffn':
            self.attention = nn.ModuleList([MultiHeadedSelfAttention() for _ in range(n_layers)])
            self.proj = nn.ModuleList([nn.Linear(dim, dim) for _ in range(n_layers)])
            self.feedforward = PositionWiseFeedForward()
        elif self.share == 'att':
            self.attention = MultiHeadedSelfAttention()
            self.proj = nn.Linear(dim, dim)
            self.feedforward = nn.ModuleList([PositionWiseFeedForward() for _ in range(n_layers)])
        elif self.share == 'all':
            self.attention = MultiHeadedSelfAttention()
            self.proj = nn.Linear(dim, dim)
            self.feedforward = PositionWiseFeedForward()
        elif self.share == 'none':
            self.attention = nn.ModuleList([MultiHeadedSelfAttention() for _ in range(n_layers)])
            self.proj = nn.ModuleList([nn.Linear(dim, dim) for _ in range(n_layers)])
            self.feedforward = nn.ModuleList([PositionWiseFeedForward() for _ in range(n_layers)])
    def forward(self, hidden_states, attention_mask, layer_num):
        attention_mask = torch.cuda.LongTensor(attention_mask)
        if self.norm_pos == 'pre':
            if isinstance(self.attention, nn.ModuleList):
                h = self.proj[layer_num](self.attention[layer_num](self.norm1(hidden_states), attention_mask))
            else:
                h = self.proj(self.attention(self.norm1(hidden_states), attention_mask))
            out = hidden_states + self.drop1(h)
            if isinstance(self.feedforward, nn.ModuleList):
                h = self.feedforward[layer_num](self.norm1(out))
            else:
                h = self.feedforward(self.norm1(out))
            out = out + self.drop2(h)
        if self.norm_pos == 'post':
            if isinstance(self.attention, nn.ModuleList):
                h = self.proj[layer_num](self.attention[layer_num](hidden_states, attention_mask))
            else:
                h = self.proj(self.attention(hidden_states, attention_mask))
            out = self.norm1(hidden_states + self.drop1(h))
            if isinstance(self.feedforward, nn.ModuleList):
                h = self.feedforward[layer_num](out)
            else:
                h = self.feedforward(out)
            out = self.norm2(out + self.drop2(h))
        return out
