import torch.nn as nn
from text_encoder import Embeddings
from image_encoder import Transfer
from fusion import MultiHeadedSelfAttention, PositionWiseFeedForward

class Transformer(nn.Module):
    def __init__(self, vocab_size, dim, max_len, n_layers, heads, drop=0.1):
        super().__init__()
        self.embed = Embeddings(vocab_size, dim, max_len, drop)
        self.trans = Transfer(dim=dim)
        self.attention = MultiHeadedSelfAttention(dim, heads, drop)
        self.feedforward = PositionWiseFeedForward(dim)
        self.n_layers = n_layers

    def forward(self, image_paths, input_ids, seg, mask):
        image_features = self.trans(image_paths)
        text_features = self.embed(input_ids, seg)
        for i, key in enumerate(["conv2", "conv3", "conv4", "conv5", "conv7"]):
            text_features[:, i+1] = image_features[key]
        for _ in range(self.n_layers):
            text_features = self.attention(text_features, mask)
            text_features = self.feedforward(text_features)
        return text_features
