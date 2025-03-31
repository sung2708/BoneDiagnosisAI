from torch import nn
import torch
from ..configs import MODEL_CONFIG
from transformer_model import Transformer
from ..utils import gelu

dim = MODEL_CONFIG.dim
gen_len = MODEL_CONFIG.gen_len


class mod_model1(nn.Module):
    def __init__(self):
        super().__init__()
        self.transformer = Transformer()
        self.fc1 = nn.Linear(dim, dim)
        self.activ1 = nn.Tanh()
        self.classifier = nn.Linear(dim, 43)
    def forward(self, name, input_ids, segment_ids, input_mask):
        h = self.transformer(name, input_ids, segment_ids, input_mask)
        pooled_h = self.activ1(self.fc1(h[:, 0]))
        logits_clsf = self.classifier(pooled_h)
        return logits_clsf

class mod_model2(nn.Module):
    def __init__(self):
        super().__init__()
        self.transformer = Transformer()
        self.fc1 = nn.Linear(dim, dim)
        self.activ1 = nn.Tanh()
        self.classifier = nn.Linear(dim, 43)
    def forward(self, name, input_ids, segment_ids, input_mask):
        h = self.transformer(name, input_ids, segment_ids, input_mask)
        pooled_h = self.activ1(self.fc1(h[:, 0]))
        logits_clsf = self.classifier(pooled_h)
        return logits_clsf

class mod_model3(nn.Module):
    def __init__(self):
        super().__init__()
        self.transformer = Transformer()
        self.fc1 = nn.Linear(dim, dim)
        self.activ1 = nn.Tanh()
        self.classifier = nn.Linear(dim, 43)
    def forward(self, name, input_ids, segment_ids, input_mask):
        h = self.transformer(name, input_ids, segment_ids, input_mask)
        pooled_h = self.activ1(self.fc1(h[:, 0]))
        logits_clsf = self.classifier(pooled_h)
        return logits_clsf

class mod_yn_model1(nn.Module):
    def __init__(self):
        super().__init__()
        self.transformer = Transformer()
        self.fc1 = nn.Linear(dim, dim)
        self.activ1 = nn.Tanh()
        self.classifier = nn.Linear(dim, 2)
    def forward(self, name, input_ids, segment_ids, input_mask):
        h = self.transformer(name, input_ids, segment_ids, input_mask)
        pooled_h = self.activ1(self.fc1(h[:, 0]))
        logits_clsf = self.classifier(pooled_h)
        return logits_clsf

class mod_yn_model2(nn.Module):
    def __init__(self):
        super().__init__()
        self.transformer = Transformer()
        self.fc1 = nn.Linear(dim, dim)
        self.activ1 = nn.Tanh()
        self.classifier = nn.Linear(dim, 2)
    def forward(self, name, input_ids, segment_ids, input_mask):
        h = self.transformer(name, input_ids, segment_ids, input_mask)
        pooled_h = self.activ1(self.fc1(h[:, 0]))
        logits_clsf = self.classifier(pooled_h)
        return logits_clsf

class mod_yn_model3(nn.Module):
    def __init__(self):
        super().__init__()
        self.transformer = Transformer()
        self.fc1 = nn.Linear(dim, dim)
        self.activ1 = nn.Tanh()
        self.classifier = nn.Linear(dim, 2)
    def forward(self, name, input_ids, segment_ids, input_mask):
        h = self.transformer(name, input_ids, segment_ids, input_mask)
        pooled_h = self.activ1(self.fc1(h[:, 0]))
        logits_clsf = self.classifier(pooled_h)
        return logits_clsf

class pla_model1(nn.Module):
    def __init__(self):
        super().__init__()
        self.transformer = Transformer()
        self.fc1 = nn.Linear(dim, dim)
        self.activ1 = nn.Tanh()
        self.classifier = nn.Linear(dim, 16)
    def forward(self, name, input_ids, segment_ids, input_mask):
        h = self.transformer(name, input_ids, segment_ids, input_mask)
        pooled_h = self.activ1(self.fc1(h[:, 0]))
        logits_clsf = self.classifier(pooled_h)
        return logits_clsf

class pla_model2(nn.Module):
    def __init__(self):
        super().__init__()
        self.transformer = Transformer()
        self.fc1 = nn.Linear(dim, dim)
        self.activ1 = nn.Tanh()
        self.classifier = nn.Linear(dim, 16)
    def forward(self, name, input_ids, segment_ids, input_mask):
        h = self.transformer(name, input_ids, segment_ids, input_mask)
        pooled_h = self.activ1(self.fc1(h[:, 0]))
        logits_clsf = self.classifier(pooled_h)
        return logits_clsf

class pla_model3(nn.Module):
    def __init__(self):
        super().__init__()
        self.transformer = Transformer()
        self.fc1 = nn.Linear(dim, dim)
        self.activ1 = nn.Tanh()
        self.classifier = nn.Linear(dim, 16)
    def forward(self, name, input_ids, segment_ids, input_mask):
        h = self.transformer(name, input_ids, segment_ids, input_mask)
        pooled_h = self.activ1(self.fc1(h[:, 0]))
        logits_clsf = self.classifier(pooled_h)
        return logits_clsf

class org_model1(nn.Module):
    def __init__(self):
        super().__init__()
        self.transformer = Transformer()
        self.fc1 = nn.Linear(dim, dim)
        self.activ1 = nn.Tanh()
        self.classifier = nn.Linear(dim, 10)
    def forward(self, name, input_ids, segment_ids, input_mask):
        h = self.transformer(name, input_ids, segment_ids, input_mask)
        pooled_h = self.activ1(self.fc1(h[:, 0]))
        logits_clsf = self.classifier(pooled_h)
        return logits_clsf

class org_model2(nn.Module):
    def __init__(self):
        super().__init__()
        self.transformer = Transformer()
        self.fc1 = nn.Linear(dim, dim)
        self.activ1 = nn.Tanh()
        self.classifier = nn.Linear(dim, 10)
    def forward(self, name, input_ids, segment_ids, input_mask):
        h = self.transformer(name, input_ids, segment_ids, input_mask)
        pooled_h = self.activ1(self.fc1(h[:, 0]))
        logits_clsf = self.classifier(pooled_h)
        return logits_clsf

class org_model3(nn.Module):
    def __init__(self):
        super().__init__()
        self.transformer = Transformer()
        self.fc1 = nn.Linear(dim, dim)
        self.activ1 = nn.Tanh()
        self.classifier = nn.Linear(dim, 10)
    def forward(self, name, input_ids, segment_ids, input_mask):
        h = self.transformer(name, input_ids, segment_ids, input_mask)
        pooled_h = self.activ1(self.fc1(h[:, 0]))
        logits_clsf = self.classifier(pooled_h)
        return logits_clsf

class abn_model(nn.Module):
    def __init__(self):
        super().__init__()
        self.transformer = Transformer(gen_len)
        self.fc2 = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim, eps=1e-12)
        embed_weight = self.transformer.embed.word_embeddings.weight
        n_vocab, embed_dim = embed_weight.size()
        self.decoder = nn.Linear(dim, embed_dim, bias=False)
        self.decoder_2 = nn.Linear(embed_dim, n_vocab, bias=False)
        self.decoder_2.weight = embed_weight
        self.decoder_bias = nn.Parameter(torch.zeros(n_vocab))
    def forward(self, name, input_ids, segment_ids, input_mask, masked_pos):
        h = self.transformer(name, input_ids, segment_ids, input_mask)
        masked_pos = torch.cuda.LongTensor(masked_pos)[:, :, None].expand(-1, -1, h.size(-1))
        h_masked = torch.gather(h, 1, masked_pos)
        h_masked = self.norm(gelu(self.fc2(h_masked)))
        logits_lm = self.decoder_2(self.decoder(h_masked)) + self.decoder_bias
        return logits_lm

class abn_yn_model1(nn.Module):
    def __init__(self):
        super().__init__()
        self.transformer = Transformer()
        self.fc1 = nn.Linear(dim, dim)
        self.activ1 = nn.Tanh()
        self.classifier = nn.Linear(dim, 2)
    def forward(self, name, input_ids, segment_ids, input_mask):
        h = self.transformer(name, input_ids, segment_ids, input_mask)
        pooled_h = self.activ1(self.fc1(h[:, 0]))
        logits_clsf = self.classifier(pooled_h)
        return logits_clsf

class abn_yn_model2(nn.Module):
    def __init__(self):
        super().__init__()
        self.transformer = Transformer()
        self.fc1 = nn.Linear(dim, dim)
        self.activ1 = nn.Tanh()
        self.classifier = nn.Linear(dim, 2)
    def forward(self, name, input_ids, segment_ids, input_mask):
        h = self.transformer(name, input_ids, segment_ids, input_mask)
        pooled_h = self.activ1(self.fc1(h[:, 0]))
        logits_clsf = self.classifier(pooled_h)
        return logits_clsf

class abn_yn_model3(nn.Module):
    def __init__(self):
        super().__init__()
        self.transformer = Transformer()
        self.fc1 = nn.Linear(dim, dim)
        self.activ1 = nn.Tanh()
        self.classifier = nn.Linear(dim, 2)
    def forward(self, name, input_ids, segment_ids, input_mask):
        h = self.transformer(name, input_ids, segment_ids, input_mask)
        pooled_h = self.activ1(self.fc1(h[:, 0]))
        logits_clsf = self.classifier(pooled_h)
        return logits_clsf
