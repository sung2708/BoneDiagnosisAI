import torch

device = "cuda:0" if torch.cuda.is_available() else "cpu"

DATA_PATHS = {
    'image_dir': 'data/images/',
    'label_path': 'data/advanced_vqa_labels.json'
}

MODEL_CONFIG = {
    "vocab_size": 30522,
    "max_len": 20,
    "gen_len": 33,
    "heads": 12,
    "epochs": 100,
    "lr": 0.0001,
    "clip": True,
    "share": "all",
    "norm": "pre",
    "dim": 768,
    "drop": 0.0,
    "n_layers": 4,
    "batch_size": 64,
}
