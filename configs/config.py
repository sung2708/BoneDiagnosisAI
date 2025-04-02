import torch

DATA_PATHS = {
    'image_dir': 'data/images/',
    'label_path': 'data/processed_labels.json',
    'save_dir': 'saved_models',
    'eval_dir': 'evaluation'
}

MODEL_CONFIG = {
    "vocab_size": 30522,          # PhoBERT vocab size
    "max_len": 32,                # Max sequence length for questions
    "dim": 768,                   # Embedding dimension
    "heads": 8,                   # Number of attention heads
    "n_layers": 6,                # Number of transformer layers
    "drop": 0.1,                  # Dropout rate
    "batch_size": 32,             # Batch size
    "epochs": 50,                 # Number of epochs
    "lr": 3e-5,                   # Learning rate
    "clip": True,                 # Gradient clipping
    "warmup_steps": 1000,         # Warmup steps for scheduler
    "weight_decay": 0.01,         # Weight decay for regularization
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "train_ratio": 0.7,           # 70% for training
    "val_ratio": 0.15,            # 15% for validation
    "test_ratio": 0.15            # 15% for testing
}
