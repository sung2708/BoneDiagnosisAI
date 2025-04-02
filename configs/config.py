import torch

DATA_PATHS = {
    'image_dir': './data/images/',
    'label_path': './data/advanced_vqa_labels.json',
    'save_dir': './saved_models'
}

MODEL_CONFIG = {
    "vocab_size": 30522,  # PhoBERT vocab size
    "max_len": 32,        # Tăng độ dài tối đa cho câu hỏi
    "dim": 768,           # Chiều dài vector embedding
    "heads": 8,           # Số head trong MultiHeadAttention
    "n_layers": 6,        # Tăng số layer transformer
    "drop": 0.1,          # Thêm dropout để tránh overfitting
    "batch_size": 32,     # Giảm batch size nếu GPU yếu
    "epochs": 50,         # Đủ để hội tụ
    "lr": 3e-5,           # Learning rate nhỏ hơn
    "clip": True,         # Gradient clipping
    "warmup_steps": 1000, # Warmup cho scheduler
    "weight_decay": 0.01, # Regularization
    "device": "cuda" if torch.cuda.is_available() else "cpu"
}
