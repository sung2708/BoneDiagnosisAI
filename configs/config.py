import torch

MODEL_CONFIG = {
    # Model architecture
    "dim": 768,
    "heads": 8,
    "drop": 0.1,

    # Input processing
    "text_model": "vinai/phobert-base",
    "max_len": 64,
    "image_size": 224,
    "image_mean": [0.485, 0.456, 0.406],
    "image_std": [0.229, 0.224, 0.225],

    # Classes (sẽ được tự động cập nhật từ tên thư mục)
    "class_names": ["U_xuong", "Viem_nhiem", "Chan_thuong"],  # Cố định 3 lớp
    "medical_keywords": [
        "u xương", "viêm", "nhiễm trùng", "gãy", "chấn thương",
        "tổn thương", "đau", "X-quang", "xương", "khớp"
    ],

    # Device
    "device": "cuda" if torch.cuda.is_available() else "cpu"
}

DATA_PATHS = {
    "image_dir": "data",  # Thư mục gốc chứa các thư mục con theo lớp
    "save_dir": "saved_models",
    "eval_dir": "evaluation"
}

TRAINING_CONFIG = {
    "batch_size": 32,
    "epochs": 30,
    "learning_rate": 3e-5,
    "oversample_minority": True
}
