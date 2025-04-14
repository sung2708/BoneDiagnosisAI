import os
from pathlib import Path

class Config:
    freeze_text_encoder = True

    # Model configuration
    dim = 512
    heads = 8
    drop = 0.2
    max_text_len = 64

    # Training
    lr = 1e-4
    img_lr = 1e-5
    text_lr = 5e-5
    batch_size = 16
    num_epochs = 20
    patience = 5

    # Paths - use absolute paths to avoid issues
    BASE_DIR = Path('Data')
    IMAGE_DIR = BASE_DIR
    TRAIN_CSV = Path('images_train_label.csv')
    VAL_CSV = Path('images_valid_label.csv')
    DATA_PATHS = {
        'pretrained_mura': Path('saved_models/resnet50_mura.pth'),
    }
    SAVE_DIR = Path('saved_models')

    # Question types and answers
    QUESTION_TYPES = {
        'disease_group': ['U xương', 'Viêm nhiễm', 'Chấn thương', 'Khác'],
        'bone_tumor_type': [
            'Sarcom tạo xương', 'Sarcoma màng', 'Sarcom sợi',
            'Theo dõi sarcoma', 'Bướu đại bào', 'Bướu sợi không sinh xương',
            'Sarcom tạo xương + gãy bệnh lý', 'Bướu phần mềm', 'Khác'
        ],
        'location': [
            'Xương đùi trái', 'Xương đùi phải', 'Xương chày trái',
            'Xương chày phải', 'Xương mác trái', 'Xương mác phải',
            'Xương cánh tay trái', 'Xương cánh tay phải', 'Khác'
        ],
        'imaging_modality': ['X quang', 'CT', 'MRI', 'Khác'],
        'tumor_nature': ['Ác tính', 'Lành tính', 'Không xác định']
    }

    @classmethod
    def create_dirs(cls):
        os.makedirs(cls.SAVE_DIR, exist_ok=True)
        os.makedirs('pretrained', exist_ok=True)
        os.makedirs(cls.IMAGE_DIR, exist_ok=True)

    @classmethod
    def check_paths(cls):
        """Method to ensure paths exist before use."""
        if not cls.BASE_DIR.exists():
            raise FileNotFoundError(f"BASE_DIR {cls.BASE_DIR} does not exist.")
        if not cls.TRAIN_CSV.exists():
            raise FileNotFoundError(f"Training CSV file {cls.TRAIN_CSV} not found.")
        if not cls.VAL_CSV.exists():
            raise FileNotFoundError(f"Validation CSV file {cls.VAL_CSV} not found.")
        if not cls.SAVE_DIR.exists():
            print(f"Warning: {cls.SAVE_DIR} does not exist, creating it now.")
            os.makedirs(cls.SAVE_DIR)

    @classmethod
    def to_dict(cls):
        """Return a serializable dictionary of config values"""
        return {
            'freeze_text_encoder': cls.freeze_text_encoder,
            'dim': cls.dim,
            'heads': cls.heads,
            'drop': cls.drop,
            'max_text_len': cls.max_text_len,
            'lr': cls.lr,
            'img_lr': cls.img_lr,
            'text_lr': cls.text_lr,
            'batch_size': cls.batch_size,
            'num_epochs': cls.num_epochs,
            'patience': cls.patience,
            'BASE_DIR': str(cls.BASE_DIR),
            'IMAGE_DIR': str(cls.IMAGE_DIR),
            'TRAIN_CSV': str(cls.TRAIN_CSV),
            'VAL_CSV': str(cls.VAL_CSV),
            'DATA_PATHS': {k: str(v) for k, v in cls.DATA_PATHS.items()},
            'SAVE_DIR': str(cls.SAVE_DIR),
            'QUESTION_TYPES': cls.QUESTION_TYPES
        }
