import os
import torch
from torch.utils.data import DataLoader
from bone_dataset import BoneDataset
from vqa_system import VQASystem
from Configs.config import Config
from torchvision import transforms

def get_transforms():
    return transforms.Compose([
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                          std=[0.229, 0.224, 0.225])
    ])

def custom_collate_fn(batch):
    """Custom collate function to handle multiple questions per image"""
    images = torch.stack([item['image'] for item in batch])
    all_questions = [item['questions'] for item in batch]
    all_question_types = [item['question_types'] for item in batch]
    all_labels = torch.stack([item['labels'] for item in batch])

    # Flatten the questions and labels
    questions = []
    question_types = []
    labels = []

    for i in range(len(batch)):
        for j in range(len(Config.QUESTION_TYPES)):
            questions.append(all_questions[i][j])
            question_types.append(all_question_types[i][j])

    labels = all_labels.view(-1)  # Flatten the labels

    return images, questions, question_types, labels

def main():
    Config.create_dirs()
    Config.check_paths()

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Transforms
    transform = get_transforms()

    # Datasets
    train_dataset = BoneDataset(
        Config.TRAIN_CSV,
        Config.IMAGE_DIR,
        transform=transform,
        is_train=True
    )

    val_dataset = BoneDataset(
        Config.VAL_CSV,
        Config.IMAGE_DIR,
        transform=transform,
        is_train=False
    )

    # DataLoaders with custom collate
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        collate_fn=custom_collate_fn
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
        collate_fn=custom_collate_fn
    )

    # Model
    model = VQASystem().to(device)

    # Check if pretrained weights exist
    if Config.DATA_PATHS['pretrained_mura'].exists():
        print("Loading pretrained MURA weights for image encoder...")
        model.load_pretrained_image_encoder(Config.DATA_PATHS['pretrained_mura'])
    else:
        print("Warning: Pretrained MURA weights not found, training from scratch")

    # Training
    try:
        model.train_model(train_loader, val_loader)

        # Evaluation
        print("\n🧪 Final Evaluation:")
        metrics = model.evaluate(val_loader)
        print(f"✅ Overall Accuracy: {metrics['accuracy']:.4f}")
        print(f"✅ Macro F1: {metrics['macro_f1']:.4f}")

        print("\n📊 Metrics by question type:")
        for q_type, q_metrics in metrics['type_metrics'].items():
            print(f" - {q_type}: Accuracy = {q_metrics['accuracy']:.4f}, F1 = {q_metrics['f1']:.4f}")

        # Save report
        report_path = Config.SAVE_DIR / 'classification_report.txt'
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(metrics['report'])
        print(f"\n📄 Classification report saved to: {report_path}")

    except Exception as e:
        print(f"\n❌ Training failed: {str(e)}")
        raise e

if __name__ == "__main__":
    main()
