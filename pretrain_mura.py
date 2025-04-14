import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms, models
from torchvision.models.resnet import ResNet50_Weights
import pandas as pd
from PIL import Image
import os
from tqdm import tqdm
import logging

class Config:
    # Corrected BASE_PATH - should point to the directory containing MURA-v1.1 folder
    BASE_PATH = "MURA-v1.1/"  # Changed from "MURA-v1.1/MURA-v1.1/"
    IMAGE_PATHS_FILE = os.path.join(BASE_PATH, "train_image_paths.csv")
    LABELED_STUDIES_FILE = os.path.join(BASE_PATH, "train_labeled_studies.csv")
    MODEL_SAVE_PATH = os.path.join("saved_models", "resnet50_mura.pth")
    BATCH_SIZE = 32
    NUM_EPOCHS = 15
    LEARNING_RATE = 1e-4
    IMG_SIZE = 224
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    TRAIN_MEAN = [0.485, 0.456, 0.406]
    TRAIN_STD = [0.229, 0.224, 0.225]

config = Config()

# Create saved_models directory if it doesn't exist
os.makedirs("saved_models", exist_ok=True)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('pretrain.log'),
        logging.StreamHandler()
    ]
)

class MuraDataset(torch.utils.data.Dataset):
    def __init__(self, img_paths_file, label_file, transform=None):
        try:
            # Check if files exist
            if not os.path.exists(img_paths_file):
                raise FileNotFoundError(f"Image paths file not found: {img_paths_file}")
            if not os.path.exists(label_file):
                raise FileNotFoundError(f"Label file not found: {label_file}")

            # Read CSV files
            self.img_paths = pd.read_csv(img_paths_file, header=None, names=['path'])
            self.labels_df = pd.read_csv(label_file, header=None, names=['study', 'label'])

            # Create study to label mapping (without BASE_PATH prefix)
            self.study_to_label = {
                study.strip(): label
                for study, label in zip(self.labels_df['study'], self.labels_df['label'])
            }

            # Find valid indices
            self.valid_indices = []
            for i, path in enumerate(self.img_paths['path']):
                full_path = path.strip()  # Paths in CSV are already complete
                study_path = os.path.dirname(full_path) + '/'

                if study_path in self.study_to_label and self._validate_path(full_path):
                    self.valid_indices.append(i)

            if len(self.valid_indices) == 0:
                raise ValueError("No valid images found in the dataset")

            logging.info(f"Loaded dataset with {len(self.valid_indices)} valid images")

            # Define transformations
            self.transform = transform or transforms.Compose([
                transforms.Resize(config.IMG_SIZE + 32),
                transforms.RandomResizedCrop(config.IMG_SIZE, scale=(0.8, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(15),
                transforms.ColorJitter(brightness=0.1, contrast=0.1),
                transforms.ToTensor(),
                transforms.Normalize(config.TRAIN_MEAN, config.TRAIN_STD)
            ])

        except Exception as e:
            logging.error(f"Dataset initialization error: {str(e)}")
            raise

    def _validate_path(self, path):
        """Check if path exists and is a valid image file."""
        path = str(path).strip()
        if not path.lower().endswith(('.png', '.jpg', '.jpeg')):
            return False
        if not os.path.exists(path):
            logging.warning(f"Image file not found: {path}")
            return False
        return True

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        real_idx = self.valid_indices[idx]
        full_path = str(self.img_paths.iloc[real_idx]['path']).strip()
        study_path = os.path.dirname(full_path) + '/'

        try:
            label = self.study_to_label[study_path]
            img = Image.open(full_path).convert('RGB')

            if self.transform:
                img = self.transform(img)

            return img, torch.tensor(label, dtype=torch.float32)

        except Exception as e:
            logging.warning(f"Error loading image {full_path}: {str(e)}")
            return None, None

def collate_fn(batch):
    """Filter out None samples and stack the rest."""
    batch = [item for item in batch if item[0] is not None]
    if len(batch) == 0:
        return torch.Tensor(), torch.Tensor()
    images = torch.stack([item[0] for item in batch])
    labels = torch.stack([item[1] for item in batch])
    return images, labels

def init_model():
    """Initialize ResNet50 model with pretrained weights."""
    model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
    num_features = model.fc.in_features

    # Modify final layers for binary classification
    model.fc = nn.Sequential(
        nn.Dropout(0.2),
        nn.Linear(num_features, 1),
        nn.Sigmoid()
    )

    # Freeze all layers except the last block and fc layer
    for name, param in model.named_parameters():
        if 'layer4' not in name and 'fc' not in name:
            param.requires_grad = False

    return model.to(config.DEVICE)

def train_model():
    """Main training loop."""
    try:
        model = init_model()
        criterion = nn.BCELoss()
        optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2)

        # Initialize dataset and dataloader
        train_dataset = MuraDataset(config.IMAGE_PATHS_FILE, config.LABELED_STUDIES_FILE)
        train_loader = DataLoader(
            train_dataset,
            batch_size=config.BATCH_SIZE,
            shuffle=True,
            num_workers=4,
            collate_fn=collate_fn,
            pin_memory=True
        )

        best_loss = float('inf')
        for epoch in range(config.NUM_EPOCHS):
            model.train()
            running_loss = 0.0
            progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{config.NUM_EPOCHS}')

            for images, labels in progress_bar:
                if len(images) == 0:
                    continue

                images, labels = images.to(config.DEVICE), labels.to(config.DEVICE)

                outputs = model(images).squeeze()
                loss = criterion(outputs, labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                progress_bar.set_postfix(loss=loss.item())

            epoch_loss = running_loss / len(train_loader)
            scheduler.step(epoch_loss)
            logging.info(f'Epoch {epoch+1} - Loss: {epoch_loss:.4f}')

            if epoch_loss < best_loss:
                best_loss = epoch_loss
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'epoch': epoch,
                    'best_loss': best_loss,
                }, config.MODEL_SAVE_PATH)
                logging.info(f"Saved best model with loss: {best_loss:.4f}")

    except Exception as e:
        logging.error(f"Error in training: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    logging.info("Starting MURA pretraining...")
    logging.info(f"Device: {config.DEVICE}")
    logging.info(f"Epochs: {config.NUM_EPOCHS}")
    logging.info(f"Batch size: {config.BATCH_SIZE}")

    try:
        train_model()
        logging.info("Training completed successfully!")
    except Exception as e:
        logging.error(f"Training failed: {str(e)}", exc_info=True)
