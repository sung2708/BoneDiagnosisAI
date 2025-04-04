from torch.utils.data import Dataset
import os
from PIL import Image

class BoneDiseaseDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.classes = ['U_xuong', 'Viem_nhiem', 'Chan_thuong']
        self.samples = []
        self.transform = transform

        for class_idx, class_name in enumerate(self.classes):
            class_dir = os.path.join(root_dir, class_name)
            for img_name in os.listdir(class_dir):
                if img_name.endswith(('.png', '.jpg')):
                    self.samples.append({
                        'image_path': os.path.join(class_dir, img_name),
                        'label': class_idx
                    })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        image = Image.open(item['image_path']).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return {
            'image': image,
            'label': item['label']
        }
