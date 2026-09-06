"""
dataset.py

PyTorch Dataset for loading face crop images from the CSV splits produced
by make_splits.py.

Labels: real -> 0, fake -> 1
"""

from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


LABEL_MAP = {"real": 0, "fake": 1}


def get_transforms(train=True, image_size=224):
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )
    if train:
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
            transforms.ToTensor(),
            normalize,
        ])
    else:
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            normalize,
        ])


class DeepfakeFaceDataset(Dataset):
    def __init__(self, csv_path, image_size=224, train=True):
        self.df = pd.read_csv(csv_path)
        self.transform = get_transforms(train=train, image_size=image_size)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_path = row["filepath"]
        label_str = row["label"]

        image = Image.open(image_path).convert("RGB")
        image = self.transform(image)
        label = LABEL_MAP[label_str]

        return image, torch.tensor(label, dtype=torch.long)


if __name__ == "__main__":
    import sys
    csv_path = sys.argv[1] if len(sys.argv) > 1 else "data/splits/train.csv"
    ds = DeepfakeFaceDataset(csv_path, train=True)
    print(f"Dataset size: {len(ds)}")
    img, label = ds[0]
    print(f"Sample image shape: {img.shape}, label: {label.item()}")
