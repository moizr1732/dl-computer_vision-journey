"""
train.py

Trains a deepfake detector using transfer learning on EfficientNet-B0.
"""

import argparse
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import models

from dataset import DeepfakeFaceDataset


def parse_args():
    parser = argparse.ArgumentParser(description="Train deepfake detector.")
    parser.add_argument("--train_csv", type=str, default="data/splits/train.csv")
    parser.add_argument("--dev_csv", type=str, default="data/splits/dev.csv")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--checkpoint_dir", type=str, default="outputs/checkpoints")
    parser.add_argument("--num_workers", type=int, default=0)
    return parser.parse_args()


def build_model(device):
    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)

    for param in model.parameters():
        param.requires_grad = False

    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(in_features, 1)
    )

    model = model.to(device)
    return model


def run_epoch(model, loader, criterion, optimizer, device, train=True):
    model.train() if train else model.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    context = torch.enable_grad() if train else torch.no_grad()
    with context:
        for images, labels in loader:
            images = images.to(device)
            labels = labels.float().to(device)

            if train:
                optimizer.zero_grad()

            outputs = model(images).squeeze(1)
            loss = criterion(outputs, labels)

            if train:
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * images.size(0)
            preds = (torch.sigmoid(outputs) >= 0.5).float()
            correct += (preds == labels).sum().item()
            total += images.size(0)

    avg_loss = total_loss / total
    accuracy = correct / total
    return avg_loss, accuracy


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    train_ds = DeepfakeFaceDataset(args.train_csv, image_size=args.image_size, train=True)
    dev_ds = DeepfakeFaceDataset(args.dev_csv, image_size=args.image_size, train=False)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers)
    dev_loader = DataLoader(dev_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers)

    print(f"Train samples: {len(train_ds)} | Dev samples: {len(dev_ds)}")

    model = build_model(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad], lr=args.lr
    )

    best_dev_acc = 0.0
    history = []

    for epoch in range(1, args.epochs + 1):
        start = time.time()
        train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer, device, train=True)
        dev_loss, dev_acc = run_epoch(model, dev_loader, criterion, optimizer, device, train=False)
        elapsed = time.time() - start

        print(f"Epoch {epoch}/{args.epochs} | "
              f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
              f"dev_loss={dev_loss:.4f} dev_acc={dev_acc:.4f} | "
              f"{elapsed:.1f}s")

        history.append({
            "epoch": epoch, "train_loss": train_loss, "train_acc": train_acc,
            "dev_loss": dev_loss, "dev_acc": dev_acc
        })

        if dev_acc > best_dev_acc:
            best_dev_acc = dev_acc
            checkpoint_path = checkpoint_dir / "best_model.pt"
            torch.save(model.state_dict(), checkpoint_path)
            print(f"  -> New best dev accuracy ({dev_acc:.4f}). Saved to {checkpoint_path}")

    print(f"Training complete. Best dev accuracy: {best_dev_acc:.4f}")

    import pandas as pd
    history_path = checkpoint_dir.parent / "results" / "training_history.csv"
    history_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(history).to_csv(history_path, index=False)
    print(f"Training history saved to {history_path}")


if __name__ == "__main__":
    main()