import os
import sys
import time
import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from torchvision.models import ResNet18_Weights
from PIL import Image
import pandas as pd
import numpy as np

sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import BASE_DIR, MANIFEST_PATH, ARTIFACTS_DIR, TARGET_CLASSES
from core.evaluation.metrics import compute_classification_metrics


class PlantLeafDataset(Dataset):
    def __init__(self, df: pd.DataFrame, transform=None):
        self.df = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = BASE_DIR / row["path"]
        with Image.open(img_path) as pil_img:
            img = pil_img.convert("RGB")

        if self.transform:
            img = self.transform(img)

        label = int(row["label_id"])
        return img, label


class TinyCNN(nn.Module):
    def __init__(self, num_classes=6):
        super().__init__()
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )
        self.conv_block3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = self.classifier(x)
        return x


def train_and_save_dl():
    dl_dir = ARTIFACTS_DIR / "dl"
    dl_dir.mkdir(parents=True, exist_ok=True)

    print("Loading manifest for Deep Learning...")
    df = pd.read_csv(MANIFEST_PATH)
    train_df = df[df["split"] == "train"].copy()
    test_df = df[df["split"] == "test"].copy()

    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_loader = DataLoader(PlantLeafDataset(train_df, transform=train_transform), batch_size=32, shuffle=True)
    test_loader = DataLoader(PlantLeafDataset(test_df, transform=test_transform), batch_size=32, shuffle=False)

    device = torch.device("cpu")
    criterion = nn.CrossEntropyLoss()

    print("\n--- 1. Training Classic TinyCNN (3 Conv Blocks) ---")
    tiny_model = TinyCNN(num_classes=len(TARGET_CLASSES)).to(device)
    optimizer_tiny = optim.Adam(tiny_model.parameters(), lr=0.001)

    tiny_model.train()
    for epoch in range(3):
        running_loss = 0.0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer_tiny.zero_grad()
            outputs = tiny_model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer_tiny.step()
            running_loss += loss.item() * imgs.size(0)
        epoch_loss = running_loss / len(train_df)
        print(f"TinyCNN Epoch {epoch+1}/3 - Loss: {epoch_loss:.4f}")

    tiny_pt_path = dl_dir / "tinycnn_leaf.pt"
    torch.save(tiny_model.state_dict(), tiny_pt_path)
    print(f"TinyCNN saved to {tiny_pt_path} ({tiny_pt_path.stat().st_size / (1024*1024):.2f} MB)")

    print("\n--- 2. Fine-tuning ResNet-18 (Lightweight Pretrained Backbone) ---")
    resnet_model = models.resnet18(weights=ResNet18_Weights.DEFAULT)
    for param in resnet_model.parameters():
        param.requires_grad = False

    num_features = resnet_model.fc.in_features
    resnet_model.fc = nn.Linear(num_features, len(TARGET_CLASSES))
    resnet_model = resnet_model.to(device)

    optimizer_resnet = optim.Adam(resnet_model.fc.parameters(), lr=0.001)

    resnet_model.train()
    for epoch in range(2):
        running_loss = 0.0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer_resnet.zero_grad()
            outputs = resnet_model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer_resnet.step()
            running_loss += loss.item() * imgs.size(0)
        epoch_loss = running_loss / len(train_df)
        print(f"ResNet-18 Epoch {epoch+1}/2 - Loss: {epoch_loss:.4f}")

    resnet_pt_path = dl_dir / "resnet18_leaf.pt"
    torch.save(resnet_model.state_dict(), resnet_pt_path)
    print(f"ResNet-18 saved to {resnet_pt_path} ({resnet_pt_path.stat().st_size / (1024*1024):.2f} MB)")

    print("\n--- 3. Evaluating on Test Set ---")
    models_to_eval = [("tinycnn", tiny_model), ("resnet18", resnet_model)]
    dl_metrics = {}

    for name, m in models_to_eval:
        m.eval()
        all_preds = []
        all_targets = []
        t0 = time.perf_counter()
        with torch.no_grad():
            for imgs, labels in test_loader:
                imgs = imgs.to(device)
                outputs = m(imgs)
                _, preds = torch.max(outputs, 1)
                all_preds.extend(preds.cpu().numpy().tolist())
                all_targets.extend(labels.numpy().tolist())
        infer_ms = (time.perf_counter() - t0) / len(test_df) * 1000.0

        metrics = compute_classification_metrics(all_targets, all_preds, TARGET_CLASSES)
        pt_path = dl_dir / f"{name}_leaf.pt"

        dl_metrics[name] = {
            "model_name": name,
            "accuracy": round(metrics["accuracy"], 4),
            "macro_f1": round(metrics["f1_macro"], 4),
            "infer_ms_per_image": round(infer_ms, 2),
            "model_size_mb": round(pt_path.stat().st_size / (1024 * 1024), 2),
            "confusion_matrix": metrics["confusion_matrix"]
        }
        print(f"[{name.upper()}] Accuracy: {metrics['accuracy']:.4f} | F1: {metrics['f1_macro']:.4f} | Latency: {infer_ms:.2f} ms")

    metrics_save_path = dl_dir / "dl_metrics.json"
    with open(metrics_save_path, "w", encoding="utf-8") as f:
        json.dump(dl_metrics, f, ensure_ascii=False, indent=2)

    print(f"\nEvaluation metrics saved to {metrics_save_path}")


if __name__ == "__main__":
    train_and_save_dl()
