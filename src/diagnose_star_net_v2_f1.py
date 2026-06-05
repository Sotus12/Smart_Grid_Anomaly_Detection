"""
Diagnostic script to print the exact validation metrics (precision, recall, F1)
for both class 0 and class 1 at various thresholds.
"""

import os, sys
import numpy as np
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from sklearn.metrics import (
    recall_score, precision_score, f1_score, accuracy_score,
    confusion_matrix
)

sys.path.insert(0, str(Path(__file__).parent))
from star_net import build_star_net

DATA_DIR  = r'C:\SOFTWARE\DL Project\outputs\spectrograms'
MODEL_DIR = r'C:\SOFTWARE\DL Project\models'
MODEL_PATH = os.path.join(MODEL_DIR, 'star_net_v2_best.pth')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

val_ds = datasets.ImageFolder(os.path.join(DATA_DIR, 'val'), val_transform)
print(f"Class mapping: {val_ds.class_to_idx}")
val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=0)

model = build_star_net(num_classes=2, metric_dim=64, dropout=0.3).to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()
print(f"[OK] Model loaded from {MODEL_PATH}")

@torch.no_grad()
def get_probs_labels(loader):
    all_probs  = []
    all_labels = []
    for images, labels in loader:
        images = images.to(device)
        logits = model(images)
        probs  = torch.softmax(logits, dim=1)[:, 1] # prob of class 1 (normal)
        all_probs.extend(probs.cpu().numpy())
        all_labels.extend(labels.numpy())
    return np.array(all_probs), np.array(all_labels)

val_probs, val_labels = get_probs_labels(val_loader)

# print counts of each class
unique, counts = np.unique(val_labels, return_counts=True)
print(f"Validation label counts: {dict(zip(unique, counts))}")

print("\n--- Sweeping thresholds for class 1 (normal) ---")
print(f"  {'Thresh':>7} | {'Recall':>7} | {'Prec':>7} | {'F1':>7} | {'TN':>5} {'FP':>5} {'FN':>5} {'TP':>5}")
print("-" * 65)
for t in [0.01, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.55, 0.60, 0.70, 0.80]:
    p = (val_probs >= t).astype(int)
    cm = confusion_matrix(val_labels, p, labels=[0,1])
    rec  = recall_score(val_labels, p, zero_division=0)
    prec = precision_score(val_labels, p, zero_division=0)
    f1   = f1_score(val_labels, p, zero_division=0)
    print(f"  {t:>7.3f} | {rec:>7.4f} | {prec:>7.4f} | {f1:>7.4f} | "
          f"{cm[0,0]:>5} {cm[0,1]:>5} {cm[1,0]:>5} {cm[1,1]:>5}")
