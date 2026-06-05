"""
GAResNet-18 - Post-Training Diagnostic and Threshold Sweep
==========================================================
Loads the best trained garesnet_fixed_best.pth model and:
1. Evaluates raw anomaly/normal probabilities on validation and test sets
2. Sweeps decision thresholds to maximize precision (either for Anomaly or Normal)
3. Prints exact predicted vs actual numbers (confusion matrix) for each threshold.
"""

import os, sys, json
import numpy as np
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from sklearn.metrics import (
    recall_score, precision_score, f1_score, accuracy_score,
    confusion_matrix, roc_auc_score, classification_report
)

sys.path.insert(0, str(Path(__file__).parent))
from grid_resnet_model import garesnet18

DATA_DIR  = r'C:\SOFTWARE\DL Project\outputs\spectrograms'
MODEL_DIR = r'C:\SOFTWARE\DL Project\models'
MODEL_PATH = os.path.join(MODEL_DIR, 'garesnet_fixed_best.pth')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

test_ds  = datasets.ImageFolder(os.path.join(DATA_DIR, 'test'),  val_transform)
val_ds   = datasets.ImageFolder(os.path.join(DATA_DIR, 'val'),   val_transform)

class_to_idx = test_ds.class_to_idx  # {'anomaly': 0, 'normal': 1}
idx_to_class = {v: k for k, v in class_to_idx.items()}
anomaly_idx = class_to_idx['anomaly']  # 0
normal_idx  = class_to_idx['normal']   # 1

print(f"Class mapping: {class_to_idx}")
print(f"Anomaly index: {anomaly_idx}, Normal index: {normal_idx}")

test_loader = DataLoader(test_ds, batch_size=32, shuffle=False, num_workers=0)
val_loader  = DataLoader(val_ds,  batch_size=32, shuffle=False, num_workers=0)

model = garesnet18(num_classes=2).to(device)
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
        probs  = torch.softmax(logits, dim=1)
        all_probs.append(probs.cpu().numpy())
        all_labels.extend(labels.numpy())
    return np.concatenate(all_probs), np.array(all_labels)

print("\n--- Collecting probabilities ---")
val_probs,  val_labels  = get_probs_labels(val_loader)
test_probs, test_labels = get_probs_labels(test_loader)

# Anomaly probabilities (index 0)
val_anom_probs = val_probs[:, anomaly_idx]
test_anom_probs = test_probs[:, anomaly_idx]

# Normal probabilities (index 1)
val_norm_probs = val_probs[:, normal_idx]
test_norm_probs = test_probs[:, normal_idx]

# ── Probability Distributions ────────────────────────────────────────────────
print("\n=== PROBABILITY DISTRIBUTION ===")
print("Val Set Anomaly Probabilities (class 0):")
print(f"  Actual Anomaly samples (N={np.sum(val_labels==0)}) — mean: {val_anom_probs[val_labels==0].mean():.4f} "
      f"min: {val_anom_probs[val_labels==0].min():.4f} max: {val_anom_probs[val_labels==0].max():.4f}")
print(f"  Actual Normal  samples (N={np.sum(val_labels==1)}) — mean: {val_anom_probs[val_labels==1].mean():.4f} "
      f"min: {val_anom_probs[val_labels==1].min():.4f} max: {val_anom_probs[val_labels==1].max():.4f}")

print("\nTest Set Anomaly Probabilities (class 0):")
print(f"  Actual Anomaly samples (N={np.sum(test_labels==0)}) — mean: {test_anom_probs[test_labels==0].mean():.4f} "
      f"min: {test_anom_probs[test_labels==0].min():.4f} max: {test_anom_probs[test_labels==0].max():.4f}")
print(f"  Actual Normal  samples (N={np.sum(test_labels==1)}) — mean: {test_anom_probs[test_labels==1].mean():.4f} "
      f"min: {test_anom_probs[test_labels==1].min():.4f} max: {test_anom_probs[test_labels==1].max():.4f}")

# ── Anomaly Classification Threshold Sweep ───────────────────────────────────
print("\n=== SWEEPING THRESHOLD FOR ANOMALY CLASS (Class 0) ===")
print("We classify an image as ANOMALY if prob(anomaly) >= threshold, else NORMAL.")
print(f"  {'Thresh':>7} | {'Recall':>7} | {'Prec':>7} | {'F1':>7} | {'TP (Anom)':>9} {'FP (Norm)':>9} {'FN (Anom)':>9} {'TN (Norm)':>9}")
print("-" * 90)

# Threshold list from 0.05 to 0.95
for t in [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95]:
    # If prob(anomaly) >= t, predict anomaly (0), else normal (1)
    preds = np.where(test_anom_probs >= t, 0, 1)
    
    # Confusion matrix:
    # cm[0,0]: True Anomaly (TP for anomaly)
    # cm[0,1]: False Normal (FN for anomaly)
    # cm[1,0]: False Anomaly (FP for anomaly)
    # cm[1,1]: True Normal (TN for anomaly)
    cm = confusion_matrix(test_labels, preds)
    
    tp = cm[0,0]
    fn = cm[0,1]
    fp = cm[1,0]
    tn = cm[1,1]
    
    # Metrics for anomaly class (0)
    # To compute precision/recall for anomaly class, we treat anomaly as positive class (1)
    # and normal as negative class (0)
    y_true_binary = (test_labels == 0).astype(int)
    y_pred_binary = (preds == 0).astype(int)
    
    prec = precision_score(y_true_binary, y_pred_binary, zero_division=0)
    rec = recall_score(y_true_binary, y_pred_binary, zero_division=0)
    f1 = f1_score(y_true_binary, y_pred_binary, zero_division=0)
    
    print(f"  {t:>7.2f} | {rec:>7.4f} | {prec:>7.4f} | {f1:>7.4f} | "
          f"{tp:>9} {fp:>9} {fn:>9} {tn:>9}")

# ── Normal Classification Threshold Sweep ─────────────────────────────────────
print("\n=== SWEEPING THRESHOLD FOR NORMAL CLASS (Class 1) ===")
print("We classify an image as NORMAL if prob(normal) >= threshold, else ANOMALY.")
print(f"  {'Thresh':>7} | {'Recall':>7} | {'Prec':>7} | {'F1':>7} | {'TP (Norm)':>9} {'FP (Anom)':>9} {'FN (Norm)':>9} {'TN (Anom)':>9}")
print("-" * 90)

for t in [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95]:
    # If prob(normal) >= t, predict normal (1), else anomaly (0)
    preds = np.where(test_norm_probs >= t, 1, 0)
    
    cm = confusion_matrix(test_labels, preds)
    
    tn_anom = cm[0,0] # True Anomaly
    fp_norm = cm[0,1] # False Normal
    fn_norm = cm[1,0] # False Anomaly
    tp_norm = cm[1,1] # True Normal
    
    y_true_binary = (test_labels == 1).astype(int)
    y_pred_binary = (preds == 1).astype(int)
    
    prec = precision_score(y_true_binary, y_pred_binary, zero_division=0)
    rec = recall_score(y_true_binary, y_pred_binary, zero_division=0)
    f1 = f1_score(y_true_binary, y_pred_binary, zero_division=0)
    
    print(f"  {t:>7.2f} | {rec:>7.4f} | {prec:>7.4f} | {f1:>7.4f} | "
          f"{tp_norm:>9} {fp_norm:>9} {fn_norm:>9} {tn_anom:>9}")
