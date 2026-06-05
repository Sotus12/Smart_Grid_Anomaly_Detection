"""
STAR-Net v2 — Post-Training Diagnostic
========================================
Run this after training completes to inspect:
1. Actual anomaly probability distributions (normal vs anomaly)
2. What threshold achieves 85%+ recall on the test set
3. Full confusion matrix sweep across thresholds
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
    confusion_matrix, roc_auc_score
)

sys.path.insert(0, str(Path(__file__).parent))
from star_net import build_star_net
from acfl_loss import AsymmetricContrastiveFocalLoss

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

test_ds  = datasets.ImageFolder(os.path.join(DATA_DIR, 'test'),  val_transform)
val_ds   = datasets.ImageFolder(os.path.join(DATA_DIR, 'val'),   val_transform)

print(f"Class mapping: {test_ds.class_to_idx}")
# Identify which index is anomaly
anomaly_idx = test_ds.class_to_idx.get('anomaly', 0)
print(f"Anomaly class index: {anomaly_idx}")

test_loader = DataLoader(test_ds, batch_size=32, shuffle=False, num_workers=0)
val_loader  = DataLoader(val_ds,  batch_size=32, shuffle=False, num_workers=0)

model = build_star_net(num_classes=2, metric_dim=64, dropout=0.3).to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()
print(f"[OK] Model loaded from {MODEL_PATH}")

@torch.no_grad()
def get_probs_labels(loader, anomaly_cls_idx):
    all_probs  = []
    all_labels = []
    for images, labels in loader:
        images = images.to(device)
        logits = model(images)
        probs  = torch.softmax(logits, dim=1)[:, anomaly_cls_idx]
        all_probs.extend(probs.cpu().numpy())
        # Remap: anomaly=1, normal=0 for sklearn metrics
        remapped = [1 if l == anomaly_cls_idx else 0 for l in labels.numpy()]
        all_labels.extend(remapped)
    return np.array(all_probs), np.array(all_labels)

print("\n--- Collecting probabilities ---")
val_probs,  val_labels  = get_probs_labels(val_loader,  anomaly_idx)
test_probs, test_labels = get_probs_labels(test_loader, anomaly_idx)

# ── Distribution Analysis ──────────────────────────────────────────────────
print("\n=== PROBABILITY DISTRIBUTION ===")
print(f"VAL SET:")
print(f"  Normal  samples — mean prob: {val_probs[val_labels==0].mean():.4f}  "
      f"min: {val_probs[val_labels==0].min():.4f}  max: {val_probs[val_labels==0].max():.4f}")
print(f"  Anomaly samples — mean prob: {val_probs[val_labels==1].mean():.4f}  "
      f"min: {val_probs[val_labels==1].min():.4f}  max: {val_probs[val_labels==1].max():.4f}")

print(f"\nTEST SET:")
print(f"  Normal  samples — mean prob: {test_probs[test_labels==0].mean():.4f}  "
      f"min: {test_probs[test_labels==0].min():.4f}  max: {test_probs[test_labels==0].max():.4f}")
print(f"  Anomaly samples — mean prob: {test_probs[test_labels==1].mean():.4f}  "
      f"min: {test_probs[test_labels==1].min():.4f}  max: {test_probs[test_labels==1].max():.4f}")

# ── Threshold Sweep on Val, Apply to Test ─────────────────────────────────
print("\n=== THRESHOLD SWEEP (val -> test) ===")
thresholds = np.linspace(0.001, 0.999, 500)

best_val_recall = 0.0
best_thresh = 0.5
for t in thresholds:
    preds = (val_probs >= t).astype(int)
    rec  = recall_score(val_labels, preds, zero_division=0)
    prec = precision_score(val_labels, preds, zero_division=0)
    if prec >= 0.08 and rec > best_val_recall:
        best_val_recall = rec
        best_thresh = t

print(f"\nBest val threshold (recall, prec>=8%): {best_thresh:.4f}")
print(f"Val recall at this threshold: {best_val_recall:.4f}")

# Apply to test
test_preds = (test_probs >= best_thresh).astype(int)
cm = confusion_matrix(test_labels, test_preds, labels=[0,1])

print(f"\n=== TEST RESULTS (thresh={best_thresh:.4f}) ===")
print(f"  Accuracy:  {accuracy_score(test_labels, test_preds):.4f}")
print(f"  Precision: {precision_score(test_labels, test_preds, zero_division=0):.4f}")
print(f"  Recall:    {recall_score(test_labels, test_preds, zero_division=0):.4f}")
print(f"  F1:        {f1_score(test_labels, test_preds, zero_division=0):.4f}")
try:
    print(f"  ROC-AUC:   {roc_auc_score(test_labels, test_probs):.4f}")
except:
    pass
print(f"  Confusion Matrix: TN={cm[0,0]} FP={cm[0,1]} FN={cm[1,0]} TP={cm[1,1]}")

# ── Print full sweep for multiple thresholds ───────────────────────────────
print(f"\n=== MULTI-THRESHOLD SWEEP (TEST SET) ===")
print(f"  {'Thresh':>7} | {'Recall':>7} | {'Prec':>7} | {'F1':>7} | {'TN':>5} {'FP':>5} {'FN':>5} {'TP':>5}")
print("  " + "-"*65)
for t in [0.01, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]:
    p = (test_probs >= t).astype(int)
    cm2 = confusion_matrix(test_labels, p, labels=[0,1])
    rec  = recall_score(test_labels, p, zero_division=0)
    prec = precision_score(test_labels, p, zero_division=0)
    f1   = f1_score(test_labels, p, zero_division=0)
    flag = " <-- TARGET" if rec >= 0.85 else ""
    print(f"  {t:>7.3f} | {rec:>7.4f} | {prec:>7.4f} | {f1:>7.4f} | "
          f"{cm2[0,0]:>5} {cm2[0,1]:>5} {cm2[1,0]:>5} {cm2[1,1]:>5}{flag}")

garesnet_recall = 0.7895
final_recall = recall_score(test_labels, test_preds, zero_division=0)
print(f"\n  GAResNet recall: {garesnet_recall:.4f}")
print(f"  STAR-Net v2:     {final_recall:.4f}")
if final_recall > garesnet_recall:
    print(f"  [SUCCESS] STAR-Net v2 BEATS GAResNet by {(final_recall-garesnet_recall)*100:.2f}%!")
else:
    print(f"  [INFO] Gap to GAResNet: {(garesnet_recall-final_recall)*100:.2f}%")
