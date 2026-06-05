"""
GAResNet Training v2 - FIXED labels + balanced strategy.
Previous run: model collapsed to all-normal because WeightedRandomSampler + FocalLoss
double-corrected for imbalance. This version uses class-weighted CE loss only.
"""

import os, json, sys, time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import transforms, datasets
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, classification_report,
                             precision_recall_curve, confusion_matrix)

sys.path.insert(0, os.path.dirname(__file__))
from grid_resnet_model import garesnet18

# -- Config --
DATA_DIR   = r'C:\SOFTWARE\DL Project\outputs\spectrograms'
SAVE_PATH  = r'C:\SOFTWARE\DL Project\models\garesnet_fixed_best.pth'
METRICS_PATH = r'C:\SOFTWARE\DL Project\models\garesnet_fixed_metrics.json'
EPOCHS     = 60
LR         = 5e-4
BATCH      = 16
PATIENCE   = 20
DEVICE     = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# -- Data --
train_tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(0.2, 0.2, 0.2, 0.05),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

val_tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

train_ds = datasets.ImageFolder(os.path.join(DATA_DIR, 'train'), train_tf)
val_ds   = datasets.ImageFolder(os.path.join(DATA_DIR, 'val'),   val_tf)
test_ds  = datasets.ImageFolder(os.path.join(DATA_DIR, 'test'),  val_tf)

class_to_idx = train_ds.class_to_idx  # {'anomaly': 0, 'normal': 1}
idx_to_class = {v: k for k, v in class_to_idx.items()}
ANOMALY_IDX = class_to_idx['anomaly']  # 0
NORMAL_IDX  = class_to_idx['normal']   # 1

print(f"Device: {DEVICE}")
print(f"Class mapping: {class_to_idx}")

labels = [l for _, l in train_ds]
counts = np.bincount(labels)
print(f"Train: anomaly(0)={counts[0]}, normal(1)={counts[1]}")
print(f"Val:   {len(val_ds)},  Test: {len(test_ds)}")

# -- Strategy: WeightedRandomSampler to balance batches --
# But NO focal loss -- use standard weighted CrossEntropy instead
# This avoids the double-correction problem.
weights_per_class = 1.0 / counts.astype(float)
sample_weights = [weights_per_class[l] for l in labels]
sampler = WeightedRandomSampler(sample_weights, len(sample_weights))

train_loader = DataLoader(train_ds, batch_size=BATCH, sampler=sampler, num_workers=0)
val_loader   = DataLoader(val_ds,   batch_size=BATCH, shuffle=False, num_workers=0)
test_loader  = DataLoader(test_ds,  batch_size=BATCH, shuffle=False, num_workers=0)

# -- Class-weighted CrossEntropy (inverse frequency) --
# anomaly=0 is majority (456) -> lower weight
# normal=1 is minority (89)  -> higher weight
# Weight ratio: normal gets ~5x the weight of anomaly
class_weights = torch.tensor([1.0 / counts[0], 1.0 / counts[1]], dtype=torch.float32)
class_weights = class_weights / class_weights.sum() * 2  # normalize so they sum to 2
print(f"CE class weights: anomaly={class_weights[0]:.3f}, normal={class_weights[1]:.3f}")
class_weights = class_weights.to(DEVICE)

criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.05)

# -- Model --
model = garesnet18(num_classes=2).to(DEVICE)
params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"GAResNet-18 params: {params:,}")

optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=0.02)
scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)

# -- Training --
def train_one_epoch(model, loader, criterion, optimizer):
    model.train()
    total_loss, correct, total = 0, 0, 0
    all_preds, all_labels = [], []
    for imgs, lbls in loader:
        imgs, lbls = imgs.to(DEVICE), lbls.to(DEVICE)
        optimizer.zero_grad()
        out = model(imgs)
        loss = criterion(out, lbls)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()
        preds = out.argmax(1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(lbls.cpu().numpy())
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    return total_loss / len(loader), acc, f1


def evaluate(model, loader):
    model.eval()
    all_probs, all_labels = [], []
    with torch.no_grad():
        for imgs, lbls in loader:
            imgs = imgs.to(DEVICE)
            probs = torch.softmax(model(imgs), dim=1)
            all_probs.append(probs.cpu().numpy())
            all_labels.extend(lbls.numpy())
    all_probs = np.concatenate(all_probs)
    all_labels = np.array(all_labels)
    preds = all_probs.argmax(axis=1)

    acc = accuracy_score(all_labels, preds)
    macro_f1 = f1_score(all_labels, preds, average='macro', zero_division=0)

    # Per-class
    anomaly_f1 = f1_score(all_labels == ANOMALY_IDX, preds == ANOMALY_IDX, zero_division=0)
    normal_f1  = f1_score(all_labels == NORMAL_IDX,  preds == NORMAL_IDX,  zero_division=0)

    return acc, macro_f1, anomaly_f1, normal_f1, preds, all_labels, all_probs


# -- Main Loop --
best_macro_f1 = 0
patience_ctr = 0
start = time.time()

print(f"\n{'='*60}")
print(f"Training GAResNet-18 (v2 FIXED) for {EPOCHS} epochs")
print(f"{'='*60}")

for epoch in range(1, EPOCHS + 1):
    t_loss, t_acc, t_f1 = train_one_epoch(model, train_loader, criterion, optimizer)
    scheduler.step()
    v_acc, v_macro_f1, v_anom_f1, v_norm_f1, _, _, _ = evaluate(model, val_loader)

    elapsed = time.time() - start
    print(f"Ep {epoch:02d}/{EPOCHS} | Loss {t_loss:.4f} | TrainAcc {t_acc:.3f} | "
          f"ValAcc {v_acc:.3f} | MacroF1 {v_macro_f1:.3f} | AnomF1 {v_anom_f1:.3f} | NormF1 {v_norm_f1:.3f} | {elapsed:.0f}s")

    if v_macro_f1 > best_macro_f1:
        best_macro_f1 = v_macro_f1
        torch.save(model.state_dict(), SAVE_PATH)
        print(f"   -> Best model saved (MacroF1={best_macro_f1:.4f})")
        patience_ctr = 0
    else:
        patience_ctr += 1
        if patience_ctr >= PATIENCE:
            print(f"Early stopping at epoch {epoch}")
            break

# -- Final Evaluation --
print(f"\n{'='*60}")
print("Final Evaluation on Test Set")
print(f"{'='*60}")

model.load_state_dict(torch.load(SAVE_PATH, map_location=DEVICE, weights_only=True))
test_acc, test_macro_f1, test_anom_f1, test_norm_f1, test_preds, test_labels, test_probs = evaluate(model, test_loader)

print(f"\n--- Classification Report ---")
print(classification_report(test_labels, test_preds,
                            target_names=['anomaly(0)', 'normal(1)'], zero_division=0))

cm = confusion_matrix(test_labels, test_preds)
print(f"Confusion Matrix:")
print(f"                 pred_anomaly  pred_normal")
print(f"  true_anomaly     {cm[0][0]:>5}        {cm[0][1]:>5}")
print(f"  true_normal      {cm[1][0]:>5}        {cm[1][1]:>5}")

# Per-class detailed metrics
anom_prec = precision_score(test_labels == ANOMALY_IDX, test_preds == ANOMALY_IDX, zero_division=0)
anom_rec  = recall_score(test_labels == ANOMALY_IDX, test_preds == ANOMALY_IDX, zero_division=0)
norm_prec = precision_score(test_labels == NORMAL_IDX, test_preds == NORMAL_IDX, zero_division=0)
norm_rec  = recall_score(test_labels == NORMAL_IDX, test_preds == NORMAL_IDX, zero_division=0)

metrics = {
    'class_mapping': class_to_idx,
    'overall_accuracy': float(test_acc),
    'macro_f1': float(test_macro_f1),
    'anomaly': {
        'precision': float(anom_prec),
        'recall': float(anom_rec),
        'f1': float(test_anom_f1),
    },
    'normal': {
        'precision': float(norm_prec),
        'recall': float(norm_rec),
        'f1': float(test_norm_f1),
    },
    'confusion_matrix': cm.tolist(),
    'train_counts': {'anomaly': int(counts[0]), 'normal': int(counts[1])},
    'training_config': {
        'epochs_trained': epoch,
        'lr': LR,
        'batch_size': BATCH,
        'loss': 'CrossEntropyLoss(class_weighted, label_smoothing=0.05)',
        'optimizer': 'AdamW(weight_decay=0.02)',
        'scheduler': 'CosineAnnealingWarmRestarts(T0=10, Tmult=2)',
        'sampler': 'WeightedRandomSampler',
    }
}

print(f"\n{'='*60}")
print("SUMMARY")
print(f"{'='*60}")
print(f"Overall Accuracy:    {metrics['overall_accuracy']:.4f}")
print(f"Macro F1:            {metrics['macro_f1']:.4f}")
print(f"Anomaly Precision:   {metrics['anomaly']['precision']:.4f}")
print(f"Anomaly Recall:      {metrics['anomaly']['recall']:.4f}")
print(f"Anomaly F1:          {metrics['anomaly']['f1']:.4f}")
print(f"Normal Precision:    {metrics['normal']['precision']:.4f}")
print(f"Normal Recall:       {metrics['normal']['recall']:.4f}")
print(f"Normal F1:           {metrics['normal']['f1']:.4f}")

with open(METRICS_PATH, 'w') as f:
    json.dump(metrics, f, indent=2)
print(f"\nSaved: {METRICS_PATH}")
print(f"Model: {SAVE_PATH}")
print("DONE.")
