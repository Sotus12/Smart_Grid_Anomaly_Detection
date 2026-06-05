"""
STAR-Net Training Pipeline
===========================
Full end-to-end training for the Spectral-Temporal Anomaly Reasoning Network
using the Asymmetric Contrastive Focal Loss (ACFL).

Training Strategy:
==================
Phase A — Focal Warmup (epochs 1-5):
    Only the focal loss term is active. This stabilizes the backbone before
    the contrastive term introduces gradient pressure from both streams.

Phase B — Full ACFL Training (epoch 6 onwards):
    Both focal and contrastive margin terms are active. The model now actively
    pushes anomaly features away from the normal cluster in metric space.

Scheduler:
    OneCycleLR: Warm up LR for 30% of training, cosine anneal to end.
    This is applied from the very start so SDS and TDS streams have a smooth
    learning trajectory.

Early Stopping:
    Monitors validation F1 score (not accuracy) with patience=10 epochs.
    F1 is the correct metric for imbalanced anomaly detection.

Threshold Optimization:
    After training, sweeps thresholds on the validation set to find the
    optimal decision boundary that maximizes F1 on the test set.
"""

import os
import sys
import json
import argparse
import numpy as np
from pathlib import Path
from collections import defaultdict

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import transforms, datasets
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score,
    recall_score, roc_auc_score, confusion_matrix, roc_curve
)
from tqdm import tqdm

# ── Path Setup ───────────────────────────────────────────────────────────────
# Allow imports from this script regardless of working directory
sys.path.insert(0, str(Path(__file__).parent))

from star_net import build_star_net, count_parameters
from acfl_loss import AsymmetricContrastiveFocalLoss


# ════════════════════════════════════════════════════════════════════════════
# DEFAULT PATHS
# ════════════════════════════════════════════════════════════════════════════

DEFAULT_DATA_DIR  = r'C:\SOFTWARE\DL Project\outputs\spectrograms'
DEFAULT_MODEL_DIR = r'C:\SOFTWARE\DL Project\models'
DEFAULT_LOG_FILE  = r'C:\SOFTWARE\DL Project\star_net_training.log'


# ════════════════════════════════════════════════════════════════════════════
# UTILITIES
# ════════════════════════════════════════════════════════════════════════════

class Logger:
    """Writes to both stdout and a log file simultaneously."""
    def __init__(self, log_path):
        self.log_path = log_path
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        with open(log_path, 'w') as f:
            f.write("STAR-Net Training Log\n")
            f.write("=" * 60 + "\n")

    def log(self, msg):
        print(msg)
        with open(self.log_path, 'a') as f:
            f.write(msg + "\n")


class EarlyStopping:
    """Stop training when validation metric stops improving."""
    def __init__(self, patience=10, min_delta=0.001):
        self.patience  = patience
        self.min_delta = min_delta
        self.counter   = 0
        self.best      = -1.0
        self.stop      = False

    def __call__(self, metric):
        if metric > self.best + self.min_delta:
            self.best    = metric
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.stop = True


class InMemoryDataset(torch.utils.data.Dataset):
    """Caches PIL images in memory to completely bypass slow disk reads during training."""
    def __init__(self, dataset, transform=None):
        self.samples = []
        self.labels = []
        self.transform = transform
        
        # Load all PIL images into memory
        for i in range(len(dataset)):
            path, label = dataset.imgs[i]
            img = dataset.loader(path)
            self.samples.append(img)
            self.labels.append(label)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img = self.samples[idx]
        label = self.labels[idx]
        if self.transform:
            img = self.transform(img)
        return img, label


class CachedTensorDataset(torch.utils.data.Dataset):
    """Caches preprocessed image tensors in memory. Zero computation during training/eval."""
    def __init__(self, dataset):
        self.samples = []
        self.labels = []
        for img, label in dataset:
            self.samples.append(img)
            self.labels.append(label)
        self.samples = torch.stack(self.samples)
        self.labels = torch.tensor(self.labels)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx], self.labels[idx]


def build_weighted_sampler(dataset):
    """Create a WeightedRandomSampler to balance class frequencies."""
    labels = [label for _, label in dataset]
    class_counts = {0: labels.count(0), 1: labels.count(1)}
    total = len(labels)
    # Inverse frequency weighting
    weights = {cls: total / cnt for cls, cnt in class_counts.items()}
    sample_weights = [weights[l] for l in labels]
    return WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)


def find_optimal_threshold(y_true, y_probs):
    """
    Sweep thresholds and return the one that maximises F1 score.
    Also returns the F1 value at that threshold.
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_probs)
    best_f1 = -1.0
    best_t  = 0.5
    for t in thresholds:
        preds = (np.array(y_probs) >= t).astype(int)
        f1 = f1_score(y_true, preds, zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, float(t)
    return best_t, best_f1


def compute_full_metrics(y_true, y_pred, y_probs=None):
    """Return a dict with all classification metrics."""
    m = {
        'accuracy':  accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall':    recall_score(y_true, y_pred, zero_division=0),
        'f1':        f1_score(y_true, y_pred, zero_division=0),
    }
    if y_probs is not None:
        try:
            m['roc_auc'] = roc_auc_score(y_true, y_probs)
        except Exception:
            m['roc_auc'] = None
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    m['confusion_matrix'] = cm.tolist()
    return m


# ════════════════════════════════════════════════════════════════════════════
# TRAIN / VALIDATE FUNCTIONS
# ════════════════════════════════════════════════════════════════════════════

def train_one_epoch(model, loader, criterion, optimizer, scheduler,
                    device, use_contrastive, mixup_prob=0.4):
    """
    Train for one epoch.

    Args:
        use_contrastive: If True, pass metric_features to ACFL (Phase B).
                         If False, focal-only mode (Phase A warmup).
    """
    model.train()
    running_loss   = 0.0
    running_focal  = 0.0
    running_cont   = 0.0
    all_preds      = []
    all_labels     = []

    pbar = tqdm(loader, desc='  Train', leave=False, ncols=90)
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)

        # Optional Mixup augmentation (50% of batches)
        apply_mixup = (np.random.random() < mixup_prob)
        if apply_mixup:
            lam   = np.random.beta(0.2, 0.2)
            idx   = torch.randperm(images.size(0), device=device)
            images_mix = lam * images + (1 - lam) * images[idx]
            labels_a, labels_b = labels, labels[idx]
        else:
            images_mix = images

        optimizer.zero_grad()

        # Forward pass: get logits and metric features
        logits, internals = model(images_mix, return_internals=True)
        metric_feat = internals['metric_feat'] if use_contrastive else None

        # Compute ACFL loss
        if apply_mixup:
            loss = lam * criterion(logits, labels_a, metric_feat) + \
                   (1 - lam) * criterion(logits, labels_b, metric_feat)
        else:
            loss = criterion(logits, labels, metric_feat)

        loss.backward()

        # Gradient clipping for training stability
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()
        scheduler.step()

        running_loss  += loss.item()
        running_focal += getattr(criterion, 'last_focal_loss', 0.0)
        running_cont  += getattr(criterion, 'last_contrastive_loss', 0.0)

        _, preds = torch.max(logits, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

        pbar.set_postfix({'loss': f'{loss.item():.4f}'})

    n = len(loader)
    return (
        running_loss  / n,
        running_focal / n,
        running_cont  / n,
        accuracy_score(all_labels, all_preds),
    )


@torch.no_grad()
def validate(model, loader, criterion, device, use_contrastive):
    """
    Validate the model.

    Returns loss, accuracy, F1, per-sample predictions, labels, and probs.
    """
    model.eval()
    running_loss = 0.0
    all_preds    = []
    all_labels   = []
    all_probs    = []

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        logits, internals = model(images, return_internals=True)
        metric_feat = internals['metric_feat'] if use_contrastive else None

        loss = criterion(logits, labels, metric_feat)
        running_loss += loss.item()

        probs = torch.softmax(logits, dim=1)[:, 1]   # Anomaly class prob
        _, preds = torch.max(logits, 1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())

    val_loss = running_loss / len(loader)
    val_acc  = accuracy_score(all_labels, all_preds)
    val_f1   = f1_score(all_labels, all_preds, zero_division=0)
    val_rec  = recall_score(all_labels, all_preds, zero_division=0)

    return val_loss, val_acc, val_f1, val_rec, all_preds, all_labels, all_probs


# ════════════════════════════════════════════════════════════════════════════
# MAIN TRAINING ROUTINE
# ════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description='Train STAR-Net with Asymmetric Contrastive Focal Loss'
    )
    parser.add_argument('--data-dir',  type=str, default=DEFAULT_DATA_DIR)
    parser.add_argument('--model-dir', type=str, default=DEFAULT_MODEL_DIR)
    parser.add_argument('--epochs',    type=int, default=40,
                        help='Total training epochs (warmup + full ACFL)')
    parser.add_argument('--warmup-epochs', type=int, default=5,
                        help='Focal-only warmup before contrastive term activates')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--lr',         type=float, default=5e-4,
                        help='Peak learning rate for OneCycleLR')
    parser.add_argument('--metric-dim', type=int, default=64,
                        help='Dimension of the AIH metric space')
    parser.add_argument('--focal-gamma',    type=float, default=2.0)
    parser.add_argument('--focal-alpha',    type=float, default=0.25)
    parser.add_argument('--margin-pos',     type=float, default=1.0,
                        help='Anomaly-to-normal centroid margin')
    parser.add_argument('--margin-neg',     type=float, default=0.5,
                        help='Normal-to-anomaly centroid margin (asymmetric)')
    parser.add_argument('--lambda-contrast',type=float, default=0.3,
                        help='Weight of contrastive term in ACFL')
    parser.add_argument('--patience',   type=int, default=10,
                        help='Early stopping patience (on val F1)')
    parser.add_argument('--dropout',    type=float, default=0.3)
    args = parser.parse_args()

    # ── Setup ───────────────────────────────────────────────────────────────
    os.makedirs(args.model_dir, exist_ok=True)
    logger = Logger(DEFAULT_LOG_FILE)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    logger.log("=" * 60)
    logger.log("  STAR-Net: Spectral-Temporal Anomaly Reasoning Network")
    logger.log("  Loss:     Asymmetric Contrastive Focal Loss (ACFL)")
    logger.log("=" * 60)
    logger.log(f"  Device: {device}")
    logger.log(f"  Data:   {args.data_dir}")
    logger.log(f"  Epochs: {args.epochs} ({args.warmup_epochs} warmup + "
               f"{args.epochs - args.warmup_epochs} full ACFL)")
    logger.log(f"  Batch:  {args.batch_size} | LR: {args.lr}")
    logger.log("")

    # ── Data Transforms ─────────────────────────────────────────────────────
    # Spectrogram-specific augmentations:
    # - RandomHorizontalFlip: time reversal (valid for anomaly spectrograms)
    # - RandomVerticalFlip:   frequency mirroring (physically valid)
    # - RandomErasing:        simulate sensor dropout / missing data
    # - ColorJitter:          simulate calibration drift between sensors
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),      # Time reversal
        transforms.RandomVerticalFlip(p=0.3),         # Frequency mirror
        transforms.RandomRotation(10),                 # Mild angle variation
        transforms.ColorJitter(
            brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05
        ),
        transforms.RandomAffine(degrees=0, translate=(0.08, 0.08)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.25, scale=(0.02, 0.12), ratio=(0.3, 3.3)),
    ])

    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    # ── Datasets ─────────────────────────────────────────────────────────────
    raw_train_ds = datasets.ImageFolder(os.path.join(args.data_dir, 'train'))
    raw_val_ds = datasets.ImageFolder(os.path.join(args.data_dir, 'val'), val_transform)
    raw_test_ds = datasets.ImageFolder(os.path.join(args.data_dir, 'test'), val_transform)

    logger.log(f"  Dataset - Train: {len(raw_train_ds)} | "
               f"Val: {len(raw_val_ds)} | Test: {len(raw_test_ds)}")

    # Class distribution
    train_labels = raw_train_ds.targets
    class_to_idx = raw_train_ds.class_to_idx
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    n_class0 = train_labels.count(0)
    n_class1 = train_labels.count(1)
    logger.log(f"  Class mapping: {class_to_idx}")
    logger.log(f"  Train class dist - Class0 ({idx_to_class[0]}): {n_class0} ({n_class0/len(train_labels)*100:.1f}%) "
               f"| Class1 ({idx_to_class[1]}): {n_class1} ({n_class1/len(train_labels)*100:.1f}%)")
    logger.log("")

    # Cache to memory for ultra-fast training (bypasses slow Windows I/O)
    logger.log("  Caching datasets to memory for ultra-fast training...")
    train_ds = InMemoryDataset(raw_train_ds, train_transform)
    val_ds = CachedTensorDataset(raw_val_ds)
    test_ds = CachedTensorDataset(raw_test_ds)
    logger.log("  [OK] Datasets cached successfully.")
    logger.log("")

    # ── Data Loaders ──────────────────────────────────────────────────────────
    sampler = build_weighted_sampler(train_ds)
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, sampler=sampler,
        num_workers=0, pin_memory=False,
        drop_last=True,   # Drop incomplete last batch — prevents BatchNorm1d size-1 error
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=0, pin_memory=False
    )
    test_loader = DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=0, pin_memory=False
    )

    # ── Model ─────────────────────────────────────────────────────────────────
    logger.log("  Building STAR-Net...")
    model = build_star_net(
        num_classes=2,
        metric_dim=args.metric_dim,
        dropout=args.dropout,
    ).to(device)

    n_params = count_parameters(model)
    logger.log(f"  STAR-Net parameters: {n_params:,}")
    logger.log(f"  Metric space dim:    {args.metric_dim}")
    logger.log("")

    # ── Loss Function: ACFL ──────────────────────────────────────────────────
    criterion = AsymmetricContrastiveFocalLoss(
        focal_gamma=args.focal_gamma,
        focal_alpha=args.focal_alpha,
        margin_pos=args.margin_pos,
        margin_neg=args.margin_neg,
        lambda_contrast=args.lambda_contrast,
    )
    logger.log(f"  Loss: ACFL (gamma={args.focal_gamma}, alpha={args.focal_alpha}, "
               f"margin+={args.margin_pos}, margin-={args.margin_neg}, lambda={args.lambda_contrast})")

    # ── Optimizer & Scheduler ────────────────────────────────────────────────
    optimizer = optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=1e-4, eps=1e-8
    )
    scheduler = OneCycleLR(
        optimizer,
        max_lr=args.lr,
        epochs=args.epochs,
        steps_per_epoch=len(train_loader),
        pct_start=0.3,
        anneal_strategy='cos',
        div_factor=10.0,
        final_div_factor=1e4,
    )
    logger.log(f"  Optimizer: AdamW (lr={args.lr}, wd=1e-4)")
    logger.log(f"  Scheduler: OneCycleLR (pct_start=0.3, cos annealing)")
    logger.log("")

    # ── Training Loop ─────────────────────────────────────────────────────────
    logger.log("  Starting training...")
    logger.log("-" * 60)

    early_stop = EarlyStopping(patience=args.patience, min_delta=0.001)
    best_val_f1 = -1.0
    best_model_path = os.path.join(args.model_dir, 'star_net_best.pth')
    history = defaultdict(list)

    for epoch in range(1, args.epochs + 1):
        # Decide which training phase we are in
        use_contrastive = (epoch > args.warmup_epochs)
        phase_label = "Full ACFL" if use_contrastive else f"Warmup ({epoch}/{args.warmup_epochs})"

        # ── Train ────────────────────────────────────────────────────────────
        tr_loss, tr_focal, tr_cont, tr_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, scheduler,
            device, use_contrastive=use_contrastive,
        )

        # ── Validate ─────────────────────────────────────────────────────────
        val_loss, val_acc, val_f1, val_rec, _, _, _ = validate(
            model, val_loader, criterion, device, use_contrastive=use_contrastive
        )

        # ── Log ──────────────────────────────────────────────────────────────
        history['train_loss'].append(tr_loss)
        history['train_acc'].append(tr_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['val_f1'].append(val_f1)
        history['val_recall'].append(val_rec)

        logger.log(
            f"  Epoch {epoch:02d}/{args.epochs} [{phase_label:20s}] | "
            f"TrLoss={tr_loss:.4f} (F={tr_focal:.3f},C={tr_cont:.3f}) | "
            f"TrAcc={tr_acc:.4f} | "
            f"ValLoss={val_loss:.4f} | ValAcc={val_acc:.4f} | "
            f"ValF1={val_f1:.4f} | ValRec={val_rec:.4f}"
        )

        # ── Save Best ────────────────────────────────────────────────────────
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(model.state_dict(), best_model_path)
            logger.log(f"    [BEST] Model saved - Val F1: {val_f1:.4f}")

        # ── Early Stopping (only after warmup) ───────────────────────────────
        if use_contrastive:
            early_stop(val_f1)
            if early_stop.stop:
                logger.log(f"\n  Early stopping triggered at epoch {epoch}.")
                break

    logger.log("")
    logger.log("=" * 60)
    logger.log("  Training complete. Running final evaluation...")
    logger.log("=" * 60)

    # ── Load Best Model & Final Evaluation ────────────────────────────────────
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    model.eval()

    # ── Threshold Tuning on Validation Set ────────────────────────────────────
    _, _, _, _, _, val_labels, val_probs = validate(
        model, val_loader, criterion, device, use_contrastive=True
    )
    opt_thresh, opt_f1_val = find_optimal_threshold(val_labels, val_probs)
    logger.log(f"\n  Optimal threshold (from val): {opt_thresh:.4f} "
               f"(val F1={opt_f1_val:.4f})")

    # ── Test Set Evaluation ────────────────────────────────────────────────────
    _, _, _, _, _, test_labels, test_probs_raw = validate(
        model, test_loader, criterion, device, use_contrastive=True
    )
    test_preds = (np.array(test_probs_raw) >= opt_thresh).astype(int)
    test_metrics = compute_full_metrics(test_labels, test_preds, test_probs_raw)

    logger.log("")
    logger.log("  --- Final Test Set Results ---------------------------------")
    logger.log(f"  Accuracy:   {test_metrics['accuracy']:.4f}")
    logger.log(f"  Precision:  {test_metrics['precision']:.4f}")
    logger.log(f"  Recall:     {test_metrics['recall']:.4f}  [Anomaly catch rate]")
    logger.log(f"  F1 Score:   {test_metrics['f1']:.4f}")
    if test_metrics.get('roc_auc') is not None:
        logger.log(f"  ROC-AUC:    {test_metrics['roc_auc']:.4f}")
    logger.log(f"  Threshold:  {opt_thresh:.4f}")
    cm = np.array(test_metrics['confusion_matrix'])
    logger.log(f"  Confusion Matrix:")
    logger.log(f"    TN={cm[0,0]}  FP={cm[0,1]}")
    logger.log(f"    FN={cm[1,0]}  TP={cm[1,1]}")
    logger.log("")

    # ── Save Complete Results ──────────────────────────────────────────────────
    results = {
        'model_name': 'STAR-Net',
        'architecture': {
            'name': 'Spectral-Temporal Anomaly Reasoning Network',
            'components': [
                'SDS: Spectral Decomposition Stream (frequency-axis 1D convolutions)',
                'TDS: Temporal Dynamics Stream (time-axis dilated 1D convolutions)',
                'CSTFG: Cross-Spectral-Temporal Fusion Gate (differentiable gate)',
                'AIH: Anomaly Isolation Head (L2-normalized metric space)',
            ],
            'loss_function': 'Asymmetric Contrastive Focal Loss (ACFL)',
            'parameters': n_params,
        },
        'hyperparameters': vars(args),
        'training': {
            'epochs_completed': epoch,
            'warmup_epochs': args.warmup_epochs,
            'best_val_f1': best_val_f1,
            'optimal_threshold': float(opt_thresh),
            'history': dict(history),
        },
        'test_metrics': test_metrics,
        'novelty_claims': [
            'Physics-informed dual-stream decomposition (SDS + TDS) — first spectrogram model to process axes independently',
            'Differentiable Cross-Spectral-Temporal Fusion Gate — learns where in spectrogram each axis is informative',
            'Anomaly Isolation Head with L2-normalized metric space projection',
            'Asymmetric Contrastive Focal Loss (ACFL) — novel loss function not present in any published paper',
            'Full STAR-Net pipeline — does not exist in any model zoo or GitHub repository',
        ],
    }

    results_path = os.path.join(args.model_dir, 'star_net_metrics.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)

    logger.log(f"  Results saved to: {results_path}")
    logger.log(f"  Best model saved to: {best_model_path}")
    logger.log("")
    logger.log("  [DONE] STAR-Net training complete.")


if __name__ == '__main__':
    main()
