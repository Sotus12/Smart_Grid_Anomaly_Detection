"""
STAR-Net Training Pipeline v2 — High-Recall Edition
=====================================================
TARGET: >85% recall on anomaly class (beats GAResNet's 78.95%)

KEY CHANGES vs v1:
==================
1. ACFL focal_alpha FIXED: alpha=0.75 (was 0.25!). In Focal Loss, alpha is the
   weight applied to the MINORITY class (anomaly=1). The v1 value of 0.25
   was effectively DOWNWEIGHTING anomalies, causing the model to miss them.

2. class_weights in CE baseline: Added explicit nn.CrossEntropyLoss weight
   vector to directly balance the loss gradient. This is orthogonal to focal
   loss and compounds the effect.

3. Threshold strategy changed: Instead of threshold = argmax(F1), we now use
   threshold = argmax(Recall) subject to Precision >= 0.10. This guarantees
   maximum anomaly catch rate while avoiding trivially predicting everything
   as anomalous.

4. Training duration extended: 50 epochs with patience=15 (more time to converge).

5. Stronger mixup: probability raised to 0.6 with label smoothing regularization.

6. ACFL hyperparameters:
   - focal_gamma: 3.0 (was 2.0) — more aggressive hard-example mining
   - margin_pos:  1.5 (was 1.0) — wider anomaly-to-normal separation
   - lambda_contrast: 0.2 (was 0.3) — reduce contrastive term, let focal lead

7. LR increased: 1e-3 (was 5e-4) with more warmup (40% of cycle).
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
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import transforms, datasets
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score,
    recall_score, roc_auc_score, confusion_matrix, roc_curve,
    precision_recall_curve
)
from tqdm import tqdm

# ── Path Setup ──────────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))

from star_net import build_star_net, count_parameters
from acfl_loss import AsymmetricContrastiveFocalLoss


# ════════════════════════════════════════════════════════════════════════════
# DEFAULT PATHS
# ════════════════════════════════════════════════════════════════════════════

DEFAULT_DATA_DIR  = r'C:\SOFTWARE\DL Project\outputs\spectrograms'
DEFAULT_MODEL_DIR = r'C:\SOFTWARE\DL Project\models'
DEFAULT_LOG_FILE  = r'C:\SOFTWARE\DL Project\star_net_v2_training.log'


# ════════════════════════════════════════════════════════════════════════════
# UTILITIES
# ════════════════════════════════════════════════════════════════════════════

class Logger:
    """Writes to both stdout and a log file simultaneously."""
    def __init__(self, log_path):
        self.log_path = log_path
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        with open(log_path, 'w', encoding='utf-8', errors='replace') as f:
            f.write("STAR-Net v2 Training Log\n")
            f.write("=" * 60 + "\n")

    def log(self, msg):
        safe_msg = msg.encode('ascii', errors='replace').decode('ascii')
        print(safe_msg)
        with open(self.log_path, 'a', encoding='utf-8', errors='replace') as f:
            f.write(msg + "\n")


class EarlyStopping:
    """Stop training when validation recall stops improving."""
    def __init__(self, patience=15, min_delta=0.005):
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
    """Caches PIL images in memory to completely bypass slow disk reads."""
    def __init__(self, dataset, transform=None):
        self.samples = []
        self.labels = []
        self.transform = transform

        print(f"  Caching {len(dataset)} images to RAM...")
        for i in range(len(dataset)):
            path, label = dataset.imgs[i]
            img = dataset.loader(path)
            self.samples.append(img)
            self.labels.append(label)
        print(f"  [OK] {len(self.samples)} images cached.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img = self.samples[idx]
        label = self.labels[idx]
        if self.transform:
            img = self.transform(img)
        return img, label


class CachedTensorDataset(torch.utils.data.Dataset):
    """Caches preprocessed tensors. Zero computation during eval."""
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
    """WeightedRandomSampler to balance class frequencies."""
    labels = [label for _, label in dataset]
    class_counts = {0: labels.count(0), 1: labels.count(1)}
    total = len(labels)
    weights = {cls: total / cnt for cls, cnt in class_counts.items()}
    sample_weights = [weights[l] for l in labels]
    return WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)


def compute_class_weights(dataset):
    """Compute inverse-frequency class weights for CrossEntropyLoss."""
    labels = [label for _, label in dataset]
    counts = np.bincount(labels)
    n_total = len(labels)
    # Inverse frequency: minority class gets higher weight
    weights = n_total / (len(counts) * counts)
    return torch.FloatTensor(weights)


def find_optimal_threshold_recall(y_true, y_probs, min_precision=0.08):
    """
    Find threshold that maximizes recall subject to precision >= min_precision.

    This is critical for anomaly detection where missing an anomaly (FN) is
    far worse than a false alarm (FP). We accept lower precision to catch more
    true anomalies.

    Args:
        y_true: Ground truth labels (0=normal, 1=anomaly)
        y_probs: Model's predicted probability for the anomaly class
        min_precision: Minimum acceptable precision (default 8%)

    Returns:
        (best_threshold, best_recall, achieved_precision, achieved_f1)
    """
    y_true = np.array(y_true)
    y_probs = np.array(y_probs)

    # Sweep thresholds from high to low (low threshold = more recall)
    thresholds = np.linspace(0.01, 0.99, 200)

    best_recall = 0.0
    best_thresh = 0.5
    best_prec   = 0.0
    best_f1     = 0.0

    for t in thresholds:
        preds = (y_probs >= t).astype(int)
        rec  = recall_score(y_true, preds, zero_division=0)
        prec = precision_score(y_true, preds, zero_division=0)
        f1   = f1_score(y_true, preds, zero_division=0)

        if prec >= min_precision and rec > best_recall:
            best_recall = rec
            best_thresh = float(t)
            best_prec   = prec
            best_f1     = f1

    # If no threshold satisfies min_precision, fall back to max-recall threshold
    if best_recall == 0.0:
        for t in thresholds:
            preds = (y_probs >= t).astype(int)
            rec  = recall_score(y_true, preds, zero_division=0)
            if rec > best_recall:
                best_recall = rec
                best_thresh = float(t)
                best_prec   = precision_score(y_true, preds, zero_division=0)
                best_f1     = f1_score(y_true, preds, zero_division=0)

    return best_thresh, best_recall, best_prec, best_f1


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

def train_one_epoch(model, loader, criterion, class_weights_ce, optimizer,
                    scheduler, device, use_contrastive, mixup_prob=0.6,
                    label_smoothing=0.05):
    """
    Train for one epoch with:
    - Weighted CrossEntropy as an auxiliary signal (compounded with ACFL)
    - Mixup augmentation at increased probability
    - Label smoothing for regularization
    """
    model.train()
    running_loss   = 0.0
    all_preds      = []
    all_labels     = []

    # Weighted CE loss — hard class-weight signal for anomaly upweighting
    ce_criterion = nn.CrossEntropyLoss(
        weight=class_weights_ce.to(device),
        label_smoothing=label_smoothing
    )

    pbar = tqdm(loader, desc='  Train', leave=False, ncols=90)
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)

        # Mixup augmentation (more frequent in v2: 60% of batches)
        apply_mixup = (np.random.random() < mixup_prob)
        if apply_mixup:
            lam   = np.random.beta(0.4, 0.4)  # Stronger mixing than v1
            idx   = torch.randperm(images.size(0), device=device)
            images_mix = lam * images + (1 - lam) * images[idx]
            labels_a, labels_b = labels, labels[idx]
        else:
            images_mix = images

        optimizer.zero_grad()

        # Forward pass
        logits, internals = model(images_mix, return_internals=True)
        metric_feat = internals['metric_feat'] if use_contrastive else None

        # ACFL loss
        if apply_mixup:
            acfl_loss = (lam * criterion(logits, labels_a, metric_feat) +
                         (1 - lam) * criterion(logits, labels_b, metric_feat))
            ce_loss = (lam * ce_criterion(logits, labels_a) +
                       (1 - lam) * ce_criterion(logits, labels_b))
        else:
            acfl_loss = criterion(logits, labels, metric_feat)
            ce_loss = ce_criterion(logits, labels)

        # Compound loss: ACFL + weighted CE (jointly optimize recall-friendly signals)
        loss = 0.7 * acfl_loss + 0.3 * ce_loss

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        running_loss += loss.item()
        _, preds = torch.max(logits, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

        pbar.set_postfix({'loss': f'{loss.item():.4f}'})

    n = len(loader)
    return (
        running_loss / n,
        accuracy_score(all_labels, all_preds),
        recall_score(all_labels, all_preds, zero_division=0),
    )


@torch.no_grad()
def validate(model, loader, criterion, device, use_contrastive):
    """Validate the model. Returns soft probabilities for threshold search."""
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

        probs = torch.softmax(logits, dim=1)[:, 1]
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
        description='Train STAR-Net v2 — High-Recall Edition'
    )
    parser.add_argument('--data-dir',       type=str,   default=DEFAULT_DATA_DIR)
    parser.add_argument('--model-dir',      type=str,   default=DEFAULT_MODEL_DIR)
    parser.add_argument('--epochs',         type=int,   default=30)
    parser.add_argument('--warmup-epochs',  type=int,   default=5)
    parser.add_argument('--batch-size',     type=int,   default=32)
    parser.add_argument('--lr',             type=float, default=1e-3)
    parser.add_argument('--metric-dim',     type=int,   default=64)
    # KEY FIX: alpha=0.75 (anomaly class heavily upweighted)
    parser.add_argument('--focal-gamma',    type=float, default=3.0)
    parser.add_argument('--focal-alpha',    type=float, default=0.75,
                        help='Alpha for anomaly class. Must be >> 0.5 to upweight anomalies.')
    parser.add_argument('--margin-pos',     type=float, default=1.5)
    parser.add_argument('--margin-neg',     type=float, default=0.5)
    parser.add_argument('--lambda-contrast',type=float, default=0.2)
    parser.add_argument('--patience',       type=int,   default=15)
    parser.add_argument('--dropout',        type=float, default=0.3)
    parser.add_argument('--min-precision',  type=float, default=0.22,
                        help='Minimum precision floor for recall-optimized threshold')
    args = parser.parse_args()

    # ── Setup ───────────────────────────────────────────────────────────────
    os.makedirs(args.model_dir, exist_ok=True)
    logger = Logger(DEFAULT_LOG_FILE)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    logger.log("=" * 60)
    logger.log("  STAR-Net v2: High-Recall Edition")
    logger.log("  Loss: ACFL (alpha=0.75) + Weighted CE (compound)")
    logger.log("  Target: >85% Recall on Anomaly Class")
    logger.log("=" * 60)
    logger.log(f"  Device: {device}")
    logger.log(f"  Epochs: {args.epochs} ({args.warmup_epochs} warmup + "
               f"{args.epochs - args.warmup_epochs} full ACFL)")
    logger.log(f"  Batch:  {args.batch_size} | LR: {args.lr}")
    logger.log(f"  ACFL: gamma={args.focal_gamma}, alpha={args.focal_alpha}, "
               f"margin+={args.margin_pos}, lambda={args.lambda_contrast}")
    logger.log("")

    # ── Data Transforms ─────────────────────────────────────────────────────
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.4),
        transforms.RandomRotation(15),
        transforms.ColorJitter(
            brightness=0.4, contrast=0.4, saturation=0.3, hue=0.08
        ),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.3, scale=(0.02, 0.15), ratio=(0.3, 3.3)),
    ])

    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    # ── Datasets ──────────────────────────────────────────────────────────
    raw_train_ds = datasets.ImageFolder(os.path.join(args.data_dir, 'train'))
    raw_val_ds   = datasets.ImageFolder(os.path.join(args.data_dir, 'val'),   val_transform)
    raw_test_ds  = datasets.ImageFolder(os.path.join(args.data_dir, 'test'),  val_transform)

    class_to_idx = raw_train_ds.class_to_idx
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    train_labels = raw_train_ds.targets
    n_class0 = train_labels.count(0)
    n_class1 = train_labels.count(1)
    logger.log(f"  Class mapping: {class_to_idx}")
    logger.log(f"  Train: {n_class0} normal | {n_class1} anomaly "
               f"(ratio {n_class0/max(n_class1,1):.1f}:1)")
    logger.log(f"  Val: {len(raw_val_ds)} | Test: {len(raw_test_ds)}")
    logger.log("")

    # ── Compute class weights for CE auxiliary loss ────────────────────────
    counts = np.array([n_class0, n_class1])
    class_weights_ce = torch.FloatTensor(
        len(counts) * len(train_labels) / (counts * counts.sum())
    )
    logger.log(f"  CE class weights: normal={class_weights_ce[0]:.3f}, "
               f"anomaly={class_weights_ce[1]:.3f}")
    logger.log("")

    # ── Cache datasets to memory ───────────────────────────────────────────
    logger.log("  Caching datasets to memory...")
    train_ds = InMemoryDataset(raw_train_ds, train_transform)
    val_ds   = CachedTensorDataset(raw_val_ds)
    test_ds  = CachedTensorDataset(raw_test_ds)
    logger.log("  [OK] All datasets cached.")
    logger.log("")

    # ── Data Loaders ──────────────────────────────────────────────────────
    sampler = build_weighted_sampler(train_ds)
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, sampler=sampler,
        num_workers=0, pin_memory=False, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=0, pin_memory=False,
    )
    test_loader = DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=0, pin_memory=False,
    )

    # ── Model ──────────────────────────────────────────────────────────────
    logger.log("  Building STAR-Net v2...")
    model = build_star_net(
        num_classes=2,
        metric_dim=args.metric_dim,
        dropout=args.dropout,
    ).to(device)
    n_params = count_parameters(model)
    logger.log(f"  Parameters: {n_params:,}")
    logger.log("")

    # ── ACFL Loss (with corrected alpha) ──────────────────────────────────
    # CRITICAL: focal_alpha=0.75 means anomaly samples (class=1) get weight 0.75
    # while normal samples get weight (1-0.75)=0.25. This heavily penalizes
    # missing anomalies in the focal loss gradient.
    criterion = AsymmetricContrastiveFocalLoss(
        focal_gamma=args.focal_gamma,
        focal_alpha=args.focal_alpha,
        margin_pos=args.margin_pos,
        margin_neg=args.margin_neg,
        lambda_contrast=args.lambda_contrast,
    )
    logger.log(f"  ACFL: gamma={args.focal_gamma}, alpha={args.focal_alpha} [ANOMALY UPWEIGHTED]")
    logger.log(f"  Compound loss: 0.7*ACFL + 0.3*WeightedCE")

    # ── Optimizer & Scheduler ─────────────────────────────────────────────
    optimizer = optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=1e-4, eps=1e-8
    )
    scheduler = OneCycleLR(
        optimizer,
        max_lr=args.lr,
        epochs=args.epochs,
        steps_per_epoch=len(train_loader),
        pct_start=0.4,        # 40% warmup (more than v1's 30%)
        anneal_strategy='cos',
        div_factor=25.0,       # Start at lr/25
        final_div_factor=1e4,
    )
    logger.log(f"  Optimizer: AdamW lr={args.lr}, wd=1e-4")
    logger.log(f"  Scheduler: OneCycleLR (pct_start=0.40, div=25)")
    logger.log("")

    # ── Training Loop ──────────────────────────────────────────────────────
    logger.log("  Starting training...")
    logger.log("-" * 60)

    # Early stopping monitors F1-score
    early_stop = EarlyStopping(patience=args.patience, min_delta=0.002)
    best_val_f1 = -1.0
    best_model_path = os.path.join(args.model_dir, 'star_net_v2_best.pth')
    history = defaultdict(list)

    for epoch in range(1, args.epochs + 1):
        use_contrastive = (epoch > args.warmup_epochs)
        phase_label = "Full ACFL" if use_contrastive else f"Warmup ({epoch}/{args.warmup_epochs})"

        # Train
        tr_loss, tr_acc, tr_rec = train_one_epoch(
            model, train_loader, criterion, class_weights_ce,
            optimizer, scheduler, device,
            use_contrastive=use_contrastive,
            mixup_prob=0.6,
            label_smoothing=0.05,
        )

        # Validate
        val_loss, val_acc, val_f1, val_rec, _, val_labels, val_probs = validate(
            model, val_loader, criterion, device, use_contrastive=use_contrastive
        )

        # Recall-optimized threshold on val set
        val_thresh, val_rec_opt, val_prec_opt, val_f1_opt = find_optimal_threshold_recall(
            val_labels, val_probs, min_precision=args.min_precision
        )

        history['train_loss'].append(tr_loss)
        history['train_acc'].append(tr_acc)
        history['train_recall'].append(tr_rec)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['val_f1'].append(val_f1)
        history['val_recall_default'].append(val_rec)
        history['val_recall_tuned'].append(val_rec_opt)

        logger.log(
            f"  Epoch {epoch:02d}/{args.epochs} [{phase_label:20s}] | "
            f"TrLoss={tr_loss:.4f} TrRec={tr_rec:.4f} | "
            f"ValLoss={val_loss:.4f} ValF1={val_f1:.4f} | "
            f"ValRec(default)={val_rec:.4f} ValRec(tuned)={val_rec_opt:.4f} "
            f"[thresh={val_thresh:.3f}]"
        )

        # Save best by tuned F1 (ensures solid representation learning and no trivial collapse)
        if val_f1_opt > best_val_f1:
            best_val_f1 = val_f1_opt
            torch.save(model.state_dict(), best_model_path)
            logger.log(f"    [BEST] Saved — Val F1(tuned)={val_f1_opt:.4f} Recall(tuned)={val_rec_opt:.4f} "
                       f"Prec={val_prec_opt:.4f} thresh={val_thresh:.4f}")

        # Early stopping disabled to allow full OneCycleLR cosine decay phase convergence
        # if use_contrastive:
        #     early_stop(val_f1_opt)
        #     if early_stop.stop:
        #         logger.log(f"\n  Early stopping at epoch {epoch}.")
        #         break

    logger.log("")
    logger.log("=" * 60)
    logger.log("  Training complete. Running final evaluation...")
    logger.log("=" * 60)

    # ── Load Best Model ────────────────────────────────────────────────────
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    model.eval()

    # ── Threshold Tuning on Validation Set ────────────────────────────────
    _, _, _, _, _, val_labels, val_probs = validate(
        model, val_loader, criterion, device, use_contrastive=True
    )
    opt_thresh, opt_recall, opt_prec, opt_f1 = find_optimal_threshold_recall(
        val_labels, val_probs, min_precision=args.min_precision
    )
    logger.log(f"\n  Recall-optimized threshold (from val): {opt_thresh:.4f}")
    logger.log(f"  Val Recall={opt_recall:.4f} | Val Precision={opt_prec:.4f} | Val F1={opt_f1:.4f}")

    # ── Test Set Evaluation ────────────────────────────────────────────────
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

    # Compare vs GAResNet
    garesnet_recall = 0.7895
    star_recall = test_metrics['recall']
    if star_recall > garesnet_recall:
        logger.log(f"\n  [SUCCESS] STAR-Net v2 ({star_recall:.2%}) BEATS GAResNet ({garesnet_recall:.2%})!")
    else:
        logger.log(f"\n  [INFO] STAR-Net v2: {star_recall:.2%} | GAResNet: {garesnet_recall:.2%}")
        logger.log(f"  Gap: {(garesnet_recall - star_recall):.2%} — try lowering min-precision threshold")

    # ── Save Results ───────────────────────────────────────────────────────
    results = {
        'model_name':    'STAR-Net v2 (High-Recall Edition)',
        'version':       'v2',
        'architecture':  {
            'name': 'Spectral-Temporal Anomaly Reasoning Network',
            'components': [
                'SDS: Spectral Decomposition Stream (frequency-axis 2D convolutions)',
                'TDS: Temporal Dynamics Stream (time-axis dilated 2D convolutions)',
                'CSTFG: Cross-Spectral-Temporal Fusion Gate (differentiable gate)',
                'AIH: Anomaly Isolation Head (L2-normalized metric space)',
            ],
            'loss_function': 'ACFL(alpha=0.75, gamma=3.0) + 0.3*WeightedCE',
            'parameters': n_params,
        },
        'hyperparameters': {
            **vars(args),
            'compound_loss_weights': '0.7*ACFL + 0.3*WeightedCE',
            'threshold_strategy': 'maximize recall s.t. precision >= min_precision',
        },
        'training': {
            'epochs_completed': epoch,
            'warmup_epochs':    args.warmup_epochs,
            'best_val_f1':      best_val_f1,
            'optimal_threshold': float(opt_thresh),
            'history': dict(history),
        },
        'test_metrics':   test_metrics,
        'comparison': {
            'GAResNet_recall':   garesnet_recall,
            'STAR_Net_v2_recall': test_metrics['recall'],
            'improvement':       test_metrics['recall'] - garesnet_recall,
            'beats_GAResNet':    test_metrics['recall'] > garesnet_recall,
        },
        'novelty_claims': [
            'Physics-informed dual-stream decomposition (SDS + TDS)',
            'Differentiable Cross-Spectral-Temporal Fusion Gate (CSTFG)',
            'Anomaly Isolation Head with L2-normalized metric space',
            'Asymmetric Contrastive Focal Loss with alpha=0.75 anomaly upweighting',
            'Compound loss: 0.7*ACFL + 0.3*WeightedCE for recall maximization',
            'Recall-optimized threshold selection (vs standard F1-optimized)',
        ],
    }

    results_path = os.path.join(args.model_dir, 'star_net_v2_metrics.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)

    logger.log(f"\n  Results saved: {results_path}")
    logger.log(f"  Best model:   {best_model_path}")
    logger.log("\n  [DONE] STAR-Net v2 training complete.")


if __name__ == '__main__':
    main()
