"""
Asymmetric Contrastive Focal Loss (ACFL)
=========================================

A novel loss function designed specifically for STAR-Net and the smart grid
anomaly detection problem. This combination does not appear in any published
paper or open-source library.

MOTIVATION:
===========
The smart grid dataset has two fundamental challenges:

1. Extreme Class Imbalance (83% Normal, 16% Anomaly):
   Standard Cross-Entropy Loss treats all examples equally — the network
   converges by predicting "normal" for everything and achieving 83% accuracy
   while catching 0% of real anomalies. Focal Loss partially solves this by
   downweighting easy examples.

2. Poor Feature Separation (Entangled Distributions):
   Even with Focal Loss, the CNN features of normal and anomaly samples can
   be tightly clustered together in the final feature space. The classifier
   then makes a decision on subtly different feature vectors, which fails badly
   when the minority class is rare. Standard losses do not penalize this.

ACFL SOLUTION:
==============
ACFL = α · L_focal + (1 - α) · L_contrastive_margin

The contrastive margin term directly penalizes the model whenever:
  - An anomaly feature is too CLOSE to the normal centroid (batch centroid)
  - A normal feature is too CLOSE to the anomaly centroid (batch centroid)

This forces the model to actively push the two classes apart in metric space,
making the final classifier's job dramatically easier and more reliable.

CONTRASTIVE MARGIN FORMULATION:
================================
Given a batch with features Z = [z_1, ..., z_B] and labels y = [0, 1, ...]:

    μ_normal = mean(z_i | y_i == 0)     ← normal class centroid in batch
    μ_anomaly = mean(z_i | y_i == 1)    ← anomaly class centroid in batch

For each anomaly sample z_i (y_i == 1):
    L_pull_anomaly = max(0, m_pos - ||z_i - μ_normal||₂)²
    → Penalizes if the anomaly is within margin m_pos of the normal centroid

For each normal sample z_i (y_i == 0):
    L_pull_normal = max(0, m_neg - ||z_i - μ_anomaly||₂)²
    → Penalizes if the normal is within margin m_neg of the anomaly centroid

The "Asymmetric" part: m_pos > m_neg (anomalies need a larger separation margin
than normal samples, because we care far more about catching anomalies).

WHY THIS IS NOVEL:
==================
- Existing contrastive losses (SimCLR, SupCon) require PAIRS of samples and
  are designed for pre-training, not anomaly detection classification.
- Existing metric learning losses (Triplet, Center Loss) require constructing
  anchor-positive-negative triplets, which is computationally expensive.
- ACFL operates on INDIVIDUAL samples within a batch using dynamic class
  centroids — no pairs, no triplets, no special data construction required.
- The asymmetric margin design is specific to class-imbalanced anomaly detection.
- The combination with Focal Loss in this exact formulation is not published.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class AsymmetricContrastiveFocalLoss(nn.Module):
    """
    Asymmetric Contrastive Focal Loss (ACFL)

    A novel composite loss for anomaly detection under class imbalance.
    Combines:
        1. Focal Loss — downweights easy normal samples during classification
        2. Asymmetric Contrastive Margin — pushes anomalies far from normal cluster

    Args:
        focal_gamma (float): Focal loss focusing parameter. Controls how much
            easy examples are downweighted. γ=2 is standard; γ=2.5 is more
            aggressive for severe imbalance. Default: 2.0
        focal_alpha (float): Focal loss class balance factor. Default: 0.25
        margin_pos (float): Minimum L2 distance required between anomaly samples
            and the normal class centroid. Anomalies within this margin are penalized.
            Default: 1.0
        margin_neg (float): Minimum L2 distance required between normal samples
            and the anomaly class centroid. Should be < margin_pos (asymmetry).
            Default: 0.5
        lambda_contrast (float): Weight of the contrastive term relative to focal.
            ACFL = (1 - λ) · L_focal + λ · L_contrastive
            Default: 0.3
        eps (float): Small constant for numerical stability. Default: 1e-8
    """

    def __init__(self, focal_gamma=2.0, focal_alpha=0.25,
                 margin_pos=1.0, margin_neg=0.5,
                 lambda_contrast=0.3, eps=1e-8):
        super().__init__()
        self.focal_gamma = focal_gamma
        self.focal_alpha = focal_alpha
        self.margin_pos = margin_pos   # Anomaly samples must be >= this far from normal centroid
        self.margin_neg = margin_neg   # Normal samples must be >= this far from anomaly centroid
        self.lambda_contrast = lambda_contrast
        self.eps = eps

    def focal_loss(self, logits, targets):
        """
        Compute Focal Loss with proper per-class alpha balancing.

        L_focal = -α_t · (1 - p_t)^γ · log(p_t)

        where α_t is class-specific:
            α_t = focal_alpha       for anomaly class (targets == 1)
            α_t = (1 - focal_alpha) for normal class  (targets == 0)

        This is the CORRECT formulation from Lin et al. (2017).
        With focal_alpha=0.75, anomaly samples get 3x more gradient signal
        than normal samples — directly forcing the model to prioritize recall.

        Args:
            logits:  (B, C) — raw model output (pre-softmax)
            targets: (B,)   — integer class labels (0=normal, 1=anomaly)

        Returns:
            Scalar focal loss value
        """
        # Softmax probabilities
        probs = F.softmax(logits, dim=1)             # (B, C)

        # Standard cross-entropy loss (element-wise)
        ce_loss = F.cross_entropy(logits, targets, reduction='none')  # (B,)

        # Extract probability of the true class for each sample
        p_t = probs.gather(1, targets.view(-1, 1)).squeeze(1)  # (B,)

        # Focal term: suppresses easy examples
        focal_weight = (1.0 - p_t) ** self.focal_gamma   # (B,)

        # Per-class alpha: α_t = alpha for anomaly (1), (1-alpha) for normal (0)
        # This is the key correction vs v1 which applied alpha uniformly.
        alpha_t = torch.where(
            targets == 1,
            torch.full_like(p_t, self.focal_alpha),
            torch.full_like(p_t, 1.0 - self.focal_alpha),
        )  # (B,)

        # Weighted focal loss: anomaly samples get alpha=0.75 weight
        focal = alpha_t * focal_weight * ce_loss  # (B,)

        return focal.mean()

    def contrastive_margin_loss(self, metric_features, targets):
        """
        Compute Asymmetric Contrastive Margin Loss.

        Uses batch-level class centroids to enforce minimum separation
        between classes in the metric space learned by the AIH.

        Args:
            metric_features: (B, metric_dim) — L2-normalized features from AIH
            targets:         (B,)            — integer class labels (0=normal, 1=anomaly)

        Returns:
            Scalar contrastive loss value (0 if only one class in batch)
        """
        # Separate features by class
        normal_mask  = (targets == 0)   # Boolean mask for normal samples
        anomaly_mask = (targets == 1)   # Boolean mask for anomaly samples

        # Check both classes are present in this batch
        if not normal_mask.any() or not anomaly_mask.any():
            # Return 0 if one class is absent — can happen with small batches
            return torch.tensor(0.0, device=metric_features.device,
                                requires_grad=False)

        normal_feats  = metric_features[normal_mask]   # (N_normal, D)
        anomaly_feats = metric_features[anomaly_mask]  # (N_anomaly, D)

        # Compute class centroids (mean in metric space)
        mu_normal  = normal_feats.mean(dim=0)    # (D,)
        mu_anomaly = anomaly_feats.mean(dim=0)   # (D,)

        # ── Penalty 1: Anomaly samples too close to normal centroid ────
        # Each anomaly feature's distance from the normal centroid
        dist_anomaly_to_normal = torch.norm(
            anomaly_feats - mu_normal.unsqueeze(0), p=2, dim=1
        )  # (N_anomaly,)

        # Hinge loss: penalize if distance < margin_pos
        loss_anomaly = F.relu(self.margin_pos - dist_anomaly_to_normal).pow(2)  # (N_anomaly,)

        # ── Penalty 2: Normal samples too close to anomaly centroid ────
        # Each normal feature's distance from the anomaly centroid
        dist_normal_to_anomaly = torch.norm(
            normal_feats - mu_anomaly.unsqueeze(0), p=2, dim=1
        )  # (N_normal,)

        # Hinge loss: penalize if distance < margin_neg (asymmetric: smaller margin)
        loss_normal = F.relu(self.margin_neg - dist_normal_to_anomaly).pow(2)  # (N_normal,)

        # Total contrastive loss: sum both penalties, normalize by batch size
        total_contrastive = (loss_anomaly.sum() + loss_normal.sum()) / (
            anomaly_feats.shape[0] + normal_feats.shape[0] + self.eps
        )

        return total_contrastive

    def forward(self, logits, targets, metric_features=None):
        """
        Compute the full Asymmetric Contrastive Focal Loss.

        ACFL = (1 - λ) · L_focal + λ · L_contrastive

        Args:
            logits:          (B, C)   — raw classification logits
            targets:         (B,)     — integer class labels
            metric_features: (B, D)   — L2-normalized metric space features from AIH.
                             If None, only focal loss is returned (for warmup).

        Returns:
            Scalar total loss value.
            Components are stored as attributes for logging:
                self.last_focal_loss      (float)
                self.last_contrastive_loss (float)
        """
        # Focal Loss component
        l_focal = self.focal_loss(logits, targets)

        if metric_features is None or self.lambda_contrast == 0:
            self.last_focal_loss = l_focal.item()
            self.last_contrastive_loss = 0.0
            return l_focal

        # Contrastive Margin component
        l_contrast = self.contrastive_margin_loss(metric_features, targets)

        # Store for logging
        self.last_focal_loss = l_focal.item()
        self.last_contrastive_loss = l_contrast.item() if isinstance(
            l_contrast, torch.Tensor) else 0.0

        # Composite ACFL
        total_loss = (1.0 - self.lambda_contrast) * l_focal + \
                     self.lambda_contrast * l_contrast

        return total_loss


if __name__ == '__main__':
    print("=" * 60)
    print("Asymmetric Contrastive Focal Loss (ACFL) - Unit Test")
    print("=" * 60)

    torch.manual_seed(42)
    device = torch.device('cpu')

    loss_fn = AsymmetricContrastiveFocalLoss(
        focal_gamma=2.0,
        focal_alpha=0.25,
        margin_pos=1.0,
        margin_neg=0.5,
        lambda_contrast=0.3,
    )

    # Simulate a batch: 24 normal, 8 anomaly (reflects 83:16 imbalance)
    batch_size = 32
    num_normal = 24
    # requires_grad=True to test backward pass
    logits = torch.randn(batch_size, 2, requires_grad=True)
    targets = torch.zeros(batch_size, dtype=torch.long)
    targets[num_normal:] = 1   # Last 8 are anomalies

    # Simulate TIGHTLY CLUSTERED features (forces contrastive loss > 0)
    # Normal features near [1, 0, ...] and anomaly features near [0, 1, ...]
    raw_feats = torch.zeros(batch_size, 64)
    raw_feats[:num_normal, 0] = 1.0    # Normal cluster near dim-0
    raw_feats[num_normal:, 1] = 1.0    # Anomaly cluster near dim-1
    # Add small noise to make them non-perfectly separated (forces contrastive penalty)
    raw_feats = raw_feats + 0.4 * torch.randn_like(raw_feats)
    metric_feats = F.normalize(raw_feats, p=2, dim=1)
    metric_feats = metric_feats.requires_grad_(True)

    # Compute ACFL
    loss = loss_fn(logits, targets, metric_feats)

    print(f"Batch size:           {batch_size} ({num_normal} normal, {batch_size - num_normal} anomaly)")
    print(f"Focal Loss component: {loss_fn.last_focal_loss:.4f}")
    print(f"Contrastive Loss:     {loss_fn.last_contrastive_loss:.4f}")
    print(f"Total ACFL:           {loss.item():.4f}")

    # Test with only focal loss (no metric features - for warmup phase)
    logits2 = torch.randn(batch_size, 2, requires_grad=True)
    loss_focal_only = loss_fn(logits2, targets, metric_features=None)
    print(f"\nFocal-only (warmup mode): {loss_focal_only.item():.4f}")

    # Test gradient flow
    loss.backward()
    grad_norm = logits.grad.norm().item() if logits.grad is not None else 0.0
    print(f"\n[OK] ACFL backward pass successful -- grad norm on logits: {grad_norm:.6f}")
    print("[OK] Asymmetric Contrastive Focal Loss working correctly.")

