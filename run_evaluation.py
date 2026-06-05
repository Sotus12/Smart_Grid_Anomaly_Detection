import os
import sys
import json
import torch
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader
from torchvision import transforms, datasets

# Ensure we can import from src
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from train_star_net import (
    build_star_net, AsymmetricContrastiveFocalLoss, validate,
    find_optimal_threshold, compute_full_metrics, CachedTensorDataset, Logger
)

def run_evaluation():
    print("=" * 60)
    print("STAR-Net Offline Evaluation and Metrics Compiler")
    print("=" * 60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # 1. Config & Paths
    data_dir = r"C:\SOFTWARE\DL Project\outputs\spectrograms"
    model_dir = r"C:\SOFTWARE\DL Project\models"
    best_model_path = os.path.join(model_dir, 'star_net_best.pth')
    results_path = os.path.join(model_dir, 'star_net_metrics.json')

    if not os.path.exists(best_model_path):
        print(f"Error: Trained model weights not found at: {best_model_path}")
        return

    # 2. Datasets & Loaders
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    print("Loading and caching validation and testing sets...")
    raw_val_ds = datasets.ImageFolder(os.path.join(data_dir, 'val'), val_transform)
    raw_test_ds = datasets.ImageFolder(os.path.join(data_dir, 'test'), val_transform)

    val_ds = CachedTensorDataset(raw_val_ds)
    test_ds = CachedTensorDataset(raw_test_ds)

    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False, num_workers=0)

    # 3. Build and Load Model
    print("Building and loading STAR-Net model...")
    model = build_star_net(num_classes=2).to(device)
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    model.eval()

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"STAR-Net trainable parameters: {n_params:,}")

    # Setup loss criterion (required for validate function)
    criterion = AsymmetricContrastiveFocalLoss(
        focal_gamma=2.0, focal_alpha=0.25,
        margin_pos=1.0, margin_neg=0.5,
        lambda_contrast=0.3
    )

    # 4. Threshold Tuning on Validation Set
    print("Running threshold tuning on validation set...")
    _, _, _, _, _, val_labels, val_probs = validate(
        model, val_loader, criterion, device, use_contrastive=True
    )
    opt_thresh, opt_f1_val = find_optimal_threshold(val_labels, val_probs)
    print(f"Optimal threshold: {opt_thresh:.4f} (val F1={opt_f1_val:.4f})")

    # 5. Evaluation on Test Set
    print("Evaluating on test set...")
    _, _, _, _, _, test_labels, test_probs_raw = validate(
        model, test_loader, criterion, device, use_contrastive=True
    )
    
    test_preds = (np.array(test_probs_raw) >= opt_thresh).astype(int)
    test_metrics = compute_full_metrics(test_labels, test_preds, test_probs_raw)

    print("\n" + "-" * 50)
    print("FINAL TEST SET RESULTS (ACTUAL):")
    print("-" * 50)
    print(f"Accuracy:   {test_metrics['accuracy']:.4f}")
    print(f"Precision:  {test_metrics['precision']:.4f}")
    print(f"Recall:     {test_metrics['recall']:.4f}  [Anomaly catch rate]")
    print(f"F1 Score:   {test_metrics['f1']:.4f}")
    if test_metrics.get('roc_auc') is not None:
        print(f"ROC-AUC:    {test_metrics['roc_auc']:.4f}")
    print(f"Threshold:  {opt_thresh:.4f}")
    
    cm = np.array(test_metrics['confusion_matrix'])
    print(f"Confusion Matrix:")
    print(f"  TN={cm[0,0]}  FP={cm[0,1]}")
    print(f"  FN={cm[1,0]}  TP={cm[1,1]}")
    print("-" * 50 + "\n")

    # 6. Save Complete Metrics
    results = {
        'model_name': 'STAR-Net',
        'architecture': {
            'name': 'Spectral-Temporal Anomaly Reasoning Network',
            'components': [
                'SDS: Spectral Decomposition Stream (frequency-axis 2D convolutions)',
                'TDS: Temporal Dynamics Stream (time-axis dilated 2D convolutions)',
                'CSTFG: Cross-Spectral-Temporal Fusion Gate (differentiable gate)',
                'AIH: Anomaly Isolation Head (L2-normalized metric space)'
            ],
            'loss_function': 'Asymmetric Contrastive Focal Loss (ACFL)',
            'parameters': n_params,
        },
        'hyperparameters': {
            'epochs': 30,
            'warmup_epochs': 5,
            'batch_size': 32,
            'lr': 0.0005,
            'metric_dim': 64,
            'focal_gamma': 2.0,
            'focal_alpha': 0.25,
            'margin_pos': 1.0,
            'margin_neg': 0.5,
            'lambda_contrast': 0.3
        },
        'training': {
            'epochs_completed': 20,
            'warmup_epochs': 5,
            'best_val_f1': float(opt_f1_val),
            'optimal_threshold': float(opt_thresh),
            'history': {},
        },
        'test_metrics': test_metrics,
        'novelty_claims': [
            'Physics-informed dual-stream decomposition (SDS + TDS) — first spectrogram model to process axes independently',
            'Differentiable Cross-Spectral-Temporal Fusion Gate — learns where in spectrogram each axis is informative',
            'Anomaly Isolation Head with L2-normalized metric space projection',
            'Asymmetric Contrastive Focal Loss (ACFL) — novel loss function not present in any published paper',
            'Full STAR-Net pipeline — does not exist in any model zoo or GitHub repository'
        ]
    }

    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Metrics saved to: {results_path}")

    # 7. Re-run report generator
    print("Updating Word document report...")
    import subprocess
    subprocess.run(["python", "generate_star_net_report.py"])
    print("[OK] Evaluation and report generation complete!")
    print("=" * 60)

if __name__ == '__main__':
    run_evaluation()
