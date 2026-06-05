"""
Improved training script with multiple optimization techniques:
1. Learning rate scheduling (OneCycleLR)
2. Early stopping with patience
3. Better hyperparameter tuning
4. Label smoothing
5. Mixup data augmentation
6. Model checkpoint saving based on validation accuracy
7. Gradient clipping to prevent vanishing/exploding gradients
"""

import os
import json
import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import OneCycleLR
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, WeightedRandomSampler
from sklearn.metrics import accuracy_score, f1_score, classification_report
from models import get_model

DEFAULT_DATA_DIR = r"C:\SOFTWARE\DL Project\outputs\spectrograms"
MODELS_DIR = r"C:\SOFTWARE\DL Project\models"


def mixup_batch(x, y, alpha=0.2):
    """Mixup data augmentation in batch."""
    lam = np.random.beta(alpha, alpha)
    batch_size = x.size(0)
    index = torch.randperm(batch_size)
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """Compute mixup loss."""
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


class EarlyStopping:
    """Early stopping to prevent overfitting."""
    def __init__(self, patience=5, verbose=False, delta=0.0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.delta = delta

    def __call__(self, val_score):
        if self.best_score is None:
            self.best_score = val_score
        elif val_score > self.best_score + self.delta:
            self.best_score = val_score
            self.counter = 0
        else:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter}/{self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
                if self.verbose:
                    print('EarlyStopping triggered')


def train_model_improved(model_name, data_dir=DEFAULT_DATA_DIR, epochs=20, batch_size=32, 
                         lr=3e-4, device=None, use_mixup=True, use_label_smoothing=True,
                         patience=7, augment_strength='high'):
    """
    Improved training with multiple optimization techniques.
    
    Args:
        model_name: 'resnet18', 'densenet121', 'efficientnet_b0'
        data_dir: Path to spectrogram directory
        epochs: Maximum epochs (may stop early)
        batch_size: Batch size
        lr: Initial learning rate
        device: 'cuda' or 'cpu'
        use_mixup: Use mixup data augmentation
        use_label_smoothing: Use label smoothing
        patience: Early stopping patience
        augment_strength: 'low', 'medium', 'high'
    """
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'\n{"="*60}')
    print(f'Training {model_name} on {device}')
    print(f'Config: epochs={epochs}, batch={batch_size}, lr={lr}')
    print(f'Mixup: {use_mixup}, Label Smoothing: {use_label_smoothing}')
    print(f'Augmentation: {augment_strength}, Early Stopping Patience: {patience}')
    print(f'{"="*60}\n')

    # Data augmentation strategies
    if augment_strength == 'high':
        train_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.3),
            transforms.RandomRotation(20),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            transforms.RandomErasing(p=0.2, scale=(0.02, 0.1))
        ])
    elif augment_strength == 'medium':
        train_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:  # low
        train_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    eval_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Load datasets
    train_ds = datasets.ImageFolder(os.path.join(data_dir, 'train'), transform=train_transform)
    val_ds = datasets.ImageFolder(os.path.join(data_dir, 'val'), transform=eval_transform)
    test_ds = datasets.ImageFolder(os.path.join(data_dir, 'test'), transform=eval_transform)

    # Compute class weights and create WeightedRandomSampler
    labels = [label for _, label in train_ds]
    class_counts = {}
    for l in labels:
        class_counts[l] = class_counts.get(l, 0) + 1
    num_samples = len(labels)
    class_weights = {cls: num_samples / count for cls, count in class_counts.items()}
    sample_weights = [class_weights[l] for l in labels]
    sampler = WeightedRandomSampler(sample_weights, num_samples=num_samples, replacement=True)

    print(f'Class distribution: {class_counts}')
    print(f'Class weights: {class_weights}\n')

    train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    # Model and optimizer
    model = get_model(model_name, num_classes=2, pretrained=True)
    model = model.to(device)
    
    # Loss with label smoothing
    if use_label_smoothing:
        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    else:
        criterion = nn.CrossEntropyLoss()
    
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    
    # Learning rate scheduler
    steps_per_epoch = len(train_loader)
    scheduler = OneCycleLR(
        optimizer,
        max_lr=lr,
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        pct_start=0.3,
        anneal_strategy='cos'
    )

    early_stopping = EarlyStopping(patience=patience, verbose=True, delta=0.001)
    
    best_val_acc = 0.0
    history = {
        'train_loss': [], 'val_loss': [], 'val_acc': [], 'val_f1': [],
        'train_acc': [], 'learning_rate': []
    }

    for epoch in range(epochs):
        # Training
        model.train()
        train_losses = []
        train_preds = []
        train_trues = []
        
        for batch_idx, (xb, yb) in enumerate(train_loader):
            xb, yb = xb.to(device), yb.to(device)
            
            # Mixup
            if use_mixup and np.random.random() < 0.5:
                xb_mixed, yb_a, yb_b, lam = mixup_batch(xb, yb, alpha=0.2)
                optimizer.zero_grad()
                out = model(xb_mixed)
                loss = mixup_criterion(criterion, out, yb_a, yb_b, lam)
            else:
                optimizer.zero_grad()
                out = model(xb)
                loss = criterion(out, yb)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            
            train_losses.append(loss.item())
            
            # Track predictions
            preds = out.argmax(dim=1).detach().cpu().numpy()
            train_preds.extend(preds.tolist())
            train_trues.extend(yb.detach().cpu().numpy().tolist())
        
        train_loss = np.mean(train_losses)
        train_acc = accuracy_score(train_trues, train_preds)

        # Validation
        model.eval()
        val_losses = []
        val_preds = []
        val_trues = []
        
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                out = model(xb)
                loss = criterion(out, yb)
                val_losses.append(loss.item())
                
                preds = out.argmax(dim=1).cpu().numpy()
                val_preds.extend(preds.tolist())
                val_trues.extend(yb.cpu().numpy().tolist())
        
        val_loss = np.mean(val_losses)
        val_acc = accuracy_score(val_trues, val_preds)
        val_f1 = f1_score(val_trues, val_preds, average='binary', zero_division=0)
        
        history['train_loss'].append(float(train_loss))
        history['val_loss'].append(float(val_loss))
        history['train_acc'].append(float(train_acc))
        history['val_acc'].append(float(val_acc))
        history['val_f1'].append(float(val_f1))
        history['learning_rate'].append(float(optimizer.param_groups[0]['lr']))
        
        print(f'Epoch {epoch+1}/{epochs} | '
              f'Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f} | '
              f'Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, F1: {val_f1:.4f} | '
              f'LR: {optimizer.param_groups[0]["lr"]:.2e}')
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            os.makedirs(MODELS_DIR, exist_ok=True)
            model_path = os.path.join(MODELS_DIR, f'{model_name}_improved_best.pth')
            torch.save(model.state_dict(), model_path)
            print(f'  ✓ Best model saved (val_acc: {val_acc:.4f})')
        
        # Early stopping
        early_stopping(val_acc)
        if early_stopping.early_stop:
            print(f'\nEarly stopping at epoch {epoch+1}')
            break

    # Test evaluation
    print('\n' + '='*60)
    print('FINAL TEST EVALUATION')
    print('='*60)
    
    # Reload best model
    best_model_path = os.path.join(MODELS_DIR, f'{model_name}_improved_best.pth')
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    model.eval()
    
    test_preds = []
    test_trues = []
    with torch.no_grad():
        for xb, yb in test_loader:
            xb = xb.to(device)
            out = model(xb)
            preds = out.argmax(dim=1).cpu().numpy()
            test_preds.extend(preds.tolist())
            test_trues.extend(yb.numpy().tolist())
    
    test_acc = accuracy_score(test_trues, test_preds)
    test_f1 = f1_score(test_trues, test_preds, average='binary', zero_division=0)
    
    print(f'\nTest Accuracy: {test_acc:.4f}')
    print(f'Test F1-Score: {test_f1:.4f}')
    print(f'Best Validation Accuracy: {best_val_acc:.4f}')
    print('\nClassification Report:')
    print(classification_report(test_trues, test_preds, target_names=['Normal', 'Anomaly']))
    
    # Save metrics
    metrics = {
        'model': model_name,
        'best_val_acc': float(best_val_acc),
        'test_acc': float(test_acc),
        'test_f1': float(test_f1),
        'epochs_trained': epoch + 1,
        'history': history
    }
    
    metrics_path = os.path.join(MODELS_DIR, f'{model_name}_improved_metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print(f'\nMetrics saved to: {metrics_path}')
    return metrics


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--models', nargs='+', default=['resnet18', 'densenet121', 'efficientnet_b0'])
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch', type=int, default=32)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--patience', type=int, default=7)
    parser.add_argument('--augment', choices=['low', 'medium', 'high'], default='high')
    args = parser.parse_args()
    
    results = {}
    for model_name in args.models:
        print(f'\n{"#"*60}')
        print(f'# Training {model_name.upper()}')
        print(f'{"#"*60}')
        try:
            metrics = train_model_improved(
                model_name,
                epochs=args.epochs,
                batch_size=args.batch,
                lr=args.lr,
                patience=args.patience,
                augment_strength=args.augment
            )
            results[model_name] = metrics
        except Exception as e:
            print(f'Error training {model_name}: {e}')
            import traceback
            traceback.print_exc()
    
    # Save all results
    results_path = os.path.join(MODELS_DIR, 'all_results_improved.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f'\n\n{"="*60}')
    print('All training completed!')
    print(f'Results saved to: {results_path}')
    print(f'{"="*60}')
