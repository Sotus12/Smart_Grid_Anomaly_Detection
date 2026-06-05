"""
Training script for Hybrid CNN-Attention Model

This script trains the novel custom architecture with the same optimization
techniques as the improved transfer learning models, enabling fair comparison.

Novel features:
- Custom lightweight CNN backbone (1.58M parameters)
- Spatial + Channel Attention mechanisms
- Interpretable attention weights extraction
- Same optimization pipeline for fair comparison
"""

import os
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
from tqdm import tqdm

from models import get_model
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


class EarlyStopping:
    """Early stopping to prevent overfitting."""
    def __init__(self, patience=7, verbose=False, delta=0.001):
        self.patience = patience
        self.verbose = verbose
        self.delta = delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, val_metric):
        if self.best_score is None:
            self.best_score = val_metric
        elif val_metric > self.best_score + self.delta:
            self.best_score = val_metric
            self.counter = 0
        else:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter}/{self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True


def compute_metrics(y_true, y_pred):
    """Compute classification metrics."""
    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0),
    }


def mixup_batch(x, y, alpha=0.2):
    """Apply mixup data augmentation."""
    lam = np.random.beta(alpha, alpha)
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def train_epoch(model, train_loader, criterion, optimizer, device, mixup_prob=0.5):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(train_loader, desc='Training', leave=False)
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)

        # Apply mixup with probability
        if np.random.random() < mixup_prob:
            images, y_a, y_b, lam = mixup_batch(images, labels, alpha=0.2)
            optimizer.zero_grad()
            outputs = model(images)
            loss = lam * criterion(outputs, y_a) + (1 - lam) * criterion(outputs, y_b)
        else:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

        pbar.set_postfix({'loss': loss.item()})

    return total_loss / len(train_loader), correct / total


def validate(model, val_loader, criterion, device):
    """Validate the model."""
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    val_loss = total_loss / len(val_loader)
    val_acc = np.mean(np.array(all_preds) == np.array(all_labels))
    
    return val_loss, val_acc, all_preds, all_labels


def main():
    parser = argparse.ArgumentParser(description='Train Hybrid CNN-Attention Model')
    parser.add_argument('--train-dir', type=str, default=r'C:\SOFTWARE\DL Project\outputs\spectrograms\train',
                        help='Path to training data')
    parser.add_argument('--val-dir', type=str, default=r'C:\SOFTWARE\DL Project\outputs\spectrograms\val',
                        help='Path to validation data')
    parser.add_argument('--test-dir', type=str, default=r'C:\SOFTWARE\DL Project\outputs\spectrograms\test',
                        help='Path to test data')
    parser.add_argument('--epochs', type=int, default=20,
                        help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=3e-4,
                        help='Initial learning rate')
    parser.add_argument('--model-dir', type=str, default=r'C:\SOFTWARE\DL Project\models',
                        help='Directory to save models')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    # Create model directory
    os.makedirs(args.model_dir, exist_ok=True)

    # ===== LOAD DATA =====
    print('\n=== Loading Data ===')
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.3),
        transforms.RandomRotation(20),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.2, scale=(0.02, 0.1))
    ])

    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])

    train_ds = datasets.ImageFolder(args.train_dir, transform=train_transform)
    val_ds = datasets.ImageFolder(args.val_dir, transform=val_transform)
    test_ds = datasets.ImageFolder(args.test_dir, transform=val_transform)

    print(f'Train: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)}')

    # ===== WEIGHTED SAMPLER FOR CLASS IMBALANCE =====
    labels = [label for _, label in train_ds]
    class_counts = {0: labels.count(0), 1: labels.count(1)}
    num_samples = len(labels)
    class_weights = {cls: num_samples / count for cls, count in class_counts.items()}
    sample_weights = [class_weights[l] for l in labels]
    sampler = WeightedRandomSampler(sample_weights, num_samples, replacement=True)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    # ===== MODEL, LOSS, OPTIMIZER =====
    print('\n=== Building Model ===')
    model = get_model('hybrid_attention', num_classes=2, pretrained=False).to(device)
    
    # Count parameters
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Model: Hybrid CNN-Attention')
    print(f'Parameters: {num_params:,} (Novel custom architecture)')

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = OneCycleLR(
        optimizer,
        max_lr=args.lr,
        epochs=args.epochs,
        steps_per_epoch=len(train_loader),
        pct_start=0.3,
        anneal_strategy='cos'
    )

    # ===== TRAINING LOOP =====
    print('\n=== Training ===')
    early_stopping = EarlyStopping(patience=7, verbose=True, delta=0.001)
    best_val_acc = 0
    history = defaultdict(list)

    for epoch in range(args.epochs):
        print(f'\nEpoch [{epoch+1}/{args.epochs}]')

        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        scheduler.step()

        val_loss, val_acc, val_preds, val_labels = validate(model, val_loader, criterion, device)

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        print(f'Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}')
        print(f'Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}')

        # Early stopping
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 
                      os.path.join(args.model_dir, 'hybrid_attention_best.pth'))
            print(f'✓ Best model saved (Val Acc: {val_acc:.4f})')
        else:
            early_stopping(val_acc)
            if early_stopping.early_stop:
                print('\nEarly stopping triggered')
                break

    # ===== EVALUATION =====
    print('\n=== Evaluation ===')
    model.load_state_dict(torch.load(os.path.join(args.model_dir, 'hybrid_attention_best.pth')))
    model.eval()

    # Test set evaluation
    test_loss = 0.0
    test_preds = []
    test_labels = []
    
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            test_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            test_preds.extend(predicted.cpu().numpy())
            test_labels.extend(labels.cpu().numpy())

    test_loss /= len(test_loader)
    test_metrics = compute_metrics(test_labels, test_preds)

    print(f'Test Loss: {test_loss:.4f}')
    print(f'Test Accuracy: {test_metrics["accuracy"]:.4f}')
    print(f'Test F1 Score: {test_metrics["f1"]:.4f}')
    print(f'Test Precision: {test_metrics["precision"]:.4f}')
    print(f'Test Recall: {test_metrics["recall"]:.4f}')

    # ===== SAVE RESULTS =====
    metrics = {
        'model': 'hybrid_attention',
        'architecture': 'Custom CNN with Spatial + Channel Attention',
        'parameters': num_params,
        'epochs_trained': epoch + 1,
        'train_history': {k: v for k, v in history.items()},
        'test_loss': float(test_loss),
        'test_accuracy': test_metrics['accuracy'],
        'test_precision': test_metrics['precision'],
        'test_recall': test_metrics['recall'],
        'test_f1': test_metrics['f1'],
        'novelty': 'Custom architecture with interpretable attention mechanisms',
    }

    metrics_file = os.path.join(args.model_dir, 'hybrid_attention_metrics.json')
    with open(metrics_file, 'w') as f:
        json.dump(metrics, f, indent=2)

    print(f'\n✓ Metrics saved to {metrics_file}')
    print('\n=== Training Complete ===')


if __name__ == '__main__':
    main()
