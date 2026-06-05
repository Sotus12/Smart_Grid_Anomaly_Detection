"""
Training Script for GAResNet (Grid-Attention Residual Network)
Optimized for high-precision and high-recall anomaly detection in Smart Grids.
"""

import os
import json
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import transforms, datasets
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_curve, precision_recall_curve
from tqdm import tqdm
from collections import defaultdict

from models import get_model
from focal_loss import FocalLoss

class EarlyStopping:
    def __init__(self, patience=10, delta=0.0001):
        self.patience = patience
        self.delta = delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, val_f1):
        if self.best_score is None:
            self.best_score = val_f1
        elif val_f1 < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = val_f1
            self.counter = 0

def train_epoch(model, loader, criterion, optimizer, scheduler, device):
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []

    for images, labels in tqdm(loader, desc="Training", leave=False):
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    return total_loss / len(loader), acc

def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    all_probs = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            probs = torch.softmax(outputs, dim=1)[:, 1]
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    
    # Calculate best F1 threshold
    precision, recall, thresholds = precision_recall_curve(all_labels, all_probs)
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
    best_idx = np.argmax(f1_scores)
    best_threshold = thresholds[best_idx] if best_idx < len(thresholds) else 0.5
    best_f1 = f1_scores[best_idx]
    
    return total_loss / len(loader), best_f1, best_threshold, all_probs, all_labels

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=40)
    parser.add_argument('--lr', type=float, default=2e-3)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--data_dir', type=str, default=r'C:\SOFTWARE\DL Project\outputs\spectrograms')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_save_path = r'C:\SOFTWARE\DL Project\models\garesnet_best.pth'
    
    # Data Augmentation
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(0.2, 0.2, 0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    train_ds = datasets.ImageFolder(os.path.join(args.data_dir, 'train'), train_transform)
    val_ds = datasets.ImageFolder(os.path.join(args.data_dir, 'val'), val_transform)
    test_ds = datasets.ImageFolder(os.path.join(args.data_dir, 'test'), val_transform)

    # Weighted Sampler for Imbalance
    labels = [l for _, l in train_ds]
    counts = np.bincount(labels)
    weights = 1. / counts
    sample_weights = weights[labels]
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights))

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size)

    # Model, Loss, Optimizer
    model = get_model('garesnet', num_classes=2, pretrained=False).to(device)
    criterion = FocalLoss(alpha=0.25, gamma=2.5) # Higher gamma for harder focus
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    
    scheduler = OneCycleLR(optimizer, max_lr=args.lr, 
                           steps_per_epoch=len(train_loader), 
                           epochs=args.epochs)

    early_stopping = EarlyStopping(patience=12)
    best_f1 = 0
    history = defaultdict(list)

    print(f"Starting Training GAResNet on {device}...")
    
    for epoch in range(args.epochs):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, scheduler, device)
        val_loss, val_f1, threshold, _, _ = validate(model, val_loader, criterion, device)
        
        history['train_loss'].append(train_loss)
        history['val_f1'].append(val_f1)
        
        print(f"Epoch {epoch+1}/{args.epochs} | Loss: {train_loss:.4f} | Val F1: {val_f1:.4f} | Thresh: {threshold:.3f}")
        
        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save(model.state_dict(), model_save_path)
            print(f"* New Best Model Saved (F1: {best_f1:.4f})")
        
        early_stopping(val_f1)
        if early_stopping.early_stop:
            print("Early stopping triggered.")
            break

    # Final Evaluation
    print("\n--- Final Evaluation ---")
    model.load_state_dict(torch.load(model_save_path))
    _, val_f1, opt_threshold, _, _ = validate(model, val_loader, criterion, device)
    
    model.eval()
    test_probs = []
    test_labels = []
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)[:, 1]
            test_probs.extend(probs.cpu().numpy())
            test_labels.extend(labels.cpu().numpy())
    
    test_preds = (np.array(test_probs) >= opt_threshold).astype(int)
    
    metrics = {
        'test_accuracy': accuracy_score(test_labels, test_preds),
        'test_precision': precision_score(test_labels, test_preds, zero_division=0),
        'test_recall': recall_score(test_labels, test_preds, zero_division=0),
        'test_f1': f1_score(test_labels, test_preds, zero_division=0),
        'optimal_threshold': float(opt_threshold)
    }
    
    print(json.dumps(metrics, indent=2))
    
    with open(r'C:\SOFTWARE\DL Project\models\garesnet_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print("\n[SUCCESS] Training and Evaluation Complete.")

if __name__ == '__main__':
    main()
