"""
Breakthrough Training Script for Attentional ResNet-50
Goal: Exceed 83% accuracy baseline while maintaining anomaly detection performance.
"""

import os
import json
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import transforms, datasets
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import numpy as np
from tqdm import tqdm

from models import get_model

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    
    for images, labels in tqdm(loader, desc="Training", leave=False):
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        
    return running_loss / len(loader), accuracy_score(all_labels, all_preds)

def validate(model, loader, criterion, device):
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    recall = recall_score(all_labels, all_preds, zero_division=0)
    return acc, f1, recall, all_preds, all_labels

def find_best_accuracy_threshold(y_true, y_probs):
    from sklearn.metrics import roc_curve
    fpr, tpr, thresholds = roc_curve(y_true, y_probs)
    best_acc = 0
    best_thresh = 0.5
    for thresh in thresholds:
        y_pred = (y_probs >= thresh).astype(int)
        acc = accuracy_score(y_true, y_pred)
        if acc > best_acc:
            best_acc = acc
            best_thresh = thresh
    return best_thresh, best_acc

def validate_probs(model, loader, device):
    model.eval()
    all_probs = []
    all_labels = []
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)[:, 1]
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    return np.array(all_probs), np.array(all_labels)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--data_dir', type=str, default=r'C:\SOFTWARE\DL Project\outputs\spectrograms')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_path = r'C:\SOFTWARE\DL Project\models\breakthrough_best.pth'

    # Stronger Augmentation for Accuracy
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
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

    # Balanced Sampler (slightly biased towards anomalies but less than before)
    labels = [l for _, l in train_ds]
    class_sample_count = np.array([len(np.where(labels == t)[0]) for t in np.unique(labels)])
    weight = 1. / class_sample_count
    # Moderate balancing: Increase weight of anomaly but keep it reasonable for accuracy
    weight[1] = weight[1] * 1.5 
    samples_weight = np.array([weight[t] for t in labels])
    sampler = WeightedRandomSampler(samples_weight, len(samples_weight))

    train_loader = DataLoader(train_ds, batch_size=16, sampler=sampler)
    val_loader = DataLoader(val_ds, batch_size=16)
    test_loader = DataLoader(test_ds, batch_size=16)

    # Initialize Model
    print(f"Building Breakthrough Model on {device}...")
    model = get_model('att_resnet50', num_classes=2, pretrained=True).to(device)
    
    # Loss & Optimizer
    # Using Class Weights in CrossEntropy for Accuracy focus
    weights = torch.tensor([1.0, 2.0]).to(device) # Anomaly is 2x more important
    criterion = nn.CrossEntropyLoss(weight=weights)
    
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-3)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=2)

    best_acc = 0
    print("Starting Breakthrough Training (Phase: Accuracy Focus)...")
    
    for epoch in range(args.epochs):
        loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_acc, val_f1, val_recall, _, _ = validate(model, val_loader, criterion, device)
        scheduler.step()
        
        print(f"Epoch {epoch+1}/{args.epochs} | Loss: {loss:.4f} | Val Acc: {val_acc:.4f} | Val F1: {val_f1:.4f}")
        
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), model_path)
            print(f"[*] New High Accuracy Saved: {best_acc:.4f}")

    # Final Evaluation with Threshold Tuning
    print("\n--- Final Test Evaluation (Optimized for Accuracy) ---")
    model.load_state_dict(torch.load(model_path))
    
    # Use validation set to find best threshold for accuracy
    val_probs, val_labels = validate_probs(model, val_loader, device)
    best_thresh, _ = find_best_accuracy_threshold(val_labels, val_probs)
    print(f"Optimal Threshold for Accuracy: {best_thresh:.4f}")

    # Apply to test set
    test_probs, test_labels = validate_probs(model, test_loader, device)
    test_preds = (test_probs >= best_thresh).astype(int)
            
    metrics = {
        'test_accuracy': accuracy_score(test_labels, test_preds),
        'test_precision': precision_score(test_labels, test_preds, zero_division=0),
        'test_recall': recall_score(test_labels, test_preds, zero_division=0),
        'test_f1': f1_score(test_labels, test_preds, zero_division=0),
        'optimal_threshold': float(best_thresh)
    }
    
    print(json.dumps(metrics, indent=2))
    
    with open(r'C:\SOFTWARE\DL Project\models\breakthrough_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print("\n[SUCCESS] Breakthrough Training Complete.")

if __name__ == '__main__':
    main()
