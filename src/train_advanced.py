import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, accuracy_score
from models import get_model

DEFAULT_DATA_DIR = r"C:\SOFTWARE\DL Project\outputs\spectrograms"
MODELS_DIR = r"C:\SOFTWARE\DL Project\models"

class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=None, reduction='mean'):
        super().__init__()
        self.gamma = gamma
        if alpha is not None:
            if isinstance(alpha, (list,tuple)):
                self.alpha = torch.tensor(alpha, dtype=torch.float32)
            else:
                self.alpha = torch.tensor([1.0-alpha, alpha], dtype=torch.float32)
        else:
            self.alpha = None
        self.reduction = reduction

    def forward(self, inputs, targets):
        # inputs: logits [B, C]
        ce = nn.functional.cross_entropy(inputs, targets, reduction='none')
        p = torch.softmax(inputs, dim=1)
        pt = p.gather(1, targets.unsqueeze(1)).squeeze(1)
        loss = (1-pt) ** self.gamma * ce
        if self.alpha is not None:
            if self.alpha.device != inputs.device:
                self.alpha = self.alpha.to(inputs.device)
            at = self.alpha.gather(0, targets)
            loss = at * loss
        if self.reduction == 'mean':
            return loss.mean()
        if self.reduction == 'sum':
            return loss.sum()
        return loss


def tune_threshold(model, loader, device):
    model.eval()
    probs = []
    trues = []
    softmax = nn.Softmax(dim=1)
    with torch.no_grad():
        for xb,yb in loader:
            xb = xb.to(device)
            out = model(xb)
            p = softmax(out)[:,1].cpu().numpy()
            probs.extend(p.tolist())
            trues.extend(yb.numpy().tolist())
    probs = np.array(probs)
    trues = np.array(trues)
    best_f1 = 0.0
    best_t = 0.5
    for t in np.linspace(0.0,1.0,101):
        preds = (probs >= t).astype(int)
        f1 = f1_score(trues, preds, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_t = float(t)
    return best_t, best_f1


def evaluate(model, loader, device, threshold=0.5):
    model.eval()
    preds = []
    trues = []
    probs = []
    softmax = nn.Softmax(dim=1)
    with torch.no_grad():
        for xb,yb in loader:
            xb = xb.to(device)
            out = model(xb)
            p = softmax(out)[:,1].cpu().numpy()
            preds.extend((p >= threshold).astype(int).tolist())
            probs.extend(p.tolist())
            trues.extend(yb.numpy().tolist())
    acc = accuracy_score(trues, preds)
    f1 = f1_score(trues, preds, zero_division=0)
    return acc, f1, probs, trues


def train_model(model_name, data_dir=DEFAULT_DATA_DIR, epochs=10, batch_size=32, lr=1e-4, device=None, gamma=2.0):
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    print('Device', device)
    train_transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    ])
    eval_transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    ])

    train_ds = datasets.ImageFolder(os.path.join(data_dir,'train'), transform=train_transform)
    val_ds = datasets.ImageFolder(os.path.join(data_dir,'val'), transform=eval_transform)
    test_ds = datasets.ImageFolder(os.path.join(data_dir,'test'), transform=eval_transform)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    # compute class weights from train set
    labels = [y for _,y in train_ds]
    counts = np.bincount(labels, minlength=2).astype(float)
    total = counts.sum()
    class_weights = total / (counts + 1e-8)
    # normalize
    class_weights = class_weights / class_weights.sum()
    alpha = [float(class_weights[0]), float(class_weights[1])]
    print('Class counts', counts.tolist(), 'alpha', alpha)

    model = get_model(model_name, num_classes=2, pretrained=True)
    model = model.to(device)

    criterion = FocalLoss(gamma=gamma, alpha=alpha)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    best_val_f1 = 0.0
    best_threshold = 0.5
    history = {'train_loss':[], 'val_loss':[], 'val_f1':[], 'threshold':[]}

    for epoch in range(epochs):
        model.train()
        losses = []
        for xb,yb in train_loader:
            xb,yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            out = model(xb)
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        train_loss = float(np.mean(losses)) if losses else 0.0

        # validate
        model.eval()
        val_losses = []
        with torch.no_grad():
            for xb,yb in val_loader:
                xb,yb = xb.to(device), yb.to(device)
                out = model(xb)
                loss = criterion(out, yb)
                val_losses.append(loss.item())
        val_loss = float(np.mean(val_losses)) if val_losses else 0.0

        # tune threshold on val set
        thresh, val_f1 = tune_threshold(model, val_loader, device)
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_f1'].append(val_f1)
        history['threshold'].append(thresh)
        print(f'Epoch {epoch+1}/{epochs} train_loss={train_loss:.4f} val_loss={val_loss:.4f} val_f1={val_f1:.4f} thresh={thresh:.2f}')

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_threshold = thresh
            os.makedirs(MODELS_DIR, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(MODELS_DIR, f'{model_name}_adv_best.pth'))

    # evaluate on test with best threshold
    # reload best model
    model.load_state_dict(torch.load(os.path.join(MODELS_DIR, f'{model_name}_adv_best.pth'), map_location=device))
    acc, f1, probs, trues = evaluate(model, test_loader, device, threshold=best_threshold)
    print('Final test acc', acc, 'test f1', f1, 'threshold', best_threshold)

    metrics = {'model':model_name, 'best_val_f1':best_val_f1, 'best_threshold':best_threshold, 'test_acc':float(acc), 'test_f1':float(f1), 'history':history}
    with open(os.path.join(MODELS_DIR, f'{model_name}_adv_metrics.json'),'w') as f:
        json.dump(metrics, f, indent=2)
    return metrics

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--models', nargs='+', default=['resnet18','densenet121'])
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch', type=int, default=32)
    parser.add_argument('--gamma', type=float, default=2.0)
    args = parser.parse_args()
    results = {}
    for m in args.models:
        print('Training advanced', m)
        try:
            results[m] = train_model(m, epochs=args.epochs, batch_size=args.batch, gamma=args.gamma)
        except Exception as e:
            print('Error', m, e)
    with open(os.path.join(MODELS_DIR,'all_results_adv.json'),'w') as f:
        json.dump(results, f, indent=2)
    print('Done')
