import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, WeightedRandomSampler
from sklearn.metrics import accuracy_score, f1_score
from models import get_model

DEFAULT_DATA_DIR = r"C:\SOFTWARE\DL Project\outputs\spectrograms"


def train_model_balanced(model_name, data_dir=DEFAULT_DATA_DIR, epochs=5, batch_size=32, lr=1e-4, device=None, augment=True):
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device', device)

    train_transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    ])
    eval_transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    ])

    train_ds = datasets.ImageFolder(os.path.join(data_dir,'train'), transform=train_transform if augment else eval_transform)
    val_ds = datasets.ImageFolder(os.path.join(data_dir,'val'), transform=eval_transform)
    test_ds = datasets.ImageFolder(os.path.join(data_dir,'test'), transform=eval_transform)

    # compute weights for WeightedRandomSampler to balance classes
    labels = [label for _,label in train_ds]
    class_sample_count = {}
    for l in labels:
        class_sample_count[l] = class_sample_count.get(l,0) + 1
    num_samples = len(labels)
    class_weights = {cls: num_samples/count for cls,count in class_sample_count.items()}
    sample_weights = [class_weights[l] for l in labels]
    sampler = WeightedRandomSampler(sample_weights, num_samples=num_samples, replacement=True)

    train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    model = get_model(model_name, num_classes=2, pretrained=True)
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    best_val_f1 = 0.0
    history = {'train_loss':[], 'val_loss':[], 'val_f1':[]}

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
        train_loss = sum(losses)/len(losses) if losses else 0.0

        # validation
        model.eval()
        preds, trues = [], []
        val_losses = []
        with torch.no_grad():
            for xb,yb in val_loader:
                xb,yb = xb.to(device), yb.to(device)
                out = model(xb)
                loss = criterion(out, yb)
                val_losses.append(loss.item())
                p = out.argmax(dim=1).cpu().numpy()
                preds.extend(p.tolist())
                trues.extend(yb.cpu().numpy().tolist())
        val_loss = sum(val_losses)/len(val_losses) if val_losses else 0.0
        val_f1 = f1_score(trues, preds, average='binary', zero_division=0)
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_f1'].append(val_f1)
        print(f'Epoch {epoch+1}/{epochs} train_loss={train_loss:.4f} val_loss={val_loss:.4f} val_f1={val_f1:.4f}')

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            os.makedirs(r'C:\SOFTWARE\DL Project\models', exist_ok=True)
            torch.save(model.state_dict(), os.path.join(r'C:\SOFTWARE\DL Project\models', f'{model_name}_best.pth'))

    # test
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for xb,yb in test_loader:
            xb,yb = xb.to(device), yb.to(device)
            out = model(xb)
            p = out.argmax(dim=1).cpu().numpy()
            preds.extend(p.tolist())
            trues.extend(yb.cpu().numpy().tolist())
    test_acc = accuracy_score(trues, preds)
    test_f1 = f1_score(trues, preds, average='binary', zero_division=0)
    print('Test acc', test_acc, 'Test f1', test_f1)

    metrics = {'model':model_name, 'best_val_f1':best_val_f1, 'test_acc':float(test_acc), 'test_f1':float(test_f1), 'history':history}
    with open(os.path.join(r'C:\SOFTWARE\DL Project\models', f'{model_name}_balanced_metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)
    return metrics


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--models', nargs='+', default=['resnet18','densenet121','efficientnet_b0'])
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--batch', type=int, default=32)
    args = parser.parse_args()
    results = {}
    for m in args.models:
        print('Training balanced', m)
        try:
            metrics = train_model_balanced(m, epochs=args.epochs, batch_size=args.batch)
            results[m] = metrics
        except Exception as e:
            print('Error training', m, e)
    with open('C:\\SOFTWARE\\DL Project\\models\\all_results_balanced.json','w') as f:
        json.dump(results, f, indent=2)
    print('Done')
