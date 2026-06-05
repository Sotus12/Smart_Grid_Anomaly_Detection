import os
import time
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score
from models import get_model


DEFAULT_DATA_DIR = r"C:\SOFTWARE\DL Project\outputs\spectrograms"


def train_model(model_name, data_dir=DEFAULT_DATA_DIR, epochs=5, batch_size=32, lr=1e-4, device=None):
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device', device)
    transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    ])
    train_ds = datasets.ImageFolder(os.path.join(data_dir,'train'), transform=transform)
    val_ds = datasets.ImageFolder(os.path.join(data_dir,'val'), transform=transform)
    test_ds = datasets.ImageFolder(os.path.join(data_dir,'test'), transform=transform)

    # sanity checks: ensure both classes exist and have enough samples
    def count_samples(folder):
        out = {}
        if not os.path.isdir(folder):
            return out
        for cls in os.listdir(folder):
            cls_folder = os.path.join(folder, cls)
            if os.path.isdir(cls_folder):
                out[cls] = len([f for f in os.listdir(cls_folder) if f.lower().endswith(('.png','.jpg','.jpeg','.bmp','.tif','.tiff','.webp'))])
        return out

    train_counts = count_samples(os.path.join(data_dir,'train'))
    if len(train_counts) < 2 or min(train_counts.values()) < 5:
        raise RuntimeError(f"Not enough class samples for training: {train_counts}. Regenerate spectrograms with balanced classes or adjust parameters.")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=2)

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
        val_f1 = f1_score(trues, preds, average='binary')
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_f1'].append(val_f1)
        print(f'Epoch {epoch+1}/{epochs} train_loss={train_loss:.4f} val_loss={val_loss:.4f} val_f1={val_f1:.4f}')

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            # save best model
            os.makedirs('C:\\SOFTWARE\\DL Project\\models', exist_ok=True)
            torch.save(model.state_dict(), os.path.join('C:\\SOFTWARE\\DL Project\\models', f'{model_name}_best.pth'))

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
    test_f1 = f1_score(trues, preds, average='binary')
    print('Test acc', test_acc, 'Test f1', test_f1)

    # save metrics
    metrics = {'model':model_name, 'best_val_f1':best_val_f1, 'test_acc':test_acc, 'test_f1':test_f1, 'history':history}
    with open(os.path.join('C:\\SOFTWARE\\DL Project\\models', f'{model_name}_metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)
    return metrics


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--models', nargs='+', default=['resnet18','densenet121'])
    parser.add_argument('--epochs', type=int, default=5)
    args = parser.parse_args()
    results = {}
    for m in args.models:
        print('Training', m)
        try:
            metrics = train_model(m, epochs=args.epochs)
            results[m] = metrics
        except Exception as e:
            print('Error training', m, e)
    with open('C:\\SOFTWARE\\DL Project\\models\\all_results.json','w') as f:
        json.dump(results, f, indent=2)
    print('Done')
