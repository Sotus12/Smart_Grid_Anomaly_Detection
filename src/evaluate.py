import os
import json
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, roc_curve, auc, accuracy_score, f1_score, classification_report
from models import get_model

DATA_DIR = r"C:\SOFTWARE\DL Project\outputs\spectrograms"
MODELS_DIR = r"C:\SOFTWARE\DL Project\models"
PLOTS_DIR = r"C:\SOFTWARE\DL Project\outputs\plots"
os.makedirs(PLOTS_DIR, exist_ok=True)

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
])

test_ds = datasets.ImageFolder(os.path.join(DATA_DIR,'test'), transform=transform)
if len(test_ds) == 0:
    raise RuntimeError('Test dataset is empty. Did you generate spectrograms?')

test_loader = DataLoader(test_ds, batch_size=32, shuffle=False, num_workers=0)
true_labels = [y for _,y in test_ds]

models_to_eval = ['resnet18','densenet121','efficientnet_b0']
results = {}
roc_curves = {}

device = 'cuda' if torch.cuda.is_available() else 'cpu'

for name in models_to_eval:
    weight_file = os.path.join(MODELS_DIR, f'{name}_best.pth')
    metrics_file = os.path.join(MODELS_DIR, f'{name}_metrics.json')
    if not os.path.exists(weight_file):
        print('Skipping', name, '- weights not found at', weight_file)
        continue
    print('Evaluating', name)
    # instantiate model (no pretrained to avoid downloads)
    try:
        model = get_model(name, num_classes=2, pretrained=False)
    except Exception as e:
        print('Error creating model', name, e)
        continue
    model.load_state_dict(torch.load(weight_file, map_location=device))
    model = model.to(device).eval()

    probs_all = []
    preds_all = []
    trues_all = []

    softmax = torch.nn.Softmax(dim=1)
    with torch.no_grad():
        for xb,yb in test_loader:
            xb = xb.to(device)
            out = model(xb)
            p = softmax(out).cpu().numpy()
            preds = p.argmax(axis=1)
            probs_pos = p[:,1]
            probs_all.extend(probs_pos.tolist())
            preds_all.extend(preds.tolist())
            trues_all.extend(yb.numpy().tolist())

    # metrics
    acc = float(accuracy_score(trues_all, preds_all))
    f1 = float(f1_score(trues_all, preds_all, zero_division=0))
    try:
        fpr, tpr, _ = roc_curve(trues_all, probs_all)
        roc_auc = float(auc(fpr, tpr))
    except Exception:
        fpr, tpr, roc_auc = [0],[0],0.0

    cm = confusion_matrix(trues_all, preds_all)

    # save confusion matrix plot
    plt.figure(figsize=(4,4))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(f'Confusion Matrix: {name}')
    plt.colorbar()
    ticks = [0,1]
    plt.xticks(ticks, ['normal','anomaly'], rotation=45)
    plt.yticks(ticks, ['normal','anomaly'])
    thresh = cm.max() / 2. if cm.size else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'), horizontalalignment='center', color='white' if cm[i, j] > thresh else 'black')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    cm_path = os.path.join(PLOTS_DIR, f'cm_{name}.png')
    plt.savefig(cm_path)
    plt.close()

    # store ROC for combined plot
    roc_curves[name] = {'fpr': list(map(float,fpr)), 'tpr': list(map(float,tpr)), 'auc': roc_auc}

    results[name] = {'test_acc': acc, 'test_f1': f1, 'roc_auc': roc_auc, 'confusion_matrix': cm.tolist()}

    # update metrics json with roc_auc
    if os.path.exists(metrics_file):
        try:
            with open(metrics_file,'r') as f:
                js = json.load(f)
        except Exception:
            js = {}
        js['roc_auc'] = roc_auc
        js['test_acc'] = acc
        js['test_f1'] = f1
        with open(metrics_file,'w') as f:
            json.dump(js, f, indent=2)

# save combined results
with open(os.path.join(MODELS_DIR,'all_results.json'),'w') as f:
    json.dump(results, f, indent=2)

# plot ROC curves together
plt.figure(figsize=(6,6))
for name,rc in roc_curves.items():
    plt.plot(rc['fpr'], rc['tpr'], label=f"{name} (AUC={rc['auc']:.3f})")
plt.plot([0,1],[0,1],'k--', alpha=0.5)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves')
plt.legend()
plt.tight_layout()
roc_path = os.path.join(PLOTS_DIR, 'roc_all.png')
plt.savefig(roc_path)
plt.close()

# write summary table (CSV)
import csv
summary_csv = os.path.join(MODELS_DIR,'comparison_table.csv')
with open(summary_csv,'w',newline='') as f:
    w = csv.writer(f)
    w.writerow(['model','test_acc','test_f1','roc_auc','confusion_matrix'])
    for name,vals in results.items():
        w.writerow([name, vals['test_acc'], vals['test_f1'], vals['roc_auc'], json.dumps(vals['confusion_matrix'])])

print('Saved plots to', PLOTS_DIR)
print('Saved comparison CSV to', summary_csv)
print('Results:', json.dumps(results, indent=2))
