import sys
sys.path.insert(0, r'C:\SOFTWARE\DL Project\src')

import torch
from models import get_model
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import numpy as np

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = get_model('hybrid_attention', num_classes=2).to(device)
model.load_state_dict(torch.load(r'C:\SOFTWARE\DL Project\models\hybrid_attention_best.pth'))
model.eval()

test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

test_ds = datasets.ImageFolder(r'C:\SOFTWARE\DL Project\outputs\spectrograms\test', transform=test_transform)
test_loader = DataLoader(test_ds, batch_size=32, shuffle=False)

preds = []
labels = []
probs = []

with torch.no_grad():
    for images, batch_labels in test_loader:
        outputs = model(images.to(device))
        prob = torch.softmax(outputs, dim=1)
        _, pred = torch.max(outputs, 1)
        preds.extend(pred.cpu().numpy())
        labels.extend(batch_labels.numpy())
        probs.extend(prob.cpu().numpy())

print(f'Total predictions: {len(preds)}')
print(f'Predicted 0 (normal): {sum(p == 0 for p in preds)}')
print(f'Predicted 1 (anomaly): {sum(p == 1 for p in preds)}')
print(f'Actual 0 (normal): {sum(l == 0 for l in labels)}')
print(f'Actual 1 (anomaly): {sum(l == 1 for l in labels)}')

# Check confidence scores
anomaly_probs = np.array([probs[i][1] for i in range(len(probs)) if labels[i] == 1])
normal_probs = np.array([probs[i][1] for i in range(len(probs)) if labels[i] == 0])

print(f'\nAnomalies - Predicted as anomaly prob (mean): {anomaly_probs.mean():.4f}')
print(f'Normals - Predicted as anomaly prob (mean): {normal_probs.mean():.4f}')
