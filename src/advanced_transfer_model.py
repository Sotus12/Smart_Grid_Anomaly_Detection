"""
Attentional ResNet-50 (Breakthrough Model)
Combines deep transfer learning with custom attention for high accuracy.

Design:
1. Backbone: Pre-trained ResNet50 (Frozen initially).
2. Attention Injection: Spatial and Channel Attention applied at multiple scales.
3. Optimization: Designed to exceed 83% accuracy baseline.
"""

import torch
import torch.nn as nn
import torchvision.models as models
from grid_resnet_model import SpatialAttentionModule, ChannelAttentionModule

class AttentionWrapper(nn.Module):
    """Wraps a layer with Channel and Spatial Attention."""
    def __init__(self, in_planes, ratio=16):
        super().__init__()
        self.ca = ChannelAttentionModule(in_planes, ratio=ratio)
        self.sa = SpatialAttentionModule(kernel_size=7)

    def forward(self, x):
        # Attention modules return weights, so we multiply manually here
        # or reuse the module logic. Let's reuse the logic from grid_resnet_model 
        # where SAM returns sigmoid weights and CAM returns sigmoid weights.
        
        # Note: In grid_resnet_model.py, CAM and SAM return the sigmoid maps.
        # We multiply them by the input.
        out = x * self.ca(x)
        out = out * self.sa(out)
        return out

class AttentionalResNet50(nn.Module):
    """
    Advanced model that injects attention into pre-trained ResNet50 stages.
    """
    def __init__(self, num_classes=2, pretrained=True):
        super().__init__()
        # Load base model
        base = models.resnet50(pretrained=pretrained)
        
        # Initial layers
        self.initial = nn.Sequential(
            base.conv1, base.bn1, base.relu, base.maxpool
        )
        
        # ResNet Stages with injected attention
        self.stage1 = nn.Sequential(base.layer1, AttentionWrapper(256))
        self.stage2 = nn.Sequential(base.layer2, AttentionWrapper(512))
        self.stage3 = nn.Sequential(base.layer3, AttentionWrapper(1024))
        self.stage4 = nn.Sequential(base.layer4, AttentionWrapper(2048))
        
        # New Head
        self.avgpool = base.avgpool
        self.dropout = nn.Dropout(0.4)
        self.fc = nn.Linear(2048, num_classes)

    def forward(self, x):
        x = self.initial(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        logits = self.fc(x)
        return logits

def attentional_resnet50(num_classes=2, pretrained=True):
    return AttentionalResNet50(num_classes=num_classes, pretrained=pretrained)

if __name__ == '__main__':
    model = attentional_resnet50(pretrained=False)
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Attentional ResNet-50 Parameters: {params:,}")
    
    x = torch.randn(2, 3, 224, 224)
    y = model(x)
    print(f"Output shape: {y.shape}")
    print("✓ Breakthrough Model Ready")
