import torch
import torch.nn as nn
import torchvision.models as tv
from attention_model import HybridCNNAttention
from grid_resnet_model import garesnet18
from advanced_transfer_model import attentional_resnet50
from star_net import build_star_net


def get_model(name, num_classes=2, pretrained=True, **kwargs):
    name = name.lower()
    if name == 'resnet18':
        m = tv.resnet18(pretrained=pretrained)
        in_f = m.fc.in_features
        m.fc = nn.Linear(in_f, num_classes)
        return m
    if name == 'densenet121':
        m = tv.densenet121(pretrained=pretrained)
        in_f = m.classifier.in_features
        m.classifier = nn.Linear(in_f, num_classes)
        return m
    if name == 'efficientnet_b0':
        try:
            m = tv.efficientnet_b0(pretrained=pretrained)
            in_f = m.classifier[1].in_features
            m.classifier[1] = nn.Linear(in_f, num_classes)
            return m
        except Exception:
            raise ValueError('efficientnet_b0 not available in this torchvision version')
    if name == 'hybrid_attention':
        # Custom CNN with Spatial + Channel Attention modules
        return HybridCNNAttention(num_classes=num_classes)
    if name == 'garesnet':
        # Grid-Attention Residual Network (custom residual attention)
        return garesnet18(num_classes=num_classes)
    if name == 'att_resnet50':
        # Attentional ResNet-50 (breakthrough model)
        return attentional_resnet50(num_classes=num_classes, pretrained=pretrained)
    if name == 'star_net':
        # STAR-Net: Spectral-Temporal Anomaly Reasoning Network (novel)
        # Dual-stream physics-informed architecture with CSTFG + AIH
        # Use with ACFL loss from acfl_loss.py for best performance
        metric_dim = kwargs.get('metric_dim', 64)
        dropout    = kwargs.get('dropout', 0.3)
        return build_star_net(
            num_classes=num_classes,
            metric_dim=metric_dim,
            dropout=dropout,
        )
    raise ValueError(f'Unknown model: {name}. '
                     f'Valid options: resnet18, densenet121, efficientnet_b0, '
                     f'hybrid_attention, garesnet, att_resnet50, star_net')
