"""
Focal Loss Implementation for Class Imbalance

Focal Loss is a key technique for handling extreme class imbalance:
- Standard CrossEntropyLoss: L(p) = -log(p_t)
- Focal Loss: L(p) = -(1-p_t)^gamma * log(p_t)

The (1-p_t)^gamma term:
- For easy examples (p_t high): becomes very small, down-weights easy negatives
- For hard examples (p_t low): remains close to 1, focuses on hard positives
- Gamma controls focusing strength (typical values: 2-5)

This is especially important for our 83:16 class imbalance.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance
    Reference: https://arxiv.org/abs/1708.02002 (Lin et al.)
    """
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        """
        Args:
            inputs: Model outputs (logits) of shape (B, C)
            targets: Target labels of shape (B,)
        
        Returns:
            Focal loss value
        """
        p = F.softmax(inputs, dim=1)  # (B, C)
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')  # (B,)
        
        # Get probability of the true class
        p_t = p.gather(1, targets.view(-1, 1)).squeeze(1)  # (B,)
        
        # Focal term: (1 - p_t)^gamma
        focal_weight = (1 - p_t) ** self.gamma
        
        # Apply focal loss
        focal_loss = self.alpha * focal_weight * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss
