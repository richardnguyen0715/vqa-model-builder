"""
Loss Functions Module
Provides loss functions for training Vietnamese VQA models.
"""

from .vqa_losses import (
    LossType,
    LossConfig,
    CrossEntropyLoss,
    BinaryCrossEntropyLoss,
    FocalLoss,
    LabelSmoothingLoss,
    SoftTargetLoss,
    ContrastiveLoss,
    TripletLoss,
    InfoNCELoss,
    MOELoadBalancingLoss,
    VQAMultiTaskLoss,
    create_loss
)

__all__ = [
    'LossType',
    'LossConfig',
    'CrossEntropyLoss',
    'BinaryCrossEntropyLoss',
    'FocalLoss',
    'LabelSmoothingLoss',
    'SoftTargetLoss',
    'ContrastiveLoss',
    'TripletLoss',
    'InfoNCELoss',
    'MOELoadBalancingLoss',
    'VQAMultiTaskLoss',
    'create_loss'
]
