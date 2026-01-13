"""
Loss Functions for VQA
Provides various loss functions for training Vietnamese VQA models.
"""

from typing import Optional, Dict, Any, List, Union
from dataclasses import dataclass
from enum import Enum

import torch
import torch.nn as nn
import torch.nn.functional as F


class LossType(Enum):
    """Enumeration of loss types."""
    CROSS_ENTROPY = 'cross_entropy'
    BINARY_CROSS_ENTROPY = 'binary_cross_entropy'
    FOCAL = 'focal'
    LABEL_SMOOTHING = 'label_smoothing'
    SOFT_TARGET = 'soft_target'
    CONTRASTIVE = 'contrastive'
    TRIPLET = 'triplet'


@dataclass
class LossConfig:
    """
    Configuration for loss functions.
    
    Attributes:
        loss_type: Type of loss function
        weight: Loss weight
        reduction: Reduction method
        label_smoothing: Label smoothing factor
        focal_gamma: Focal loss gamma
        class_weights: Per-class weights
    """
    loss_type: str = 'cross_entropy'
    weight: float = 1.0
    reduction: str = 'mean'
    label_smoothing: float = 0.0
    focal_gamma: float = 2.0
    class_weights: Optional[List[float]] = None


class CrossEntropyLoss(nn.Module):
    """
    Standard cross-entropy loss for VQA classification.
    """
    
    def __init__(
        self,
        weight: Optional[torch.Tensor] = None,
        ignore_index: int = -100,
        label_smoothing: float = 0.0,
        reduction: str = 'mean'
    ):
        """
        Initialize cross-entropy loss.
        
        Args:
            weight: Per-class weights
            ignore_index: Index to ignore
            label_smoothing: Label smoothing factor
            reduction: Reduction method
        """
        super().__init__()
        
        self.weight = weight
        self.ignore_index = ignore_index
        self.label_smoothing = label_smoothing
        self.reduction = reduction
        
        self.ce_loss = nn.CrossEntropyLoss(
            weight=weight,
            ignore_index=ignore_index,
            label_smoothing=label_smoothing,
            reduction=reduction
        )
    
    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute cross-entropy loss.
        
        Args:
            logits: Prediction logits [batch, num_classes]
            targets: Target indices [batch]
            
        Returns:
            Loss value
        """
        return self.ce_loss(logits, targets)


class BinaryCrossEntropyLoss(nn.Module):
    """
    Binary cross-entropy loss for multi-label VQA.
    """
    
    def __init__(
        self,
        weight: Optional[torch.Tensor] = None,
        pos_weight: Optional[torch.Tensor] = None,
        reduction: str = 'mean'
    ):
        """
        Initialize binary cross-entropy loss.
        
        Args:
            weight: Per-sample weights
            pos_weight: Positive class weights
            reduction: Reduction method
        """
        super().__init__()
        
        self.weight = weight
        self.pos_weight = pos_weight
        self.reduction = reduction
    
    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute binary cross-entropy loss.
        
        Args:
            logits: Prediction logits [batch, num_classes]
            targets: Binary targets [batch, num_classes]
            
        Returns:
            Loss value
        """
        return F.binary_cross_entropy_with_logits(
            logits,
            targets.float(),
            weight=self.weight,
            pos_weight=self.pos_weight,
            reduction=self.reduction
        )


class FocalLoss(nn.Module):
    """
    Focal loss for handling class imbalance in VQA.
    Focuses on hard examples by down-weighting easy ones.
    """
    
    def __init__(
        self,
        alpha: Optional[torch.Tensor] = None,
        gamma: float = 2.0,
        reduction: str = 'mean'
    ):
        """
        Initialize focal loss.
        
        Args:
            alpha: Per-class weights
            gamma: Focusing parameter (higher = more focus on hard examples)
            reduction: Reduction method
        """
        super().__init__()
        
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute focal loss.
        
        Args:
            logits: Prediction logits [batch, num_classes]
            targets: Target indices [batch]
            
        Returns:
            Loss value
        """
        # Compute softmax probabilities
        probs = F.softmax(logits, dim=-1)
        
        # Get probabilities for true class
        batch_size = targets.size(0)
        target_probs = probs[torch.arange(batch_size), targets]
        
        # Compute focal weight
        focal_weight = (1 - target_probs) ** self.gamma
        
        # Compute cross-entropy
        ce_loss = F.cross_entropy(logits, targets, reduction='none')
        
        # Apply focal weight
        focal_loss = focal_weight * ce_loss
        
        # Apply class weights if provided
        if self.alpha is not None:
            alpha_weight = self.alpha[targets]
            focal_loss = alpha_weight * focal_loss
        
        # Reduce
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class LabelSmoothingLoss(nn.Module):
    """
    Cross-entropy loss with label smoothing for regularization.
    """
    
    def __init__(
        self,
        num_classes: int,
        smoothing: float = 0.1,
        reduction: str = 'mean'
    ):
        """
        Initialize label smoothing loss.
        
        Args:
            num_classes: Number of classes
            smoothing: Smoothing factor (0 = no smoothing, 1 = uniform)
            reduction: Reduction method
        """
        super().__init__()
        
        self.num_classes = num_classes
        self.smoothing = smoothing
        self.reduction = reduction
        self.confidence = 1.0 - smoothing
    
    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute label smoothing loss.
        
        Args:
            logits: Prediction logits [batch, num_classes]
            targets: Target indices [batch]
            
        Returns:
            Loss value
        """
        # Create smoothed targets
        smooth_targets = torch.zeros_like(logits)
        smooth_targets.fill_(self.smoothing / (self.num_classes - 1))
        smooth_targets.scatter_(1, targets.unsqueeze(1), self.confidence)
        
        # Compute log softmax
        log_probs = F.log_softmax(logits, dim=-1)
        
        # Compute loss
        loss = -torch.sum(smooth_targets * log_probs, dim=-1)
        
        # Reduce
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class SoftTargetLoss(nn.Module):
    """
    Loss for soft/multi-answer VQA (e.g., VQA v2 with soft scores).
    """
    
    def __init__(
        self,
        reduction: str = 'mean',
        normalize_targets: bool = True
    ):
        """
        Initialize soft target loss.
        
        Args:
            reduction: Reduction method
            normalize_targets: Whether to normalize soft targets
        """
        super().__init__()
        
        self.reduction = reduction
        self.normalize_targets = normalize_targets
    
    def forward(
        self,
        logits: torch.Tensor,
        soft_targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute soft target loss.
        
        Args:
            logits: Prediction logits [batch, num_classes]
            soft_targets: Soft target scores [batch, num_classes]
            
        Returns:
            Loss value
        """
        # Normalize targets if needed
        if self.normalize_targets:
            target_sum = soft_targets.sum(dim=-1, keepdim=True).clamp(min=1e-8)
            soft_targets = soft_targets / target_sum
        
        # Compute log softmax
        log_probs = F.log_softmax(logits, dim=-1)
        
        # Binary cross-entropy style loss
        loss = -torch.sum(soft_targets * log_probs, dim=-1)
        
        # Reduce
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class ContrastiveLoss(nn.Module):
    """
    Contrastive loss for learning discriminative embeddings.
    """
    
    def __init__(
        self,
        margin: float = 1.0,
        reduction: str = 'mean'
    ):
        """
        Initialize contrastive loss.
        
        Args:
            margin: Margin for negative pairs
            reduction: Reduction method
        """
        super().__init__()
        
        self.margin = margin
        self.reduction = reduction
    
    def forward(
        self,
        embeddings1: torch.Tensor,
        embeddings2: torch.Tensor,
        labels: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute contrastive loss.
        
        Args:
            embeddings1: First embeddings [batch, embed_dim]
            embeddings2: Second embeddings [batch, embed_dim]
            labels: Binary labels (1 = similar, 0 = dissimilar)
            
        Returns:
            Loss value
        """
        # Compute pairwise distances
        distances = F.pairwise_distance(embeddings1, embeddings2)
        
        # Contrastive loss
        # For similar pairs: minimize distance
        # For dissimilar pairs: maximize distance up to margin
        loss = labels * distances.pow(2) + \
               (1 - labels) * F.relu(self.margin - distances).pow(2)
        
        # Reduce
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class TripletLoss(nn.Module):
    """
    Triplet loss for metric learning in VQA embeddings.
    """
    
    def __init__(
        self,
        margin: float = 1.0,
        p: int = 2,
        reduction: str = 'mean'
    ):
        """
        Initialize triplet loss.
        
        Args:
            margin: Margin between positive and negative
            p: Norm degree for distance
            reduction: Reduction method
        """
        super().__init__()
        
        self.margin = margin
        self.p = p
        self.reduction = reduction
        
        self.triplet_loss = nn.TripletMarginLoss(
            margin=margin,
            p=p,
            reduction=reduction
        )
    
    def forward(
        self,
        anchor: torch.Tensor,
        positive: torch.Tensor,
        negative: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute triplet loss.
        
        Args:
            anchor: Anchor embeddings [batch, embed_dim]
            positive: Positive embeddings [batch, embed_dim]
            negative: Negative embeddings [batch, embed_dim]
            
        Returns:
            Loss value
        """
        return self.triplet_loss(anchor, positive, negative)


class InfoNCELoss(nn.Module):
    """
    InfoNCE loss for contrastive learning.
    Used in vision-language pretraining.
    """
    
    def __init__(
        self,
        temperature: float = 0.07,
        reduction: str = 'mean'
    ):
        """
        Initialize InfoNCE loss.
        
        Args:
            temperature: Temperature for softmax
            reduction: Reduction method
        """
        super().__init__()
        
        self.temperature = temperature
        self.reduction = reduction
    
    def forward(
        self,
        query: torch.Tensor,
        positive_key: torch.Tensor,
        negative_keys: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute InfoNCE loss.
        
        Args:
            query: Query embeddings [batch, embed_dim]
            positive_key: Positive key embeddings [batch, embed_dim]
            negative_keys: Negative key embeddings [batch, num_negatives, embed_dim]
            
        Returns:
            Loss value
        """
        # Normalize embeddings
        query = F.normalize(query, dim=-1)
        positive_key = F.normalize(positive_key, dim=-1)
        
        # Positive logits
        pos_logits = torch.sum(query * positive_key, dim=-1, keepdim=True)
        pos_logits = pos_logits / self.temperature
        
        # Negative logits
        if negative_keys is not None:
            negative_keys = F.normalize(negative_keys, dim=-1)
            neg_logits = torch.bmm(negative_keys, query.unsqueeze(-1)).squeeze(-1)
            neg_logits = neg_logits / self.temperature
            
            # Concatenate
            logits = torch.cat([pos_logits, neg_logits], dim=-1)
        else:
            # Use other samples in batch as negatives
            all_logits = torch.mm(query, positive_key.t()) / self.temperature
            labels = torch.arange(query.size(0), device=query.device)
            
            return F.cross_entropy(all_logits, labels, reduction=self.reduction)
        
        # Labels: positive is always at index 0
        labels = torch.zeros(query.size(0), dtype=torch.long, device=query.device)
        
        loss = F.cross_entropy(logits, labels, reduction=self.reduction)
        
        return loss


class MOELoadBalancingLoss(nn.Module):
    """
    Load balancing loss for Mixture of Experts.
    Encourages uniform expert utilization.
    """
    
    def __init__(
        self,
        num_experts: int,
        alpha: float = 0.01,
        reduction: str = 'mean'
    ):
        """
        Initialize MOE load balancing loss.
        
        Args:
            num_experts: Number of experts
            alpha: Loss weight
            reduction: Reduction method
        """
        super().__init__()
        
        self.num_experts = num_experts
        self.alpha = alpha
        self.reduction = reduction
    
    def forward(
        self,
        router_probs: torch.Tensor,
        expert_indices: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute load balancing loss.
        
        Args:
            router_probs: Router probabilities [batch, num_experts]
            expert_indices: Selected expert indices [batch, top_k]
            
        Returns:
            Loss value
        """
        batch_size = router_probs.size(0)
        
        # Mean router probability per expert
        mean_probs = router_probs.mean(dim=0)
        
        # Expert selection frequency
        expert_counts = torch.zeros(self.num_experts, device=router_probs.device)
        for idx in expert_indices.view(-1):
            expert_counts[idx] += 1
        expert_freq = expert_counts / (batch_size * expert_indices.size(1))
        
        # Load balancing loss: encourage uniform distribution
        # cv^2 of selection frequency
        load_balance_loss = self.num_experts * torch.sum(mean_probs * expert_freq)
        
        return self.alpha * load_balance_loss


class VQAMultiTaskLoss(nn.Module):
    """
    Multi-task loss for VQA combining multiple objectives.
    """
    
    def __init__(
        self,
        answer_loss_weight: float = 1.0,
        aux_loss_weight: float = 0.1,
        consistency_loss_weight: float = 0.1,
        use_uncertainty_weighting: bool = False
    ):
        """
        Initialize multi-task loss.
        
        Args:
            answer_loss_weight: Weight for answer prediction loss
            aux_loss_weight: Weight for auxiliary losses
            consistency_loss_weight: Weight for consistency loss
            use_uncertainty_weighting: Use learned uncertainty weights
        """
        super().__init__()
        
        self.answer_loss_weight = answer_loss_weight
        self.aux_loss_weight = aux_loss_weight
        self.consistency_loss_weight = consistency_loss_weight
        self.use_uncertainty_weighting = use_uncertainty_weighting
        
        # Main answer loss
        self.answer_loss = CrossEntropyLoss()
        
        # Uncertainty parameters (learned)
        if use_uncertainty_weighting:
            self.log_var_answer = nn.Parameter(torch.zeros(1))
            self.log_var_aux = nn.Parameter(torch.zeros(1))
            self.log_var_consistency = nn.Parameter(torch.zeros(1))
    
    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        aux_logits: Optional[torch.Tensor] = None,
        aux_targets: Optional[torch.Tensor] = None,
        features1: Optional[torch.Tensor] = None,
        features2: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Compute multi-task loss.
        
        Args:
            logits: Answer prediction logits
            targets: Answer targets
            aux_logits: Auxiliary task logits
            aux_targets: Auxiliary task targets
            features1: Features for consistency
            features2: Augmented features for consistency
            
        Returns:
            Dictionary of losses
        """
        losses = {}
        
        # Answer loss
        answer_loss = self.answer_loss(logits, targets)
        losses['answer_loss'] = answer_loss
        
        # Auxiliary loss
        if aux_logits is not None and aux_targets is not None:
            aux_loss = F.cross_entropy(aux_logits, aux_targets)
            losses['aux_loss'] = aux_loss
        else:
            aux_loss = torch.tensor(0.0, device=logits.device)
        
        # Consistency loss
        if features1 is not None and features2 is not None:
            consistency_loss = F.mse_loss(features1, features2)
            losses['consistency_loss'] = consistency_loss
        else:
            consistency_loss = torch.tensor(0.0, device=logits.device)
        
        # Combine losses
        if self.use_uncertainty_weighting:
            # Uncertainty weighting
            total_loss = (
                torch.exp(-self.log_var_answer) * answer_loss + self.log_var_answer +
                torch.exp(-self.log_var_aux) * aux_loss + self.log_var_aux +
                torch.exp(-self.log_var_consistency) * consistency_loss + self.log_var_consistency
            )
        else:
            total_loss = (
                self.answer_loss_weight * answer_loss +
                self.aux_loss_weight * aux_loss +
                self.consistency_loss_weight * consistency_loss
            )
        
        losses['total_loss'] = total_loss
        
        return losses


def create_loss(config: LossConfig) -> nn.Module:
    """
    Factory function to create loss from config.
    
    Args:
        config: Loss configuration
        
    Returns:
        Loss module
    """
    loss_type = config.loss_type.lower()
    
    if loss_type == 'cross_entropy':
        weight = torch.tensor(config.class_weights) if config.class_weights else None
        return CrossEntropyLoss(
            weight=weight,
            label_smoothing=config.label_smoothing,
            reduction=config.reduction
        )
    
    elif loss_type == 'binary_cross_entropy':
        return BinaryCrossEntropyLoss(reduction=config.reduction)
    
    elif loss_type == 'focal':
        alpha = torch.tensor(config.class_weights) if config.class_weights else None
        return FocalLoss(
            alpha=alpha,
            gamma=config.focal_gamma,
            reduction=config.reduction
        )
    
    elif loss_type == 'soft_target':
        return SoftTargetLoss(reduction=config.reduction)
    
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")


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
