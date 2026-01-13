"""
MOE Utilities Module
Helper functions and utilities for Mixture of Experts.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Any
import json


def compute_expert_capacity(
    num_tokens: int,
    num_experts: int,
    top_k: int,
    capacity_factor: float = 1.25
) -> int:
    """
    Compute capacity per expert.
    
    Capacity = (num_tokens * top_k * capacity_factor) / num_experts
    
    Args:
        num_tokens: Total number of tokens
        num_experts: Number of experts
        top_k: Experts per token
        capacity_factor: Multiplier for capacity
        
    Returns:
        Capacity per expert
    """
    return int(capacity_factor * num_tokens * top_k / num_experts)


def compute_load_balance_loss(
    router_probs: torch.Tensor,
    expert_indices: torch.Tensor,
    num_experts: int,
    weight: float = 0.01
) -> torch.Tensor:
    """
    Compute load balancing auxiliary loss.
    Encourages uniform distribution of tokens across experts.
    
    Args:
        router_probs: Router probabilities [batch_size, seq_len, num_experts]
        expert_indices: Selected expert indices [batch_size, seq_len, top_k]
        num_experts: Total number of experts
        weight: Loss weight
        
    Returns:
        Load balance loss
    """
    batch_size, seq_len, _ = router_probs.shape
    num_tokens = batch_size * seq_len
    
    # Fraction of tokens per expert
    expert_mask = torch.zeros(
        batch_size, seq_len, num_experts,
        device=expert_indices.device
    )
    expert_mask.scatter_add_(
        2,
        expert_indices,
        torch.ones_like(expert_indices, dtype=torch.float)
    )
    tokens_per_expert = expert_mask.sum(dim=[0, 1])
    fraction_tokens = tokens_per_expert / num_tokens
    
    # Mean routing probability per expert
    mean_router_prob = router_probs.mean(dim=[0, 1])
    
    # Load balance loss
    loss = num_experts * torch.sum(fraction_tokens * mean_router_prob)
    
    return weight * loss


def compute_router_z_loss(
    router_logits: torch.Tensor,
    weight: float = 0.001
) -> torch.Tensor:
    """
    Compute router z-loss for training stability.
    Penalizes large router logits.
    
    Args:
        router_logits: Raw router logits [batch_size, seq_len, num_experts]
        weight: Loss weight
        
    Returns:
        Z-loss
    """
    # Mean of squared log-sum-exp of logits
    z_loss = torch.logsumexp(router_logits, dim=-1).pow(2).mean()
    return weight * z_loss


def get_expert_utilization(
    expert_indices: torch.Tensor,
    num_experts: int
) -> Dict[int, float]:
    """
    Compute utilization ratio for each expert.
    
    Args:
        expert_indices: Expert assignments [batch_size, seq_len, top_k]
        num_experts: Total number of experts
        
    Returns:
        Dictionary mapping expert_id to utilization ratio
    """
    total_assignments = expert_indices.numel()
    utilization = {}
    
    for expert_id in range(num_experts):
        count = (expert_indices == expert_id).sum().item()
        utilization[expert_id] = count / total_assignments
    
    return utilization


def compute_expert_entropy(
    router_probs: torch.Tensor
) -> torch.Tensor:
    """
    Compute entropy of expert routing distribution.
    Higher entropy indicates more uniform routing.
    
    Args:
        router_probs: Router probabilities [batch_size, seq_len, num_experts]
        
    Returns:
        Mean entropy value
    """
    # Add small epsilon for numerical stability
    eps = 1e-10
    entropy = -torch.sum(router_probs * torch.log(router_probs + eps), dim=-1)
    return entropy.mean()


class ExpertDropout(nn.Module):
    """
    Expert-level dropout.
    Randomly drops entire experts during training.
    """
    
    def __init__(self, num_experts: int, drop_rate: float = 0.1):
        """
        Initialize expert dropout.
        
        Args:
            num_experts: Total number of experts
            drop_rate: Probability of dropping an expert
        """
        super().__init__()
        self.num_experts = num_experts
        self.drop_rate = drop_rate
    
    def forward(
        self,
        expert_weights: torch.Tensor,
        expert_indices: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply expert dropout.
        
        Args:
            expert_weights: Routing weights [batch_size, seq_len, top_k]
            expert_indices: Expert indices [batch_size, seq_len, top_k]
            
        Returns:
            Tuple of (modified_weights, modified_indices)
        """
        if not self.training or self.drop_rate == 0:
            return expert_weights, expert_indices
        
        # Generate dropout mask for experts
        drop_mask = torch.bernoulli(
            torch.full((self.num_experts,), 1 - self.drop_rate, device=expert_indices.device)
        )
        
        # Apply mask to weights
        expert_mask = drop_mask[expert_indices]  # [B, S, K]
        modified_weights = expert_weights * expert_mask
        
        # Renormalize weights
        weight_sum = modified_weights.sum(dim=-1, keepdim=True)
        modified_weights = modified_weights / (weight_sum + 1e-10)
        
        return modified_weights, expert_indices


class ExpertParallelWrapper(nn.Module):
    """
    Wrapper for parallelizing expert computation across devices.
    Distributes experts across multiple GPUs.
    """
    
    def __init__(
        self,
        experts: nn.ModuleList,
        device_ids: Optional[List[int]] = None
    ):
        """
        Initialize parallel wrapper.
        
        Args:
            experts: List of expert modules
            device_ids: GPU device IDs to use
        """
        super().__init__()
        self.experts = experts
        self.num_experts = len(experts)
        
        if device_ids is None:
            device_ids = list(range(torch.cuda.device_count()))
        self.device_ids = device_ids
        
        # Distribute experts across devices
        if len(device_ids) > 0:
            experts_per_device = self.num_experts // len(device_ids)
            for i, expert in enumerate(experts):
                device_idx = min(i // experts_per_device, len(device_ids) - 1)
                expert.to(f'cuda:{device_ids[device_idx]}')
    
    def forward(
        self,
        x: torch.Tensor,
        expert_id: int,
        **kwargs
    ) -> torch.Tensor:
        """
        Forward pass through specific expert.
        
        Args:
            x: Input tensor
            expert_id: Expert index
            **kwargs: Additional arguments
            
        Returns:
            Expert output
        """
        expert = self.experts[expert_id]
        device = next(expert.parameters()).device
        
        # Move input to expert's device
        x_device = x.to(device)
        
        # Process
        output = expert(x_device, **kwargs)
        
        # Move output back
        return output.to(x.device)


def save_moe_checkpoint(
    moe_layer: nn.Module,
    path: str,
    additional_info: Optional[Dict] = None
):
    """
    Save MOE layer checkpoint with metadata.
    
    Args:
        moe_layer: MOE layer to save
        path: Save path
        additional_info: Additional metadata to save
    """
    checkpoint = {
        'state_dict': moe_layer.state_dict(),
        'num_experts': moe_layer.num_experts,
        'input_dim': moe_layer.input_dim,
        'output_dim': moe_layer.output_dim,
    }
    
    if additional_info:
        checkpoint['additional_info'] = additional_info
    
    torch.save(checkpoint, path)


def load_moe_checkpoint(
    moe_layer: nn.Module,
    path: str,
    strict: bool = True
) -> Dict:
    """
    Load MOE layer checkpoint.
    
    Args:
        moe_layer: MOE layer to load into
        path: Checkpoint path
        strict: Whether to strictly enforce state dict matching
        
    Returns:
        Additional info from checkpoint
    """
    checkpoint = torch.load(path, map_location='cpu')
    moe_layer.load_state_dict(checkpoint['state_dict'], strict=strict)
    
    return checkpoint.get('additional_info', {})


def analyze_routing_patterns(
    router_probs: torch.Tensor,
    expert_indices: torch.Tensor,
    num_experts: int
) -> Dict[str, Any]:
    """
    Analyze expert routing patterns.
    
    Args:
        router_probs: Router probabilities
        expert_indices: Selected expert indices
        num_experts: Number of experts
        
    Returns:
        Dictionary with analysis results
    """
    analysis = {
        'expert_utilization': get_expert_utilization(expert_indices, num_experts),
        'routing_entropy': compute_expert_entropy(router_probs).item(),
        'max_prob_mean': router_probs.max(dim=-1).values.mean().item(),
        'min_prob_mean': router_probs.min(dim=-1).values.mean().item(),
    }
    
    # Compute correlation between expert selections
    flat_indices = expert_indices.view(-1, expert_indices.size(-1))
    co_selection = torch.zeros(num_experts, num_experts, device=expert_indices.device)
    
    for i in range(flat_indices.size(0)):
        for j in range(flat_indices.size(-1)):
            for k in range(j + 1, flat_indices.size(-1)):
                e1, e2 = flat_indices[i, j].item(), flat_indices[i, k].item()
                co_selection[e1, e2] += 1
                co_selection[e2, e1] += 1
    
    analysis['expert_co_selection'] = co_selection.cpu().numpy().tolist()
    
    return analysis
