"""
Expert Router Module
Implements various routing mechanisms for Mixture of Experts.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from abc import ABC, abstractmethod
from typing import Optional, Tuple, Dict, Any
import math


class BaseRouter(ABC, nn.Module):
    """
    Abstract base class for expert routers.
    Routes input tokens to appropriate experts based on learned gating.
    """
    
    def __init__(
        self,
        input_dim: int,
        num_experts: int,
        top_k: int = 2
    ):
        """
        Initialize base router.
        
        Args:
            input_dim: Dimension of input features
            num_experts: Total number of experts
            top_k: Number of experts to route each token to
        """
        super().__init__()
        self.input_dim = input_dim
        self.num_experts = num_experts
        self.top_k = top_k
        
        # Gating network
        self.gate = nn.Linear(input_dim, num_experts, bias=False)
    
    @abstractmethod
    def forward(
        self,
        x: torch.Tensor,
        **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """
        Route input to experts.
        
        Args:
            x: Input tensor [batch_size, seq_len, input_dim]
            
        Returns:
            Tuple of:
                - routing_weights: [batch_size, seq_len, num_experts]
                - expert_indices: [batch_size, seq_len, top_k]
                - aux_outputs: Dictionary with auxiliary outputs (load balancing loss, etc.)
        """
        pass
    
    def _compute_routing_logits(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute routing logits from input.
        
        Args:
            x: Input tensor
            
        Returns:
            Routing logits [batch_size, seq_len, num_experts]
        """
        return self.gate(x)


class TopKRouter(BaseRouter):
    """
    Top-K Router implementation.
    Routes each token to top-k experts based on gating scores.
    
    Reference: Shazeer et al. "Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer"
    """
    
    def __init__(
        self,
        input_dim: int,
        num_experts: int,
        top_k: int = 2,
        use_aux_loss: bool = True,
        load_balance_weight: float = 0.01
    ):
        """
        Initialize Top-K router.
        
        Args:
            input_dim: Dimension of input features
            num_experts: Total number of experts
            top_k: Number of experts per token
            use_aux_loss: Whether to compute load balancing loss
            load_balance_weight: Weight for load balancing loss
        """
        super().__init__(input_dim, num_experts, top_k)
        self.use_aux_loss = use_aux_loss
        self.load_balance_weight = load_balance_weight
    
    def forward(
        self,
        x: torch.Tensor,
        **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """
        Route input using top-k selection.
        
        Args:
            x: Input tensor [batch_size, seq_len, input_dim]
            
        Returns:
            Tuple of routing_weights, expert_indices, aux_outputs
        """
        batch_size, seq_len, _ = x.shape
        
        # Compute routing logits
        logits = self._compute_routing_logits(x)  # [B, S, E]
        
        # Get top-k experts
        routing_weights, expert_indices = torch.topk(
            F.softmax(logits, dim=-1),
            self.top_k,
            dim=-1
        )  # [B, S, K], [B, S, K]
        
        # Normalize routing weights
        routing_weights = routing_weights / routing_weights.sum(dim=-1, keepdim=True)
        
        # Compute auxiliary outputs
        aux_outputs = {}
        if self.use_aux_loss:
            aux_outputs['load_balance_loss'] = self._compute_load_balance_loss(
                logits, expert_indices
            )
            aux_outputs['router_probs'] = F.softmax(logits, dim=-1)
        
        return routing_weights, expert_indices, aux_outputs
    
    def _compute_load_balance_loss(
        self,
        logits: torch.Tensor,
        expert_indices: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute load balancing auxiliary loss.
        Encourages uniform distribution of tokens across experts.
        
        Args:
            logits: Router logits [batch_size, seq_len, num_experts]
            expert_indices: Selected expert indices [batch_size, seq_len, top_k]
            
        Returns:
            Load balance loss scalar
        """
        batch_size, seq_len, _ = logits.shape
        num_tokens = batch_size * seq_len
        
        # Compute fraction of tokens routed to each expert
        expert_mask = F.one_hot(expert_indices, self.num_experts).float()
        expert_mask = expert_mask.sum(dim=2)  # [B, S, E]
        tokens_per_expert = expert_mask.sum(dim=[0, 1])  # [E]
        fraction_tokens = tokens_per_expert / num_tokens
        
        # Compute mean routing probability to each expert
        router_probs = F.softmax(logits, dim=-1)
        mean_router_prob = router_probs.mean(dim=[0, 1])  # [E]
        
        # Load balance loss: dot product of fractions
        load_balance_loss = self.num_experts * torch.sum(
            fraction_tokens * mean_router_prob
        )
        
        return self.load_balance_weight * load_balance_loss


class SoftRouter(BaseRouter):
    """
    Soft Router implementation.
    Routes tokens to all experts with soft weights (no sparsity).
    Useful for smaller models where computation is not a bottleneck.
    """
    
    def __init__(
        self,
        input_dim: int,
        num_experts: int,
        temperature: float = 1.0
    ):
        """
        Initialize soft router.
        
        Args:
            input_dim: Dimension of input features
            num_experts: Total number of experts
            temperature: Softmax temperature for sharper/smoother routing
        """
        super().__init__(input_dim, num_experts, num_experts)  # top_k = all experts
        self.temperature = temperature
    
    def forward(
        self,
        x: torch.Tensor,
        **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """
        Route input using soft weighting.
        
        Args:
            x: Input tensor [batch_size, seq_len, input_dim]
            
        Returns:
            Tuple of routing_weights, expert_indices, aux_outputs
        """
        # Compute routing logits
        logits = self._compute_routing_logits(x) / self.temperature  # [B, S, E]
        
        # Soft routing weights
        routing_weights = F.softmax(logits, dim=-1)  # [B, S, E]
        
        # All experts are selected
        expert_indices = torch.arange(
            self.num_experts, device=x.device
        ).expand(x.size(0), x.size(1), -1)  # [B, S, E]
        
        aux_outputs = {
            'router_probs': routing_weights,
            'entropy': self._compute_entropy(routing_weights)
        }
        
        return routing_weights, expert_indices, aux_outputs
    
    def _compute_entropy(self, probs: torch.Tensor) -> torch.Tensor:
        """
        Compute routing entropy for monitoring.
        
        Args:
            probs: Routing probabilities
            
        Returns:
            Mean entropy value
        """
        entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=-1)
        return entropy.mean()


class NoisyTopKRouter(BaseRouter):
    """
    Noisy Top-K Router with trainable noise.
    Adds learned noise to routing decisions for better exploration.
    
    Reference: Shazeer et al. "Outrageously Large Neural Networks"
    """
    
    def __init__(
        self,
        input_dim: int,
        num_experts: int,
        top_k: int = 2,
        noise_std: float = 1.0,
        use_aux_loss: bool = True,
        load_balance_weight: float = 0.01
    ):
        """
        Initialize noisy top-k router.
        
        Args:
            input_dim: Dimension of input features
            num_experts: Total number of experts
            top_k: Number of experts per token
            noise_std: Standard deviation of routing noise
            use_aux_loss: Whether to compute load balancing loss
            load_balance_weight: Weight for load balancing loss
        """
        super().__init__(input_dim, num_experts, top_k)
        self.noise_std = noise_std
        self.use_aux_loss = use_aux_loss
        self.load_balance_weight = load_balance_weight
        
        # Learnable noise scale
        self.w_noise = nn.Linear(input_dim, num_experts, bias=False)
    
    def forward(
        self,
        x: torch.Tensor,
        **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """
        Route input using noisy top-k selection.
        
        Args:
            x: Input tensor [batch_size, seq_len, input_dim]
            
        Returns:
            Tuple of routing_weights, expert_indices, aux_outputs
        """
        # Compute clean routing logits
        clean_logits = self._compute_routing_logits(x)  # [B, S, E]
        
        # Add noise during training
        if self.training:
            noise_scale = F.softplus(self.w_noise(x))
            noise = torch.randn_like(clean_logits) * noise_scale * self.noise_std
            logits = clean_logits + noise
        else:
            logits = clean_logits
        
        # Get top-k experts
        routing_weights, expert_indices = torch.topk(
            F.softmax(logits, dim=-1),
            self.top_k,
            dim=-1
        )
        
        # Normalize routing weights
        routing_weights = routing_weights / routing_weights.sum(dim=-1, keepdim=True)
        
        # Auxiliary outputs
        aux_outputs = {}
        if self.use_aux_loss:
            aux_outputs['load_balance_loss'] = self._compute_load_balance_loss(
                clean_logits, expert_indices
            )
            aux_outputs['router_probs'] = F.softmax(clean_logits, dim=-1)
            aux_outputs['noise_scale'] = noise_scale.mean() if self.training else 0.0
        
        return routing_weights, expert_indices, aux_outputs
    
    def _compute_load_balance_loss(
        self,
        logits: torch.Tensor,
        expert_indices: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute load balancing loss.
        
        Args:
            logits: Router logits
            expert_indices: Selected expert indices
            
        Returns:
            Load balance loss
        """
        batch_size, seq_len, _ = logits.shape
        num_tokens = batch_size * seq_len
        
        # Fraction of tokens per expert
        expert_mask = F.one_hot(expert_indices, self.num_experts).float()
        expert_mask = expert_mask.sum(dim=2)
        tokens_per_expert = expert_mask.sum(dim=[0, 1])
        fraction_tokens = tokens_per_expert / num_tokens
        
        # Mean routing probability
        router_probs = F.softmax(logits, dim=-1)
        mean_router_prob = router_probs.mean(dim=[0, 1])
        
        # Load balance loss
        load_balance_loss = self.num_experts * torch.sum(
            fraction_tokens * mean_router_prob
        )
        
        return self.load_balance_weight * load_balance_loss


class ExpertChoiceRouter(BaseRouter):
    """
    Expert Choice Router.
    Instead of tokens choosing experts, experts choose tokens.
    Ensures perfect load balancing.
    
    Reference: Zhou et al. "Mixture-of-Experts with Expert Choice Routing"
    """
    
    def __init__(
        self,
        input_dim: int,
        num_experts: int,
        capacity_factor: float = 1.25
    ):
        """
        Initialize expert choice router.
        
        Args:
            input_dim: Dimension of input features
            num_experts: Total number of experts
            capacity_factor: Factor to determine expert capacity
        """
        super().__init__(input_dim, num_experts, 1)
        self.capacity_factor = capacity_factor
    
    def forward(
        self,
        x: torch.Tensor,
        **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """
        Route using expert choice mechanism.
        
        Args:
            x: Input tensor [batch_size, seq_len, input_dim]
            
        Returns:
            Tuple of routing_weights, expert_indices, aux_outputs
        """
        batch_size, seq_len, _ = x.shape
        num_tokens = batch_size * seq_len
        
        # Compute capacity per expert
        capacity = int(self.capacity_factor * num_tokens / self.num_experts)
        
        # Compute routing scores
        logits = self._compute_routing_logits(x)  # [B, S, E]
        scores = F.softmax(logits, dim=1)  # Softmax over tokens
        
        # Reshape for expert selection
        scores_flat = scores.view(num_tokens, self.num_experts)  # [N, E]
        
        # Each expert selects top tokens
        expert_indices = torch.zeros(
            num_tokens, dtype=torch.long, device=x.device
        )
        routing_weights = torch.zeros(
            num_tokens, device=x.device
        )
        
        for expert_id in range(self.num_experts):
            expert_scores = scores_flat[:, expert_id]
            top_scores, top_indices = torch.topk(
                expert_scores, 
                min(capacity, num_tokens),
                dim=0
            )
            expert_indices[top_indices] = expert_id
            routing_weights[top_indices] = top_scores
        
        # Reshape back
        expert_indices = expert_indices.view(batch_size, seq_len, 1)
        routing_weights = routing_weights.view(batch_size, seq_len, 1)
        
        aux_outputs = {
            'router_probs': scores,
            'capacity': capacity
        }
        
        return routing_weights, expert_indices, aux_outputs


def create_router(
    router_type: str,
    input_dim: int,
    num_experts: int,
    **kwargs
) -> BaseRouter:
    """
    Factory function to create router by type.
    
    Args:
        router_type: Type of router (topk, soft, noisy_topk, expert_choice)
        input_dim: Input feature dimension
        num_experts: Number of experts
        **kwargs: Additional router-specific arguments
        
    Returns:
        Router instance
    """
    routers = {
        'topk': TopKRouter,
        'soft': SoftRouter,
        'noisy_topk': NoisyTopKRouter,
        'expert_choice': ExpertChoiceRouter
    }
    
    if router_type not in routers:
        raise ValueError(f"Unknown router type: {router_type}. Available: {list(routers.keys())}")
    
    return routers[router_type](input_dim, num_experts, **kwargs)
