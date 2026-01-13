"""
Base Expert Module
Defines abstract base class for all expert implementations.
"""

import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, Tuple


class BaseExpert(ABC, nn.Module):
    """
    Abstract base class for expert networks in MOE.
    All expert implementations should inherit from this class.
    
    Attributes:
        input_dim: Dimension of input features
        hidden_dim: Dimension of hidden layers
        output_dim: Dimension of output features
        expert_id: Unique identifier for the expert
    """
    
    def __init__(
        self,
        input_dim: int = 768,
        hidden_dim: int = 3072,
        output_dim: int = 768,
        expert_id: Optional[int] = None,
        dropout: float = 0.1
    ):
        """
        Initialize base expert.
        
        Args:
            input_dim: Input feature dimension
            hidden_dim: Hidden layer dimension
            output_dim: Output feature dimension
            expert_id: Unique identifier for this expert
            dropout: Dropout rate
        """
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.expert_id = expert_id
        self.dropout_rate = dropout
        
        # Track expert usage for load balancing
        self.register_buffer('usage_count', torch.tensor(0.0))
        self.register_buffer('total_tokens', torch.tensor(0.0))
    
    @abstractmethod
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        **kwargs
    ) -> torch.Tensor:
        """
        Forward pass through expert.
        
        Args:
            x: Input tensor [batch_size, seq_len, input_dim] or [batch_size, input_dim]
            mask: Optional attention mask
            **kwargs: Additional arguments for specific experts
            
        Returns:
            Expert output tensor
        """
        pass
    
    def update_usage_stats(self, num_tokens: int):
        """
        Update expert usage statistics.
        
        Args:
            num_tokens: Number of tokens processed by this expert
        """
        self.usage_count += 1
        self.total_tokens += num_tokens
    
    def get_usage_ratio(self) -> float:
        """
        Get ratio of tokens processed by this expert.
        
        Returns:
            Usage ratio (0 to 1)
        """
        if self.total_tokens == 0:
            return 0.0
        return (self.usage_count / self.total_tokens).item()
    
    def reset_usage_stats(self):
        """Reset usage statistics."""
        self.usage_count.zero_()
        self.total_tokens.zero_()
    
    def get_expert_info(self) -> Dict[str, Any]:
        """
        Get expert information dictionary.
        
        Returns:
            Dictionary containing expert metadata
        """
        return {
            'expert_id': self.expert_id,
            'input_dim': self.input_dim,
            'hidden_dim': self.hidden_dim,
            'output_dim': self.output_dim,
            'usage_ratio': self.get_usage_ratio(),
            'num_parameters': sum(p.numel() for p in self.parameters())
        }


class ExpertWithCapacity(BaseExpert):
    """
    Expert with capacity constraint.
    Limits number of tokens that can be processed to prevent memory issues.
    """
    
    def __init__(
        self,
        input_dim: int = 768,
        hidden_dim: int = 3072,
        output_dim: int = 768,
        expert_id: Optional[int] = None,
        dropout: float = 0.1,
        capacity: Optional[int] = None
    ):
        """
        Initialize expert with capacity.
        
        Args:
            input_dim: Input feature dimension
            hidden_dim: Hidden layer dimension
            output_dim: Output feature dimension
            expert_id: Unique identifier
            dropout: Dropout rate
            capacity: Maximum number of tokens (None for unlimited)
        """
        super().__init__(input_dim, hidden_dim, output_dim, expert_id, dropout)
        self.capacity = capacity
    
    def apply_capacity_constraint(
        self,
        x: torch.Tensor,
        routing_weights: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Apply capacity constraint to input tokens.
        
        Args:
            x: Input tensor [num_tokens, input_dim]
            routing_weights: Routing weights for each token
            
        Returns:
            Tuple of (selected_tokens, selected_weights, selected_indices)
        """
        if self.capacity is None or x.size(0) <= self.capacity:
            indices = torch.arange(x.size(0), device=x.device)
            return x, routing_weights, indices
        
        # Select top-capacity tokens based on routing weights
        _, top_indices = torch.topk(routing_weights, self.capacity)
        selected_x = x[top_indices]
        selected_weights = routing_weights[top_indices]
        
        return selected_x, selected_weights, top_indices
