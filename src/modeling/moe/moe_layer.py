"""
MOE Layer Module
Implements Mixture of Experts layers for VQA.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any, List, Tuple, Union

from src.modeling.moe.base_expert import BaseExpert
from src.modeling.moe.router import (
    BaseRouter,
    TopKRouter,
    NoisyTopKRouter,
    SoftRouter,
    create_router
)
from src.modeling.moe.expert_types import (
    FeedForwardExpert,
    VisionExpert,
    TextExpert,
    MultimodalExpert,
    create_expert
)
from src.modeling.moe.moe_config import MOEConfig


class MOELayer(nn.Module):
    """
    Standard Mixture of Experts Layer.
    Routes tokens to multiple experts and combines outputs.
    
    Architecture:
    1. Router computes expert assignments
    2. Tokens are dispatched to selected experts
    3. Expert outputs are combined using routing weights
    
    Reference: Shazeer et al. "Outrageously Large Neural Networks"
    """
    
    def __init__(
        self,
        config: Optional[MOEConfig] = None,
        input_dim: int = 768,
        hidden_dim: int = 3072,
        output_dim: int = 768,
        num_experts: int = 8,
        top_k: int = 2,
        router_type: str = 'topk',
        expert_type: str = 'feedforward',
        dropout: float = 0.1,
        use_aux_loss: bool = True,
        load_balance_weight: float = 0.01
    ):
        """
        Initialize MOE layer.
        
        Args:
            config: MOE configuration (overrides other arguments if provided)
            input_dim: Input feature dimension
            hidden_dim: Hidden dimension for experts
            output_dim: Output feature dimension
            num_experts: Number of experts
            top_k: Number of experts per token
            router_type: Type of router (topk, soft, noisy_topk)
            expert_type: Type of expert (feedforward, vision, text, multimodal)
            dropout: Dropout rate
            use_aux_loss: Whether to compute auxiliary load balancing loss
            load_balance_weight: Weight for load balancing loss
        """
        super().__init__()
        
        # Use config if provided
        if config is not None:
            input_dim = config.input_dim
            hidden_dim = config.hidden_dim
            output_dim = config.output_dim
            num_experts = config.num_experts
            top_k = config.num_experts_per_token
            dropout = config.expert_dropout
            if config.router_config:
                router_type = config.router_config.router_type
                use_aux_loss = config.router_config.use_aux_loss
                load_balance_weight = config.router_config.load_balance_weight
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_experts = num_experts
        self.top_k = top_k
        
        # Create router
        self.router = create_router(
            router_type=router_type,
            input_dim=input_dim,
            num_experts=num_experts,
            top_k=top_k,
            use_aux_loss=use_aux_loss,
            load_balance_weight=load_balance_weight
        )
        
        # Create experts
        self.experts = nn.ModuleList([
            create_expert(
                expert_type=expert_type,
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                output_dim=output_dim,
                expert_id=i,
                dropout=dropout
            )
            for i in range(num_experts)
        ])
        
        # Output layer norm
        self.output_norm = nn.LayerNorm(output_dim)
        
        # Track auxiliary outputs
        self.aux_outputs: Dict[str, Any] = {}
    
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        **kwargs
    ) -> torch.Tensor:
        """
        Forward pass through MOE layer.
        
        Args:
            x: Input tensor [batch_size, seq_len, input_dim]
            mask: Optional attention mask
            **kwargs: Additional arguments passed to experts
            
        Returns:
            Output tensor [batch_size, seq_len, output_dim]
        """
        batch_size, seq_len, _ = x.shape
        
        # Get routing weights and expert indices
        routing_weights, expert_indices, aux_outputs = self.router(x)
        self.aux_outputs = aux_outputs
        
        # Initialize output
        output = torch.zeros(
            batch_size, seq_len, self.output_dim,
            device=x.device, dtype=x.dtype
        )
        
        # Process each expert
        for expert_id, expert in enumerate(self.experts):
            # Find tokens assigned to this expert
            expert_mask = (expert_indices == expert_id).any(dim=-1)  # [B, S]
            
            if not expert_mask.any():
                continue
            
            # Get weights for this expert
            expert_weight_mask = (expert_indices == expert_id).float()  # [B, S, K]
            expert_weights = (routing_weights * expert_weight_mask).sum(dim=-1)  # [B, S]
            
            # Process through expert
            expert_output = expert(x, mask=mask, **kwargs)  # [B, S, D]
            
            # Weight and accumulate
            weighted_output = expert_output * expert_weights.unsqueeze(-1)
            output = output + weighted_output
        
        # Layer norm
        output = self.output_norm(output)
        
        return output
    
    def get_aux_loss(self) -> torch.Tensor:
        """
        Get auxiliary load balancing loss.
        
        Returns:
            Auxiliary loss tensor
        """
        if 'load_balance_loss' in self.aux_outputs:
            return self.aux_outputs['load_balance_loss']
        return torch.tensor(0.0)
    
    def get_expert_usage(self) -> Dict[int, float]:
        """
        Get usage statistics for each expert.
        
        Returns:
            Dictionary mapping expert_id to usage ratio
        """
        usage = {}
        for i, expert in enumerate(self.experts):
            usage[i] = expert.get_usage_ratio()
        return usage


class SparseMOELayer(nn.Module):
    """
    Sparse Mixture of Experts Layer.
    Memory-efficient implementation using token dispatch.
    
    Key differences from standard MOE:
    - Only computes expert outputs for assigned tokens
    - Uses capacity constraint to limit tokens per expert
    - More memory efficient for large models
    """
    
    def __init__(
        self,
        input_dim: int = 768,
        hidden_dim: int = 3072,
        output_dim: int = 768,
        num_experts: int = 8,
        top_k: int = 2,
        capacity_factor: float = 1.25,
        dropout: float = 0.1,
        use_aux_loss: bool = True,
        expert_type: str = 'feedforward'
    ):
        """
        Initialize sparse MOE layer.
        
        Args:
            input_dim: Input feature dimension
            hidden_dim: Hidden dimension
            output_dim: Output feature dimension
            num_experts: Number of experts
            top_k: Experts per token
            capacity_factor: Factor for expert capacity
            dropout: Dropout rate
            use_aux_loss: Whether to use auxiliary loss
            expert_type: Type of expert
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_experts = num_experts
        self.top_k = top_k
        self.capacity_factor = capacity_factor
        
        # Router with noisy top-k for better exploration
        self.router = NoisyTopKRouter(
            input_dim=input_dim,
            num_experts=num_experts,
            top_k=top_k,
            use_aux_loss=use_aux_loss
        )
        
        # Experts
        self.experts = nn.ModuleList([
            create_expert(
                expert_type=expert_type,
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                output_dim=output_dim,
                expert_id=i,
                dropout=dropout
            )
            for i in range(num_experts)
        ])
        
        self.output_norm = nn.LayerNorm(output_dim)
        self.aux_outputs: Dict[str, Any] = {}
    
    def _compute_capacity(self, num_tokens: int) -> int:
        """
        Compute capacity per expert.
        
        Args:
            num_tokens: Total number of tokens
            
        Returns:
            Capacity per expert
        """
        return int(self.capacity_factor * num_tokens * self.top_k / self.num_experts)
    
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        **kwargs
    ) -> torch.Tensor:
        """
        Forward pass through sparse MOE layer.
        
        Args:
            x: Input tensor [batch_size, seq_len, input_dim]
            mask: Optional attention mask
            
        Returns:
            Output tensor [batch_size, seq_len, output_dim]
        """
        batch_size, seq_len, _ = x.shape
        num_tokens = batch_size * seq_len
        capacity = self._compute_capacity(num_tokens)
        
        # Get routing
        routing_weights, expert_indices, aux_outputs = self.router(x)
        self.aux_outputs = aux_outputs
        
        # Flatten for dispatch
        x_flat = x.view(num_tokens, self.input_dim)  # [N, D]
        routing_flat = routing_weights.view(num_tokens, self.top_k)  # [N, K]
        indices_flat = expert_indices.view(num_tokens, self.top_k)  # [N, K]
        
        # Initialize output
        output_flat = torch.zeros(
            num_tokens, self.output_dim,
            device=x.device, dtype=x.dtype
        )
        
        # Dispatch to experts
        for expert_id, expert in enumerate(self.experts):
            # Find tokens for this expert (across all top-k positions)
            token_mask = (indices_flat == expert_id)  # [N, K]
            
            if not token_mask.any():
                continue
            
            # Get token indices and their weights
            token_indices = token_mask.any(dim=-1).nonzero(as_tuple=True)[0]
            
            # Apply capacity constraint
            if len(token_indices) > capacity:
                # Get routing weights for sorting
                expert_weights = (routing_flat * token_mask.float()).sum(dim=-1)
                top_weights, top_idx = torch.topk(
                    expert_weights[token_indices],
                    capacity
                )
                token_indices = token_indices[top_idx]
            
            # Get tokens and weights
            selected_tokens = x_flat[token_indices]  # [C, D]
            selected_weights = (routing_flat[token_indices] * token_mask[token_indices].float()).sum(dim=-1)
            
            # Process through expert
            expert_output = expert(selected_tokens.unsqueeze(0)).squeeze(0)  # [C, D]
            
            # Scatter weighted outputs
            weighted_output = expert_output * selected_weights.unsqueeze(-1)
            output_flat.index_add_(0, token_indices, weighted_output)
        
        # Reshape and normalize
        output = output_flat.view(batch_size, seq_len, self.output_dim)
        output = self.output_norm(output)
        
        return output
    
    def get_aux_loss(self) -> torch.Tensor:
        """Get auxiliary loss."""
        if 'load_balance_loss' in self.aux_outputs:
            return self.aux_outputs['load_balance_loss']
        return torch.tensor(0.0)


class HierarchicalMOE(nn.Module):
    """
    Hierarchical Mixture of Experts.
    Two-level routing: first to expert groups, then to individual experts.
    
    Architecture:
    1. First-level router selects expert group
    2. Second-level router selects expert within group
    3. Hierarchical combination of outputs
    
    Benefits:
    - Better scalability for large numbers of experts
    - Structured expert organization
    - Reduced routing complexity
    """
    
    def __init__(
        self,
        input_dim: int = 768,
        hidden_dim: int = 3072,
        output_dim: int = 768,
        num_expert_groups: int = 4,
        experts_per_group: int = 4,
        top_k_groups: int = 2,
        top_k_experts: int = 1,
        dropout: float = 0.1,
        expert_types: Optional[List[str]] = None
    ):
        """
        Initialize hierarchical MOE.
        
        Args:
            input_dim: Input feature dimension
            hidden_dim: Hidden dimension
            output_dim: Output feature dimension
            num_expert_groups: Number of expert groups
            experts_per_group: Experts in each group
            top_k_groups: Groups to select per token
            top_k_experts: Experts to select per group
            dropout: Dropout rate
            expert_types: List of expert types for each group
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_expert_groups = num_expert_groups
        self.experts_per_group = experts_per_group
        self.top_k_groups = top_k_groups
        self.top_k_experts = top_k_experts
        
        # Default expert types for groups
        if expert_types is None:
            expert_types = ['vision', 'text', 'multimodal', 'feedforward']
            expert_types = (expert_types * num_expert_groups)[:num_expert_groups]
        
        # First-level router (group selection)
        self.group_router = TopKRouter(
            input_dim=input_dim,
            num_experts=num_expert_groups,
            top_k=top_k_groups,
            use_aux_loss=True
        )
        
        # Second-level routers (expert selection within group)
        self.expert_routers = nn.ModuleList([
            TopKRouter(
                input_dim=input_dim,
                num_experts=experts_per_group,
                top_k=top_k_experts,
                use_aux_loss=True
            )
            for _ in range(num_expert_groups)
        ])
        
        # Expert groups
        self.expert_groups = nn.ModuleList([
            nn.ModuleList([
                create_expert(
                    expert_type=expert_types[g],
                    input_dim=input_dim,
                    hidden_dim=hidden_dim,
                    output_dim=output_dim,
                    expert_id=g * experts_per_group + e,
                    dropout=dropout
                )
                for e in range(experts_per_group)
            ])
            for g in range(num_expert_groups)
        ])
        
        # Output combination
        self.output_proj = nn.Linear(output_dim, output_dim)
        self.output_norm = nn.LayerNorm(output_dim)
        
        self.aux_outputs: Dict[str, Any] = {}
    
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        **kwargs
    ) -> torch.Tensor:
        """
        Forward pass through hierarchical MOE.
        
        Args:
            x: Input tensor [batch_size, seq_len, input_dim]
            mask: Optional attention mask
            
        Returns:
            Output tensor [batch_size, seq_len, output_dim]
        """
        batch_size, seq_len, _ = x.shape
        
        # First-level routing
        group_weights, group_indices, group_aux = self.group_router(x)
        
        # Initialize output
        output = torch.zeros(
            batch_size, seq_len, self.output_dim,
            device=x.device, dtype=x.dtype
        )
        
        total_aux_loss = group_aux.get('load_balance_loss', torch.tensor(0.0))
        
        # Process each selected group
        for k in range(self.top_k_groups):
            group_idx = group_indices[:, :, k]  # [B, S]
            group_weight = group_weights[:, :, k]  # [B, S]
            
            # Process each group
            for g in range(self.num_expert_groups):
                group_mask = (group_idx == g)  # [B, S]
                
                if not group_mask.any():
                    continue
                
                # Second-level routing within group
                expert_weights, expert_indices, expert_aux = self.expert_routers[g](x)
                total_aux_loss = total_aux_loss + expert_aux.get(
                    'load_balance_loss', torch.tensor(0.0)
                )
                
                # Process experts in group
                group_output = torch.zeros_like(output)
                
                for e_k in range(self.top_k_experts):
                    expert_idx = expert_indices[:, :, e_k]  # [B, S]
                    expert_weight = expert_weights[:, :, e_k]  # [B, S]
                    
                    for e in range(self.experts_per_group):
                        expert_mask = (expert_idx == e)  # [B, S]
                        
                        if not expert_mask.any():
                            continue
                        
                        expert = self.expert_groups[g][e]
                        expert_out = expert(x, mask=mask, **kwargs)
                        
                        # Weight by expert and group weights
                        combined_weight = expert_weight * group_weight
                        weighted_out = expert_out * combined_weight.unsqueeze(-1)
                        
                        # Apply expert mask
                        weighted_out = weighted_out * expert_mask.unsqueeze(-1).float()
                        group_output = group_output + weighted_out
                
                # Apply group mask
                group_output = group_output * group_mask.unsqueeze(-1).float()
                output = output + group_output
        
        # Store auxiliary outputs
        self.aux_outputs = {
            'load_balance_loss': total_aux_loss,
            'group_probs': group_aux.get('router_probs', None)
        }
        
        # Output processing
        output = self.output_proj(output)
        output = self.output_norm(output)
        
        return output
    
    def get_aux_loss(self) -> torch.Tensor:
        """Get auxiliary loss."""
        return self.aux_outputs.get('load_balance_loss', torch.tensor(0.0))


class VQAMOELayer(MOELayer):
    """
    VQA-specific MOE Layer.
    Combines different expert types optimized for Visual Question Answering.
    
    Expert composition:
    - Vision experts: For image understanding
    - Text experts: For question understanding (Vietnamese optimized)
    - Multimodal experts: For cross-modal reasoning
    - Specialized experts: Segmentation, detection, OCR, etc.
    """
    
    def __init__(
        self,
        input_dim: int = 768,
        hidden_dim: int = 3072,
        output_dim: int = 768,
        num_vision_experts: int = 2,
        num_text_experts: int = 2,
        num_multimodal_experts: int = 2,
        num_specialized_experts: int = 2,
        top_k: int = 2,
        dropout: float = 0.1,
        vietnamese_optimized: bool = True
    ):
        """
        Initialize VQA MOE layer.
        
        Args:
            input_dim: Input feature dimension
            hidden_dim: Hidden dimension
            output_dim: Output feature dimension
            num_vision_experts: Number of vision experts
            num_text_experts: Number of text experts
            num_multimodal_experts: Number of multimodal experts
            num_specialized_experts: Number of specialized experts
            top_k: Experts per token
            dropout: Dropout rate
            vietnamese_optimized: Whether to optimize for Vietnamese
        """
        # Calculate total experts
        total_experts = (
            num_vision_experts +
            num_text_experts +
            num_multimodal_experts +
            num_specialized_experts
        )
        
        # Initialize parent without creating experts
        nn.Module.__init__(self)
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_experts = total_experts
        self.top_k = top_k
        
        # Create router
        self.router = NoisyTopKRouter(
            input_dim=input_dim,
            num_experts=total_experts,
            top_k=top_k,
            use_aux_loss=True
        )
        
        # Create diverse experts
        self.experts = nn.ModuleList()
        expert_id = 0
        
        # Vision experts
        for _ in range(num_vision_experts):
            self.experts.append(
                VisionExpert(
                    input_dim=input_dim,
                    hidden_dim=hidden_dim,
                    output_dim=output_dim,
                    expert_id=expert_id,
                    dropout=dropout
                )
            )
            expert_id += 1
        
        # Text experts
        for _ in range(num_text_experts):
            self.experts.append(
                TextExpert(
                    input_dim=input_dim,
                    hidden_dim=hidden_dim,
                    output_dim=output_dim,
                    expert_id=expert_id,
                    dropout=dropout
                )
            )
            expert_id += 1
        
        # Multimodal experts
        for _ in range(num_multimodal_experts):
            self.experts.append(
                MultimodalExpert(
                    input_dim=input_dim,
                    hidden_dim=hidden_dim,
                    output_dim=output_dim,
                    expert_id=expert_id,
                    dropout=dropout
                )
            )
            expert_id += 1
        
        # Specialized experts
        from src.modeling.moe.specialized_experts import (
            SegmentationExpert,
            ObjectDetectionExpert,
            OCRExpert,
            SceneUnderstandingExpert
        )
        
        specialized_types = [
            SegmentationExpert,
            ObjectDetectionExpert,
            OCRExpert,
            SceneUnderstandingExpert
        ]
        
        for i in range(num_specialized_experts):
            expert_class = specialized_types[i % len(specialized_types)]
            expert_kwargs = {
                'input_dim': input_dim,
                'hidden_dim': hidden_dim,
                'output_dim': output_dim,
                'expert_id': expert_id,
                'dropout': dropout
            }
            
            # Add Vietnamese optimization for OCR
            if expert_class == OCRExpert:
                expert_kwargs['vietnamese_optimized'] = vietnamese_optimized
            
            self.experts.append(expert_class(**expert_kwargs))
            expert_id += 1
        
        self.output_norm = nn.LayerNorm(output_dim)
        self.aux_outputs: Dict[str, Any] = {}
