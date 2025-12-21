"""
Expert Types Module
Implements various expert architectures for different modalities.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any

from src.modeling.moe.base_expert import BaseExpert


class FeedForwardExpert(BaseExpert):
    """
    Standard Feed-Forward Expert.
    Two-layer MLP with GELU activation.
    
    Architecture: Linear -> GELU -> Dropout -> Linear -> Dropout
    """
    
    def __init__(
        self,
        input_dim: int = 768,
        hidden_dim: int = 3072,
        output_dim: int = 768,
        expert_id: Optional[int] = None,
        dropout: float = 0.1,
        activation: str = 'gelu'
    ):
        """
        Initialize feed-forward expert.
        
        Args:
            input_dim: Input feature dimension
            hidden_dim: Hidden layer dimension (typically 4x input_dim)
            output_dim: Output feature dimension
            expert_id: Unique identifier
            dropout: Dropout rate
            activation: Activation function (gelu, relu, silu)
        """
        super().__init__(input_dim, hidden_dim, output_dim, expert_id, dropout)
        
        # Select activation function
        activations = {
            'gelu': nn.GELU(),
            'relu': nn.ReLU(),
            'silu': nn.SiLU(),
            'tanh': nn.Tanh()
        }
        self.activation = activations.get(activation, nn.GELU())
        
        # Feed-forward layers
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(output_dim)
    
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        **kwargs
    ) -> torch.Tensor:
        """
        Forward pass through feed-forward expert.
        
        Args:
            x: Input tensor [batch_size, seq_len, input_dim]
            mask: Optional mask (not used in FF expert)
            
        Returns:
            Output tensor [batch_size, seq_len, output_dim]
        """
        # Store residual if dimensions match
        residual = x if x.size(-1) == self.output_dim else None
        
        # Feed-forward
        h = self.fc1(x)
        h = self.activation(h)
        h = self.dropout(h)
        h = self.fc2(h)
        h = self.dropout(h)
        
        # Residual connection
        if residual is not None:
            h = h + residual
        
        # Layer normalization
        h = self.layer_norm(h)
        
        return h


class VisionExpert(BaseExpert):
    """
    Vision-specialized Expert.
    Designed to process visual features with spatial awareness.
    
    Features:
    - Spatial attention mechanism
    - Multi-scale feature processing
    - Position-aware transformations
    """
    
    def __init__(
        self,
        input_dim: int = 768,
        hidden_dim: int = 3072,
        output_dim: int = 768,
        expert_id: Optional[int] = None,
        dropout: float = 0.1,
        num_heads: int = 8,
        use_spatial_attention: bool = True
    ):
        """
        Initialize vision expert.
        
        Args:
            input_dim: Input feature dimension
            hidden_dim: Hidden layer dimension
            output_dim: Output feature dimension
            expert_id: Unique identifier
            dropout: Dropout rate
            num_heads: Number of attention heads
            use_spatial_attention: Whether to use spatial attention
        """
        super().__init__(input_dim, hidden_dim, output_dim, expert_id, dropout)
        
        self.num_heads = num_heads
        self.use_spatial_attention = use_spatial_attention
        
        # Input projection
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        # Spatial attention
        if use_spatial_attention:
            self.spatial_attention = nn.MultiheadAttention(
                embed_dim=hidden_dim,
                num_heads=num_heads,
                dropout=dropout,
                batch_first=True
            )
            self.spatial_norm = nn.LayerNorm(hidden_dim)
        
        # Feature transformation
        self.transform = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Dropout(dropout)
        )
        
        # Output projection
        self.output_proj = nn.Linear(hidden_dim, output_dim)
        self.output_norm = nn.LayerNorm(output_dim)
    
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        spatial_positions: Optional[torch.Tensor] = None,
        **kwargs
    ) -> torch.Tensor:
        """
        Forward pass through vision expert.
        
        Args:
            x: Input tensor [batch_size, num_patches, input_dim]
            mask: Optional attention mask
            spatial_positions: Optional spatial position embeddings
            
        Returns:
            Output tensor [batch_size, num_patches, output_dim]
        """
        # Input projection
        h = self.input_proj(x)
        
        # Add spatial positions if provided
        if spatial_positions is not None:
            h = h + spatial_positions
        
        # Spatial attention
        if self.use_spatial_attention:
            residual = h
            h_attn, _ = self.spatial_attention(h, h, h, key_padding_mask=mask)
            h = self.spatial_norm(residual + h_attn)
        
        # Feature transformation
        residual = h
        h = self.transform(h)
        h = residual + h
        
        # Output projection
        output = self.output_proj(h)
        output = self.output_norm(output)
        
        return output


class TextExpert(BaseExpert):
    """
    Text-specialized Expert.
    Designed to process textual features with linguistic awareness.
    
    Features:
    - Self-attention for contextual understanding
    - Positional encoding support
    - Designed for Vietnamese language processing
    """
    
    def __init__(
        self,
        input_dim: int = 768,
        hidden_dim: int = 3072,
        output_dim: int = 768,
        expert_id: Optional[int] = None,
        dropout: float = 0.1,
        num_heads: int = 8,
        use_self_attention: bool = True,
        max_seq_length: int = 512
    ):
        """
        Initialize text expert.
        
        Args:
            input_dim: Input feature dimension
            hidden_dim: Hidden layer dimension
            output_dim: Output feature dimension
            expert_id: Unique identifier
            dropout: Dropout rate
            num_heads: Number of attention heads
            use_self_attention: Whether to use self-attention
            max_seq_length: Maximum sequence length
        """
        super().__init__(input_dim, hidden_dim, output_dim, expert_id, dropout)
        
        self.num_heads = num_heads
        self.use_self_attention = use_self_attention
        self.max_seq_length = max_seq_length
        
        # Input projection
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        # Self-attention
        if use_self_attention:
            self.self_attention = nn.MultiheadAttention(
                embed_dim=hidden_dim,
                num_heads=num_heads,
                dropout=dropout,
                batch_first=True
            )
            self.attention_norm = nn.LayerNorm(hidden_dim)
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Dropout(dropout)
        )
        self.ffn_norm = nn.LayerNorm(hidden_dim)
        
        # Output projection
        self.output_proj = nn.Linear(hidden_dim, output_dim)
        self.output_norm = nn.LayerNorm(output_dim)
    
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        **kwargs
    ) -> torch.Tensor:
        """
        Forward pass through text expert.
        
        Args:
            x: Input tensor [batch_size, seq_len, input_dim]
            mask: Optional attention mask [batch_size, seq_len]
            
        Returns:
            Output tensor [batch_size, seq_len, output_dim]
        """
        # Input projection
        h = self.input_proj(x)
        
        # Self-attention
        if self.use_self_attention:
            residual = h
            # Convert mask to attention mask format
            attn_mask = None
            if mask is not None:
                attn_mask = ~mask.bool()  # Invert for attention
            
            h_attn, _ = self.self_attention(
                h, h, h,
                key_padding_mask=attn_mask
            )
            h = self.attention_norm(residual + h_attn)
        
        # Feed-forward
        residual = h
        h = self.ffn(h)
        h = self.ffn_norm(residual + h)
        
        # Output projection
        output = self.output_proj(h)
        output = self.output_norm(output)
        
        return output


class MultimodalExpert(BaseExpert):
    """
    Multimodal Expert.
    Processes combined vision and text features with cross-modal attention.
    
    Features:
    - Cross-attention between modalities
    - Modality-aware gating
    - Fusion-oriented processing
    """
    
    def __init__(
        self,
        input_dim: int = 768,
        hidden_dim: int = 3072,
        output_dim: int = 768,
        expert_id: Optional[int] = None,
        dropout: float = 0.1,
        num_heads: int = 8,
        use_cross_attention: bool = True,
        use_modality_gate: bool = True
    ):
        """
        Initialize multimodal expert.
        
        Args:
            input_dim: Input feature dimension
            hidden_dim: Hidden layer dimension
            output_dim: Output feature dimension
            expert_id: Unique identifier
            dropout: Dropout rate
            num_heads: Number of attention heads
            use_cross_attention: Whether to use cross-modal attention
            use_modality_gate: Whether to use modality-aware gating
        """
        super().__init__(input_dim, hidden_dim, output_dim, expert_id, dropout)
        
        self.num_heads = num_heads
        self.use_cross_attention = use_cross_attention
        self.use_modality_gate = use_modality_gate
        
        # Input projection
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        # Cross-modal attention
        if use_cross_attention:
            self.cross_attention = nn.MultiheadAttention(
                embed_dim=hidden_dim,
                num_heads=num_heads,
                dropout=dropout,
                batch_first=True
            )
            self.cross_norm = nn.LayerNorm(hidden_dim)
        
        # Modality gate
        if use_modality_gate:
            self.modality_gate = nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.Sigmoid()
            )
        
        # Feature transformation
        self.transform = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Dropout(dropout)
        )
        self.transform_norm = nn.LayerNorm(hidden_dim)
        
        # Output projection
        self.output_proj = nn.Linear(hidden_dim, output_dim)
        self.output_norm = nn.LayerNorm(output_dim)
    
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        context: Optional[torch.Tensor] = None,
        context_mask: Optional[torch.Tensor] = None,
        **kwargs
    ) -> torch.Tensor:
        """
        Forward pass through multimodal expert.
        
        Args:
            x: Input tensor [batch_size, seq_len, input_dim]
            mask: Optional mask for input
            context: Optional context features from other modality
            context_mask: Optional mask for context
            
        Returns:
            Output tensor [batch_size, seq_len, output_dim]
        """
        # Input projection
        h = self.input_proj(x)
        
        # Cross-modal attention
        if self.use_cross_attention and context is not None:
            context_h = self.input_proj(context)
            residual = h
            
            # Convert mask to attention format
            attn_mask = None
            if context_mask is not None:
                attn_mask = ~context_mask.bool()
            
            h_cross, _ = self.cross_attention(
                h, context_h, context_h,
                key_padding_mask=attn_mask
            )
            
            # Apply modality gate
            if self.use_modality_gate:
                gate_input = torch.cat([h, h_cross], dim=-1)
                gate = self.modality_gate(gate_input)
                h_cross = gate * h_cross
            
            h = self.cross_norm(residual + h_cross)
        
        # Feature transformation
        residual = h
        h = self.transform(h)
        h = self.transform_norm(residual + h)
        
        # Output projection
        output = self.output_proj(h)
        output = self.output_norm(output)
        
        return output


class GatedLinearExpert(BaseExpert):
    """
    Gated Linear Unit Expert.
    Uses gated linear units for more expressive transformations.
    
    Reference: Dauphin et al. "Language Modeling with Gated Convolutional Networks"
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
        Initialize GLU expert.
        
        Args:
            input_dim: Input feature dimension
            hidden_dim: Hidden layer dimension
            output_dim: Output feature dimension
            expert_id: Unique identifier
            dropout: Dropout rate
        """
        super().__init__(input_dim, hidden_dim, output_dim, expert_id, dropout)
        
        # GLU layers (output 2x hidden for gating)
        self.fc1 = nn.Linear(input_dim, hidden_dim * 2)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(output_dim)
    
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        **kwargs
    ) -> torch.Tensor:
        """
        Forward pass through GLU expert.
        
        Args:
            x: Input tensor
            mask: Optional mask
            
        Returns:
            Output tensor
        """
        residual = x if x.size(-1) == self.output_dim else None
        
        # GLU: split into value and gate
        h = self.fc1(x)
        h, gate = h.chunk(2, dim=-1)
        h = h * F.sigmoid(gate)  # Gated linear unit
        h = self.dropout(h)
        
        # Output
        h = self.fc2(h)
        h = self.dropout(h)
        
        if residual is not None:
            h = h + residual
        
        h = self.layer_norm(h)
        
        return h


def create_expert(
    expert_type: str,
    input_dim: int,
    hidden_dim: int,
    output_dim: int,
    expert_id: Optional[int] = None,
    **kwargs
) -> BaseExpert:
    """
    Factory function to create expert by type.
    
    Args:
        expert_type: Type of expert
        input_dim: Input dimension
        hidden_dim: Hidden dimension
        output_dim: Output dimension
        expert_id: Expert identifier
        **kwargs: Additional expert-specific arguments
        
    Returns:
        Expert instance
    """
    experts = {
        'feedforward': FeedForwardExpert,
        'vision': VisionExpert,
        'text': TextExpert,
        'multimodal': MultimodalExpert,
        'glu': GatedLinearExpert
    }
    
    if expert_type not in experts:
        raise ValueError(f"Unknown expert type: {expert_type}. Available: {list(experts.keys())}")
    
    return experts[expert_type](
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        output_dim=output_dim,
        expert_id=expert_id,
        **kwargs
    )
