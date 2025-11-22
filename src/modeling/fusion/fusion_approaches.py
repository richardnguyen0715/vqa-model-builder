"""
Multimodal Fusion Approaches for VQA
Implements different strategies to combine vision and language features.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
from abc import ABC, abstractmethod


class BaseFusion(ABC, nn.Module):
    """
    Abstract base class for multimodal fusion modules.
    All fusion approaches should inherit from this class.
    """
    
    def __init__(self, vision_dim: int, text_dim: int, output_dim: int):
        """
        Args:
            vision_dim: Dimension of vision features
            text_dim: Dimension of text features
            output_dim: Dimension of fused output features
        """
        super().__init__()
        self.vision_dim = vision_dim
        self.text_dim = text_dim
        self.output_dim = output_dim
    
    @abstractmethod
    def forward(
        self,
        vision_features: torch.Tensor,
        text_features: torch.Tensor,
        vision_mask: Optional[torch.Tensor] = None,
        text_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Fuse vision and text features.
        
        Args:
            vision_features: [batch_size, num_vision_tokens, vision_dim]
            text_features: [batch_size, num_text_tokens, text_dim]
            vision_mask: [batch_size, num_vision_tokens] (optional)
            text_mask: [batch_size, num_text_tokens] (optional)
            
        Returns:
            fused_features: [batch_size, output_dim] or [batch_size, seq_len, output_dim]
        """
        pass
    
    @abstractmethod
    def get_output_dim(self) -> int:
        """Return the output feature dimension."""
        pass


class CrossAttentionFusion(BaseFusion):
    """
    Cross-Attention Fusion approach.
    Vision and text features attend to each other through cross-attention mechanism.
    
    Architecture:
    1. Vision attends to Text (Text-guided Vision)
    2. Text attends to Vision (Vision-guided Text)
    3. Concatenate or add the attended features
    4. Final fusion layer
    
    Reference: 
    - ViLBERT: Pretraining Task-Agnostic Visiolinguistic Representations (Lu et al., 2019)
    - LXMERT: Learning Cross-Modality Encoder Representations from Transformers (Tan & Bansal, 2019)
    """
    
    def __init__(
        self,
        vision_dim: int = 768,
        text_dim: int = 768,
        output_dim: int = 768,
        num_attention_heads: int = 8,
        num_layers: int = 4,
        intermediate_dim: int = 3072,
        dropout: float = 0.1,
        fusion_method: str = 'concat'
    ):
        """
        Args:
            vision_dim: Dimension of vision features
            text_dim: Dimension of text features
            output_dim: Dimension of fused output
            num_attention_heads: Number of attention heads
            num_layers: Number of cross-attention layers
            intermediate_dim: Dimension of feed-forward intermediate layer
            dropout: Dropout rate
            fusion_method: How to combine attended features ('concat', 'add', 'multiply')
        """
        super().__init__(vision_dim, text_dim, output_dim)
        
        self.num_attention_heads = num_attention_heads
        self.num_layers = num_layers
        self.fusion_method = fusion_method
        
        # Project vision and text to same dimension if needed
        if vision_dim != output_dim:
            self.vision_projection = nn.Linear(vision_dim, output_dim)
        else:
            self.vision_projection = nn.Identity()
            
        if text_dim != output_dim:
            self.text_projection = nn.Linear(text_dim, output_dim)
        else:
            self.text_projection = nn.Identity()
        
        # Cross-attention layers
        self.cross_attention_layers = nn.ModuleList([
            CrossAttentionBlock(
                dim=output_dim,
                num_heads=num_attention_heads,
                intermediate_dim=intermediate_dim,
                dropout=dropout
            )
            for _ in range(num_layers)
        ])
        
        # Final fusion layer
        if fusion_method == 'concat':
            fusion_input_dim = output_dim * 2
        else:  # add or multiply
            fusion_input_dim = output_dim
            
        self.fusion_layer = nn.Sequential(
            nn.Linear(fusion_input_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(output_dim, output_dim),
            nn.LayerNorm(output_dim)
        )
        
        # Global pooling for final output
        self.pooling = nn.AdaptiveAvgPool1d(1)
    
    def forward(
        self,
        vision_features: torch.Tensor,
        text_features: torch.Tensor,
        vision_mask: Optional[torch.Tensor] = None,
        text_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            vision_features: [batch_size, num_vision_tokens, vision_dim]
            text_features: [batch_size, num_text_tokens, text_dim]
            vision_mask: [batch_size, num_vision_tokens]
            text_mask: [batch_size, num_text_tokens]
            
        Returns:
            fused_features: [batch_size, output_dim]
        """
        # Project to common dimension
        vision_feat = self.vision_projection(vision_features)  # [B, V, D]
        text_feat = self.text_projection(text_features)        # [B, T, D]
        
        # Apply cross-attention layers
        for layer in self.cross_attention_layers:
            vision_feat, text_feat = layer(
                vision_feat, text_feat,
                vision_mask, text_mask
            )
        
        # Pool features
        vision_pooled = torch.mean(vision_feat, dim=1)  # [B, D]
        text_pooled = torch.mean(text_feat, dim=1)      # [B, D]
        
        # Fuse vision and text
        if self.fusion_method == 'concat':
            fused = torch.cat([vision_pooled, text_pooled], dim=-1)  # [B, 2D]
        elif self.fusion_method == 'add':
            fused = vision_pooled + text_pooled  # [B, D]
        elif self.fusion_method == 'multiply':
            fused = vision_pooled * text_pooled  # [B, D]
        else:
            raise ValueError(f"Unknown fusion method: {self.fusion_method}")
        
        # Final fusion
        output = self.fusion_layer(fused)  # [B, D]
        
        return output
    
    def get_output_dim(self) -> int:
        return self.output_dim


class CrossAttentionBlock(nn.Module):
    """
    Single cross-attention block with bidirectional attention.
    Vision attends to Text, Text attends to Vision.
    """
    
    def __init__(
        self,
        dim: int,
        num_heads: int,
        intermediate_dim: int,
        dropout: float = 0.1
    ):
        super().__init__()
        
        # Vision-to-Text attention (text queries vision)
        self.v2t_attention = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.v2t_norm1 = nn.LayerNorm(dim)
        self.v2t_norm2 = nn.LayerNorm(dim)
        self.v2t_ffn = nn.Sequential(
            nn.Linear(dim, intermediate_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(intermediate_dim, dim),
            nn.Dropout(dropout)
        )
        
        # Text-to-Vision attention (vision queries text)
        self.t2v_attention = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.t2v_norm1 = nn.LayerNorm(dim)
        self.t2v_norm2 = nn.LayerNorm(dim)
        self.t2v_ffn = nn.Sequential(
            nn.Linear(dim, intermediate_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(intermediate_dim, dim),
            nn.Dropout(dropout)
        )
    
    def forward(
        self,
        vision_features: torch.Tensor,
        text_features: torch.Tensor,
        vision_mask: Optional[torch.Tensor] = None,
        text_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            vision_features: [B, V, D]
            text_features: [B, T, D]
            vision_mask: [B, V]
            text_mask: [B, T]
            
        Returns:
            vision_output: [B, V, D]
            text_output: [B, T, D]
        """
        # Text attends to Vision (Vision-to-Text)
        text_attn_output, _ = self.v2t_attention(
            query=text_features,
            key=vision_features,
            value=vision_features,
            key_padding_mask=~vision_mask if vision_mask is not None else None
        )
        text_features = self.v2t_norm1(text_features + text_attn_output)
        text_features = self.v2t_norm2(text_features + self.v2t_ffn(text_features))
        
        # Vision attends to Text (Text-to-Vision)
        vision_attn_output, _ = self.t2v_attention(
            query=vision_features,
            key=text_features,
            value=text_features,
            key_padding_mask=~text_mask if text_mask is not None else None
        )
        vision_features = self.t2v_norm1(vision_features + vision_attn_output)
        vision_features = self.t2v_norm2(vision_features + self.t2v_ffn(vision_features))
        
        return vision_features, text_features


class QFormerFusion(BaseFusion):
    """
    Q-Former (Querying Transformer) Fusion approach.
    Uses learnable query tokens to extract relevant information from both modalities.
    
    Architecture:
    1. Learnable query tokens
    2. Self-attention among queries
    3. Cross-attention: Queries attend to vision features
    4. Cross-attention: Queries attend to text features
    5. Output: Query representations as fused features
    
    Reference:
    - BLIP-2: Bootstrapping Language-Image Pre-training (Li et al., 2023)
    - Inspired by Perceiver and Flamingo architectures
    """
    
    def __init__(
        self,
        vision_dim: int = 768,
        text_dim: int = 768,
        output_dim: int = 768,
        num_query_tokens: int = 32,
        num_attention_heads: int = 8,
        num_layers: int = 6,
        intermediate_dim: int = 3072,
        dropout: float = 0.1
    ):
        """
        Args:
            vision_dim: Dimension of vision features
            text_dim: Dimension of text features
            output_dim: Dimension of query tokens (and output)
            num_query_tokens: Number of learnable query tokens
            num_attention_heads: Number of attention heads
            num_layers: Number of Q-Former layers
            intermediate_dim: Dimension of feed-forward intermediate layer
            dropout: Dropout rate
        """
        super().__init__(vision_dim, text_dim, output_dim)
        
        self.num_query_tokens = num_query_tokens
        self.num_layers = num_layers
        
        # Learnable query tokens
        self.query_tokens = nn.Parameter(
            torch.randn(1, num_query_tokens, output_dim)
        )
        nn.init.trunc_normal_(self.query_tokens, std=0.02)
        
        # Project vision and text to output dimension
        self.vision_projection = nn.Linear(vision_dim, output_dim)
        self.text_projection = nn.Linear(text_dim, output_dim)
        
        # Q-Former layers
        self.qformer_layers = nn.ModuleList([
            QFormerLayer(
                dim=output_dim,
                num_heads=num_attention_heads,
                intermediate_dim=intermediate_dim,
                dropout=dropout
            )
            for _ in range(num_layers)
        ])
        
        # Final output projection
        self.output_projection = nn.Sequential(
            nn.LayerNorm(output_dim),
            nn.Linear(output_dim, output_dim)
        )
    
    def forward(
        self,
        vision_features: torch.Tensor,
        text_features: torch.Tensor,
        vision_mask: Optional[torch.Tensor] = None,
        text_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            vision_features: [batch_size, num_vision_tokens, vision_dim]
            text_features: [batch_size, num_text_tokens, text_dim]
            vision_mask: [batch_size, num_vision_tokens]
            text_mask: [batch_size, num_text_tokens]
            
        Returns:
            query_output: [batch_size, output_dim] (pooled from queries)
        """
        batch_size = vision_features.size(0)
        
        # Project vision and text features
        vision_feat = self.vision_projection(vision_features)  # [B, V, D]
        text_feat = self.text_projection(text_features)        # [B, T, D]
        
        # Initialize query tokens for this batch
        queries = self.query_tokens.expand(batch_size, -1, -1)  # [B, Q, D]
        
        # Apply Q-Former layers
        for layer in self.qformer_layers:
            queries = layer(
                queries,
                vision_feat,
                text_feat,
                vision_mask,
                text_mask
            )
        
        # Project output
        queries = self.output_projection(queries)  # [B, Q, D]
        
        # Pool queries to get final representation
        output = torch.mean(queries, dim=1)  # [B, D]
        
        return output
    
    def get_output_dim(self) -> int:
        return self.output_dim


class QFormerLayer(nn.Module):
    """
    Single Q-Former layer.
    Queries perform self-attention and cross-attention to vision and text.
    """
    
    def __init__(
        self,
        dim: int,
        num_heads: int,
        intermediate_dim: int,
        dropout: float = 0.1
    ):
        super().__init__()
        
        # Self-attention among queries
        self.self_attention = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.self_norm1 = nn.LayerNorm(dim)
        self.self_norm2 = nn.LayerNorm(dim)
        self.self_ffn = nn.Sequential(
            nn.Linear(dim, intermediate_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(intermediate_dim, dim),
            nn.Dropout(dropout)
        )
        
        # Cross-attention to vision
        self.vision_cross_attention = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.vision_norm1 = nn.LayerNorm(dim)
        self.vision_norm2 = nn.LayerNorm(dim)
        self.vision_ffn = nn.Sequential(
            nn.Linear(dim, intermediate_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(intermediate_dim, dim),
            nn.Dropout(dropout)
        )
        
        # Cross-attention to text
        self.text_cross_attention = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.text_norm1 = nn.LayerNorm(dim)
        self.text_norm2 = nn.LayerNorm(dim)
        self.text_ffn = nn.Sequential(
            nn.Linear(dim, intermediate_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(intermediate_dim, dim),
            nn.Dropout(dropout)
        )
    
    def forward(
        self,
        queries: torch.Tensor,
        vision_features: torch.Tensor,
        text_features: torch.Tensor,
        vision_mask: Optional[torch.Tensor] = None,
        text_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            queries: [B, Q, D]
            vision_features: [B, V, D]
            text_features: [B, T, D]
            vision_mask: [B, V]
            text_mask: [B, T]
            
        Returns:
            queries: [B, Q, D]
        """
        # Self-attention among queries
        self_attn_output, _ = self.self_attention(queries, queries, queries)
        queries = self.self_norm1(queries + self_attn_output)
        queries = self.self_norm2(queries + self.self_ffn(queries))
        
        # Cross-attention to vision
        vision_attn_output, _ = self.vision_cross_attention(
            query=queries,
            key=vision_features,
            value=vision_features,
            key_padding_mask=~vision_mask if vision_mask is not None else None
        )
        queries = self.vision_norm1(queries + vision_attn_output)
        queries = self.vision_norm2(queries + self.vision_ffn(queries))
        
        # Cross-attention to text
        text_attn_output, _ = self.text_cross_attention(
            query=queries,
            key=text_features,
            value=text_features,
            key_padding_mask=~text_mask if text_mask is not None else None
        )
        queries = self.text_norm1(queries + text_attn_output)
        queries = self.text_norm2(queries + self.text_ffn(queries))
        
        return queries


class SingleStreamFusion(BaseFusion):
    """
    Single-Stream Fusion (ViLT-style) approach.
    Concatenates vision and text tokens and processes them jointly in a single transformer.
    
    Architecture:
    1. Project vision and text to same dimension
    2. Add modality-specific embeddings
    3. Concatenate vision and text tokens
    4. Process through unified transformer
    5. Extract fused representation
    
    Reference:
    - ViLT: Vision-and-Language Transformer Without Convolution or Region Supervision (Kim et al., 2021)
    - Simple yet effective approach
    """
    
    def __init__(
        self,
        vision_dim: int = 768,
        text_dim: int = 768,
        output_dim: int = 768,
        num_attention_heads: int = 12,
        num_layers: int = 6,
        intermediate_dim: int = 3072,
        dropout: float = 0.1,
        max_vision_tokens: int = 200,
        max_text_tokens: int = 128
    ):
        """
        Args:
            vision_dim: Dimension of vision features
            text_dim: Dimension of text features
            output_dim: Dimension of unified representation
            num_attention_heads: Number of attention heads
            num_layers: Number of transformer layers
            intermediate_dim: Dimension of feed-forward intermediate layer
            dropout: Dropout rate
            max_vision_tokens: Maximum number of vision tokens
            max_text_tokens: Maximum number of text tokens
        """
        super().__init__(vision_dim, text_dim, output_dim)
        
        self.num_layers = num_layers
        self.max_vision_tokens = max_vision_tokens
        self.max_text_tokens = max_text_tokens
        
        # Project vision and text to output dimension
        self.vision_projection = nn.Linear(vision_dim, output_dim)
        self.text_projection = nn.Linear(text_dim, output_dim)
        
        # Modality type embeddings (to distinguish vision vs text)
        self.modality_embeddings = nn.Embedding(2, output_dim)  # 0: vision, 1: text
        
        # Position embeddings
        max_total_tokens = max_vision_tokens + max_text_tokens
        self.position_embeddings = nn.Parameter(
            torch.randn(1, max_total_tokens, output_dim)
        )
        nn.init.trunc_normal_(self.position_embeddings, std=0.02)
        
        # Unified transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=output_dim,
            nhead=num_attention_heads,
            dim_feedforward=intermediate_dim,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )
        
        # Layer norm
        self.norm = nn.LayerNorm(output_dim)
        
        # CLS token for global representation
        self.cls_token = nn.Parameter(torch.randn(1, 1, output_dim))
        nn.init.trunc_normal_(self.cls_token, std=0.02)
    
    def forward(
        self,
        vision_features: torch.Tensor,
        text_features: torch.Tensor,
        vision_mask: Optional[torch.Tensor] = None,
        text_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            vision_features: [batch_size, num_vision_tokens, vision_dim]
            text_features: [batch_size, num_text_tokens, text_dim]
            vision_mask: [batch_size, num_vision_tokens]
            text_mask: [batch_size, num_text_tokens]
            
        Returns:
            fused_features: [batch_size, output_dim]
        """
        batch_size = vision_features.size(0)
        num_vision_tokens = vision_features.size(1)
        num_text_tokens = text_features.size(1)
        
        # Project to common dimension
        vision_feat = self.vision_projection(vision_features)  # [B, V, D]
        text_feat = self.text_projection(text_features)        # [B, T, D]
        
        # Add modality type embeddings
        vision_type = self.modality_embeddings(
            torch.zeros(batch_size, num_vision_tokens, dtype=torch.long, device=vision_feat.device)
        )
        text_type = self.modality_embeddings(
            torch.ones(batch_size, num_text_tokens, dtype=torch.long, device=text_feat.device)
        )
        
        vision_feat = vision_feat + vision_type
        text_feat = text_feat + text_type
        
        # Add CLS token at the beginning
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)  # [B, 1, D]
        
        # Concatenate: [CLS] + Vision + Text
        combined_features = torch.cat([cls_tokens, vision_feat, text_feat], dim=1)  # [B, 1+V+T, D]
        total_tokens = combined_features.size(1)
        
        # Add position embeddings
        combined_features = combined_features + self.position_embeddings[:, :total_tokens, :]
        
        # Create attention mask
        if vision_mask is not None or text_mask is not None:
            # CLS token is always attended
            cls_mask = torch.ones(batch_size, 1, dtype=torch.bool, device=combined_features.device)
            
            if vision_mask is None:
                vision_mask = torch.ones(batch_size, num_vision_tokens, dtype=torch.bool, device=combined_features.device)
            if text_mask is None:
                text_mask = torch.ones(batch_size, num_text_tokens, dtype=torch.bool, device=combined_features.device)
            
            combined_mask = torch.cat([cls_mask, vision_mask, text_mask], dim=1)  # [B, 1+V+T]
            # Convert to attention mask (True = attend, False = ignore)
            # Transformer expects: True = ignore, False = attend (opposite!)
            src_key_padding_mask = ~combined_mask
        else:
            src_key_padding_mask = None
        
        # Pass through unified transformer
        encoded_features = self.transformer(
            combined_features,
            src_key_padding_mask=src_key_padding_mask
        )
        
        # Normalize
        encoded_features = self.norm(encoded_features)
        
        # Extract CLS token as fused representation
        cls_output = encoded_features[:, 0, :]  # [B, D]
        
        return cls_output
    
    def get_output_dim(self) -> int:
        return self.output_dim


# Factory function for easy model creation
def create_fusion_model(
    fusion_type: str,
    **kwargs
) -> BaseFusion:
    """
    Factory function to create fusion models.
    
    Args:
        fusion_type: Type of fusion ('cross_attention', 'qformer', 'single_stream')
        **kwargs: Additional arguments for the specific fusion model
        
    Returns:
        Fusion model
        
    Examples:
        >>> # Cross-Attention Fusion
        >>> fusion = create_fusion_model(
        ...     'cross_attention',
        ...     vision_dim=768,
        ...     text_dim=768,
        ...     output_dim=768
        ... )
        
        >>> # Q-Former Fusion
        >>> fusion = create_fusion_model(
        ...     'qformer',
        ...     vision_dim=768,
        ...     text_dim=768,
        ...     num_query_tokens=32
        ... )
        
        >>> # Single-Stream Fusion (ViLT)
        >>> fusion = create_fusion_model(
        ...     'single_stream',
        ...     vision_dim=768,
        ...     text_dim=768
        ... )
    """
    fusion_registry = {
        'cross_attention': CrossAttentionFusion,
        'qformer': QFormerFusion,
        'q_former': QFormerFusion,  # Alias
        'single_stream': SingleStreamFusion,
        'vilt': SingleStreamFusion  # Alias
    }
    
    if fusion_type not in fusion_registry:
        available = ', '.join(fusion_registry.keys())
        raise ValueError(
            f"Unknown fusion type: {fusion_type}. "
            f"Available types: {available}"
        )
    
    return fusion_registry[fusion_type](**kwargs)
