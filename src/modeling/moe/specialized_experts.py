"""
Specialized Experts Module
Implements domain-specific experts for VQA tasks.
Includes SAM, Object Detection, OCR, and other specialized experts.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any, List, Tuple

from src.modeling.moe.base_expert import BaseExpert


class SegmentationExpert(BaseExpert):
    """
    Segmentation Expert using SAM (Segment Anything Model) inspired architecture.
    Specializes in understanding object boundaries and spatial relationships.
    
    Features:
    - Mask prediction head
    - Boundary-aware feature extraction
    - Spatial relationship modeling
    
    Reference: Kirillov et al. "Segment Anything" (2023)
    """
    
    def __init__(
        self,
        input_dim: int = 768,
        hidden_dim: int = 3072,
        output_dim: int = 768,
        expert_id: Optional[int] = None,
        dropout: float = 0.1,
        num_mask_tokens: int = 4,
        use_pretrained_sam: bool = False,
        sam_model_type: str = 'vit_b'
    ):
        """
        Initialize segmentation expert.
        
        Args:
            input_dim: Input feature dimension
            hidden_dim: Hidden layer dimension
            output_dim: Output feature dimension
            expert_id: Unique identifier
            dropout: Dropout rate
            num_mask_tokens: Number of learnable mask tokens
            use_pretrained_sam: Whether to load pretrained SAM weights
            sam_model_type: SAM model variant (vit_b, vit_l, vit_h)
        """
        super().__init__(input_dim, hidden_dim, output_dim, expert_id, dropout)
        
        self.num_mask_tokens = num_mask_tokens
        self.use_pretrained_sam = use_pretrained_sam
        self.sam_model_type = sam_model_type
        
        # Learnable mask tokens
        self.mask_tokens = nn.Parameter(
            torch.randn(1, num_mask_tokens, hidden_dim) * 0.02
        )
        
        # Input projection
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        # Mask prediction transformer
        self.mask_transformer = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(
                d_model=hidden_dim,
                nhead=8,
                dim_feedforward=hidden_dim * 2,
                dropout=dropout,
                activation='gelu',
                batch_first=True
            ),
            num_layers=2
        )
        
        # Boundary feature extractor
        self.boundary_conv = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.GELU()
        )
        
        # Spatial relationship MLP
        self.spatial_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Output projection
        self.output_proj = nn.Linear(hidden_dim, output_dim)
        self.output_norm = nn.LayerNorm(output_dim)
        
        # Optional: Load SAM encoder weights
        if use_pretrained_sam:
            self._load_sam_weights()
    
    def _load_sam_weights(self):
        """
        Load pretrained SAM weights.
        Requires segment_anything package.
        """
        try:
            from segment_anything import sam_model_registry
            # Note: Actual weight loading would require model checkpoint
            # This is a placeholder for the integration pattern
            pass
        except ImportError:
            import warnings
            warnings.warn(
                "segment_anything not installed. "
                "Install with: pip install segment-anything"
            )
    
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        image_features: Optional[torch.Tensor] = None,
        **kwargs
    ) -> torch.Tensor:
        """
        Forward pass through segmentation expert.
        
        Args:
            x: Input tensor [batch_size, seq_len, input_dim]
            mask: Optional attention mask
            image_features: Optional raw image features for boundary detection
            
        Returns:
            Output tensor [batch_size, seq_len, output_dim]
        """
        batch_size, seq_len, _ = x.shape
        
        # Input projection
        h = self.input_proj(x)  # [B, S, H]
        
        # Expand mask tokens
        mask_tokens = self.mask_tokens.expand(batch_size, -1, -1)  # [B, M, H]
        
        # Mask transformer: mask tokens attend to image features
        mask_features = self.mask_transformer(
            mask_tokens,  # query
            h  # memory
        )  # [B, M, H]
        
        # Boundary feature extraction
        h_t = h.transpose(1, 2)  # [B, H, S]
        boundary_features = self.boundary_conv(h_t)  # [B, H, S]
        boundary_features = boundary_features.transpose(1, 2)  # [B, S, H]
        
        # Combine with mask features through spatial relationship
        # Pool mask features
        mask_pooled = mask_features.mean(dim=1, keepdim=True)  # [B, 1, H]
        mask_pooled = mask_pooled.expand(-1, seq_len, -1)  # [B, S, H]
        
        # Spatial relationship
        spatial_input = torch.cat([boundary_features, mask_pooled], dim=-1)
        spatial_features = self.spatial_mlp(spatial_input)
        
        # Combine features
        h = h + boundary_features + spatial_features
        
        # Output projection
        output = self.output_proj(h)
        output = self.output_norm(output)
        
        return output


class ObjectDetectionExpert(BaseExpert):
    """
    Object Detection Expert.
    Specializes in identifying and localizing objects in images.
    
    Features:
    - Region proposal mechanism
    - Object classification head
    - Bounding box regression
    - Multi-scale feature processing
    
    Reference: Inspired by DETR (Carion et al., 2020)
    """
    
    def __init__(
        self,
        input_dim: int = 768,
        hidden_dim: int = 3072,
        output_dim: int = 768,
        expert_id: Optional[int] = None,
        dropout: float = 0.1,
        num_queries: int = 100,
        num_decoder_layers: int = 3
    ):
        """
        Initialize object detection expert.
        
        Args:
            input_dim: Input feature dimension
            hidden_dim: Hidden layer dimension
            output_dim: Output feature dimension
            expert_id: Unique identifier
            dropout: Dropout rate
            num_queries: Number of object queries
            num_decoder_layers: Number of transformer decoder layers
        """
        super().__init__(input_dim, hidden_dim, output_dim, expert_id, dropout)
        
        self.num_queries = num_queries
        
        # Object queries (learnable)
        self.object_queries = nn.Parameter(
            torch.randn(1, num_queries, hidden_dim) * 0.02
        )
        
        # Input projection
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        # Object detection decoder
        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(
                d_model=hidden_dim,
                nhead=8,
                dim_feedforward=hidden_dim * 2,
                dropout=dropout,
                activation='gelu',
                batch_first=True
            ),
            num_layers=num_decoder_layers
        )
        
        # Object feature aggregation
        self.object_aggregation = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # Cross-attention for query-feature interaction
        self.query_feature_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=8,
            dropout=dropout,
            batch_first=True
        )
        
        # Output projection
        self.output_proj = nn.Linear(hidden_dim, output_dim)
        self.output_norm = nn.LayerNorm(output_dim)
    
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        return_object_features: bool = False,
        **kwargs
    ) -> torch.Tensor:
        """
        Forward pass through object detection expert.
        
        Args:
            x: Input tensor [batch_size, seq_len, input_dim]
            mask: Optional attention mask
            return_object_features: Whether to return object query features
            
        Returns:
            Output tensor [batch_size, seq_len, output_dim]
        """
        batch_size, seq_len, _ = x.shape
        
        # Input projection
        h = self.input_proj(x)  # [B, S, H]
        
        # Expand object queries
        queries = self.object_queries.expand(batch_size, -1, -1)  # [B, Q, H]
        
        # Object detection decoder
        object_features = self.decoder(
            queries,  # query
            h  # memory
        )  # [B, Q, H]
        
        # Aggregate object features
        object_features = self.object_aggregation(object_features)  # [B, Q, H]
        
        # Cross-attention: input features attend to object features
        h_enhanced, _ = self.query_feature_attention(
            h,  # query
            object_features,  # key
            object_features  # value
        )  # [B, S, H]
        
        # Residual connection
        h = h + h_enhanced
        
        # Output projection
        output = self.output_proj(h)
        output = self.output_norm(output)
        
        if return_object_features:
            return output, object_features
        
        return output


class OCRExpert(BaseExpert):
    """
    OCR (Optical Character Recognition) Expert.
    Specializes in reading and understanding text within images.
    Optimized for Vietnamese text recognition.
    
    Features:
    - Text region detection
    - Character/word recognition
    - Vietnamese diacritics handling
    - Reading order inference
    """
    
    def __init__(
        self,
        input_dim: int = 768,
        hidden_dim: int = 3072,
        output_dim: int = 768,
        expert_id: Optional[int] = None,
        dropout: float = 0.1,
        max_text_length: int = 128,
        num_attention_heads: int = 8,
        vietnamese_optimized: bool = True
    ):
        """
        Initialize OCR expert.
        
        Args:
            input_dim: Input feature dimension
            hidden_dim: Hidden layer dimension
            output_dim: Output feature dimension
            expert_id: Unique identifier
            dropout: Dropout rate
            max_text_length: Maximum text sequence length
            num_attention_heads: Number of attention heads
            vietnamese_optimized: Whether to optimize for Vietnamese
        """
        super().__init__(input_dim, hidden_dim, output_dim, expert_id, dropout)
        
        self.max_text_length = max_text_length
        self.vietnamese_optimized = vietnamese_optimized
        
        # Text region queries
        self.text_queries = nn.Parameter(
            torch.randn(1, max_text_length, hidden_dim) * 0.02
        )
        
        # Input projection
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        # Text detection transformer
        self.text_detector = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(
                d_model=hidden_dim,
                nhead=num_attention_heads,
                dim_feedforward=hidden_dim * 2,
                dropout=dropout,
                activation='gelu',
                batch_first=True
            ),
            num_layers=2
        )
        
        # Reading order attention
        self.reading_order_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_attention_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Vietnamese-specific processing
        if vietnamese_optimized:
            # Additional layer for handling Vietnamese diacritics
            self.diacritic_processor = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, hidden_dim)
            )
        
        # Feature aggregation
        self.aggregator = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
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
        Forward pass through OCR expert.
        
        Args:
            x: Input tensor [batch_size, seq_len, input_dim]
            mask: Optional attention mask
            
        Returns:
            Output tensor [batch_size, seq_len, output_dim]
        """
        batch_size, seq_len, _ = x.shape
        
        # Input projection
        h = self.input_proj(x)  # [B, S, H]
        
        # Expand text queries
        text_queries = self.text_queries.expand(batch_size, -1, -1)  # [B, T, H]
        
        # Text detection
        text_features = self.text_detector(
            text_queries,
            h
        )  # [B, T, H]
        
        # Vietnamese-specific processing
        if self.vietnamese_optimized:
            text_features = text_features + self.diacritic_processor(text_features)
        
        # Reading order attention
        ordered_features, _ = self.reading_order_attention(
            text_features,
            text_features,
            text_features
        )
        
        # Aggregate text features back to input sequence
        # Cross-attention: input attends to text features
        h_text, _ = self.reading_order_attention(
            h,
            ordered_features,
            ordered_features
        )
        
        # Combine
        h = h + self.aggregator(h_text)
        
        # Output projection
        output = self.output_proj(h)
        output = self.output_norm(output)
        
        return output


class SceneUnderstandingExpert(BaseExpert):
    """
    Scene Understanding Expert.
    Specializes in understanding overall scene context and semantics.
    
    Features:
    - Global scene representation
    - Scene classification
    - Contextual relationship modeling
    - Attribute detection
    """
    
    def __init__(
        self,
        input_dim: int = 768,
        hidden_dim: int = 3072,
        output_dim: int = 768,
        expert_id: Optional[int] = None,
        dropout: float = 0.1,
        num_scene_tokens: int = 16,
        use_global_pooling: bool = True
    ):
        """
        Initialize scene understanding expert.
        
        Args:
            input_dim: Input feature dimension
            hidden_dim: Hidden layer dimension
            output_dim: Output feature dimension
            expert_id: Unique identifier
            dropout: Dropout rate
            num_scene_tokens: Number of learnable scene tokens
            use_global_pooling: Whether to use global average pooling
        """
        super().__init__(input_dim, hidden_dim, output_dim, expert_id, dropout)
        
        self.num_scene_tokens = num_scene_tokens
        self.use_global_pooling = use_global_pooling
        
        # Scene tokens
        self.scene_tokens = nn.Parameter(
            torch.randn(1, num_scene_tokens, hidden_dim) * 0.02
        )
        
        # Input projection
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        # Scene understanding layers
        self.scene_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=8,
                dim_feedforward=hidden_dim * 2,
                dropout=dropout,
                activation='gelu',
                batch_first=True
            ),
            num_layers=2
        )
        
        # Global scene aggregation
        if use_global_pooling:
            self.global_pool = nn.Sequential(
                nn.AdaptiveAvgPool1d(1),
                nn.Flatten(start_dim=1)
            )
            self.global_proj = nn.Linear(hidden_dim, hidden_dim)
        
        # Context distribution
        self.context_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=8,
            dropout=dropout,
            batch_first=True
        )
        
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
        Forward pass through scene understanding expert.
        
        Args:
            x: Input tensor [batch_size, seq_len, input_dim]
            mask: Optional attention mask
            
        Returns:
            Output tensor [batch_size, seq_len, output_dim]
        """
        batch_size, seq_len, _ = x.shape
        
        # Input projection
        h = self.input_proj(x)  # [B, S, H]
        
        # Expand scene tokens
        scene_tokens = self.scene_tokens.expand(batch_size, -1, -1)  # [B, T, H]
        
        # Concatenate scene tokens with input
        combined = torch.cat([scene_tokens, h], dim=1)  # [B, T+S, H]
        
        # Scene encoding
        encoded = self.scene_encoder(combined)  # [B, T+S, H]
        
        # Split back
        scene_features = encoded[:, :self.num_scene_tokens, :]  # [B, T, H]
        h_encoded = encoded[:, self.num_scene_tokens:, :]  # [B, S, H]
        
        # Global scene representation
        if self.use_global_pooling:
            scene_t = scene_features.transpose(1, 2)  # [B, H, T]
            global_scene = self.global_pool(scene_t)  # [B, H]
            global_scene = self.global_proj(global_scene)  # [B, H]
            global_scene = global_scene.unsqueeze(1).expand(-1, seq_len, -1)  # [B, S, H]
        else:
            global_scene = scene_features.mean(dim=1, keepdim=True).expand(-1, seq_len, -1)
        
        # Distribute scene context to features
        h_context, _ = self.context_attention(
            h_encoded,
            scene_features,
            scene_features
        )
        
        # Combine all
        h = h_encoded + h_context + global_scene
        
        # Output projection
        output = self.output_proj(h)
        output = self.output_norm(output)
        
        return output


class SpatialReasoningExpert(BaseExpert):
    """
    Spatial Reasoning Expert.
    Specializes in understanding spatial relationships between objects.
    
    Features:
    - Relative position encoding
    - Distance computation
    - Spatial relationship classification
    - Graph-based reasoning
    """
    
    def __init__(
        self,
        input_dim: int = 768,
        hidden_dim: int = 3072,
        output_dim: int = 768,
        expert_id: Optional[int] = None,
        dropout: float = 0.1,
        num_spatial_relations: int = 16,
        use_graph_attention: bool = True
    ):
        """
        Initialize spatial reasoning expert.
        
        Args:
            input_dim: Input feature dimension
            hidden_dim: Hidden layer dimension
            output_dim: Output feature dimension
            expert_id: Unique identifier
            dropout: Dropout rate
            num_spatial_relations: Number of spatial relation types
            use_graph_attention: Whether to use graph attention
        """
        super().__init__(input_dim, hidden_dim, output_dim, expert_id, dropout)
        
        self.num_spatial_relations = num_spatial_relations
        self.use_graph_attention = use_graph_attention
        
        # Input projection
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        # Spatial relation embeddings
        self.relation_embeddings = nn.Embedding(
            num_spatial_relations,
            hidden_dim
        )
        
        # Pairwise feature computation
        self.pairwise_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Relation prediction
        self.relation_predictor = nn.Linear(hidden_dim, num_spatial_relations)
        
        # Graph attention (if enabled)
        if use_graph_attention:
            self.graph_attention = nn.MultiheadAttention(
                embed_dim=hidden_dim,
                num_heads=8,
                dropout=dropout,
                batch_first=True
            )
        
        # Spatial aggregation
        self.spatial_aggregator = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Output projection
        self.output_proj = nn.Linear(hidden_dim, output_dim)
        self.output_norm = nn.LayerNorm(output_dim)
    
    def compute_pairwise_features(
        self,
        h: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute pairwise features between all positions.
        
        Args:
            h: Input features [batch_size, seq_len, hidden_dim]
            
        Returns:
            Tuple of pairwise features and relation logits
        """
        batch_size, seq_len, hidden_dim = h.shape
        
        # Expand for pairwise computation
        h_i = h.unsqueeze(2).expand(-1, -1, seq_len, -1)  # [B, S, S, H]
        h_j = h.unsqueeze(1).expand(-1, seq_len, -1, -1)  # [B, S, S, H]
        
        # Concatenate pairs
        pairs = torch.cat([h_i, h_j], dim=-1)  # [B, S, S, 2H]
        
        # Compute pairwise features
        pairwise = self.pairwise_mlp(pairs)  # [B, S, S, H]
        
        # Predict relations
        relation_logits = self.relation_predictor(pairwise)  # [B, S, S, R]
        
        return pairwise, relation_logits
    
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        spatial_positions: Optional[torch.Tensor] = None,
        **kwargs
    ) -> torch.Tensor:
        """
        Forward pass through spatial reasoning expert.
        
        Args:
            x: Input tensor [batch_size, seq_len, input_dim]
            mask: Optional attention mask
            spatial_positions: Optional position information
            
        Returns:
            Output tensor [batch_size, seq_len, output_dim]
        """
        batch_size, seq_len, _ = x.shape
        
        # Input projection
        h = self.input_proj(x)  # [B, S, H]
        
        # Add spatial positions if provided
        if spatial_positions is not None:
            h = h + spatial_positions
        
        # Compute pairwise features and relations
        pairwise_features, relation_logits = self.compute_pairwise_features(h)
        
        # Get relation weights
        relation_weights = F.softmax(relation_logits, dim=-1)  # [B, S, S, R]
        
        # Weighted relation embeddings
        # [B, S, S, R] @ [R, H] -> [B, S, S, H]
        relation_features = torch.einsum(
            'bssr,rh->bssh',
            relation_weights,
            self.relation_embeddings.weight
        )
        
        # Combine pairwise and relation features
        spatial_context = pairwise_features + relation_features  # [B, S, S, H]
        
        # Aggregate spatial context for each position
        spatial_context = spatial_context.mean(dim=2)  # [B, S, H]
        
        # Graph attention (if enabled)
        if self.use_graph_attention:
            h_graph, _ = self.graph_attention(h, h, h)
            h = h + h_graph
        
        # Combine with spatial context
        combined = torch.cat([h, spatial_context], dim=-1)  # [B, S, 2H]
        h = self.spatial_aggregator(combined)  # [B, S, H]
        
        # Output projection
        output = self.output_proj(h)
        output = self.output_norm(output)
        
        return output


class CountingExpert(BaseExpert):
    """
    Counting Expert.
    Specializes in counting objects in images.
    
    Features:
    - Object counting mechanism
    - Density estimation
    - Attention-based counting
    """
    
    def __init__(
        self,
        input_dim: int = 768,
        hidden_dim: int = 3072,
        output_dim: int = 768,
        expert_id: Optional[int] = None,
        dropout: float = 0.1,
        max_count: int = 20
    ):
        """
        Initialize counting expert.
        
        Args:
            input_dim: Input feature dimension
            hidden_dim: Hidden layer dimension
            output_dim: Output feature dimension
            expert_id: Unique identifier
            dropout: Dropout rate
            max_count: Maximum count value
        """
        super().__init__(input_dim, hidden_dim, output_dim, expert_id, dropout)
        
        self.max_count = max_count
        
        # Input projection
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        # Count queries
        self.count_queries = nn.Parameter(
            torch.randn(1, max_count + 1, hidden_dim) * 0.02
        )
        
        # Counting transformer
        self.count_transformer = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(
                d_model=hidden_dim,
                nhead=8,
                dim_feedforward=hidden_dim * 2,
                dropout=dropout,
                activation='gelu',
                batch_first=True
            ),
            num_layers=2
        )
        
        # Density estimation
        self.density_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
        # Feature aggregation
        self.aggregator = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
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
        Forward pass through counting expert.
        
        Args:
            x: Input tensor [batch_size, seq_len, input_dim]
            mask: Optional attention mask
            
        Returns:
            Output tensor [batch_size, seq_len, output_dim]
        """
        batch_size, seq_len, _ = x.shape
        
        # Input projection
        h = self.input_proj(x)  # [B, S, H]
        
        # Expand count queries
        count_queries = self.count_queries.expand(batch_size, -1, -1)  # [B, C, H]
        
        # Count transformer
        count_features = self.count_transformer(
            count_queries,
            h
        )  # [B, C, H]
        
        # Density estimation for each position
        density = self.density_head(h)  # [B, S, 1]
        
        # Weight features by density
        h_weighted = h * density
        
        # Aggregate count features
        count_agg = count_features.mean(dim=1, keepdim=True)  # [B, 1, H]
        count_agg = count_agg.expand(-1, seq_len, -1)  # [B, S, H]
        
        # Combine
        h = h + h_weighted + self.aggregator(count_agg)
        
        # Output projection
        output = self.output_proj(h)
        output = self.output_norm(output)
        
        return output
