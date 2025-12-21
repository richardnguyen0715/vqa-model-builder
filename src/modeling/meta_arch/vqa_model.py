"""
VQA Meta Architecture
Complete Vietnamese VQA model architecture combining all components.
"""

from typing import Optional, List, Dict, Any, Tuple, Union
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .vqa_config import (
    VQAModelConfig,
    VisualEncoderConfig,
    TextEncoderConfig,
    FusionConfig,
    MOEConfig,
    KnowledgeConfig,
    AnswerHeadConfig
)


@dataclass
class VQAOutput:
    """
    Container for VQA model output.
    
    Attributes:
        logits: Answer prediction logits [batch, num_answers]
        loss: Total loss value
        predictions: Predicted answer indices [batch]
        visual_features: Visual features
        text_features: Text features
        fused_features: Fused features
        knowledge_features: Knowledge features (if used)
        moe_info: MOE routing information (if used)
        auxiliary_outputs: Additional outputs
    """
    logits: torch.Tensor
    loss: Optional[torch.Tensor] = None
    predictions: Optional[torch.Tensor] = None
    visual_features: Optional[torch.Tensor] = None
    text_features: Optional[torch.Tensor] = None
    fused_features: Optional[torch.Tensor] = None
    knowledge_features: Optional[torch.Tensor] = None
    moe_info: Optional[Dict[str, Any]] = None
    auxiliary_outputs: Optional[Dict[str, Any]] = None


class VisualEncoder(nn.Module):
    """
    Visual encoder module.
    Supports various vision backbones.
    """
    
    def __init__(self, config: VisualEncoderConfig):
        """
        Initialize visual encoder.
        
        Args:
            config: Visual encoder configuration
        """
        super().__init__()
        self.config = config
        
        self._init_backbone()
        
        # Optional projection
        if hasattr(self, 'backbone_dim') and self.backbone_dim != config.output_dim:
            self.projection = nn.Linear(self.backbone_dim, config.output_dim)
        else:
            self.projection = None
    
    def _init_backbone(self):
        """Initialize visual backbone."""
        try:
            from transformers import AutoImageProcessor
            
            self.processor = AutoImageProcessor.from_pretrained(self.config.model_name)
            
            # Use vision-only model for CLIP to avoid requiring text input
            if 'clip' in self.config.model_name.lower():
                from transformers import CLIPVisionModel
                self.backbone = CLIPVisionModel.from_pretrained(self.config.model_name)
            else:
                from transformers import AutoModel
                self.backbone = AutoModel.from_pretrained(self.config.model_name)
            
            # Get backbone output dimension
            if hasattr(self.backbone.config, 'hidden_size'):
                self.backbone_dim = self.backbone.config.hidden_size
            else:
                self.backbone_dim = self.config.output_dim
            
            if self.config.freeze_backbone:
                for param in self.backbone.parameters():
                    param.requires_grad = False
                    
        except ImportError:
            raise ImportError("transformers required for visual encoder")
    
    def forward(
        self,
        pixel_values: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            pixel_values: Input images [batch, 3, H, W]
            
        Returns:
            Tuple of (pooled_features, spatial_features)
        """
        outputs = self.backbone(pixel_values=pixel_values)
        
        # Get features
        if hasattr(outputs, 'last_hidden_state'):
            spatial_features = outputs.last_hidden_state
            pooled_features = spatial_features[:, 0, :]  # CLS token
        else:
            pooled_features = outputs.image_embeds if hasattr(outputs, 'image_embeds') else outputs[0]
            spatial_features = pooled_features.unsqueeze(1)
        
        # Project if needed
        if self.projection is not None:
            pooled_features = self.projection(pooled_features)
            spatial_features = self.projection(spatial_features)
        
        return pooled_features, spatial_features


class TextEncoder(nn.Module):
    """
    Text encoder module.
    Optimized for Vietnamese with PhoBERT support.
    """
    
    def __init__(self, config: TextEncoderConfig):
        """
        Initialize text encoder.
        
        Args:
            config: Text encoder configuration
        """
        super().__init__()
        self.config = config
        
        self._init_encoder()
        
        # Optional projection
        if hasattr(self, 'encoder_dim') and self.encoder_dim != config.output_dim:
            self.projection = nn.Linear(self.encoder_dim, config.output_dim)
        else:
            self.projection = None
    
    def _init_encoder(self):
        """Initialize text encoder."""
        try:
            from transformers import AutoModel, AutoTokenizer
            
            self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
            self.encoder = AutoModel.from_pretrained(self.config.model_name)
            
            # Get encoder output dimension
            if hasattr(self.encoder.config, 'hidden_size'):
                self.encoder_dim = self.encoder.config.hidden_size
            else:
                self.encoder_dim = self.config.output_dim
            
            if self.config.freeze_encoder:
                for param in self.encoder.parameters():
                    param.requires_grad = False
                    
        except ImportError:
            raise ImportError("transformers required for text encoder")
    
    def _pool_features(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Pool hidden states.
        
        Args:
            hidden_states: Hidden states [batch, seq, hidden]
            attention_mask: Attention mask [batch, seq]
            
        Returns:
            Pooled features [batch, hidden]
        """
        if self.config.pooling_strategy == 'cls':
            return hidden_states[:, 0, :]
        elif self.config.pooling_strategy == 'max':
            mask = attention_mask.unsqueeze(-1).expand(hidden_states.size())
            hidden_states = hidden_states.masked_fill(mask == 0, -1e9)
            return hidden_states.max(dim=1)[0]
        else:  # mean
            mask = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
            sum_hidden = (hidden_states * mask).sum(dim=1)
            sum_mask = mask.sum(dim=1).clamp(min=1e-9)
            return sum_hidden / sum_mask
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            input_ids: Token IDs [batch, seq]
            attention_mask: Attention mask [batch, seq]
            
        Returns:
            Tuple of (pooled_features, sequence_features)
        """
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        sequence_features = outputs.last_hidden_state
        pooled_features = self._pool_features(sequence_features, attention_mask)
        
        # Project if needed
        if self.projection is not None:
            pooled_features = self.projection(pooled_features)
            sequence_features = self.projection(sequence_features)
        
        return pooled_features, sequence_features


class CrossModalAttention(nn.Module):
    """
    Cross-modal attention for vision-language fusion.
    """
    
    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 8,
        dropout: float = 0.1
    ):
        """
        Initialize cross-modal attention.
        
        Args:
            embed_dim: Embedding dimension
            num_heads: Number of attention heads
            dropout: Dropout rate
        """
        super().__init__()
        
        self.self_attn = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.cross_attn = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )
        
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 4, embed_dim),
            nn.Dropout(dropout)
        )
        
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.norm3 = nn.LayerNorm(embed_dim)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        query: torch.Tensor,
        key_value: torch.Tensor,
        query_mask: Optional[torch.Tensor] = None,
        kv_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            query: Query features [batch, q_seq, embed_dim]
            key_value: Key/value features [batch, kv_seq, embed_dim]
            query_mask: Query attention mask
            kv_mask: Key/value attention mask
            
        Returns:
            Output features [batch, q_seq, embed_dim]
        """
        # Self attention
        x = query
        attn_out, _ = self.self_attn(x, x, x, key_padding_mask=query_mask)
        x = self.norm1(x + self.dropout(attn_out))
        
        # Cross attention
        cross_out, _ = self.cross_attn(x, key_value, key_value, key_padding_mask=kv_mask)
        x = self.norm2(x + self.dropout(cross_out))
        
        # FFN
        ffn_out = self.ffn(x)
        x = self.norm3(x + ffn_out)
        
        return x


class MultimodalFusion(nn.Module):
    """
    Multimodal fusion module.
    """
    
    def __init__(self, config: FusionConfig):
        """
        Initialize fusion module.
        
        Args:
            config: Fusion configuration
        """
        super().__init__()
        self.config = config
        
        if config.fusion_type == 'cross_attention':
            self.fusion_layers = nn.ModuleList([
                CrossModalAttention(
                    config.hidden_dim,
                    config.num_heads,
                    config.dropout
                )
                for _ in range(config.num_layers)
            ])
            self.output_proj = nn.Linear(config.hidden_dim, config.output_dim)
            
        elif config.fusion_type == 'concat':
            self.fusion_layer = nn.Sequential(
                nn.Linear(config.hidden_dim * 2, config.hidden_dim),
                nn.ReLU(),
                nn.Dropout(config.dropout),
                nn.Linear(config.hidden_dim, config.output_dim)
            )
            
        elif config.fusion_type == 'bilinear':
            self.bilinear = nn.Bilinear(
                config.hidden_dim, config.hidden_dim, config.output_dim
            )
            
        else:
            self.fusion_layer = nn.Linear(config.hidden_dim, config.output_dim)
        
        if config.use_layer_norm:
            self.layer_norm = nn.LayerNorm(config.output_dim)
        else:
            self.layer_norm = None
    
    def forward(
        self,
        visual_features: torch.Tensor,
        text_features: torch.Tensor,
        visual_mask: Optional[torch.Tensor] = None,
        text_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            visual_features: Visual features [batch, v_seq, hidden]
            text_features: Text features [batch, t_seq, hidden]
            visual_mask: Visual attention mask
            text_mask: Text attention mask
            
        Returns:
            Fused features [batch, output_dim]
        """
        if self.config.fusion_type == 'cross_attention':
            # Cross-modal attention fusion
            for layer in self.fusion_layers:
                text_features = layer(text_features, visual_features, text_mask, visual_mask)
            
            # Pool to single vector
            fused = text_features[:, 0, :]  # Use CLS position
            fused = self.output_proj(fused)
            
        elif self.config.fusion_type == 'concat':
            # Pool features first
            if visual_features.dim() == 3:
                visual_pooled = visual_features[:, 0, :]
            else:
                visual_pooled = visual_features
            
            if text_features.dim() == 3:
                text_pooled = text_features[:, 0, :]
            else:
                text_pooled = text_features
            
            combined = torch.cat([visual_pooled, text_pooled], dim=-1)
            fused = self.fusion_layer(combined)
            
        elif self.config.fusion_type == 'bilinear':
            if visual_features.dim() == 3:
                visual_pooled = visual_features[:, 0, :]
            else:
                visual_pooled = visual_features
            
            if text_features.dim() == 3:
                text_pooled = text_features[:, 0, :]
            else:
                text_pooled = text_features
            
            fused = self.bilinear(visual_pooled, text_pooled)
            
        else:  # add or default
            if visual_features.dim() == 3:
                visual_pooled = visual_features[:, 0, :]
            else:
                visual_pooled = visual_features
            
            if text_features.dim() == 3:
                text_pooled = text_features[:, 0, :]
            else:
                text_pooled = text_features
            
            fused = self.fusion_layer(visual_pooled + text_pooled)
        
        if self.layer_norm is not None:
            fused = self.layer_norm(fused)
        
        return fused


class AnswerHead(nn.Module):
    """
    Answer prediction head.
    """
    
    def __init__(self, config: AnswerHeadConfig, input_dim: int):
        """
        Initialize answer head.
        
        Args:
            config: Answer head configuration
            input_dim: Input dimension
        """
        super().__init__()
        self.config = config
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in config.hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(config.dropout)
            ])
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, config.num_answers))
        
        self.classifier = nn.Sequential(*layers)
    
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            features: Input features [batch, input_dim]
            
        Returns:
            Logits [batch, num_answers]
        """
        return self.classifier(features)


class VietnameseVQAModel(nn.Module):
    """
    Complete Vietnamese VQA model.
    Combines visual encoder, text encoder, fusion, MOE, knowledge, and answer head.
    """
    
    def __init__(self, config: VQAModelConfig):
        """
        Initialize VQA model.
        
        Args:
            config: Model configuration
        """
        super().__init__()
        self.config = config
        
        # Visual encoder
        self.visual_encoder = VisualEncoder(config.visual_encoder)
        
        # Text encoder
        self.text_encoder = TextEncoder(config.text_encoder)
        
        # Fusion
        self.fusion = MultimodalFusion(config.fusion)
        
        # MOE (optional)
        if config.moe.use_moe:
            self._init_moe(config.moe)
        else:
            self.moe_layer = None
        
        # Knowledge augmentation (optional)
        if config.knowledge.use_knowledge:
            self._init_knowledge(config.knowledge)
        else:
            self.knowledge_module = None
        
        # Answer head
        self.answer_head = AnswerHead(
            config.answer_head,
            input_dim=config.fusion.output_dim
        )
        
        # Dropout
        self.dropout = nn.Dropout(config.dropout)
    
    def _init_moe(self, config: MOEConfig):
        """Initialize MOE layer."""
        try:
            from src.modeling.moe import VQAMOELayer
            
            # Distribute experts evenly across types
            # VQAMOELayer expects num_vision_experts, num_text_experts, etc.
            num_per_type = max(1, config.num_experts // 4)
            remainder = config.num_experts % 4
            
            self.moe_layer = VQAMOELayer(
                input_dim=self.config.fusion.output_dim,
                hidden_dim=config.hidden_dim,
                output_dim=self.config.fusion.output_dim,
                num_vision_experts=num_per_type + (1 if remainder > 0 else 0),
                num_text_experts=num_per_type + (1 if remainder > 1 else 0),
                num_multimodal_experts=num_per_type + (1 if remainder > 2 else 0),
                num_specialized_experts=num_per_type,
                top_k=config.top_k,
                dropout=self.config.dropout
            )
        except ImportError:
            import warnings
            warnings.warn("MOE module not available, disabling MOE")
            self.moe_layer = None
    
    def _init_knowledge(self, config: KnowledgeConfig):
        """Initialize knowledge augmentation."""
        try:
            from src.modeling.knowledge_base import (
                RAGModule,
                TextKnowledgeEncoder,
                DenseRetriever,
                FAISSVectorStore
            )
            
            # Create components
            encoder = TextKnowledgeEncoder(
                model_name=self.config.text_encoder.model_name,
                embed_dim=self.config.text_encoder.output_dim
            )
            
            # Vector store (needs to be loaded with knowledge)
            self.knowledge_encoder = encoder
            self.knowledge_module = None  # Will be set when knowledge base is loaded
            
        except ImportError:
            import warnings
            warnings.warn("Knowledge base module not available")
            self.knowledge_module = None
            self.knowledge_encoder = None
    
    def set_knowledge_base(self, retriever, context_encoder=None):
        """
        Set knowledge base retriever.
        
        Args:
            retriever: Knowledge retriever
            context_encoder: Optional context encoder
        """
        try:
            from src.modeling.knowledge_base import RAGModule
            
            encoder = context_encoder or self.knowledge_encoder
            
            self.knowledge_module = RAGModule(
                retriever=retriever,
                context_encoder=encoder,
                embed_dim=self.config.fusion.output_dim,
                num_contexts=self.config.knowledge.num_contexts
            )
        except ImportError:
            pass
    
    def encode_visual(
        self,
        pixel_values: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode visual input.
        
        Args:
            pixel_values: Input images
            
        Returns:
            Tuple of (pooled, spatial) features
        """
        return self.visual_encoder(pixel_values)
    
    def encode_text(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode text input.
        
        Args:
            input_ids: Token IDs
            attention_mask: Attention mask
            
        Returns:
            Tuple of (pooled, sequence) features
        """
        return self.text_encoder(input_ids, attention_mask)
    
    def forward(
        self,
        pixel_values: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        questions: Optional[List[str]] = None,
        labels: Optional[torch.Tensor] = None,
        return_features: bool = False
    ) -> VQAOutput:
        """
        Forward pass.
        
        Args:
            pixel_values: Input images [batch, 3, H, W]
            input_ids: Token IDs [batch, seq]
            attention_mask: Attention mask [batch, seq]
            questions: Question strings for knowledge retrieval
            labels: Target answer indices for loss computation
            return_features: Whether to return intermediate features
            
        Returns:
            VQAOutput
        """
        # Encode visual
        visual_pooled, visual_spatial = self.encode_visual(pixel_values)
        
        # Encode text
        text_pooled, text_sequence = self.encode_text(input_ids, attention_mask)
        
        # Fusion
        fused = self.fusion(
            visual_spatial,
            text_sequence,
            text_mask=~attention_mask.bool()  # Invert mask
        )
        
        # MOE
        moe_info = None
        if self.moe_layer is not None:
            # MOE layer expects 3D input [batch, seq, hidden]
            # Fused features are 2D [batch, hidden], so add seq dimension
            if fused.dim() == 2:
                fused_3d = fused.unsqueeze(1)  # [batch, 1, hidden]
                moe_output = self.moe_layer(fused_3d)
                if isinstance(moe_output, tuple):
                    fused, moe_info = moe_output
                    fused = fused.squeeze(1)  # Back to [batch, hidden]
                else:
                    fused = moe_output.squeeze(1)
            else:
                moe_output = self.moe_layer(fused)
                if isinstance(moe_output, tuple):
                    fused, moe_info = moe_output
                else:
                    fused = moe_output
        
        # Knowledge augmentation
        knowledge_features = None
        if self.knowledge_module is not None and questions is not None:
            # Retrieve and incorporate knowledge
            for i, question in enumerate(questions):
                rag_output = self.knowledge_module(question, fused[i])
                if knowledge_features is None:
                    knowledge_features = [rag_output.output]
                else:
                    knowledge_features.append(rag_output.output)
            
            if knowledge_features:
                knowledge_features = torch.stack(knowledge_features, dim=0)
                # Combine with fused features
                fused = fused + 0.5 * knowledge_features
        
        # Dropout
        fused = self.dropout(fused)
        
        # Answer prediction
        logits = self.answer_head(fused)
        
        # Compute loss
        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits, labels)
        
        # Get predictions
        predictions = logits.argmax(dim=-1)
        
        return VQAOutput(
            logits=logits,
            loss=loss,
            predictions=predictions,
            visual_features=visual_pooled if return_features else None,
            text_features=text_pooled if return_features else None,
            fused_features=fused if return_features else None,
            knowledge_features=knowledge_features if return_features else None,
            moe_info=moe_info
        )


def create_vqa_model(
    config: Optional[VQAModelConfig] = None,
    **kwargs
) -> VietnameseVQAModel:
    """
    Factory function to create VQA model.
    
    Args:
        config: Model configuration
        **kwargs: Override configuration values
        
    Returns:
        VQA model instance
    """
    if config is None:
        from .vqa_config import get_default_vietnamese_vqa_config
        config = get_default_vietnamese_vqa_config()
    
    # Apply overrides
    if kwargs:
        config_dict = config.to_dict()
        for key, value in kwargs.items():
            if key in config_dict:
                config_dict[key] = value
        config = VQAModelConfig.from_dict(config_dict)
    
    return VietnameseVQAModel(config)


__all__ = [
    'VQAOutput',
    'VisualEncoder',
    'TextEncoder',
    'CrossModalAttention',
    'MultimodalFusion',
    'AnswerHead',
    'VietnameseVQAModel',
    'create_vqa_model'
]
