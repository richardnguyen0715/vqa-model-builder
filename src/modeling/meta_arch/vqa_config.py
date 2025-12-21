"""
VQA Meta Architecture Configuration
Configuration dataclasses for Vietnamese VQA models.
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from enum import Enum


class BackboneType(Enum):
    """Enumeration of visual backbone types."""
    RESNET = 'resnet'
    VIT = 'vit'
    SWIN = 'swin'
    CLIP = 'clip'
    DINO = 'dino'


class TextEncoderType(Enum):
    """Enumeration of text encoder types."""
    PHOBERT = 'phobert'
    BERT = 'bert'
    ROBERTA = 'roberta'
    BARTPHO = 'bartpho'
    CLIP_TEXT = 'clip_text'


class FusionType(Enum):
    """Enumeration of fusion types."""
    CONCAT = 'concat'
    BILINEAR = 'bilinear'
    ATTENTION = 'attention'
    CROSS_ATTENTION = 'cross_attention'
    MCAN = 'mcan'
    MUTAN = 'mutan'


@dataclass
class VisualEncoderConfig:
    """
    Configuration for visual encoder.
    
    Attributes:
        backbone_type: Type of visual backbone
        model_name: Pretrained model name
        pretrained: Whether to use pretrained weights
        freeze_backbone: Whether to freeze backbone
        output_dim: Output feature dimension
        use_spatial_features: Whether to use spatial features
        num_spatial_tokens: Number of spatial tokens
    """
    backbone_type: str = 'vit'
    model_name: str = 'openai/clip-vit-base-patch32'
    pretrained: bool = True
    freeze_backbone: bool = False
    output_dim: int = 768
    use_spatial_features: bool = True
    num_spatial_tokens: int = 196


@dataclass
class TextEncoderConfig:
    """
    Configuration for text encoder.
    
    Attributes:
        encoder_type: Type of text encoder
        model_name: Pretrained model name
        pretrained: Whether to use pretrained weights
        freeze_encoder: Whether to freeze encoder
        output_dim: Output feature dimension
        max_length: Maximum sequence length
        pooling_strategy: Pooling strategy
    """
    encoder_type: str = 'phobert'
    model_name: str = 'vinai/phobert-base'
    pretrained: bool = True
    freeze_encoder: bool = False
    output_dim: int = 768
    max_length: int = 128
    pooling_strategy: str = 'cls'


@dataclass
class FusionConfig:
    """
    Configuration for multimodal fusion.
    
    Attributes:
        fusion_type: Type of fusion
        hidden_dim: Hidden dimension
        output_dim: Output dimension
        num_heads: Number of attention heads
        num_layers: Number of fusion layers
        dropout: Dropout rate
        use_layer_norm: Whether to use layer normalization
    """
    fusion_type: str = 'cross_attention'
    hidden_dim: int = 512
    output_dim: int = 512
    num_heads: int = 8
    num_layers: int = 2
    dropout: float = 0.1
    use_layer_norm: bool = True


@dataclass
class MOEConfig:
    """
    Configuration for Mixture of Experts.
    
    Attributes:
        use_moe: Whether to use MOE
        num_experts: Number of experts
        top_k: Number of experts to select
        router_type: Router type
        expert_type: Expert type
        hidden_dim: Expert hidden dimension
        load_balance_weight: Load balancing loss weight
    """
    use_moe: bool = False
    num_experts: int = 8
    top_k: int = 2
    router_type: str = 'top_k'
    expert_type: str = 'feedforward'
    hidden_dim: int = 2048
    load_balance_weight: float = 0.01


@dataclass
class KnowledgeConfig:
    """
    Configuration for knowledge augmentation.
    
    Attributes:
        use_knowledge: Whether to use knowledge augmentation
        num_contexts: Number of contexts to retrieve
        retriever_type: Type of retriever
        vector_store_type: Type of vector store
        context_fusion: Context fusion method
        knowledge_base_path: Path to knowledge base
    """
    use_knowledge: bool = False
    num_contexts: int = 5
    retriever_type: str = 'hybrid'
    vector_store_type: str = 'faiss'
    context_fusion: str = 'attention'
    knowledge_base_path: Optional[str] = None


@dataclass
class AnswerHeadConfig:
    """
    Configuration for answer prediction head.
    
    Attributes:
        num_answers: Number of answer classes
        hidden_dims: Hidden layer dimensions
        dropout: Dropout rate
        use_sigmoid: Whether to use sigmoid (multi-label)
        classifier_type: Classifier type
    """
    num_answers: int = 3000
    hidden_dims: List[int] = field(default_factory=lambda: [512, 256])
    dropout: float = 0.3
    use_sigmoid: bool = False
    classifier_type: str = 'mlp'


@dataclass
class VQAModelConfig:
    """
    Complete configuration for VQA model.
    
    Attributes:
        visual_encoder: Visual encoder config
        text_encoder: Text encoder config
        fusion: Fusion config
        moe: MOE config
        knowledge: Knowledge config
        answer_head: Answer head config
        embed_dim: Embedding dimension
        dropout: Global dropout rate
    """
    visual_encoder: VisualEncoderConfig = field(default_factory=VisualEncoderConfig)
    text_encoder: TextEncoderConfig = field(default_factory=TextEncoderConfig)
    fusion: FusionConfig = field(default_factory=FusionConfig)
    moe: MOEConfig = field(default_factory=MOEConfig)
    knowledge: KnowledgeConfig = field(default_factory=KnowledgeConfig)
    answer_head: AnswerHeadConfig = field(default_factory=AnswerHeadConfig)
    embed_dim: int = 768
    dropout: float = 0.1
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'VQAModelConfig':
        """
        Create config from dictionary.
        
        Args:
            config_dict: Configuration dictionary
            
        Returns:
            VQAModelConfig instance
        """
        visual_encoder = VisualEncoderConfig(**config_dict.get('visual_encoder', {}))
        text_encoder = TextEncoderConfig(**config_dict.get('text_encoder', {}))
        fusion = FusionConfig(**config_dict.get('fusion', {}))
        moe = MOEConfig(**config_dict.get('moe', {}))
        knowledge = KnowledgeConfig(**config_dict.get('knowledge', {}))
        answer_head = AnswerHeadConfig(**config_dict.get('answer_head', {}))
        
        return cls(
            visual_encoder=visual_encoder,
            text_encoder=text_encoder,
            fusion=fusion,
            moe=moe,
            knowledge=knowledge,
            answer_head=answer_head,
            embed_dim=config_dict.get('embed_dim', 768),
            dropout=config_dict.get('dropout', 0.1)
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert config to dictionary.
        
        Returns:
            Configuration dictionary
        """
        from dataclasses import asdict
        return asdict(self)


def get_default_vietnamese_vqa_config() -> VQAModelConfig:
    """
    Get default configuration for Vietnamese VQA.
    
    Returns:
        VQAModelConfig with Vietnamese-optimized settings
    """
    return VQAModelConfig(
        visual_encoder=VisualEncoderConfig(
            backbone_type='vit',
            model_name='openai/clip-vit-base-patch32',
            pretrained=True,
            freeze_backbone=False,
            output_dim=768
        ),
        text_encoder=TextEncoderConfig(
            encoder_type='phobert',
            model_name='vinai/phobert-base',
            pretrained=True,
            freeze_encoder=False,
            output_dim=768,
            max_length=128,
            pooling_strategy='cls'
        ),
        fusion=FusionConfig(
            fusion_type='cross_attention',
            hidden_dim=768,
            output_dim=768,
            num_heads=8,
            num_layers=2,
            dropout=0.1
        ),
        moe=MOEConfig(
            use_moe=True,
            num_experts=8,
            top_k=2,
            router_type='top_k',
            expert_type='feedforward'
        ),
        knowledge=KnowledgeConfig(
            use_knowledge=True,
            num_contexts=5,
            retriever_type='hybrid',
            context_fusion='attention'
        ),
        answer_head=AnswerHeadConfig(
            num_answers=3000,
            hidden_dims=[768, 512],
            dropout=0.3
        )
    )
