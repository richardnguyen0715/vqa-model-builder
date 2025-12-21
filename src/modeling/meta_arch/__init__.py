"""
Meta Architecture Module
Provides complete Vietnamese VQA model architectures.
"""

from .vqa_config import (
    BackboneType,
    TextEncoderType,
    FusionType,
    VisualEncoderConfig,
    TextEncoderConfig,
    FusionConfig,
    MOEConfig,
    KnowledgeConfig,
    AnswerHeadConfig,
    VQAModelConfig,
    get_default_vietnamese_vqa_config
)

from .vqa_model import (
    VQAOutput,
    VisualEncoder,
    TextEncoder,
    CrossModalAttention,
    MultimodalFusion,
    AnswerHead,
    VietnameseVQAModel,
    create_vqa_model
)

__all__ = [
    # Config enums
    'BackboneType',
    'TextEncoderType',
    'FusionType',
    
    # Config dataclasses
    'VisualEncoderConfig',
    'TextEncoderConfig',
    'FusionConfig',
    'MOEConfig',
    'KnowledgeConfig',
    'AnswerHeadConfig',
    'VQAModelConfig',
    'get_default_vietnamese_vqa_config',
    
    # Model components
    'VQAOutput',
    'VisualEncoder',
    'TextEncoder',
    'CrossModalAttention',
    'MultimodalFusion',
    'AnswerHead',
    'VietnameseVQAModel',
    'create_vqa_model'
]
