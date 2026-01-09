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

from .generative_vqa_model import (
    GenerativeVQAConfig,
    GenerativeVQAOutput,
    GenerativeVQAModel,
    create_generative_vqa_model,
    get_default_generative_vqa_config
)

__all__ = [
    # Config enums
    'BackboneType',
    'TextEncoderType',
    'FusionType',
    
    # Classification VQA Config dataclasses
    'VisualEncoderConfig',
    'TextEncoderConfig',
    'FusionConfig',
    'MOEConfig',
    'KnowledgeConfig',
    'AnswerHeadConfig',
    'VQAModelConfig',
    'get_default_vietnamese_vqa_config',
    
    # Classification VQA Model components
    'VQAOutput',
    'VisualEncoder',
    'TextEncoder',
    'CrossModalAttention',
    'MultimodalFusion',
    'AnswerHead',
    'VietnameseVQAModel',
    'create_vqa_model',
    
    # Generative VQA
    'GenerativeVQAConfig',
    'GenerativeVQAOutput',
    'GenerativeVQAModel',
    'create_generative_vqa_model',
    'get_default_generative_vqa_config',
]
