"""
Modeling Module
Provides model architectures for Vietnamese VQA.
"""

# Meta architecture imports
from .meta_arch import (
    # Config
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
    get_default_vietnamese_vqa_config,
    
    # Model
    VQAOutput,
    VisualEncoder,
    TextEncoder,
    CrossModalAttention,
    MultimodalFusion,
    AnswerHead,
    VietnameseVQAModel,
    create_vqa_model
)

# Inference imports
from .inference import (
    # Config
    InferenceMode,
    DecodingStrategy,
    InferenceConfig,
    AnswerConfig,
    PreprocessConfig,
    VQAInferenceConfig,
    get_default_inference_config,
    
    # Predictor
    PredictionResult,
    BatchPredictionResult,
    VQAPredictor,
    load_predictor
)

__all__ = [
    # Meta architecture - Config
    'BackboneType',
    'TextEncoderType',
    'FusionType',
    'VisualEncoderConfig',
    'TextEncoderConfig',
    'FusionConfig',
    'MOEConfig',
    'KnowledgeConfig',
    'AnswerHeadConfig',
    'VQAModelConfig',
    'get_default_vietnamese_vqa_config',
    
    # Meta architecture - Model
    'VQAOutput',
    'VisualEncoder',
    'TextEncoder',
    'CrossModalAttention',
    'MultimodalFusion',
    'AnswerHead',
    'VietnameseVQAModel',
    'create_vqa_model',
    
    # Inference - Config
    'InferenceMode',
    'DecodingStrategy',
    'InferenceConfig',
    'AnswerConfig',
    'PreprocessConfig',
    'VQAInferenceConfig',
    'get_default_inference_config',
    
    # Inference - Predictor
    'PredictionResult',
    'BatchPredictionResult',
    'VQAPredictor',
    'load_predictor'
]
