"""
Inference Module
Provides inference pipeline for Vietnamese VQA models.
"""

from .inference_config import (
    InferenceMode,
    DecodingStrategy,
    InferenceConfig,
    AnswerConfig,
    PreprocessConfig,
    VQAInferenceConfig,
    get_default_inference_config
)

from .vqa_predictor import (
    PredictionResult,
    BatchPredictionResult,
    VQAPredictor,
    load_predictor
)

__all__ = [
    # Config enums
    'InferenceMode',
    'DecodingStrategy',
    
    # Config dataclasses
    'InferenceConfig',
    'AnswerConfig',
    'PreprocessConfig',
    'VQAInferenceConfig',
    'get_default_inference_config',
    
    # Predictor
    'PredictionResult',
    'BatchPredictionResult',
    'VQAPredictor',
    'load_predictor'
]
