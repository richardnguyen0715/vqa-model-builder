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

from .result_manager import (
    InferenceResult,
    InferenceResultManager,
    save_inference_results
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
    'load_predictor',
    
    # Result management
    'InferenceResult',
    'InferenceResultManager',
    'save_inference_results'
]
