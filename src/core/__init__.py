"""
Core Pipeline Module
Provides complete VQA pipeline with comprehensive logging and validation.
"""

from src.core.pipeline_logger import (
    PipelineLogger,
    get_pipeline_logger,
    LogSection
)

from src.core.data_pipeline import (
    DataPipeline,
    DataPipelineConfig,
    DataPipelineOutput
)

from src.core.model_pipeline import (
    ModelPipeline,
    ModelPipelineConfig,
    ModelPipelineOutput
)

from src.core.training_pipeline import (
    TrainingPipeline,
    TrainingPipelineConfig,
    TrainingPipelineOutput
)

from src.core.vqa_pipeline import (
    VQAPipeline,
    VQAPipelineConfig,
    VQAPipelineOutput,
    main
)

from src.core.generative_training_pipeline import (
    GenerativeTrainingPipeline,
    GenerativeTrainingConfig,
    GenerativeTrainingOutput
)

from src.core.generative_vqa_pipeline import (
    GenerativeVQAPipeline,
    GenerativeVQAPipelineConfig,
    GenerativeVQAPipelineOutput
)


__all__ = [
    # Logger
    'PipelineLogger',
    'get_pipeline_logger',
    'LogSection',
    
    # Data Pipeline
    'DataPipeline',
    'DataPipelineConfig',
    'DataPipelineOutput',
    
    # Model Pipeline
    'ModelPipeline',
    'ModelPipelineConfig',
    'ModelPipelineOutput',
    
    # Training Pipeline
    'TrainingPipeline',
    'TrainingPipelineConfig',
    'TrainingPipelineOutput',
    
    # Main Pipeline
    'VQAPipeline',
    'VQAPipelineConfig',
    'VQAPipelineOutput',
    'main',
    
    # Generative VQA
    'GenerativeTrainingPipeline',
    'GenerativeTrainingConfig',
    'GenerativeTrainingOutput',
    'GenerativeVQAPipeline',
    'GenerativeVQAPipelineConfig',
    'GenerativeVQAPipelineOutput'
]
