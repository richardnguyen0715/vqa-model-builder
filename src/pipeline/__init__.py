"""
Pipeline Module
Provides training and evaluation pipelines for Vietnamese VQA models.
"""

from .trainer import (
    # Config
    TrainingStrategy,
    MixedPrecisionMode,
    GradientCheckpointMode,
    OptimizerConfig,
    SchedulerConfig,
    LossConfig,
    DataConfig,
    LoggingConfig,
    CheckpointConfig,
    EarlyStoppingConfig,
    TrainingConfig,
    get_default_training_config,
    
    # Utilities
    set_seed,
    get_device,
    count_parameters,
    freeze_module,
    unfreeze_module,
    save_checkpoint,
    load_checkpoint,
    
    # Trainer
    TrainingState,
    VQATrainer,
    create_trainer
)

from .evaluator import (
    # Config
    EvaluationMode,
    MetricConfig,
    AnalysisConfig,
    EvaluationConfig,
    get_default_evaluation_config,
    
    # Evaluator
    EvaluationResult,
    VQAEvaluator,
    create_evaluator
)

__all__ = [
    # Training
    'TrainingStrategy',
    'MixedPrecisionMode',
    'GradientCheckpointMode',
    'OptimizerConfig',
    'SchedulerConfig',
    'LossConfig',
    'DataConfig',
    'LoggingConfig',
    'CheckpointConfig',
    'EarlyStoppingConfig',
    'TrainingConfig',
    'get_default_training_config',
    'set_seed',
    'get_device',
    'count_parameters',
    'freeze_module',
    'unfreeze_module',
    'save_checkpoint',
    'load_checkpoint',
    'TrainingState',
    'VQATrainer',
    'create_trainer',
    
    # Evaluation
    'EvaluationMode',
    'MetricConfig',
    'AnalysisConfig',
    'EvaluationConfig',
    'get_default_evaluation_config',
    'EvaluationResult',
    'VQAEvaluator',
    'create_evaluator'
]
