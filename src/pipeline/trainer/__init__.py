"""
Trainer Module
Provides training pipeline for Vietnamese VQA models.
"""

from .trainer_config import (
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
    get_default_training_config
)

from .training_utils import (
    set_seed,
    get_device,
    count_parameters,
    get_parameter_groups,
    GradientAccumulator,
    EarlyStopping,
    AverageMeter,
    ProgressTracker,
    freeze_module,
    unfreeze_module,
    apply_training_strategy,
    clip_gradients,
    save_checkpoint,
    load_checkpoint
)

from .vqa_trainer import (
    TrainingState,
    VQATrainer,
    create_trainer
)

__all__ = [
    # Config enums
    'TrainingStrategy',
    'MixedPrecisionMode',
    'GradientCheckpointMode',
    
    # Config dataclasses
    'OptimizerConfig',
    'SchedulerConfig',
    'LossConfig',
    'DataConfig',
    'LoggingConfig',
    'CheckpointConfig',
    'EarlyStoppingConfig',
    'TrainingConfig',
    'get_default_training_config',
    
    # Utilities
    'set_seed',
    'get_device',
    'count_parameters',
    'get_parameter_groups',
    'GradientAccumulator',
    'EarlyStopping',
    'AverageMeter',
    'ProgressTracker',
    'freeze_module',
    'unfreeze_module',
    'apply_training_strategy',
    'clip_gradients',
    'save_checkpoint',
    'load_checkpoint',
    
    # Trainer
    'TrainingState',
    'VQATrainer',
    'create_trainer'
]
