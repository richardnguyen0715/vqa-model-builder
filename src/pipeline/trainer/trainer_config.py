"""
Trainer Configuration Module
Provides configuration dataclasses for VQA training.
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from enum import Enum


class TrainingStrategy(Enum):
    """Training strategy enumeration."""
    FULL = "full"
    FREEZE_VISUAL = "freeze_visual"
    FREEZE_TEXT = "freeze_text"
    LINEAR_PROBE = "linear_probe"
    GRADUAL_UNFREEZE = "gradual_unfreeze"


class MixedPrecisionMode(Enum):
    """Mixed precision training mode."""
    OFF = "off"
    FP16 = "fp16"
    BF16 = "bf16"


class GradientCheckpointMode(Enum):
    """Gradient checkpointing mode."""
    OFF = "off"
    FULL = "full"
    SELECTIVE = "selective"


@dataclass
class OptimizerConfig:
    """
    Optimizer configuration.
    
    Attributes:
        name: Optimizer name (adam, adamw, sgd, lamb)
        lr: Base learning rate
        weight_decay: Weight decay coefficient
        betas: Adam betas
        momentum: SGD momentum
        eps: Epsilon for numerical stability
        use_lookahead: Whether to use lookahead wrapper
        lookahead_k: Lookahead update frequency
        lookahead_alpha: Lookahead interpolation coefficient
    """
    name: str = "adamw"
    lr: float = 1e-4
    weight_decay: float = 0.01
    betas: tuple = (0.9, 0.999)
    momentum: float = 0.9
    eps: float = 1e-8
    use_lookahead: bool = False
    lookahead_k: int = 5
    lookahead_alpha: float = 0.5


@dataclass
class SchedulerConfig:
    """
    Learning rate scheduler configuration.
    
    Attributes:
        name: Scheduler name (cosine, linear, polynomial, step)
        warmup_steps: Number of warmup steps
        warmup_ratio: Warmup ratio of total steps
        num_cycles: Number of cycles for cosine scheduler
        power: Power for polynomial scheduler
        step_size: Step size for step scheduler
        gamma: Gamma for step scheduler
        min_lr: Minimum learning rate
    """
    name: str = "cosine"
    warmup_steps: int = 0
    warmup_ratio: float = 0.1
    num_cycles: int = 1
    power: float = 1.0
    step_size: int = 10
    gamma: float = 0.1
    min_lr: float = 0.0


@dataclass
class LossConfig:
    """
    Loss function configuration.
    
    Attributes:
        classification_loss: Loss for classification (cross_entropy, focal, label_smoothing)
        classification_weight: Weight for classification loss
        contrastive_loss: Optional contrastive loss (triplet, infonce)
        contrastive_weight: Weight for contrastive loss
        moe_aux_weight: Weight for MOE auxiliary loss
        rag_aux_weight: Weight for RAG auxiliary loss
        label_smoothing: Label smoothing factor
        focal_gamma: Focal loss gamma
        focal_alpha: Focal loss alpha
    """
    classification_loss: str = "cross_entropy"
    classification_weight: float = 1.0
    contrastive_loss: Optional[str] = None
    contrastive_weight: float = 0.1
    moe_aux_weight: float = 0.01
    rag_aux_weight: float = 0.1
    label_smoothing: float = 0.0
    focal_gamma: float = 2.0
    focal_alpha: float = 0.25


@dataclass
class DataConfig:
    """
    Data loading configuration.
    
    Attributes:
        batch_size: Training batch size
        eval_batch_size: Evaluation batch size
        num_workers: Number of data loading workers
        pin_memory: Whether to pin memory
        drop_last: Whether to drop last incomplete batch
        shuffle: Whether to shuffle training data
        prefetch_factor: Prefetch factor for data loading
    """
    batch_size: int = 32
    eval_batch_size: int = 64
    num_workers: int = 4
    pin_memory: bool = True
    drop_last: bool = True
    shuffle: bool = True
    prefetch_factor: int = 2


@dataclass
class LoggingConfig:
    """
    Logging configuration.
    
    Attributes:
        log_dir: Directory for logs
        log_interval: Logging interval in steps
        save_interval: Checkpoint save interval in epochs
        eval_interval: Evaluation interval in epochs
        use_tensorboard: Whether to use TensorBoard
        use_wandb: Whether to use Weights & Biases
        wandb_project: W&B project name
        wandb_entity: W&B entity name
        wandb_run_name: W&B run name
    """
    log_dir: str = "logs"
    log_interval: int = 100
    save_interval: int = 1
    eval_interval: int = 1
    use_tensorboard: bool = True
    use_wandb: bool = False
    wandb_project: str = "vietnamese-vqa"
    wandb_entity: Optional[str] = None
    wandb_run_name: Optional[str] = None


@dataclass
class CheckpointConfig:
    """
    Checkpoint configuration.
    
    Attributes:
        save_dir: Directory for checkpoints
        save_best: Whether to save best model
        save_last: Whether to save last model
        max_keep: Maximum checkpoints to keep
        metric_for_best: Metric to use for best model selection
        mode: Whether to maximize or minimize metric
        resume_from: Path to checkpoint to resume from
    """
    save_dir: str = "checkpoints"
    save_best: bool = True
    save_last: bool = True
    max_keep: int = 3
    metric_for_best: str = "accuracy"
    mode: str = "max"
    resume_from: Optional[str] = None


@dataclass
class EarlyStoppingConfig:
    """
    Early stopping configuration.
    
    Attributes:
        enabled: Whether to enable early stopping
        patience: Number of epochs without improvement
        min_delta: Minimum change to qualify as improvement
        metric: Metric to monitor
        mode: Whether to maximize or minimize metric
    """
    enabled: bool = True
    patience: int = 5
    min_delta: float = 0.001
    metric: str = "val_accuracy"
    mode: str = "max"


@dataclass
class TrainingConfig:
    """
    Complete training configuration.
    
    Attributes:
        num_epochs: Number of training epochs
        max_steps: Maximum training steps (overrides epochs if set)
        gradient_accumulation_steps: Number of gradient accumulation steps
        max_grad_norm: Maximum gradient norm for clipping
        strategy: Training strategy
        mixed_precision: Mixed precision mode
        gradient_checkpointing: Gradient checkpointing mode
        compile_model: Whether to compile model with torch.compile
        seed: Random seed
        deterministic: Whether to use deterministic algorithms
        optimizer: Optimizer configuration
        scheduler: Scheduler configuration
        loss: Loss configuration
        data: Data configuration
        logging: Logging configuration
        checkpoint: Checkpoint configuration
        early_stopping: Early stopping configuration
    """
    num_epochs: int = 10
    max_steps: Optional[int] = None
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0
    strategy: TrainingStrategy = TrainingStrategy.FULL
    mixed_precision: MixedPrecisionMode = MixedPrecisionMode.OFF
    gradient_checkpointing: GradientCheckpointMode = GradientCheckpointMode.OFF
    compile_model: bool = False
    seed: int = 42
    deterministic: bool = False
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    data: DataConfig = field(default_factory=DataConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)
    early_stopping: EarlyStoppingConfig = field(default_factory=EarlyStoppingConfig)


def get_default_training_config() -> TrainingConfig:
    """
    Get default training configuration for Vietnamese VQA.
    
    Returns:
        Default TrainingConfig instance
    """
    return TrainingConfig(
        num_epochs=20,
        gradient_accumulation_steps=2,
        max_grad_norm=1.0,
        strategy=TrainingStrategy.GRADUAL_UNFREEZE,
        mixed_precision=MixedPrecisionMode.FP16,
        gradient_checkpointing=GradientCheckpointMode.SELECTIVE,
        optimizer=OptimizerConfig(
            name="adamw",
            lr=2e-5,
            weight_decay=0.01,
            use_lookahead=True
        ),
        scheduler=SchedulerConfig(
            name="cosine",
            warmup_ratio=0.1
        ),
        loss=LossConfig(
            classification_loss="label_smoothing",
            label_smoothing=0.1,
            moe_aux_weight=0.01
        ),
        data=DataConfig(
            batch_size=16,
            eval_batch_size=32,
            num_workers=4
        ),
        logging=LoggingConfig(
            log_interval=50,
            save_interval=1,
            eval_interval=1
        ),
        checkpoint=CheckpointConfig(
            metric_for_best="accuracy"
        ),
        early_stopping=EarlyStoppingConfig(
            patience=5
        )
    )
