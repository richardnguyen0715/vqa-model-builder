"""
VQA Trainer Module
Provides the main trainer class for Vietnamese VQA models.
Integrates with centralized configuration system and checkpoint management.
Includes resource monitoring and automatic backup functionality.
"""

import os
import time
import signal
import logging
from typing import Optional, Dict, Any, Callable, List
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.amp import GradScaler

from .trainer_config import (
    TrainingConfig,
    TrainingStrategy,
    MixedPrecisionMode,
    GradientCheckpointMode,
    get_default_training_config
)
from .training_utils import (
    set_seed,
    get_device,
    count_parameters,
    get_parameter_groups,
    GradientAccumulator,
    EarlyStopping,
    ProgressTracker,
    apply_training_strategy,
    clip_gradients,
    save_checkpoint,
    load_checkpoint
)
from .checkpoint_manager import CheckpointManager
from src.middleware.config_loader import get_config, get_config_manager

# Resource management imports
try:
    from src.resource_management import (
        ResourceManager,
        get_resource_manager,
        resource_managed_training,
        load_resource_config,
    )
    RESOURCE_MANAGEMENT_AVAILABLE = True
except ImportError:
    RESOURCE_MANAGEMENT_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class TrainingState:
    """
    Training state container.
    
    Attributes:
        epoch: Current epoch
        global_step: Global training step
        best_metric: Best validation metric
        is_best: Whether current epoch is best
        resource_task_id: Task ID for resource tracking
    """
    epoch: int = 0
    global_step: int = 0
    best_metric: float = 0.0
    is_best: bool = False
    resource_task_id: Optional[str] = None


class VQATrainer:
    """
    Vietnamese VQA model trainer.
    
    Provides complete training loop with:
    - Mixed precision training
    - Gradient accumulation
    - Learning rate scheduling
    - Early stopping
    - Checkpointing (integrated with CheckpointManager)
    - Logging (TensorBoard, W&B)
    - Configuration from YAML files
    - Resource monitoring and automatic backup
    
    Attributes:
        model: VQA model to train
        config: Training configuration
        device: Training device
        optimizer: Model optimizer
        scheduler: Learning rate scheduler
        scaler: Gradient scaler for mixed precision
        early_stopping: Early stopping handler
        checkpoint_manager: Checkpoint management handler
        state: Current training state
        resource_manager: Resource monitoring and backup handler
    """
    
    def __init__(
        self,
        model: nn.Module,
        config: Optional[TrainingConfig] = None,
        train_dataloader: Optional[DataLoader] = None,
        val_dataloader: Optional[DataLoader] = None,
        loss_fn: Optional[Callable] = None,
        metrics_fn: Optional[Callable] = None,
        use_yaml_config: bool = True
    ):
        """
        Initialize trainer.
        
        Args:
            model: VQA model to train
            config: Training configuration (overrides YAML if provided)
            train_dataloader: Training data loader
            val_dataloader: Validation data loader
            loss_fn: Loss function (receives model output and targets)
            metrics_fn: Metrics function (receives predictions and targets)
            use_yaml_config: Whether to load settings from YAML config files
        """
        # Load configuration
        self.use_yaml_config = use_yaml_config
        self.config = config or self._load_config_from_yaml() or get_default_training_config()
        self.device = get_device()
        
        # Set seed for reproducibility
        seed = self._get_training_param("seed", self.config.seed)
        deterministic = self._get_training_param("deterministic", self.config.deterministic)
        set_seed(seed, deterministic)
        
        # Setup model
        self.model = model.to(self.device)
        self._setup_gradient_checkpointing()
        self._compile_model()
        
        # Data loaders
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        
        # Loss and metrics
        self.loss_fn = loss_fn or self._default_loss_fn
        self.metrics_fn = metrics_fn or self._default_metrics_fn
        
        # Setup optimizer and scheduler
        self.optimizer = self._create_optimizer()
        self.scheduler = self._create_scheduler()
        
        # Setup mixed precision
        self.scaler = self._setup_mixed_precision()
        
        # Setup early stopping
        self.early_stopping = self._setup_early_stopping()
        
        # Setup checkpoint manager
        self.checkpoint_manager = self._setup_checkpoint_manager()
        
        # Setup resource manager for monitoring and automatic backup
        self.resource_manager = self._setup_resource_manager()
        
        # Gradient accumulation
        grad_accum_steps = self._get_training_param(
            "gradient_accumulation_steps",
            self.config.gradient_accumulation_steps
        )
        self.grad_accumulator = GradientAccumulator(grad_accum_steps)
        
        # Progress tracking
        self.train_tracker = ProgressTracker(["loss", "accuracy"])
        self.val_tracker = ProgressTracker(["loss", "accuracy"])
        
        # Training state
        self.state = TrainingState()
        
        # Logging
        self.writers = self._setup_logging()
        
        # Setup interrupt handling
        self._setup_interrupt_handler()
        
        # Resume from checkpoint if specified
        resume_path = self._get_checkpoint_resume_path()
        if resume_path:
            self._resume_from_checkpoint(resume_path)
        
        logger.info(f"Initialized trainer with {count_parameters(self.model):,} trainable parameters")
    
    def _load_config_from_yaml(self) -> Optional[TrainingConfig]:
        """
        Load training configuration from YAML files.
        
        Returns:
            TrainingConfig instance loaded from YAML, or None if loading fails.
        """
        if not self.use_yaml_config:
            return None
        
        try:
            from .trainer_config import (
                OptimizerConfig, SchedulerConfig, LossConfig,
                DataConfig, LoggingConfig, CheckpointConfig, EarlyStoppingConfig
            )
            
            # Load optimizer config
            opt_cfg = get_config("training", "optimizer", {})
            optimizer_config = OptimizerConfig(
                name=opt_cfg.get("name", "adamw"),
                lr=opt_cfg.get("learning_rate", 1e-4),
                weight_decay=opt_cfg.get("weight_decay", 0.01),
                betas=tuple(opt_cfg.get("betas", [0.9, 0.999])),
                momentum=opt_cfg.get("momentum", 0.9),
                eps=opt_cfg.get("eps", 1e-8),
                use_lookahead=opt_cfg.get("use_lookahead", False),
                lookahead_k=opt_cfg.get("lookahead_k", 5),
                lookahead_alpha=opt_cfg.get("lookahead_alpha", 0.5)
            )
            
            # Load scheduler config
            sched_cfg = get_config("training", "scheduler", {})
            scheduler_config = SchedulerConfig(
                name=sched_cfg.get("name", "cosine"),
                warmup_steps=sched_cfg.get("warmup_steps", 0),
                warmup_ratio=sched_cfg.get("warmup_ratio", 0.1),
                num_cycles=sched_cfg.get("num_cycles", 1),
                power=sched_cfg.get("power", 1.0),
                step_size=sched_cfg.get("step_size", 10),
                gamma=sched_cfg.get("gamma", 0.1),
                min_lr=sched_cfg.get("min_lr", 0.0)
            )
            
            # Load loss config
            loss_cfg = get_config("training", "loss", {})
            loss_config = LossConfig(
                classification_loss=loss_cfg.get("classification_loss", "cross_entropy"),
                classification_weight=loss_cfg.get("classification_weight", 1.0),
                contrastive_loss=loss_cfg.get("contrastive_loss"),
                contrastive_weight=loss_cfg.get("contrastive_weight", 0.1),
                moe_aux_weight=loss_cfg.get("moe_aux_weight", 0.01),
                rag_aux_weight=loss_cfg.get("rag_aux_weight", 0.1),
                label_smoothing=loss_cfg.get("label_smoothing", 0.0),
                focal_gamma=loss_cfg.get("focal_gamma", 2.0),
                focal_alpha=loss_cfg.get("focal_alpha", 0.25)
            )
            
            # Load data config
            data_cfg = get_config("training", "data", {})
            data_config = DataConfig(
                batch_size=data_cfg.get("batch_size", 32),
                eval_batch_size=data_cfg.get("eval_batch_size", 64),
                num_workers=data_cfg.get("num_workers", 4),
                pin_memory=data_cfg.get("pin_memory", True),
                drop_last=data_cfg.get("drop_last", True),
                shuffle=data_cfg.get("shuffle", True),
                prefetch_factor=data_cfg.get("prefetch_factor", 2)
            )
            
            # Load logging config
            log_cfg = get_config("training", "logging", {})
            logging_config = LoggingConfig(
                log_dir=log_cfg.get("log_dir", "logs"),
                log_interval=log_cfg.get("log_interval", 100),
                save_interval=get_config("checkpoint", "save.save_every_k_epochs", 1),
                eval_interval=log_cfg.get("eval_interval", 1),
                use_tensorboard=log_cfg.get("use_tensorboard", True),
                use_wandb=log_cfg.get("use_wandb", False),
                wandb_project=log_cfg.get("wandb_project", "vietnamese-vqa"),
                wandb_entity=log_cfg.get("wandb_entity"),
                wandb_run_name=log_cfg.get("wandb_run_name")
            )
            
            # Load checkpoint config
            ckpt_cfg = get_config("checkpoint", {})
            checkpoint_config = CheckpointConfig(
                save_dir=ckpt_cfg.get("save", {}).get("base_dir", "checkpoints"),
                save_best=ckpt_cfg.get("save", {}).get("keep_best", True),
                save_last=True,
                max_keep=ckpt_cfg.get("save", {}).get("max_keep", 3),
                metric_for_best=ckpt_cfg.get("best_model", {}).get("metric", "accuracy"),
                mode=ckpt_cfg.get("best_model", {}).get("mode", "max"),
                resume_from=ckpt_cfg.get("resume", {}).get("checkpoint_path")
            )
            
            # Load early stopping config
            es_cfg = get_config("training", "early_stopping", {})
            early_stopping_config = EarlyStoppingConfig(
                enabled=es_cfg.get("enabled", True),
                patience=es_cfg.get("patience", 5),
                min_delta=es_cfg.get("min_delta", 0.001),
                metric=es_cfg.get("metric", "val_accuracy"),
                mode=es_cfg.get("mode", "max")
            )
            
            # Load training params
            train_cfg = get_config("training", "training", {})
            
            # Map string to enum for strategy
            strategy_map = {
                "full": TrainingStrategy.FULL,
                "freeze_visual": TrainingStrategy.FREEZE_VISUAL,
                "freeze_text": TrainingStrategy.FREEZE_TEXT,
                "linear_probe": TrainingStrategy.LINEAR_PROBE,
                "gradual_unfreeze": TrainingStrategy.GRADUAL_UNFREEZE
            }
            strategy_str = train_cfg.get("strategy", "full")
            strategy = strategy_map.get(strategy_str, TrainingStrategy.FULL)
            
            # Map string to enum for mixed precision
            mp_map = {
                "off": MixedPrecisionMode.OFF,
                "fp16": MixedPrecisionMode.FP16,
                "bf16": MixedPrecisionMode.BF16
            }
            mp_str = train_cfg.get("mixed_precision", "off")
            mixed_precision = mp_map.get(mp_str, MixedPrecisionMode.OFF)
            
            # Map string to enum for gradient checkpointing
            gc_map = {
                "off": GradientCheckpointMode.OFF,
                "full": GradientCheckpointMode.FULL,
                "selective": GradientCheckpointMode.SELECTIVE
            }
            gc_str = train_cfg.get("gradient_checkpointing", "off")
            gradient_checkpointing = gc_map.get(gc_str, GradientCheckpointMode.OFF)
            
            return TrainingConfig(
                num_epochs=train_cfg.get("num_epochs", 10),
                max_steps=train_cfg.get("max_steps"),
                gradient_accumulation_steps=train_cfg.get("gradient_accumulation_steps", 1),
                max_grad_norm=train_cfg.get("max_grad_norm", 1.0),
                strategy=strategy,
                mixed_precision=mixed_precision,
                gradient_checkpointing=gradient_checkpointing,
                compile_model=train_cfg.get("compile_model", False),
                seed=train_cfg.get("seed", 42),
                deterministic=train_cfg.get("deterministic", False),
                optimizer=optimizer_config,
                scheduler=scheduler_config,
                loss=loss_config,
                data=data_config,
                logging=logging_config,
                checkpoint=checkpoint_config,
                early_stopping=early_stopping_config
            )
            
        except Exception as e:
            logger.warning(f"Failed to load config from YAML: {e}. Using defaults.")
            return None
    
    def _get_training_param(self, key: str, default: Any) -> Any:
        """
        Get training parameter from YAML config or use default.
        
        Args:
            key: Configuration key in training.training section.
            default: Default value if not found.
            
        Returns:
            Configuration value.
        """
        if self.use_yaml_config:
            yaml_value = get_config("training", f"training.{key}")
            if yaml_value is not None:
                return yaml_value
        return default
    
    def _get_checkpoint_resume_path(self) -> Optional[str]:
        """
        Get checkpoint path for resuming training.
        
        Returns:
            Checkpoint path or None.
        """
        # Check config object first
        if self.config.checkpoint.resume_from:
            return self.config.checkpoint.resume_from
        
        # Check YAML config
        if self.use_yaml_config:
            return get_config("checkpoint", "resume.checkpoint_path")
        
        return None
    
    def _setup_checkpoint_manager(self) -> CheckpointManager:
        """
        Setup checkpoint manager for saving and loading checkpoints.
        
        Returns:
            Initialized CheckpointManager instance.
        """
        return CheckpointManager(
            save_dir=self.config.checkpoint.save_dir,
            max_keep=self.config.checkpoint.max_keep,
            metric_name=self.config.checkpoint.metric_for_best,
            metric_mode=self.config.checkpoint.mode
        )
    
    def _setup_resource_manager(self) -> Optional["ResourceManager"]:
        """
        Setup resource manager for monitoring and automatic backup.
        
        Creates and configures resource manager to monitor system resources
        during training and automatically create backups when thresholds
        are exceeded.
        
        Returns:
            Initialized ResourceManager instance or None if not available.
        """
        if not RESOURCE_MANAGEMENT_AVAILABLE:
            logger.info("Resource management module not available")
            return None
        
        try:
            # Load resource configuration from YAML
            resource_config = load_resource_config()
            
            # Create resource manager
            resource_manager = ResourceManager(resource_config)
            
            # Register model and optimizer for automatic backup
            resource_manager.register_model(
                model=self.model,
                optimizer=self.optimizer,
                scheduler=self.scheduler,
            )
            
            # Start monitoring
            resource_manager.start()
            
            logger.info("Resource manager initialized with monitoring and auto-backup")
            return resource_manager
            
        except Exception as e:
            logger.warning(f"Failed to initialize resource manager: {e}")
            return None
    
    def _setup_interrupt_handler(self) -> None:
        """Setup handler to save checkpoint on keyboard interrupt."""
        def interrupt_handler(signum, frame):
            logger.info("Interrupt received. Saving checkpoint before exit...")
            self._save_interrupt_checkpoint()
            raise KeyboardInterrupt("Training interrupted by user")
        
        signal.signal(signal.SIGINT, interrupt_handler)
    
    def _setup_gradient_checkpointing(self) -> None:
        """Setup gradient checkpointing if enabled."""
        if self.config.gradient_checkpointing == GradientCheckpointMode.OFF:
            return
        
        if hasattr(self.model, "gradient_checkpointing_enable"):
            self.model.gradient_checkpointing_enable()
            logger.info("Enabled gradient checkpointing")
    
    def _compile_model(self) -> None:
        """Compile model with torch.compile if enabled."""
        if not self.config.compile_model:
            return
        
        if hasattr(torch, "compile"):
            self.model = torch.compile(self.model)
            logger.info("Compiled model with torch.compile")
    
    def _create_optimizer(self) -> torch.optim.Optimizer:
        """Create optimizer based on configuration."""
        opt_config = self.config.optimizer
        
        # Get parameter groups
        param_groups = get_parameter_groups(
            self.model,
            opt_config.lr,
            opt_config.weight_decay
        )
        
        # Create optimizer
        if opt_config.name.lower() == "adam":
            optimizer = torch.optim.Adam(
                param_groups,
                lr=opt_config.lr,
                betas=opt_config.betas,
                eps=opt_config.eps
            )
        elif opt_config.name.lower() == "adamw":
            optimizer = torch.optim.AdamW(
                param_groups,
                lr=opt_config.lr,
                betas=opt_config.betas,
                eps=opt_config.eps
            )
        elif opt_config.name.lower() == "sgd":
            optimizer = torch.optim.SGD(
                param_groups,
                lr=opt_config.lr,
                momentum=opt_config.momentum,
                weight_decay=opt_config.weight_decay
            )
        else:
            raise ValueError(f"Unknown optimizer: {opt_config.name}")
        
        # Wrap with lookahead if enabled
        if opt_config.use_lookahead:
            try:
                from ...solvers.optimizers import Lookahead
                optimizer = Lookahead(
                    optimizer,
                    k=opt_config.lookahead_k,
                    alpha=opt_config.lookahead_alpha
                )
                logger.info("Applied Lookahead optimizer wrapper")
            except ImportError:
                logger.warning("Lookahead not available, using base optimizer")
        
        return optimizer
    
    def _create_scheduler(self) -> Any:
        """Create learning rate scheduler."""
        sched_config = self.config.scheduler
        
        if self.train_dataloader is None:
            total_steps = self.config.max_steps or 1000
        else:
            steps_per_epoch = len(self.train_dataloader) // self.config.gradient_accumulation_steps
            total_steps = self.config.max_steps or (steps_per_epoch * self.config.num_epochs)
        
        warmup_steps = sched_config.warmup_steps or int(total_steps * sched_config.warmup_ratio)
        
        if sched_config.name.lower() == "cosine":
            from torch.optim.lr_scheduler import CosineAnnealingLR
            main_scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=total_steps - warmup_steps,
                eta_min=sched_config.min_lr
            )
        elif sched_config.name.lower() == "linear":
            from torch.optim.lr_scheduler import LinearLR
            main_scheduler = LinearLR(
                self.optimizer,
                start_factor=1.0,
                end_factor=sched_config.min_lr / self.config.optimizer.lr,
                total_iters=total_steps - warmup_steps
            )
        elif sched_config.name.lower() == "step":
            from torch.optim.lr_scheduler import StepLR
            main_scheduler = StepLR(
                self.optimizer,
                step_size=sched_config.step_size,
                gamma=sched_config.gamma
            )
        else:
            from torch.optim.lr_scheduler import ConstantLR
            main_scheduler = ConstantLR(self.optimizer, factor=1.0, total_iters=total_steps)
        
        # Add warmup
        if warmup_steps > 0:
            from torch.optim.lr_scheduler import SequentialLR, LinearLR
            warmup_scheduler = LinearLR(
                self.optimizer,
                start_factor=0.1,
                end_factor=1.0,
                total_iters=warmup_steps
            )
            scheduler = SequentialLR(
                self.optimizer,
                schedulers=[warmup_scheduler, main_scheduler],
                milestones=[warmup_steps]
            )
        else:
            scheduler = main_scheduler
        
        return scheduler
    
    def _setup_mixed_precision(self) -> Optional[GradScaler]:
        """Setup mixed precision training."""
        if self.config.mixed_precision == MixedPrecisionMode.OFF:
            return None
        
        device_type = self.device.type if hasattr(self.device, 'type') else str(self.device).split(':')[0]
        
        if self.config.mixed_precision == MixedPrecisionMode.FP16:
            logger.info("Using FP16 mixed precision")
            return torch.amp.GradScaler(device_type)
        elif self.config.mixed_precision == MixedPrecisionMode.BF16:
            logger.info("Using BF16 mixed precision")
            is_bf16_supported = device_type == 'cuda' and torch.cuda.is_bf16_supported()
            return torch.amp.GradScaler(device_type, enabled=is_bf16_supported)
        
        return None
    
    def _setup_early_stopping(self) -> Optional[EarlyStopping]:
        """Setup early stopping if enabled."""
        if not self.config.early_stopping.enabled:
            return None
        
        return EarlyStopping(
            patience=self.config.early_stopping.patience,
            min_delta=self.config.early_stopping.min_delta,
            mode=self.config.early_stopping.mode
        )
    
    def _setup_logging(self) -> Dict[str, Any]:
        """Setup logging writers."""
        writers = {}
        
        log_config = self.config.logging
        
        if log_config.use_tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter
                tb_dir = os.path.join(log_config.log_dir, "tensorboard")
                os.makedirs(tb_dir, exist_ok=True)
                writers["tensorboard"] = SummaryWriter(tb_dir)
                logger.info(f"TensorBoard logging to {tb_dir}")
            except ImportError:
                logger.warning("TensorBoard not available")
        
        if log_config.use_wandb:
            try:
                import wandb
                wandb.init(
                    project=log_config.wandb_project,
                    entity=log_config.wandb_entity,
                    name=log_config.wandb_run_name,
                    config=self.config.__dict__
                )
                writers["wandb"] = wandb
                logger.info(f"W&B logging to project {log_config.wandb_project}")
            except ImportError:
                logger.warning("Weights & Biases not available")
        
        return writers
    
    def _resume_from_checkpoint(self, path: str) -> None:
        """
        Resume training from checkpoint.
        
        Args:
            path: Path to checkpoint file.
        """
        # Try using checkpoint manager first
        try:
            loaded_state = self.checkpoint_manager.load(
                checkpoint_path=path,
                model=self.model,
                optimizer=self.optimizer,
                scheduler=self.scheduler,
                early_stopping=self.early_stopping
            )
            
            self.state.epoch = loaded_state.get("epoch", 0)
            self.state.global_step = loaded_state.get("global_step", 0)
            if loaded_state.get("metrics"):
                self.state.best_metric = loaded_state["metrics"].get(
                    self.config.checkpoint.metric_for_best, 0.0
                )
            
            logger.info(f"Resumed from checkpoint: epoch {self.state.epoch}, step {self.state.global_step}")
            
        except Exception as e:
            # Fallback to legacy load function
            logger.warning(f"CheckpointManager load failed: {e}. Using legacy loader.")
            epoch, step, metrics = load_checkpoint(
                path,
                self.model,
                self.optimizer,
                self.scheduler,
                self.early_stopping
            )
            
            self.state.epoch = epoch
            self.state.global_step = step
            if metrics:
                self.state.best_metric = metrics.get(
                    self.config.checkpoint.metric_for_best, 0.0
                )
    
    def _save_interrupt_checkpoint(self) -> None:
        """Save checkpoint when training is interrupted."""
        try:
            self.checkpoint_manager.save(
                model=self.model,
                optimizer=self.optimizer,
                scheduler=self.scheduler,
                scaler=self.scaler,
                early_stopping=self.early_stopping,
                epoch=self.state.epoch,
                global_step=self.state.global_step,
                metrics={"accuracy": self.state.best_metric},
                is_interrupt=True
            )
            logger.info("Interrupt checkpoint saved successfully")
        except Exception as e:
            logger.error(f"Failed to save interrupt checkpoint: {e}")
    
    def _default_loss_fn(self, outputs: Dict[str, Any], batch: Dict[str, Any]) -> torch.Tensor:
        """Default loss function using cross entropy."""
        # Handle both dict and dataclass outputs
        if hasattr(outputs, 'logits'):
            logits = outputs.logits
        elif hasattr(outputs, 'get'):
            logits = outputs.get("logits", outputs.get("answer_logits"))
        else:
            logits = outputs
        
        # Handle both dict and different key names for targets
        if hasattr(batch, 'get'):
            targets = batch.get("labels", batch.get("answer_ids"))
        else:
            targets = batch.labels if hasattr(batch, 'labels') else batch.answer_ids
        
        return nn.functional.cross_entropy(logits, targets)
    
    def _default_metrics_fn(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor
    ) -> Dict[str, float]:
        """Default metrics function computing accuracy."""
        pred_ids = predictions.argmax(dim=-1)
        correct = (pred_ids == targets).float().sum()
        total = targets.numel()
        return {"accuracy": (correct / total).item()}
    
    def _get_amp_dtype(self) -> torch.dtype:
        """Get dtype for automatic mixed precision."""
        if self.config.mixed_precision == MixedPrecisionMode.BF16:
            return torch.bfloat16
        elif self.config.mixed_precision == MixedPrecisionMode.FP16:
            return torch.float16
        return torch.float32
    
    def _to_device(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """Move batch to device."""
        result = {}
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                result[key] = value.to(self.device)
            elif isinstance(value, dict):
                result[key] = self._to_device(value)
            else:
                result[key] = value
        return result
    
    def train_step(self, batch: Dict[str, Any]) -> Dict[str, float]:
        """
        Single training step.
        
        Args:
            batch: Input batch
            
        Returns:
            Dictionary of step metrics
        """
        batch = self._to_device(batch)
        
        # Forward pass with mixed precision
        device_type = self.device.type if hasattr(self.device, 'type') else str(self.device).split(':')[0]
        with torch.amp.autocast(
            device_type=device_type,
            enabled=self.scaler is not None,
            dtype=self._get_amp_dtype()
        ):
            outputs = self.model(
                pixel_values=batch.get("pixel_values", batch.get("image", batch.get("images"))),
                input_ids=batch.get("input_ids"),
                attention_mask=batch.get("attention_mask"),
                labels=batch.get("labels", batch.get("answer_ids"))
            )
            
            loss = self.loss_fn(outputs, batch)
            
            # Add auxiliary losses
            if hasattr(outputs, "aux_loss") and outputs.aux_loss is not None:
                loss = loss + self.config.loss.moe_aux_weight * outputs.aux_loss
            
            # Scale loss for gradient accumulation
            loss = loss / self.config.gradient_accumulation_steps
        
        # Backward pass
        if self.scaler is not None:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()
        
        # Compute metrics
        with torch.no_grad():
            logits = outputs.logits if hasattr(outputs, "logits") else outputs.get("logits")
            targets = batch.get("labels", batch.get("answer_ids"))
            metrics = self._default_metrics_fn(logits, targets)
        
        step_metrics = {
            "loss": loss.item() * self.config.gradient_accumulation_steps,
            **metrics
        }
        
        # Optimizer step if accumulated enough
        if self.grad_accumulator.should_step():
            # Gradient clipping
            if self.scaler is not None:
                self.scaler.unscale_(self.optimizer)
            
            grad_norm = clip_gradients(self.model, self.config.max_grad_norm)
            step_metrics["grad_norm"] = grad_norm
            
            # Optimizer step
            if self.scaler is not None:
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                self.optimizer.step()
            
            self.optimizer.zero_grad()
            self.scheduler.step()
            self.state.global_step += 1
            
            step_metrics["lr"] = self.optimizer.param_groups[0]["lr"]
            
            # Update resource tracking with step progress
            self._update_resource_step(step_metrics)
        
        return step_metrics
    
    def _update_resource_step(self, step_metrics: Dict[str, float]) -> None:
        """
        Update resource manager with step progress.
        
        Args:
            step_metrics: Metrics from the training step.
        """
        if self.resource_manager is None or self.state.resource_task_id is None:
            return
        
        try:
            self.resource_manager.update_training_step(
                task_id=self.state.resource_task_id,
                step=self.state.global_step,
                loss=step_metrics.get("loss", 0.0),
                metrics={
                    k: v for k, v in step_metrics.items() 
                    if k not in ("loss", "lr", "grad_norm")
                },
                learning_rate=step_metrics.get("lr"),
            )
        except Exception as e:
            # Do not interrupt training if resource tracking fails
            pass
    
    @torch.no_grad()
    def eval_step(self, batch: Dict[str, Any]) -> Dict[str, float]:
        """
        Single evaluation step.
        
        Args:
            batch: Input batch
            
        Returns:
            Dictionary of step metrics
        """
        batch = self._to_device(batch)
        
        device_type = self.device.type if hasattr(self.device, 'type') else str(self.device).split(':')[0]
        with torch.amp.autocast(
            device_type=device_type,
            enabled=self.scaler is not None,
            dtype=self._get_amp_dtype()
        ):
            outputs = self.model(
                pixel_values=batch.get("pixel_values", batch.get("image", batch.get("images"))),
                input_ids=batch.get("input_ids"),
                attention_mask=batch.get("attention_mask"),
                labels=batch.get("labels", batch.get("answer_ids"))
            )
            
            loss = self.loss_fn(outputs, batch)
        
        logits = outputs.logits if hasattr(outputs, "logits") else outputs.get("logits")
        targets = batch.get("labels", batch.get("answer_ids"))
        metrics = self._default_metrics_fn(logits, targets)
        
        return {"loss": loss.item(), **metrics}
    
    def train_epoch(self) -> Dict[str, float]:
        """
        Train for one epoch.
        
        Returns:
            Dictionary of epoch metrics
        """
        self.model.train()
        self.train_tracker.reset()
        
        # Apply training strategy
        apply_training_strategy(
            self.model,
            self.config.strategy.value,
            self.state.epoch,
            self.config.num_epochs
        )
        
        start_time = time.time()
        
        for step, batch in enumerate(self.train_dataloader):
            step_metrics = self.train_step(batch)
            batch_size = batch.get("image", batch.get("images")).size(0)
            self.train_tracker.update(step_metrics, batch_size)
            
            # Logging
            if (step + 1) % self.config.logging.log_interval == 0:
                self._log_step(step, step_metrics, "train")
        
        epoch_time = time.time() - start_time
        epoch_metrics = self.train_tracker.epoch_end()
        epoch_metrics["epoch_time"] = epoch_time
        
        return epoch_metrics
    
    @torch.no_grad()
    def evaluate(self) -> Dict[str, float]:
        """
        Evaluate model on validation set.
        
        Returns:
            Dictionary of evaluation metrics
        """
        if self.val_dataloader is None:
            return {}
        
        self.model.eval()
        self.val_tracker.reset()
        
        for batch in self.val_dataloader:
            step_metrics = self.eval_step(batch)
            batch_size = batch.get("image", batch.get("images")).size(0)
            self.val_tracker.update(step_metrics, batch_size)
        
        return self.val_tracker.epoch_end()
    
    def _log_step(self, step: int, metrics: Dict[str, float], prefix: str) -> None:
        """Log step metrics."""
        # Console logging
        metric_str = " | ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
        logger.info(f"{prefix.upper()} Step {step} | {metric_str}")
        
        # TensorBoard
        if "tensorboard" in self.writers:
            for key, value in metrics.items():
                self.writers["tensorboard"].add_scalar(
                    f"{prefix}/{key}",
                    value,
                    self.state.global_step
                )
        
        # W&B
        if "wandb" in self.writers:
            self.writers["wandb"].log(
                {f"{prefix}/{key}": value for key, value in metrics.items()},
                step=self.state.global_step
            )
    
    def _log_epoch(self, epoch: int, train_metrics: Dict[str, float], val_metrics: Dict[str, float]) -> None:
        """Log epoch metrics."""
        train_str = " | ".join([f"{k}: {v:.4f}" for k, v in train_metrics.items()])
        logger.info(f"Epoch {epoch} TRAIN | {train_str}")
        
        if val_metrics:
            val_str = " | ".join([f"{k}: {v:.4f}" for k, v in val_metrics.items()])
            logger.info(f"Epoch {epoch} VAL | {val_str}")
        
        # TensorBoard
        if "tensorboard" in self.writers:
            for key, value in train_metrics.items():
                self.writers["tensorboard"].add_scalar(f"epoch/train_{key}", value, epoch)
            for key, value in val_metrics.items():
                self.writers["tensorboard"].add_scalar(f"epoch/val_{key}", value, epoch)
        
        # W&B
        if "wandb" in self.writers:
            self.writers["wandb"].log({
                **{f"epoch/train_{k}": v for k, v in train_metrics.items()},
                **{f"epoch/val_{k}": v for k, v in val_metrics.items()},
                "epoch": epoch
            })
    
    def _save_checkpoint(self, is_best: bool = False) -> None:
        """
        Save checkpoint using checkpoint manager.
        
        Args:
            is_best: Whether current model is the best so far.
        """
        current_metrics = {self.config.checkpoint.metric_for_best: self.state.best_metric}
        
        # Use checkpoint manager for saving
        try:
            self.checkpoint_manager.save(
                model=self.model,
                optimizer=self.optimizer,
                scheduler=self.scheduler,
                scaler=self.scaler,
                early_stopping=self.early_stopping,
                epoch=self.state.epoch,
                global_step=self.state.global_step,
                metrics=current_metrics,
                is_best=is_best
            )
            
        except Exception as e:
            logger.error(f"CheckpointManager save failed: {e}. Using fallback save.")
            self._fallback_save_checkpoint(is_best)
    
    def _fallback_save_checkpoint(self, is_best: bool = False) -> None:
        """
        Fallback checkpoint saving using legacy function.
        
        Args:
            is_best: Whether current model is the best so far.
        """
        ckpt_config = self.config.checkpoint
        
        # Save last checkpoint
        if ckpt_config.save_last:
            path = os.path.join(ckpt_config.save_dir, "last.pt")
            save_checkpoint(
                self.model,
                self.optimizer,
                self.scheduler,
                self.state.epoch,
                self.state.global_step,
                {"accuracy": self.state.best_metric},
                path,
                self.early_stopping
            )
        
        # Save best checkpoint
        if is_best and ckpt_config.save_best:
            path = os.path.join(ckpt_config.save_dir, "best.pt")
            save_checkpoint(
                self.model,
                self.optimizer,
                self.scheduler,
                self.state.epoch,
                self.state.global_step,
                {"accuracy": self.state.best_metric},
                path,
                self.early_stopping
            )
        
        # Save periodic checkpoint
        save_interval = get_config(
            "checkpoint", 
            "save.save_every_k_epochs",
            ckpt_config.save_interval
        ) if self.use_yaml_config else ckpt_config.save_interval
        
        if self.state.epoch % save_interval == 0:
            path = os.path.join(ckpt_config.save_dir, f"epoch_{self.state.epoch}.pt")
            save_checkpoint(
                self.model,
                self.optimizer,
                self.scheduler,
                self.state.epoch,
                self.state.global_step,
                {"accuracy": self.state.best_metric},
                path,
                self.early_stopping
            )
    
    def train(self) -> Dict[str, Any]:
        """
        Run full training loop.
        
        Integrates resource monitoring and automatic backup.
        Tracks training progress for reporting.
        
        Returns:
            Dictionary of training results
        """
        logger.info("Starting training...")
        logger.info(f"Config: epochs={self.config.num_epochs}, batch_size={self.config.data.batch_size}")
        
        training_results = {
            "best_metric": 0.0,
            "best_epoch": 0,
            "history": {"train": [], "val": []}
        }
        
        # Start resource tracking if available
        steps_per_epoch = (
            len(self.train_dataloader) if self.train_dataloader else 100
        )
        self._start_resource_tracking(steps_per_epoch)
        
        try:
            for epoch in range(self.state.epoch, self.config.num_epochs):
                self.state.epoch = epoch
                
                # Notify resource manager of epoch start
                self._on_epoch_start(epoch)
                
                # Training
                train_metrics = self.train_epoch()
                training_results["history"]["train"].append(train_metrics)
                
                # Evaluation
                val_metrics = {}
                if (epoch + 1) % self.config.logging.eval_interval == 0:
                    val_metrics = self.evaluate()
                    training_results["history"]["val"].append(val_metrics)
                
                # Log epoch results
                self._log_epoch(epoch, train_metrics, val_metrics)
                
                # Notify resource manager of epoch end
                self._on_epoch_end(
                    epoch,
                    train_metrics.get("loss", 0.0),
                    val_metrics.get("loss"),
                    val_metrics.get(self.config.checkpoint.metric_for_best)
                )
                
                # Check for best model
                metric_key = self.config.checkpoint.metric_for_best
                current_metric = val_metrics.get(metric_key, train_metrics.get(metric_key, 0.0))
                
                if self.config.checkpoint.mode == "max":
                    is_best = current_metric > self.state.best_metric
                else:
                    is_best = current_metric < self.state.best_metric
                
                if is_best:
                    self.state.best_metric = current_metric
                    self.state.is_best = True
                    training_results["best_metric"] = current_metric
                    training_results["best_epoch"] = epoch
                else:
                    self.state.is_best = False
                
                # Save checkpoint
                self._save_checkpoint(is_best)
                
                # Early stopping
                if self.early_stopping is not None:
                    if self.early_stopping(current_metric):
                        logger.info("Early stopping triggered")
                        break
        
        finally:
            # Complete resource tracking
            self._complete_resource_tracking(
                final_metric=training_results.get("best_metric", 0.0)
            )
            
            # Cleanup
            self._cleanup()
        
        logger.info(f"Training complete. Best metric: {training_results['best_metric']:.4f} at epoch {training_results['best_epoch']}")
        
        return training_results
    
    def _start_resource_tracking(self, steps_per_epoch: int) -> None:
        """
        Start resource tracking for the training session.
        
        Args:
            steps_per_epoch: Number of training steps per epoch.
        """
        if self.resource_manager is None:
            return
        
        try:
            task_id = self.resource_manager.start_training(
                name="VQA Training",
                total_epochs=self.config.num_epochs,
                steps_per_epoch=steps_per_epoch,
                metadata={
                    "batch_size": self.config.data.batch_size,
                    "learning_rate": self.config.optimizer.lr,
                    "model_params": count_parameters(self.model),
                    "mixed_precision": self.config.mixed_precision.value,
                }
            )
            self.state.resource_task_id = task_id
            logger.info(f"Started resource tracking with task ID: {task_id}")
        except Exception as e:
            logger.warning(f"Failed to start resource tracking: {e}")
    
    def _on_epoch_start(self, epoch: int) -> None:
        """
        Notify resource manager of epoch start.
        
        Args:
            epoch: Current epoch number.
        """
        if self.resource_manager is None or self.state.resource_task_id is None:
            return
        
        try:
            current_lr = self.optimizer.param_groups[0]["lr"]
            self.resource_manager.start_epoch(
                task_id=self.state.resource_task_id,
                epoch=epoch,
                learning_rate=current_lr,
            )
        except Exception as e:
            logger.warning(f"Failed to notify epoch start: {e}")
    
    def _on_epoch_end(
        self,
        epoch: int,
        train_loss: float,
        val_loss: Optional[float] = None,
        val_metric: Optional[float] = None,
    ) -> None:
        """
        Notify resource manager of epoch end.
        
        Args:
            epoch: Current epoch number.
            train_loss: Training loss for the epoch.
            val_loss: Optional validation loss.
            val_metric: Optional validation metric.
        """
        if self.resource_manager is None or self.state.resource_task_id is None:
            return
        
        try:
            self.resource_manager.end_epoch(
                task_id=self.state.resource_task_id,
                epoch=epoch,
                train_loss=train_loss,
                val_loss=val_loss,
                val_metric=val_metric,
            )
        except Exception as e:
            logger.warning(f"Failed to notify epoch end: {e}")
    
    def _complete_resource_tracking(
        self,
        final_metric: float = 0.0,
    ) -> None:
        """
        Complete resource tracking for the training session.
        
        Args:
            final_metric: Final metric value.
        """
        if self.resource_manager is None or self.state.resource_task_id is None:
            return
        
        try:
            self.resource_manager.complete_training(
                task_id=self.state.resource_task_id,
                final_metrics={
                    self.config.checkpoint.metric_for_best: final_metric,
                    "best_epoch": self.state.epoch,
                }
            )
            logger.info("Resource tracking completed")
        except Exception as e:
            logger.warning(f"Failed to complete resource tracking: {e}")
    
    def _cleanup(self) -> None:
        """Cleanup resources including resource manager."""
        # Stop resource manager
        if self.resource_manager is not None:
            try:
                self.resource_manager.stop()
            except Exception as e:
                logger.warning(f"Error stopping resource manager: {e}")
        
        # Close logging writers
        if "tensorboard" in self.writers:
            self.writers["tensorboard"].close()
        
        if "wandb" in self.writers:
            self.writers["wandb"].finish()


def create_trainer(
    model: nn.Module,
    config: Optional[TrainingConfig] = None,
    train_dataloader: Optional[DataLoader] = None,
    val_dataloader: Optional[DataLoader] = None,
    **kwargs
) -> VQATrainer:
    """
    Factory function to create a VQA trainer.
    
    Args:
        model: VQA model to train
        config: Training configuration
        train_dataloader: Training data loader
        val_dataloader: Validation data loader
        **kwargs: Additional trainer arguments
        
    Returns:
        Initialized VQATrainer instance
    """
    return VQATrainer(
        model=model,
        config=config,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        **kwargs
    )
