"""
VQA Trainer Module
Provides the main trainer class for Vietnamese VQA models.
"""

import os
import time
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
    """
    epoch: int = 0
    global_step: int = 0
    best_metric: float = 0.0
    is_best: bool = False


class VQATrainer:
    """
    Vietnamese VQA model trainer.
    
    Provides complete training loop with:
    - Mixed precision training
    - Gradient accumulation
    - Learning rate scheduling
    - Early stopping
    - Checkpointing
    - Logging (TensorBoard, W&B)
    
    Attributes:
        model: VQA model to train
        config: Training configuration
        device: Training device
        optimizer: Model optimizer
        scheduler: Learning rate scheduler
        scaler: Gradient scaler for mixed precision
        early_stopping: Early stopping handler
        state: Current training state
    """
    
    def __init__(
        self,
        model: nn.Module,
        config: Optional[TrainingConfig] = None,
        train_dataloader: Optional[DataLoader] = None,
        val_dataloader: Optional[DataLoader] = None,
        loss_fn: Optional[Callable] = None,
        metrics_fn: Optional[Callable] = None
    ):
        """
        Initialize trainer.
        
        Args:
            model: VQA model to train
            config: Training configuration
            train_dataloader: Training data loader
            val_dataloader: Validation data loader
            loss_fn: Loss function (receives model output and targets)
            metrics_fn: Metrics function (receives predictions and targets)
        """
        self.config = config or get_default_training_config()
        self.device = get_device()
        
        # Set seed for reproducibility
        set_seed(self.config.seed, self.config.deterministic)
        
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
        
        # Gradient accumulation
        self.grad_accumulator = GradientAccumulator(
            self.config.gradient_accumulation_steps
        )
        
        # Progress tracking
        self.train_tracker = ProgressTracker(["loss", "accuracy"])
        self.val_tracker = ProgressTracker(["loss", "accuracy"])
        
        # Training state
        self.state = TrainingState()
        
        # Logging
        self.writers = self._setup_logging()
        
        # Resume from checkpoint if specified
        if self.config.checkpoint.resume_from:
            self._resume_from_checkpoint()
        
        logger.info(f"Initialized trainer with {count_parameters(self.model):,} trainable parameters")
    
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
    
    def _resume_from_checkpoint(self) -> None:
        """Resume training from checkpoint."""
        path = self.config.checkpoint.resume_from
        
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
            self.state.best_metric = metrics.get(self.config.checkpoint.metric_for_best, 0.0)
    
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
        
        return step_metrics
    
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
        """Save checkpoint."""
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
        if self.state.epoch % ckpt_config.save_interval == 0:
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
        
        for epoch in range(self.state.epoch, self.config.num_epochs):
            self.state.epoch = epoch
            
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
        
        # Cleanup
        self._cleanup()
        
        logger.info(f"Training complete. Best metric: {training_results['best_metric']:.4f} at epoch {training_results['best_epoch']}")
        
        return training_results
    
    def _cleanup(self) -> None:
        """Cleanup resources."""
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
