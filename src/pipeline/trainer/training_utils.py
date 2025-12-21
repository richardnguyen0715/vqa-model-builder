"""
Training Utilities Module
Provides utility functions for VQA model training.
"""

import os
import random
import logging
from typing import Optional, Dict, Any, Tuple, List
from dataclasses import dataclass

import torch
import torch.nn as nn
import numpy as np

logger = logging.getLogger(__name__)


def set_seed(seed: int, deterministic: bool = False) -> None:
    """
    Set random seed for reproducibility.
    
    Args:
        seed: Random seed value
        deterministic: Whether to use deterministic algorithms
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        if hasattr(torch, "use_deterministic_algorithms"):
            torch.use_deterministic_algorithms(True)
    else:
        torch.backends.cudnn.benchmark = True
    
    logger.info(f"Set random seed to {seed}, deterministic={deterministic}")


def get_device() -> torch.device:
    """
    Get the best available device.
    
    Returns:
        torch.device for training
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info(f"Using CUDA: {torch.cuda.get_device_name(0)}")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
        logger.info("Using Apple MPS")
    else:
        device = torch.device("cpu")
        logger.info("Using CPU")
    
    return device


def count_parameters(model: nn.Module, trainable_only: bool = True) -> int:
    """
    Count model parameters.
    
    Args:
        model: PyTorch model
        trainable_only: Whether to count only trainable parameters
        
    Returns:
        Number of parameters
    """
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


def get_parameter_groups(
    model: nn.Module,
    base_lr: float,
    weight_decay: float,
    layer_decay: Optional[float] = None,
    no_decay_keywords: List[str] = None
) -> List[Dict[str, Any]]:
    """
    Get parameter groups with optional layer-wise decay.
    
    Args:
        model: PyTorch model
        base_lr: Base learning rate
        weight_decay: Weight decay coefficient
        layer_decay: Layer-wise decay factor
        no_decay_keywords: Keywords for parameters without decay
        
    Returns:
        List of parameter group dictionaries
    """
    if no_decay_keywords is None:
        no_decay_keywords = ["bias", "LayerNorm", "layernorm", "bn", "BatchNorm"]
    
    if layer_decay is None or layer_decay >= 1.0:
        # Standard parameter groups without layer decay
        decay_params = []
        no_decay_params = []
        
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            
            if any(kw in name for kw in no_decay_keywords):
                no_decay_params.append(param)
            else:
                decay_params.append(param)
        
        return [
            {"params": decay_params, "lr": base_lr, "weight_decay": weight_decay},
            {"params": no_decay_params, "lr": base_lr, "weight_decay": 0.0}
        ]
    
    # Layer-wise learning rate decay
    parameter_groups = {}
    num_layers = 0
    
    # Count layers (assuming encoder structure)
    for name, _ in model.named_parameters():
        if "encoder.layer" in name or "layers." in name:
            layer_id = int(name.split(".layer")[-1].split(".")[0]) if "encoder.layer" in name else \
                       int(name.split("layers.")[-1].split(".")[0])
            num_layers = max(num_layers, layer_id + 1)
    
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        
        # Determine layer id
        if "encoder.layer" in name:
            layer_id = int(name.split(".layer")[-1].split(".")[0])
        elif "layers." in name:
            layer_id = int(name.split("layers.")[-1].split(".")[0])
        elif "embeddings" in name:
            layer_id = 0
        else:
            layer_id = num_layers  # Top layers
        
        # Calculate learning rate
        lr_scale = layer_decay ** (num_layers - layer_id)
        lr = base_lr * lr_scale
        
        # Determine weight decay
        wd = 0.0 if any(kw in name for kw in no_decay_keywords) else weight_decay
        
        group_key = (lr, wd)
        if group_key not in parameter_groups:
            parameter_groups[group_key] = []
        parameter_groups[group_key].append(param)
    
    return [
        {"params": params, "lr": lr, "weight_decay": wd}
        for (lr, wd), params in parameter_groups.items()
    ]


@dataclass
class GradientAccumulator:
    """
    Gradient accumulation helper.
    
    Attributes:
        accumulation_steps: Number of accumulation steps
        current_step: Current accumulation step
    """
    accumulation_steps: int = 1
    current_step: int = 0
    
    def should_step(self) -> bool:
        """Check if optimizer step should be performed."""
        self.current_step += 1
        if self.current_step >= self.accumulation_steps:
            self.current_step = 0
            return True
        return False
    
    def reset(self) -> None:
        """Reset accumulation counter."""
        self.current_step = 0


class EarlyStopping:
    """
    Early stopping handler.
    
    Attributes:
        patience: Number of epochs without improvement
        min_delta: Minimum change to qualify as improvement
        mode: Whether to maximize or minimize metric
        counter: Current patience counter
        best_score: Best metric value seen
        should_stop: Whether training should stop
    """
    
    def __init__(
        self,
        patience: int = 5,
        min_delta: float = 0.0,
        mode: str = "max"
    ):
        """
        Initialize early stopping.
        
        Args:
            patience: Number of epochs without improvement
            min_delta: Minimum change to qualify as improvement
            mode: Whether to maximize ('max') or minimize ('min') metric
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.should_stop = False
    
    def __call__(self, score: float) -> bool:
        """
        Check if training should stop.
        
        Args:
            score: Current metric value
            
        Returns:
            True if training should stop
        """
        if self.best_score is None:
            self.best_score = score
            return False
        
        if self.mode == "max":
            improved = score > self.best_score + self.min_delta
        else:
            improved = score < self.best_score - self.min_delta
        
        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
                logger.info(f"Early stopping triggered after {self.counter} epochs")
        
        return self.should_stop
    
    def state_dict(self) -> Dict[str, Any]:
        """Get state dictionary."""
        return {
            "counter": self.counter,
            "best_score": self.best_score,
            "should_stop": self.should_stop
        }
    
    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Load state dictionary."""
        self.counter = state_dict["counter"]
        self.best_score = state_dict["best_score"]
        self.should_stop = state_dict["should_stop"]


class AverageMeter:
    """
    Computes and stores average and current values.
    
    Attributes:
        name: Meter name
        fmt: Format string
        val: Current value
        avg: Running average
        sum: Running sum
        count: Number of updates
    """
    
    def __init__(self, name: str = "", fmt: str = ":f"):
        """
        Initialize meter.
        
        Args:
            name: Meter name
            fmt: Format string
        """
        self.name = name
        self.fmt = fmt
        self.reset()
    
    def reset(self) -> None:
        """Reset meter values."""
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val: float, n: int = 1) -> None:
        """
        Update meter with new value.
        
        Args:
            val: Value to add
            n: Number of samples
        """
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count > 0 else 0
    
    def __str__(self) -> str:
        """Get string representation."""
        fmtstr = f"{{name}} {{val{self.fmt}}} ({{avg{self.fmt}}})"
        return fmtstr.format(**self.__dict__)


class ProgressTracker:
    """
    Training progress tracker.
    
    Tracks loss, metrics, and other statistics during training.
    """
    
    def __init__(self, metric_names: List[str] = None):
        """
        Initialize progress tracker.
        
        Args:
            metric_names: List of metric names to track
        """
        self.metric_names = metric_names or ["loss", "accuracy"]
        self.meters = {name: AverageMeter(name) for name in self.metric_names}
        self.history = {name: [] for name in self.metric_names}
    
    def update(self, values: Dict[str, float], n: int = 1) -> None:
        """
        Update trackers with new values.
        
        Args:
            values: Dictionary of metric values
            n: Number of samples
        """
        for name, value in values.items():
            if name in self.meters:
                self.meters[name].update(value, n)
    
    def reset(self) -> None:
        """Reset all meters for new epoch."""
        for meter in self.meters.values():
            meter.reset()
    
    def epoch_end(self) -> Dict[str, float]:
        """
        End epoch and record history.
        
        Returns:
            Dictionary of epoch averages
        """
        results = {}
        for name, meter in self.meters.items():
            results[name] = meter.avg
            self.history[name].append(meter.avg)
        return results
    
    def get_current(self) -> Dict[str, float]:
        """Get current meter values."""
        return {name: meter.val for name, meter in self.meters.items()}
    
    def get_average(self) -> Dict[str, float]:
        """Get average meter values."""
        return {name: meter.avg for name, meter in self.meters.items()}


def freeze_module(module: nn.Module) -> None:
    """
    Freeze module parameters.
    
    Args:
        module: Module to freeze
    """
    for param in module.parameters():
        param.requires_grad = False


def unfreeze_module(module: nn.Module) -> None:
    """
    Unfreeze module parameters.
    
    Args:
        module: Module to unfreeze
    """
    for param in module.parameters():
        param.requires_grad = True


def apply_training_strategy(
    model: nn.Module,
    strategy: str,
    current_epoch: int = 0,
    total_epochs: int = 10,
    unfreeze_schedule: Dict[int, List[str]] = None
) -> None:
    """
    Apply training strategy to model.
    
    Args:
        model: Model to apply strategy to
        strategy: Strategy name (full, freeze_visual, freeze_text, linear_probe, gradual_unfreeze)
        current_epoch: Current training epoch
        total_epochs: Total training epochs
        unfreeze_schedule: Schedule for gradual unfreezing
    """
    if strategy == "full":
        unfreeze_module(model)
    
    elif strategy == "freeze_visual":
        if hasattr(model, "visual_encoder"):
            freeze_module(model.visual_encoder)
    
    elif strategy == "freeze_text":
        if hasattr(model, "text_encoder"):
            freeze_module(model.text_encoder)
    
    elif strategy == "linear_probe":
        freeze_module(model)
        if hasattr(model, "answer_head"):
            unfreeze_module(model.answer_head)
    
    elif strategy == "gradual_unfreeze":
        if unfreeze_schedule is None:
            # Default gradual unfreeze: answer head -> fusion -> encoders
            unfreeze_fraction = current_epoch / total_epochs
            
            if hasattr(model, "answer_head"):
                unfreeze_module(model.answer_head)
            
            if unfreeze_fraction >= 0.3:
                if hasattr(model, "fusion"):
                    unfreeze_module(model.fusion)
            
            if unfreeze_fraction >= 0.6:
                unfreeze_module(model)
        else:
            for epoch, module_names in unfreeze_schedule.items():
                if current_epoch >= epoch:
                    for name in module_names:
                        if hasattr(model, name):
                            unfreeze_module(getattr(model, name))
    
    logger.info(f"Applied training strategy: {strategy}, trainable params: {count_parameters(model):,}")


def clip_gradients(
    model: nn.Module,
    max_norm: float,
    norm_type: float = 2.0
) -> float:
    """
    Clip gradients by norm.
    
    Args:
        model: Model with gradients
        max_norm: Maximum gradient norm
        norm_type: Norm type
        
    Returns:
        Total gradient norm before clipping
    """
    if max_norm <= 0:
        return 0.0
    
    parameters = [p for p in model.parameters() if p.grad is not None]
    if len(parameters) == 0:
        return 0.0
    
    total_norm = torch.nn.utils.clip_grad_norm_(parameters, max_norm, norm_type)
    return total_norm.item()


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    epoch: int,
    step: int,
    metrics: Dict[str, float],
    path: str,
    early_stopping: Optional[EarlyStopping] = None
) -> None:
    """
    Save training checkpoint.
    
    Args:
        model: Model to save
        optimizer: Optimizer state
        scheduler: Scheduler state
        epoch: Current epoch
        step: Current global step
        metrics: Current metrics
        path: Path to save checkpoint
        early_stopping: Optional early stopping state
    """
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
        "epoch": epoch,
        "step": step,
        "metrics": metrics
    }
    
    if early_stopping:
        checkpoint["early_stopping_state_dict"] = early_stopping.state_dict()
    
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(checkpoint, path)
    logger.info(f"Saved checkpoint to {path}")


def load_checkpoint(
    path: str,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Any = None,
    early_stopping: Optional[EarlyStopping] = None,
    strict: bool = True
) -> Tuple[int, int, Dict[str, float]]:
    """
    Load training checkpoint.
    
    Args:
        path: Path to checkpoint
        model: Model to load weights into
        optimizer: Optional optimizer to load state
        scheduler: Optional scheduler to load state
        early_stopping: Optional early stopping to load state
        strict: Whether to strictly enforce state dict matching
        
    Returns:
        Tuple of (epoch, step, metrics)
    """
    checkpoint = torch.load(path, map_location="cpu")
    
    model.load_state_dict(checkpoint["model_state_dict"], strict=strict)
    
    if optimizer and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    
    if scheduler and checkpoint.get("scheduler_state_dict"):
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    
    if early_stopping and checkpoint.get("early_stopping_state_dict"):
        early_stopping.load_state_dict(checkpoint["early_stopping_state_dict"])
    
    epoch = checkpoint.get("epoch", 0)
    step = checkpoint.get("step", 0)
    metrics = checkpoint.get("metrics", {})
    
    logger.info(f"Loaded checkpoint from {path} (epoch={epoch}, step={step})")
    
    return epoch, step, metrics
