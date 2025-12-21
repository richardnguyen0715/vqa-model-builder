"""
Checkpoint Manager Module
Provides comprehensive checkpoint saving, loading, and management functionality.
"""

import os
import json
import shutil
import logging
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

import torch
import torch.nn as nn

from src.middleware.config_loader import get_config

logger = logging.getLogger(__name__)


class CheckpointManager:
    """
    Manages model checkpoint saving, loading, and retention.
    
    Features:
    - Save checkpoints at specified intervals
    - Track and save best model based on metrics
    - Automatic checkpoint cleanup based on retention policy
    - Resume training from checkpoint
    - Save on interrupt handling
    
    Attributes:
        save_dir: Directory for saving checkpoints.
        prefix: Checkpoint filename prefix.
        max_keep: Maximum number of checkpoints to keep.
        best_metric: Best metric value seen.
        best_mode: Whether to maximize or minimize metric.
    """
    
    def __init__(
        self,
        save_dir: Optional[str] = None,
        prefix: Optional[str] = None,
        max_keep: Optional[int] = None,
        metric_name: Optional[str] = None,
        metric_mode: Optional[str] = None,
    ) -> None:
        """
        Initialize checkpoint manager.
        
        Args:
            save_dir: Directory for checkpoints (from config if None).
            prefix: Checkpoint filename prefix (from config if None).
            max_keep: Maximum checkpoints to keep (from config if None).
            metric_name: Metric for best model selection (from config if None).
            metric_mode: Metric mode: max or min (from config if None).
        """
        # Load from config with provided overrides
        self.save_dir = Path(save_dir or get_config("checkpoint", "save.base_dir", "checkpoints"))
        self.prefix = prefix or get_config("checkpoint", "save.prefix", "vqa_model")
        self.max_keep = max_keep or get_config("checkpoint", "save.max_keep", 3)
        self.metric_name = metric_name or get_config("checkpoint", "best_model.metric", "accuracy")
        self.metric_mode = metric_mode or get_config("checkpoint", "best_model.mode", "max")
        
        # Configuration flags
        self.save_optimizer = get_config("checkpoint", "save.save_optimizer", True)
        self.save_scheduler = get_config("checkpoint", "save.save_scheduler", True)
        self.save_scaler = get_config("checkpoint", "save.save_scaler", True)
        self.save_early_stopping = get_config("checkpoint", "save.save_early_stopping", True)
        self.save_training_state = get_config("checkpoint", "save.save_training_state", True)
        self.save_config = get_config("checkpoint", "save.save_config", True)
        self.keep_best = get_config("checkpoint", "save.keep_best", True)
        
        # Periodic save settings
        self.save_every_k_epochs = get_config("checkpoint", "save.save_every_k_epochs", 5)
        self.save_every_k_steps = get_config("checkpoint", "save.save_every_k_steps", 0)
        
        # State tracking
        self.best_metric: Optional[float] = None
        self.checkpoint_history: List[Dict[str, Any]] = []
        
        # Create save directory
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"CheckpointManager initialized. Save dir: {self.save_dir}")
    
    def _is_better(self, current: float, best: float) -> bool:
        """
        Check if current metric is better than best.
        
        Args:
            current: Current metric value.
            best: Best metric value.
            
        Returns:
            True if current is better than best.
        """
        if self.metric_mode == "max":
            return current > best
        return current < best
    
    def _get_checkpoint_path(self, name: str) -> Path:
        """
        Get full path for checkpoint file.
        
        Args:
            name: Checkpoint filename.
            
        Returns:
            Full path to checkpoint file.
        """
        return self.save_dir / name
    
    def _generate_checkpoint_name(
        self,
        epoch: Optional[int] = None,
        step: Optional[int] = None,
        is_best: bool = False,
        suffix: str = "",
    ) -> str:
        """
        Generate checkpoint filename.
        
        Args:
            epoch: Current epoch number.
            step: Current step number.
            is_best: Whether this is best checkpoint.
            suffix: Additional suffix for filename.
            
        Returns:
            Generated checkpoint filename.
        """
        if is_best:
            return get_config("checkpoint", "best_model.filename", "best_model.pt")
        
        timestamp = datetime.now().strftime(
            get_config("checkpoint", "save.timestamp_format", "%Y%m%d_%H%M%S")
        )
        
        parts = [self.prefix]
        if epoch is not None:
            parts.append(f"epoch{epoch}")
        if step is not None:
            parts.append(f"step{step}")
        parts.append(timestamp)
        if suffix:
            parts.append(suffix)
        
        return "_".join(parts) + ".pt"
    
    def _create_checkpoint_dict(
        self,
        model: nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        scaler: Optional[Any] = None,
        early_stopping: Optional[Any] = None,
        epoch: int = 0,
        global_step: int = 0,
        metrics: Optional[Dict[str, float]] = None,
        config: Optional[Dict[str, Any]] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Create checkpoint dictionary.
        
        Args:
            model: Model to save.
            optimizer: Optimizer state.
            scheduler: Scheduler state.
            scaler: Gradient scaler state.
            early_stopping: Early stopping state.
            epoch: Current epoch.
            global_step: Current global step.
            metrics: Current metrics.
            config: Model and training config.
            extra: Additional data to save.
            
        Returns:
            Checkpoint dictionary.
        """
        checkpoint = {
            "model_state_dict": model.state_dict(),
        }
        
        if self.save_optimizer and optimizer is not None:
            checkpoint["optimizer_state_dict"] = optimizer.state_dict()
        
        if self.save_scheduler and scheduler is not None:
            checkpoint["scheduler_state_dict"] = scheduler.state_dict()
        
        if self.save_scaler and scaler is not None:
            checkpoint["scaler_state_dict"] = scaler.state_dict()
        
        if self.save_early_stopping and early_stopping is not None:
            if hasattr(early_stopping, "state_dict"):
                checkpoint["early_stopping_state"] = early_stopping.state_dict()
        
        if self.save_training_state:
            checkpoint["training_state"] = {
                "epoch": epoch,
                "global_step": global_step,
                "best_metric": self.best_metric,
                "metrics": metrics or {},
            }
        
        if self.save_config and config is not None:
            checkpoint["config"] = config
        
        # Metadata
        checkpoint["metadata"] = {
            "timestamp": datetime.now().isoformat(),
            "pytorch_version": torch.__version__,
            "metric_name": self.metric_name,
            "metric_mode": self.metric_mode,
        }
        
        if extra:
            checkpoint["extra"] = extra
        
        return checkpoint
    
    def save(
        self,
        model: nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        scaler: Optional[Any] = None,
        early_stopping: Optional[Any] = None,
        epoch: int = 0,
        global_step: int = 0,
        metrics: Optional[Dict[str, float]] = None,
        config: Optional[Dict[str, Any]] = None,
        extra: Optional[Dict[str, Any]] = None,
        is_best: bool = False,
        suffix: str = "",
    ) -> Path:
        """
        Save checkpoint.
        
        Args:
            model: Model to save.
            optimizer: Optimizer state.
            scheduler: Scheduler state.
            scaler: Gradient scaler state.
            early_stopping: Early stopping state.
            epoch: Current epoch.
            global_step: Current global step.
            metrics: Current metrics.
            config: Model and training config.
            extra: Additional data to save.
            is_best: Whether this is best checkpoint.
            suffix: Additional suffix for filename.
            
        Returns:
            Path to saved checkpoint.
        """
        checkpoint = self._create_checkpoint_dict(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            early_stopping=early_stopping,
            epoch=epoch,
            global_step=global_step,
            metrics=metrics,
            config=config,
            extra=extra,
        )
        
        filename = self._generate_checkpoint_name(
            epoch=epoch,
            step=global_step,
            is_best=is_best,
            suffix=suffix,
        )
        
        filepath = self._get_checkpoint_path(filename)
        torch.save(checkpoint, filepath)
        
        # Track checkpoint
        self.checkpoint_history.append({
            "path": str(filepath),
            "epoch": epoch,
            "step": global_step,
            "metrics": metrics,
            "is_best": is_best,
            "timestamp": datetime.now().isoformat(),
        })
        
        logger.info(f"Saved checkpoint: {filepath}")
        
        # Cleanup old checkpoints
        if not is_best:
            self._cleanup_checkpoints()
        
        return filepath
    
    def save_best(
        self,
        model: nn.Module,
        current_metric: float,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        scaler: Optional[Any] = None,
        early_stopping: Optional[Any] = None,
        epoch: int = 0,
        global_step: int = 0,
        metrics: Optional[Dict[str, float]] = None,
        config: Optional[Dict[str, Any]] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> Optional[Path]:
        """
        Save checkpoint if current metric is best.
        
        Args:
            model: Model to save.
            current_metric: Current metric value.
            optimizer: Optimizer state.
            scheduler: Scheduler state.
            scaler: Gradient scaler state.
            early_stopping: Early stopping state.
            epoch: Current epoch.
            global_step: Current global step.
            metrics: Current metrics.
            config: Model and training config.
            extra: Additional data to save.
            
        Returns:
            Path to saved checkpoint if best, None otherwise.
        """
        is_best = False
        
        if self.best_metric is None:
            is_best = True
            self.best_metric = current_metric
        elif self._is_better(current_metric, self.best_metric):
            is_best = True
            self.best_metric = current_metric
        
        if is_best:
            logger.info(f"New best {self.metric_name}: {current_metric:.4f}")
            return self.save(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                early_stopping=early_stopping,
                epoch=epoch,
                global_step=global_step,
                metrics=metrics,
                config=config,
                extra=extra,
                is_best=True,
            )
        
        return None
    
    def should_save_periodic(self, epoch: int, global_step: int) -> bool:
        """
        Check if periodic checkpoint should be saved.
        
        Args:
            epoch: Current epoch.
            global_step: Current global step.
            
        Returns:
            True if checkpoint should be saved.
        """
        if self.save_every_k_epochs > 0 and epoch > 0:
            if epoch % self.save_every_k_epochs == 0:
                return True
        
        if self.save_every_k_steps > 0 and global_step > 0:
            if global_step % self.save_every_k_steps == 0:
                return True
        
        return False
    
    def _cleanup_checkpoints(self) -> None:
        """Remove old checkpoints based on retention policy."""
        if self.max_keep <= 0:
            return
        
        # Get non-best checkpoints
        regular_checkpoints = [
            entry for entry in self.checkpoint_history
            if not entry.get("is_best", False)
        ]
        
        # Remove oldest checkpoints if exceeding max_keep
        while len(regular_checkpoints) > self.max_keep:
            oldest = regular_checkpoints.pop(0)
            checkpoint_path = Path(oldest["path"])
            
            if checkpoint_path.exists():
                checkpoint_path.unlink()
                logger.debug(f"Removed old checkpoint: {checkpoint_path}")
            
            self.checkpoint_history.remove(oldest)
    
    def load(
        self,
        checkpoint_path: Union[str, Path],
        model: nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        scaler: Optional[Any] = None,
        early_stopping: Optional[Any] = None,
        strict: bool = True,
        load_optimizer: Optional[bool] = None,
        load_scheduler: Optional[bool] = None,
        load_scaler: Optional[bool] = None,
        load_early_stopping: Optional[bool] = None,
        map_location: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Load checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint file.
            model: Model to load weights into.
            optimizer: Optimizer to load state into.
            scheduler: Scheduler to load state into.
            scaler: Gradient scaler to load state into.
            early_stopping: Early stopping to load state into.
            strict: Whether to strictly enforce state_dict key matching.
            load_optimizer: Whether to load optimizer (from config if None).
            load_scheduler: Whether to load scheduler (from config if None).
            load_scaler: Whether to load scaler (from config if None).
            load_early_stopping: Whether to load early stopping (from config if None).
            map_location: Device to map tensors to.
            
        Returns:
            Dictionary with training state and extra data.
        """
        checkpoint_path = Path(checkpoint_path)
        
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        # Determine device mapping
        if map_location is None:
            if torch.cuda.is_available():
                map_location = "cuda"
            else:
                map_location = "cpu"
        
        checkpoint = torch.load(checkpoint_path, map_location=map_location)
        
        # Load model weights
        model.load_state_dict(checkpoint["model_state_dict"], strict=strict)
        logger.info(f"Loaded model weights from {checkpoint_path}")
        
        # Load optimizer
        load_opt = load_optimizer if load_optimizer is not None else get_config("checkpoint", "load.load_optimizer", True)
        if load_opt and optimizer is not None and "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            logger.debug("Loaded optimizer state")
        
        # Load scheduler
        load_sched = load_scheduler if load_scheduler is not None else get_config("checkpoint", "load.load_scheduler", True)
        if load_sched and scheduler is not None and "scheduler_state_dict" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            logger.debug("Loaded scheduler state")
        
        # Load scaler
        load_sc = load_scaler if load_scaler is not None else get_config("checkpoint", "load.load_scaler", True)
        if load_sc and scaler is not None and "scaler_state_dict" in checkpoint:
            scaler.load_state_dict(checkpoint["scaler_state_dict"])
            logger.debug("Loaded gradient scaler state")
        
        # Load early stopping
        load_es = load_early_stopping if load_early_stopping is not None else get_config("checkpoint", "load.load_early_stopping", True)
        if load_es and early_stopping is not None and "early_stopping_state" in checkpoint:
            if hasattr(early_stopping, "load_state_dict"):
                early_stopping.load_state_dict(checkpoint["early_stopping_state"])
                logger.debug("Loaded early stopping state")
        
        # Update best metric
        training_state = checkpoint.get("training_state", {})
        if "best_metric" in training_state:
            self.best_metric = training_state["best_metric"]
        
        return {
            "training_state": training_state,
            "config": checkpoint.get("config"),
            "metadata": checkpoint.get("metadata"),
            "extra": checkpoint.get("extra"),
        }
    
    def load_for_inference(
        self,
        checkpoint_path: Union[str, Path],
        model: nn.Module,
        map_location: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Load checkpoint for inference only (weights only).
        
        Args:
            checkpoint_path: Path to checkpoint file.
            model: Model to load weights into.
            map_location: Device to map tensors to.
            
        Returns:
            Dictionary with config and metadata.
        """
        return self.load(
            checkpoint_path=checkpoint_path,
            model=model,
            strict=get_config("checkpoint", "load.strict_loading", True),
            load_optimizer=False,
            load_scheduler=False,
            load_scaler=False,
            load_early_stopping=False,
            map_location=map_location,
        )
    
    def get_best_checkpoint_path(self) -> Optional[Path]:
        """
        Get path to best checkpoint.
        
        Returns:
            Path to best checkpoint if exists, None otherwise.
        """
        best_filename = get_config("checkpoint", "best_model.filename", "best_model.pt")
        best_path = self._get_checkpoint_path(best_filename)
        
        if best_path.exists():
            return best_path
        return None
    
    def get_latest_checkpoint_path(self) -> Optional[Path]:
        """
        Get path to most recent checkpoint.
        
        Returns:
            Path to latest checkpoint if exists, None otherwise.
        """
        if not self.checkpoint_history:
            # Scan directory for checkpoints
            checkpoints = list(self.save_dir.glob(f"{self.prefix}*.pt"))
            if not checkpoints:
                return None
            
            # Return most recently modified
            return max(checkpoints, key=lambda p: p.stat().st_mtime)
        
        # Return from history
        return Path(self.checkpoint_history[-1]["path"])
    
    def list_checkpoints(self) -> List[Dict[str, Any]]:
        """
        List all checkpoints.
        
        Returns:
            List of checkpoint information dictionaries.
        """
        checkpoints = []
        
        for checkpoint_path in self.save_dir.glob("*.pt"):
            try:
                # Load just metadata
                checkpoint = torch.load(checkpoint_path, map_location="cpu")
                
                checkpoints.append({
                    "path": str(checkpoint_path),
                    "filename": checkpoint_path.name,
                    "training_state": checkpoint.get("training_state", {}),
                    "metadata": checkpoint.get("metadata", {}),
                })
            except Exception as e:
                logger.warning(f"Could not read checkpoint {checkpoint_path}: {e}")
        
        return sorted(checkpoints, key=lambda x: x.get("metadata", {}).get("timestamp", ""))


# =============================================================================
# Convenience Functions
# =============================================================================

def save_checkpoint(
    model: nn.Module,
    path: Union[str, Path],
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    epoch: int = 0,
    global_step: int = 0,
    metrics: Optional[Dict[str, float]] = None,
    config: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Simple function to save a checkpoint.
    
    Args:
        model: Model to save.
        path: Path to save checkpoint.
        optimizer: Optional optimizer to save.
        scheduler: Optional scheduler to save.
        epoch: Current epoch.
        global_step: Current global step.
        metrics: Current metrics.
        config: Model configuration.
    """
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "training_state": {
            "epoch": epoch,
            "global_step": global_step,
            "metrics": metrics or {},
        },
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "pytorch_version": torch.__version__,
        },
    }
    
    if optimizer is not None:
        checkpoint["optimizer_state_dict"] = optimizer.state_dict()
    
    if scheduler is not None:
        checkpoint["scheduler_state_dict"] = scheduler.state_dict()
    
    if config is not None:
        checkpoint["config"] = config
    
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, path)
    
    logger.info(f"Saved checkpoint: {path}")


def load_checkpoint(
    path: Union[str, Path],
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    strict: bool = True,
    map_location: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Simple function to load a checkpoint.
    
    Args:
        path: Path to checkpoint.
        model: Model to load weights into.
        optimizer: Optional optimizer to load state into.
        scheduler: Optional scheduler to load state into.
        strict: Whether to strictly match state_dict keys.
        map_location: Device to map tensors to.
        
    Returns:
        Dictionary with training state and config.
    """
    path = Path(path)
    
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    
    if map_location is None:
        map_location = "cuda" if torch.cuda.is_available() else "cpu"
    
    checkpoint = torch.load(path, map_location=map_location)
    
    model.load_state_dict(checkpoint["model_state_dict"], strict=strict)
    
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    
    if scheduler is not None and "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    
    logger.info(f"Loaded checkpoint: {path}")
    
    return {
        "training_state": checkpoint.get("training_state", {}),
        "config": checkpoint.get("config"),
        "metadata": checkpoint.get("metadata"),
    }


# =============================================================================
# Module Exports
# =============================================================================

__all__ = [
    "CheckpointManager",
    "save_checkpoint",
    "load_checkpoint",
]
