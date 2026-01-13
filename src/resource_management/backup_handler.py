"""
Backup Handler Module

Provides automatic checkpoint backup functionality when resource thresholds
are exceeded. Manages emergency backups, rotation, and cleanup.
"""

import gzip
import logging
import os
import shutil
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

import torch
import torch.nn as nn

from .resource_config import (
    BackupConfig,
    ResourceConfig,
    ThresholdAction,
    load_resource_config,
)
from .resource_monitor import ResourceSnapshot, AggregatedMetrics


logger = logging.getLogger(__name__)


# =============================================================================
# Enumerations
# =============================================================================

class BackupType(str, Enum):
    """Types of backups that can be created."""
    
    SCHEDULED = "scheduled"
    EMERGENCY = "emergency"
    WARNING_TRIGGER = "warning_trigger"
    CRITICAL_TRIGGER = "critical_trigger"
    MANUAL = "manual"
    SHUTDOWN = "shutdown"


class BackupStatus(str, Enum):
    """Status of a backup operation."""
    
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class BackupInfo:
    """
    Information about a created backup.
    
    Attributes:
        backup_id: Unique identifier for the backup.
        backup_type: Type of backup.
        status: Current backup status.
        filepath: Path to backup file.
        timestamp: When backup was created.
        size_bytes: Size of backup file in bytes.
        trigger_reason: Reason for triggering backup.
        model_state_included: Whether model state is included.
        optimizer_state_included: Whether optimizer state is included.
        scheduler_state_included: Whether scheduler state is included.
        metadata: Additional backup metadata.
    """
    
    backup_id: str
    backup_type: BackupType
    status: BackupStatus = BackupStatus.PENDING
    filepath: Optional[str] = None
    timestamp: Optional[datetime] = None
    size_bytes: int = 0
    trigger_reason: str = ""
    model_state_included: bool = True
    optimizer_state_included: bool = False
    scheduler_state_included: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert backup info to dictionary.
        
        Returns:
            Dictionary representation of backup info.
        """
        return {
            "backup_id": self.backup_id,
            "backup_type": self.backup_type.value,
            "status": self.status.value,
            "filepath": self.filepath,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "size_bytes": self.size_bytes,
            "trigger_reason": self.trigger_reason,
            "model_state_included": self.model_state_included,
            "optimizer_state_included": self.optimizer_state_included,
            "scheduler_state_included": self.scheduler_state_included,
            "metadata": self.metadata,
        }


@dataclass
class BackupState:
    """
    State to be saved in a backup.
    
    Attributes:
        model_state: Model state dictionary.
        optimizer_state: Optimizer state dictionary.
        scheduler_state: Scheduler state dictionary.
        epoch: Current epoch number.
        step: Current step number.
        metrics: Current metrics.
        config: Training configuration.
        additional: Additional state data.
    """
    
    model_state: Optional[Dict[str, Any]] = None
    optimizer_state: Optional[Dict[str, Any]] = None
    scheduler_state: Optional[Dict[str, Any]] = None
    epoch: int = 0
    step: int = 0
    metrics: Dict[str, float] = field(default_factory=dict)
    config: Dict[str, Any] = field(default_factory=dict)
    additional: Dict[str, Any] = field(default_factory=dict)


# =============================================================================
# Backup Handler
# =============================================================================

class BackupHandler:
    """
    Handles checkpoint backup operations.
    
    Creates, manages, and rotates backup files based on
    configuration and resource threshold triggers.
    """
    
    def __init__(
        self,
        config: Optional[BackupConfig] = None,
    ) -> None:
        """
        Initialize backup handler.
        
        Args:
            config: Backup configuration. Uses defaults if not provided.
        """
        if config is None:
            resource_config = load_resource_config()
            config = resource_config.backup
        
        self.config = config
        self._backups: List[BackupInfo] = []
        self._backup_counter = 0
        self._lock = threading.Lock()
        self._callbacks: List[Callable[[BackupInfo], None]] = []
        
        # Model and optimizer references
        self._model: Optional[nn.Module] = None
        self._optimizer: Optional[torch.optim.Optimizer] = None
        self._scheduler: Optional[Any] = None
        self._state_provider: Optional[Callable[[], BackupState]] = None
        
        # Ensure backup directory exists
        self._ensure_backup_dir()
    
    def _ensure_backup_dir(self) -> None:
        """Ensure backup directory exists."""
        backup_dir = Path(self.config.backup_dir)
        backup_dir.mkdir(parents=True, exist_ok=True)
    
    def register_model(
        self,
        model: nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
    ) -> None:
        """
        Register model and training components for backup.
        
        Args:
            model: PyTorch model to backup.
            optimizer: Optional optimizer to include in backup.
            scheduler: Optional scheduler to include in backup.
        """
        self._model = model
        self._optimizer = optimizer
        self._scheduler = scheduler
        
        logger.info("Registered model for backup")
    
    def register_state_provider(
        self,
        provider: Callable[[], BackupState]
    ) -> None:
        """
        Register a state provider function for backups.
        
        Args:
            provider: Function that returns current BackupState.
        """
        self._state_provider = provider
        logger.info("Registered state provider for backup")
    
    def _generate_backup_id(self, backup_type: BackupType) -> str:
        """
        Generate unique backup ID.
        
        Args:
            backup_type: Type of backup.
            
        Returns:
            Unique backup ID string.
        """
        with self._lock:
            self._backup_counter += 1
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            return f"backup_{backup_type.value}_{timestamp}_{self._backup_counter}"
    
    def create_backup(
        self,
        backup_type: BackupType = BackupType.MANUAL,
        trigger_reason: str = "",
        additional_state: Optional[Dict[str, Any]] = None,
    ) -> BackupInfo:
        """
        Create a backup checkpoint.
        
        Args:
            backup_type: Type of backup to create.
            trigger_reason: Reason for creating backup.
            additional_state: Additional state to include.
            
        Returns:
            BackupInfo with backup details.
        """
        if not self.config.enabled:
            logger.warning("Backup is disabled in configuration")
            return BackupInfo(
                backup_id="disabled",
                backup_type=backup_type,
                status=BackupStatus.FAILED,
                trigger_reason="Backup disabled in configuration",
            )
        
        backup_id = self._generate_backup_id(backup_type)
        backup_info = BackupInfo(
            backup_id=backup_id,
            backup_type=backup_type,
            status=BackupStatus.IN_PROGRESS,
            timestamp=datetime.now(),
            trigger_reason=trigger_reason,
        )
        
        try:
            # Collect state
            state = self._collect_state(additional_state)
            
            # Generate filepath
            extension = ".pt.gz" if self.config.compress_backup else ".pt"
            filename = f"{backup_id}{extension}"
            filepath = Path(self.config.backup_dir) / filename
            
            # Save checkpoint
            checkpoint = self._build_checkpoint(state, backup_info)
            
            if self.config.compress_backup:
                self._save_compressed(checkpoint, filepath)
            else:
                torch.save(checkpoint, filepath)
            
            # Update backup info
            backup_info.status = BackupStatus.COMPLETED
            backup_info.filepath = str(filepath)
            backup_info.size_bytes = filepath.stat().st_size
            backup_info.model_state_included = state.model_state is not None
            backup_info.optimizer_state_included = (
                state.optimizer_state is not None and
                self.config.include_optimizer_state
            )
            backup_info.scheduler_state_included = (
                state.scheduler_state is not None and
                self.config.include_scheduler_state
            )
            
            # Track backup
            with self._lock:
                self._backups.append(backup_info)
            
            # Rotate old backups
            self._rotate_backups()
            
            # Notify callbacks
            for callback in self._callbacks:
                try:
                    callback(backup_info)
                except Exception as e:
                    logger.error(f"Error in backup callback: {e}")
            
            logger.info(
                f"Created backup: {backup_id} "
                f"({backup_info.size_bytes / (1024 * 1024):.2f} MB)"
            )
            
        except Exception as e:
            backup_info.status = BackupStatus.FAILED
            backup_info.metadata["error"] = str(e)
            logger.error(f"Backup failed: {e}")
        
        return backup_info
    
    def _collect_state(
        self,
        additional_state: Optional[Dict[str, Any]] = None
    ) -> BackupState:
        """
        Collect current state for backup.
        
        Args:
            additional_state: Additional state data.
            
        Returns:
            BackupState with collected state.
        """
        # Use state provider if available
        if self._state_provider:
            state = self._state_provider()
        else:
            state = BackupState()
        
        # Collect model state
        if self._model is not None and state.model_state is None:
            state.model_state = self._model.state_dict()
        
        # Collect optimizer state
        if (self._optimizer is not None and 
            self.config.include_optimizer_state and
            state.optimizer_state is None):
            state.optimizer_state = self._optimizer.state_dict()
        
        # Collect scheduler state
        if (self._scheduler is not None and 
            self.config.include_scheduler_state and
            state.scheduler_state is None):
            if hasattr(self._scheduler, "state_dict"):
                state.scheduler_state = self._scheduler.state_dict()
        
        # Add additional state
        if additional_state:
            state.additional.update(additional_state)
        
        return state
    
    def _build_checkpoint(
        self,
        state: BackupState,
        backup_info: BackupInfo
    ) -> Dict[str, Any]:
        """
        Build checkpoint dictionary from state.
        
        Args:
            state: Backup state data.
            backup_info: Backup metadata.
            
        Returns:
            Checkpoint dictionary.
        """
        checkpoint = {
            "backup_info": backup_info.to_dict(),
            "timestamp": datetime.now().isoformat(),
            "epoch": state.epoch,
            "step": state.step,
            "metrics": state.metrics,
        }
        
        if state.model_state is not None:
            checkpoint["model_state_dict"] = state.model_state
        
        if state.optimizer_state is not None:
            checkpoint["optimizer_state_dict"] = state.optimizer_state
        
        if state.scheduler_state is not None:
            checkpoint["scheduler_state_dict"] = state.scheduler_state
        
        if state.config:
            checkpoint["config"] = state.config
        
        if state.additional:
            checkpoint["additional"] = state.additional
        
        return checkpoint
    
    def _save_compressed(
        self,
        checkpoint: Dict[str, Any],
        filepath: Path
    ) -> None:
        """
        Save checkpoint with gzip compression.
        
        Args:
            checkpoint: Checkpoint dictionary.
            filepath: Path to save to.
        """
        import io
        
        buffer = io.BytesIO()
        torch.save(checkpoint, buffer)
        buffer.seek(0)
        
        with gzip.open(filepath, "wb") as f:
            f.write(buffer.read())
    
    def _rotate_backups(self) -> None:
        """Remove old backups to stay within max_backups limit."""
        if self.config.max_backups <= 0:
            return
        
        with self._lock:
            # Sort backups by timestamp
            sorted_backups = sorted(
                self._backups,
                key=lambda b: b.timestamp or datetime.min,
                reverse=True
            )
            
            # Keep only max_backups most recent
            to_remove = sorted_backups[self.config.max_backups:]
            
            for backup in to_remove:
                if backup.filepath and Path(backup.filepath).exists():
                    try:
                        Path(backup.filepath).unlink()
                        logger.debug(f"Removed old backup: {backup.backup_id}")
                    except Exception as e:
                        logger.error(f"Failed to remove backup {backup.filepath}: {e}")
                
                self._backups.remove(backup)
    
    def load_backup(
        self,
        filepath: Union[str, Path],
        map_location: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Load a backup checkpoint.
        
        Args:
            filepath: Path to backup file.
            map_location: Device to map tensors to.
            
        Returns:
            Loaded checkpoint dictionary.
        """
        filepath = Path(filepath)
        
        if not filepath.exists():
            raise FileNotFoundError(f"Backup file not found: {filepath}")
        
        if filepath.suffix == ".gz":
            # Load compressed backup
            with gzip.open(filepath, "rb") as f:
                import io
                buffer = io.BytesIO(f.read())
                checkpoint = torch.load(buffer, map_location=map_location)
        else:
            checkpoint = torch.load(filepath, map_location=map_location)
        
        logger.info(f"Loaded backup from: {filepath}")
        return checkpoint
    
    def restore_from_backup(
        self,
        filepath: Union[str, Path],
        model: Optional[nn.Module] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        map_location: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Restore model and training state from backup.
        
        Args:
            filepath: Path to backup file.
            model: Model to restore state to.
            optimizer: Optimizer to restore state to.
            scheduler: Scheduler to restore state to.
            map_location: Device to map tensors to.
            
        Returns:
            Checkpoint metadata (epoch, step, metrics).
        """
        checkpoint = self.load_backup(filepath, map_location)
        
        # Use registered components if not provided
        model = model or self._model
        optimizer = optimizer or self._optimizer
        scheduler = scheduler or self._scheduler
        
        # Restore model state
        if model is not None and "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
            logger.info("Restored model state from backup")
        
        # Restore optimizer state
        if optimizer is not None and "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            logger.info("Restored optimizer state from backup")
        
        # Restore scheduler state
        if scheduler is not None and "scheduler_state_dict" in checkpoint:
            if hasattr(scheduler, "load_state_dict"):
                scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
                logger.info("Restored scheduler state from backup")
        
        return {
            "epoch": checkpoint.get("epoch", 0),
            "step": checkpoint.get("step", 0),
            "metrics": checkpoint.get("metrics", {}),
            "config": checkpoint.get("config", {}),
            "additional": checkpoint.get("additional", {}),
        }
    
    def get_backups(
        self,
        backup_type: Optional[BackupType] = None
    ) -> List[BackupInfo]:
        """
        Get list of created backups.
        
        Args:
            backup_type: Filter by backup type.
            
        Returns:
            List of BackupInfo objects.
        """
        with self._lock:
            backups = list(self._backups)
        
        if backup_type:
            backups = [b for b in backups if b.backup_type == backup_type]
        
        return backups
    
    def get_latest_backup(
        self,
        backup_type: Optional[BackupType] = None
    ) -> Optional[BackupInfo]:
        """
        Get the most recent backup.
        
        Args:
            backup_type: Filter by backup type.
            
        Returns:
            Most recent BackupInfo or None.
        """
        backups = self.get_backups(backup_type)
        if not backups:
            return None
        
        return max(backups, key=lambda b: b.timestamp or datetime.min)
    
    def register_callback(
        self,
        callback: Callable[[BackupInfo], None]
    ) -> None:
        """
        Register callback for backup events.
        
        Args:
            callback: Function to call when backup is created.
        """
        self._callbacks.append(callback)
    
    def cleanup_all_backups(self) -> int:
        """
        Remove all backup files.
        
        Returns:
            Number of backups removed.
        """
        count = 0
        with self._lock:
            for backup in self._backups:
                if backup.filepath and Path(backup.filepath).exists():
                    try:
                        Path(backup.filepath).unlink()
                        count += 1
                    except Exception as e:
                        logger.error(f"Failed to remove {backup.filepath}: {e}")
            
            self._backups.clear()
        
        logger.info(f"Cleaned up {count} backup files")
        return count


# =============================================================================
# Auto-Backup Trigger
# =============================================================================

class AutoBackupTrigger:
    """
    Automatic backup trigger based on resource thresholds.
    
    Monitors resource usage and triggers backups when
    warning or critical thresholds are exceeded.
    """
    
    def __init__(
        self,
        backup_handler: BackupHandler,
        config: Optional[ResourceConfig] = None,
    ) -> None:
        """
        Initialize auto-backup trigger.
        
        Args:
            backup_handler: BackupHandler instance.
            config: Resource configuration.
        """
        if config is None:
            config = load_resource_config()
        
        self.backup_handler = backup_handler
        self.config = config
        
        self._last_warning_backup: Optional[datetime] = None
        self._last_critical_backup: Optional[datetime] = None
        self._min_backup_interval_seconds = 60  # Minimum time between backups
        self._shutdown_callbacks: List[Callable[[], None]] = []
        self._lock = threading.Lock()
    
    def on_resource_alert(
        self,
        snapshot: ResourceSnapshot,
        alert_level: str
    ) -> Optional[BackupInfo]:
        """
        Handle resource alert and trigger backup if needed.
        
        Args:
            snapshot: Resource snapshot that triggered alert.
            alert_level: Level of alert (warning or critical).
            
        Returns:
            BackupInfo if backup was created, None otherwise.
        """
        backup_config = self.config.backup
        
        # Check if backup should be triggered
        should_backup = False
        backup_type = BackupType.MANUAL
        
        if alert_level == "warning" and backup_config.backup_on_warning:
            if self._can_create_backup("warning"):
                should_backup = True
                backup_type = BackupType.WARNING_TRIGGER
        
        elif alert_level == "critical" and backup_config.backup_on_critical:
            if self._can_create_backup("critical"):
                should_backup = True
                backup_type = BackupType.CRITICAL_TRIGGER
        
        if should_backup:
            # Create backup
            reason = (
                f"{snapshot.resource_type.value.upper()} usage at "
                f"{snapshot.usage_percent:.1f}% ({alert_level})"
            )
            
            backup_info = self.backup_handler.create_backup(
                backup_type=backup_type,
                trigger_reason=reason,
                additional_state={
                    "trigger_snapshot": snapshot.to_dict(),
                    "alert_level": alert_level,
                },
            )
            
            # Update last backup time
            with self._lock:
                if alert_level == "warning":
                    self._last_warning_backup = datetime.now()
                else:
                    self._last_critical_backup = datetime.now()
            
            # Handle critical action
            if alert_level == "critical":
                self._handle_critical_action(snapshot, backup_info)
            
            return backup_info
        
        return None
    
    def _can_create_backup(self, alert_level: str) -> bool:
        """
        Check if enough time has passed since last backup.
        
        Args:
            alert_level: Alert level to check.
            
        Returns:
            True if backup can be created.
        """
        with self._lock:
            if alert_level == "warning":
                last_backup = self._last_warning_backup
            else:
                last_backup = self._last_critical_backup
            
            if last_backup is None:
                return True
            
            elapsed = (datetime.now() - last_backup).total_seconds()
            return elapsed >= self._min_backup_interval_seconds
    
    def _handle_critical_action(
        self,
        snapshot: ResourceSnapshot,
        backup_info: BackupInfo
    ) -> None:
        """
        Handle critical threshold action.
        
        Args:
            snapshot: Resource snapshot.
            backup_info: Backup that was created.
        """
        action = self.config.thresholds.critical_action
        
        if action == ThresholdAction.WARN:
            logger.critical(
                f"CRITICAL: {snapshot.resource_type.value.upper()} at "
                f"{snapshot.usage_percent:.1f}%"
            )
        
        elif action == ThresholdAction.BACKUP:
            # Already handled by backup creation
            pass
        
        elif action == ThresholdAction.SHUTDOWN:
            self._initiate_shutdown(snapshot, backup_info, force=True)
        
        elif action == ThresholdAction.BACKUP_AND_SHUTDOWN:
            self._initiate_shutdown(snapshot, backup_info, force=False)
    
    def _initiate_shutdown(
        self,
        snapshot: ResourceSnapshot,
        backup_info: BackupInfo,
        force: bool = False
    ) -> None:
        """
        Initiate graceful shutdown.
        
        Args:
            snapshot: Resource snapshot that triggered shutdown.
            backup_info: Backup created before shutdown.
            force: Whether to force immediate shutdown.
        """
        logger.critical(
            f"Initiating shutdown due to {snapshot.resource_type.value.upper()} "
            f"at {snapshot.usage_percent:.1f}%"
        )
        
        if backup_info.filepath:
            logger.critical(f"Backup saved to: {backup_info.filepath}")
        
        # Notify shutdown callbacks
        for callback in self._shutdown_callbacks:
            try:
                callback()
            except Exception as e:
                logger.error(f"Error in shutdown callback: {e}")
        
        if force or not self.config.enable_auto_shutdown:
            return
        
        # Grace period
        grace_period = self.config.shutdown_grace_period_seconds
        logger.critical(
            f"Shutdown in {grace_period} seconds. Save work and exit gracefully."
        )
        
        # Schedule shutdown
        def delayed_shutdown():
            time.sleep(grace_period)
            logger.critical("Grace period ended. Terminating process.")
            os._exit(1)
        
        shutdown_thread = threading.Thread(
            target=delayed_shutdown,
            daemon=True,
            name="shutdown_timer"
        )
        shutdown_thread.start()
    
    def register_shutdown_callback(
        self,
        callback: Callable[[], None]
    ) -> None:
        """
        Register callback for shutdown event.
        
        Args:
            callback: Function to call before shutdown.
        """
        self._shutdown_callbacks.append(callback)
    
    def set_min_backup_interval(self, seconds: float) -> None:
        """
        Set minimum interval between automatic backups.
        
        Args:
            seconds: Minimum seconds between backups.
        """
        self._min_backup_interval_seconds = seconds


# =============================================================================
# Module Exports
# =============================================================================

__all__ = [
    "BackupType",
    "BackupStatus",
    "BackupInfo",
    "BackupState",
    "BackupHandler",
    "AutoBackupTrigger",
]
