"""
Resource Manager Module

Provides the main facade for resource management functionality.
Integrates monitoring, progress tracking, reporting, and backup
into a unified interface.
"""

import atexit
import logging
import signal
import sys
import threading
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Callable, Dict, Generator, List, Optional, Union

import torch.nn as nn

from .resource_config import (
    ResourceConfig,
    load_resource_config,
    ThresholdAction,
)
from .resource_monitor import (
    ResourceMonitor,
    AggregatedMetrics,
    ResourceSnapshot,
)
from .progress_tracker import (
    ProgressTracker,
    TrainingProgressTracker,
    TaskInfo,
    TaskStatus,
)
from .report_manager import (
    ReportManager,
    ReportType,
    ReportData,
)
from .backup_handler import (
    BackupHandler,
    AutoBackupTrigger,
    BackupInfo,
    BackupType,
    BackupState,
)


logger = logging.getLogger(__name__)


# =============================================================================
# Global Resource Manager Instance
# =============================================================================

_resource_manager: Optional["ResourceManager"] = None


def get_resource_manager(
    config: Optional[ResourceConfig] = None,
    create_if_missing: bool = True,
) -> Optional["ResourceManager"]:
    """
    Get the global resource manager instance.
    
    Args:
        config: Configuration to use if creating new instance.
        create_if_missing: Create instance if not exists.
        
    Returns:
        ResourceManager instance or None.
    """
    global _resource_manager
    
    if _resource_manager is None and create_if_missing:
        _resource_manager = ResourceManager(config)
    
    return _resource_manager


def reset_resource_manager() -> None:
    """Reset the global resource manager instance."""
    global _resource_manager
    
    if _resource_manager is not None:
        _resource_manager.stop()
        _resource_manager = None


# =============================================================================
# Resource Context
# =============================================================================

@dataclass
class ResourceContext:
    """
    Context information for resource management.
    
    Attributes:
        is_training: Whether training is in progress.
        current_task_id: ID of current tracked task.
        model_registered: Whether model is registered for backup.
        monitoring_active: Whether monitoring is active.
        last_metrics: Most recent aggregated metrics.
        active_alerts: Current active alerts.
    """
    
    is_training: bool = False
    current_task_id: Optional[str] = None
    model_registered: bool = False
    monitoring_active: bool = False
    last_metrics: Optional[AggregatedMetrics] = None
    active_alerts: List[str] = None
    
    def __post_init__(self) -> None:
        if self.active_alerts is None:
            self.active_alerts = []
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert context to dictionary."""
        return {
            "is_training": self.is_training,
            "current_task_id": self.current_task_id,
            "model_registered": self.model_registered,
            "monitoring_active": self.monitoring_active,
            "last_metrics": (
                self.last_metrics.to_dict() if self.last_metrics else None
            ),
            "active_alerts": self.active_alerts,
        }


# =============================================================================
# Resource Manager
# =============================================================================

class ResourceManager:
    """
    Main facade for resource management functionality.
    
    Integrates all resource management components:
    - Resource monitoring (CPU, GPU, Memory, Disk)
    - Progress tracking for training tasks
    - Automatic report generation
    - Emergency backup on threshold breach
    
    Usage:
        manager = ResourceManager()
        manager.start()
        
        # Register model for backup
        manager.register_model(model, optimizer, scheduler)
        
        # Start training tracking
        task_id = manager.start_training(
            name="VQA Training",
            total_epochs=10,
            steps_per_epoch=1000
        )
        
        # Update progress
        manager.update_training_step(task_id, step=1, loss=0.5)
        
        # Complete training
        manager.complete_training(task_id)
        
        manager.stop()
    """
    
    def __init__(
        self,
        config: Optional[ResourceConfig] = None,
    ) -> None:
        """
        Initialize resource manager.
        
        Args:
            config: Resource configuration. Loads from YAML if not provided.
        """
        if config is None:
            config = load_resource_config()
        
        self.config = config
        self._context = ResourceContext()
        self._lock = threading.Lock()
        self._is_running = False
        
        # Initialize components
        self.monitor = ResourceMonitor(config)
        self.progress_tracker = TrainingProgressTracker()
        self.report_manager = ReportManager(config.report)
        self.backup_handler = BackupHandler(config.backup)
        self.auto_backup_trigger = AutoBackupTrigger(
            self.backup_handler, config
        )
        
        # Register internal callbacks
        self._setup_callbacks()
        
        # Setup signal handlers for graceful shutdown
        self._setup_signal_handlers()
        
        logger.info("ResourceManager initialized")
    
    def _setup_callbacks(self) -> None:
        """Setup internal callbacks between components."""
        # Alert callback for auto-backup
        self.monitor.register_alert_callback(self._on_resource_alert)
        
        # Metrics callback for periodic reporting
        self.monitor.register_metrics_callback(self._on_metrics_update)
        
        # Shutdown callback
        self.auto_backup_trigger.register_shutdown_callback(
            self._on_shutdown_triggered
        )
    
    def _setup_signal_handlers(self) -> None:
        """Setup signal handlers for graceful shutdown."""
        def signal_handler(signum, frame):
            logger.warning(f"Received signal {signum}, initiating shutdown")
            self._save_emergency_state()
            self.stop()
            sys.exit(0)
        
        # Register for SIGINT (Ctrl+C) and SIGTERM
        try:
            signal.signal(signal.SIGINT, signal_handler)
            signal.signal(signal.SIGTERM, signal_handler)
        except Exception as e:
            logger.warning(f"Could not register signal handlers: {e}")
        
        # Register atexit handler
        atexit.register(self._on_exit)
    
    def _on_resource_alert(
        self,
        snapshot: ResourceSnapshot,
        alert_level: str
    ) -> None:
        """
        Handle resource alert from monitor.
        
        Args:
            snapshot: Resource snapshot that triggered alert.
            alert_level: Level of alert.
        """
        with self._lock:
            alert_msg = (
                f"{alert_level.upper()}: {snapshot.resource_type.value} "
                f"at {snapshot.usage_percent:.1f}%"
            )
            
            if alert_msg not in self._context.active_alerts:
                self._context.active_alerts.append(alert_msg)
        
        # Trigger auto-backup if needed
        self.auto_backup_trigger.on_resource_alert(snapshot, alert_level)
    
    def _on_metrics_update(self, metrics: AggregatedMetrics) -> None:
        """
        Handle metrics update from monitor.
        
        Args:
            metrics: Updated aggregated metrics.
        """
        with self._lock:
            self._context.last_metrics = metrics
            
            # Clear old alerts, keep only current ones
            self._context.active_alerts = list(metrics.alerts)
    
    def _on_shutdown_triggered(self) -> None:
        """Handle shutdown triggered by resource threshold."""
        logger.critical("Resource manager shutting down due to threshold breach")
        
        # Save emergency report
        if self.config.report.save_on_shutdown:
            try:
                self.save_emergency_report("Threshold breach shutdown")
            except Exception as e:
                logger.error(f"Failed to save emergency report: {e}")
    
    def _on_exit(self) -> None:
        """Handle process exit."""
        if self._is_running:
            try:
                if self.config.report.save_on_shutdown:
                    self._save_final_report()
            except Exception as e:
                logger.error(f"Error during exit cleanup: {e}")
    
    def _save_emergency_state(self) -> None:
        """Save emergency state before shutdown."""
        try:
            self.backup_handler.create_backup(
                backup_type=BackupType.SHUTDOWN,
                trigger_reason="Emergency shutdown",
            )
        except Exception as e:
            logger.error(f"Failed to create emergency backup: {e}")
        
        try:
            self.save_emergency_report("Emergency shutdown")
        except Exception as e:
            logger.error(f"Failed to save emergency report: {e}")
    
    def _save_final_report(self) -> None:
        """Save final report before normal exit."""
        try:
            metrics = self.monitor.get_latest_metrics()
            tasks = self.progress_tracker.get_all_tasks()
            
            self.report_manager.generate_and_save(
                report_type=ReportType.COMBINED,
                metrics=metrics,
                tasks=tasks,
                additional_data={
                    "reason": "Normal shutdown",
                    "context": self._context.to_dict(),
                },
            )
        except Exception as e:
            logger.error(f"Failed to save final report: {e}")
    
    # =========================================================================
    # Lifecycle Methods
    # =========================================================================
    
    def start(self) -> None:
        """Start resource management (monitoring and auto-save)."""
        if self._is_running:
            logger.warning("ResourceManager is already running")
            return
        
        self._is_running = True
        
        # Start monitoring
        if self.config.enable_monitoring:
            self.monitor.start()
            self._context.monitoring_active = True
        
        # Start auto-save for reports
        self.report_manager.start_auto_save(
            metrics_provider=self.monitor.get_latest_metrics,
            tasks_provider=self.progress_tracker.get_all_tasks,
        )
        
        logger.info("ResourceManager started")
    
    def stop(self) -> None:
        """Stop resource management."""
        if not self._is_running:
            return
        
        self._is_running = False
        
        # Stop components
        self.monitor.stop()
        self.report_manager.stop_auto_save()
        
        self._context.monitoring_active = False
        
        logger.info("ResourceManager stopped")
    
    @property
    def is_running(self) -> bool:
        """Check if resource manager is running."""
        return self._is_running
    
    # =========================================================================
    # Model Registration
    # =========================================================================
    
    def register_model(
        self,
        model: nn.Module,
        optimizer: Optional[Any] = None,
        scheduler: Optional[Any] = None,
        state_provider: Optional[Callable[[], BackupState]] = None,
    ) -> None:
        """
        Register model and training components for backup.
        
        Args:
            model: PyTorch model.
            optimizer: Optional optimizer.
            scheduler: Optional scheduler.
            state_provider: Optional function to provide backup state.
        """
        self.backup_handler.register_model(model, optimizer, scheduler)
        
        if state_provider:
            self.backup_handler.register_state_provider(state_provider)
        
        self._context.model_registered = True
        logger.info("Model registered for resource management")
    
    # =========================================================================
    # Training Progress Methods
    # =========================================================================
    
    def start_training(
        self,
        name: str,
        total_epochs: int,
        steps_per_epoch: int,
        task_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Start tracking a training task.
        
        Args:
            name: Name of the training task.
            total_epochs: Total number of epochs.
            steps_per_epoch: Steps per epoch.
            task_id: Optional custom task ID.
            metadata: Additional metadata.
            
        Returns:
            Task ID for the training task.
        """
        task_id = self.progress_tracker.create_training_task(
            name=name,
            total_epochs=total_epochs,
            steps_per_epoch=steps_per_epoch,
            task_id=task_id,
            metadata=metadata,
        )
        
        self.progress_tracker.start_task(task_id)
        
        with self._lock:
            self._context.is_training = True
            self._context.current_task_id = task_id
        
        logger.info(f"Started training task: {name} ({task_id})")
        return task_id
    
    def start_epoch(
        self,
        task_id: Optional[str] = None,
        epoch: int = 0,
        learning_rate: Optional[float] = None,
    ) -> None:
        """
        Mark the start of a new epoch.
        
        Args:
            task_id: Task ID (uses current if not provided).
            epoch: Epoch number.
            learning_rate: Current learning rate.
        """
        task_id = task_id or self._context.current_task_id
        if task_id is None:
            raise ValueError("No active training task")
        
        self.progress_tracker.start_epoch(task_id, epoch, learning_rate)
    
    def update_training_step(
        self,
        task_id: Optional[str] = None,
        step: Optional[int] = None,
        loss: float = 0.0,
        metrics: Optional[Dict[str, float]] = None,
        learning_rate: Optional[float] = None,
    ) -> TaskInfo:
        """
        Update training progress for a single step.
        
        Args:
            task_id: Task ID (uses current if not provided).
            step: Global step number.
            loss: Current loss value.
            metrics: Additional metrics.
            learning_rate: Current learning rate.
            
        Returns:
            Updated TaskInfo.
        """
        task_id = task_id or self._context.current_task_id
        if task_id is None:
            raise ValueError("No active training task")
        
        return self.progress_tracker.update_training_step(
            task_id=task_id,
            step=step,
            loss=loss,
            metrics=metrics,
            learning_rate=learning_rate,
        )
    
    def end_epoch(
        self,
        task_id: Optional[str] = None,
        epoch: int = 0,
        train_loss: float = 0.0,
        val_loss: Optional[float] = None,
        metrics: Optional[Dict[str, float]] = None,
    ) -> None:
        """
        Mark the end of an epoch.
        
        Args:
            task_id: Task ID (uses current if not provided).
            epoch: Epoch number that ended.
            train_loss: Average training loss.
            val_loss: Validation loss.
            metrics: Additional epoch metrics.
        """
        task_id = task_id or self._context.current_task_id
        if task_id is None:
            raise ValueError("No active training task")
        
        self.progress_tracker.end_epoch(
            task_id=task_id,
            epoch=epoch,
            train_loss=train_loss,
            val_loss=val_loss,
            metrics=metrics,
        )
    
    def complete_training(
        self,
        task_id: Optional[str] = None,
        final_metrics: Optional[Dict[str, Any]] = None,
    ) -> TaskInfo:
        """
        Mark training as completed.
        
        Args:
            task_id: Task ID (uses current if not provided).
            final_metrics: Final training metrics.
            
        Returns:
            Completed TaskInfo.
        """
        task_id = task_id or self._context.current_task_id
        if task_id is None:
            raise ValueError("No active training task")
        
        task = self.progress_tracker.complete_task(task_id, final_metrics)
        
        with self._lock:
            self._context.is_training = False
            if self._context.current_task_id == task_id:
                self._context.current_task_id = None
        
        # Generate completion report
        try:
            summary = self.progress_tracker.get_training_summary(task_id)
            self.report_manager.generate_and_save(
                report_type=ReportType.TRAINING_PROGRESS,
                tasks={task_id: task},
                additional_data={"summary": summary},
            )
        except Exception as e:
            logger.error(f"Failed to generate completion report: {e}")
        
        return task
    
    def fail_training(
        self,
        task_id: Optional[str] = None,
        error_message: str = "Training failed",
        error_details: Optional[Dict[str, Any]] = None,
    ) -> TaskInfo:
        """
        Mark training as failed.
        
        Args:
            task_id: Task ID (uses current if not provided).
            error_message: Error message.
            error_details: Additional error details.
            
        Returns:
            Failed TaskInfo.
        """
        task_id = task_id or self._context.current_task_id
        if task_id is None:
            raise ValueError("No active training task")
        
        task = self.progress_tracker.fail_task(
            task_id=task_id,
            error_message=error_message,
            error_details=error_details,
        )
        
        with self._lock:
            self._context.is_training = False
            if self._context.current_task_id == task_id:
                self._context.current_task_id = None
        
        # Create emergency backup and report
        self.create_backup(
            backup_type=BackupType.EMERGENCY,
            trigger_reason=f"Training failed: {error_message}",
        )
        
        self.save_emergency_report(
            f"Training failed: {error_message}",
            error_info=error_details,
        )
        
        return task
    
    # =========================================================================
    # Resource Monitoring Methods
    # =========================================================================
    
    def get_current_metrics(self) -> AggregatedMetrics:
        """
        Get current resource metrics.
        
        Returns:
            Current AggregatedMetrics.
        """
        return self.monitor.get_all_metrics()
    
    def get_latest_metrics(self) -> Optional[AggregatedMetrics]:
        """
        Get most recent cached metrics.
        
        Returns:
            Most recent AggregatedMetrics or None.
        """
        return self.monitor.get_latest_metrics()
    
    def is_resource_critical(self) -> bool:
        """
        Check if any resource is at critical level.
        
        Returns:
            True if any resource exceeds critical threshold.
        """
        return self.monitor.is_critical()
    
    def is_resource_warning(self) -> bool:
        """
        Check if any resource is at warning level.
        
        Returns:
            True if any resource exceeds warning threshold.
        """
        return self.monitor.is_warning()
    
    def get_active_alerts(self) -> List[str]:
        """
        Get list of active resource alerts.
        
        Returns:
            List of alert strings.
        """
        with self._lock:
            return list(self._context.active_alerts)
    
    # =========================================================================
    # Backup Methods
    # =========================================================================
    
    def create_backup(
        self,
        backup_type: BackupType = BackupType.MANUAL,
        trigger_reason: str = "",
        additional_state: Optional[Dict[str, Any]] = None,
    ) -> BackupInfo:
        """
        Create a backup checkpoint.
        
        Args:
            backup_type: Type of backup.
            trigger_reason: Reason for backup.
            additional_state: Additional state to include.
            
        Returns:
            BackupInfo with backup details.
        """
        return self.backup_handler.create_backup(
            backup_type=backup_type,
            trigger_reason=trigger_reason,
            additional_state=additional_state,
        )
    
    def restore_backup(
        self,
        filepath: str,
        model: Optional[nn.Module] = None,
        optimizer: Optional[Any] = None,
        scheduler: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """
        Restore from a backup checkpoint.
        
        Args:
            filepath: Path to backup file.
            model: Model to restore to.
            optimizer: Optimizer to restore to.
            scheduler: Scheduler to restore to.
            
        Returns:
            Restored checkpoint metadata.
        """
        return self.backup_handler.restore_from_backup(
            filepath=filepath,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
        )
    
    def get_latest_backup(
        self,
        backup_type: Optional[BackupType] = None
    ) -> Optional[BackupInfo]:
        """
        Get most recent backup.
        
        Args:
            backup_type: Filter by backup type.
            
        Returns:
            Most recent BackupInfo or None.
        """
        return self.backup_handler.get_latest_backup(backup_type)
    
    # =========================================================================
    # Report Methods
    # =========================================================================
    
    def generate_report(
        self,
        report_type: ReportType = ReportType.COMBINED,
        additional_data: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Generate and save a report.
        
        Args:
            report_type: Type of report to generate.
            additional_data: Additional data to include.
            
        Returns:
            Path to saved report file.
        """
        metrics = self.monitor.get_latest_metrics()
        tasks = self.progress_tracker.get_all_tasks()
        
        return self.report_manager.generate_and_save(
            report_type=report_type,
            metrics=metrics,
            tasks=tasks,
            additional_data=additional_data,
        )
    
    def save_emergency_report(
        self,
        reason: str,
        error_info: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Generate and save emergency report.
        
        Args:
            reason: Reason for emergency.
            error_info: Error information.
            
        Returns:
            Path to saved report file.
        """
        metrics = self.monitor.get_latest_metrics()
        tasks = self.progress_tracker.get_all_tasks()
        
        return self.report_manager.save_emergency_report(
            reason=reason,
            metrics=metrics,
            tasks=tasks,
            error_info=error_info,
        )
    
    # =========================================================================
    # Context Methods
    # =========================================================================
    
    def get_context(self) -> ResourceContext:
        """
        Get current resource context.
        
        Returns:
            Current ResourceContext.
        """
        with self._lock:
            return ResourceContext(
                is_training=self._context.is_training,
                current_task_id=self._context.current_task_id,
                model_registered=self._context.model_registered,
                monitoring_active=self._context.monitoring_active,
                last_metrics=self._context.last_metrics,
                active_alerts=list(self._context.active_alerts),
            )
    
    def get_status_summary(self) -> Dict[str, Any]:
        """
        Get comprehensive status summary.
        
        Returns:
            Dictionary containing status summary.
        """
        context = self.get_context()
        metrics = self.monitor.get_latest_metrics()
        tasks = self.progress_tracker.get_all_tasks()
        backups = self.backup_handler.get_backups()
        
        active_tasks = [
            t for t in tasks.values()
            if t.status == TaskStatus.RUNNING
        ]
        
        summary = {
            "timestamp": datetime.now().isoformat(),
            "is_running": self._is_running,
            "context": context.to_dict(),
            "resources": {
                "monitoring_active": context.monitoring_active,
                "active_alerts": context.active_alerts,
                "is_critical": self.monitor.is_critical(),
                "is_warning": self.monitor.is_warning(),
            },
            "training": {
                "is_training": context.is_training,
                "current_task_id": context.current_task_id,
                "active_tasks": len(active_tasks),
                "total_tasks": len(tasks),
            },
            "backup": {
                "model_registered": context.model_registered,
                "total_backups": len(backups),
                "latest_backup": (
                    backups[-1].to_dict() if backups else None
                ),
            },
        }
        
        if metrics:
            summary["resources"]["current"] = {
                "cpu_percent": metrics.cpu.usage_percent if metrics.cpu else None,
                "memory_percent": (
                    metrics.memory.usage_percent if metrics.memory else None
                ),
                "gpu_percent": (
                    metrics.gpu.usage_percent 
                    if metrics.gpu and metrics.gpu.additional_info.get("available", True)
                    else None
                ),
                "disk_percent": metrics.disk.usage_percent if metrics.disk else None,
            }
        
        return summary
    
    # =========================================================================
    # Context Manager Support
    # =========================================================================
    
    def __enter__(self) -> "ResourceManager":
        """Enter context manager."""
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        """Exit context manager."""
        if exc_type is not None:
            # Save emergency state on exception
            try:
                self.save_emergency_report(
                    f"Exception: {exc_type.__name__}: {exc_val}",
                    error_info={
                        "exception_type": str(exc_type),
                        "exception_value": str(exc_val),
                    },
                )
                self.create_backup(
                    backup_type=BackupType.EMERGENCY,
                    trigger_reason=f"Exception: {exc_val}",
                )
            except Exception as e:
                logger.error(f"Failed to save emergency state: {e}")
        
        self.stop()
        return False  # Do not suppress exceptions


@contextmanager
def resource_managed_training(
    model: nn.Module,
    optimizer: Optional[Any] = None,
    scheduler: Optional[Any] = None,
    config: Optional[ResourceConfig] = None,
) -> Generator[ResourceManager, None, None]:
    """
    Context manager for resource-managed training.
    
    Args:
        model: PyTorch model.
        optimizer: Optional optimizer.
        scheduler: Optional scheduler.
        config: Optional resource configuration.
        
    Yields:
        ResourceManager instance.
        
    Example:
        with resource_managed_training(model, optimizer) as manager:
            task_id = manager.start_training("Training", epochs, steps)
            for epoch in range(epochs):
                manager.start_epoch(epoch=epoch)
                for step in range(steps):
                    # Training step
                    manager.update_training_step(step=step, loss=loss)
                manager.end_epoch(epoch=epoch, train_loss=avg_loss)
            manager.complete_training(task_id)
    """
    manager = ResourceManager(config)
    manager.register_model(model, optimizer, scheduler)
    
    try:
        manager.start()
        yield manager
    finally:
        manager.stop()


# =============================================================================
# Module Exports
# =============================================================================

__all__ = [
    "ResourceManager",
    "ResourceContext",
    "get_resource_manager",
    "reset_resource_manager",
    "resource_managed_training",
]
