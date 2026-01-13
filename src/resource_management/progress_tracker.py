"""
Progress Tracker Module

Provides progress tracking functionality for training tasks and other
long-running operations. Supports task status management, time estimation,
and progress history recording.
"""

import logging
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Deque, Union

logger = logging.getLogger(__name__)


# =============================================================================
# Enumerations
# =============================================================================

class TaskStatus(str, Enum):
    """Status of a tracked task."""
    
    PENDING = "pending"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class TaskPriority(str, Enum):
    """Priority level of a task."""
    
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    CRITICAL = "critical"


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class TaskInfo:
    """
    Information about a tracked task.
    
    Attributes:
        task_id: Unique identifier for the task.
        name: Human-readable task name.
        status: Current task status.
        priority: Task priority level.
        total_steps: Total number of steps to complete.
        current_step: Current step number.
        start_time: When the task was started.
        end_time: When the task was completed/failed.
        estimated_completion: Estimated completion time.
        progress_percent: Current progress percentage.
        metadata: Additional task metadata.
        error_message: Error message if task failed.
    """
    
    task_id: str
    name: str
    status: TaskStatus = TaskStatus.PENDING
    priority: TaskPriority = TaskPriority.NORMAL
    total_steps: int = 0
    current_step: int = 0
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    estimated_completion: Optional[datetime] = None
    progress_percent: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    error_message: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert task info to dictionary.
        
        Returns:
            Dictionary representation of task info.
        """
        return {
            "task_id": self.task_id,
            "name": self.name,
            "status": self.status.value,
            "priority": self.priority.value,
            "total_steps": self.total_steps,
            "current_step": self.current_step,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "estimated_completion": (
                self.estimated_completion.isoformat() 
                if self.estimated_completion else None
            ),
            "progress_percent": self.progress_percent,
            "metadata": self.metadata,
            "error_message": self.error_message,
        }


@dataclass
class ProgressSnapshot:
    """
    Snapshot of progress at a specific point in time.
    
    Attributes:
        timestamp: When the snapshot was taken.
        task_id: ID of the task.
        step: Current step number.
        progress_percent: Progress percentage.
        step_duration_seconds: Duration of the last step.
        metrics: Metrics associated with this step.
    """
    
    timestamp: datetime
    task_id: str
    step: int
    progress_percent: float
    step_duration_seconds: float
    metrics: Dict[str, float] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert snapshot to dictionary.
        
        Returns:
            Dictionary representation of snapshot.
        """
        return {
            "timestamp": self.timestamp.isoformat(),
            "task_id": self.task_id,
            "step": self.step,
            "progress_percent": self.progress_percent,
            "step_duration_seconds": self.step_duration_seconds,
            "metrics": self.metrics,
        }


# =============================================================================
# Progress Tracker Class
# =============================================================================

class ProgressTracker:
    """
    General-purpose progress tracker for long-running tasks.
    
    Supports multiple concurrent tasks, progress estimation,
    and history tracking.
    """
    
    def __init__(
        self,
        history_size: int = 1000,
    ) -> None:
        """
        Initialize progress tracker.
        
        Args:
            history_size: Maximum number of history entries to keep per task.
        """
        self.history_size = history_size
        
        self._tasks: Dict[str, TaskInfo] = {}
        self._history: Dict[str, Deque[ProgressSnapshot]] = {}
        self._step_times: Dict[str, Deque[float]] = {}
        self._callbacks: List[Callable[[TaskInfo], None]] = []
        self._lock = threading.Lock()
        self._task_counter = 0
    
    def create_task(
        self,
        name: str,
        total_steps: int,
        task_id: Optional[str] = None,
        priority: TaskPriority = TaskPriority.NORMAL,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Create a new tracked task.
        
        Args:
            name: Human-readable task name.
            total_steps: Total number of steps in the task.
            task_id: Optional custom task ID.
            priority: Task priority level.
            metadata: Additional metadata for the task.
            
        Returns:
            Task ID of the created task.
        """
        with self._lock:
            if task_id is None:
                self._task_counter += 1
                task_id = f"task_{self._task_counter}_{int(time.time())}"
            
            task = TaskInfo(
                task_id=task_id,
                name=name,
                status=TaskStatus.PENDING,
                priority=priority,
                total_steps=total_steps,
                current_step=0,
                metadata=metadata or {},
            )
            
            self._tasks[task_id] = task
            self._history[task_id] = deque(maxlen=self.history_size)
            self._step_times[task_id] = deque(maxlen=100)
            
            logger.info(f"Created task '{name}' with ID: {task_id}")
            return task_id
    
    def start_task(self, task_id: str) -> None:
        """
        Start a task.
        
        Args:
            task_id: ID of the task to start.
        """
        with self._lock:
            if task_id not in self._tasks:
                raise ValueError(f"Task not found: {task_id}")
            
            task = self._tasks[task_id]
            task.status = TaskStatus.RUNNING
            task.start_time = datetime.now()
            
            logger.info(f"Started task: {task.name}")
            self._notify_callbacks(task)
    
    def update_progress(
        self,
        task_id: str,
        step: Optional[int] = None,
        increment: int = 1,
        metrics: Optional[Dict[str, float]] = None,
    ) -> TaskInfo:
        """
        Update task progress.
        
        Args:
            task_id: ID of the task to update.
            step: Absolute step number (overrides increment if provided).
            increment: Number of steps to increment.
            metrics: Metrics for this step.
            
        Returns:
            Updated TaskInfo.
        """
        with self._lock:
            if task_id not in self._tasks:
                raise ValueError(f"Task not found: {task_id}")
            
            task = self._tasks[task_id]
            
            if task.status != TaskStatus.RUNNING:
                logger.warning(
                    f"Updating progress for non-running task: {task.name}"
                )
            
            # Calculate step duration
            now = datetime.now()
            last_snapshot = (
                self._history[task_id][-1]
                if self._history[task_id]
                else None
            )
            
            if last_snapshot:
                step_duration = (
                    now - datetime.fromisoformat(
                        last_snapshot.timestamp.isoformat()
                    )
                ).total_seconds()
            elif task.start_time:
                step_duration = (now - task.start_time).total_seconds()
            else:
                step_duration = 0.0
            
            # Update step
            if step is not None:
                task.current_step = step
            else:
                task.current_step += increment
            
            # Update progress percentage
            if task.total_steps > 0:
                task.progress_percent = (
                    task.current_step / task.total_steps
                ) * 100
            else:
                task.progress_percent = 0.0
            
            # Record step time for estimation
            self._step_times[task_id].append(step_duration)
            
            # Estimate completion time
            task.estimated_completion = self._estimate_completion(task_id)
            
            # Create snapshot
            snapshot = ProgressSnapshot(
                timestamp=now,
                task_id=task_id,
                step=task.current_step,
                progress_percent=task.progress_percent,
                step_duration_seconds=step_duration,
                metrics=metrics or {},
            )
            self._history[task_id].append(snapshot)
            
            self._notify_callbacks(task)
            return task
    
    def _estimate_completion(self, task_id: str) -> Optional[datetime]:
        """
        Estimate task completion time based on historical step times.
        
        Args:
            task_id: ID of the task.
            
        Returns:
            Estimated completion datetime or None.
        """
        task = self._tasks[task_id]
        step_times = self._step_times[task_id]
        
        if not step_times or task.total_steps == 0:
            return None
        
        remaining_steps = task.total_steps - task.current_step
        if remaining_steps <= 0:
            return datetime.now()
        
        # Use weighted average (recent steps weighted more)
        weights = [i + 1 for i in range(len(step_times))]
        weighted_avg = sum(
            t * w for t, w in zip(step_times, weights)
        ) / sum(weights)
        
        estimated_seconds = weighted_avg * remaining_steps
        return datetime.now() + timedelta(seconds=estimated_seconds)
    
    def complete_task(
        self,
        task_id: str,
        final_metrics: Optional[Dict[str, Any]] = None,
    ) -> TaskInfo:
        """
        Mark a task as completed.
        
        Args:
            task_id: ID of the task to complete.
            final_metrics: Final metrics to record.
            
        Returns:
            Completed TaskInfo.
        """
        with self._lock:
            if task_id not in self._tasks:
                raise ValueError(f"Task not found: {task_id}")
            
            task = self._tasks[task_id]
            task.status = TaskStatus.COMPLETED
            task.end_time = datetime.now()
            task.progress_percent = 100.0
            task.current_step = task.total_steps
            
            if final_metrics:
                task.metadata["final_metrics"] = final_metrics
            
            # Calculate total duration
            if task.start_time:
                duration = (task.end_time - task.start_time).total_seconds()
                task.metadata["total_duration_seconds"] = duration
            
            logger.info(
                f"Completed task: {task.name} "
                f"({task.total_steps} steps)"
            )
            self._notify_callbacks(task)
            return task
    
    def fail_task(
        self,
        task_id: str,
        error_message: str,
        error_details: Optional[Dict[str, Any]] = None,
    ) -> TaskInfo:
        """
        Mark a task as failed.
        
        Args:
            task_id: ID of the task that failed.
            error_message: Error message describing the failure.
            error_details: Additional error details.
            
        Returns:
            Failed TaskInfo.
        """
        with self._lock:
            if task_id not in self._tasks:
                raise ValueError(f"Task not found: {task_id}")
            
            task = self._tasks[task_id]
            task.status = TaskStatus.FAILED
            task.end_time = datetime.now()
            task.error_message = error_message
            
            if error_details:
                task.metadata["error_details"] = error_details
            
            logger.error(f"Task failed: {task.name} - {error_message}")
            self._notify_callbacks(task)
            return task
    
    def pause_task(self, task_id: str) -> TaskInfo:
        """
        Pause a running task.
        
        Args:
            task_id: ID of the task to pause.
            
        Returns:
            Paused TaskInfo.
        """
        with self._lock:
            if task_id not in self._tasks:
                raise ValueError(f"Task not found: {task_id}")
            
            task = self._tasks[task_id]
            if task.status == TaskStatus.RUNNING:
                task.status = TaskStatus.PAUSED
                logger.info(f"Paused task: {task.name}")
            
            self._notify_callbacks(task)
            return task
    
    def resume_task(self, task_id: str) -> TaskInfo:
        """
        Resume a paused task.
        
        Args:
            task_id: ID of the task to resume.
            
        Returns:
            Resumed TaskInfo.
        """
        with self._lock:
            if task_id not in self._tasks:
                raise ValueError(f"Task not found: {task_id}")
            
            task = self._tasks[task_id]
            if task.status == TaskStatus.PAUSED:
                task.status = TaskStatus.RUNNING
                logger.info(f"Resumed task: {task.name}")
            
            self._notify_callbacks(task)
            return task
    
    def cancel_task(self, task_id: str) -> TaskInfo:
        """
        Cancel a task.
        
        Args:
            task_id: ID of the task to cancel.
            
        Returns:
            Cancelled TaskInfo.
        """
        with self._lock:
            if task_id not in self._tasks:
                raise ValueError(f"Task not found: {task_id}")
            
            task = self._tasks[task_id]
            task.status = TaskStatus.CANCELLED
            task.end_time = datetime.now()
            
            logger.info(f"Cancelled task: {task.name}")
            self._notify_callbacks(task)
            return task
    
    def get_task(self, task_id: str) -> Optional[TaskInfo]:
        """
        Get task information.
        
        Args:
            task_id: ID of the task.
            
        Returns:
            TaskInfo or None if not found.
        """
        with self._lock:
            return self._tasks.get(task_id)
    
    def get_all_tasks(self) -> Dict[str, TaskInfo]:
        """
        Get all tracked tasks.
        
        Returns:
            Dictionary of task ID to TaskInfo.
        """
        with self._lock:
            return dict(self._tasks)
    
    def get_active_tasks(self) -> List[TaskInfo]:
        """
        Get all running or paused tasks.
        
        Returns:
            List of active TaskInfo objects.
        """
        with self._lock:
            return [
                task for task in self._tasks.values()
                if task.status in [TaskStatus.RUNNING, TaskStatus.PAUSED]
            ]
    
    def get_history(self, task_id: str) -> List[ProgressSnapshot]:
        """
        Get progress history for a task.
        
        Args:
            task_id: ID of the task.
            
        Returns:
            List of ProgressSnapshot objects.
        """
        with self._lock:
            if task_id not in self._history:
                return []
            return list(self._history[task_id])
    
    def remove_task(self, task_id: str) -> None:
        """
        Remove a task from tracking.
        
        Args:
            task_id: ID of the task to remove.
        """
        with self._lock:
            if task_id in self._tasks:
                del self._tasks[task_id]
            if task_id in self._history:
                del self._history[task_id]
            if task_id in self._step_times:
                del self._step_times[task_id]
            
            logger.info(f"Removed task: {task_id}")
    
    def clear_completed(self) -> int:
        """
        Remove all completed, failed, or cancelled tasks.
        
        Returns:
            Number of tasks removed.
        """
        with self._lock:
            to_remove = [
                task_id for task_id, task in self._tasks.items()
                if task.status in [
                    TaskStatus.COMPLETED,
                    TaskStatus.FAILED,
                    TaskStatus.CANCELLED
                ]
            ]
            
            for task_id in to_remove:
                del self._tasks[task_id]
                if task_id in self._history:
                    del self._history[task_id]
                if task_id in self._step_times:
                    del self._step_times[task_id]
            
            logger.info(f"Cleared {len(to_remove)} completed tasks")
            return len(to_remove)
    
    def register_callback(
        self,
        callback: Callable[[TaskInfo], None]
    ) -> None:
        """
        Register callback for task updates.
        
        Args:
            callback: Function to call when task is updated.
        """
        self._callbacks.append(callback)
    
    def _notify_callbacks(self, task: TaskInfo) -> None:
        """
        Notify registered callbacks of task update.
        
        Args:
            task: Updated task info.
        """
        for callback in self._callbacks:
            try:
                callback(task)
            except Exception as e:
                logger.error(f"Error in progress callback: {e}")


# =============================================================================
# Training Progress Tracker
# =============================================================================

class TrainingProgressTracker(ProgressTracker):
    """
    Specialized progress tracker for training tasks.
    
    Extends base ProgressTracker with training-specific features
    like epoch tracking, loss monitoring, and learning rate tracking.
    """
    
    def __init__(
        self,
        history_size: int = 1000,
    ) -> None:
        """
        Initialize training progress tracker.
        
        Args:
            history_size: Maximum history entries per task.
        """
        super().__init__(history_size=history_size)
        
        self._epoch_info: Dict[str, Dict[int, Dict[str, Any]]] = {}
        self._best_metrics: Dict[str, Dict[str, float]] = {}
    
    def create_training_task(
        self,
        name: str,
        total_epochs: int,
        steps_per_epoch: int,
        task_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Create a new training task.
        
        Args:
            name: Name of the training task.
            total_epochs: Total number of epochs.
            steps_per_epoch: Number of steps per epoch.
            task_id: Optional custom task ID.
            metadata: Additional metadata.
            
        Returns:
            Task ID of the created task.
        """
        total_steps = total_epochs * steps_per_epoch
        
        task_metadata = metadata or {}
        task_metadata.update({
            "task_type": "training",
            "total_epochs": total_epochs,
            "steps_per_epoch": steps_per_epoch,
            "current_epoch": 0,
        })
        
        task_id = self.create_task(
            name=name,
            total_steps=total_steps,
            task_id=task_id,
            priority=TaskPriority.HIGH,
            metadata=task_metadata,
        )
        
        self._epoch_info[task_id] = {}
        self._best_metrics[task_id] = {}
        
        return task_id
    
    def start_epoch(
        self,
        task_id: str,
        epoch: int,
        learning_rate: Optional[float] = None,
    ) -> None:
        """
        Mark the start of a new epoch.
        
        Args:
            task_id: ID of the training task.
            epoch: Epoch number (0-indexed).
            learning_rate: Current learning rate.
        """
        with self._lock:
            if task_id not in self._tasks:
                raise ValueError(f"Task not found: {task_id}")
            
            task = self._tasks[task_id]
            task.metadata["current_epoch"] = epoch
            
            self._epoch_info[task_id][epoch] = {
                "start_time": datetime.now().isoformat(),
                "learning_rate": learning_rate,
                "train_loss": [],
                "val_loss": None,
                "metrics": {},
            }
            
            logger.info(
                f"Started epoch {epoch + 1}/{task.metadata.get('total_epochs', '?')} "
                f"for task: {task.name}"
            )
    
    def update_training_step(
        self,
        task_id: str,
        step: int,
        loss: float,
        metrics: Optional[Dict[str, float]] = None,
        learning_rate: Optional[float] = None,
    ) -> TaskInfo:
        """
        Update training progress for a single step.
        
        Args:
            task_id: ID of the training task.
            step: Global step number.
            loss: Current loss value.
            metrics: Additional metrics.
            learning_rate: Current learning rate.
            
        Returns:
            Updated TaskInfo.
        """
        step_metrics = metrics or {}
        step_metrics["loss"] = loss
        if learning_rate is not None:
            step_metrics["learning_rate"] = learning_rate
        
        task = self.update_progress(
            task_id=task_id,
            step=step,
            metrics=step_metrics,
        )
        
        # Record loss in epoch info
        with self._lock:
            current_epoch = task.metadata.get("current_epoch", 0)
            if (task_id in self._epoch_info and 
                current_epoch in self._epoch_info[task_id]):
                self._epoch_info[task_id][current_epoch]["train_loss"].append(loss)
        
        return task
    
    def end_epoch(
        self,
        task_id: str,
        epoch: int,
        train_loss: float,
        val_loss: Optional[float] = None,
        metrics: Optional[Dict[str, float]] = None,
    ) -> None:
        """
        Mark the end of an epoch with summary metrics.
        
        Args:
            task_id: ID of the training task.
            epoch: Epoch number that ended.
            train_loss: Average training loss for the epoch.
            val_loss: Validation loss (if applicable).
            metrics: Additional epoch-level metrics.
        """
        with self._lock:
            if task_id not in self._tasks:
                raise ValueError(f"Task not found: {task_id}")
            
            if task_id not in self._epoch_info:
                self._epoch_info[task_id] = {}
            
            if epoch not in self._epoch_info[task_id]:
                self._epoch_info[task_id][epoch] = {}
            
            epoch_data = self._epoch_info[task_id][epoch]
            epoch_data["end_time"] = datetime.now().isoformat()
            epoch_data["train_loss_avg"] = train_loss
            epoch_data["val_loss"] = val_loss
            
            if metrics:
                epoch_data["metrics"] = metrics
            
            # Update best metrics
            self._update_best_metrics(task_id, train_loss, val_loss, metrics)
            
            # Calculate epoch duration
            if "start_time" in epoch_data:
                start = datetime.fromisoformat(epoch_data["start_time"])
                end = datetime.fromisoformat(epoch_data["end_time"])
                epoch_data["duration_seconds"] = (end - start).total_seconds()
            
            task = self._tasks[task_id]
            logger.info(
                f"Completed epoch {epoch + 1}/{task.metadata.get('total_epochs', '?')} "
                f"- Train Loss: {train_loss:.4f}"
                f"{f', Val Loss: {val_loss:.4f}' if val_loss is not None else ''}"
            )
    
    def _update_best_metrics(
        self,
        task_id: str,
        train_loss: float,
        val_loss: Optional[float],
        metrics: Optional[Dict[str, float]],
    ) -> None:
        """
        Update best metrics tracking.
        
        Args:
            task_id: ID of the training task.
            train_loss: Current training loss.
            val_loss: Current validation loss.
            metrics: Additional metrics.
        """
        if task_id not in self._best_metrics:
            self._best_metrics[task_id] = {}
        
        best = self._best_metrics[task_id]
        
        # Track best train loss (lower is better)
        if "train_loss" not in best or train_loss < best["train_loss"]:
            best["train_loss"] = train_loss
        
        # Track best val loss (lower is better)
        if val_loss is not None:
            if "val_loss" not in best or val_loss < best["val_loss"]:
                best["val_loss"] = val_loss
        
        # Track other metrics (assume higher is better for accuracy-type metrics)
        if metrics:
            for key, value in metrics.items():
                if "loss" in key.lower():
                    # Lower is better for loss metrics
                    if key not in best or value < best[key]:
                        best[key] = value
                else:
                    # Higher is better for other metrics
                    if key not in best or value > best[key]:
                        best[key] = value
    
    def get_epoch_info(
        self,
        task_id: str,
        epoch: Optional[int] = None,
    ) -> Union[Dict[int, Dict[str, Any]], Dict[str, Any]]:
        """
        Get epoch information for a task.
        
        Args:
            task_id: ID of the training task.
            epoch: Specific epoch number (returns all if not provided).
            
        Returns:
            Epoch info dictionary.
        """
        with self._lock:
            if task_id not in self._epoch_info:
                return {} if epoch is None else {}
            
            if epoch is not None:
                return self._epoch_info[task_id].get(epoch, {})
            
            return dict(self._epoch_info[task_id])
    
    def get_best_metrics(self, task_id: str) -> Dict[str, float]:
        """
        Get best metrics achieved during training.
        
        Args:
            task_id: ID of the training task.
            
        Returns:
            Dictionary of best metric values.
        """
        with self._lock:
            return dict(self._best_metrics.get(task_id, {}))
    
    def get_training_summary(self, task_id: str) -> Dict[str, Any]:
        """
        Get comprehensive training summary.
        
        Args:
            task_id: ID of the training task.
            
        Returns:
            Dictionary containing training summary.
        """
        with self._lock:
            if task_id not in self._tasks:
                return {}
            
            task = self._tasks[task_id]
            history = list(self._history.get(task_id, []))
            epoch_info = dict(self._epoch_info.get(task_id, {}))
            best_metrics = dict(self._best_metrics.get(task_id, {}))
            
            # Calculate average step time
            step_times = list(self._step_times.get(task_id, []))
            avg_step_time = sum(step_times) / len(step_times) if step_times else 0
            
            return {
                "task_info": task.to_dict(),
                "total_epochs": task.metadata.get("total_epochs", 0),
                "completed_epochs": len(epoch_info),
                "total_steps": task.total_steps,
                "completed_steps": task.current_step,
                "best_metrics": best_metrics,
                "average_step_time_seconds": avg_step_time,
                "epoch_info": epoch_info,
                "history_size": len(history),
            }


# =============================================================================
# Module Exports
# =============================================================================

__all__ = [
    "TaskStatus",
    "TaskPriority",
    "TaskInfo",
    "ProgressSnapshot",
    "ProgressTracker",
    "TrainingProgressTracker",
]
