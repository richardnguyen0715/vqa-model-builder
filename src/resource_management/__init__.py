"""
Resource Management Module

Provides comprehensive resource monitoring, progress tracking, and automatic
backup functionality for the VQA Model Builder project.

Components:
    - resource_config: Configuration schema and loader for resource management
    - resource_monitor: Real-time monitoring of CPU, GPU, and memory usage
    - progress_tracker: Training progress and task tracking
    - report_manager: Automated report generation and saving
    - backup_handler: Automatic checkpoint backup on resource threshold breach
    - resource_manager: Main facade integrating all components

Usage:
    Basic usage with context manager:
    
        from src.resource_management import resource_managed_training
        
        with resource_managed_training(model, optimizer) as manager:
            task_id = manager.start_training("VQA Training", epochs=10, steps_per_epoch=100)
            for epoch in range(10):
                manager.start_epoch(epoch=epoch)
                for step in range(100):
                    loss = train_step()
                    manager.update_training_step(step=step, loss=loss)
                manager.end_epoch(epoch=epoch, train_loss=avg_loss)
            manager.complete_training(task_id)
    
    Manual usage:
    
        from src.resource_management import ResourceManager
        
        manager = ResourceManager()
        manager.start()
        manager.register_model(model, optimizer)
        # ... training code ...
        manager.stop()
"""

from .resource_config import (
    ResourceConfig,
    ResourceThresholds,
    MonitoringInterval,
    BackupConfig,
    ReportConfig,
    ResourceType,
    ThresholdAction,
    load_resource_config,
    get_default_resource_config,
)

from .resource_monitor import (
    ResourceSnapshot,
    AggregatedMetrics,
    ResourceMonitor,
    CPUMonitor,
    GPUMonitor,
    MemoryMonitor,
    DiskMonitor,
    BaseResourceMonitor,
)

from .progress_tracker import (
    TaskStatus,
    TaskPriority,
    TaskInfo,
    ProgressSnapshot,
    ProgressTracker,
    TrainingProgressTracker,
)

from .report_manager import (
    ReportType,
    ReportFormat,
    ReportData,
    ReportMetadata,
    ReportManager,
    ReportGenerator,
)

from .backup_handler import (
    BackupType,
    BackupStatus,
    BackupInfo,
    BackupState,
    BackupHandler,
    AutoBackupTrigger,
)

from .resource_manager import (
    ResourceManager,
    ResourceContext,
    get_resource_manager,
    reset_resource_manager,
    resource_managed_training,
)


__all__ = [
    # Configuration
    "ResourceConfig",
    "ResourceThresholds",
    "MonitoringInterval",
    "BackupConfig",
    "ReportConfig",
    "ResourceType",
    "ThresholdAction",
    "load_resource_config",
    "get_default_resource_config",
    # Monitoring
    "ResourceSnapshot",
    "AggregatedMetrics",
    "ResourceMonitor",
    "CPUMonitor",
    "GPUMonitor",
    "MemoryMonitor",
    "DiskMonitor",
    "BaseResourceMonitor",
    # Progress Tracking
    "TaskStatus",
    "TaskPriority",
    "TaskInfo",
    "ProgressSnapshot",
    "ProgressTracker",
    "TrainingProgressTracker",
    # Reporting
    "ReportType",
    "ReportFormat",
    "ReportData",
    "ReportMetadata",
    "ReportManager",
    "ReportGenerator",
    # Backup
    "BackupType",
    "BackupStatus",
    "BackupInfo",
    "BackupState",
    "BackupHandler",
    "AutoBackupTrigger",
    # Main Manager
    "ResourceManager",
    "ResourceContext",
    "get_resource_manager",
    "reset_resource_manager",
    "resource_managed_training",
]
