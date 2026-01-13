"""
Resource Configuration Module

Defines configuration schemas and loaders for resource management settings.
All configurations are loaded from the resource_configs.yaml file.
"""

import logging
from pathlib import Path
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Union
from enum import Enum

from src.middleware.config_loader import (
    get_config_dir,
    load_yaml,
    get_config_manager,
)


logger = logging.getLogger(__name__)


# =============================================================================
# Enumerations
# =============================================================================

class ResourceType(str, Enum):
    """Types of resources that can be monitored."""
    
    CPU = "cpu"
    GPU = "gpu"
    MEMORY = "memory"
    DISK = "disk"


class ThresholdAction(str, Enum):
    """Actions to take when resource threshold is exceeded."""
    
    WARN = "warn"
    BACKUP = "backup"
    SHUTDOWN = "shutdown"
    BACKUP_AND_SHUTDOWN = "backup_and_shutdown"


# =============================================================================
# Configuration Data Classes
# =============================================================================

@dataclass
class ResourceThresholds:
    """
    Resource threshold configuration.
    
    Defines warning and critical thresholds for different resources.
    When thresholds are exceeded, corresponding actions are triggered.
    
    Attributes:
        cpu_warning_percent: CPU usage percentage to trigger warning.
        cpu_critical_percent: CPU usage percentage to trigger critical action.
        memory_warning_percent: Memory usage percentage to trigger warning.
        memory_critical_percent: Memory usage percentage to trigger critical action.
        gpu_memory_warning_percent: GPU memory percentage to trigger warning.
        gpu_memory_critical_percent: GPU memory percentage to trigger critical action.
        disk_warning_percent: Disk usage percentage to trigger warning.
        disk_critical_percent: Disk usage percentage to trigger critical action.
        critical_action: Action to take when critical threshold is exceeded.
    """
    
    cpu_warning_percent: float = 70.0
    cpu_critical_percent: float = 90.0
    memory_warning_percent: float = 70.0
    memory_critical_percent: float = 90.0
    gpu_memory_warning_percent: float = 70.0
    gpu_memory_critical_percent: float = 90.0
    disk_warning_percent: float = 80.0
    disk_critical_percent: float = 95.0
    critical_action: ThresholdAction = ThresholdAction.BACKUP_AND_SHUTDOWN
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ResourceThresholds":
        """
        Create ResourceThresholds from dictionary.
        
        Args:
            data: Dictionary containing threshold values.
            
        Returns:
            ResourceThresholds instance.
        """
        action_str = data.get("critical_action", "backup_and_shutdown")
        try:
            action = ThresholdAction(action_str)
        except ValueError:
            logger.warning(
                f"Invalid critical_action '{action_str}', using default 'backup_and_shutdown'"
            )
            action = ThresholdAction.BACKUP_AND_SHUTDOWN
        
        return cls(
            cpu_warning_percent=data.get("cpu_warning_percent", 70.0),
            cpu_critical_percent=data.get("cpu_critical_percent", 90.0),
            memory_warning_percent=data.get("memory_warning_percent", 70.0),
            memory_critical_percent=data.get("memory_critical_percent", 90.0),
            gpu_memory_warning_percent=data.get("gpu_memory_warning_percent", 70.0),
            gpu_memory_critical_percent=data.get("gpu_memory_critical_percent", 90.0),
            disk_warning_percent=data.get("disk_warning_percent", 80.0),
            disk_critical_percent=data.get("disk_critical_percent", 95.0),
            critical_action=action,
        )


@dataclass
class MonitoringInterval:
    """
    Monitoring interval configuration.
    
    Defines how frequently different resources should be monitored.
    
    Attributes:
        cpu_seconds: Interval in seconds for CPU monitoring.
        memory_seconds: Interval in seconds for memory monitoring.
        gpu_seconds: Interval in seconds for GPU monitoring.
        disk_seconds: Interval in seconds for disk monitoring.
        aggregate_seconds: Interval for aggregating all metrics.
    """
    
    cpu_seconds: float = 5.0
    memory_seconds: float = 5.0
    gpu_seconds: float = 5.0
    disk_seconds: float = 60.0
    aggregate_seconds: float = 10.0
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MonitoringInterval":
        """
        Create MonitoringInterval from dictionary.
        
        Args:
            data: Dictionary containing interval values.
            
        Returns:
            MonitoringInterval instance.
        """
        return cls(
            cpu_seconds=data.get("cpu_seconds", 5.0),
            memory_seconds=data.get("memory_seconds", 5.0),
            gpu_seconds=data.get("gpu_seconds", 5.0),
            disk_seconds=data.get("disk_seconds", 60.0),
            aggregate_seconds=data.get("aggregate_seconds", 10.0),
        )


@dataclass
class BackupConfig:
    """
    Backup configuration for automatic checkpoint saving.
    
    Attributes:
        enabled: Whether automatic backup is enabled.
        backup_dir: Directory to store backup files.
        max_backups: Maximum number of backup files to keep.
        backup_on_warning: Create backup when warning threshold is reached.
        backup_on_critical: Create backup when critical threshold is reached.
        include_optimizer_state: Include optimizer state in backup.
        include_scheduler_state: Include scheduler state in backup.
        compress_backup: Compress backup files.
    """
    
    enabled: bool = True
    backup_dir: str = "checkpoints/emergency_backups"
    max_backups: int = 5
    backup_on_warning: bool = False
    backup_on_critical: bool = True
    include_optimizer_state: bool = True
    include_scheduler_state: bool = True
    compress_backup: bool = False
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BackupConfig":
        """
        Create BackupConfig from dictionary.
        
        Args:
            data: Dictionary containing backup configuration.
            
        Returns:
            BackupConfig instance.
        """
        return cls(
            enabled=data.get("enabled", True),
            backup_dir=data.get("backup_dir", "checkpoints/emergency_backups"),
            max_backups=data.get("max_backups", 5),
            backup_on_warning=data.get("backup_on_warning", False),
            backup_on_critical=data.get("backup_on_critical", True),
            include_optimizer_state=data.get("include_optimizer_state", True),
            include_scheduler_state=data.get("include_scheduler_state", True),
            compress_backup=data.get("compress_backup", False),
        )


@dataclass
class ReportConfig:
    """
    Report configuration for automatic report generation.
    
    Attributes:
        enabled: Whether automatic reporting is enabled.
        report_dir: Directory to store report files.
        report_format: Format for report files (json, yaml, csv).
        auto_save_interval_minutes: Interval for auto-saving reports.
        include_resource_history: Include historical resource usage.
        include_progress_history: Include historical progress data.
        max_history_entries: Maximum number of history entries to keep.
        save_on_shutdown: Save report when shutting down.
        save_on_exception: Save report when exception occurs.
    """
    
    enabled: bool = True
    report_dir: str = "logs/resource_reports"
    report_format: str = "json"
    auto_save_interval_minutes: float = 30.0
    include_resource_history: bool = True
    include_progress_history: bool = True
    max_history_entries: int = 1000
    save_on_shutdown: bool = True
    save_on_exception: bool = True
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ReportConfig":
        """
        Create ReportConfig from dictionary.
        
        Args:
            data: Dictionary containing report configuration.
            
        Returns:
            ReportConfig instance.
        """
        return cls(
            enabled=data.get("enabled", True),
            report_dir=data.get("report_dir", "logs/resource_reports"),
            report_format=data.get("report_format", "json"),
            auto_save_interval_minutes=data.get("auto_save_interval_minutes", 30.0),
            include_resource_history=data.get("include_resource_history", True),
            include_progress_history=data.get("include_progress_history", True),
            max_history_entries=data.get("max_history_entries", 1000),
            save_on_shutdown=data.get("save_on_shutdown", True),
            save_on_exception=data.get("save_on_exception", True),
        )


@dataclass
class ResourceConfig:
    """
    Main resource management configuration.
    
    Aggregates all sub-configurations for resource management.
    
    Attributes:
        thresholds: Resource threshold configuration.
        intervals: Monitoring interval configuration.
        backup: Backup configuration.
        report: Report configuration.
        enable_monitoring: Whether resource monitoring is enabled.
        enable_auto_shutdown: Whether automatic shutdown is enabled.
        shutdown_grace_period_seconds: Grace period before shutdown.
        log_level: Logging level for resource management.
    """
    
    thresholds: ResourceThresholds = field(default_factory=ResourceThresholds)
    intervals: MonitoringInterval = field(default_factory=MonitoringInterval)
    backup: BackupConfig = field(default_factory=BackupConfig)
    report: ReportConfig = field(default_factory=ReportConfig)
    enable_monitoring: bool = True
    enable_auto_shutdown: bool = True
    shutdown_grace_period_seconds: float = 30.0
    log_level: str = "INFO"
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ResourceConfig":
        """
        Create ResourceConfig from dictionary.
        
        Args:
            data: Dictionary containing resource configuration.
            
        Returns:
            ResourceConfig instance.
        """
        thresholds_data = data.get("thresholds", {})
        intervals_data = data.get("intervals", {})
        backup_data = data.get("backup", {})
        report_data = data.get("report", {})
        
        return cls(
            thresholds=ResourceThresholds.from_dict(thresholds_data),
            intervals=MonitoringInterval.from_dict(intervals_data),
            backup=BackupConfig.from_dict(backup_data),
            report=ReportConfig.from_dict(report_data),
            enable_monitoring=data.get("enable_monitoring", True),
            enable_auto_shutdown=data.get("enable_auto_shutdown", True),
            shutdown_grace_period_seconds=data.get("shutdown_grace_period_seconds", 30.0),
            log_level=data.get("log_level", "INFO"),
        )


# =============================================================================
# Configuration Loading Functions
# =============================================================================

def get_resource_config_path() -> Path:
    """
    Get the path to the resource configuration file.
    
    Returns:
        Path to resource_configs.yaml file.
    """
    return get_config_dir() / "resource_configs.yaml"


def load_resource_config(
    config_path: Optional[Union[str, Path]] = None
) -> ResourceConfig:
    """
    Load resource configuration from YAML file.
    
    Args:
        config_path: Optional path to configuration file.
                    If not provided, uses default resource_configs.yaml.
                    
    Returns:
        ResourceConfig instance with loaded configuration.
    """
    if config_path is None:
        config_path = get_resource_config_path()
    else:
        config_path = Path(config_path)
    
    if not config_path.exists():
        logger.warning(
            f"Resource configuration file not found at {config_path}. "
            "Using default configuration."
        )
        return ResourceConfig()
    
    try:
        config_data = load_yaml(config_path)
        resource_data = config_data.get("resource_management", {})
        config = ResourceConfig.from_dict(resource_data)
        logger.info(f"Loaded resource configuration from {config_path}")
        return config
    except Exception as e:
        logger.error(f"Failed to load resource configuration: {e}")
        logger.warning("Using default resource configuration.")
        return ResourceConfig()


def get_default_resource_config() -> ResourceConfig:
    """
    Get default resource configuration.
    
    Returns:
        ResourceConfig instance with default values.
    """
    return ResourceConfig()


# =============================================================================
# Module Exports
# =============================================================================

__all__ = [
    "ResourceType",
    "ThresholdAction",
    "ResourceThresholds",
    "MonitoringInterval",
    "BackupConfig",
    "ReportConfig",
    "ResourceConfig",
    "get_resource_config_path",
    "load_resource_config",
    "get_default_resource_config",
]
