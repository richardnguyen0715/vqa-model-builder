"""
Report Manager Module

Provides automated report generation, formatting, and persistence for
resource usage and training progress data.
"""

import json
import logging
import os
import threading
import time
from datetime import datetime
from enum import Enum
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

import yaml

from .resource_config import ReportConfig, load_resource_config
from .resource_monitor import AggregatedMetrics, ResourceSnapshot
from .progress_tracker import TaskInfo, ProgressSnapshot


logger = logging.getLogger(__name__)


# =============================================================================
# Enumerations
# =============================================================================

class ReportType(str, Enum):
    """Types of reports that can be generated."""
    
    RESOURCE_USAGE = "resource_usage"
    TRAINING_PROGRESS = "training_progress"
    SYSTEM_STATUS = "system_status"
    COMBINED = "combined"
    EMERGENCY = "emergency"


class ReportFormat(str, Enum):
    """Output formats for reports."""
    
    JSON = "json"
    YAML = "yaml"
    CSV = "csv"
    TEXT = "text"


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class ReportData:
    """
    Container for report data.
    
    Attributes:
        report_id: Unique identifier for the report.
        report_type: Type of report.
        timestamp: When the report was generated.
        title: Report title.
        summary: Brief summary of the report.
        data: Main report data.
        metadata: Additional metadata.
    """
    
    report_id: str
    report_type: ReportType
    timestamp: datetime
    title: str
    summary: str
    data: Dict[str, Any]
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert report to dictionary.
        
        Returns:
            Dictionary representation of report.
        """
        return {
            "report_id": self.report_id,
            "report_type": self.report_type.value,
            "timestamp": self.timestamp.isoformat(),
            "title": self.title,
            "summary": self.summary,
            "data": self.data,
            "metadata": self.metadata,
        }


@dataclass
class ReportMetadata:
    """
    Metadata about generated reports.
    
    Attributes:
        total_reports: Total number of reports generated.
        last_report_time: Time of last report generation.
        report_types_count: Count of each report type.
        total_size_bytes: Total size of all reports.
    """
    
    total_reports: int = 0
    last_report_time: Optional[datetime] = None
    report_types_count: Dict[str, int] = field(default_factory=dict)
    total_size_bytes: int = 0


# =============================================================================
# Report Formatters
# =============================================================================

class BaseReportFormatter:
    """Base class for report formatters."""
    
    def format(self, report: ReportData) -> str:
        """
        Format report data to string.
        
        Args:
            report: Report data to format.
            
        Returns:
            Formatted report string.
        """
        raise NotImplementedError
    
    def get_extension(self) -> str:
        """
        Get file extension for this format.
        
        Returns:
            File extension string.
        """
        raise NotImplementedError


class JSONReportFormatter(BaseReportFormatter):
    """Formats reports as JSON."""
    
    def __init__(self, indent: int = 2) -> None:
        """
        Initialize JSON formatter.
        
        Args:
            indent: JSON indentation level.
        """
        self.indent = indent
    
    def format(self, report: ReportData) -> str:
        """Format report as JSON string."""
        return json.dumps(report.to_dict(), indent=self.indent, default=str)
    
    def get_extension(self) -> str:
        """Get JSON file extension."""
        return ".json"


class YAMLReportFormatter(BaseReportFormatter):
    """Formats reports as YAML."""
    
    def format(self, report: ReportData) -> str:
        """Format report as YAML string."""
        return yaml.dump(
            report.to_dict(),
            default_flow_style=False,
            allow_unicode=True,
            sort_keys=False,
        )
    
    def get_extension(self) -> str:
        """Get YAML file extension."""
        return ".yaml"


class CSVReportFormatter(BaseReportFormatter):
    """Formats reports as CSV (flattened)."""
    
    def format(self, report: ReportData) -> str:
        """Format report as CSV string."""
        lines = []
        
        # Header row
        lines.append("key,value")
        
        # Flatten and add data
        flat_data = self._flatten_dict(report.to_dict())
        for key, value in flat_data.items():
            # Escape commas and quotes in values
            value_str = str(value).replace('"', '""')
            if ',' in value_str or '"' in value_str or '\n' in value_str:
                value_str = f'"{value_str}"'
            lines.append(f"{key},{value_str}")
        
        return "\n".join(lines)
    
    def _flatten_dict(
        self,
        d: Dict[str, Any],
        parent_key: str = "",
        sep: str = "."
    ) -> Dict[str, Any]:
        """
        Flatten nested dictionary.
        
        Args:
            d: Dictionary to flatten.
            parent_key: Parent key prefix.
            sep: Separator for nested keys.
            
        Returns:
            Flattened dictionary.
        """
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(
                    self._flatten_dict(v, new_key, sep=sep).items()
                )
            elif isinstance(v, list):
                for i, item in enumerate(v):
                    if isinstance(item, dict):
                        items.extend(
                            self._flatten_dict(
                                item, f"{new_key}[{i}]", sep=sep
                            ).items()
                        )
                    else:
                        items.append((f"{new_key}[{i}]", item))
            else:
                items.append((new_key, v))
        return dict(items)
    
    def get_extension(self) -> str:
        """Get CSV file extension."""
        return ".csv"


class TextReportFormatter(BaseReportFormatter):
    """Formats reports as human-readable text."""
    
    def format(self, report: ReportData) -> str:
        """Format report as plain text string."""
        lines = []
        
        # Header
        lines.append("=" * 80)
        lines.append(f"REPORT: {report.title}")
        lines.append("=" * 80)
        lines.append("")
        
        # Metadata
        lines.append(f"Report ID: {report.report_id}")
        lines.append(f"Type: {report.report_type.value}")
        lines.append(f"Generated: {report.timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append("")
        
        # Summary
        lines.append("-" * 40)
        lines.append("SUMMARY")
        lines.append("-" * 40)
        lines.append(report.summary)
        lines.append("")
        
        # Data
        lines.append("-" * 40)
        lines.append("DATA")
        lines.append("-" * 40)
        lines.append(self._format_dict(report.data, indent=0))
        
        # Metadata
        if report.metadata:
            lines.append("")
            lines.append("-" * 40)
            lines.append("METADATA")
            lines.append("-" * 40)
            lines.append(self._format_dict(report.metadata, indent=0))
        
        lines.append("")
        lines.append("=" * 80)
        lines.append("END OF REPORT")
        lines.append("=" * 80)
        
        return "\n".join(lines)
    
    def _format_dict(
        self,
        d: Dict[str, Any],
        indent: int = 0
    ) -> str:
        """
        Format dictionary as indented text.
        
        Args:
            d: Dictionary to format.
            indent: Current indentation level.
            
        Returns:
            Formatted text string.
        """
        lines = []
        prefix = "  " * indent
        
        for key, value in d.items():
            if isinstance(value, dict):
                lines.append(f"{prefix}{key}:")
                lines.append(self._format_dict(value, indent + 1))
            elif isinstance(value, list):
                lines.append(f"{prefix}{key}:")
                for i, item in enumerate(value):
                    if isinstance(item, dict):
                        lines.append(f"{prefix}  [{i}]:")
                        lines.append(self._format_dict(item, indent + 2))
                    else:
                        lines.append(f"{prefix}  - {item}")
            else:
                lines.append(f"{prefix}{key}: {value}")
        
        return "\n".join(lines)
    
    def get_extension(self) -> str:
        """Get text file extension."""
        return ".txt"


# =============================================================================
# Report Generator
# =============================================================================

class ReportGenerator:
    """
    Generates reports from resource and progress data.
    
    Creates formatted reports based on collected metrics
    and training progress information.
    """
    
    def __init__(self) -> None:
        """Initialize report generator."""
        self._report_counter = 0
        self._lock = threading.Lock()
    
    def _generate_report_id(self, report_type: ReportType) -> str:
        """
        Generate unique report ID.
        
        Args:
            report_type: Type of report.
            
        Returns:
            Unique report ID string.
        """
        with self._lock:
            self._report_counter += 1
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            return f"{report_type.value}_{timestamp}_{self._report_counter}"
    
    def generate_resource_report(
        self,
        metrics: AggregatedMetrics,
        history: Optional[List[AggregatedMetrics]] = None,
    ) -> ReportData:
        """
        Generate resource usage report.
        
        Args:
            metrics: Current aggregated metrics.
            history: Optional historical metrics.
            
        Returns:
            ReportData containing resource usage report.
        """
        report_id = self._generate_report_id(ReportType.RESOURCE_USAGE)
        
        # Build data section
        data = {
            "current": metrics.to_dict(),
            "alerts": metrics.alerts,
        }
        
        if history:
            data["history_count"] = len(history)
            data["history_summary"] = self._summarize_resource_history(history)
        
        # Generate summary
        summary_parts = []
        if metrics.cpu:
            summary_parts.append(f"CPU: {metrics.cpu.usage_percent:.1f}%")
        if metrics.memory:
            summary_parts.append(f"Memory: {metrics.memory.usage_percent:.1f}%")
        if metrics.gpu and metrics.gpu.additional_info.get("available", True):
            summary_parts.append(f"GPU: {metrics.gpu.usage_percent:.1f}%")
        if metrics.disk:
            summary_parts.append(f"Disk: {metrics.disk.usage_percent:.1f}%")
        
        summary = f"Resource usage snapshot: {', '.join(summary_parts)}"
        if metrics.alerts:
            summary += f". Active alerts: {len(metrics.alerts)}"
        
        return ReportData(
            report_id=report_id,
            report_type=ReportType.RESOURCE_USAGE,
            timestamp=datetime.now(),
            title="Resource Usage Report",
            summary=summary,
            data=data,
            metadata={
                "has_history": history is not None,
                "alert_count": len(metrics.alerts),
            },
        )
    
    def _summarize_resource_history(
        self,
        history: List[AggregatedMetrics]
    ) -> Dict[str, Any]:
        """
        Summarize resource usage history.
        
        Args:
            history: List of historical metrics.
            
        Returns:
            Summary dictionary.
        """
        if not history:
            return {}
        
        cpu_values = [m.cpu.usage_percent for m in history if m.cpu]
        memory_values = [m.memory.usage_percent for m in history if m.memory]
        
        return {
            "cpu": {
                "min": min(cpu_values) if cpu_values else 0,
                "max": max(cpu_values) if cpu_values else 0,
                "avg": sum(cpu_values) / len(cpu_values) if cpu_values else 0,
            },
            "memory": {
                "min": min(memory_values) if memory_values else 0,
                "max": max(memory_values) if memory_values else 0,
                "avg": sum(memory_values) / len(memory_values) if memory_values else 0,
            },
            "sample_count": len(history),
            "time_range": {
                "start": history[0].timestamp.isoformat(),
                "end": history[-1].timestamp.isoformat(),
            },
        }
    
    def generate_progress_report(
        self,
        tasks: Dict[str, TaskInfo],
        history: Optional[Dict[str, List[ProgressSnapshot]]] = None,
    ) -> ReportData:
        """
        Generate training progress report.
        
        Args:
            tasks: Dictionary of tracked tasks.
            history: Optional progress history per task.
            
        Returns:
            ReportData containing progress report.
        """
        report_id = self._generate_report_id(ReportType.TRAINING_PROGRESS)
        
        # Build data section
        data = {
            "tasks": {
                task_id: task.to_dict()
                for task_id, task in tasks.items()
            },
            "task_count": len(tasks),
        }
        
        # Categorize tasks by status
        status_counts = {}
        for task in tasks.values():
            status = task.status.value
            status_counts[status] = status_counts.get(status, 0) + 1
        data["status_summary"] = status_counts
        
        if history:
            data["history_summary"] = {
                task_id: len(snapshots)
                for task_id, snapshots in history.items()
            }
        
        # Generate summary
        active_tasks = [t for t in tasks.values() if t.status.value == "running"]
        summary = f"Tracking {len(tasks)} tasks. "
        if active_tasks:
            summary += f"{len(active_tasks)} currently running. "
            for task in active_tasks[:3]:  # Show up to 3 active tasks
                summary += f"{task.name}: {task.progress_percent:.1f}%. "
        
        return ReportData(
            report_id=report_id,
            report_type=ReportType.TRAINING_PROGRESS,
            timestamp=datetime.now(),
            title="Training Progress Report",
            summary=summary.strip(),
            data=data,
            metadata={
                "active_task_count": len(active_tasks),
                "has_history": history is not None,
            },
        )
    
    def generate_combined_report(
        self,
        metrics: Optional[AggregatedMetrics] = None,
        tasks: Optional[Dict[str, TaskInfo]] = None,
        additional_data: Optional[Dict[str, Any]] = None,
    ) -> ReportData:
        """
        Generate combined system status report.
        
        Args:
            metrics: Current resource metrics.
            tasks: Current tracked tasks.
            additional_data: Additional data to include.
            
        Returns:
            ReportData containing combined report.
        """
        report_id = self._generate_report_id(ReportType.COMBINED)
        
        data = {
            "generated_at": datetime.now().isoformat(),
        }
        
        summary_parts = []
        
        if metrics:
            data["resources"] = metrics.to_dict()
            if metrics.alerts:
                summary_parts.append(f"{len(metrics.alerts)} resource alerts")
        
        if tasks:
            data["tasks"] = {
                task_id: task.to_dict()
                for task_id, task in tasks.items()
            }
            active = len([t for t in tasks.values() if t.status.value == "running"])
            summary_parts.append(f"{active} active tasks")
        
        if additional_data:
            data["additional"] = additional_data
        
        summary = "Combined system report. " + ", ".join(summary_parts)
        
        return ReportData(
            report_id=report_id,
            report_type=ReportType.COMBINED,
            timestamp=datetime.now(),
            title="Combined System Status Report",
            summary=summary,
            data=data,
            metadata={
                "has_resources": metrics is not None,
                "has_tasks": tasks is not None,
                "has_additional": additional_data is not None,
            },
        )
    
    def generate_emergency_report(
        self,
        reason: str,
        metrics: Optional[AggregatedMetrics] = None,
        tasks: Optional[Dict[str, TaskInfo]] = None,
        error_info: Optional[Dict[str, Any]] = None,
    ) -> ReportData:
        """
        Generate emergency report for critical situations.
        
        Args:
            reason: Reason for emergency report.
            metrics: Current resource metrics.
            tasks: Current tracked tasks.
            error_info: Error information if applicable.
            
        Returns:
            ReportData containing emergency report.
        """
        report_id = self._generate_report_id(ReportType.EMERGENCY)
        
        data = {
            "reason": reason,
            "emergency_time": datetime.now().isoformat(),
        }
        
        if metrics:
            data["resources"] = metrics.to_dict()
            data["alerts"] = metrics.alerts
        
        if tasks:
            data["tasks"] = {
                task_id: task.to_dict()
                for task_id, task in tasks.items()
            }
        
        if error_info:
            data["error"] = error_info
        
        return ReportData(
            report_id=report_id,
            report_type=ReportType.EMERGENCY,
            timestamp=datetime.now(),
            title="EMERGENCY REPORT",
            summary=f"Emergency report generated: {reason}",
            data=data,
            metadata={
                "severity": "critical",
                "requires_attention": True,
            },
        )


# =============================================================================
# Report Manager
# =============================================================================

class ReportManager:
    """
    Manages report generation, storage, and auto-saving.
    
    Coordinates report generators, formatters, and persistence
    with configurable auto-save intervals.
    """
    
    def __init__(
        self,
        config: Optional[ReportConfig] = None,
    ) -> None:
        """
        Initialize report manager.
        
        Args:
            config: Report configuration. Uses defaults if not provided.
        """
        if config is None:
            resource_config = load_resource_config()
            config = resource_config.report
        
        self.config = config
        self.generator = ReportGenerator()
        
        # Initialize formatters
        self._formatters: Dict[ReportFormat, BaseReportFormatter] = {
            ReportFormat.JSON: JSONReportFormatter(),
            ReportFormat.YAML: YAMLReportFormatter(),
            ReportFormat.CSV: CSVReportFormatter(),
            ReportFormat.TEXT: TextReportFormatter(),
        }
        
        # Report storage
        self._reports: List[ReportData] = []
        self._metadata = ReportMetadata()
        self._lock = threading.Lock()
        
        # Auto-save thread
        self._auto_save_thread: Optional[threading.Thread] = None
        self._is_running = False
        
        # Callbacks
        self._callbacks: List[Callable[[ReportData], None]] = []
        
        # Ensure report directory exists
        self._ensure_report_dir()
    
    def _ensure_report_dir(self) -> None:
        """Ensure report directory exists."""
        report_dir = Path(self.config.report_dir)
        report_dir.mkdir(parents=True, exist_ok=True)
    
    def _get_formatter(
        self,
        format_type: Optional[Union[str, ReportFormat]] = None
    ) -> BaseReportFormatter:
        """
        Get formatter for specified format.
        
        Args:
            format_type: Format type or None for default.
            
        Returns:
            Report formatter instance.
        """
        if format_type is None:
            format_type = self.config.report_format
        
        if isinstance(format_type, str):
            try:
                format_type = ReportFormat(format_type)
            except ValueError:
                logger.warning(
                    f"Unknown format '{format_type}', using JSON"
                )
                format_type = ReportFormat.JSON
        
        return self._formatters.get(format_type, self._formatters[ReportFormat.JSON])
    
    def generate_and_save(
        self,
        report_type: ReportType,
        metrics: Optional[AggregatedMetrics] = None,
        tasks: Optional[Dict[str, TaskInfo]] = None,
        additional_data: Optional[Dict[str, Any]] = None,
        format_type: Optional[Union[str, ReportFormat]] = None,
    ) -> str:
        """
        Generate and save a report.
        
        Args:
            report_type: Type of report to generate.
            metrics: Resource metrics for report.
            tasks: Task information for report.
            additional_data: Additional data to include.
            format_type: Output format (uses config default if not specified).
            
        Returns:
            Path to saved report file.
        """
        # Generate report
        if report_type == ReportType.RESOURCE_USAGE:
            report = self.generator.generate_resource_report(metrics)
        elif report_type == ReportType.TRAINING_PROGRESS:
            report = self.generator.generate_progress_report(tasks or {})
        elif report_type == ReportType.COMBINED:
            report = self.generator.generate_combined_report(
                metrics, tasks, additional_data
            )
        else:
            report = self.generator.generate_combined_report(
                metrics, tasks, additional_data
            )
        
        # Save and return path
        return self.save_report(report, format_type)
    
    def save_report(
        self,
        report: ReportData,
        format_type: Optional[Union[str, ReportFormat]] = None,
    ) -> str:
        """
        Save a report to file.
        
        Args:
            report: Report data to save.
            format_type: Output format.
            
        Returns:
            Path to saved report file.
        """
        formatter = self._get_formatter(format_type)
        formatted_content = formatter.format(report)
        
        # Generate filename
        timestamp = report.timestamp.strftime("%Y%m%d_%H%M%S")
        filename = f"{report.report_type.value}_{timestamp}{formatter.get_extension()}"
        filepath = Path(self.config.report_dir) / filename
        
        # Write file
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(formatted_content)
        
        # Update metadata
        with self._lock:
            self._reports.append(report)
            self._metadata.total_reports += 1
            self._metadata.last_report_time = datetime.now()
            
            type_key = report.report_type.value
            self._metadata.report_types_count[type_key] = (
                self._metadata.report_types_count.get(type_key, 0) + 1
            )
            self._metadata.total_size_bytes += len(formatted_content.encode("utf-8"))
        
        # Notify callbacks
        for callback in self._callbacks:
            try:
                callback(report)
            except Exception as e:
                logger.error(f"Error in report callback: {e}")
        
        logger.info(f"Saved report to: {filepath}")
        return str(filepath)
    
    def save_emergency_report(
        self,
        reason: str,
        metrics: Optional[AggregatedMetrics] = None,
        tasks: Optional[Dict[str, TaskInfo]] = None,
        error_info: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Generate and save emergency report.
        
        Args:
            reason: Reason for emergency.
            metrics: Current resource metrics.
            tasks: Current tasks.
            error_info: Error information.
            
        Returns:
            Path to saved report file.
        """
        report = self.generator.generate_emergency_report(
            reason=reason,
            metrics=metrics,
            tasks=tasks,
            error_info=error_info,
        )
        
        # Always save as JSON for emergency reports
        return self.save_report(report, ReportFormat.JSON)
    
    def get_reports(
        self,
        report_type: Optional[ReportType] = None,
        limit: Optional[int] = None,
    ) -> List[ReportData]:
        """
        Get saved reports.
        
        Args:
            report_type: Filter by report type.
            limit: Maximum number of reports to return.
            
        Returns:
            List of ReportData objects.
        """
        with self._lock:
            reports = list(self._reports)
        
        if report_type:
            reports = [r for r in reports if r.report_type == report_type]
        
        if limit:
            reports = reports[-limit:]
        
        return reports
    
    def get_metadata(self) -> ReportMetadata:
        """
        Get report metadata.
        
        Returns:
            ReportMetadata instance.
        """
        with self._lock:
            return ReportMetadata(
                total_reports=self._metadata.total_reports,
                last_report_time=self._metadata.last_report_time,
                report_types_count=dict(self._metadata.report_types_count),
                total_size_bytes=self._metadata.total_size_bytes,
            )
    
    def register_callback(
        self,
        callback: Callable[[ReportData], None]
    ) -> None:
        """
        Register callback for report generation.
        
        Args:
            callback: Function to call when report is generated.
        """
        self._callbacks.append(callback)
    
    def start_auto_save(
        self,
        metrics_provider: Optional[Callable[[], AggregatedMetrics]] = None,
        tasks_provider: Optional[Callable[[], Dict[str, TaskInfo]]] = None,
    ) -> None:
        """
        Start auto-save background thread.
        
        Args:
            metrics_provider: Function that returns current metrics.
            tasks_provider: Function that returns current tasks.
        """
        if not self.config.enabled:
            logger.info("Report auto-save is disabled in configuration")
            return
        
        if self._is_running:
            logger.warning("Auto-save is already running")
            return
        
        self._is_running = True
        self._auto_save_thread = threading.Thread(
            target=self._auto_save_loop,
            args=(metrics_provider, tasks_provider),
            daemon=True,
            name="report_auto_save",
        )
        self._auto_save_thread.start()
        
        logger.info(
            f"Started report auto-save (interval: "
            f"{self.config.auto_save_interval_minutes} minutes)"
        )
    
    def stop_auto_save(self) -> None:
        """Stop auto-save background thread."""
        self._is_running = False
        if self._auto_save_thread and self._auto_save_thread.is_alive():
            self._auto_save_thread.join(
                timeout=self.config.auto_save_interval_minutes * 60 / 10
            )
        logger.info("Stopped report auto-save")
    
    def _auto_save_loop(
        self,
        metrics_provider: Optional[Callable[[], AggregatedMetrics]],
        tasks_provider: Optional[Callable[[], Dict[str, TaskInfo]]],
    ) -> None:
        """
        Background loop for auto-saving reports.
        
        Args:
            metrics_provider: Function that returns current metrics.
            tasks_provider: Function that returns current tasks.
        """
        interval_seconds = self.config.auto_save_interval_minutes * 60
        
        while self._is_running:
            try:
                metrics = metrics_provider() if metrics_provider else None
                tasks = tasks_provider() if tasks_provider else None
                
                self.generate_and_save(
                    report_type=ReportType.COMBINED,
                    metrics=metrics,
                    tasks=tasks,
                )
            except Exception as e:
                logger.error(f"Error in auto-save: {e}")
            
            # Sleep in small intervals to allow quick shutdown
            for _ in range(int(interval_seconds)):
                if not self._is_running:
                    break
                time.sleep(1)
    
    def cleanup_old_reports(
        self,
        max_age_days: int = 30,
        max_count: Optional[int] = None,
    ) -> int:
        """
        Clean up old report files.
        
        Args:
            max_age_days: Maximum age of reports to keep.
            max_count: Maximum number of reports to keep.
            
        Returns:
            Number of reports deleted.
        """
        report_dir = Path(self.config.report_dir)
        if not report_dir.exists():
            return 0
        
        deleted_count = 0
        now = datetime.now()
        
        # Get all report files sorted by modification time
        report_files = sorted(
            report_dir.iterdir(),
            key=lambda p: p.stat().st_mtime,
        )
        
        for filepath in report_files:
            if not filepath.is_file():
                continue
            
            # Check age
            mtime = datetime.fromtimestamp(filepath.stat().st_mtime)
            age_days = (now - mtime).days
            
            should_delete = False
            
            if max_age_days and age_days > max_age_days:
                should_delete = True
            
            if max_count and len(report_files) - deleted_count > max_count:
                should_delete = True
            
            if should_delete:
                try:
                    filepath.unlink()
                    deleted_count += 1
                except Exception as e:
                    logger.error(f"Failed to delete {filepath}: {e}")
        
        if deleted_count > 0:
            logger.info(f"Cleaned up {deleted_count} old report files")
        
        return deleted_count


# =============================================================================
# Module Exports
# =============================================================================

__all__ = [
    "ReportType",
    "ReportFormat",
    "ReportData",
    "ReportMetadata",
    "BaseReportFormatter",
    "JSONReportFormatter",
    "YAMLReportFormatter",
    "CSVReportFormatter",
    "TextReportFormatter",
    "ReportGenerator",
    "ReportManager",
]
