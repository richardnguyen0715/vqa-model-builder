"""
Resource Monitor Module

Provides real-time monitoring of system resources including CPU, GPU, memory, and disk.
Supports threshold-based alerting and historical data collection.
"""

import logging
import threading
import time
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Deque

import psutil

from .resource_config import (
    ResourceConfig,
    ResourceThresholds,
    ResourceType,
    MonitoringInterval,
)


logger = logging.getLogger(__name__)


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class ResourceSnapshot:
    """
    Snapshot of resource usage at a specific point in time.
    
    Attributes:
        timestamp: Time when snapshot was taken.
        resource_type: Type of resource being measured.
        usage_percent: Usage percentage (0-100).
        used_value: Absolute used value (bytes, cores, etc.).
        total_value: Total available value.
        available_value: Available/free value.
        additional_info: Additional resource-specific information.
    """
    
    timestamp: datetime
    resource_type: ResourceType
    usage_percent: float
    used_value: float
    total_value: float
    available_value: float
    additional_info: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert snapshot to dictionary representation.
        
        Returns:
            Dictionary containing snapshot data.
        """
        return {
            "timestamp": self.timestamp.isoformat(),
            "resource_type": self.resource_type.value,
            "usage_percent": self.usage_percent,
            "used_value": self.used_value,
            "total_value": self.total_value,
            "available_value": self.available_value,
            "additional_info": self.additional_info,
        }


@dataclass
class AggregatedMetrics:
    """
    Aggregated metrics from all resource monitors.
    
    Attributes:
        timestamp: Time when metrics were aggregated.
        cpu: CPU resource snapshot.
        memory: Memory resource snapshot.
        gpu: GPU resource snapshot (optional).
        disk: Disk resource snapshot.
        alerts: List of current alerts.
    """
    
    timestamp: datetime
    cpu: Optional[ResourceSnapshot] = None
    memory: Optional[ResourceSnapshot] = None
    gpu: Optional[ResourceSnapshot] = None
    disk: Optional[ResourceSnapshot] = None
    alerts: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert aggregated metrics to dictionary.
        
        Returns:
            Dictionary containing all metrics.
        """
        return {
            "timestamp": self.timestamp.isoformat(),
            "cpu": self.cpu.to_dict() if self.cpu else None,
            "memory": self.memory.to_dict() if self.memory else None,
            "gpu": self.gpu.to_dict() if self.gpu else None,
            "disk": self.disk.to_dict() if self.disk else None,
            "alerts": self.alerts,
        }


# =============================================================================
# Base Monitor Class
# =============================================================================

class BaseResourceMonitor(ABC):
    """
    Abstract base class for resource monitors.
    
    Provides common functionality for monitoring specific resources,
    including threshold checking and historical data storage.
    """
    
    def __init__(
        self,
        resource_type: ResourceType,
        warning_threshold: float,
        critical_threshold: float,
        check_interval: float = 5.0,
        history_size: int = 1000,
    ) -> None:
        """
        Initialize the base resource monitor.
        
        Args:
            resource_type: Type of resource being monitored.
            warning_threshold: Percentage threshold for warnings.
            critical_threshold: Percentage threshold for critical alerts.
            check_interval: Interval between checks in seconds.
            history_size: Maximum number of historical snapshots to store.
        """
        self.resource_type = resource_type
        self.warning_threshold = warning_threshold
        self.critical_threshold = critical_threshold
        self.check_interval = check_interval
        self.history_size = history_size
        
        self._history: Deque[ResourceSnapshot] = deque(maxlen=history_size)
        self._callbacks: List[Callable[[ResourceSnapshot, str], None]] = []
        self._is_running = False
        self._monitor_thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        
        self._last_snapshot: Optional[ResourceSnapshot] = None
        self._warning_issued = False
        self._critical_issued = False
    
    @abstractmethod
    def _collect_metrics(self) -> ResourceSnapshot:
        """
        Collect current resource metrics.
        
        Returns:
            ResourceSnapshot with current metrics.
        """
        pass
    
    def get_snapshot(self) -> ResourceSnapshot:
        """
        Get current resource snapshot.
        
        Returns:
            Current ResourceSnapshot.
        """
        snapshot = self._collect_metrics()
        
        with self._lock:
            self._last_snapshot = snapshot
            self._history.append(snapshot)
        
        self._check_thresholds(snapshot)
        return snapshot
    
    def _check_thresholds(self, snapshot: ResourceSnapshot) -> None:
        """
        Check if snapshot exceeds configured thresholds.
        
        Args:
            snapshot: Resource snapshot to check.
        """
        usage = snapshot.usage_percent
        
        # Check critical threshold
        if usage >= self.critical_threshold:
            if not self._critical_issued:
                self._critical_issued = True
                self._notify_callbacks(snapshot, "critical")
                logger.critical(
                    f"{self.resource_type.value.upper()} usage at {usage:.1f}% "
                    f"exceeds critical threshold of {self.critical_threshold}%"
                )
        else:
            self._critical_issued = False
        
        # Check warning threshold
        if usage >= self.warning_threshold and usage < self.critical_threshold:
            if not self._warning_issued:
                self._warning_issued = True
                self._notify_callbacks(snapshot, "warning")
                logger.warning(
                    f"{self.resource_type.value.upper()} usage at {usage:.1f}% "
                    f"exceeds warning threshold of {self.warning_threshold}%"
                )
        elif usage < self.warning_threshold:
            self._warning_issued = False
    
    def _notify_callbacks(self, snapshot: ResourceSnapshot, alert_level: str) -> None:
        """
        Notify registered callbacks of threshold breach.
        
        Args:
            snapshot: Resource snapshot that triggered alert.
            alert_level: Level of alert (warning or critical).
        """
        for callback in self._callbacks:
            try:
                callback(snapshot, alert_level)
            except Exception as e:
                logger.error(f"Error in callback: {e}")
    
    def register_callback(
        self,
        callback: Callable[[ResourceSnapshot, str], None]
    ) -> None:
        """
        Register callback for threshold alerts.
        
        Args:
            callback: Function to call when threshold is breached.
                     Receives (snapshot, alert_level) parameters.
        """
        self._callbacks.append(callback)
    
    def unregister_callback(
        self,
        callback: Callable[[ResourceSnapshot, str], None]
    ) -> None:
        """
        Unregister a previously registered callback.
        
        Args:
            callback: Callback function to remove.
        """
        if callback in self._callbacks:
            self._callbacks.remove(callback)
    
    def get_history(self) -> List[ResourceSnapshot]:
        """
        Get historical snapshots.
        
        Returns:
            List of historical ResourceSnapshots.
        """
        with self._lock:
            return list(self._history)
    
    def get_latest(self) -> Optional[ResourceSnapshot]:
        """
        Get the most recent snapshot.
        
        Returns:
            Most recent ResourceSnapshot or None.
        """
        with self._lock:
            return self._last_snapshot
    
    def clear_history(self) -> None:
        """Clear historical data."""
        with self._lock:
            self._history.clear()
            self._last_snapshot = None
    
    def start_monitoring(self) -> None:
        """Start background monitoring thread."""
        if self._is_running:
            logger.warning(
                f"{self.resource_type.value} monitor is already running"
            )
            return
        
        self._is_running = True
        self._monitor_thread = threading.Thread(
            target=self._monitoring_loop,
            daemon=True,
            name=f"{self.resource_type.value}_monitor"
        )
        self._monitor_thread.start()
        logger.info(f"Started {self.resource_type.value} monitoring")
    
    def stop_monitoring(self) -> None:
        """Stop background monitoring thread."""
        self._is_running = False
        if self._monitor_thread and self._monitor_thread.is_alive():
            self._monitor_thread.join(timeout=self.check_interval * 2)
        logger.info(f"Stopped {self.resource_type.value} monitoring")
    
    def _monitoring_loop(self) -> None:
        """Background monitoring loop."""
        while self._is_running:
            try:
                self.get_snapshot()
            except Exception as e:
                logger.error(f"Error collecting {self.resource_type.value} metrics: {e}")
            time.sleep(self.check_interval)
    
    @property
    def is_running(self) -> bool:
        """Check if monitoring is active."""
        return self._is_running


# =============================================================================
# CPU Monitor
# =============================================================================

class CPUMonitor(BaseResourceMonitor):
    """
    Monitor for CPU resource usage.
    
    Tracks CPU utilization percentage and per-core usage.
    """
    
    def __init__(
        self,
        warning_threshold: float = 70.0,
        critical_threshold: float = 90.0,
        check_interval: float = 5.0,
        history_size: int = 1000,
    ) -> None:
        """
        Initialize CPU monitor.
        
        Args:
            warning_threshold: CPU usage percentage for warning.
            critical_threshold: CPU usage percentage for critical alert.
            check_interval: Interval between checks in seconds.
            history_size: Maximum historical snapshots to store.
        """
        super().__init__(
            resource_type=ResourceType.CPU,
            warning_threshold=warning_threshold,
            critical_threshold=critical_threshold,
            check_interval=check_interval,
            history_size=history_size,
        )
        
        # Initialize CPU percent to get accurate readings
        psutil.cpu_percent(interval=None)
    
    def _collect_metrics(self) -> ResourceSnapshot:
        """
        Collect current CPU metrics.
        
        Returns:
            ResourceSnapshot with CPU usage data.
        """
        cpu_percent = psutil.cpu_percent(interval=None)
        cpu_count = psutil.cpu_count()
        cpu_count_logical = psutil.cpu_count(logical=True)
        per_cpu = psutil.cpu_percent(percpu=True)
        
        # Get CPU frequency if available
        try:
            cpu_freq = psutil.cpu_freq()
            freq_info = {
                "current_mhz": cpu_freq.current if cpu_freq else None,
                "min_mhz": cpu_freq.min if cpu_freq else None,
                "max_mhz": cpu_freq.max if cpu_freq else None,
            }
        except Exception:
            freq_info = {}
        
        return ResourceSnapshot(
            timestamp=datetime.now(),
            resource_type=ResourceType.CPU,
            usage_percent=cpu_percent,
            used_value=cpu_percent,
            total_value=100.0,
            available_value=100.0 - cpu_percent,
            additional_info={
                "cpu_count_physical": cpu_count,
                "cpu_count_logical": cpu_count_logical,
                "per_cpu_percent": per_cpu,
                **freq_info,
            },
        )


# =============================================================================
# Memory Monitor
# =============================================================================

class MemoryMonitor(BaseResourceMonitor):
    """
    Monitor for system memory (RAM) usage.
    
    Tracks memory utilization and swap usage.
    """
    
    def __init__(
        self,
        warning_threshold: float = 70.0,
        critical_threshold: float = 90.0,
        check_interval: float = 5.0,
        history_size: int = 1000,
    ) -> None:
        """
        Initialize memory monitor.
        
        Args:
            warning_threshold: Memory usage percentage for warning.
            critical_threshold: Memory usage percentage for critical alert.
            check_interval: Interval between checks in seconds.
            history_size: Maximum historical snapshots to store.
        """
        super().__init__(
            resource_type=ResourceType.MEMORY,
            warning_threshold=warning_threshold,
            critical_threshold=critical_threshold,
            check_interval=check_interval,
            history_size=history_size,
        )
    
    def _collect_metrics(self) -> ResourceSnapshot:
        """
        Collect current memory metrics.
        
        Returns:
            ResourceSnapshot with memory usage data.
        """
        vm = psutil.virtual_memory()
        swap = psutil.swap_memory()
        
        return ResourceSnapshot(
            timestamp=datetime.now(),
            resource_type=ResourceType.MEMORY,
            usage_percent=vm.percent,
            used_value=vm.used,
            total_value=vm.total,
            available_value=vm.available,
            additional_info={
                "used_mb": vm.used / (1024 ** 2),
                "total_mb": vm.total / (1024 ** 2),
                "available_mb": vm.available / (1024 ** 2),
                "cached_mb": getattr(vm, "cached", 0) / (1024 ** 2),
                "buffers_mb": getattr(vm, "buffers", 0) / (1024 ** 2),
                "swap_used_mb": swap.used / (1024 ** 2),
                "swap_total_mb": swap.total / (1024 ** 2),
                "swap_percent": swap.percent,
            },
        )


# =============================================================================
# GPU Monitor
# =============================================================================

class GPUMonitor(BaseResourceMonitor):
    """
    Monitor for GPU resource usage.
    
    Tracks GPU memory and utilization. Supports NVIDIA GPUs via pynvml.
    Falls back gracefully if GPU monitoring is not available.
    """
    
    def __init__(
        self,
        warning_threshold: float = 70.0,
        critical_threshold: float = 90.0,
        check_interval: float = 5.0,
        history_size: int = 1000,
        device_index: int = 0,
    ) -> None:
        """
        Initialize GPU monitor.
        
        Args:
            warning_threshold: GPU memory percentage for warning.
            critical_threshold: GPU memory percentage for critical alert.
            check_interval: Interval between checks in seconds.
            history_size: Maximum historical snapshots to store.
            device_index: GPU device index to monitor.
        """
        super().__init__(
            resource_type=ResourceType.GPU,
            warning_threshold=warning_threshold,
            critical_threshold=critical_threshold,
            check_interval=check_interval,
            history_size=history_size,
        )
        
        self.device_index = device_index
        self._nvml_initialized = False
        self._torch_available = False
        
        self._initialize_gpu_monitoring()
    
    def _initialize_gpu_monitoring(self) -> None:
        """Initialize GPU monitoring backend."""
        # Try pynvml first
        try:
            import pynvml
            pynvml.nvmlInit()
            self._nvml_initialized = True
            self._pynvml = pynvml
            logger.info("GPU monitoring initialized via pynvml")
            return
        except ImportError:
            logger.debug("pynvml not available")
        except Exception as e:
            logger.debug(f"pynvml initialization failed: {e}")
        
        # Fall back to PyTorch
        try:
            import torch
            if torch.cuda.is_available():
                self._torch_available = True
                logger.info("GPU monitoring initialized via PyTorch")
                return
        except ImportError:
            logger.debug("PyTorch not available")
        
        logger.warning("GPU monitoring not available. No GPU backend found.")
    
    def _collect_metrics(self) -> ResourceSnapshot:
        """
        Collect current GPU metrics.
        
        Returns:
            ResourceSnapshot with GPU usage data.
        """
        if self._nvml_initialized:
            return self._collect_nvml_metrics()
        elif self._torch_available:
            return self._collect_torch_metrics()
        else:
            return self._create_unavailable_snapshot()
    
    def _collect_nvml_metrics(self) -> ResourceSnapshot:
        """Collect GPU metrics using pynvml."""
        try:
            handle = self._pynvml.nvmlDeviceGetHandleByIndex(self.device_index)
            mem_info = self._pynvml.nvmlDeviceGetMemoryInfo(handle)
            util = self._pynvml.nvmlDeviceGetUtilizationRates(handle)
            
            # Get additional info
            try:
                name = self._pynvml.nvmlDeviceGetName(handle)
                if isinstance(name, bytes):
                    name = name.decode("utf-8")
            except Exception:
                name = "Unknown"
            
            try:
                temp = self._pynvml.nvmlDeviceGetTemperature(
                    handle,
                    self._pynvml.NVML_TEMPERATURE_GPU
                )
            except Exception:
                temp = None
            
            try:
                power = self._pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0
            except Exception:
                power = None
            
            memory_percent = (mem_info.used / mem_info.total) * 100
            
            return ResourceSnapshot(
                timestamp=datetime.now(),
                resource_type=ResourceType.GPU,
                usage_percent=memory_percent,
                used_value=mem_info.used,
                total_value=mem_info.total,
                available_value=mem_info.free,
                additional_info={
                    "device_index": self.device_index,
                    "device_name": name,
                    "used_mb": mem_info.used / (1024 ** 2),
                    "total_mb": mem_info.total / (1024 ** 2),
                    "free_mb": mem_info.free / (1024 ** 2),
                    "gpu_utilization_percent": util.gpu,
                    "memory_utilization_percent": util.memory,
                    "temperature_celsius": temp,
                    "power_watts": power,
                },
            )
        except Exception as e:
            logger.error(f"Error collecting NVML metrics: {e}")
            return self._create_unavailable_snapshot()
    
    def _collect_torch_metrics(self) -> ResourceSnapshot:
        """Collect GPU metrics using PyTorch."""
        try:
            import torch
            
            if not torch.cuda.is_available():
                return self._create_unavailable_snapshot()
            
            device = torch.device(f"cuda:{self.device_index}")
            
            mem_allocated = torch.cuda.memory_allocated(device)
            mem_reserved = torch.cuda.memory_reserved(device)
            
            # Get total memory
            props = torch.cuda.get_device_properties(device)
            total_memory = props.total_memory
            
            memory_percent = (mem_allocated / total_memory) * 100 if total_memory > 0 else 0
            
            return ResourceSnapshot(
                timestamp=datetime.now(),
                resource_type=ResourceType.GPU,
                usage_percent=memory_percent,
                used_value=mem_allocated,
                total_value=total_memory,
                available_value=total_memory - mem_allocated,
                additional_info={
                    "device_index": self.device_index,
                    "device_name": props.name,
                    "allocated_mb": mem_allocated / (1024 ** 2),
                    "reserved_mb": mem_reserved / (1024 ** 2),
                    "total_mb": total_memory / (1024 ** 2),
                    "compute_capability": f"{props.major}.{props.minor}",
                },
            )
        except Exception as e:
            logger.error(f"Error collecting PyTorch GPU metrics: {e}")
            return self._create_unavailable_snapshot()
    
    def _create_unavailable_snapshot(self) -> ResourceSnapshot:
        """Create snapshot indicating GPU monitoring is unavailable."""
        return ResourceSnapshot(
            timestamp=datetime.now(),
            resource_type=ResourceType.GPU,
            usage_percent=0.0,
            used_value=0.0,
            total_value=0.0,
            available_value=0.0,
            additional_info={
                "available": False,
                "reason": "GPU monitoring not available",
            },
        )
    
    def __del__(self) -> None:
        """Cleanup GPU monitoring resources."""
        if self._nvml_initialized:
            try:
                self._pynvml.nvmlShutdown()
            except Exception:
                pass


# =============================================================================
# Disk Monitor
# =============================================================================

class DiskMonitor(BaseResourceMonitor):
    """
    Monitor for disk storage usage.
    
    Tracks disk space utilization for specified paths.
    """
    
    def __init__(
        self,
        warning_threshold: float = 80.0,
        critical_threshold: float = 95.0,
        check_interval: float = 60.0,
        history_size: int = 1000,
        monitor_path: str = "/",
    ) -> None:
        """
        Initialize disk monitor.
        
        Args:
            warning_threshold: Disk usage percentage for warning.
            critical_threshold: Disk usage percentage for critical alert.
            check_interval: Interval between checks in seconds.
            history_size: Maximum historical snapshots to store.
            monitor_path: Path to monitor for disk usage.
        """
        super().__init__(
            resource_type=ResourceType.DISK,
            warning_threshold=warning_threshold,
            critical_threshold=critical_threshold,
            check_interval=check_interval,
            history_size=history_size,
        )
        
        self.monitor_path = monitor_path
    
    def _collect_metrics(self) -> ResourceSnapshot:
        """
        Collect current disk metrics.
        
        Returns:
            ResourceSnapshot with disk usage data.
        """
        try:
            disk = psutil.disk_usage(self.monitor_path)
            
            # Get IO counters if available
            try:
                io = psutil.disk_io_counters()
                io_info = {
                    "read_bytes": io.read_bytes,
                    "write_bytes": io.write_bytes,
                    "read_count": io.read_count,
                    "write_count": io.write_count,
                }
            except Exception:
                io_info = {}
            
            return ResourceSnapshot(
                timestamp=datetime.now(),
                resource_type=ResourceType.DISK,
                usage_percent=disk.percent,
                used_value=disk.used,
                total_value=disk.total,
                available_value=disk.free,
                additional_info={
                    "monitor_path": self.monitor_path,
                    "used_gb": disk.used / (1024 ** 3),
                    "total_gb": disk.total / (1024 ** 3),
                    "free_gb": disk.free / (1024 ** 3),
                    **io_info,
                },
            )
        except Exception as e:
            logger.error(f"Error collecting disk metrics: {e}")
            return ResourceSnapshot(
                timestamp=datetime.now(),
                resource_type=ResourceType.DISK,
                usage_percent=0.0,
                used_value=0.0,
                total_value=0.0,
                available_value=0.0,
                additional_info={
                    "error": str(e),
                },
            )


# =============================================================================
# Aggregated Resource Monitor
# =============================================================================

class ResourceMonitor:
    """
    Aggregated resource monitor combining all individual monitors.
    
    Provides unified interface for monitoring all system resources
    and triggering callbacks based on configurable thresholds.
    """
    
    def __init__(
        self,
        config: Optional[ResourceConfig] = None,
    ) -> None:
        """
        Initialize aggregated resource monitor.
        
        Args:
            config: Resource configuration. Uses defaults if not provided.
        """
        if config is None:
            from .resource_config import load_resource_config
            config = load_resource_config()
        
        self.config = config
        self._callbacks: List[Callable[[AggregatedMetrics], None]] = []
        self._alert_callbacks: List[Callable[[ResourceSnapshot, str], None]] = []
        self._is_running = False
        self._aggregate_thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        self._latest_metrics: Optional[AggregatedMetrics] = None
        
        # Initialize individual monitors
        thresholds = config.thresholds
        intervals = config.intervals
        
        self.cpu_monitor = CPUMonitor(
            warning_threshold=thresholds.cpu_warning_percent,
            critical_threshold=thresholds.cpu_critical_percent,
            check_interval=intervals.cpu_seconds,
        )
        
        self.memory_monitor = MemoryMonitor(
            warning_threshold=thresholds.memory_warning_percent,
            critical_threshold=thresholds.memory_critical_percent,
            check_interval=intervals.memory_seconds,
        )
        
        self.gpu_monitor = GPUMonitor(
            warning_threshold=thresholds.gpu_memory_warning_percent,
            critical_threshold=thresholds.gpu_memory_critical_percent,
            check_interval=intervals.gpu_seconds,
        )
        
        self.disk_monitor = DiskMonitor(
            warning_threshold=thresholds.disk_warning_percent,
            critical_threshold=thresholds.disk_critical_percent,
            check_interval=intervals.disk_seconds,
        )
        
        # Register alert callbacks to propagate to aggregated callbacks
        for monitor in [
            self.cpu_monitor,
            self.memory_monitor,
            self.gpu_monitor,
            self.disk_monitor
        ]:
            monitor.register_callback(self._on_alert)
    
    def _on_alert(self, snapshot: ResourceSnapshot, alert_level: str) -> None:
        """
        Handle alerts from individual monitors.
        
        Args:
            snapshot: Resource snapshot that triggered alert.
            alert_level: Level of alert (warning or critical).
        """
        for callback in self._alert_callbacks:
            try:
                callback(snapshot, alert_level)
            except Exception as e:
                logger.error(f"Error in alert callback: {e}")
    
    def get_all_metrics(self) -> AggregatedMetrics:
        """
        Get current metrics from all monitors.
        
        Returns:
            AggregatedMetrics containing all resource snapshots.
        """
        cpu = self.cpu_monitor.get_snapshot()
        memory = self.memory_monitor.get_snapshot()
        gpu = self.gpu_monitor.get_snapshot()
        disk = self.disk_monitor.get_snapshot()
        
        # Collect alerts
        alerts = []
        thresholds = self.config.thresholds
        
        if cpu.usage_percent >= thresholds.cpu_critical_percent:
            alerts.append(f"CRITICAL: CPU at {cpu.usage_percent:.1f}%")
        elif cpu.usage_percent >= thresholds.cpu_warning_percent:
            alerts.append(f"WARNING: CPU at {cpu.usage_percent:.1f}%")
        
        if memory.usage_percent >= thresholds.memory_critical_percent:
            alerts.append(f"CRITICAL: Memory at {memory.usage_percent:.1f}%")
        elif memory.usage_percent >= thresholds.memory_warning_percent:
            alerts.append(f"WARNING: Memory at {memory.usage_percent:.1f}%")
        
        if gpu.additional_info.get("available", True):
            if gpu.usage_percent >= thresholds.gpu_memory_critical_percent:
                alerts.append(f"CRITICAL: GPU memory at {gpu.usage_percent:.1f}%")
            elif gpu.usage_percent >= thresholds.gpu_memory_warning_percent:
                alerts.append(f"WARNING: GPU memory at {gpu.usage_percent:.1f}%")
        
        if disk.usage_percent >= thresholds.disk_critical_percent:
            alerts.append(f"CRITICAL: Disk at {disk.usage_percent:.1f}%")
        elif disk.usage_percent >= thresholds.disk_warning_percent:
            alerts.append(f"WARNING: Disk at {disk.usage_percent:.1f}%")
        
        metrics = AggregatedMetrics(
            timestamp=datetime.now(),
            cpu=cpu,
            memory=memory,
            gpu=gpu,
            disk=disk,
            alerts=alerts,
        )
        
        with self._lock:
            self._latest_metrics = metrics
        
        return metrics
    
    def get_latest_metrics(self) -> Optional[AggregatedMetrics]:
        """
        Get the most recently collected aggregated metrics.
        
        Returns:
            Most recent AggregatedMetrics or None.
        """
        with self._lock:
            return self._latest_metrics
    
    def register_metrics_callback(
        self,
        callback: Callable[[AggregatedMetrics], None]
    ) -> None:
        """
        Register callback for periodic metrics updates.
        
        Args:
            callback: Function to call with aggregated metrics.
        """
        self._callbacks.append(callback)
    
    def register_alert_callback(
        self,
        callback: Callable[[ResourceSnapshot, str], None]
    ) -> None:
        """
        Register callback for threshold alerts.
        
        Args:
            callback: Function to call when threshold is breached.
        """
        self._alert_callbacks.append(callback)
    
    def start(self) -> None:
        """Start all monitors and aggregation."""
        if self._is_running:
            logger.warning("Resource monitor is already running")
            return
        
        self._is_running = True
        
        # Start individual monitors
        self.cpu_monitor.start_monitoring()
        self.memory_monitor.start_monitoring()
        self.gpu_monitor.start_monitoring()
        self.disk_monitor.start_monitoring()
        
        # Start aggregation thread
        self._aggregate_thread = threading.Thread(
            target=self._aggregation_loop,
            daemon=True,
            name="resource_aggregator"
        )
        self._aggregate_thread.start()
        
        logger.info("Started all resource monitors")
    
    def stop(self) -> None:
        """Stop all monitors and aggregation."""
        self._is_running = False
        
        # Stop aggregation thread
        if self._aggregate_thread and self._aggregate_thread.is_alive():
            self._aggregate_thread.join(
                timeout=self.config.intervals.aggregate_seconds * 2
            )
        
        # Stop individual monitors
        self.cpu_monitor.stop_monitoring()
        self.memory_monitor.stop_monitoring()
        self.gpu_monitor.stop_monitoring()
        self.disk_monitor.stop_monitoring()
        
        logger.info("Stopped all resource monitors")
    
    def _aggregation_loop(self) -> None:
        """Background loop for aggregating metrics."""
        while self._is_running:
            try:
                metrics = self.get_all_metrics()
                for callback in self._callbacks:
                    try:
                        callback(metrics)
                    except Exception as e:
                        logger.error(f"Error in metrics callback: {e}")
            except Exception as e:
                logger.error(f"Error in aggregation loop: {e}")
            
            time.sleep(self.config.intervals.aggregate_seconds)
    
    def is_critical(self) -> bool:
        """
        Check if any resource is at critical level.
        
        Returns:
            True if any resource exceeds critical threshold.
        """
        metrics = self.get_latest_metrics()
        if metrics is None:
            return False
        
        return any("CRITICAL" in alert for alert in metrics.alerts)
    
    def is_warning(self) -> bool:
        """
        Check if any resource is at warning level.
        
        Returns:
            True if any resource exceeds warning threshold.
        """
        metrics = self.get_latest_metrics()
        if metrics is None:
            return False
        
        return len(metrics.alerts) > 0


# =============================================================================
# Module Exports
# =============================================================================

__all__ = [
    "ResourceSnapshot",
    "AggregatedMetrics",
    "BaseResourceMonitor",
    "CPUMonitor",
    "MemoryMonitor",
    "GPUMonitor",
    "DiskMonitor",
    "ResourceMonitor",
]
