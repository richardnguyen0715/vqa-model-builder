"""
Data Loading Exceptions and Warnings.
Handles memory overflow, data validation, and resource constraints.
"""

import warnings
import psutil
import sys
from typing import Optional


class DataLoadingException(Exception):
    """Base exception for data loading errors."""
    pass


class MemoryOverflowException(DataLoadingException):
    """Exception raised when RAM usage exceeds the configured threshold."""
    
    def __init__(self, current_usage_percent: float, max_threshold_percent: float):
        """
        Args:
            current_usage_percent: Current RAM usage percentage
            max_threshold_percent: Maximum allowed RAM usage percentage
        """
        self.current_usage = current_usage_percent
        self.max_threshold = max_threshold_percent
        
        message = (
            f"Memory Overflow Detected! "
            f"Current RAM usage: {current_usage_percent:.2f}% "
            f"exceeds the configured threshold of {max_threshold_percent}%. "
            f"Process will be terminated to prevent system crash."
        )
        super().__init__(message)


class DataValidationException(DataLoadingException):
    """Exception raised when data validation fails."""
    pass


class LargeFileWarning(UserWarning):
    """Warning issued when large files are detected during data loading."""
    pass


class MemoryWarning(UserWarning):
    """Warning issued when memory usage approaches the threshold."""
    pass


class MemoryMonitor:
    """
    Handler class for data loading warnings.
    Monitors RAM usage and issues warnings based on configured thresholds.
    """
    
    def __init__(self, max_ram_percentage: float = 80.0, warn_threshold_percent: float = 70.0):
        """
        Args:
            max_ram_percentage: Maximum allowed RAM usage percentage before killing process
            warn_threshold_percent: Warning issued when RAM reaches this percentage
        """
        self.max_ram_percentage = max_ram_percentage
        self.warn_threshold_percent = warn_threshold_percent
        self.initial_memory = self._get_memory_usage()
    
    def _get_memory_usage(self) -> float:
        """Get current RAM usage percentage."""
        return psutil.virtual_memory().percent
    
    def _get_memory_used_mb(self) -> float:
        """Get memory used in MB."""
        return psutil.virtual_memory().used / (1024 ** 2)
    
    def _get_available_memory_mb(self) -> float:
        """Get available memory in MB."""
        return psutil.virtual_memory().available / (1024 ** 2)
    
    def check_memory_usage(self, operation_name: str = "Data Loading") -> None:
        """
        Check current memory usage and raise exception if threshold exceeded.
        
        Args:
            operation_name: Name of the operation being performed
            
        Raises:
            MemoryOverflowException: If RAM usage exceeds max_ram_percentage
        """
        current_usage = self._get_memory_usage()
        
        # Issue warning if approaching max threshold
        if current_usage >= self.warn_threshold_percent:
            warnings.warn(
                f"{operation_name}: RAM usage at {current_usage:.2f}% "
                f"(Available: {self._get_available_memory_mb():.2f} MB). "
                f"Approaching threshold of {self.max_ram_percentage}%",
                MemoryWarning,
                stacklevel=2
            )
        
        # Kill process if exceeds max threshold
        if current_usage >= self.max_ram_percentage:
            raise MemoryOverflowException(
                current_usage_percent=current_usage,
                max_threshold_percent=self.max_ram_percentage
            )
    
    def check_file_size(self, file_size_mb: float, warn_on_large_files: bool = True) -> None:
        """
        Check file size and issue warning if file is considered large.
        
        Args:
            file_size_mb: File size in megabytes
            warn_on_large_files: Whether to issue warnings for large files
        """
        if warn_on_large_files and file_size_mb > 500:  # 500 MB threshold
            warnings.warn(
                f"Large file detected: {file_size_mb:.2f} MB. "
                f"This may impact memory usage during processing.",
                LargeFileWarning,
                stacklevel=2
            )
    
    def get_memory_report(self) -> dict:
        """
        Get a detailed memory usage report.
        
        Returns:
            Dictionary containing memory statistics
        """
        vm = psutil.virtual_memory()
        return {
            'total_memory_mb': vm.total / (1024 ** 2),
            'used_memory_mb': vm.used / (1024 ** 2),
            'available_memory_mb': vm.available / (1024 ** 2),
            'used_percentage': vm.percent,
            'initial_memory_percentage': self.initial_memory,
            'memory_increase_percentage': vm.percent - self.initial_memory,
            'max_allowed_percentage': self.max_ram_percentage,
            'warning_threshold_percentage': self.warn_threshold_percent
        }
    
    def kill_process(self, message: str = "Process terminated due to memory overflow") -> None:
        """
        Terminate the current process.
        
        Args:
            message: Message to display before termination
        """
        print(f"\n[FATAL] {message}", file=sys.stderr)
        sys.exit(1)


def handle_memory_overflow(func):
    """
    Decorator to automatically check memory usage before and after function execution.
    
    Usage:
        @handle_memory_overflow
        def load_large_dataset():
            ...
    """
    def wrapper(*args, **kwargs):
        global memory_monitor
        
        try:
            memory_monitor.check_memory_usage(f"Before executing {func.__name__}")
            result = func(*args, **kwargs)
            memory_monitor.check_memory_usage(f"After executing {func.__name__}")
            return result
        except MemoryOverflowException as e:
            memory_monitor.kill_process(str(e))
    
    return wrapper
