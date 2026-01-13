from src.middleware.config import memory_max_ram_percent, memory_warn_threshold_percent
from src.exception.data_exception_handling import MemoryMonitor


memory_monitor = MemoryMonitor(
    max_ram_percentage=memory_max_ram_percent,
    warn_threshold_percent=memory_warn_threshold_percent
)