from src.middleware.config import warn_exception_config

print(f"Memory Monitor Config: max_ram_percentage={warn_exception_config.get('memory_monitor', {}).get('max_ram_percentage', '')}, warn_threshold_percent={warn_exception_config.get('memory_monitor', {}).get('warn_threshold_percent', '')}")