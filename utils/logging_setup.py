import yaml
import logging
import logging.config
from pathlib import Path
from typing import Optional
from datetime import datetime

_configured = False


def setup_logging(config_path: str, log_engine: Optional[str] = None) -> logging.Logger:
    """Load YAML logging configuration (dictConfig), ensure file paths exist, and return a logger.

    Args:
        config_path: Path to YAML file containing a logging.dictConfig-style mapping.
        log_engine: Name of the logger to return. If None, returns the root logger.

    Returns:
        logging.Logger: The requested logger instance.
    """
    global _configured

    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)

    # Generate timestamp for this run
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Update handler filenames with timestamp
    handlers = config.get('handlers', {}) or {}
    for handler in handlers.values():
        filename = handler.get('filename')
        if filename:
            # Insert timestamp before file extension
            path = Path(filename)
            new_filename = path.parent / f"{path.stem}_{timestamp}{path.suffix}"
            handler['filename'] = str(new_filename)
            new_filename.parent.mkdir(parents=True, exist_ok=True)
            
    if not _configured:
        logging.config.dictConfig(config)
        _configured = True

    logger_name = log_engine if log_engine else config.get('root', {}).get('name', '')
    return logging.getLogger(logger_name)


# Example usage:
# from src.utils.logging_setup import setup_logging
# logger = setup_logging('configs/logging_configs.yaml', 'data_loader_logger')