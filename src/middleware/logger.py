
import logging
from typing import Optional

from utils.logging_setup import setup_logging
from utils.path_management import ROOT_DIR


DEFAULT_CONFIG_PATH = ROOT_DIR / 'configs' / 'logging_configs.yaml'

def get_logger(name: str, config_path: Optional[str] = None) -> logging.Logger:
	"""Return a configured logger by name. Loads config the first time it's called.

	Args:
		name: Logger name matching a key in the YAML `loggers` section (e.g. 'data_loader_logger').
		config_path: Optional path to YAML config; defaults to `configs/logging_configs.yaml` in project root.
	"""
	config_path = config_path or str(DEFAULT_CONFIG_PATH)
	return setup_logging(config_path, name)


def test_loggers():
    # Example usage
    logger = get_logger('data_loader_logger')
    logger.info('Logger is configured and ready to use.')
    model_logger = get_logger('model_trainer_logger')
    model_logger.info('Model trainer logger is also configured.')


data_loader_logger = get_logger('data_loader_logger')
model_trainer_logger = get_logger('model_trainer_logger')