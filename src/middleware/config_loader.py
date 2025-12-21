"""
Configuration Loader Module
Provides centralized configuration loading and management.
Loads all configurations from YAML files in the configs directory.
"""

import os
import yaml
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Union
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


# =============================================================================
# Configuration File Paths
# =============================================================================

def get_project_root() -> Path:
    """
    Get the project root directory.
    
    Returns:
        Path to project root directory.
    """
    current_file = Path(__file__).resolve()
    # Navigate up from src/middleware to project root
    return current_file.parent.parent.parent


def get_config_dir() -> Path:
    """
    Get the configuration directory path.
    
    Returns:
        Path to configs directory.
    """
    return get_project_root() / "configs"


# =============================================================================
# YAML Loading Utilities
# =============================================================================

def load_yaml(config_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Load a YAML configuration file.
    
    Args:
        config_path: Path to the YAML file.
        
    Returns:
        Dictionary containing configuration values.
        
    Raises:
        FileNotFoundError: If config file does not exist.
        yaml.YAMLError: If YAML parsing fails.
    """
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    
    return config or {}


def merge_configs(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """
    Recursively merge two configuration dictionaries.
    Override values take precedence over base values.
    
    Args:
        base: Base configuration dictionary.
        override: Override configuration dictionary.
        
    Returns:
        Merged configuration dictionary.
    """
    result = base.copy()
    
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = merge_configs(result[key], value)
        else:
            result[key] = value
    
    return result


# =============================================================================
# Configuration Classes
# =============================================================================

@dataclass
class ConfigPaths:
    """
    Container for configuration file paths.
    
    Attributes:
        data: Path to data configuration file.
        model: Path to model configuration file.
        training: Path to training configuration file.
        checkpoint: Path to checkpoint configuration file.
        inference: Path to inference configuration file.
        logging: Path to logging configuration file.
        warning: Path to warning configuration file.
    """
    data: Path = field(default_factory=lambda: get_config_dir() / "data_configs.yaml")
    model: Path = field(default_factory=lambda: get_config_dir() / "model_configs.yaml")
    training: Path = field(default_factory=lambda: get_config_dir() / "training_configs.yaml")
    checkpoint: Path = field(default_factory=lambda: get_config_dir() / "checkpoint_configs.yaml")
    inference: Path = field(default_factory=lambda: get_config_dir() / "inference_configs.yaml")
    logging: Path = field(default_factory=lambda: get_config_dir() / "logging_configs.yaml")
    warning: Path = field(default_factory=lambda: get_config_dir() / "warning_configs.yaml")


class ConfigManager:
    """
    Centralized configuration manager.
    Loads and provides access to all configuration values.
    
    This class implements singleton pattern to ensure consistent
    configuration access throughout the application.
    """
    
    _instance: Optional["ConfigManager"] = None
    _initialized: bool = False
    
    def __new__(cls) -> "ConfigManager":
        """Create singleton instance."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self) -> None:
        """Initialize configuration manager."""
        if self._initialized:
            return
        
        self.paths = ConfigPaths()
        self._configs: Dict[str, Dict[str, Any]] = {}
        self._load_all_configs()
        self._initialized = True
        
        logger.info("ConfigManager initialized successfully")
    
    def _load_all_configs(self) -> None:
        """Load all configuration files."""
        config_files = {
            "data": self.paths.data,
            "model": self.paths.model,
            "training": self.paths.training,
            "checkpoint": self.paths.checkpoint,
            "inference": self.paths.inference,
            "logging": self.paths.logging,
            "warning": self.paths.warning,
        }
        
        for name, path in config_files.items():
            try:
                if path.exists():
                    self._configs[name] = load_yaml(path)
                    logger.debug(f"Loaded config: {name} from {path}")
                else:
                    self._configs[name] = {}
                    logger.warning(f"Config file not found: {path}")
            except Exception as e:
                logger.error(f"Failed to load config {name}: {e}")
                self._configs[name] = {}
    
    def reload(self) -> None:
        """Reload all configuration files."""
        self._configs.clear()
        self._load_all_configs()
        logger.info("Configuration files reloaded")
    
    def get(self, config_name: str, key: str = None, default: Any = None) -> Any:
        """
        Get configuration value.
        
        Args:
            config_name: Name of configuration (data, model, training, etc.).
            key: Optional dot-separated key path (e.g., "optimizer.learning_rate").
            default: Default value if key not found.
            
        Returns:
            Configuration value or default.
        """
        config = self._configs.get(config_name, {})
        
        if key is None:
            return config
        
        # Support dot-separated key paths
        keys = key.split(".")
        value = config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    @property
    def data(self) -> Dict[str, Any]:
        """Get data configuration."""
        return self._configs.get("data", {})
    
    @property
    def model(self) -> Dict[str, Any]:
        """Get model configuration."""
        return self._configs.get("model", {})
    
    @property
    def training(self) -> Dict[str, Any]:
        """Get training configuration."""
        return self._configs.get("training", {})
    
    @property
    def checkpoint(self) -> Dict[str, Any]:
        """Get checkpoint configuration."""
        return self._configs.get("checkpoint", {})
    
    @property
    def inference(self) -> Dict[str, Any]:
        """Get inference configuration."""
        return self._configs.get("inference", {})
    
    @property
    def logging_config(self) -> Dict[str, Any]:
        """Get logging configuration."""
        return self._configs.get("logging", {})
    
    @property
    def warning(self) -> Dict[str, Any]:
        """Get warning configuration."""
        return self._configs.get("warning", {})


# =============================================================================
# Global Configuration Access
# =============================================================================

def get_config_manager() -> ConfigManager:
    """
    Get the global configuration manager instance.
    
    Returns:
        ConfigManager singleton instance.
    """
    return ConfigManager()


def get_config(config_name: str, key: str = None, default: Any = None) -> Any:
    """
    Convenience function to get configuration value.
    
    Args:
        config_name: Name of configuration.
        key: Optional dot-separated key path.
        default: Default value if key not found.
        
    Returns:
        Configuration value or default.
    """
    return get_config_manager().get(config_name, key, default)


# =============================================================================
# Module Exports
# =============================================================================

__all__ = [
    "get_project_root",
    "get_config_dir",
    "load_yaml",
    "merge_configs",
    "ConfigPaths",
    "ConfigManager",
    "get_config_manager",
    "get_config",
]
