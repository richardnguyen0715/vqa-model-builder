"""
Pipeline Logger Module
Provides comprehensive logging for the VQA pipeline with detailed validation output.
"""

import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass
import json

import torch
import torch.nn as nn


@dataclass
class LogSection:
    """Represents a logging section with formatted output."""
    title: str
    level: str = "INFO"
    width: int = 80
    char: str = "="


class PipelineLogger:
    """
    Comprehensive logger for VQA pipeline with detailed validation output.
    
    Provides structured logging with:
    - Section headers for different pipeline stages
    - Data validation output
    - Model architecture visualization
    - Progress tracking
    - Summary statistics
    """
    
    def __init__(
        self,
        name: str = "vqa_pipeline",
        log_dir: Optional[str] = None,
        console_level: int = logging.INFO,
        file_level: int = logging.DEBUG
    ):
        """
        Initialize pipeline logger.
        
        Args:
            name: Logger name
            log_dir: Directory for log files
            console_level: Console logging level
            file_level: File logging level
        """
        self.name = name
        self.log_dir = Path(log_dir) if log_dir else Path("logs/pipeline")
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Create timestamp for this run
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Setup logger
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.DEBUG)
        self.logger.propagate = False  # Prevent duplicate logging
        
        # Clear existing handlers
        if self.logger.handlers:
            self.logger.handlers.clear()
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(console_level)
        console_format = logging.Formatter(
            '%(asctime)s | %(levelname)-8s | %(message)s',
            datefmt='%H:%M:%S'
        )
        console_handler.setFormatter(console_format)
        self.logger.addHandler(console_handler)
        
        # File handler
        log_file = self.log_dir / f"pipeline_{self.timestamp}.log"
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(file_level)
        file_format = logging.Formatter(
            '%(asctime)s | %(levelname)-8s | %(funcName)-20s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(file_format)
        self.logger.addHandler(file_handler)
        
        # Statistics tracking
        self.stats: Dict[str, Any] = {}
        self.start_time = datetime.now()
        
    def section(self, title: str, char: str = "=", width: int = 80):
        """Print a section header."""
        border = char * width
        self.logger.info("")
        self.logger.info(border)
        self.logger.info(f" {title.upper()} ".center(width, char))
        self.logger.info(border)
        
    def subsection(self, title: str, char: str = "-", width: int = 60):
        """Print a subsection header."""
        self.logger.info("")
        self.logger.info(f" {title} ".center(width, char))
        
    def info(self, message: str):
        """Log info message."""
        self.logger.info(message)
        
    def debug(self, message: str):
        """Log debug message."""
        self.logger.debug(message)
        
    def warning(self, message: str):
        """Log warning message."""
        self.logger.warning(message)
        
    def error(self, message: str):
        """Log error message."""
        self.logger.error(message)
        
    def success(self, message: str):
        """Log success message with checkmark."""
        self.logger.info(f"✓ {message}")
        
    def failure(self, message: str):
        """Log failure message with X mark."""
        self.logger.error(f"✗ {message}")
        
    def bullet(self, message: str, indent: int = 2):
        """Log bullet point message."""
        spaces = " " * indent
        self.logger.info(f"{spaces}• {message}")
        
    def key_value(self, key: str, value: Any, indent: int = 4):
        """Log key-value pair."""
        spaces = " " * indent
        self.logger.info(f"{spaces}{key}: {value}")
        
    def table(self, headers: List[str], rows: List[List[Any]], col_widths: Optional[List[int]] = None):
        """Print a formatted table."""
        if col_widths is None:
            col_widths = [max(len(str(h)), max(len(str(r[i])) for r in rows)) + 2 
                         for i, h in enumerate(headers)]
        
        # Header
        header_row = "|".join(str(h).center(w) for h, w in zip(headers, col_widths))
        separator = "+".join("-" * w for w in col_widths)
        
        self.logger.info(f"+{separator}+")
        self.logger.info(f"|{header_row}|")
        self.logger.info(f"+{separator}+")
        
        # Rows
        for row in rows:
            row_str = "|".join(str(r).center(w) for r, w in zip(row, col_widths))
            self.logger.info(f"|{row_str}|")
        
        self.logger.info(f"+{separator}+")
        
    def log_dict(self, data: Dict[str, Any], title: Optional[str] = None, indent: int = 4):
        """Log a dictionary with formatted output."""
        if title:
            self.subsection(title)
        
        for key, value in data.items():
            if isinstance(value, dict):
                self.logger.info(f"{'  ' * (indent // 2)}{key}:")
                for k, v in value.items():
                    self.key_value(k, v, indent + 4)
            else:
                self.key_value(key, value, indent)
                
    def log_data_sample(self, sample: Any, idx: int = 0):
        """Log a data sample for validation."""
        self.subsection(f"Sample #{idx}")
        
        if hasattr(sample, 'image'):
            img = sample.image
            if hasattr(img, 'shape'):
                self.key_value("Image shape", img.shape)
            elif isinstance(img, torch.Tensor):
                self.key_value("Image tensor shape", list(img.shape))
                
        if hasattr(sample, 'question'):
            question = sample.question
            if len(question) > 100:
                question = question[:100] + "..."
            self.key_value("Question", f'"{question}"')
            
        if hasattr(sample, 'answers'):
            self.key_value("Answers", sample.answers)
            
        if hasattr(sample, 'metadata') and sample.metadata:
            self.key_value("Metadata", sample.metadata)
            
    def log_model_architecture(self, model: nn.Module, input_shapes: Optional[Dict[str, tuple]] = None):
        """Log model architecture details."""
        self.subsection("Model Architecture")
        
        # Model name
        self.key_value("Model class", model.__class__.__name__)
        
        # Parameter count
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        frozen_params = total_params - trainable_params
        
        self.key_value("Total parameters", f"{total_params:,}")
        self.key_value("Trainable parameters", f"{trainable_params:,}")
        self.key_value("Frozen parameters", f"{frozen_params:,}")
        self.key_value("Model size (MB)", f"{total_params * 4 / 1024 / 1024:.2f}")
        
        # Module breakdown
        self.subsection("Module Breakdown")
        module_params = {}
        for name, module in model.named_children():
            params = sum(p.numel() for p in module.parameters())
            trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
            module_params[name] = {
                'total': params,
                'trainable': trainable,
                'frozen': params - trainable
            }
            
        for name, counts in module_params.items():
            self.logger.info(f"    {name}:")
            self.logger.info(f"        Total: {counts['total']:,} | Trainable: {counts['trainable']:,} | Frozen: {counts['frozen']:,}")
            
        # Full model structure (abbreviated)
        self.subsection("Model Structure (Top-level)")
        model_str = str(model)
        lines = model_str.split('\n')
        if len(lines) > 50:
            for line in lines[:25]:
                self.logger.debug(line)
            self.logger.debug(f"    ... ({len(lines) - 50} lines omitted) ...")
            for line in lines[-25:]:
                self.logger.debug(line)
        else:
            for line in lines:
                self.logger.debug(line)
                
    def log_training_config(self, config: Any):
        """Log training configuration."""
        self.subsection("Training Configuration")
        
        if hasattr(config, '__dict__'):
            config_dict = config.__dict__
        elif isinstance(config, dict):
            config_dict = config
        else:
            config_dict = {'config': str(config)}
            
        for key, value in config_dict.items():
            if not key.startswith('_'):
                self.key_value(key, value)
                
    def log_batch_sample(self, batch: Dict[str, torch.Tensor], batch_idx: int = 0):
        """Log a batch sample for validation."""
        self.subsection(f"Batch #{batch_idx} Sample")
        
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                self.key_value(f"{key} shape", list(value.shape))
                self.key_value(f"{key} dtype", str(value.dtype))
                self.key_value(f"{key} device", str(value.device))
            else:
                self.key_value(key, type(value).__name__)
                
    def log_augmentation_result(
        self,
        original_shape: tuple,
        augmented_shape: tuple,
        augmentations_applied: List[str]
    ):
        """Log augmentation results."""
        self.subsection("Augmentation Result")
        self.key_value("Original shape", original_shape)
        self.key_value("Augmented shape", augmented_shape)
        self.key_value("Augmentations applied", augmentations_applied)
        
    def log_dataloader_info(
        self,
        name: str,
        num_samples: int,
        batch_size: int,
        num_batches: int,
        num_workers: int
    ):
        """Log DataLoader information."""
        self.subsection(f"{name} DataLoader")
        self.key_value("Number of samples", num_samples)
        self.key_value("Batch size", batch_size)
        self.key_value("Number of batches", num_batches)
        self.key_value("Number of workers", num_workers)
        
    def log_metrics(self, metrics: Dict[str, float], epoch: Optional[int] = None):
        """Log training/evaluation metrics."""
        prefix = f"Epoch {epoch}" if epoch is not None else "Metrics"
        self.subsection(prefix)
        
        for name, value in metrics.items():
            if isinstance(value, float):
                self.key_value(name, f"{value:.4f}")
            else:
                self.key_value(name, value)
                
    def log_checkpoint(self, path: str, metrics: Dict[str, float], is_best: bool = False):
        """Log checkpoint save information."""
        best_str = " [BEST]" if is_best else ""
        self.success(f"Checkpoint saved{best_str}: {path}")
        for name, value in metrics.items():
            self.key_value(name, f"{value:.4f}" if isinstance(value, float) else value, indent=6)
            
    def start_stage(self, stage_name: str):
        """Start a pipeline stage with timing."""
        self.stats[f"{stage_name}_start"] = datetime.now()
        self.section(stage_name)
        
    def end_stage(self, stage_name: str, success: bool = True):
        """End a pipeline stage with timing."""
        end_time = datetime.now()
        start_key = f"{stage_name}_start"
        
        if start_key in self.stats:
            duration = (end_time - self.stats[start_key]).total_seconds()
            self.stats[f"{stage_name}_duration"] = duration
            
            if success:
                self.success(f"{stage_name} completed in {duration:.2f}s")
            else:
                self.failure(f"{stage_name} failed after {duration:.2f}s")
        else:
            if success:
                self.success(f"{stage_name} completed")
            else:
                self.failure(f"{stage_name} failed")
                
    def summary(self):
        """Print final summary of pipeline execution."""
        self.section("PIPELINE SUMMARY")
        
        total_duration = (datetime.now() - self.start_time).total_seconds()
        
        self.key_value("Total execution time", f"{total_duration:.2f}s")
        self.key_value("Log file", str(self.log_dir / f"pipeline_{self.timestamp}.log"))
        
        # Stage durations
        self.subsection("Stage Durations")
        for key, value in self.stats.items():
            if key.endswith("_duration"):
                stage_name = key.replace("_duration", "")
                self.key_value(stage_name, f"{value:.2f}s")
                
    def save_stats(self, filepath: Optional[str] = None):
        """Save statistics to JSON file."""
        if filepath is None:
            filepath = self.log_dir / f"pipeline_stats_{self.timestamp}.json"
        
        stats_to_save = {}
        for key, value in self.stats.items():
            if isinstance(value, datetime):
                stats_to_save[key] = value.isoformat()
            else:
                stats_to_save[key] = value
                
        stats_to_save['total_duration'] = (datetime.now() - self.start_time).total_seconds()
        stats_to_save['timestamp'] = self.timestamp
        
        with open(filepath, 'w') as f:
            json.dump(stats_to_save, f, indent=2)
            
        self.info(f"Statistics saved to: {filepath}")


# Global pipeline logger instance
_pipeline_logger: Optional[PipelineLogger] = None


def get_pipeline_logger(
    name: str = "vqa_pipeline",
    log_dir: Optional[str] = None,
    reset: bool = False
) -> PipelineLogger:
    """
    Get or create the global pipeline logger.
    
    Args:
        name: Logger name
        log_dir: Directory for log files
        reset: Force create new logger
        
    Returns:
        PipelineLogger instance
    """
    global _pipeline_logger
    
    if _pipeline_logger is None or reset:
        _pipeline_logger = PipelineLogger(name=name, log_dir=log_dir)
        
    return _pipeline_logger
