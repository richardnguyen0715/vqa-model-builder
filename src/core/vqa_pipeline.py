"""
VQA Pipeline Orchestrator
Main entry point for running the complete VQA pipeline.
"""

import os
import sys
import warnings

# Suppress common warnings BEFORE importing torch
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=RuntimeWarning)
warnings.filterwarnings('ignore', message='.*pynvml.*')

import argparse
from pathlib import Path
from typing import Any, Dict, Optional
from dataclasses import dataclass, field
from datetime import datetime
import json

import torch

from src.core.pipeline_logger import PipelineLogger, get_pipeline_logger
from src.core.data_pipeline import DataPipeline, DataPipelineConfig, DataPipelineOutput
from src.core.model_pipeline import ModelPipeline, ModelPipelineConfig, ModelPipelineOutput
from src.core.training_pipeline import TrainingPipeline, TrainingPipelineConfig, TrainingPipelineOutput


@dataclass
class VQAPipelineConfig:
    """Configuration for the complete VQA pipeline."""
    
    # Pipeline mode
    mode: str = "train"  # train, evaluate, inference
    
    # Data configuration
    data: DataPipelineConfig = field(default_factory=DataPipelineConfig)
    
    # Model configuration
    model: ModelPipelineConfig = field(default_factory=ModelPipelineConfig)
    
    # Training configuration
    training: TrainingPipelineConfig = field(default_factory=TrainingPipelineConfig)
    
    # Paths
    output_dir: str = "outputs"
    log_dir: str = "logs/pipeline"
    
    # Resume training
    resume_from: Optional[str] = None
    
    @classmethod
    def from_yaml(cls, yaml_path: str) -> "VQAPipelineConfig":
        """Load configuration from YAML file."""
        import yaml
        
        with open(yaml_path, 'r') as f:
            config_dict = yaml.safe_load(f)
            
        # Parse nested configs
        data_config = DataPipelineConfig(**config_dict.get('data', {}))
        model_config = ModelPipelineConfig(**config_dict.get('model', {}))
        training_config = TrainingPipelineConfig(**config_dict.get('training', {}))
        
        return cls(
            mode=config_dict.get('mode', 'train'),
            data=data_config,
            model=model_config,
            training=training_config,
            output_dir=config_dict.get('output_dir', 'outputs'),
            log_dir=config_dict.get('log_dir', 'logs/pipeline'),
            resume_from=config_dict.get('resume_from')
        )


@dataclass
class VQAPipelineOutput:
    """Output from VQA pipeline."""
    
    data_output: Optional[DataPipelineOutput] = None
    model_output: Optional[ModelPipelineOutput] = None
    training_output: Optional[TrainingPipelineOutput] = None
    
    success: bool = True
    error: Optional[str] = None
    
    execution_time: float = 0.0
    timestamp: str = ""


class VQAPipeline:
    """
    Main VQA Pipeline Orchestrator.
    
    Coordinates the execution of:
    1. Data Pipeline - Load, validate, and prepare data
    2. Model Pipeline - Build and initialize model
    3. Training Pipeline - Train and evaluate model
    
    All steps are logged with comprehensive validation output.
    """
    
    def __init__(
        self,
        config: Optional[VQAPipelineConfig] = None,
        config_path: Optional[str] = None
    ):
        """
        Initialize VQA pipeline.
        
        Args:
            config: Pipeline configuration object
            config_path: Path to YAML configuration file
        """
        # Load config
        if config is not None:
            self.config = config
        elif config_path is not None:
            self.config = VQAPipelineConfig.from_yaml(config_path)
        else:
            self.config = VQAPipelineConfig()
            
        # Setup directories
        Path(self.config.output_dir).mkdir(parents=True, exist_ok=True)
        Path(self.config.log_dir).mkdir(parents=True, exist_ok=True)
        
        # Initialize logger
        self.logger = get_pipeline_logger(
            name="vqa_pipeline",
            log_dir=self.config.log_dir,
            reset=True
        )
        
        # Pipeline components
        self.data_pipeline: Optional[DataPipeline] = None
        self.model_pipeline: Optional[ModelPipeline] = None
        self.training_pipeline: Optional[TrainingPipeline] = None
        
        # Outputs
        self.data_output: Optional[DataPipelineOutput] = None
        self.model_output: Optional[ModelPipelineOutput] = None
        self.training_output: Optional[TrainingPipelineOutput] = None
        
    def run(self) -> VQAPipelineOutput:
        """
        Execute the complete VQA pipeline.
        
        Returns:
            VQAPipelineOutput with all results
        """
        start_time = datetime.now()
        
        self._print_banner()
        
        self.logger.section("VQA PIPELINE STARTED")
        self.logger.key_value("Mode", self.config.mode)
        self.logger.key_value("Output directory", self.config.output_dir)
        self.logger.key_value("Start time", start_time.strftime("%Y-%m-%d %H:%M:%S"))
        
        try:
            # Log system info
            self._log_system_info()
            
            # Log configuration
            self._log_configuration()
            
            if self.config.mode == "train":
                # Full training pipeline
                self._run_data_pipeline()
                self._run_model_pipeline()
                self._run_training_pipeline()
                
            elif self.config.mode == "evaluate":
                # Evaluation only
                self._run_data_pipeline()
                self._run_model_pipeline()
                self._run_evaluation()
                
            elif self.config.mode == "inference":
                # Inference only
                self._run_model_pipeline()
                self._run_inference()
                
            else:
                raise ValueError(f"Unknown mode: {self.config.mode}")
                
            # Final summary
            execution_time = (datetime.now() - start_time).total_seconds()
            self._log_final_summary(execution_time)
            
            return VQAPipelineOutput(
                data_output=self.data_output,
                model_output=self.model_output,
                training_output=self.training_output,
                success=True,
                execution_time=execution_time,
                timestamp=start_time.isoformat()
            )
            
        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds()
            self.logger.error(f"Pipeline failed: {e}")
            self.logger.section("PIPELINE FAILED")
            
            import traceback
            self.logger.error(traceback.format_exc())
            
            return VQAPipelineOutput(
                data_output=self.data_output,
                model_output=self.model_output,
                training_output=self.training_output,
                success=False,
                error=str(e),
                execution_time=execution_time,
                timestamp=start_time.isoformat()
            )
            
    def _print_banner(self):
        """Print pipeline banner."""
        banner = """
╔══════════════════════════════════════════════════════════════════════════════════════════╗
║                                                                                          ║
║  ██╗   ██╗ ██████╗  █████╗     ██████╗ ██╗██████╗ ███████╗██╗     ██╗███╗   ██╗███████╗  ║
║  ██║   ██║██╔═══██╗██╔══██╗    ██╔══██╗██║██╔══██╗██╔════╝██║     ██║████╗  ██║██╔════╝  ║
║  ██║   ██║██║   ██║███████║    ██████╔╝██║██████╔╝█████╗  ██║     ██║██╔██╗ ██║█████╗    ║
║  ╚██╗ ██╔╝██║▄▄ ██║██╔══██║    ██╔═══╝ ██║██╔═══╝ ██╔══╝  ██║     ██║██║╚██╗██║██╔══╝    ║
║   ╚████╔╝ ╚██████╔╝██║  ██║    ██║     ██║██║     ███████╗███████╗██║██║ ╚████║███████╗  ║
║    ╚═══╝   ╚══▀▀═╝ ╚═╝  ╚═╝    ╚═╝     ╚═╝╚═╝     ╚══════╝╚══════╝╚═╝╚═╝  ╚═══╝╚══════╝  ║
║                                                                                          ║
║                            Vietnamese Visual Question Answering                          ║
║                                  AutoViVQA Model Builder                                 ║
║                                                                                          ║
╚══════════════════════════════════════════════════════════════════════════════════════════╝
        """
        print(banner)
        
    def _log_system_info(self):
        """Log system information."""
        self.logger.subsection("System Information")
        
        import platform
        
        self.logger.key_value("Platform", platform.platform())
        self.logger.key_value("Python version", platform.python_version())
        self.logger.key_value("PyTorch version", torch.__version__)
        
        # CUDA info
        if torch.cuda.is_available():
            self.logger.key_value("CUDA available", True)
            self.logger.key_value("CUDA version", torch.version.cuda)
            self.logger.key_value("GPU count", torch.cuda.device_count())
            for i in range(torch.cuda.device_count()):
                gpu_name = torch.cuda.get_device_name(i)
                gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
                self.logger.key_value(f"GPU {i}", f"{gpu_name} ({gpu_memory:.1f} GB)")
        else:
            self.logger.key_value("CUDA available", False)
            
        # Memory info
        import psutil
        total_ram = psutil.virtual_memory().total / 1024**3
        available_ram = psutil.virtual_memory().available / 1024**3
        self.logger.key_value("Total RAM", f"{total_ram:.1f} GB")
        self.logger.key_value("Available RAM", f"{available_ram:.1f} GB")
        
        # CPU info
        self.logger.key_value("CPU cores", psutil.cpu_count(logical=False))
        self.logger.key_value("CPU threads", psutil.cpu_count(logical=True))
        
    def _log_configuration(self):
        """Log complete configuration."""
        self.logger.subsection("Pipeline Configuration")
        
        # Data config
        self.logger.info("Data Configuration:")
        self.logger.key_value("Images directory", self.config.data.images_dir)
        self.logger.key_value("Text file", self.config.data.text_file)
        self.logger.key_value("Batch size", self.config.data.batch_size)
        self.logger.key_value("Train/Val/Test split", 
                            f"{self.config.data.train_ratio}/{self.config.data.val_ratio}/{self.config.data.test_ratio}")
        
        # Model config
        self.logger.info("Model Configuration:")
        self.logger.key_value("Visual backbone", self.config.model.visual_backbone)
        self.logger.key_value("Text encoder", self.config.model.text_encoder_type)
        self.logger.key_value("Fusion type", self.config.model.fusion_type)
        self.logger.key_value("Use MOE", self.config.model.use_moe)
        self.logger.key_value("Use Knowledge/RAG", self.config.model.use_knowledge)
        
        # Training config
        self.logger.info("Training Configuration:")
        self.logger.key_value("Epochs", self.config.training.num_epochs)
        self.logger.key_value("Learning rate", self.config.training.learning_rate)
        self.logger.key_value("Optimizer", self.config.training.optimizer_name)
        self.logger.key_value("Scheduler", self.config.training.scheduler_name)
        self.logger.key_value("Mixed precision", self.config.training.use_amp)
        
    def _run_data_pipeline(self):
        """Run data pipeline."""
        self.logger.section("STAGE 1: DATA PIPELINE")
        
        self.data_pipeline = DataPipeline(
            config=self.config.data,
            logger=self.logger
        )
        
        self.data_output = self.data_pipeline.run()
        
        self.logger.success("Data pipeline completed successfully")
        
    def _run_model_pipeline(self):
        """Run model pipeline."""
        self.logger.section("STAGE 2: MODEL PIPELINE")
        
        # Get number of answer classes from data pipeline
        num_answers = None
        if self.data_output is not None:
            num_answers = self.data_output.num_classes
            
        self.model_pipeline = ModelPipeline(
            config=self.config.model,
            logger=self.logger
        )
        
        self.model_output = self.model_pipeline.run(num_answers=num_answers)
        
        self.logger.success("Model pipeline completed successfully")
        
    def _run_training_pipeline(self):
        """Run training pipeline."""
        self.logger.section("STAGE 3: TRAINING PIPELINE")
        
        if self.data_output is None or self.model_output is None:
            raise RuntimeError("Data and model pipelines must be run first")
            
        self.training_pipeline = TrainingPipeline(
            model=self.model_output.model,
            train_loader=self.data_output.train_loader,
            val_loader=self.data_output.val_loader,
            config=self.config.training,
            logger=self.logger,
            device=self.model_output.device,
            model_config=self.model_output.config,
            vocabulary=self.data_output.answer2id,
            id2answer=self.data_output.id2answer,
            num_samples_to_display=5
        )
        
        self.training_output = self.training_pipeline.run()
        
        self.logger.success("Training pipeline completed successfully")
        
    def _run_evaluation(self):
        """Run evaluation only."""
        self.logger.section("EVALUATION")
        
        if self.data_output is None or self.model_output is None:
            raise RuntimeError("Data and model pipelines must be run first")
            
        # Load checkpoint if specified
        if self.config.resume_from:
            self.model_pipeline.load_checkpoint(self.config.resume_from)
            
        # Run evaluation
        from src.core.training_pipeline import TrainingPipeline
        
        eval_pipeline = TrainingPipeline(
            model=self.model_output.model,
            train_loader=self.data_output.train_loader,
            val_loader=self.data_output.val_loader,
            config=self.config.training,
            logger=self.logger,
            device=self.model_output.device,
            id2answer=self.data_output.id2answer,
            num_samples_to_display=10  # Show more samples in evaluation mode
        )
        
        metrics = eval_pipeline._validate_epoch(show_samples=True)
        
        self.logger.success("Evaluation completed")
        self.logger.log_metrics(metrics)
        
    def _run_inference(self):
        """Run inference only."""
        self.logger.section("INFERENCE")
        
        if self.model_output is None:
            raise RuntimeError("Model pipeline must be run first")
            
        # Load checkpoint if specified
        if self.config.resume_from:
            self.model_pipeline.load_checkpoint(self.config.resume_from)
            
        self.logger.info("Model ready for inference")
        self.logger.info("Use model_output.model for predictions")
        
    def _log_final_summary(self, execution_time: float):
        """Log final pipeline summary."""
        self.logger.section("PIPELINE SUMMARY")
        
        self.logger.key_value("Status", "SUCCESS")
        self.logger.key_value("Total execution time", f"{execution_time:.2f}s ({execution_time/60:.1f} min)")
        
        if self.data_output:
            self.logger.info("Data Summary:")
            self.logger.key_value("Training samples", self.data_output.train_samples)
            self.logger.key_value("Validation samples", self.data_output.val_samples)
            self.logger.key_value("Test samples", self.data_output.test_samples)
            self.logger.key_value("Answer classes", self.data_output.num_classes)
            
        if self.model_output:
            self.logger.info("Model Summary:")
            self.logger.key_value("Total parameters", f"{self.model_output.total_params:,}")
            self.logger.key_value("Trainable parameters", f"{self.model_output.trainable_params:,}")
            self.logger.key_value("Model size", f"{self.model_output.model_size_mb:.2f} MB")
            
        if self.training_output:
            self.logger.info("Training Summary:")
            self.logger.key_value("Total epochs", self.training_output.total_epochs)
            self.logger.key_value("Total steps", self.training_output.total_steps)
            self.logger.key_value("Best metric", f"{self.training_output.best_metric:.4f}")
            self.logger.key_value("Best model path", self.training_output.best_model_path)
            
            self.logger.info("Final Metrics:")
            for name, value in self.training_output.final_metrics.items():
                self.logger.key_value(name, f"{value:.4f}" if isinstance(value, float) else value)
                
        # Save summary to file
        self._save_summary(execution_time)
        
        self.logger.logger.handlers[0].flush()  # Flush to file
        
    def _save_summary(self, execution_time: float):
        """Save pipeline summary to JSON."""
        summary = {
            "timestamp": datetime.now().isoformat(),
            "execution_time": execution_time,
            "mode": self.config.mode,
            "status": "SUCCESS"
        }
        
        if self.data_output:
            summary["data"] = {
                "train_samples": self.data_output.train_samples,
                "val_samples": self.data_output.val_samples,
                "test_samples": self.data_output.test_samples,
                "num_classes": self.data_output.num_classes
            }
            
        if self.model_output:
            summary["model"] = {
                "total_params": self.model_output.total_params,
                "trainable_params": self.model_output.trainable_params,
                "model_size_mb": self.model_output.model_size_mb
            }
            
        if self.training_output:
            summary["training"] = {
                "total_epochs": self.training_output.total_epochs,
                "total_steps": self.training_output.total_steps,
                "best_metric": self.training_output.best_metric,
                "best_model_path": self.training_output.best_model_path,
                "final_metrics": self.training_output.final_metrics
            }
            
        summary_path = Path(self.config.output_dir) / "pipeline_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
            
        self.logger.info(f"Summary saved to: {summary_path}")


def main():
    """Main entry point for CLI."""
    parser = argparse.ArgumentParser(
        description="Vietnamese VQA Pipeline - AutoViVQA Model Builder"
    )
    
    parser.add_argument(
        "--mode",
        type=str,
        default="train",
        choices=["train", "evaluate", "inference"],
        help="Pipeline mode: train, evaluate, or inference"
    )
    
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to YAML configuration file"
    )
    
    # Data arguments
    parser.add_argument("--images-dir", type=str, default="data/raw/images")
    parser.add_argument("--text-file", type=str, default="data/raw/texts/evaluate_60k_data_balanced_preprocessed.csv")
    parser.add_argument("--batch-size", type=int, default=32)
    
    # Model arguments
    parser.add_argument("--visual-backbone", type=str, default="vit")
    parser.add_argument("--text-encoder", type=str, default="phobert")
    parser.add_argument("--use-moe", action="store_true")
    parser.add_argument("--use-knowledge", action="store_true")
    
    # Training arguments
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--output-dir", type=str, default="outputs")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    
    # Resume
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    
    args = parser.parse_args()
    
    # Build config
    if args.config:
        config = VQAPipelineConfig.from_yaml(args.config)
    else:
        config = VQAPipelineConfig(
            mode=args.mode,
            data=DataPipelineConfig(
                images_dir=args.images_dir,
                text_file=args.text_file,
                batch_size=args.batch_size,
                seed=args.seed
            ),
            model=ModelPipelineConfig(
                visual_backbone=args.visual_backbone,
                text_encoder_type=args.text_encoder,
                use_moe=args.use_moe,
                use_knowledge=args.use_knowledge
            ),
            training=TrainingPipelineConfig(
                num_epochs=args.epochs,
                learning_rate=args.learning_rate,
                seed=args.seed
            ),
            output_dir=args.output_dir,
            resume_from=args.resume
        )
        
    # Run pipeline
    pipeline = VQAPipeline(config=config)
    output = pipeline.run()
    
    # Exit code
    sys.exit(0 if output.success else 1)


if __name__ == "__main__":
    main()
