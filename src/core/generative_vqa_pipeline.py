"""
Generative VQA Pipeline
CLI entry point for training and inference with generative VQA model.

Features:
- Multiple modes: train, evaluate, inference, demo
- Resource management integration (CPU, GPU, Memory monitoring)
- Automatic backup on threshold breach
- Progress tracking
- Comprehensive logging
- YAML configuration support
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
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import json

import torch
from transformers import AutoTokenizer

from src.core.pipeline_logger import PipelineLogger, get_pipeline_logger
from src.modeling.meta_arch.generative_vqa_model import (
    GenerativeVQAModel, 
    GenerativeVQAConfig,
    create_generative_vqa_model,
    get_default_generative_vqa_config
)
from src.data.generative_dataset import (
    GenerativeVQADataset,
    generative_vqa_collate_fn,
    create_generative_dataloader
)
from src.core.generative_training_pipeline import (
    GenerativeTrainingPipeline,
    GenerativeTrainingConfig,
    GenerativeTrainingOutput
)

# Resource Management (optional)
try:
    from src.resource_management import (
        ResourceManager,
        resource_managed_training,
        get_resource_manager,
        load_resource_config,
    )
    RESOURCE_MANAGEMENT_AVAILABLE = True
except ImportError:
    RESOURCE_MANAGEMENT_AVAILABLE = False


@dataclass
class GenerativeVQAPipelineConfig:
    """Configuration for generative VQA pipeline."""
    
    # Mode: train, evaluate, inference, demo
    mode: str = "train"
    
    # Data paths
    images_dir: str = "data/raw/images"
    text_file: str = "data/raw/texts/evaluate_60k_data_balanced_preprocessed.csv"
    
    # Data settings
    batch_size: int = 16
    max_question_length: int = 128
    max_answer_length: int = 64
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    test_ratio: float = 0.1
    num_workers: int = 4
    
    # Model settings
    visual_backbone: str = "openai/clip-vit-base-patch32"
    text_encoder: str = "vinai/phobert-base"
    hidden_size: int = 768
    num_decoder_layers: int = 6
    num_attention_heads: int = 8
    freeze_visual_encoder: bool = True
    freeze_question_encoder: bool = True
    
    # MOE settings (optional)
    use_moe: bool = False
    moe_type: str = "standard"  # 'standard', 'vqa', 'sparse'
    num_experts: int = 4
    expert_capacity_factor: float = 1.25
    moe_loss_weight: float = 0.01
    moe_position: str = "fusion"  # 'fusion', 'decoder', 'both'
    
    # VQA MOE specialized experts (when moe_type='vqa')
    num_vision_experts: int = 2
    num_text_experts: int = 2
    num_multimodal_experts: int = 2
    num_specialized_experts: int = 2
    vietnamese_optimized: bool = True
    
    # Knowledge/RAG settings (optional)
    use_knowledge: bool = False
    knowledge_base_path: Optional[str] = None
    retriever_top_k: int = 3
    
    # Training settings
    num_epochs: int = 20
    learning_rate: float = 5e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    use_amp: bool = True
    gradient_accumulation_steps: int = 2
    early_stopping: bool = True
    patience: int = 5
    
    # Generation settings
    max_generate_length: int = 64
    num_beams: int = 1
    do_sample: bool = False
    temperature: float = 1.0
    top_k: int = 50
    top_p: float = 0.95
    
    # Paths
    output_dir: str = "outputs/generative"
    checkpoint_dir: str = "checkpoints/generative"
    log_dir: str = "logs/generative_pipeline"
    
    # Resume
    resume_from: Optional[str] = None
    
    # Resource Management
    enable_resource_management: bool = True
    resource_config_path: Optional[str] = None
    
    # Seed
    seed: int = 42
    
    @classmethod
    def from_yaml(cls, yaml_path: str) -> "GenerativeVQAPipelineConfig":
        """Load configuration from YAML file."""
        import yaml
        
        with open(yaml_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        # Extract generative config section if exists
        if 'generative' in config_dict:
            config_dict = config_dict['generative']
        
        # Handle nested configs
        data_config = config_dict.get('data', {})
        model_config = config_dict.get('model', {})
        training_config = config_dict.get('training', {})
        generation_config = config_dict.get('generation', {})
        moe_config = config_dict.get('moe', {})
        knowledge_config = config_dict.get('knowledge', {})
        
        # Helper function to safely convert to float
        def to_float(value, default):
            if value is None:
                return default
            try:
                return float(value)
            except (ValueError, TypeError):
                return default
        
        # Helper function to safely convert to int
        def to_int(value, default):
            if value is None:
                return default
            try:
                return int(value)
            except (ValueError, TypeError):
                return default
        
        return cls(
            # Mode
            mode=config_dict.get('mode', 'train'),
            
            # Data settings
            images_dir=data_config.get('images_dir', 'data/raw/images'),
            text_file=data_config.get('text_file', 'data/raw/texts/evaluate_60k_data_balanced_preprocessed.csv'),
            batch_size=to_int(data_config.get('batch_size'), 16),
            max_question_length=to_int(data_config.get('max_question_length'), 128),
            max_answer_length=to_int(data_config.get('max_answer_length'), 64),
            train_ratio=to_float(data_config.get('train_ratio'), 0.8),
            val_ratio=to_float(data_config.get('val_ratio'), 0.1),
            test_ratio=to_float(data_config.get('test_ratio'), 0.1),
            num_workers=to_int(data_config.get('num_workers'), 4),
            
            # Model settings
            visual_backbone=model_config.get('visual_backbone', 'openai/clip-vit-base-patch32'),
            text_encoder=model_config.get('text_encoder', 'vinai/phobert-base'),
            hidden_size=to_int(model_config.get('hidden_size'), 768),
            num_decoder_layers=to_int(model_config.get('num_decoder_layers'), 6),
            num_attention_heads=to_int(model_config.get('num_attention_heads'), 8),
            freeze_visual_encoder=model_config.get('freeze_visual_encoder', True),
            freeze_question_encoder=model_config.get('freeze_question_encoder', True),
            
            # MOE settings
            use_moe=moe_config.get('enabled', False),
            moe_type=moe_config.get('type', 'standard'),
            moe_position=moe_config.get('position', 'fusion'),
            num_experts=to_int(moe_config.get('num_experts'), 4),
            expert_capacity_factor=to_float(moe_config.get('capacity_factor'), 1.25),
            moe_loss_weight=to_float(moe_config.get('loss_weight'), 0.01),
            # VQA MOE specialized experts
            num_vision_experts=to_int(moe_config.get('num_vision_experts'), 2),
            num_text_experts=to_int(moe_config.get('num_text_experts'), 2),
            num_multimodal_experts=to_int(moe_config.get('num_multimodal_experts'), 2),
            num_specialized_experts=to_int(moe_config.get('num_specialized_experts'), 2),
            vietnamese_optimized=moe_config.get('vietnamese_optimized', True),
            
            # Knowledge/RAG settings
            use_knowledge=knowledge_config.get('enabled', False),
            knowledge_base_path=knowledge_config.get('path'),
            retriever_top_k=to_int(knowledge_config.get('top_k'), 3),
            
            # Training settings
            num_epochs=to_int(training_config.get('num_epochs'), 20),
            learning_rate=to_float(training_config.get('learning_rate'), 5e-5),
            weight_decay=to_float(training_config.get('weight_decay'), 0.01),
            warmup_ratio=to_float(training_config.get('warmup_ratio'), 0.1),
            use_amp=training_config.get('use_amp', True),
            gradient_accumulation_steps=to_int(training_config.get('gradient_accumulation_steps'), 2),
            early_stopping=training_config.get('early_stopping', True),
            patience=to_int(training_config.get('patience'), 5),
            
            # Generation settings
            max_generate_length=to_int(generation_config.get('max_length'), 64),
            num_beams=to_int(generation_config.get('num_beams'), 1),
            do_sample=generation_config.get('do_sample', False),
            temperature=to_float(generation_config.get('temperature'), 1.0),
            top_k=to_int(generation_config.get('top_k'), 50),
            top_p=to_float(generation_config.get('top_p'), 0.95),
            
            # Paths
            output_dir=config_dict.get('output_dir', 'outputs/generative'),
            checkpoint_dir=config_dict.get('checkpoint_dir', 'checkpoints/generative'),
            log_dir=config_dict.get('log_dir', 'logs/generative_pipeline'),
            
            # Resume
            resume_from=config_dict.get('resume_from'),
            
            # Resource Management
            enable_resource_management=config_dict.get('enable_resource_management', True),
            resource_config_path=config_dict.get('resource_config_path'),
            
            # Seed
            seed=to_int(config_dict.get('seed'), 42),
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            'mode': self.mode,
            'data': {
                'images_dir': self.images_dir,
                'text_file': self.text_file,
                'batch_size': self.batch_size,
                'max_question_length': self.max_question_length,
                'max_answer_length': self.max_answer_length,
                'train_ratio': self.train_ratio,
                'val_ratio': self.val_ratio,
                'test_ratio': self.test_ratio,
                'num_workers': self.num_workers,
            },
            'model': {
                'visual_backbone': self.visual_backbone,
                'text_encoder': self.text_encoder,
                'hidden_size': self.hidden_size,
                'num_decoder_layers': self.num_decoder_layers,
                'num_attention_heads': self.num_attention_heads,
                'freeze_visual_encoder': self.freeze_visual_encoder,
                'freeze_question_encoder': self.freeze_question_encoder,
            },
            'moe': {
                'enabled': self.use_moe,
                'num_experts': self.num_experts,
                'capacity_factor': self.expert_capacity_factor,
                'loss_weight': self.moe_loss_weight,
            },
            'knowledge': {
                'enabled': self.use_knowledge,
                'path': self.knowledge_base_path,
                'top_k': self.retriever_top_k,
            },
            'training': {
                'num_epochs': self.num_epochs,
                'learning_rate': self.learning_rate,
                'weight_decay': self.weight_decay,
                'warmup_ratio': self.warmup_ratio,
                'use_amp': self.use_amp,
                'gradient_accumulation_steps': self.gradient_accumulation_steps,
                'early_stopping': self.early_stopping,
                'patience': self.patience,
            },
            'generation': {
                'max_length': self.max_generate_length,
                'num_beams': self.num_beams,
                'do_sample': self.do_sample,
                'temperature': self.temperature,
                'top_k': self.top_k,
                'top_p': self.top_p,
            },
            'output_dir': self.output_dir,
            'checkpoint_dir': self.checkpoint_dir,
            'log_dir': self.log_dir,
            'resume_from': self.resume_from,
            'enable_resource_management': self.enable_resource_management,
            'seed': self.seed,
        }


@dataclass
class GenerativeVQAPipelineOutput:
    """Output from generative VQA pipeline."""
    
    success: bool = True
    error: Optional[str] = None
    
    # Model info
    model_config: Optional[GenerativeVQAConfig] = None
    total_params: int = 0
    trainable_params: int = 0
    model_size_mb: float = 0.0
    
    # Data info
    train_samples: int = 0
    val_samples: int = 0
    test_samples: int = 0
    vocabulary_size: int = 0
    
    # Training results
    training_output: Optional[GenerativeTrainingOutput] = None
    final_metrics: Dict[str, float] = field(default_factory=dict)
    
    # Inference results
    inference_results: Optional[List[Dict[str, Any]]] = None
    
    execution_time: float = 0.0
    timestamp: str = ""
    
    # Resource stats
    resource_summary: Optional[Dict[str, Any]] = None


class GenerativeVQAPipeline:
    """
    Main pipeline for Generative VQA.
    
    Handles complete workflow:
    1. Data loading and tokenization
    2. Model creation and initialization
    3. Training with language modeling loss
    4. Evaluation with NLG metrics
    5. Inference with answer generation
    6. Resource management and monitoring
    
    Features:
    - Multiple modes: train, evaluate, inference, demo
    - MOE (Mixture of Experts) support
    - Knowledge Base / RAG support
    - Resource management (CPU, GPU, Memory monitoring)
    - Automatic backup on threshold breach
    - Comprehensive logging
    """
    
    def __init__(
        self,
        config: Optional[GenerativeVQAPipelineConfig] = None,
        config_path: Optional[str] = None
    ):
        """
        Initialize pipeline.
        
        Args:
            config: Pipeline configuration object
            config_path: Path to YAML configuration file
        """
        # Load config
        if config is not None:
            self.config = config
        elif config_path is not None:
            self.config = GenerativeVQAPipelineConfig.from_yaml(config_path)
        else:
            self.config = GenerativeVQAPipelineConfig()
        
        # Setup directories
        Path(self.config.output_dir).mkdir(parents=True, exist_ok=True)
        Path(self.config.checkpoint_dir).mkdir(parents=True, exist_ok=True)
        Path(self.config.log_dir).mkdir(parents=True, exist_ok=True)
        
        # Initialize logger
        self.logger = get_pipeline_logger(
            name="generative_vqa",
            log_dir=self.config.log_dir,
            reset=True
        )
        
        # Components
        self.model: Optional[GenerativeVQAModel] = None
        self.tokenizer = None
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None
        
        # Resource Manager
        self.resource_manager: Optional[ResourceManager] = None
        self._setup_resource_management()
        
        # Device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Results tracking
        self.training_output: Optional[GenerativeTrainingOutput] = None
        self.inference_results: List[Dict[str, Any]] = []
    
    def _setup_resource_management(self):
        """Setup resource management if available and enabled."""
        if not RESOURCE_MANAGEMENT_AVAILABLE:
            self.logger.warning("Resource management not available. Install dependencies.")
            return
            
        if not self.config.enable_resource_management:
            self.logger.info("Resource management disabled by config")
            return
            
        try:
            if self.config.resource_config_path:
                resource_config = load_resource_config(self.config.resource_config_path)
            else:
                resource_config = load_resource_config()
            
            self.resource_manager = ResourceManager(resource_config)
            self.logger.info("Resource management initialized")
        except Exception as e:
            self.logger.warning(f"Failed to initialize resource management: {e}")
    
    def run(self) -> GenerativeVQAPipelineOutput:
        """Execute pipeline based on mode."""
        start_time = datetime.now()
        
        self._print_banner()
        
        self.logger.section("GENERATIVE VQA PIPELINE STARTED")
        self.logger.key_value("Mode", self.config.mode)
        self.logger.key_value("Output directory", self.config.output_dir)
        self.logger.key_value("Start time", start_time.strftime("%Y-%m-%d %H:%M:%S"))
        
        # Start resource management if available
        resource_summary = None
        if self.resource_manager:
            self.resource_manager.start()
            self.logger.success("Resource management started")
        
        try:
            # Log system info
            self._log_system_info()
            
            # Log configuration
            self._log_configuration()
            
            # Initialize tokenizer
            self._setup_tokenizer()
            
            training_output = None
            
            if self.config.mode == "train":
                # Full training pipeline
                self._setup_data()
                self._setup_model()
                
                # Register model with resource manager for backup
                if self.resource_manager and self.model:
                    self.resource_manager.register_model(self.model)
                    self.logger.info("Model registered for resource monitoring and backup")
                
                training_output = self._run_training()
                
            elif self.config.mode == "evaluate":
                # Evaluation only
                self._setup_data()
                self._setup_model()
                self._load_checkpoint()
                training_output = self._run_evaluation()
                
            elif self.config.mode == "inference":
                # Inference only (batch prediction)
                self._setup_model()
                self._load_checkpoint()
                self._run_inference()
                
            elif self.config.mode == "demo":
                # Interactive demo
                self._setup_model()
                self._load_checkpoint()
                training_output = self._run_demo()
                
            else:
                raise ValueError(f"Unknown mode: {self.config.mode}")
            
            execution_time = (datetime.now() - start_time).total_seconds()
            
            # Get resource summary
            if self.resource_manager:
                resource_summary = self.resource_manager.get_status_summary()
            
            # Log final summary
            self._log_final_summary(execution_time, training_output)
            
            # Save summary to file
            self._save_summary(execution_time, training_output, resource_summary)
            
            # Calculate model stats
            total_params = sum(p.numel() for p in self.model.parameters()) if self.model else 0
            trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad) if self.model else 0
            model_size_mb = total_params * 4 / (1024 ** 2) if total_params > 0 else 0
            
            return GenerativeVQAPipelineOutput(
                success=True,
                model_config=self.model.config if self.model else None,
                total_params=total_params,
                trainable_params=trainable_params,
                model_size_mb=model_size_mb,
                train_samples=len(self.train_loader.dataset) if self.train_loader else 0,
                val_samples=len(self.val_loader.dataset) if self.val_loader else 0,
                test_samples=len(self.test_loader.dataset) if self.test_loader else 0,
                vocabulary_size=len(self.tokenizer) if self.tokenizer else 0,
                training_output=training_output if isinstance(training_output, GenerativeTrainingOutput) else None,
                final_metrics=training_output.final_metrics if isinstance(training_output, GenerativeTrainingOutput) else {},
                inference_results=self.inference_results if self.inference_results else None,
                execution_time=execution_time,
                timestamp=start_time.isoformat(),
                resource_summary=resource_summary
            )
            
        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds()
            self.logger.error(f"Pipeline failed: {e}")
            self.logger.section("PIPELINE FAILED")
            
            import traceback
            self.logger.error(traceback.format_exc())
            
            return GenerativeVQAPipelineOutput(
                success=False,
                error=str(e),
                execution_time=execution_time,
                timestamp=start_time.isoformat()
            )
        
        finally:
            # Stop resource management
            if self.resource_manager:
                self.resource_manager.stop()
                self.logger.info("Resource management stopped")
    
    def _print_banner(self):
        """Print pipeline banner."""
        banner = """
╔══════════════════════════════════════════════════════════════════════════════════════════╗
║                                                                                          ║
║   ██████╗ ███████╗███╗   ██╗███████╗██████╗  █████╗ ████████╗██╗██╗   ██╗███████╗        ║
║  ██╔════╝ ██╔════╝████╗  ██║██╔════╝██╔══██╗██╔══██╗╚══██╔══╝██║██║   ██║██╔════╝        ║
║  ██║  ███╗█████╗  ██╔██╗ ██║█████╗  ██████╔╝███████║   ██║   ██║██║   ██║█████╗          ║
║  ██║   ██║██╔══╝  ██║╚██╗██║██╔══╝  ██╔══██╗██╔══██║   ██║   ██║╚██╗ ██╔╝██╔══╝          ║
║  ╚██████╔╝███████╗██║ ╚████║███████╗██║  ██║██║  ██║   ██║   ██║ ╚████╔╝ ███████╗        ║
║   ╚═════╝ ╚══════╝╚═╝  ╚═══╝╚══════╝╚═╝  ╚═╝╚═╝  ╚═╝   ╚═╝   ╚═╝  ╚═══╝  ╚══════╝        ║
║                                                                                          ║
║           ██╗   ██╗ ██████╗  █████╗     ███╗   ███╗ ██████╗ ██████╗ ███████╗██╗          ║
║           ██║   ██║██╔═══██╗██╔══██╗    ████╗ ████║██╔═══██╗██╔══██╗██╔════╝██║          ║
║           ██║   ██║██║   ██║███████║    ██╔████╔██║██║   ██║██║  ██║█████╗  ██║          ║
║           ╚██╗ ██╔╝██║▄▄ ██║██╔══██║    ██║╚██╔╝██║██║   ██║██║  ██║██╔══╝  ██║          ║
║            ╚████╔╝ ╚██████╔╝██║  ██║    ██║ ╚═╝ ██║╚██████╔╝██████╔╝███████╗███████╗     ║
║             ╚═══╝   ╚══▀▀═╝ ╚═╝  ╚═╝    ╚═╝     ╚═╝ ╚═════╝ ╚═════╝ ╚══════╝╚══════╝     ║
║                                                                                          ║
║                           Vietnamese Generative VQA Model                                ║
║                       Encoder-Decoder with Answer Generation                             ║
║                                                                                          ║
╚══════════════════════════════════════════════════════════════════════════════════════════╝
        """
        print(banner)
    
    def _log_system_info(self):
        """Log system information."""
        self.logger.subsection("System Information")
        
        import platform
        import psutil
        
        self.logger.key_value("Platform", platform.platform())
        self.logger.key_value("Python version", platform.python_version())
        self.logger.key_value("PyTorch version", torch.__version__)
        
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
        total_ram = psutil.virtual_memory().total / 1024**3
        available_ram = psutil.virtual_memory().available / 1024**3
        self.logger.key_value("Total RAM", f"{total_ram:.1f} GB")
        self.logger.key_value("Available RAM", f"{available_ram:.1f} GB")
        
        # CPU info
        self.logger.key_value("CPU cores", psutil.cpu_count(logical=False))
        self.logger.key_value("CPU threads", psutil.cpu_count(logical=True))
        
        self.logger.key_value("Device", str(self.device))
        
        # Resource management status
        if self.resource_manager:
            self.logger.key_value("Resource management", "Enabled")
        else:
            self.logger.key_value("Resource management", "Disabled")
    
    def _log_configuration(self):
        """Log complete configuration."""
        self.logger.subsection("Pipeline Configuration")
        
        # Data config
        self.logger.info("📂 Data Configuration:")
        self.logger.key_value("  Images directory", self.config.images_dir)
        self.logger.key_value("  Text file", self.config.text_file)
        self.logger.key_value("  Batch size", self.config.batch_size)
        self.logger.key_value("  Train/Val/Test split", 
                            f"{self.config.train_ratio}/{self.config.val_ratio}/{self.config.test_ratio}")
        self.logger.key_value("  Max question length", self.config.max_question_length)
        self.logger.key_value("  Max answer length", self.config.max_answer_length)
        
        # Model config
        self.logger.info("🧠 Model Configuration:")
        self.logger.key_value("  Visual backbone", self.config.visual_backbone)
        self.logger.key_value("  Text encoder", self.config.text_encoder)
        self.logger.key_value("  Hidden size", self.config.hidden_size)
        self.logger.key_value("  Decoder layers", self.config.num_decoder_layers)
        self.logger.key_value("  Attention heads", self.config.num_attention_heads)
        self.logger.key_value("  Freeze visual", self.config.freeze_visual_encoder)
        self.logger.key_value("  Freeze text", self.config.freeze_question_encoder)
        
        # MOE config
        if self.config.use_moe:
            self.logger.info("🎯 MOE Configuration:")
            self.logger.key_value("  Use MOE", True)
            self.logger.key_value("  MOE Type", self.config.moe_type.upper())
            self.logger.key_value("  MOE Position", self.config.moe_position)
            if self.config.moe_type == 'vqa':
                total_experts = (self.config.num_vision_experts + 
                               self.config.num_text_experts + 
                               self.config.num_multimodal_experts + 
                               self.config.num_specialized_experts)
                self.logger.key_value("  Total Experts", total_experts)
                self.logger.key_value("  Vision Experts", self.config.num_vision_experts)
                self.logger.key_value("  Text Experts", self.config.num_text_experts)
                self.logger.key_value("  Multimodal Experts", self.config.num_multimodal_experts)
                self.logger.key_value("  Specialized Experts", self.config.num_specialized_experts)
                self.logger.key_value("  Vietnamese Optimized", self.config.vietnamese_optimized)
            else:
                self.logger.key_value("  Num experts", self.config.num_experts)
            self.logger.key_value("  Capacity factor", self.config.expert_capacity_factor)
            self.logger.key_value("  Loss weight", self.config.moe_loss_weight)
        
        # Knowledge/RAG config
        if self.config.use_knowledge:
            self.logger.info("📚 Knowledge/RAG Configuration:")
            self.logger.key_value("  Use Knowledge", True)
            self.logger.key_value("  Knowledge base path", self.config.knowledge_base_path)
            self.logger.key_value("  Retriever top-k", self.config.retriever_top_k)
        
        # Training config
        self.logger.info("🏋️ Training Configuration:")
        self.logger.key_value("  Epochs", self.config.num_epochs)
        self.logger.key_value("  Learning rate", self.config.learning_rate)
        self.logger.key_value("  Weight decay", self.config.weight_decay)
        self.logger.key_value("  Warmup ratio", self.config.warmup_ratio)
        self.logger.key_value("  Mixed precision", self.config.use_amp)
        self.logger.key_value("  Gradient accumulation", self.config.gradient_accumulation_steps)
        self.logger.key_value("  Early stopping", self.config.early_stopping)
        if self.config.early_stopping:
            self.logger.key_value("  Patience", self.config.patience)
        
        # Generation config
        self.logger.info("✨ Generation Configuration:")
        self.logger.key_value("  Max generate length", self.config.max_generate_length)
        self.logger.key_value("  Num beams", self.config.num_beams)
        self.logger.key_value("  Do sample", self.config.do_sample)
        self.logger.key_value("  Temperature", self.config.temperature)
        if self.config.do_sample:
            self.logger.key_value("  Top-k", self.config.top_k)
            self.logger.key_value("  Top-p", self.config.top_p)
    
    def _setup_tokenizer(self):
        """Setup tokenizer for answer generation."""
        self.logger.subsection("Setting up Tokenizer")
        
        self.logger.info(f"Loading tokenizer: {self.config.text_encoder}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.text_encoder)
        
        # Add special tokens if needed
        special_tokens = {}
        if self.tokenizer.bos_token is None:
            special_tokens['bos_token'] = '<s>'
        if self.tokenizer.eos_token is None:
            special_tokens['eos_token'] = '</s>'
        if self.tokenizer.pad_token is None:
            special_tokens['pad_token'] = '<pad>'
            
        if special_tokens:
            self.tokenizer.add_special_tokens(special_tokens)
            self.logger.info(f"Added special tokens: {special_tokens}")
        
        self.logger.key_value("Vocabulary size", len(self.tokenizer))
        self.logger.key_value("BOS token", self.tokenizer.bos_token)
        self.logger.key_value("EOS token", self.tokenizer.eos_token)
        self.logger.key_value("PAD token", self.tokenizer.pad_token)
    
    def _setup_data(self):
        """Setup dataloaders."""
        self.logger.subsection("Setting up Data")
        
        # Load data
        import pandas as pd
        self.logger.info(f"Loading data from: {self.config.text_file}")
        df = pd.read_csv(self.config.text_file)
        
        self.logger.key_value("Total samples", len(df))
        self.logger.key_value("Columns", list(df.columns))
        
        # Split data
        from sklearn.model_selection import train_test_split
        
        train_df, temp_df = train_test_split(
            df, 
            test_size=1 - self.config.train_ratio,
            random_state=self.config.seed
        )
        
        val_ratio = self.config.val_ratio / (self.config.val_ratio + self.config.test_ratio)
        val_df, test_df = train_test_split(
            temp_df,
            test_size=1 - val_ratio,
            random_state=self.config.seed
        )
        
        self.logger.key_value("Train samples", len(train_df))
        self.logger.key_value("Val samples", len(val_df))
        self.logger.key_value("Test samples", len(test_df))
        
        # Create datasets
        self.logger.info("Creating datasets...")
        
        train_dataset = GenerativeVQADataset(
            df=train_df,
            images_dir=self.config.images_dir,
            tokenizer=self.tokenizer,
            max_question_length=self.config.max_question_length,
            max_answer_length=self.config.max_answer_length
        )
        
        val_dataset = GenerativeVQADataset(
            df=val_df,
            images_dir=self.config.images_dir,
            tokenizer=self.tokenizer,
            max_question_length=self.config.max_question_length,
            max_answer_length=self.config.max_answer_length
        )
        
        test_dataset = GenerativeVQADataset(
            df=test_df,
            images_dir=self.config.images_dir,
            tokenizer=self.tokenizer,
            max_question_length=self.config.max_question_length,
            max_answer_length=self.config.max_answer_length
        )
        
        # Create dataloaders
        from torch.utils.data import DataLoader
        
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
            collate_fn=generative_vqa_collate_fn,
            pin_memory=True if torch.cuda.is_available() else False
        )
        
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            collate_fn=generative_vqa_collate_fn,
            pin_memory=True if torch.cuda.is_available() else False
        )
        
        self.test_loader = DataLoader(
            test_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            collate_fn=generative_vqa_collate_fn,
            pin_memory=True if torch.cuda.is_available() else False
        )
        
        self.logger.success(f"Created dataloaders (batch_size={self.config.batch_size})")
        
        # Display sample
        self._display_data_sample(train_dataset)
    
    def _display_data_sample(self, dataset):
        """Display a data sample."""
        self.logger.info("")
        self.logger.info("📋 Sample data item:")
        
        sample = dataset[0]
        self.logger.info(f"  • Question: {sample['question']}")
        self.logger.info(f"  • All answers: {sample['all_answers']}")
        self.logger.info(f"  • Image shape: {sample['image'].shape}")
        self.logger.info(f"  • Input IDs shape: {sample['input_ids'].shape}")
        self.logger.info(f"  • Decoder input IDs shape: {sample['decoder_input_ids'].shape}")
        self.logger.info(f"  • Labels shape: {sample['labels'].shape}")
        
        # Decode tokens
        decoded_input = self.tokenizer.decode(sample['input_ids'], skip_special_tokens=True)
        decoded_decoder = self.tokenizer.decode(sample['decoder_input_ids'], skip_special_tokens=True)
        
        self.logger.info(f"  • Decoded question: \"{decoded_input}\"")
        self.logger.info(f"  • Decoded decoder input: \"{decoded_decoder}\"")
    
    def _setup_model(self):
        """Setup generative VQA model."""
        self.logger.subsection("Setting up Model")
        
        # Create model config with MOE settings
        model_config = get_default_generative_vqa_config(
            visual_backbone=self.config.visual_backbone,
            text_encoder=self.config.text_encoder,
            vocab_size=len(self.tokenizer),
            bos_token_id=self.tokenizer.bos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
            # MOE configuration
            use_moe=self.config.use_moe,
            moe_type=self.config.moe_type,  # 'standard', 'vqa', 'sparse'
            num_experts=self.config.num_experts,
            num_experts_per_token=2,  # Default: top-2 routing
            expert_capacity_factor=self.config.expert_capacity_factor,
            moe_loss_weight=self.config.moe_loss_weight,
            moe_position=self.config.moe_position,
            # VQA MOE specialized experts
            num_vision_experts=self.config.num_vision_experts,
            num_text_experts=self.config.num_text_experts,
            num_multimodal_experts=self.config.num_multimodal_experts,
            num_specialized_experts=self.config.num_specialized_experts,
            vietnamese_optimized=self.config.vietnamese_optimized
        )
        
        # Override with pipeline config
        model_config.hidden_size = self.config.hidden_size
        model_config.num_decoder_layers = self.config.num_decoder_layers
        model_config.num_attention_heads = self.config.num_attention_heads
        model_config.freeze_visual_encoder = self.config.freeze_visual_encoder
        model_config.freeze_question_encoder = self.config.freeze_question_encoder
        model_config.max_answer_length = self.config.max_answer_length
        
        self.logger.info("Model configuration:")
        self.logger.key_value("Visual backbone", model_config.visual_backbone)
        self.logger.key_value("Text encoder", model_config.text_encoder)
        self.logger.key_value("Hidden size", model_config.hidden_size)
        self.logger.key_value("Decoder layers", model_config.num_decoder_layers)
        self.logger.key_value("Attention heads", model_config.num_attention_heads)
        self.logger.key_value("Vocabulary size", model_config.vocab_size)
        self.logger.key_value("Freeze visual", model_config.freeze_visual_encoder)
        self.logger.key_value("Freeze text", model_config.freeze_question_encoder)
        
        # Log MOE configuration if enabled
        if model_config.use_moe:
            self.logger.info("")
            self.logger.info("🎛️  MOE (Mixture of Experts) Configuration:")
            self.logger.key_value("  MOE Enabled", "✅ Yes")
            self.logger.key_value("  MOE Type", model_config.moe_type.upper())
            self.logger.key_value("  MOE Position", model_config.moe_position)
            
            if model_config.moe_type == 'vqa':
                # VQA MOE with specialized experts
                total_experts = (model_config.num_vision_experts + 
                               model_config.num_text_experts + 
                               model_config.num_multimodal_experts + 
                               model_config.num_specialized_experts)
                self.logger.key_value("  Total VQA Experts", total_experts)
                self.logger.key_value("  Vision Experts", model_config.num_vision_experts)
                self.logger.key_value("  Text Experts", model_config.num_text_experts)
                self.logger.key_value("  Multimodal Experts", model_config.num_multimodal_experts)
                self.logger.key_value("  Specialized Experts", model_config.num_specialized_experts)
                self.logger.key_value("  Vietnamese Optimized", model_config.vietnamese_optimized)
            else:
                self.logger.key_value("  Number of Experts", model_config.num_experts)
            
            self.logger.key_value("  Experts per Token", model_config.num_experts_per_token)
            self.logger.key_value("  Capacity Factor", model_config.expert_capacity_factor)
            self.logger.key_value("  Load Balance Loss Weight", model_config.moe_loss_weight)
        else:
            self.logger.key_value("MOE Enabled", "❌ No")
        
        # Create model
        self.logger.info("")
        self.logger.info("Creating model...")
        self.model = create_generative_vqa_model(model_config)
        
        # Resize embeddings if tokenizer was modified
        if hasattr(self.model, 'resize_token_embeddings'):
            self.model.resize_token_embeddings(len(self.tokenizer))
        
        # Move to device
        self.model.to(self.device)
        
        # Count parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        model_size_mb = total_params * 4 / (1024 ** 2)  # Assuming float32
        
        # Count MOE parameters if enabled
        moe_params = 0
        if model_config.use_moe and hasattr(self.model, 'fusion') and hasattr(self.model.fusion, 'moe_layer'):
            if self.model.fusion.moe_layer is not None:
                moe_params = sum(p.numel() for p in self.model.fusion.moe_layer.parameters())
        
        self.logger.success("Model created successfully!")
        self.logger.key_value("Total parameters", f"{total_params:,}")
        self.logger.key_value("Trainable parameters", f"{trainable_params:,}")
        if moe_params > 0:
            self.logger.key_value("MOE parameters", f"{moe_params:,}")
        self.logger.key_value("Model size", f"{model_size_mb:.2f} MB")
        
        # Log architecture summary
        self._log_model_architecture()
        
        # Log detailed MOE information if enabled
        if model_config.use_moe:
            self._log_moe_details()
    
    def _log_moe_details(self):
        """Log detailed information about MOE layer and experts."""
        if not hasattr(self.model, 'fusion') or not hasattr(self.model.fusion, 'moe_layer'):
            return
        
        moe_layer = self.model.fusion.moe_layer
        if moe_layer is None:
            return
        
        moe_type = getattr(self.model.fusion, 'moe_type', 'standard')
        
        self.logger.info("")
        self.logger.info("🔬 MOE Layer Details:")
        self.logger.info("  ╔════════════════════════════════════════════════════════════════╗")
        
        # Different header based on MOE type
        if moe_type == 'vqa':
            self.logger.info("  ║              VQA MIXTURE OF EXPERTS                            ║")
            self.logger.info("  ║          (Specialized Visual & Text Experts)                  ║")
        elif moe_type == 'sparse':
            self.logger.info("  ║              SPARSE MIXTURE OF EXPERTS                        ║")
        else:
            self.logger.info("  ║                    MIXTURE OF EXPERTS                         ║")
        
        self.logger.info("  ╠════════════════════════════════════════════════════════════════╣")
        
        # MOE Type info
        self.logger.info(f"  ║  🏷️  MOE Type: {moe_type.upper():<46} ║")
        
        # Router information
        if hasattr(moe_layer, 'router'):
            router = moe_layer.router
            router_type = type(router).__name__
            self.logger.info("  ╠════════════════════════════════════════════════════════════════╣")
            self.logger.info(f"  ║  📡 ROUTER                                                     ║")
            self.logger.info(f"  ║     Type: {router_type:<50} ║")
            if hasattr(router, 'top_k'):
                self.logger.info(f"  ║     Top-K: {str(router.top_k):<49} ║")
            if hasattr(router, 'num_experts'):
                self.logger.info(f"  ║     Num Experts: {str(router.num_experts):<43} ║")
            if hasattr(router, 'load_balance_weight'):
                self.logger.info(f"  ║     Load Balance Weight: {str(router.load_balance_weight):<35} ║")
            
            # Router parameters
            router_params = sum(p.numel() for p in router.parameters())
            self.logger.info(f"  ║     Parameters: {router_params:,}{'':_<{42-len(f'{router_params:,}')}} ║")
        
        self.logger.info("  ╠════════════════════════════════════════════════════════════════╣")
        
        # Expert information - categorized by type for VQA MOE
        if hasattr(moe_layer, 'experts'):
            experts = moe_layer.experts
            num_experts = len(experts)
            
            # Categorize experts by type
            expert_categories = {}
            for expert in experts:
                expert_type = type(expert).__name__
                if expert_type not in expert_categories:
                    expert_categories[expert_type] = []
                expert_categories[expert_type].append(expert)
            
            if moe_type == 'vqa':
                self.logger.info(f"  ║  🧠 EXPERTS ({num_experts} total - Specialized)                    ║")
            else:
                self.logger.info(f"  ║  🧠 EXPERTS ({num_experts} total)                                       ║")
            
            self.logger.info("  ╠════════════════════════════════════════════════════════════════╣")
            
            # Show expert categories for VQA MOE
            if len(expert_categories) > 1:
                self.logger.info("  ║  📋 Expert Categories:                                         ║")
                category_icons = {
                    'VisionExpert': '👁️ ',
                    'TextExpert': '📝',
                    'MultimodalExpert': '🔗',
                    'SegmentationExpert': '✂️ ',
                    'ObjectDetectionExpert': '🎯',
                    'OCRExpert': '🔤',
                    'SceneUnderstandingExpert': '🏞️ ',
                    'SpatialReasoningExpert': '📐',
                    'CountingExpert': '🔢',
                    'FeedForwardExpert': '➡️ ',
                    'GLUExpert': '⚡',
                }
                for cat_name, cat_experts in expert_categories.items():
                    icon = category_icons.get(cat_name, '•')
                    self.logger.info(f"  ║     {icon} {cat_name}: {len(cat_experts):<38} ║")
                self.logger.info("  ╠════════════════════════════════════════════════════════════════╣")
            
            total_expert_params = 0
            for i, expert in enumerate(experts):
                expert_type = type(expert).__name__
                expert_params = sum(p.numel() for p in expert.parameters())
                total_expert_params += expert_params
                
                # Get expert dimensions
                input_dim = getattr(expert, 'input_dim', '?')
                hidden_dim = getattr(expert, 'hidden_dim', '?')
                output_dim = getattr(expert, 'output_dim', '?')
                dim_str = f"{input_dim} → {hidden_dim} → {output_dim}"
                params_str = f"{expert_params:,}"
                
                # Get category icon
                icon = category_icons.get(expert_type, '•') if len(expert_categories) > 1 else ''
                
                self.logger.info(f"  ║  Expert {i+1}: {icon}                                                 ║")
                self.logger.info(f"  ║     Type: {expert_type:<50} ║")
                self.logger.info(f"  ║     Dimensions: {dim_str:<44} ║")
                self.logger.info(f"  ║     Parameters: {params_str:<44} ║")
                
                # Show specialized expert features
                if expert_type == 'OCRExpert' and hasattr(expert, 'vietnamese_optimized'):
                    vn_opt = "Yes" if expert.vietnamese_optimized else "No"
                    self.logger.info(f"  ║     Vietnamese Optimized: {vn_opt:<34} ║")
                
                if expert_type == 'SegmentationExpert' and hasattr(expert, 'mask_predictor'):
                    self.logger.info(f"  ║     Mask Prediction: Enabled{'':_<31} ║")
                
                if expert_type == 'ObjectDetectionExpert' and hasattr(expert, 'cross_attention'):
                    self.logger.info(f"  ║     Cross-Attention: Enabled (DETR-style){'':_<18} ║")
                
                if i < num_experts - 1:
                    self.logger.info("  ║  ─────────────────────────────────────────────────────────────  ║")
            
            self.logger.info("  ╠════════════════════════════════════════════════════════════════╣")
            self.logger.info(f"  ║  📊 SUMMARY                                                    ║")
            
            # Calculate parameter distribution
            moe_total_params = sum(p.numel() for p in moe_layer.parameters())
            router_params = sum(p.numel() for p in moe_layer.router.parameters()) if hasattr(moe_layer, 'router') else 0
            other_params = moe_total_params - total_expert_params - router_params
            
            self.logger.info(f"  ║     Total Expert Parameters: {f'{total_expert_params:,}':<31} ║")
            self.logger.info(f"  ║     Router Parameters: {f'{router_params:,}':<37} ║")
            self.logger.info(f"  ║     Other (LayerNorm, etc.): {f'{other_params:,}':<31} ║")
            self.logger.info(f"  ║     Total MOE Parameters: {f'{moe_total_params:,}':<34} ║")
        
        self.logger.info("  ╚════════════════════════════════════════════════════════════════╝")
        self.logger.info("")
    
    def _log_model_architecture(self):
        """Log model architecture with MOE if enabled."""
        use_moe = self.config.use_moe
        num_experts = self.config.num_experts
        
        self.logger.info("")
        self.logger.info("📦 Model Architecture:")
        self.logger.info("  ┌─────────────────────────────────────────┐")
        self.logger.info("  │              INPUT                      │")
        self.logger.info("  │   🖼️ Image + ❓ Question                 │")
        self.logger.info("  └───────────────┬─────────────────────────┘")
        self.logger.info("                  │")
        self.logger.info("  ┌───────────────▼─────────────────────────┐")
        self.logger.info("  │         VISUAL ENCODER                  │")
        self.logger.info("  │    CLIP ViT → Visual Features           │")
        self.logger.info("  │    [CLS] + patches → (197, 768)         │")
        self.logger.info("  └───────────────┬─────────────────────────┘")
        self.logger.info("                  │")
        self.logger.info("  ┌───────────────▼─────────────────────────┐")
        self.logger.info("  │       QUESTION ENCODER                  │")
        self.logger.info("  │    PhoBERT → Question Features          │")
        self.logger.info("  │    tokens → (seq_len, 768)              │")
        self.logger.info("  └───────────────┬─────────────────────────┘")
        self.logger.info("                  │")
        self.logger.info("  ┌───────────────▼─────────────────────────┐")
        self.logger.info("  │      CROSS-MODAL FUSION                 │")
        self.logger.info("  │    V + Q → Fused Features               │")
        self.logger.info("  │    Cross-Attention + FFN                │")
        self.logger.info("  └───────────────┬─────────────────────────┘")
        self.logger.info("                  │")
        
        # Show MOE layer if enabled
        if use_moe:
            self.logger.info("  ┌───────────────▼─────────────────────────┐")
            self.logger.info("  │      🎛️  MOE LAYER (Fusion)             │")
            self.logger.info(f"  │    {num_experts} Experts × Top-2 Routing          │")
            self.logger.info("  │    ┌─────┬─────┬─────┬─────┐           │")
            self.logger.info("  │    │ E1  │ E2  │ E3  │ E4  │  ...      │")
            self.logger.info("  │    └──┬──┴──┬──┴──┬──┴──┬──┘           │")
            self.logger.info("  │       └──── Weighted Sum ─────┘        │")
            self.logger.info("  └───────────────┬─────────────────────────┘")
            self.logger.info("                  │")
        
        self.logger.info("  ┌───────────────▼─────────────────────────┐")
        self.logger.info("  │      TRANSFORMER DECODER                │")
        self.logger.info("  │    Auto-regressive generation           │")
        self.logger.info(f"  │    {self.config.num_decoder_layers} layers × {self.config.num_attention_heads} heads                   │")
        self.logger.info("  └───────────────┬─────────────────────────┘")
        self.logger.info("                  │")
        self.logger.info("  ┌───────────────▼─────────────────────────┐")
        self.logger.info("  │            OUTPUT                       │")
        self.logger.info("  │    📝 Generated Answer (tokens)         │")
        self.logger.info("  └─────────────────────────────────────────┘")
        self.logger.info("")
    
    def _load_checkpoint(self):
        """Load model checkpoint."""
        if self.config.resume_from:
            self.logger.info(f"Loading checkpoint: {self.config.resume_from}")
            checkpoint = torch.load(
                self.config.resume_from, 
                map_location=self.device,
                weights_only=False
            )
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.logger.success("Checkpoint loaded successfully!")
    
    def _run_training(self) -> GenerativeTrainingOutput:
        """Run training pipeline."""
        self.logger.section("TRAINING")
        
        training_config = GenerativeTrainingConfig(
            num_epochs=self.config.num_epochs,
            learning_rate=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
            warmup_ratio=self.config.warmup_ratio,
            use_amp=self.config.use_amp,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            checkpoint_dir=self.config.checkpoint_dir,
            max_generate_length=self.config.max_generate_length,
            num_beams=self.config.num_beams,
            do_sample=self.config.do_sample,
            temperature=self.config.temperature,
            seed=self.config.seed
        )
        
        training_pipeline = GenerativeTrainingPipeline(
            model=self.model,
            train_loader=self.train_loader,
            val_loader=self.val_loader,
            tokenizer=self.tokenizer,
            config=training_config,
            logger=self.logger,
            device=self.device
        )
        
        return training_pipeline.run()
    
    def _run_evaluation(self) -> Dict:
        """Run evaluation only."""
        self.logger.section("EVALUATION")
        
        training_config = GenerativeTrainingConfig(
            max_generate_length=self.config.max_generate_length,
            num_beams=self.config.num_beams,
            num_samples_to_display=20
        )
        
        eval_pipeline = GenerativeTrainingPipeline(
            model=self.model,
            train_loader=self.train_loader,
            val_loader=self.val_loader,
            tokenizer=self.tokenizer,
            config=training_config,
            logger=self.logger,
            device=self.device
        )
        
        metrics = eval_pipeline._validate_epoch()
        
        self.logger.success("Evaluation completed!")
        return metrics
    
    def _run_demo(self) -> Dict:
        """Run interactive demo."""
        self.logger.section("DEMO MODE")
        
        from PIL import Image
        from transformers import CLIPProcessor
        
        clip_processor = CLIPProcessor.from_pretrained(self.config.visual_backbone)
        
        self.logger.info("Model loaded and ready for demo!")
        self.logger.info("Enter 'quit' to exit")
        self.logger.info("")
        
        self.model.eval()
        
        while True:
            # Get image path
            image_path = input("Enter image path: ").strip()
            if image_path.lower() == 'quit':
                break
                
            if not os.path.exists(image_path):
                print(f"Image not found: {image_path}")
                continue
            
            # Get question
            question = input("Enter question: ").strip()
            if question.lower() == 'quit':
                break
            
            # Process image
            image = Image.open(image_path).convert('RGB')
            image_inputs = clip_processor(images=image, return_tensors="pt")
            pixel_values = image_inputs['pixel_values'].to(self.device)
            
            # Process question
            question_inputs = self.tokenizer(
                question,
                return_tensors="pt",
                max_length=self.config.max_question_length,
                padding="max_length",
                truncation=True
            )
            input_ids = question_inputs['input_ids'].to(self.device)
            attention_mask = question_inputs['attention_mask'].to(self.device)
            
            # Generate answer
            with torch.no_grad():
                generated_ids = self.model.generate(
                    pixel_values=pixel_values,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_length=self.config.max_generate_length,
                    num_beams=self.config.num_beams,
                    do_sample=self.config.do_sample
                )
            
            # Decode answer
            answer = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
            
            print(f"\n🤖 Answer: {answer}\n")
        
        return {"status": "demo_completed"}
    
    def _run_inference(self) -> None:
        """Run batch inference on test data."""
        self.logger.section("INFERENCE MODE")
        
        if self.model is None:
            raise RuntimeError("Model must be loaded first")
        
        self.logger.info("Model ready for inference")
        
        # Check if we have test data
        if self.test_loader is None:
            self.logger.warning("No test data loaded. Running inference on sample images...")
            self._run_sample_inference()
            return
        
        from PIL import Image
        from transformers import CLIPProcessor
        
        clip_processor = CLIPProcessor.from_pretrained(self.config.visual_backbone)
        
        self.model.eval()
        self.inference_results = []
        
        self.logger.info(f"Running inference on {len(self.test_loader.dataset)} samples...")
        
        from tqdm import tqdm
        
        with torch.no_grad():
            for batch in tqdm(self.test_loader, desc="Inference"):
                # Move to device
                pixel_values = batch['image'].to(self.device)
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                
                # Generate answers
                generated_ids = self.model.generate(
                    pixel_values=pixel_values,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_length=self.config.max_generate_length,
                    num_beams=self.config.num_beams,
                    do_sample=self.config.do_sample,
                    temperature=self.config.temperature,
                    top_k=self.config.top_k if self.config.do_sample else None,
                    top_p=self.config.top_p if self.config.do_sample else None,
                )
                
                # Decode answers
                for i, gen_id in enumerate(generated_ids):
                    answer = self.tokenizer.decode(gen_id, skip_special_tokens=True)
                    result = {
                        'question': batch['question'][i] if 'question' in batch else '',
                        'predicted_answer': answer,
                        'ground_truth': batch['all_answers'][i] if 'all_answers' in batch else '',
                    }
                    self.inference_results.append(result)
        
        # Save inference results
        results_path = Path(self.config.output_dir) / "inference_results.json"
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(self.inference_results, f, ensure_ascii=False, indent=2)
        
        self.logger.success(f"Inference completed! Results saved to: {results_path}")
        
        # Display some samples
        self.logger.info("")
        self.logger.info("Sample predictions:")
        for i, result in enumerate(self.inference_results[:5]):
            self.logger.info(f"  [{i+1}] Q: {result['question'][:50]}...")
            self.logger.info(f"      Pred: {result['predicted_answer']}")
            self.logger.info(f"      GT: {result['ground_truth']}")
            self.logger.info("")
    
    def _run_sample_inference(self) -> None:
        """Run inference on sample images from images_dir."""
        from PIL import Image
        from transformers import CLIPProcessor
        
        clip_processor = CLIPProcessor.from_pretrained(self.config.visual_backbone)
        
        # Find sample images
        images_dir = Path(self.config.images_dir)
        if not images_dir.exists():
            self.logger.warning(f"Images directory not found: {images_dir}")
            return
        
        image_files = list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.png"))
        image_files = image_files[:5]  # Take first 5 images
        
        if not image_files:
            self.logger.warning("No images found in directory")
            return
        
        self.model.eval()
        self.logger.info(f"Running inference on {len(image_files)} sample images...")
        
        sample_questions = [
            "Đây là hình ảnh gì?",
            "Có bao nhiêu người trong hình?",
            "Màu sắc chủ đạo của hình ảnh là gì?",
        ]
        
        for img_path in image_files:
            image = Image.open(img_path).convert('RGB')
            image_inputs = clip_processor(images=image, return_tensors="pt")
            pixel_values = image_inputs['pixel_values'].to(self.device)
            
            self.logger.info(f"\n📷 Image: {img_path.name}")
            
            for question in sample_questions:
                question_inputs = self.tokenizer(
                    question,
                    return_tensors="pt",
                    max_length=self.config.max_question_length,
                    padding="max_length",
                    truncation=True
                )
                input_ids = question_inputs['input_ids'].to(self.device)
                attention_mask = question_inputs['attention_mask'].to(self.device)
                
                with torch.no_grad():
                    generated_ids = self.model.generate(
                        pixel_values=pixel_values,
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        max_length=self.config.max_generate_length,
                        num_beams=self.config.num_beams,
                    )
                
                answer = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
                self.logger.info(f"  Q: {question}")
                self.logger.info(f"  A: {answer}")
                
                self.inference_results.append({
                    'image': img_path.name,
                    'question': question,
                    'predicted_answer': answer,
                })
    
    def _log_final_summary(self, execution_time: float, result):
        """Log final summary."""
        self.logger.section("PIPELINE SUMMARY")
        
        self.logger.key_value("Status", "✅ SUCCESS")
        self.logger.key_value("Mode", self.config.mode.upper())
        self.logger.key_value("Total execution time", f"{execution_time:.2f}s ({execution_time/60:.1f} min)")
        
        # Data summary
        if self.train_loader or self.val_loader or self.test_loader:
            self.logger.info("")
            self.logger.info("📊 Data Summary:")
            if self.train_loader:
                self.logger.key_value("  Training samples", len(self.train_loader.dataset))
            if self.val_loader:
                self.logger.key_value("  Validation samples", len(self.val_loader.dataset))
            if self.test_loader:
                self.logger.key_value("  Test samples", len(self.test_loader.dataset))
        
        # Model summary
        if self.model:
            self.logger.info("")
            self.logger.info("🧠 Model Summary:")
            total_params = sum(p.numel() for p in self.model.parameters())
            trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            model_size_mb = total_params * 4 / (1024 ** 2)
            
            self.logger.key_value("  Total parameters", f"{total_params:,}")
            self.logger.key_value("  Trainable parameters", f"{trainable_params:,}")
            self.logger.key_value("  Model size", f"{model_size_mb:.2f} MB")
        
        # Training summary
        if isinstance(result, GenerativeTrainingOutput):
            self.logger.info("")
            self.logger.info("🏋️ Training Summary:")
            self.logger.key_value("  Total epochs", result.total_epochs)
            self.logger.key_value("  Total steps", result.total_steps)
            self.logger.key_value("  Best metric", f"{result.best_metric:.4f}")
            self.logger.key_value("  Best model path", result.best_model_path)
            
            self.logger.info("")
            self.logger.info("📈 Final Metrics:")
            for k, v in result.final_metrics.items():
                self.logger.key_value(f"  {k}", f"{v:.4f}" if isinstance(v, float) else v)
        
        # Inference summary
        if self.inference_results:
            self.logger.info("")
            self.logger.info("🔮 Inference Summary:")
            self.logger.key_value("  Predictions made", len(self.inference_results))
        
        # Resource summary
        if self.resource_manager:
            self.logger.info("")
            self.logger.info("💻 Resource Summary:")
            try:
                status = self.resource_manager.get_status_summary()
                if 'resources' in status:
                    res = status['resources']
                    if 'cpu' in res:
                        self.logger.key_value("  CPU usage", f"{res['cpu'].get('current', 0):.1f}%")
                    if 'memory' in res:
                        self.logger.key_value("  Memory usage", f"{res['memory'].get('current', 0):.1f}%")
                    if 'gpu' in res and res['gpu'].get('available'):
                        self.logger.key_value("  GPU memory", f"{res['gpu'].get('current', 0):.1f}%")
            except Exception as e:
                self.logger.warning(f"Could not get resource summary: {e}")
        
        self.logger.logger.handlers[0].flush()  # Flush to file
    
    def _save_summary(
        self, 
        execution_time: float, 
        result: Any, 
        resource_summary: Optional[Dict] = None
    ):
        """Save pipeline summary to JSON."""
        summary = {
            "timestamp": datetime.now().isoformat(),
            "execution_time": execution_time,
            "mode": self.config.mode,
            "status": "SUCCESS",
            "config": self.config.to_dict(),
        }
        
        # Data info
        if self.train_loader or self.val_loader or self.test_loader:
            summary["data"] = {
                "train_samples": len(self.train_loader.dataset) if self.train_loader else 0,
                "val_samples": len(self.val_loader.dataset) if self.val_loader else 0,
                "test_samples": len(self.test_loader.dataset) if self.test_loader else 0,
                "vocabulary_size": len(self.tokenizer) if self.tokenizer else 0,
            }
        
        # Model info
        if self.model:
            total_params = sum(p.numel() for p in self.model.parameters())
            trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            summary["model"] = {
                "total_params": total_params,
                "trainable_params": trainable_params,
                "model_size_mb": total_params * 4 / (1024 ** 2),
            }
        
        # Training info
        if isinstance(result, GenerativeTrainingOutput):
            summary["training"] = {
                "total_epochs": result.total_epochs,
                "total_steps": result.total_steps,
                "best_metric": result.best_metric,
                "best_model_path": result.best_model_path,
                "final_metrics": result.final_metrics,
            }
        
        # Inference info
        if self.inference_results:
            summary["inference"] = {
                "num_predictions": len(self.inference_results),
            }
        
        # Resource info
        if resource_summary:
            summary["resources"] = resource_summary
        
        summary_path = Path(self.config.output_dir) / "generative_pipeline_summary.json"
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"Summary saved to: {summary_path}")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Generative VQA Pipeline - Vietnamese Visual Question Answering",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train with default settings
  python -m src.core.generative_vqa_pipeline --mode train
  
  # Train with custom settings
  python -m src.core.generative_vqa_pipeline --mode train --epochs 30 --batch-size 32
  
  # Evaluate a checkpoint
  python -m src.core.generative_vqa_pipeline --mode evaluate --resume checkpoints/best_model.pt
  
  # Run inference
  python -m src.core.generative_vqa_pipeline --mode inference --resume checkpoints/best_model.pt
  
  # Interactive demo
  python -m src.core.generative_vqa_pipeline --mode demo --resume checkpoints/best_model.pt
  
  # Load from YAML config
  python -m src.core.generative_vqa_pipeline --config configs/generative_configs.yaml
  
  # Train with MOE
  python -m src.core.generative_vqa_pipeline --mode train --use-moe --num-experts 4
        """
    )
    
    parser.add_argument(
        "--mode",
        type=str,
        default="train",
        choices=["train", "evaluate", "inference", "demo"],
        help="Pipeline mode: train, evaluate, inference, or demo"
    )
    
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to YAML configuration file"
    )
    
    # Data arguments
    data_group = parser.add_argument_group('Data Configuration')
    data_group.add_argument("--images-dir", type=str, default="data/raw/images",
                           help="Path to images directory")
    data_group.add_argument("--text-file", type=str, 
                           default="data/raw/texts/evaluate_60k_data_balanced_preprocessed.csv",
                           help="Path to text/questions CSV file")
    data_group.add_argument("--batch-size", type=int, default=16,
                           help="Batch size for training/inference")
    data_group.add_argument("--max-question-length", type=int, default=128,
                           help="Maximum question length in tokens")
    data_group.add_argument("--max-answer-length", type=int, default=64,
                           help="Maximum answer length in tokens")
    data_group.add_argument("--train-ratio", type=float, default=0.8,
                           help="Training data ratio")
    data_group.add_argument("--val-ratio", type=float, default=0.1,
                           help="Validation data ratio")
    data_group.add_argument("--num-workers", type=int, default=4,
                           help="Number of data loader workers")
    
    # Model arguments
    model_group = parser.add_argument_group('Model Configuration')
    model_group.add_argument("--visual-backbone", type=str, default="openai/clip-vit-base-patch32",
                            help="Visual backbone model")
    model_group.add_argument("--text-encoder", type=str, default="vinai/phobert-base",
                            help="Text encoder model")
    model_group.add_argument("--hidden-size", type=int, default=768,
                            help="Hidden size for model")
    model_group.add_argument("--num-decoder-layers", type=int, default=6,
                            help="Number of decoder layers")
    model_group.add_argument("--num-attention-heads", type=int, default=8,
                            help="Number of attention heads")
    model_group.add_argument("--freeze-visual", action="store_true", default=True,
                            help="Freeze visual encoder")
    model_group.add_argument("--freeze-text", action="store_true", default=True,
                            help="Freeze text encoder")
    
    # MOE arguments
    moe_group = parser.add_argument_group('Mixture of Experts Configuration')
    moe_group.add_argument("--use-moe", action="store_true",
                          help="Use Mixture of Experts")
    moe_group.add_argument("--moe-type", type=str, default="standard",
                          choices=["standard", "vqa", "sparse"],
                          help="MOE type: standard (FeedForward), vqa (Specialized), sparse")
    moe_group.add_argument("--moe-position", type=str, default="fusion",
                          choices=["fusion", "decoder", "both"],
                          help="Where to apply MOE")
    moe_group.add_argument("--num-experts", type=int, default=4,
                          help="Number of experts (for standard MOE)")
    moe_group.add_argument("--expert-capacity-factor", type=float, default=1.25,
                          help="Expert capacity factor")
    moe_group.add_argument("--moe-loss-weight", type=float, default=0.01,
                          help="Weight for MOE auxiliary loss")
    
    # VQA MOE specialized experts (when --moe-type vqa)
    moe_group.add_argument("--num-vision-experts", type=int, default=2,
                          help="Number of vision experts (for VQA MOE)")
    moe_group.add_argument("--num-text-experts", type=int, default=2,
                          help="Number of text experts (for VQA MOE)")
    moe_group.add_argument("--num-multimodal-experts", type=int, default=2,
                          help="Number of multimodal experts (for VQA MOE)")
    moe_group.add_argument("--num-specialized-experts", type=int, default=2,
                          help="Number of specialized experts (Segmentation, Detection, OCR, Scene)")
    moe_group.add_argument("--vietnamese-optimized", action="store_true", default=True,
                          help="Enable Vietnamese optimization in OCR expert")
    
    # Knowledge/RAG arguments
    kb_group = parser.add_argument_group('Knowledge Base Configuration')
    kb_group.add_argument("--use-knowledge", action="store_true",
                         help="Use Knowledge Base / RAG")
    kb_group.add_argument("--knowledge-base-path", type=str, default=None,
                         help="Path to knowledge base")
    kb_group.add_argument("--retriever-top-k", type=int, default=3,
                         help="Number of documents to retrieve")
    
    # Training arguments
    train_group = parser.add_argument_group('Training Configuration')
    train_group.add_argument("--epochs", type=int, default=20,
                            help="Number of training epochs")
    train_group.add_argument("--learning-rate", type=float, default=5e-5,
                            help="Learning rate")
    train_group.add_argument("--weight-decay", type=float, default=0.01,
                            help="Weight decay")
    train_group.add_argument("--warmup-ratio", type=float, default=0.1,
                            help="Warmup ratio")
    train_group.add_argument("--use-amp", action="store_true", default=True,
                            help="Use automatic mixed precision")
    train_group.add_argument("--gradient-accumulation", type=int, default=2,
                            help="Gradient accumulation steps")
    train_group.add_argument("--early-stopping", action="store_true", default=True,
                            help="Enable early stopping")
    train_group.add_argument("--patience", type=int, default=5,
                            help="Early stopping patience")
    
    # Generation arguments
    gen_group = parser.add_argument_group('Generation Configuration')
    gen_group.add_argument("--max-generate-length", type=int, default=64,
                          help="Maximum generation length")
    gen_group.add_argument("--num-beams", type=int, default=1,
                          help="Number of beams for beam search")
    gen_group.add_argument("--do-sample", action="store_true",
                          help="Use sampling for generation")
    gen_group.add_argument("--temperature", type=float, default=1.0,
                          help="Temperature for sampling")
    gen_group.add_argument("--top-k", type=int, default=50,
                          help="Top-k for sampling")
    gen_group.add_argument("--top-p", type=float, default=0.95,
                          help="Top-p (nucleus) for sampling")
    
    # Path arguments
    path_group = parser.add_argument_group('Path Configuration')
    path_group.add_argument("--output-dir", type=str, default="outputs/generative",
                           help="Output directory")
    path_group.add_argument("--checkpoint-dir", type=str, default="checkpoints/generative",
                           help="Checkpoint directory")
    path_group.add_argument("--log-dir", type=str, default="logs/generative_pipeline",
                           help="Log directory")
    path_group.add_argument("--resume", type=str, default=None,
                           help="Path to checkpoint to resume from")
    
    # Resource management
    resource_group = parser.add_argument_group('Resource Management')
    resource_group.add_argument("--enable-resource-management", action="store_true", default=True,
                               help="Enable resource management")
    resource_group.add_argument("--disable-resource-management", action="store_true",
                               help="Disable resource management")
    resource_group.add_argument("--resource-config", type=str, default=None,
                               help="Path to resource configuration file")
    
    # Other
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed for reproducibility")
    
    args = parser.parse_args()
    
    # Build config
    if args.config:
        config = GenerativeVQAPipelineConfig.from_yaml(args.config)
        # Override with CLI args if provided
        if args.mode != "train":
            config.mode = args.mode
        if args.resume:
            config.resume_from = args.resume
    else:
        config = GenerativeVQAPipelineConfig(
            mode=args.mode,
            images_dir=args.images_dir,
            text_file=args.text_file,
            batch_size=args.batch_size,
            max_question_length=args.max_question_length,
            max_answer_length=args.max_answer_length,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            test_ratio=1 - args.train_ratio - args.val_ratio,
            num_workers=args.num_workers,
            visual_backbone=args.visual_backbone,
            text_encoder=args.text_encoder,
            hidden_size=args.hidden_size,
            num_decoder_layers=args.num_decoder_layers,
            num_attention_heads=args.num_attention_heads,
            freeze_visual_encoder=args.freeze_visual,
            freeze_question_encoder=args.freeze_text,
            use_moe=args.use_moe,
            moe_type=args.moe_type,
            moe_position=args.moe_position,
            num_experts=args.num_experts,
            expert_capacity_factor=args.expert_capacity_factor,
            moe_loss_weight=args.moe_loss_weight,
            num_vision_experts=args.num_vision_experts,
            num_text_experts=args.num_text_experts,
            num_multimodal_experts=args.num_multimodal_experts,
            num_specialized_experts=args.num_specialized_experts,
            vietnamese_optimized=args.vietnamese_optimized,
            use_knowledge=args.use_knowledge,
            knowledge_base_path=args.knowledge_base_path,
            retriever_top_k=args.retriever_top_k,
            num_epochs=args.epochs,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            warmup_ratio=args.warmup_ratio,
            use_amp=args.use_amp,
            gradient_accumulation_steps=args.gradient_accumulation,
            early_stopping=args.early_stopping,
            patience=args.patience,
            max_generate_length=args.max_generate_length,
            num_beams=args.num_beams,
            do_sample=args.do_sample,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            output_dir=args.output_dir,
            checkpoint_dir=args.checkpoint_dir,
            log_dir=args.log_dir,
            resume_from=args.resume,
            enable_resource_management=args.enable_resource_management and not args.disable_resource_management,
            resource_config_path=args.resource_config,
            seed=args.seed
        )
    
    # Run pipeline
    pipeline = GenerativeVQAPipeline(config=config)
    output = pipeline.run()
    
    sys.exit(0 if output.success else 1)


if __name__ == "__main__":
    main()
