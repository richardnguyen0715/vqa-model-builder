"""
Generative VQA Pipeline
CLI entry point for training and inference with generative VQA model.
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


@dataclass
class GenerativeVQAPipelineConfig:
    """Configuration for generative VQA pipeline."""
    
    # Mode
    mode: str = "train"  # train, evaluate, inference, demo
    
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
    
    # Training settings
    num_epochs: int = 20
    learning_rate: float = 5e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    use_amp: bool = True
    gradient_accumulation_steps: int = 2
    
    # Generation settings
    max_generate_length: int = 64
    num_beams: int = 1
    do_sample: bool = False
    temperature: float = 1.0
    
    # Paths
    output_dir: str = "outputs/generative"
    checkpoint_dir: str = "checkpoints/generative"
    log_dir: str = "logs/generative_pipeline"
    
    # Resume
    resume_from: Optional[str] = None
    
    # Seed
    seed: int = 42


@dataclass
class GenerativeVQAPipelineOutput:
    """Output from generative VQA pipeline."""
    
    success: bool = True
    error: Optional[str] = None
    
    # Model info
    model_config: Optional[GenerativeVQAConfig] = None
    total_params: int = 0
    trainable_params: int = 0
    
    # Data info
    train_samples: int = 0
    val_samples: int = 0
    test_samples: int = 0
    
    # Training results
    training_output: Optional[GenerativeTrainingOutput] = None
    
    execution_time: float = 0.0
    timestamp: str = ""


class GenerativeVQAPipeline:
    """
    Main pipeline for Generative VQA.
    
    Handles complete workflow:
    1. Data loading and tokenization
    2. Model creation and initialization
    3. Training with language modeling loss
    4. Evaluation with NLG metrics
    5. Inference with answer generation
    """
    
    def __init__(self, config: GenerativeVQAPipelineConfig):
        """Initialize pipeline."""
        self.config = config
        
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
        
        # Device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def run(self) -> GenerativeVQAPipelineOutput:
        """Execute pipeline based on mode."""
        start_time = datetime.now()
        
        self._print_banner()
        
        self.logger.section("GENERATIVE VQA PIPELINE STARTED")
        self.logger.key_value("Mode", self.config.mode)
        self.logger.key_value("Start time", start_time.strftime("%Y-%m-%d %H:%M:%S"))
        
        try:
            # Log system info
            self._log_system_info()
            
            # Initialize tokenizer
            self._setup_tokenizer()
            
            if self.config.mode == "train":
                # Full training pipeline
                self._setup_data()
                self._setup_model()
                training_output = self._run_training()
                
            elif self.config.mode == "evaluate":
                self._setup_data()
                self._setup_model()
                self._load_checkpoint()
                training_output = self._run_evaluation()
                
            elif self.config.mode == "demo":
                self._setup_model()
                self._load_checkpoint()
                training_output = self._run_demo()
                
            else:
                raise ValueError(f"Unknown mode: {self.config.mode}")
            
            execution_time = (datetime.now() - start_time).total_seconds()
            
            self._log_final_summary(execution_time, training_output)
            
            return GenerativeVQAPipelineOutput(
                success=True,
                model_config=self.model.config if self.model else None,
                total_params=sum(p.numel() for p in self.model.parameters()) if self.model else 0,
                trainable_params=sum(p.numel() for p in self.model.parameters() if p.requires_grad) if self.model else 0,
                train_samples=len(self.train_loader.dataset) if self.train_loader else 0,
                val_samples=len(self.val_loader.dataset) if self.val_loader else 0,
                test_samples=len(self.test_loader.dataset) if self.test_loader else 0,
                training_output=training_output if isinstance(training_output, GenerativeTrainingOutput) else None,
                execution_time=execution_time,
                timestamp=start_time.isoformat()
            )
            
        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds()
            self.logger.error(f"Pipeline failed: {e}")
            
            import traceback
            self.logger.error(traceback.format_exc())
            
            return GenerativeVQAPipelineOutput(
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
            for i in range(torch.cuda.device_count()):
                gpu_name = torch.cuda.get_device_name(i)
                gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
                self.logger.key_value(f"GPU {i}", f"{gpu_name} ({gpu_memory:.1f} GB)")
        else:
            self.logger.key_value("CUDA available", False)
            
        total_ram = psutil.virtual_memory().total / 1024**3
        self.logger.key_value("Total RAM", f"{total_ram:.1f} GB")
        self.logger.key_value("Device", str(self.device))
    
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
        
        # Create model config
        model_config = get_default_generative_vqa_config(
            visual_backbone=self.config.visual_backbone,
            text_encoder=self.config.text_encoder,
            vocab_size=len(self.tokenizer),
            bos_token_id=self.tokenizer.bos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id
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
        
        # Create model
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
        
        self.logger.success("Model created successfully!")
        self.logger.key_value("Total parameters", f"{total_params:,}")
        self.logger.key_value("Trainable parameters", f"{trainable_params:,}")
        self.logger.key_value("Model size", f"{model_size_mb:.2f} MB")
        
        # Log architecture summary
        self._log_model_architecture()
    
    def _log_model_architecture(self):
        """Log model architecture."""
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
        self.logger.info("  ┌───────────────▼─────────────────────────┐")
        self.logger.info("  │      TRANSFORMER DECODER                │")
        self.logger.info("  │    Auto-regressive generation           │")
        self.logger.info("  │    6 layers × 8 heads                   │")
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
    
    def _log_final_summary(self, execution_time: float, result):
        """Log final summary."""
        self.logger.section("PIPELINE SUMMARY")
        
        self.logger.key_value("Status", "SUCCESS")
        self.logger.key_value("Execution time", f"{execution_time:.2f}s ({execution_time/60:.1f} min)")
        
        if self.model:
            total_params = sum(p.numel() for p in self.model.parameters())
            trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            self.logger.key_value("Total parameters", f"{total_params:,}")
            self.logger.key_value("Trainable parameters", f"{trainable_params:,}")
        
        if self.train_loader:
            self.logger.key_value("Train samples", len(self.train_loader.dataset))
            self.logger.key_value("Val samples", len(self.val_loader.dataset))
        
        if isinstance(result, GenerativeTrainingOutput):
            self.logger.key_value("Best model", result.best_model_path)
            self.logger.key_value("Best metric", f"{result.best_metric:.4f}")
            
            self.logger.info("")
            self.logger.info("Final Metrics:")
            for k, v in result.final_metrics.items():
                self.logger.key_value(k, f"{v:.4f}" if isinstance(v, float) else v)


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Generative VQA Pipeline - Vietnamese Visual Question Answering"
    )
    
    parser.add_argument(
        "--mode",
        type=str,
        default="train",
        choices=["train", "evaluate", "demo"],
        help="Pipeline mode"
    )
    
    # Data arguments
    parser.add_argument("--images-dir", type=str, default="data/raw/images")
    parser.add_argument("--text-file", type=str, default="data/raw/texts/evaluate_60k_data_balanced_preprocessed.csv")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-question-length", type=int, default=128)
    parser.add_argument("--max-answer-length", type=int, default=64)
    
    # Model arguments
    parser.add_argument("--visual-backbone", type=str, default="openai/clip-vit-base-patch32")
    parser.add_argument("--text-encoder", type=str, default="vinai/phobert-base")
    parser.add_argument("--hidden-size", type=int, default=768)
    parser.add_argument("--num-decoder-layers", type=int, default=6)
    parser.add_argument("--freeze-visual", action="store_true", default=True)
    parser.add_argument("--freeze-text", action="store_true", default=True)
    
    # Training arguments
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--use-amp", action="store_true", default=True)
    parser.add_argument("--gradient-accumulation", type=int, default=2)
    
    # Generation arguments
    parser.add_argument("--max-generate-length", type=int, default=64)
    parser.add_argument("--num-beams", type=int, default=1)
    parser.add_argument("--do-sample", action="store_true")
    parser.add_argument("--temperature", type=float, default=1.0)
    
    # Path arguments
    parser.add_argument("--output-dir", type=str, default="outputs/generative")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints/generative")
    parser.add_argument("--resume", type=str, default=None)
    
    parser.add_argument("--seed", type=int, default=42)
    
    args = parser.parse_args()
    
    # Build config
    config = GenerativeVQAPipelineConfig(
        mode=args.mode,
        images_dir=args.images_dir,
        text_file=args.text_file,
        batch_size=args.batch_size,
        max_question_length=args.max_question_length,
        max_answer_length=args.max_answer_length,
        visual_backbone=args.visual_backbone,
        text_encoder=args.text_encoder,
        hidden_size=args.hidden_size,
        num_decoder_layers=args.num_decoder_layers,
        freeze_visual_encoder=args.freeze_visual,
        freeze_question_encoder=args.freeze_text,
        num_epochs=args.epochs,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        use_amp=args.use_amp,
        gradient_accumulation_steps=args.gradient_accumulation,
        max_generate_length=args.max_generate_length,
        num_beams=args.num_beams,
        do_sample=args.do_sample,
        temperature=args.temperature,
        output_dir=args.output_dir,
        checkpoint_dir=args.checkpoint_dir,
        resume_from=args.resume,
        seed=args.seed
    )
    
    # Run pipeline
    pipeline = GenerativeVQAPipeline(config=config)
    output = pipeline.run()
    
    sys.exit(0 if output.success else 1)


if __name__ == "__main__":
    main()
