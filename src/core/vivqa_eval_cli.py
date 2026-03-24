"""
VIVQA Model Evaluation CLI
Command-line interface for running inference and evaluation on VIVQA dataset.

Usage:
    python -m src.core.vivqa_eval_cli --checkpoint model.pt
    python -m src.core.vivqa_eval_cli --checkpoint model.pt --batch-size 64
    python -m src.core.vivqa_eval_cli --checkpoint model.pt --output-dir results/
    python -m src.core.vivqa_eval_cli --checkpoint model.pt --config configs/vivqa_evaluation_configs.yaml
"""

import argparse
import sys
import warnings
from pathlib import Path
from typing import Optional

# Suppress warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=RuntimeWarning)

import torch
from transformers import AutoTokenizer

from src.core.vivqa_evaluation_pipeline import VivqaEvaluationPipeline, EvaluationConfig
from src.core.pipeline_logger import get_pipeline_logger
from src.middleware.logger import data_process_logger


def load_model_from_checkpoint(checkpoint_path: str, device: str) -> tuple:
    """
    Load model and tokenizer from checkpoint.
    
    Args:
        checkpoint_path: Path to model checkpoint (.pt file)
        device: Device to load model on
        
    Returns:
        Tuple of (model, tokenizer)
    """
    logger = get_pipeline_logger()
    logger.info(f"Loading model from checkpoint: {checkpoint_path}")
    
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    device = torch.device(device if device != 'auto' else ('cuda' if torch.cuda.is_available() else 'cpu'))
    logger.info(f"Using device: {device}")
    
    # Load checkpoint
    try:
        # Use weights_only=False for PyTorch 2.6+ compatibility
        # This checkpoint is from the project, so it's trusted
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        logger.info(f"Checkpoint loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load checkpoint: {e}")
        raise
    
    # Load model (checkpoint should contain model state_dict and config)
    try:
        # Try to import the model class and config
        from src.modeling.meta_arch.generative_vqa_model import (
            GenerativeVQAModel, 
            GenerativeVQAConfig
        )
        
        logger.info(f"Checkpoint keys: {checkpoint.keys() if isinstance(checkpoint, dict) else 'model instance'}")
        
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            # Checkpoint is a dict with state_dict
            model_state = checkpoint['model_state_dict']
            # Note: checkpoint has 'config' key, not 'model_config'
            checkpoint_config_dict = checkpoint.get('config', {})
            logger.info(f"Model config from checkpoint: {checkpoint_config_dict}")
            
            # Always create a GenerativeVQAConfig instance
            if checkpoint_config_dict:
                # Config exists in checkpoint, use it (convert dict to dataclass if needed)
                if isinstance(checkpoint_config_dict, dict):
                    logger.info("Loading config from checkpoint dict")
                    model_config = GenerativeVQAConfig(**checkpoint_config_dict)
                else:
                    model_config = checkpoint_config_dict
                logger.info("Model config loaded from checkpoint")
            else:
                # No config in checkpoint, use defaults
                logger.warning("Empty config in checkpoint, using default GenerativeVQAConfig")
                model_config = GenerativeVQAConfig()
            
            # Initialize model with config (this is always required)
            model = GenerativeVQAModel(config=model_config)
            model.load_state_dict(model_state)
            logger.info("Model initialized and state dict loaded successfully")
        else:
            # Checkpoint is model instance or full state dict
            if isinstance(checkpoint, dict):
                logger.info(f"Loading from full state dict (keys: {list(checkpoint.keys())[:5]}...)")
                # Still need config - use defaults
                model_config = GenerativeVQAConfig()
                model = GenerativeVQAModel(config=model_config)
                model.load_state_dict(checkpoint)
                logger.info("Loading from state dict successfully")
            else:
                logger.info("Loading model instance directly")
                model = checkpoint
        
        model.to(device)
        logger.info("Model loaded successfully")
    
    except ImportError:
        logger.error("Could not import GenerativeVQAModel from src.modeling.meta_arch")
        raise
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        logger.error(f"Checkpoint structure: {type(checkpoint)}, keys: {checkpoint.keys() if isinstance(checkpoint, dict) else 'N/A'}")
        raise
    
    # Load tokenizer
    try:
        # Try to get tokenizer from checkpoint
        if isinstance(checkpoint, dict) and 'tokenizer_name' in checkpoint:
            tokenizer_name = checkpoint['tokenizer_name']
            logger.info(f"Loading tokenizer: {tokenizer_name}")
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        else:
            # Default to PhoBERT
            logger.info("Loading default tokenizer: vinai/phobert-base")
            tokenizer = AutoTokenizer.from_pretrained('vinai/phobert-base')
        
        logger.info("Tokenizer loaded successfully")
    except Exception as e:
        logger.warning(f"Failed to load tokenizer: {e}")
        # Fallback - will use default
        logger.info("Using default PhoBERT tokenizer")
        try:
            tokenizer = AutoTokenizer.from_pretrained('vinai/phobert-base')
        except:
            raise
    
    return model, tokenizer


def create_config_from_args(args: argparse.Namespace) -> EvaluationConfig:
    """Create EvaluationConfig from command-line arguments."""
    config = EvaluationConfig(
        csv_path=args.csv_path,
        images_dir=args.images_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        checkpoint_path=args.checkpoint,
        device=args.device,
        max_generate_length=args.max_length,
        num_beams=args.num_beams,
        do_sample=args.do_sample,
        temperature=args.temperature,
        top_p=args.top_p,
        output_dir=args.output_dir,
        save_predictions=args.save_predictions,
        save_metrics=args.save_metrics,
        num_samples_to_display=args.num_samples_to_display,
        log_interval=args.log_interval
    )
    return config


def main():
    """Main entry point for evaluation CLI."""
    parser = argparse.ArgumentParser(
        description="VIVQA Model Evaluation - Evaluate generative VQA models on VIVQA dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic evaluation
  python -m src.core.vivqa_eval_cli --checkpoint checkpoints/RichardNguyen_ViMoE-VQA/best_generative_model.pt
  
  # Custom batch size and output
  python -m src.core.vivqa_eval_cli --checkpoint checkpoints/RichardNguyen_ViMoE-VQA/best_generative_model.pt --batch-size 64 --output-dir results/eval/
  
  # Generation with beam search
  python -m src.core.vivqa_eval_cli --checkpoint checkpoints/RichardNguyen_ViMoE-VQA/best_generative_model.pt --num-beams 3
  
  # With sampling
  python -m src.core.vivqa_eval_cli --checkpoint checkpoints/RichardNguyen_ViMoE-VQA/best_generative_model.pt --do-sample --temperature 0.7 --top-p 0.9
  
  # CPU evaluation
  python -m src.core.vivqa_eval_cli --checkpoint checkpoints/RichardNguyen_ViMoE-VQA/best_generative_model.pt --device cpu
        """
    )
    
    # Required arguments
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint (.pt file)"
    )
    
    # Data arguments
    data_group = parser.add_argument_group('Data Configuration')
    data_group.add_argument(
        "--csv-path",
        type=str,
        default="data/vivqa/test.csv",
        help="Path to evaluation CSV file"
    )
    data_group.add_argument(
        "--images-dir",
        type=str,
        default="data/vivqa/images",
        help="Directory containing COCO images"
    )
    data_group.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for evaluation"
    )
    data_group.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of data loading workers"
    )
    
    # Device arguments
    device_group = parser.add_argument_group('Device Configuration')
    device_group.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help="Device to run evaluation on"
    )
    
    # Generation arguments
    gen_group = parser.add_argument_group('Generation Configuration')
    gen_group.add_argument(
        "--max-length",
        type=int,
        default=64,
        help="Maximum generation length"
    )
    gen_group.add_argument(
        "--num-beams",
        type=int,
        default=1,
        help="Number of beams for beam search (1=greedy)"
    )
    gen_group.add_argument(
        "--do-sample",
        action="store_true",
        help="Use sampling instead of greedy decoding"
    )
    gen_group.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Temperature for sampling"
    )
    gen_group.add_argument(
        "--top-p",
        type=float,
        default=0.9,
        help="Top-p (nucleus) probability for sampling"
    )
    
    # Output arguments
    output_group = parser.add_argument_group('Output Configuration')
    output_group.add_argument(
        "--output-dir",
        type=str,
        default="outputs/evaluation",
        help="Directory to save evaluation results"
    )
    output_group.add_argument(
        "--save-predictions",
        action="store_true",
        default=True,
        help="Save predictions to JSON"
    )
    output_group.add_argument(
        "--no-save-predictions",
        action="store_false",
        dest="save_predictions",
        help="Do not save predictions"
    )
    output_group.add_argument(
        "--save-metrics",
        action="store_true",
        default=True,
        help="Save metrics to JSON"
    )
    output_group.add_argument(
        "--no-save-metrics",
        action="store_false",
        dest="save_metrics",
        help="Do not save metrics"
    )
    
    # Logging arguments
    log_group = parser.add_argument_group('Logging Configuration')
    log_group.add_argument(
        "--num-samples",
        type=int,
        default=10,
        dest="num_samples_to_display",
        help="Number of sample predictions to display"
    )
    log_group.add_argument(
        "--log-interval",
        type=int,
        default=10,
        help="Logging interval (batches)"
    )
    
    args = parser.parse_args()
    
    logger = get_pipeline_logger()
    
    try:
        # Print header
        logger.info("\n" + "=" * 80)
        logger.info("VIVQA MODEL EVALUATION")
        logger.info("=" * 80)
        logger.info(f"Checkpoint: {args.checkpoint}")
        logger.info(f"Evaluation CSV: {args.csv_path}")
        logger.info(f"Batch size: {args.batch_size}")
        logger.info("=" * 80 + "\n")
        
        # Load model and tokenizer
        model, tokenizer = load_model_from_checkpoint(args.checkpoint, args.device)
        
        # Create config
        config = create_config_from_args(args)
        
        # Create and run pipeline
        pipeline = VivqaEvaluationPipeline(
            model=model,
            tokenizer=tokenizer,
            config=config,
            logger=logger
        )
        
        results = pipeline.evaluate()
        
        # Print final message
        logger.info("\n" + "=" * 80)
        logger.info("EVALUATION COMPLETED SUCCESSFULLY")
        logger.info("=" * 80)
        logger.info(f"Results saved to: {config.output_dir}")
        logger.info("=" * 80 + "\n")
        
        return 0
    
    except KeyboardInterrupt:
        logger.info("\nEvaluation interrupted by user")
        return 130
    except Exception as e:
        logger.error(f"Evaluation failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
