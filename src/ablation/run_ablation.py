#!/usr/bin/env python3
"""
Ablation Study Entry Point
============================

Single-command entry point for the MOE ablation pipeline.

Usage:
    # From YAML config
    python -m src.ablation.run_ablation --config configs/ablation_config.yaml

    # With overrides
    python -m src.ablation.run_ablation --config configs/ablation_config.yaml \\
        --epochs 5 --batch-size 16 --output-dir outputs/ablation_quick

    # Dry run (list experiments without running)
    python -m src.ablation.run_ablation --config configs/ablation_config.yaml --dry-run

    # Resume from previous run
    python -m src.ablation.run_ablation --config configs/ablation_config.yaml --resume
"""

import argparse
import gc
import json
import os
import sys
from pathlib import Path

# Reduce CUDA memory fragmentation
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

# Ensure project root on sys.path
project_root = str(Path(__file__).resolve().parent.parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import torch

from src.ablation.ablation_config import AblationConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="AutoViVQA MOE Ablation Study Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--config",
        type=str,
        default="configs/ablation_config.yaml",
        help="Path to ablation YAML config (default: configs/ablation_config.yaml)",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to trained model checkpoint to use as base model",
    )

    # Overrides
    parser.add_argument("--epochs", type=int, default=None, help="Override num_epochs")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch_size")
    parser.add_argument("--lr", type=float, default=None, help="Override learning_rate")
    parser.add_argument("--seed", type=int, default=None, help="Override seed")
    parser.add_argument("--output-dir", type=str, default=None, help="Override output_dir")

    # Modes
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List experiments without running them",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        default=True,
        help="Resume from previous run (skip completed experiments)",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Do not resume — re-run all experiments from scratch",
    )

    # Device
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device (cuda, cpu, cuda:0, etc.)",
    )

    return parser.parse_args()


def dry_run(config: AblationConfig) -> None:
    """List all experiments without running them."""
    experiments = config.generate_experiment_matrix()

    print(f"\n{'='*70}")
    print(f"  ABLATION STUDY DRY RUN: {config.name}")
    print(f"  Total experiments: {len(experiments)}")
    print(f"{'='*70}\n")

    for i, exp in enumerate(experiments, 1):
        print(f"  [{i:3d}] {exp.experiment_id}")
        print(f"        Expert: {exp.expert_config.description}")
        print(f"        Router: {exp.router_config.description}")
        print(f"        Priority: {exp.priority}  Tags: {exp.tags}")
        print()

    # Count by mode
    modes = {}
    for exp in experiments:
        m = exp.expert_config.mode
        modes[m] = modes.get(m, 0) + 1

    print(f"  Breakdown by mode:")
    for m, count in sorted(modes.items()):
        print(f"    {m}: {count} experiments")
    print(f"\n{'='*70}")


def main():
    args = parse_args()

    # Load config
    config = AblationConfig.from_yaml(args.config)

    # Apply overrides
    if args.epochs is not None:
        config.num_epochs = args.epochs
    if args.batch_size is not None:
        config.batch_size = args.batch_size
    if args.lr is not None:
        config.learning_rate = args.lr
    if args.seed is not None:
        config.seed = args.seed
    if args.output_dir is not None:
        config.output_dir = args.output_dir
    if args.no_resume:
        config.resume = False
    elif args.resume:
        config.resume = True

    # Dry run mode
    if args.dry_run:
        dry_run(config)
        return

    # Device
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"\nDevice: {device}")
    print(f"Config: {args.config}")
    print(f"Output: {config.output_dir}\n")

    # ── Build model and data loaders ──
    # Load from checkpoint if provided, otherwise build from config
    from src.core.data_pipeline import DataPipeline, DataPipelineConfig
    from src.core.pipeline_logger import get_pipeline_logger
    from src.modeling.meta_arch.vqa_model import create_vqa_model
    from src.modeling.meta_arch.vqa_config import VQAModelConfig

    pipeline_logger = get_pipeline_logger()

    # Data pipeline — build DataPipelineConfig from ablation config fields
    data_config = DataPipelineConfig(
        images_dir=config.images_dir,
        text_file=config.text_file,
        train_ratio=config.train_ratio,
        val_ratio=config.val_ratio,
        test_ratio=config.test_ratio,
        batch_size=config.batch_size,
        eval_batch_size=config.eval_batch_size,
    )
    data_pipeline = DataPipeline(config=data_config, logger=pipeline_logger)
    data_output = data_pipeline.run()

    train_loader = data_output.train_loader
    val_loader = data_output.val_loader
    vocabulary = data_output.answer2id
    id2answer = data_output.id2answer

    # Model
    if args.checkpoint and Path(args.checkpoint).exists():
        print(f"Loading model from checkpoint: {args.checkpoint}")
        # Load to CPU — each experiment will copy to GPU individually
        checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        model_config = checkpoint.get("model_config")
        model = create_vqa_model(config=model_config)
        model.load_state_dict(checkpoint["model_state_dict"])
        stored_vocab = checkpoint.get("vocabulary")
        # Free checkpoint tensor memory
        del checkpoint
        gc.collect()
        if stored_vocab:
            vocabulary = stored_vocab
            id2answer = {v: k for k, v in vocabulary.items()}
    else:
        # Build model from config
        from src.modeling.meta_arch.vqa_config import (
            VQAModelConfig,
            VisualEncoderConfig,
            TextEncoderConfig,
            FusionConfig,
            MOEConfig,
            AnswerHeadConfig,
        )

        model_config = VQAModelConfig(
            visual_encoder=VisualEncoderConfig(
                backbone_type=config.visual_backbone,
                model_name=config.visual_model_name,
                output_dim=config.embed_dim,
            ),
            text_encoder=TextEncoderConfig(
                encoder_type=config.text_encoder_type,
                model_name=config.text_model_name,
                output_dim=config.embed_dim,
            ),
            fusion=FusionConfig(
                fusion_type=config.fusion_type,
                hidden_dim=config.embed_dim,
                output_dim=config.embed_dim,
            ),
            moe=MOEConfig(
                use_moe=True,
                num_experts=8,
                top_k=2,
                hidden_dim=config.moe_hidden_dim,
            ),
            answer_head=AnswerHeadConfig(
                num_answers=data_output.num_classes if data_output.num_classes else 3000,
            ),
            embed_dim=config.embed_dim,
        )
        model = create_vqa_model(config=model_config)

    # Keep base model on CPU — each experiment will deep-copy and move to GPU
    # This prevents OOM from having base + experiment copy both on GPU
    model = model.cpu()

    print(f"Model: {model.__class__.__name__}")
    print(f"  MOE layer: {'Yes' if hasattr(model, 'moe_layer') and model.moe_layer else 'No'}")
    if hasattr(model, "moe_layer") and model.moe_layer:
        print(f"  Experts: {len(model.moe_layer.experts)}")
        print(f"  Router: {model.moe_layer.router.__class__.__name__}")
    print(f"  Base model kept on CPU to avoid CUDA OOM")
    print()

    # ── Run ablation ──
    from src.ablation.ablation_runner import AblationRunner

    runner = AblationRunner(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        device=device,
        pipeline_logger=pipeline_logger,
        model_config=model_config,
        vocabulary=vocabulary,
        id2answer=id2answer,
    )

    summary = runner.run()

    # Print key findings
    print(f"\n{'='*70}")
    print("  KEY FINDINGS:")
    for i, finding in enumerate(summary.get("key_findings", []), 1):
        print(f"  {i}. {finding}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
