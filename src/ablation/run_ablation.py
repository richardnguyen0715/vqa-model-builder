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
import warnings
from pathlib import Path

# Suppress known deprecation warnings before importing torch
warnings.filterwarnings('ignore', message='.*pynvml.*')
warnings.filterwarnings('ignore', message='.*Support for mismatched key_padding_mask.*')

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

    # Experiment selection
    parser.add_argument(
        "--experiments",
        type=str,
        default=None,
        help=(
            "Run only specific experiments by number (resume by default). "
            "Supports comma-separated values and ranges. "
            "Example: --experiments 1,3,5-7  (runs experiments 1,3,5,6,7). "
            "Use --dry-run first to see the full numbered list."
        ),
    )
    parser.add_argument(
        "--rerun",
        type=str,
        default=None,
        help=(
            "Force re-run specific experiments from scratch (delete old results). "
            "Supports same format as --experiments. "
            "Can be combined with --experiments for mixed resume/rerun. "
            "Example: --experiments 10,11 --rerun 10  "
            "(resume 11 if completed, re-run 10 from scratch)."
        ),
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Interactively select which experiments to run",
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
    print(f"  Model type: {config.model_type}")
    print(f"  Primary metric: {config.primary_metric}")
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


# ══════════════════════════════════════════════════════════════════════
# Experiment selection utilities
# ══════════════════════════════════════════════════════════════════════

def parse_experiment_selection(selection_str: str, total: int) -> list[int]:
    """Parse a selection string like '1,3,5-7' into a sorted list of 1-based indices.

    Supports:
        - Single numbers: '3'
        - Comma-separated: '1,3,5'
        - Ranges: '5-7' (inclusive)
        - Mixed: '1,3,5-7,10'

    Raises:
        ValueError: If any index is out of range [1, total].
    """
    indices: set[int] = set()
    parts = selection_str.strip().split(",")
    for part in parts:
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo, hi = part.split("-", 1)
            lo, hi = int(lo.strip()), int(hi.strip())
            if lo > hi:
                lo, hi = hi, lo
            indices.update(range(lo, hi + 1))
        else:
            indices.add(int(part))

    # Validate
    out_of_range = [i for i in indices if i < 1 or i > total]
    if out_of_range:
        raise ValueError(
            f"Experiment numbers out of range [1, {total}]: {sorted(out_of_range)}"
        )
    return sorted(indices)


def interactive_select(config: AblationConfig) -> list[int] | None:
    """Show experiment list and let user interactively pick which to run.

    Returns:
        Sorted list of 1-based experiment indices, or None to run all.
    """
    experiments = config.generate_experiment_matrix()
    total = len(experiments)

    print(f"\n{'='*70}")
    print(f"  EXPERIMENT LIST: {config.name}")
    print(f"  Total experiments: {total}")
    print(f"{'='*70}\n")

    # Group by mode for readability
    current_mode = None
    for i, exp in enumerate(experiments, 1):
        mode = exp.expert_config.mode
        if mode != current_mode:
            current_mode = mode
            print(f"  ── {mode.upper()} {'─'*(55 - len(mode))}")
        print(f"  [{i:3d}] {exp.experiment_id}")
        print(f"        {exp.expert_config.description} | {exp.router_config.description}")
    print()

    print(f"{'─'*70}")
    print("  Enter experiment numbers to run.")
    print("  Format: 1,3,5-7  |  'all' = run all  |  'q' = quit")
    print(f"{'─'*70}")

    while True:
        try:
            user_input = input("\n  Your selection: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n  Cancelled.")
            return []

        if not user_input or user_input.lower() == "q":
            print("  Cancelled.")
            return []
        if user_input.lower() == "all":
            return None  # None = run all

        try:
            selected = parse_experiment_selection(user_input, total)
            if not selected:
                print("  No experiments selected. Try again.")
                continue

            # Confirm
            print(f"\n  Selected {len(selected)} experiment(s):")
            for idx in selected:
                exp = experiments[idx - 1]
                print(f"    [{idx:3d}] {exp.experiment_id}")

            confirm = input("\n  Proceed? [Y/n]: ").strip().lower()
            if confirm in ("", "y", "yes"):
                return selected
            else:
                print("  Selection cleared. Try again.")
        except ValueError as e:
            print(f"  Invalid input: {e}")


# ══════════════════════════════════════════════════════════════════════
# Generative model + data setup
# ══════════════════════════════════════════════════════════════════════

def _build_generative(config, args, pipeline_logger):
    """Build GenerativeVQAModel and generative data loaders.

    Returns:
        (model, train_loader, val_loader, model_config, tokenizer,
         vocabulary, id2answer)
    """
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from transformers import AutoTokenizer

    from src.modeling.meta_arch.generative_vqa_model import (
        GenerativeVQAModel,
        GenerativeVQAConfig,
        create_generative_vqa_model,
        get_default_generative_vqa_config,
    )
    from src.data.generative_dataset import (
        GenerativeVQADataset,
        generative_vqa_collate_fn,
    )
    from torch.utils.data import DataLoader

    # ── Tokenizer ──
    tokenizer = AutoTokenizer.from_pretrained(config.text_model_name)

    # ── Data ──
    pipeline_logger.info(f"Loading data from: {config.text_file}")
    df = pd.read_csv(config.text_file)
    pipeline_logger.info(f"Total samples: {len(df)}")

    train_df, temp_df = train_test_split(
        df,
        test_size=1 - config.train_ratio,
        random_state=config.seed,
    )
    val_ratio_adj = config.val_ratio / (config.val_ratio + config.test_ratio)
    val_df, _ = train_test_split(
        temp_df,
        test_size=1 - val_ratio_adj,
        random_state=config.seed,
    )

    pipeline_logger.info(f"Train: {len(train_df)} | Val: {len(val_df)}")

    train_dataset = GenerativeVQADataset(
        df=train_df,
        images_dir=config.images_dir,
        tokenizer=tokenizer,
        max_question_length=config.max_question_length,
        max_answer_length=config.max_answer_length,
        mode="train",
    )
    val_dataset = GenerativeVQADataset(
        df=val_df,
        images_dir=config.images_dir,
        tokenizer=tokenizer,
        max_question_length=config.max_question_length,
        max_answer_length=config.max_answer_length,
        mode="val",
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=4,
        collate_fn=generative_vqa_collate_fn,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.eval_batch_size,
        shuffle=False,
        num_workers=4,
        collate_fn=generative_vqa_collate_fn,
        pin_memory=torch.cuda.is_available(),
    )

    # ── Model ──
    vocabulary = None
    id2answer = None

    if args.checkpoint and Path(args.checkpoint).exists():
        print(f"Loading generative model from checkpoint: {args.checkpoint}")
        checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        model_config = checkpoint.get("config") or checkpoint.get("model_config")
        if isinstance(model_config, GenerativeVQAConfig):
            model = create_generative_vqa_model(config=model_config)
        else:
            model = create_generative_vqa_model()
        model.load_state_dict(checkpoint["model_state_dict"])
        del checkpoint
        gc.collect()
    else:
        model_config = get_default_generative_vqa_config(
            visual_backbone=config.visual_model_name,
            text_encoder=config.text_model_name,
            vocab_size=len(tokenizer),
            bos_token_id=tokenizer.bos_token_id or 0,
            eos_token_id=tokenizer.eos_token_id or 2,
            pad_token_id=tokenizer.pad_token_id or 1,
            # Generative model settings
            freeze_visual_encoder=config.freeze_visual_encoder,
            freeze_question_encoder=config.freeze_question_encoder,
            max_answer_length=config.max_answer_length,
            # MOE
            use_moe=config.use_moe,
            moe_type=config.moe_type,
            num_experts=config.num_experts,
            num_experts_per_token=config.num_experts_per_token,
            expert_capacity_factor=config.expert_capacity_factor,
            moe_loss_weight=config.moe_loss_weight,
            moe_position=config.moe_position,
            num_vision_experts=config.num_vision_experts,
            num_text_experts=config.num_text_experts,
            num_multimodal_experts=config.num_multimodal_experts,
            num_specialized_experts=config.num_specialized_experts,
            vietnamese_optimized=config.vietnamese_optimized,
        )
        # Override extra fields
        model_config.hidden_size = config.embed_dim
        model_config.num_decoder_layers = config.num_decoder_layers
        model_config.num_attention_heads = config.num_attention_heads
        model_config.decoder_ff_dim = config.decoder_ff_dim
        model_config.decoder_dropout = config.decoder_dropout
        model_config.label_smoothing = config.label_smoothing
        model_config.tie_word_embeddings = config.tie_word_embeddings
        model_config.__post_init__()
        model = create_generative_vqa_model(config=model_config)

    return model, train_loader, val_loader, model_config, tokenizer, vocabulary, id2answer


# ══════════════════════════════════════════════════════════════════════
# Classification model + data setup (legacy)
# ══════════════════════════════════════════════════════════════════════

def _build_classification(config, args, pipeline_logger):
    """Build VQAModel (classification) and data loaders.

    Returns:
        (model, train_loader, val_loader, model_config, tokenizer,
         vocabulary, id2answer)
    """
    from src.core.data_pipeline import DataPipeline, DataPipelineConfig
    from src.modeling.meta_arch.vqa_model import create_vqa_model

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
    tokenizer = None

    if args.checkpoint and Path(args.checkpoint).exists():
        print(f"Loading classification model from checkpoint: {args.checkpoint}")
        checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        model_config = checkpoint.get("model_config")
        model = create_vqa_model(config=model_config)
        model.load_state_dict(checkpoint["model_state_dict"])
        stored_vocab = checkpoint.get("vocabulary")
        del checkpoint
        gc.collect()
        if stored_vocab:
            vocabulary = stored_vocab
            id2answer = {v: k for k, v in vocabulary.items()}
    else:
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
                num_experts=config.num_experts,
                top_k=config.num_experts_per_token,
                hidden_dim=config.moe_hidden_dim,
            ),
            answer_head=AnswerHeadConfig(
                num_answers=data_output.num_classes if data_output.num_classes else 3000,
            ),
            embed_dim=config.embed_dim,
        )
        model = create_vqa_model(config=model_config)

    return model, train_loader, val_loader, model_config, tokenizer, vocabulary, id2answer


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

    # ── Experiment selection ──
    selected_indices = None  # None = run all
    rerun_indices = None     # None = no forced reruns

    if args.interactive:
        selected_indices = interactive_select(config)
        if selected_indices is not None and len(selected_indices) == 0:
            print("No experiments selected. Exiting.")
            return
    elif args.experiments:
        experiments = config.generate_experiment_matrix()
        selected_indices = parse_experiment_selection(
            args.experiments, len(experiments)
        )
        print(f"\nSelected {len(selected_indices)} experiment(s): {selected_indices}")

    # Parse --rerun (force re-run from scratch)
    if args.rerun:
        experiments = config.generate_experiment_matrix()
        rerun_indices = parse_experiment_selection(
            args.rerun, len(experiments)
        )
        print(f"Force re-run {len(rerun_indices)} experiment(s): {rerun_indices}")
        # Auto-include rerun experiments in selected set
        if selected_indices is not None:
            merged = sorted(set(selected_indices) | set(rerun_indices))
            selected_indices = merged
        # If no --experiments but --rerun specified, only run those
        elif not args.interactive:
            selected_indices = rerun_indices

    # Device
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"\nDevice: {device}")
    print(f"Config: {args.config}")
    print(f"Model type: {config.model_type}")
    print(f"Primary metric: {config.primary_metric}")
    print(f"Output: {config.output_dir}\n")

    from src.core.pipeline_logger import get_pipeline_logger

    pipeline_logger = get_pipeline_logger()

    # ── Build model and data loaders based on model_type ──
    if config.is_generative:
        print("Building GENERATIVE VQA model and data...")
        model, train_loader, val_loader, model_config, tokenizer, vocabulary, id2answer = (
            _build_generative(config, args, pipeline_logger)
        )
    else:
        print("Building CLASSIFICATION VQA model and data...")
        model, train_loader, val_loader, model_config, tokenizer, vocabulary, id2answer = (
            _build_classification(config, args, pipeline_logger)
        )

    # Keep base model on CPU — each experiment will deep-copy and move to GPU
    # This prevents OOM from having base + experiment copy both on GPU
    model = model.cpu()

    # ── Log model info ──
    from src.ablation.ablation_trainer import _find_moe_layer

    moe = _find_moe_layer(model)

    print(f"Model: {model.__class__.__name__}")
    print(f"  MOE layer: {'Yes' if moe is not None else 'No'}")
    if moe is not None:
        print(f"  Experts: {len(moe.experts)}")
        print(f"  Router: {moe.router.__class__.__name__}")
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
        tokenizer=tokenizer,
    )

    summary = runner.run(
        selected_indices=selected_indices,
        rerun_indices=rerun_indices,
    )

    # Print key findings
    if summary.get("interrupted"):
        print(f"\n{'='*70}")
        print("  STUDY INTERRUPTED — partial results saved.")
        interrupted_at = summary.get("interrupted_at", "unknown")
        completed = summary.get("completed", summary.get("total_completed", "?"))
        total_exp = summary.get("total_experiments", "?")
        print(f"  Interrupted at: {interrupted_at}")
        print(f"  Completed: {completed}/{total_exp}")
        print(f"  Resume with: python -m src.ablation.run_ablation --config {args.config} --resume")
        print(f"{'='*70}\n")
    else:
        print(f"\n{'='*70}")
        print("  KEY FINDINGS:")
        for i, finding in enumerate(summary.get("key_findings", []), 1):
            print(f"  {i}. {finding}")
        print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
