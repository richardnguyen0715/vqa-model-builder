"""
Ablation Trainer Module
========================

Wraps the existing TrainingPipeline to support ablation experiments.
Handles:
- Expert masking (disabling specific experts in VQAMOELayer)
- Router swapping (replacing router type for each experiment)
- MOE-specific metric collection (routing weights, expert usage, load balance)
- Per-experiment checkpoint isolation
"""

import copy
import csv
import gc
import json
import logging
import time
import traceback
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.ablation.ablation_config import (
    AblationConfig,
    ExperimentConfig,
    ExpertAblationConfig,
    RouterAblationConfig,
)
from src.core.pipeline_logger import PipelineLogger, get_pipeline_logger
from src.core.training_pipeline import TrainingPipeline, TrainingPipelineConfig
from src.core.generative_training_pipeline import (
    GenerativeTrainingPipeline,
    GenerativeTrainingConfig,
)


# ══════════════════════════════════════════════════════════════════════
# Expert Mask Utilities
# ══════════════════════════════════════════════════════════════════════

def compute_expert_index_ranges(
    num_vision: int,
    num_text: int,
    num_multimodal: int,
    num_specialized: int,
) -> Dict[str, Tuple[int, int]]:
    """Compute index ranges for each expert type in the VQAMOELayer.

    VQAMOELayer creates experts in order:
        Vision → Text → Multimodal → Specialized

    Returns:
        Dict mapping expert type name to (start_idx, end_idx) inclusive range.
    """
    offset = 0
    ranges = {}
    for name, count in [
        ("vision", num_vision),
        ("text", num_text),
        ("multimodal", num_multimodal),
        ("specialized", num_specialized),
    ]:
        ranges[name] = (offset, offset + count - 1) if count > 0 else (offset, offset - 1)
        offset += count
    return ranges


def build_expert_mask(
    expert_config: ExpertAblationConfig,
    total_experts: int,
    default_per_type: Dict[str, int],
) -> List[bool]:
    """Build a boolean mask indicating which experts are *active*.

    Args:
        expert_config: Which expert types to enable.
        total_experts: Total expert count in the MOE layer.
        default_per_type: Default expert counts per type (from model construction).

    Returns:
        List[bool] of length total_experts. True = active.
    """
    ranges = compute_expert_index_ranges(
        default_per_type.get("vision", 2),
        default_per_type.get("text", 2),
        default_per_type.get("multimodal", 2),
        default_per_type.get("specialized", 2),
    )

    mask = [False] * total_experts
    active_set = set(expert_config.active_experts)

    for expert_type, (start, end) in ranges.items():
        if expert_type in active_set:
            for i in range(start, end + 1):
                if i < total_experts:
                    mask[i] = True

    return mask


# ══════════════════════════════════════════════════════════════════════
# MOE Modifier – inject expert mask & swap router
# ══════════════════════════════════════════════════════════════════════

def _find_moe_layer(model: nn.Module):
    """Find MOE layer in model, supporting both top-level and nested locations.

    Supported locations:
        - model.moe_layer  (VQAModel — classification)
        - model.fusion.moe_layer  (GenerativeVQAModel — generative)
    """
    # Top-level (VQAModel)
    if hasattr(model, "moe_layer") and model.moe_layer is not None:
        return model.moe_layer
    # Nested in fusion (GenerativeVQAModel)
    if hasattr(model, "fusion") and hasattr(model.fusion, "moe_layer") and model.fusion.moe_layer is not None:
        return model.fusion.moe_layer
    return None


class MOEModifier:
    """Applies ablation modifications to a VQA model's MOE layer.

    Supports both VQAModel (model.moe_layer) and GenerativeVQAModel
    (model.fusion.moe_layer).
    """

    def __init__(self, model: nn.Module, logger: Optional[PipelineLogger] = None):
        self.model = model
        self.logger = logger or get_pipeline_logger()
        self._original_router_state = None
        self._original_forward = None
        self._expert_mask: Optional[List[bool]] = None

    @property
    def has_moe(self) -> bool:
        return _find_moe_layer(self.model) is not None

    @property
    def moe_layer(self):
        return _find_moe_layer(self.model)

    def apply_expert_mask(self, mask: List[bool]) -> None:
        """Apply expert mask by hooking the forward pass to zero disabled expert outputs.

        Instead of removing experts from ModuleList (which breaks state_dict keys),
        we register a forward hook on the MOE layer that zeros out routing weights
        for disabled experts. This is clean and reversible.
        """
        if not self.has_moe:
            self.logger.warning("No MOE layer found, skipping expert mask")
            return

        self._expert_mask = mask
        moe = self.moe_layer
        num_experts = len(moe.experts)
        disabled = [i for i, active in enumerate(mask) if not active]

        if disabled:
            self.logger.info(
                f"Expert mask: disabling {len(disabled)}/{num_experts} experts: {disabled}"
            )

            # Store original router forward
            original_router_forward = moe.router.forward

            def masked_router_forward(x, *args, **kwargs):
                routing_weights, expert_indices, aux_outputs = original_router_forward(
                    x, *args, **kwargs
                )
                # Zero out weights for disabled experts
                for eidx in disabled:
                    routing_weights = routing_weights * (
                        (expert_indices != eidx).all(dim=-1, keepdim=True).float()
                    )
                    # Also mask expert_indices by setting disabled to -1
                    expert_indices = torch.where(
                        expert_indices == eidx,
                        torch.full_like(expert_indices, -1),
                        expert_indices,
                    )
                # Re-normalize routing weights
                weight_sum = routing_weights.sum(dim=-1, keepdim=True).clamp(min=1e-8)
                routing_weights = routing_weights / weight_sum
                return routing_weights, expert_indices, aux_outputs

            moe.router.forward = masked_router_forward
            self._original_router_state = original_router_forward
        else:
            self.logger.info("All experts active, no masking needed")

    def swap_router(self, router_config: RouterAblationConfig) -> None:
        """Replace the MOE layer's router with a different type."""
        if not self.has_moe:
            self.logger.warning("No MOE layer found, skipping router swap")
            return

        from src.modeling.moe.router import create_router

        moe = self.moe_layer
        old_type = moe.router.__class__.__name__

        new_router = create_router(
            router_type=router_config.router_type,
            input_dim=moe.input_dim,
            num_experts=moe.num_experts,
            top_k=router_config.top_k,
            noise_std=0.1 if router_config.noise_enabled else 0.0,
            use_aux_loss=True,
        )
        new_router = new_router.to(next(moe.parameters()).device)
        moe.router = new_router

        self.logger.info(
            f"Router swapped: {old_type} → {new_router.__class__.__name__} "
            f"(type={router_config.router_type}, top_k={router_config.top_k})"
        )

    def disable_moe(self) -> None:
        """Bypass the MOE layer entirely (no_moe baseline).

        Replaces MOE forward with identity (passthrough).
        """
        if not self.has_moe:
            return

        self._original_forward = self.moe_layer.forward

        def identity_forward(x, *args, **kwargs):
            return x

        self.moe_layer.forward = identity_forward
        self.logger.info("MOE layer disabled (identity passthrough)")

    def restore(self) -> None:
        """Restore original MOE state."""
        if self._original_router_state is not None and self.has_moe:
            self.moe_layer.router.forward = self._original_router_state
            self._original_router_state = None
        if self._original_forward is not None and self.has_moe:
            self.moe_layer.forward = self._original_forward
            self._original_forward = None
        self._expert_mask = None

    def collect_moe_metrics(self) -> Dict[str, Any]:
        """Collect MOE-specific metrics after a forward pass.

        Returns:
            Dict with routing entropy, expert usage, load imbalance, aux loss.
        """
        if not self.has_moe:
            return {}

        moe = self.moe_layer
        metrics: Dict[str, Any] = {}

        # Aux outputs from last forward
        if hasattr(moe, "aux_outputs") and moe.aux_outputs:
            aux = moe.aux_outputs
            if "load_balance_loss" in aux:
                lb = aux["load_balance_loss"]
                metrics["load_balance_loss"] = lb.item() if torch.is_tensor(lb) else float(lb)
            if "router_probs" in aux and aux["router_probs"] is not None:
                probs = aux["router_probs"]
                if torch.is_tensor(probs) and probs.numel() > 0:
                    # Routing entropy
                    entropy = -(probs * (probs + 1e-10).log()).sum(dim=-1).mean()
                    metrics["routing_entropy"] = entropy.item()
                    # Expert probability distribution
                    metrics["expert_prob_mean"] = probs.mean(dim=0).tolist()

        # Expert usage stats
        usage_ratios = []
        for i, expert in enumerate(moe.experts):
            if hasattr(expert, "get_usage_ratio"):
                usage_ratios.append(expert.get_usage_ratio())
            elif hasattr(expert, "usage_count"):
                usage_ratios.append(
                    expert.usage_count.item()
                    if torch.is_tensor(expert.usage_count)
                    else float(expert.usage_count)
                )

        if usage_ratios:
            metrics["expert_usage_ratios"] = usage_ratios
            total = sum(usage_ratios) or 1.0
            normalized = [u / total for u in usage_ratios]
            # Load imbalance = std / mean
            mean_usage = sum(normalized) / len(normalized)
            variance = sum((u - mean_usage) ** 2 for u in normalized) / len(normalized)
            metrics["load_imbalance"] = (variance ** 0.5) / (mean_usage + 1e-10)

        # Active expert mask
        if self._expert_mask is not None:
            metrics["active_experts"] = sum(self._expert_mask)
            metrics["expert_mask"] = self._expert_mask

        return metrics


# ══════════════════════════════════════════════════════════════════════
# Experiment Result
# ══════════════════════════════════════════════════════════════════════

@dataclass
class ExperimentResult:
    """Complete result for one ablation experiment."""

    experiment_id: str = ""
    experiment_config: Optional[Dict[str, Any]] = None

    # Standard VQA metrics
    metrics: Dict[str, float] = field(default_factory=dict)

    # MOE-specific metrics
    moe_metrics: Dict[str, Any] = field(default_factory=dict)

    # Training history
    training_history: List[Dict[str, float]] = field(default_factory=list)
    validation_history: List[Dict[str, float]] = field(default_factory=list)

    # Meta
    total_epochs: int = 0
    training_time: float = 0.0
    best_metric: float = 0.0
    checkpoint_path: str = ""
    status: str = "pending"  # pending, running, completed, failed, skipped
    error_message: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "experiment_id": self.experiment_id,
            "experiment_config": self.experiment_config,
            "metrics": self.metrics,
            "moe_metrics": {
                k: v if not isinstance(v, list) else v
                for k, v in self.moe_metrics.items()
            },
            "training_history": self.training_history,
            "validation_history": self.validation_history,
            "total_epochs": self.total_epochs,
            "training_time": self.training_time,
            "best_metric": self.best_metric,
            "checkpoint_path": self.checkpoint_path,
            "status": self.status,
            "error_message": self.error_message,
        }


# ══════════════════════════════════════════════════════════════════════
# Ablation Trainer
# ══════════════════════════════════════════════════════════════════════

class AblationTrainer:
    """Trains a single ablation experiment.

    Supports both classification (TrainingPipeline) and generative
    (GenerativeTrainingPipeline) models.

    Workflow:
    1. Deep-copy the model
    2. Apply expert mask / router swap / MOE disable
    3. Run appropriate TrainingPipeline
    4. Collect VQA + MOE metrics
    5. Return ExperimentResult
    """

    def __init__(
        self,
        base_model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        ablation_config: AblationConfig,
        device: Optional[torch.device] = None,
        logger: Optional[PipelineLogger] = None,
        model_config: Optional[Any] = None,
        vocabulary: Optional[Dict[str, int]] = None,
        id2answer: Optional[Dict[int, str]] = None,
        tokenizer: Optional[Any] = None,
    ):
        self.base_model = base_model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.ablation_config = ablation_config
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.logger = logger or get_pipeline_logger()
        self.model_config = model_config
        self.vocabulary = vocabulary
        self.id2answer = id2answer or {}
        self.tokenizer = tokenizer  # Required for generative model

        # Default per-type expert counts (deduced from model if possible)
        self._default_per_type = self._detect_expert_counts()

        # Track partial result from interrupted experiments (Ctrl+C)
        self._last_interrupted_result = None

        # Per-experiment file handler (attached/detached per run)
        self._experiment_file_handler: Optional[logging.FileHandler] = None

    @property
    def is_generative(self) -> bool:
        """Check if ablation uses generative model."""
        return self.ablation_config.is_generative

    def _detect_expert_counts(self) -> Dict[str, int]:
        """Detect expert counts per type from the model's MOE layer."""
        moe = _find_moe_layer(self.base_model)
        if moe is None:
            return {"vision": 2, "text": 2, "multimodal": 2, "specialized": 2}
        counts = {"vision": 0, "text": 0, "multimodal": 0, "specialized": 0}

        for expert in moe.experts:
            cls_name = expert.__class__.__name__.lower()
            if "vision" in cls_name:
                counts["vision"] += 1
            elif "text" in cls_name:
                counts["text"] += 1
            elif "multimodal" in cls_name:
                counts["multimodal"] += 1
            else:
                counts["specialized"] += 1

        return counts

    def _apply_ablation_modifications(
        self, model: nn.Module, experiment: ExperimentConfig
    ) -> MOEModifier:
        """Apply expert mask / router swap / MOE disable to model."""
        modifier = MOEModifier(model, self.logger)

        if experiment.expert_config.mode == "no_moe":
            modifier.disable_moe()
        elif modifier.has_moe:
            total_experts = len(modifier.moe_layer.experts)
            mask = build_expert_mask(
                experiment.expert_config,
                total_experts,
                self._default_per_type,
            )
            modifier.apply_expert_mask(mask)
            modifier.swap_router(experiment.router_config)

        return modifier

    def _create_generative_pipeline(
        self, model: nn.Module, experiment: ExperimentConfig, grad_accum: int
    ):
        """Create GenerativeTrainingPipeline for one experiment."""
        cfg = self.ablation_config
        gen_config = GenerativeTrainingConfig(
            num_epochs=experiment.num_epochs or cfg.num_epochs,
            learning_rate=experiment.learning_rate or cfg.learning_rate,
            weight_decay=0.01,
            warmup_ratio=0.1,
            gradient_accumulation_steps=grad_accum,
            max_grad_norm=1.0,
            use_amp=cfg.use_amp,
            early_stopping=cfg.early_stopping,
            patience=cfg.patience,
            checkpoint_dir=str(Path(cfg.checkpoint_dir) / experiment.experiment_id),
            save_best=True,
            save_every_epoch=False,
            metric_for_best=cfg.primary_metric,
            seed=experiment.seed,
            max_generate_length=cfg.max_generate_length,
            num_beams=cfg.num_beams,
            do_sample=cfg.do_sample,
            temperature=cfg.temperature,
        )
        return GenerativeTrainingPipeline(
            model=model,
            train_loader=self.train_loader,
            val_loader=self.val_loader,
            tokenizer=self.tokenizer,
            config=gen_config,
            logger=self.logger,
            device=self.device,
        )

    def _create_classification_pipeline(
        self, model: nn.Module, experiment: ExperimentConfig, grad_accum: int
    ):
        """Create classification TrainingPipeline for one experiment."""
        cfg = self.ablation_config
        tp_config = TrainingPipelineConfig(
            num_epochs=experiment.num_epochs or cfg.num_epochs,
            learning_rate=experiment.learning_rate or cfg.learning_rate,
            gradient_accumulation_steps=grad_accum,
            use_amp=cfg.use_amp,
            early_stopping=cfg.early_stopping,
            patience=cfg.patience,
            seed=experiment.seed,
            checkpoint_dir=str(Path(cfg.checkpoint_dir) / experiment.experiment_id),
            save_best=True,
            save_every_epoch=False,
            metric_for_best=cfg.primary_metric,
        )
        return TrainingPipeline(
            model=model,
            train_loader=self.train_loader,
            val_loader=self.val_loader,
            config=tp_config,
            logger=self.logger,
            device=self.device,
            model_config=self.model_config,
            vocabulary=self.vocabulary,
            id2answer=self.id2answer,
        )

    def _create_pipeline(
        self, model: nn.Module, experiment: ExperimentConfig, grad_accum: int
    ):
        """Create the appropriate training pipeline based on model_type."""
        if self.is_generative:
            return self._create_generative_pipeline(model, experiment, grad_accum)
        return self._create_classification_pipeline(model, experiment, grad_accum)

    # ── Per-experiment logging ────────────────────────────────────

    def _attach_experiment_log(self, experiment: ExperimentConfig) -> None:
        """Attach a dedicated file handler for this experiment's log.

        All messages emitted through ``self.logger`` will also be written to
        ``<log_dir>/<experiment_id>.log`` until ``_detach_experiment_log()``
        is called.
        """
        log_dir = Path(self.ablation_config.log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)

        safe_id = experiment.experiment_id.replace("/", "_").replace("\\", "_")
        log_path = log_dir / f"{safe_id}.log"

        handler = logging.FileHandler(log_path, mode="w", encoding="utf-8")
        handler.setLevel(logging.DEBUG)
        fmt = logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(fmt)

        # Attach to the underlying standard logger
        self.logger.logger.addHandler(handler)
        self._experiment_file_handler = handler
        self.logger.info(f"Experiment log: {log_path}")

    def _detach_experiment_log(self) -> None:
        """Remove the per-experiment file handler."""
        if self._experiment_file_handler is not None:
            self.logger.logger.removeHandler(self._experiment_file_handler)
            self._experiment_file_handler.close()
            self._experiment_file_handler = None

    # ── Per-epoch result persistence ──────────────────────────────

    def _save_epoch_results(
        self, experiment: ExperimentConfig, result: "ExperimentResult"
    ) -> Optional[str]:
        """Save per-epoch training and validation metrics for one experiment.

        Creates:
          <output_dir>/epoch_results/<experiment_id>/train_history.csv
          <output_dir>/epoch_results/<experiment_id>/val_history.csv
          <output_dir>/epoch_results/<experiment_id>/epoch_summary.json

        Returns:
            Path to the epoch results directory, or None if nothing to save.
        """
        if not result.training_history and not result.validation_history:
            return None

        out_dir = (
            Path(self.ablation_config.output_dir)
            / "epoch_results"
            / experiment.experiment_id.replace("/", "_").replace("\\", "_")
        )
        out_dir.mkdir(parents=True, exist_ok=True)

        # ── Training history CSV ──
        if result.training_history:
            train_csv = out_dir / "train_history.csv"
            train_keys = sorted(result.training_history[0].keys())
            with open(train_csv, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=["epoch"] + train_keys)
                writer.writeheader()
                for epoch_idx, row in enumerate(result.training_history, 1):
                    writer.writerow({"epoch": epoch_idx, **row})

        # ── Validation history CSV ──
        if result.validation_history:
            val_csv = out_dir / "val_history.csv"
            val_keys = sorted(result.validation_history[0].keys())
            with open(val_csv, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=["epoch"] + val_keys)
                writer.writeheader()
                for epoch_idx, row in enumerate(result.validation_history, 1):
                    writer.writerow({"epoch": epoch_idx, **row})

        # ── Epoch summary JSON (combined) ──
        epoch_summary = {
            "experiment_id": experiment.experiment_id,
            "model_type": "generative" if self.is_generative else "classification",
            "total_epochs": result.total_epochs,
            "best_metric": result.best_metric,
            "primary_metric": self.ablation_config.primary_metric,
            "status": result.status,
            "training_time": result.training_time,
            "training_history": result.training_history,
            "validation_history": result.validation_history,
            "saved_at": datetime.now().isoformat(),
        }
        summary_path = out_dir / "epoch_summary.json"
        with open(summary_path, "w") as f:
            json.dump(epoch_summary, f, indent=2, default=str)

        self.logger.info(f"Epoch results saved: {out_dir}")
        return str(out_dir)

    def run_experiment(self, experiment: ExperimentConfig) -> ExperimentResult:
        """Run a single ablation experiment.

        Args:
            experiment: Configuration for this experiment.

        Returns:
            ExperimentResult with full metrics and history.
        """
        result = ExperimentResult(
            experiment_id=experiment.experiment_id,
            experiment_config=experiment.to_dict(),
            status="running",
        )

        # ── Per-experiment log file ──
        self._attach_experiment_log(experiment)

        self.logger.start_stage(f"ABLATION: {experiment.experiment_id}")
        self.logger.info(f"Mode: {'generative' if self.is_generative else 'classification'}")
        self.logger.info(f"Expert config: {experiment.expert_config.description}")
        self.logger.info(f"Router config: {experiment.router_config.description}")

        start_time = time.time()
        grad_accum = self.ablation_config.gradient_accumulation_steps

        pipeline = None
        model = None

        try:
            # 1. Deep-copy model (base_model should be on CPU to avoid double GPU usage)
            model = copy.deepcopy(self.base_model)
            model = model.to(self.device)

            # 2. Apply ablation modifications
            modifier = self._apply_ablation_modifications(model, experiment)

            # 3. Create and run training pipeline
            pipeline = self._create_pipeline(model, experiment, grad_accum)

            try:
                output = pipeline.run()
            except (torch.cuda.OutOfMemoryError, RuntimeError) as oom_err:
                # Check if it's actually an OOM error
                if "out of memory" not in str(oom_err).lower():
                    raise

                # Cleanup current attempt
                del pipeline
                pipeline = None
                if model is not None:
                    try:
                        model.cpu()
                    except Exception:
                        pass
                    del model
                    model = None
                gc.collect()
                torch.cuda.empty_cache()

                # Retry with doubled gradient accumulation (halves memory)
                grad_accum *= 2
                self.logger.warning(
                    f"OOM detected! Retrying with gradient_accumulation_steps="
                    f"{grad_accum} (effective batch unchanged)"
                )

                # Recreate model & re-apply modifications
                model = copy.deepcopy(self.base_model)
                model = model.to(self.device)
                modifier = self._apply_ablation_modifications(model, experiment)

                pipeline = self._create_pipeline(model, experiment, grad_accum)
                output = pipeline.run()  # If this OOMs again, let it fail

            # 4. Collect MOE metrics from final state
            moe_metrics = modifier.collect_moe_metrics()

            # 5. Build result
            result.metrics = output.final_metrics
            result.moe_metrics = moe_metrics
            result.training_history = output.training_history
            result.validation_history = output.validation_history
            result.total_epochs = output.total_epochs
            result.training_time = time.time() - start_time
            result.best_metric = output.best_metric
            result.checkpoint_path = output.best_model_path or ""
            result.status = "completed"

            # Log the primary metric
            primary = self.ablation_config.primary_metric
            metric_val = result.metrics.get(primary, 0)
            self.logger.success(
                f"Experiment {experiment.experiment_id} completed: "
                f"{primary} = {metric_val:.4f} "
                f"({result.training_time:.1f}s)"
            )

            # 6. Save per-epoch results to disk
            try:
                self._save_epoch_results(experiment, result)
            except Exception as e:
                self.logger.warning(f"Failed to save epoch results: {e}")

        except KeyboardInterrupt:
            # ── Ctrl+C during training: save emergency checkpoint ──
            result.status = "interrupted"
            result.training_time = time.time() - start_time
            result.error_message = "Interrupted by user (Ctrl+C)"

            # Gather partial history from pipeline if available
            if pipeline is not None:
                result.training_history = getattr(pipeline, "training_history", [])
                result.validation_history = getattr(pipeline, "validation_history", [])
                result.total_epochs = getattr(pipeline, "current_epoch", 0)
                result.best_metric = getattr(pipeline, "best_metric", 0.0)
                result.best_model_path = getattr(pipeline, "best_model_path", None) or ""

            # Save emergency checkpoint with current model state
            ckpt_path = self._save_emergency_checkpoint(
                model=model,
                pipeline=pipeline,
                experiment=experiment,
            )
            if ckpt_path:
                result.checkpoint_path = ckpt_path

            self.logger.warning(
                f"Experiment {experiment.experiment_id} INTERRUPTED at "
                f"epoch {result.total_epochs} ({result.training_time:.1f}s)"
            )
            if ckpt_path:
                self.logger.info(f"  Emergency checkpoint saved: {ckpt_path}")

            # Save partial epoch results before exiting
            try:
                self._save_epoch_results(experiment, result)
            except Exception:
                pass

            # Store partial result so the runner can retrieve it
            self._last_interrupted_result = result

            # Re-raise so the runner can handle the pipeline-level shutdown
            raise

        except Exception as e:
            result.status = "failed"
            result.error_message = f"{type(e).__name__}: {str(e)}\n{traceback.format_exc()}"
            result.training_time = time.time() - start_time
            self.logger.error(f"Experiment {experiment.experiment_id} failed: {e}")

        finally:
            # Aggressive cleanup to free GPU memory for next experiment
            if "modifier" in locals() and modifier is not None:
                try:
                    modifier.restore()
                except Exception:
                    pass

            # Delete pipeline first (holds optimizer, scaler, scheduler — all GPU tensors)
            if pipeline is not None:
                # Clear optimizer state (large GPU allocation for Adam)
                if hasattr(pipeline, 'optimizer') and pipeline.optimizer is not None:
                    pipeline.optimizer.zero_grad(set_to_none=True)
                del pipeline
                pipeline = None

            # Move model to CPU then delete (ensures GPU memory is freed)
            if model is not None:
                try:
                    model.cpu()
                except Exception:
                    pass
                del model
                model = None

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            self.logger.end_stage(
                f"ABLATION: {experiment.experiment_id}",
                success=(result.status == "completed"),
            )

            # Detach per-experiment log file handler
            self._detach_experiment_log()

        return result

    def _save_emergency_checkpoint(
        self,
        model: Optional[nn.Module],
        pipeline: Optional[Any],
        experiment: ExperimentConfig,
    ) -> Optional[str]:
        """Save an emergency checkpoint when training is interrupted.

        Captures the current model state, optimizer state, epoch, and metrics
        so the experiment can potentially be analyzed or resumed later.

        Args:
            model: The model being trained (may be on GPU).
            pipeline: The training pipeline (holds optimizer, scheduler, etc.).
            experiment: Current experiment configuration.

        Returns:
            Path to saved checkpoint, or None if saving failed.
        """
        try:
            ckpt_dir = Path(self.ablation_config.checkpoint_dir) / "emergency_backups"
            ckpt_dir.mkdir(parents=True, exist_ok=True)

            safe_id = experiment.experiment_id.replace("/", "_").replace("\\", "_")
            ckpt_path = ckpt_dir / f"{safe_id}_interrupted.pt"

            checkpoint_data: Dict[str, Any] = {
                "experiment_id": experiment.experiment_id,
                "experiment_config": experiment.to_dict(),
                "status": "interrupted",
            }

            # Model state
            if model is not None:
                try:
                    checkpoint_data["model_state_dict"] = model.state_dict()
                except Exception:
                    pass

            # Pipeline state (optimizer, scheduler, epoch)
            if pipeline is not None:
                checkpoint_data["epoch"] = getattr(pipeline, "current_epoch", 0)
                checkpoint_data["global_step"] = getattr(pipeline, "global_step", 0)
                checkpoint_data["best_metric"] = getattr(pipeline, "best_metric", 0.0)

                if hasattr(pipeline, "optimizer") and pipeline.optimizer is not None:
                    try:
                        checkpoint_data["optimizer_state_dict"] = (
                            pipeline.optimizer.state_dict()
                        )
                    except Exception:
                        pass

                if hasattr(pipeline, "scheduler") and pipeline.scheduler is not None:
                    try:
                        checkpoint_data["scheduler_state_dict"] = (
                            pipeline.scheduler.state_dict()
                        )
                    except Exception:
                        pass

                if hasattr(pipeline, "scaler") and pipeline.scaler is not None:
                    try:
                        checkpoint_data["scaler_state_dict"] = (
                            pipeline.scaler.state_dict()
                        )
                    except Exception:
                        pass

                # Training history so far
                checkpoint_data["training_history"] = getattr(
                    pipeline, "training_history", []
                )
                checkpoint_data["validation_history"] = getattr(
                    pipeline, "validation_history", []
                )

            torch.save(checkpoint_data, ckpt_path)
            return str(ckpt_path)

        except Exception as e:
            self.logger.warning(f"Failed to save emergency checkpoint: {e}")
            return None
