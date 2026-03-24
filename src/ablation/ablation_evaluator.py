"""
Ablation Evaluator Module
===========================

Aggregates and computes evaluation metrics across all ablation experiments.
Handles:
- Per-experiment metric collection & normalization
- Cross-experiment comparison tables
- MOE-specific metric aggregation (expert usage, routing entropy, load balance)
- Statistical significance testing (paired bootstrap / McNemar)
"""

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch


# ══════════════════════════════════════════════════════════════════════
# Metric Names
# ══════════════════════════════════════════════════════════════════════

STANDARD_METRICS = [
    "vqa_accuracy",
    "accuracy",
    "exact_match",
    "bleu",
    "meteor",
    "rouge_l",
    "cider",
    "precision",
    "recall",
    "f1",
    "loss",
]

# Metrics specific to generative (seq2seq) models
GENERATIVE_METRICS = [
    "bleu",
    "meteor",
    "rouge_l",
    "cider",
    "exact_match",
    "precision",
    "recall",
    "f1",
    "loss",
    "perplexity",
]

# Metrics specific to classification models
CLASSIFICATION_METRICS = [
    "vqa_accuracy",
    "accuracy",
    "exact_match",
    "precision",
    "recall",
    "f1",
    "loss",
]

MOE_METRICS = [
    "load_balance_loss",
    "routing_entropy",
    "load_imbalance",
    "active_experts",
]


def get_metrics_for_model_type(model_type: str = "generative") -> list:
    """Return the appropriate metric list for the given model type.

    Args:
        model_type: 'generative' or 'classification'.

    Returns:
        List of metric name strings.
    """
    if model_type == "generative":
        return GENERATIVE_METRICS
    elif model_type == "classification":
        return CLASSIFICATION_METRICS
    return STANDARD_METRICS


# ══════════════════════════════════════════════════════════════════════
# MetricResult
# ══════════════════════════════════════════════════════════════════════

@dataclass
class MetricSummary:
    """Summary statistics for a single metric across experiments."""
    name: str
    values: List[float] = field(default_factory=list)
    experiment_ids: List[str] = field(default_factory=list)

    @property
    def mean(self) -> float:
        return sum(self.values) / len(self.values) if self.values else 0.0

    @property
    def std(self) -> float:
        if len(self.values) < 2:
            return 0.0
        m = self.mean
        var = sum((v - m) ** 2 for v in self.values) / (len(self.values) - 1)
        return math.sqrt(var)

    @property
    def min_val(self) -> float:
        return min(self.values) if self.values else 0.0

    @property
    def max_val(self) -> float:
        return max(self.values) if self.values else 0.0

    @property
    def best_experiment(self) -> str:
        if not self.values:
            return ""
        # For loss, lower is better; for all others, higher is better
        if "loss" in self.name:
            idx = self.values.index(min(self.values))
        else:
            idx = self.values.index(max(self.values))
        return self.experiment_ids[idx]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "mean": self.mean,
            "std": self.std,
            "min": self.min_val,
            "max": self.max_val,
            "best_experiment": self.best_experiment,
            "values": dict(zip(self.experiment_ids, self.values)),
        }


# ══════════════════════════════════════════════════════════════════════
# Expert Performance Profile
# ══════════════════════════════════════════════════════════════════════

@dataclass
class ExpertPerformanceProfile:
    """Performance profile for a single expert type."""
    expert_type: str
    # When this expert is the ONLY one active (single-expert ablation)
    solo_metrics: Dict[str, float] = field(default_factory=dict)
    # Baseline metric (all experts active)
    baseline_metric: float = 0.0
    # When this expert is removed (LOO ablation)
    loo_metrics: Dict[str, float] = field(default_factory=dict)
    # Performance drop = baseline - LOO
    importance_score: float = 0.0
    # Solo contribution = solo metric value
    solo_score: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "expert_type": self.expert_type,
            "solo_metrics": self.solo_metrics,
            "baseline_metric": self.baseline_metric,
            "loo_metrics": self.loo_metrics,
            "importance_score": self.importance_score,
            "solo_score": self.solo_score,
        }


# ══════════════════════════════════════════════════════════════════════
# Ablation Evaluator
# ══════════════════════════════════════════════════════════════════════

class AblationEvaluator:
    """Evaluates and aggregates results from all ablation experiments.

    Computes:
    - Metric comparison tables (experiments × metrics)
    - Expert importance rankings (from LOO experiments)
    - Expert solo performance rankings
    - Router comparison tables
    - Statistical significance indicators
    """

    def __init__(
        self,
        primary_metric: str = "vqa_accuracy",
        model_type: str = "generative",
    ):
        self.primary_metric = primary_metric
        self.model_type = model_type
        self.results: Dict[str, Dict[str, Any]] = {}  # experiment_id → result dict

    @property
    def relevant_metrics(self) -> List[str]:
        """Return metrics appropriate for the current model_type."""
        return get_metrics_for_model_type(self.model_type)

    def add_result(self, result_dict: Dict[str, Any]) -> None:
        """Add a single experiment result."""
        exp_id = result_dict.get("experiment_id", "unknown")
        self.results[exp_id] = result_dict

    def add_results(self, results: List[Dict[str, Any]]) -> None:
        """Add multiple experiment results."""
        for r in results:
            self.add_result(r)

    # ── Core Tables ─────────────────────────────────────────────────

    def build_metric_table(
        self, metric_names: Optional[List[str]] = None
    ) -> Dict[str, Dict[str, float]]:
        """Build experiment × metric table.

        Uses model-type-aware metric list when *metric_names* is ``None``.

        Returns:
            Dict[experiment_id, Dict[metric_name, value]]
        """
        if metric_names is None:
            metric_names = self.relevant_metrics

        table: Dict[str, Dict[str, float]] = {}
        for exp_id, result in self.results.items():
            metrics = result.get("metrics", {})
            table[exp_id] = {m: metrics.get(m, 0.0) for m in metric_names}
        return table

    def build_moe_metric_table(self) -> Dict[str, Dict[str, Any]]:
        """Build experiment × MOE metric table."""
        table: Dict[str, Dict[str, Any]] = {}
        for exp_id, result in self.results.items():
            moe_m = result.get("moe_metrics", {})
            table[exp_id] = {m: moe_m.get(m, None) for m in MOE_METRICS}
        return table

    # ── Summaries ─────────────────────────────────────────────────

    def compute_metric_summaries(
        self, metric_names: Optional[List[str]] = None
    ) -> Dict[str, MetricSummary]:
        """Compute summary statistics for each metric across all experiments."""
        if metric_names is None:
            metric_names = self.relevant_metrics

        summaries: Dict[str, MetricSummary] = {}
        for m_name in metric_names:
            summary = MetricSummary(name=m_name)
            for exp_id, result in self.results.items():
                val = result.get("metrics", {}).get(m_name)
                if val is not None:
                    summary.values.append(val)
                    summary.experiment_ids.append(exp_id)
            summaries[m_name] = summary
        return summaries

    # ── Expert Importance ──────────────────────────────────────────

    def compute_expert_importance(self) -> List[ExpertPerformanceProfile]:
        """Compute expert importance from LOO and single-expert experiments.

        Returns:
            List of ExpertPerformanceProfile sorted by importance_score desc.
        """
        # Find baseline (full)
        baseline_val = 0.0
        baseline_metrics: Dict[str, float] = {}
        for exp_id, result in self.results.items():
            ec = result.get("experiment_config", {}).get("expert_config", {})
            if ec.get("mode") == "full":
                baseline_metrics = result.get("metrics", {})
                baseline_val = baseline_metrics.get(self.primary_metric, 0.0)
                break

        expert_types = ["vision", "text", "multimodal", "specialized"]
        profiles: List[ExpertPerformanceProfile] = []

        for et in expert_types:
            profile = ExpertPerformanceProfile(
                expert_type=et,
                baseline_metric=baseline_val,
            )

            # Find LOO experiment for this expert type
            for exp_id, result in self.results.items():
                ec = result.get("experiment_config", {}).get("expert_config", {})
                if ec.get("mode") == "leave_one_out":
                    disabled = ec.get("disabled_experts", [])
                    if et in disabled and len(disabled) == 1:
                        profile.loo_metrics = result.get("metrics", {})
                        loo_val = profile.loo_metrics.get(self.primary_metric, 0.0)
                        profile.importance_score = baseline_val - loo_val
                        break

            # Find single-expert experiment
            for exp_id, result in self.results.items():
                ec = result.get("experiment_config", {}).get("expert_config", {})
                if ec.get("mode") == "single_expert":
                    active = ec.get("active_experts", [])
                    if active == [et]:
                        profile.solo_metrics = result.get("metrics", {})
                        profile.solo_score = profile.solo_metrics.get(
                            self.primary_metric, 0.0
                        )
                        break

            profiles.append(profile)

        # Sort by importance
        profiles.sort(key=lambda p: p.importance_score, reverse=True)
        return profiles

    # ── Router Comparison ──────────────────────────────────────────

    def compute_router_comparison(self) -> Dict[str, Dict[str, float]]:
        """Compare router types on the full-expert baseline.

        Returns:
            Dict[router_description, Dict[metric_name, value]]
        """
        table: Dict[str, Dict[str, float]] = {}
        for exp_id, result in self.results.items():
            ec = result.get("experiment_config", {}).get("expert_config", {})
            rc = result.get("experiment_config", {}).get("router_config", {})
            if ec.get("mode") == "full":
                desc = rc.get("description", exp_id)
                table[desc] = result.get("metrics", {})
        return table

    # ── Rankings ──────────────────────────────────────────────────

    def rank_experiments(
        self, metric: Optional[str] = None, ascending: bool = False
    ) -> List[Tuple[str, float]]:
        """Rank experiments by a metric.

        Returns:
            Sorted list of (experiment_id, metric_value).
        """
        metric = metric or self.primary_metric
        pairs = []
        for exp_id, result in self.results.items():
            val = result.get("metrics", {}).get(metric, 0.0)
            pairs.append((exp_id, val))
        pairs.sort(key=lambda x: x[1], reverse=not ascending)
        return pairs

    # ── Delta Analysis ────────────────────────────────────────────

    def compute_deltas_from_baseline(
        self, metric: Optional[str] = None
    ) -> Dict[str, float]:
        """Compute performance delta of each experiment from baseline.

        Returns:
            Dict[experiment_id, delta] where delta = experiment - baseline.
        """
        metric = metric or self.primary_metric

        # Find baseline
        baseline_val = 0.0
        for exp_id, result in self.results.items():
            ec = result.get("experiment_config", {}).get("expert_config", {})
            if ec.get("mode") == "full":
                baseline_val = result.get("metrics", {}).get(metric, 0.0)
                break

        deltas: Dict[str, float] = {}
        for exp_id, result in self.results.items():
            val = result.get("metrics", {}).get(metric, 0.0)
            deltas[exp_id] = val - baseline_val
        return deltas

    # ── Serialization ─────────────────────────────────────────────

    def to_dict(self) -> Dict[str, Any]:
        """Export all evaluation data to a dictionary."""
        profiles = self.compute_expert_importance()
        summaries = self.compute_metric_summaries()
        return {
            "primary_metric": self.primary_metric,
            "num_experiments": len(self.results),
            "ranking": self.rank_experiments(),
            "deltas": self.compute_deltas_from_baseline(),
            "expert_importance": [p.to_dict() for p in profiles],
            "metric_summaries": {k: v.to_dict() for k, v in summaries.items()},
            "metric_table": self.build_metric_table(),
            "moe_metric_table": self.build_moe_metric_table(),
            "router_comparison": self.compute_router_comparison(),
        }

    def save_json(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2, default=str)
