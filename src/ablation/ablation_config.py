"""
Ablation Configuration Module
==============================

Defines the complete search space for ablation studies, including:
- Expert-level ablation (single, LOO, subset combinations)
- Router-level ablation (type, hyperparameters)
- Capacity & routing dynamics
- Experiment matrix generation

All configs are serializable to/from YAML/JSON for reproducibility.
"""

import itertools
import json
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml


# ══════════════════════════════════════════════════════════════════════
# Enums
# ══════════════════════════════════════════════════════════════════════

class ExpertType(str, Enum):
    """Canonical expert type names matching VQAMOELayer composition."""
    VISION = "vision"
    TEXT = "text"
    MULTIMODAL = "multimodal"
    SPECIALIZED = "specialized"


class AblationMode(str, Enum):
    """Type of expert ablation experiment."""
    FULL = "full"                    # All experts enabled (baseline)
    SINGLE_EXPERT = "single_expert"  # Only one expert type enabled
    LEAVE_ONE_OUT = "leave_one_out"  # Remove one expert type
    SUBSET = "subset"                # Arbitrary subset of expert types
    NO_MOE = "no_moe"               # Entirely disable MOE


class RouterType(str, Enum):
    """Router types available for ablation."""
    TOPK = "topk"
    NOISY_TOPK = "noisy_topk"
    SOFT = "soft"
    EXPERT_CHOICE = "expert_choice"


# ══════════════════════════════════════════════════════════════════════
# Expert Ablation Config
# ══════════════════════════════════════════════════════════════════════

@dataclass
class ExpertAblationConfig:
    """Configuration for which experts are active in a given experiment.

    Attributes:
        mode: Type of ablation (full, single_expert, leave_one_out, subset, no_moe)
        active_experts: List of expert types that are *enabled*.
        disabled_experts: List of expert types that are *disabled* (derived).
        num_experts_per_type: How many instances per active expert type.
        description: Human-readable experiment label.
    """
    mode: str = "full"
    active_experts: List[str] = field(default_factory=lambda: [
        "vision", "text", "multimodal", "specialized"
    ])
    disabled_experts: List[str] = field(default_factory=list)
    num_experts_per_type: Dict[str, int] = field(default_factory=lambda: {
        "vision": 2, "text": 2, "multimodal": 2, "specialized": 2
    })
    description: str = ""

    def __post_init__(self):
        all_types = {"vision", "text", "multimodal", "specialized"}
        active_set = set(self.active_experts)
        self.disabled_experts = sorted(all_types - active_set)
        if not self.description:
            if self.mode == "full":
                self.description = "Baseline (all experts)"
            elif self.mode == "no_moe":
                self.description = "No MOE"
            elif self.mode == "single_expert":
                self.description = f"Single: {'+'.join(self.active_experts)}"
            elif self.mode == "leave_one_out":
                self.description = f"LOO: -{'+'.join(self.disabled_experts)}"
            elif self.mode == "subset":
                self.description = f"Subset: {'+'.join(self.active_experts)}"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "mode": self.mode,
            "active_experts": self.active_experts,
            "disabled_experts": self.disabled_experts,
            "num_experts_per_type": self.num_experts_per_type,
            "description": self.description,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ExpertAblationConfig":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


# ══════════════════════════════════════════════════════════════════════
# Router Ablation Config
# ══════════════════════════════════════════════════════════════════════

@dataclass
class RouterAblationConfig:
    """Configuration for router in a given experiment.

    Attributes:
        router_type: Which router to use.
        top_k: Number of experts per token.
        noise_enabled: Whether to add noise (NoisyTopK).
        load_balance_weight: Weight for auxiliary load-balancing loss.
        capacity_factor: Expert capacity factor (sparse MOE).
        description: Human-readable label.
    """
    router_type: str = "noisy_topk"
    top_k: int = 2
    noise_enabled: bool = True
    load_balance_weight: float = 0.01
    capacity_factor: float = 1.25
    description: str = ""

    def __post_init__(self):
        if not self.description:
            parts = [f"Router={self.router_type}", f"K={self.top_k}"]
            if self.router_type in ("noisy_topk",):
                parts.append(f"noise={'on' if self.noise_enabled else 'off'}")
            parts.append(f"lb={self.load_balance_weight}")
            self.description = " | ".join(parts)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "router_type": self.router_type,
            "top_k": self.top_k,
            "noise_enabled": self.noise_enabled,
            "load_balance_weight": self.load_balance_weight,
            "capacity_factor": self.capacity_factor,
            "description": self.description,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "RouterAblationConfig":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


# ══════════════════════════════════════════════════════════════════════
# Single Experiment Config
# ══════════════════════════════════════════════════════════════════════

@dataclass
class ExperimentConfig:
    """Complete configuration for one ablation experiment.

    Combines expert ablation + router ablation + training overrides.
    """
    experiment_id: str = ""
    expert_config: ExpertAblationConfig = field(default_factory=ExpertAblationConfig)
    router_config: RouterAblationConfig = field(default_factory=RouterAblationConfig)

    # Training overrides (inherit from global if None)
    num_epochs: Optional[int] = None
    learning_rate: Optional[float] = None
    batch_size: Optional[int] = None
    seed: int = 42

    # Meta
    tags: List[str] = field(default_factory=list)
    priority: int = 0  # Higher = run first

    def __post_init__(self):
        if not self.experiment_id:
            expert_str = self.expert_config.mode
            if self.expert_config.mode in ("single_expert", "subset"):
                expert_str += "_" + "_".join(sorted(self.expert_config.active_experts))
            elif self.expert_config.mode == "leave_one_out":
                expert_str += "_no_" + "_".join(sorted(self.expert_config.disabled_experts))
            router_str = f"{self.router_config.router_type}_k{self.router_config.top_k}"
            self.experiment_id = f"{expert_str}__{router_str}"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "experiment_id": self.experiment_id,
            "expert_config": self.expert_config.to_dict(),
            "router_config": self.router_config.to_dict(),
            "num_epochs": self.num_epochs,
            "learning_rate": self.learning_rate,
            "batch_size": self.batch_size,
            "seed": self.seed,
            "tags": self.tags,
            "priority": self.priority,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ExperimentConfig":
        return cls(
            experiment_id=d.get("experiment_id", ""),
            expert_config=ExpertAblationConfig.from_dict(d.get("expert_config", {})),
            router_config=RouterAblationConfig.from_dict(d.get("router_config", {})),
            num_epochs=d.get("num_epochs"),
            learning_rate=d.get("learning_rate"),
            batch_size=d.get("batch_size"),
            seed=d.get("seed", 42),
            tags=d.get("tags", []),
            priority=d.get("priority", 0),
        )


# ══════════════════════════════════════════════════════════════════════
# Ablation Search Space
# ══════════════════════════════════════════════════════════════════════

@dataclass
class AblationSearchSpace:
    """Defines the search space from which experiments are generated.

    Controls combinatorial explosion via max_subset_size and explicit includes/excludes.
    """
    # Expert ablation space
    expert_types: List[str] = field(default_factory=lambda: [
        "vision", "text", "multimodal", "specialized"
    ])
    run_single_expert: bool = True
    run_leave_one_out: bool = True
    run_subsets: bool = True
    max_subset_size: int = 3        # Max combo size to enumerate
    min_subset_size: int = 2        # Min combo size
    include_no_moe_baseline: bool = True
    include_full_baseline: bool = True

    # Router ablation space
    router_types: List[str] = field(default_factory=lambda: [
        "topk", "noisy_topk", "soft", "expert_choice"
    ])
    top_k_values: List[int] = field(default_factory=lambda: [1, 2, 4])
    load_balance_weights: List[float] = field(default_factory=lambda: [0.0, 0.01, 0.1])
    capacity_factors: List[float] = field(default_factory=lambda: [1.0, 1.25, 1.5])

    # Cross-product control
    # If True, cross every expert config with every router config.
    # If False, only test routers on the full-expert baseline.
    cross_expert_router: bool = False

    # Num experts per type (constant across experiments unless overridden)
    num_experts_per_type: Dict[str, int] = field(default_factory=lambda: {
        "vision": 2, "text": 2, "multimodal": 2, "specialized": 2
    })

    def to_dict(self) -> Dict[str, Any]:
        return {
            "expert_types": self.expert_types,
            "run_single_expert": self.run_single_expert,
            "run_leave_one_out": self.run_leave_one_out,
            "run_subsets": self.run_subsets,
            "max_subset_size": self.max_subset_size,
            "min_subset_size": self.min_subset_size,
            "include_no_moe_baseline": self.include_no_moe_baseline,
            "include_full_baseline": self.include_full_baseline,
            "router_types": self.router_types,
            "top_k_values": self.top_k_values,
            "load_balance_weights": self.load_balance_weights,
            "capacity_factors": self.capacity_factors,
            "cross_expert_router": self.cross_expert_router,
            "num_experts_per_type": self.num_experts_per_type,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "AblationSearchSpace":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    def generate_expert_configs(self) -> List[ExpertAblationConfig]:
        """Generate all expert ablation configurations from the search space."""
        configs: List[ExpertAblationConfig] = []

        # 1. Full baseline
        if self.include_full_baseline:
            configs.append(ExpertAblationConfig(
                mode="full",
                active_experts=list(self.expert_types),
                num_experts_per_type=dict(self.num_experts_per_type),
            ))

        # 2. No-MOE baseline
        if self.include_no_moe_baseline:
            configs.append(ExpertAblationConfig(
                mode="no_moe",
                active_experts=[],
                num_experts_per_type={k: 0 for k in self.expert_types},
            ))

        # 3. Single-expert-only
        if self.run_single_expert:
            for et in self.expert_types:
                nept = {k: (self.num_experts_per_type.get(k, 2) if k == et else 0)
                        for k in self.expert_types}
                configs.append(ExpertAblationConfig(
                    mode="single_expert",
                    active_experts=[et],
                    num_experts_per_type=nept,
                ))

        # 4. Leave-one-out
        if self.run_leave_one_out:
            for et in self.expert_types:
                active = [t for t in self.expert_types if t != et]
                nept = {k: (self.num_experts_per_type.get(k, 2) if k != et else 0)
                        for k in self.expert_types}
                configs.append(ExpertAblationConfig(
                    mode="leave_one_out",
                    active_experts=active,
                    num_experts_per_type=nept,
                ))

        # 5. Subset combinations
        if self.run_subsets:
            for size in range(self.min_subset_size, self.max_subset_size + 1):
                # Skip if size == len(expert_types) (that's full) or size == 1 (single)
                if size >= len(self.expert_types) or size <= 1:
                    continue
                for combo in itertools.combinations(self.expert_types, size):
                    combo_list = sorted(combo)
                    nept = {k: (self.num_experts_per_type.get(k, 2) if k in combo_list else 0)
                            for k in self.expert_types}
                    configs.append(ExpertAblationConfig(
                        mode="subset",
                        active_experts=combo_list,
                        num_experts_per_type=nept,
                    ))

        return configs

    def generate_router_configs(self) -> List[RouterAblationConfig]:
        """Generate all router ablation configurations from the search space."""
        configs: List[RouterAblationConfig] = []

        for rt in self.router_types:
            for k_val in self.top_k_values:
                for lb_w in self.load_balance_weights:
                    # SoftRouter ignores top_k (routes to all); skip redundant combos
                    if rt == "soft" and k_val != self.top_k_values[0]:
                        continue
                    # ExpertChoice doesn't use top_k directly the same way
                    if rt == "expert_choice" and k_val != self.top_k_values[0]:
                        continue
                    noise_enabled = (rt == "noisy_topk")
                    configs.append(RouterAblationConfig(
                        router_type=rt,
                        top_k=k_val,
                        noise_enabled=noise_enabled,
                        load_balance_weight=lb_w,
                        capacity_factor=self.capacity_factors[0],
                    ))

        return configs


# ══════════════════════════════════════════════════════════════════════
# Main Ablation Config
# ══════════════════════════════════════════════════════════════════════

@dataclass
class AblationConfig:
    """Top-level ablation study configuration.

    Attributes:
        name: Study name (used in output directory).
        search_space: Defines which experiments to generate.
        training: Default training hyperparameters.
        data: Data pipeline overrides.
        output_dir: Root directory for all ablation outputs.
        resume: Whether to skip completed experiments on re-run.
        max_parallel: Max parallel experiments (1 = sequential).
        explicit_experiments: Manually defined experiments (appended to generated).
    """
    name: str = "moe_ablation_study"
    search_space: AblationSearchSpace = field(default_factory=AblationSearchSpace)

    # Default training settings (individual experiments can override)
    num_epochs: int = 10
    learning_rate: float = 2e-5
    batch_size: int = 32
    eval_batch_size: int = 64
    seed: int = 42
    use_amp: bool = True
    gradient_accumulation_steps: int = 1
    early_stopping: bool = True
    patience: int = 5

    # Data settings
    images_dir: str = "data/raw/images"
    text_file: str = "data/raw/texts/evaluate_60k_data_balanced_preprocessed.csv"
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    test_ratio: float = 0.1

    # Model base settings
    visual_backbone: str = "vit"
    visual_model_name: str = "openai/clip-vit-base-patch32"
    text_encoder_type: str = "phobert"
    text_model_name: str = "vinai/phobert-base"
    fusion_type: str = "cross_attention"
    embed_dim: int = 768
    moe_hidden_dim: int = 2048

    # Output
    output_dir: str = "outputs/ablation"
    checkpoint_dir: str = "checkpoints/ablation"
    log_dir: str = "logs/ablation"

    # Execution
    resume: bool = True
    max_parallel: int = 1

    # Explicit experiments (override generated)
    explicit_experiments: List[Dict[str, Any]] = field(default_factory=list)

    def generate_experiment_matrix(self) -> List[ExperimentConfig]:
        """Generate the complete experiment matrix from search space.

        Returns:
            Sorted list of ExperimentConfig (by priority desc, then id).
        """
        expert_configs = self.search_space.generate_expert_configs()
        router_configs = self.search_space.generate_router_configs()

        experiments: List[ExperimentConfig] = []
        seen_ids = set()

        if self.search_space.cross_expert_router:
            # Cross-product: every expert config × every router config
            for ec in expert_configs:
                for rc in router_configs:
                    # Skip router ablation for no_moe
                    if ec.mode == "no_moe":
                        exp = ExperimentConfig(
                            expert_config=ec,
                            router_config=RouterAblationConfig(router_type="topk", top_k=2),
                            num_epochs=self.num_epochs,
                            learning_rate=self.learning_rate,
                            batch_size=self.batch_size,
                            seed=self.seed,
                            tags=["no_moe"],
                            priority=10,
                        )
                        if exp.experiment_id not in seen_ids:
                            seen_ids.add(exp.experiment_id)
                            experiments.append(exp)
                        continue
                    exp = ExperimentConfig(
                        expert_config=ec,
                        router_config=rc,
                        num_epochs=self.num_epochs,
                        learning_rate=self.learning_rate,
                        batch_size=self.batch_size,
                        seed=self.seed,
                        tags=self._auto_tags(ec, rc),
                        priority=self._auto_priority(ec),
                    )
                    if exp.experiment_id not in seen_ids:
                        seen_ids.add(exp.experiment_id)
                        experiments.append(exp)
        else:
            # Expert ablations use default router; router ablations use full experts
            default_rc = RouterAblationConfig(router_type="noisy_topk", top_k=2)
            for ec in expert_configs:
                exp = ExperimentConfig(
                    expert_config=ec,
                    router_config=default_rc,
                    num_epochs=self.num_epochs,
                    learning_rate=self.learning_rate,
                    batch_size=self.batch_size,
                    seed=self.seed,
                    tags=self._auto_tags(ec, default_rc),
                    priority=self._auto_priority(ec),
                )
                if exp.experiment_id not in seen_ids:
                    seen_ids.add(exp.experiment_id)
                    experiments.append(exp)

            # Router ablations on full baseline only
            full_ec = ExpertAblationConfig(
                mode="full",
                active_experts=list(self.search_space.expert_types),
                num_experts_per_type=dict(self.search_space.num_experts_per_type),
            )
            for rc in router_configs:
                exp = ExperimentConfig(
                    expert_config=full_ec,
                    router_config=rc,
                    num_epochs=self.num_epochs,
                    learning_rate=self.learning_rate,
                    batch_size=self.batch_size,
                    seed=self.seed,
                    tags=["router_ablation"],
                    priority=5,
                )
                if exp.experiment_id not in seen_ids:
                    seen_ids.add(exp.experiment_id)
                    experiments.append(exp)

        # Add explicit experiments
        for exp_dict in self.explicit_experiments:
            exp = ExperimentConfig.from_dict(exp_dict)
            if exp.experiment_id not in seen_ids:
                seen_ids.add(exp.experiment_id)
                experiments.append(exp)

        # Sort: highest priority first, then alphabetical
        experiments.sort(key=lambda e: (-e.priority, e.experiment_id))
        return experiments

    @staticmethod
    def _auto_tags(ec: ExpertAblationConfig, rc: RouterAblationConfig) -> List[str]:
        tags = [ec.mode]
        if ec.mode != "no_moe":
            tags.append(f"router_{rc.router_type}")
        return tags

    @staticmethod
    def _auto_priority(ec: ExpertAblationConfig) -> int:
        priorities = {
            "full": 10,
            "no_moe": 9,
            "leave_one_out": 8,
            "single_expert": 7,
            "subset": 5,
        }
        return priorities.get(ec.mode, 0)

    # ── Serialization ─────────────────────────────────────────────────

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "search_space": self.search_space.to_dict(),
            "num_epochs": self.num_epochs,
            "learning_rate": self.learning_rate,
            "batch_size": self.batch_size,
            "eval_batch_size": self.eval_batch_size,
            "seed": self.seed,
            "use_amp": self.use_amp,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "early_stopping": self.early_stopping,
            "patience": self.patience,
            "images_dir": self.images_dir,
            "text_file": self.text_file,
            "train_ratio": self.train_ratio,
            "val_ratio": self.val_ratio,
            "test_ratio": self.test_ratio,
            "visual_backbone": self.visual_backbone,
            "visual_model_name": self.visual_model_name,
            "text_encoder_type": self.text_encoder_type,
            "text_model_name": self.text_model_name,
            "fusion_type": self.fusion_type,
            "embed_dim": self.embed_dim,
            "moe_hidden_dim": self.moe_hidden_dim,
            "output_dir": self.output_dir,
            "checkpoint_dir": self.checkpoint_dir,
            "log_dir": self.log_dir,
            "resume": self.resume,
            "max_parallel": self.max_parallel,
            "explicit_experiments": self.explicit_experiments,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "AblationConfig":
        ss = d.pop("search_space", {})
        obj = cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})
        if ss:
            obj.search_space = AblationSearchSpace.from_dict(ss)
        return obj

    @classmethod
    def from_yaml(cls, path: str) -> "AblationConfig":
        with open(path, "r") as f:
            d = yaml.safe_load(f)
        # Support nested under 'ablation' key
        if "ablation" in d:
            d = d["ablation"]
        return cls.from_dict(d)

    def save_yaml(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, sort_keys=False)

    def save_json(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
