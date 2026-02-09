"""
Ablation Study Pipeline for Vietnamese VQA
=============================================

Provides a systematic, automated ablation framework with MOE expert analysis
as the primary focus. Supports:

- Expert-level ablation (single-expert-only, leave-one-out, subset combinations)
- Router-level ablation (TopK, NoisyTopK, Soft, ExpertChoice)
- Capacity & routing dynamics analysis
- Automated metric collection, analysis, and research-grade reporting

Usage:
    python -m src.ablation.ablation_runner --config configs/ablation_config.yaml
"""

from src.ablation.ablation_config import (
    AblationConfig,
    ExpertAblationConfig,
    RouterAblationConfig,
    ExperimentConfig,
    AblationSearchSpace,
)
from src.ablation.ablation_runner import AblationRunner
from src.ablation.ablation_trainer import AblationTrainer
from src.ablation.ablation_evaluator import AblationEvaluator
from src.ablation.ablation_analyzer import AblationAnalyzer
from src.ablation.ablation_reporter import AblationReporter

__all__ = [
    "AblationConfig",
    "ExpertAblationConfig",
    "RouterAblationConfig",
    "ExperimentConfig",
    "AblationSearchSpace",
    "AblationRunner",
    "AblationTrainer",
    "AblationEvaluator",
    "AblationAnalyzer",
    "AblationReporter",
]
