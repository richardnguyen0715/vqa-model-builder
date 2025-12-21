"""
Evaluator Configuration Module
Provides configuration for VQA model evaluation.
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from enum import Enum


class EvaluationMode(Enum):
    """Evaluation mode enumeration."""
    STANDARD = "standard"
    DETAILED = "detailed"
    ANALYSIS = "analysis"


@dataclass
class MetricConfig:
    """
    Metric configuration.
    
    Attributes:
        compute_accuracy: Whether to compute accuracy
        compute_wups: Whether to compute WUPS score
        compute_bleu: Whether to compute BLEU score
        compute_f1: Whether to compute F1 score
        compute_per_type: Whether to compute per-type metrics
        top_k_values: K values for top-k accuracy
        soft_accuracy: Whether to use soft accuracy scoring
    """
    compute_accuracy: bool = True
    compute_wups: bool = True
    compute_bleu: bool = False
    compute_f1: bool = True
    compute_per_type: bool = True
    top_k_values: List[int] = field(default_factory=lambda: [1, 3, 5])
    soft_accuracy: bool = True


@dataclass
class AnalysisConfig:
    """
    Analysis configuration for detailed evaluation.
    
    Attributes:
        analyze_question_types: Analyze performance by question type
        analyze_answer_types: Analyze performance by answer type
        analyze_difficulty: Analyze by question difficulty
        analyze_length: Analyze by question/answer length
        save_predictions: Whether to save all predictions
        save_errors: Whether to save error analysis
        num_error_examples: Number of error examples to save
    """
    analyze_question_types: bool = True
    analyze_answer_types: bool = True
    analyze_difficulty: bool = False
    analyze_length: bool = True
    save_predictions: bool = True
    save_errors: bool = True
    num_error_examples: int = 100


@dataclass
class EvaluationConfig:
    """
    Complete evaluation configuration.
    
    Attributes:
        mode: Evaluation mode
        batch_size: Evaluation batch size
        device: Device for evaluation
        use_fp16: Whether to use FP16
        num_workers: Data loading workers
        metrics: Metric configuration
        analysis: Analysis configuration
        output_dir: Directory for evaluation outputs
    """
    mode: EvaluationMode = EvaluationMode.DETAILED
    batch_size: int = 64
    device: str = "auto"
    use_fp16: bool = True
    num_workers: int = 4
    metrics: MetricConfig = field(default_factory=MetricConfig)
    analysis: AnalysisConfig = field(default_factory=AnalysisConfig)
    output_dir: str = "evaluation_results"


def get_default_evaluation_config() -> EvaluationConfig:
    """
    Get default evaluation configuration.
    
    Returns:
        Default EvaluationConfig instance
    """
    return EvaluationConfig(
        mode=EvaluationMode.DETAILED,
        batch_size=64,
        use_fp16=True,
        metrics=MetricConfig(
            compute_accuracy=True,
            compute_wups=True,
            compute_per_type=True,
            soft_accuracy=True
        ),
        analysis=AnalysisConfig(
            save_predictions=True,
            save_errors=True
        )
    )
