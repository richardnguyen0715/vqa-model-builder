"""
VQA Evaluator Module
Provides comprehensive evaluation for Vietnamese VQA models.
Includes resource monitoring during evaluation.
"""

import os
import json
import logging
from typing import Optional, Dict, Any, List, Tuple, Union
from dataclasses import dataclass, field
from collections import defaultdict
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast

from .evaluator_config import (
    EvaluationConfig,
    EvaluationMode,
    get_default_evaluation_config
)

# Resource management imports
try:
    from src.resource_management import (
        ResourceMonitor,
        ReportManager,
        load_resource_config,
    )
    RESOURCE_MANAGEMENT_AVAILABLE = True
except ImportError:
    RESOURCE_MANAGEMENT_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class EvaluationResult:
    """
    Evaluation result container.
    
    Attributes:
        overall_metrics: Overall evaluation metrics
        per_type_metrics: Metrics broken down by question type
        per_answer_metrics: Metrics by answer type
        confusion_data: Confusion matrix data
        predictions: All predictions if saved
        errors: Error examples
        metadata: Additional metadata
        resource_metrics: Resource usage metrics during evaluation
    """
    overall_metrics: Dict[str, float] = field(default_factory=dict)
    per_type_metrics: Dict[str, Dict[str, float]] = field(default_factory=dict)
    per_answer_metrics: Dict[str, Dict[str, float]] = field(default_factory=dict)
    confusion_data: Dict[str, Any] = field(default_factory=dict)
    predictions: List[Dict[str, Any]] = field(default_factory=list)
    errors: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    resource_metrics: Dict[str, Any] = field(default_factory=dict)


class VQAEvaluator:
    """
    Vietnamese VQA model evaluator.
    
    Provides comprehensive evaluation including:
    - Multiple metrics (accuracy, WUPS, BLEU, F1)
    - Per-type analysis
    - Error analysis
    - Visualization support
    - Resource monitoring during evaluation
    
    Attributes:
        config: Evaluation configuration
        device: Evaluation device
        metrics: Dictionary of metric calculators
        resource_monitor: Optional resource monitor for tracking usage
    """
    
    def __init__(
        self,
        config: Optional[EvaluationConfig] = None,
        answer_vocabulary: Optional[List[str]] = None,
        question_type_mapping: Optional[Dict[str, str]] = None,
        enable_resource_monitoring: bool = True
    ):
        """
        Initialize evaluator.
        
        Args:
            config: Evaluation configuration
            answer_vocabulary: List of answer strings
            question_type_mapping: Mapping of question to type
            enable_resource_monitoring: Enable resource monitoring during evaluation
        """
        self.config = config or get_default_evaluation_config()
        self.device = self._get_device()
        
        self.answer_vocabulary = answer_vocabulary or []
        self.question_type_mapping = question_type_mapping or {}
        
        # Setup metrics
        self.metrics = self._setup_metrics()
        
        # Setup resource monitoring
        self.resource_monitor = self._setup_resource_monitor(enable_resource_monitoring)
        
        logger.info(f"Initialized evaluator on {self.device}")
    
    def _get_device(self) -> torch.device:
        """Get evaluation device."""
        device_str = self.config.device
        
        if device_str == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return torch.device("mps")
            return torch.device("cpu")
        
        return torch.device(device_str)
    
    def _setup_metrics(self) -> Dict[str, Any]:
        """Setup metric calculators."""
        metrics = {}
        metric_config = self.config.metrics
        
        if metric_config.compute_accuracy:
            metrics["accuracy"] = AccuracyCalculator(
                soft=metric_config.soft_accuracy
            )
        
        if metric_config.compute_wups:
            metrics["wups"] = WUPSCalculator()
        
        if metric_config.compute_f1:
            metrics["f1"] = F1Calculator()
        
        if metric_config.compute_bleu:
            metrics["bleu"] = BLEUCalculator()
        
        for k in metric_config.top_k_values:
            metrics[f"top_{k}_accuracy"] = TopKAccuracyCalculator(k=k)
        
        return metrics
    
    def _setup_resource_monitor(
        self, 
        enable: bool = True
    ) -> Optional["ResourceMonitor"]:
        """
        Setup resource monitor for tracking usage during evaluation.
        
        Args:
            enable: Whether to enable resource monitoring.
            
        Returns:
            ResourceMonitor instance or None if not available.
        """
        if not enable or not RESOURCE_MANAGEMENT_AVAILABLE:
            return None
        
        try:
            resource_config = load_resource_config()
            monitor = ResourceMonitor(resource_config)
            logger.info("Resource monitoring enabled for evaluation")
            return monitor
        except Exception as e:
            logger.warning(f"Failed to setup resource monitor: {e}")
            return None
    
    def _get_resource_snapshot(self) -> Optional[Dict[str, Any]]:
        """
        Get current resource usage snapshot.
        
        Returns:
            Dictionary with resource metrics or None if monitoring not available.
        """
        if self.resource_monitor is None:
            return None
        
        try:
            metrics = self.resource_monitor.get_latest_metrics()
            if metrics is not None:
                return metrics.to_dict()
        except Exception:
            pass
        return None
    
    def _id_to_answer(self, answer_id: int) -> str:
        """Convert answer ID to string."""
        if 0 <= answer_id < len(self.answer_vocabulary):
            return self.answer_vocabulary[answer_id]
        return f"answer_{answer_id}"
    
    def _get_question_type(self, question: str) -> str:
        """Get question type from question text."""
        # Check mapping first
        if question in self.question_type_mapping:
            return self.question_type_mapping[question]
        
        # Vietnamese question type detection
        question_lower = question.lower()
        
        if any(w in question_lower for w in ["bao nhiêu", "mấy", "số lượng"]):
            return "count"
        elif any(w in question_lower for w in ["màu gì", "màu sắc", "màu nào"]):
            return "color"
        elif any(w in question_lower for w in ["ở đâu", "vị trí", "chỗ nào"]):
            return "location"
        elif any(w in question_lower for w in ["có phải", "có không", "đúng không"]):
            return "yes_no"
        elif any(w in question_lower for w in ["ai", "người nào"]):
            return "person"
        elif any(w in question_lower for w in ["cái gì", "vật gì", "đồ gì"]):
            return "object"
        elif any(w in question_lower for w in ["khi nào", "lúc nào", "thời gian"]):
            return "time"
        elif any(w in question_lower for w in ["tại sao", "vì sao", "lý do"]):
            return "why"
        elif any(w in question_lower for w in ["như thế nào", "ra sao"]):
            return "how"
        else:
            return "other"
    
    def _to_device(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """Move batch to device."""
        result = {}
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                result[key] = value.to(self.device)
            elif isinstance(value, dict):
                result[key] = self._to_device(value)
            else:
                result[key] = value
        return result
    
    @torch.no_grad()
    def evaluate(
        self,
        model: nn.Module,
        dataloader: DataLoader,
        return_predictions: bool = False
    ) -> EvaluationResult:
        """
        Evaluate model on dataset.
        
        Tracks resource usage during evaluation if monitoring is enabled.
        
        Args:
            model: VQA model to evaluate
            dataloader: Evaluation data loader
            return_predictions: Whether to return all predictions
            
        Returns:
            EvaluationResult with all metrics and resource usage
        """
        model = model.to(self.device)
        model.eval()
        
        start_time = time.time()
        
        # Start resource monitoring for evaluation
        if self.resource_monitor is not None:
            self.resource_monitor.start()
            initial_resources = self._get_resource_snapshot()
        else:
            initial_resources = None
        
        all_predictions = []
        all_targets = []
        all_questions = []
        all_question_types = []
        all_probs = []
        
        logger.info("Starting evaluation...")
        
        try:
            for batch in dataloader:
                batch = self._to_device(batch)
                
                # Forward pass
                if self.config.use_fp16 and self.device.type == "cuda":
                    with torch.amp.autocast():
                        outputs = model(
                            image=batch.get("image", batch.get("images")),
                            input_ids=batch.get("input_ids"),
                            attention_mask=batch.get("attention_mask")
                        )
                else:
                    outputs = model(
                        image=batch.get("image", batch.get("images")),
                        input_ids=batch.get("input_ids"),
                        attention_mask=batch.get("attention_mask")
                    )
                
                logits = outputs.logits if hasattr(outputs, "logits") else outputs["logits"]
                probs = torch.softmax(logits, dim=-1)
                predictions = logits.argmax(dim=-1)
                
                targets = batch.get("labels", batch.get("answer_ids"))
                questions = batch.get("questions", [""] * targets.size(0))
                
                all_predictions.extend(predictions.cpu().tolist())
                all_targets.extend(targets.cpu().tolist())
                all_probs.extend(probs.cpu().tolist())
                
                if isinstance(questions, torch.Tensor):
                    questions = questions.tolist()
                all_questions.extend(questions)
                
                # Get question types
                for q in questions:
                    if isinstance(q, str):
                        all_question_types.append(self._get_question_type(q))
                    else:
                        all_question_types.append("unknown")
        finally:
            # Stop resource monitoring
            if self.resource_monitor is not None:
                self.resource_monitor.stop()
        
        eval_time = time.time() - start_time
        
        # Compute metrics
        result = self._compute_metrics(
            all_predictions,
            all_targets,
            all_probs,
            all_questions,
            all_question_types
        )
        
        result.metadata["eval_time"] = eval_time
        result.metadata["num_samples"] = len(all_predictions)
        
        # Add resource metrics
        final_resources = self._get_resource_snapshot()
        if initial_resources or final_resources:
            result.resource_metrics = {
                "initial": initial_resources,
                "final": final_resources,
            }
        
        # Save predictions if requested
        if return_predictions or self.config.analysis.save_predictions:
            result.predictions = self._format_predictions(
                all_predictions,
                all_targets,
                all_questions,
                all_probs
            )
        
        # Error analysis
        if self.config.analysis.save_errors:
            result.errors = self._analyze_errors(
                all_predictions,
                all_targets,
                all_questions,
                all_probs
            )
        
        return result
    
    def _compute_metrics(
        self,
        predictions: List[int],
        targets: List[int],
        probs: List[List[float]],
        questions: List[str],
        question_types: List[str]
    ) -> EvaluationResult:
        """Compute all evaluation metrics."""
        result = EvaluationResult()
        
        # Convert to answer strings
        pred_answers = [self._id_to_answer(p) for p in predictions]
        target_answers = [self._id_to_answer(t) for t in targets]
        
        # Overall metrics
        for name, metric in self.metrics.items():
            if "top_" in name:
                # Top-k needs probabilities
                value = metric.compute(probs, targets)
            else:
                value = metric.compute(pred_answers, target_answers)
            result.overall_metrics[name] = value
        
        # Per question type metrics
        if self.config.metrics.compute_per_type:
            type_results = defaultdict(lambda: {"preds": [], "targets": []})
            
            for pred, target, qtype in zip(pred_answers, target_answers, question_types):
                type_results[qtype]["preds"].append(pred)
                type_results[qtype]["targets"].append(target)
            
            for qtype, data in type_results.items():
                result.per_type_metrics[qtype] = {}
                for name, metric in self.metrics.items():
                    if "top_" not in name:  # Skip top-k for per-type
                        value = metric.compute(data["preds"], data["targets"])
                        result.per_type_metrics[qtype][name] = value
                result.per_type_metrics[qtype]["count"] = len(data["preds"])
        
        return result
    
    def _format_predictions(
        self,
        predictions: List[int],
        targets: List[int],
        questions: List[str],
        probs: List[List[float]]
    ) -> List[Dict[str, Any]]:
        """Format predictions for saving."""
        formatted = []
        
        for pred, target, question, prob in zip(predictions, targets, questions, probs):
            pred_answer = self._id_to_answer(pred)
            target_answer = self._id_to_answer(target)
            confidence = prob[pred] if pred < len(prob) else 0.0
            
            formatted.append({
                "question": question if isinstance(question, str) else "",
                "predicted_answer": pred_answer,
                "predicted_id": pred,
                "target_answer": target_answer,
                "target_id": target,
                "confidence": confidence,
                "correct": pred == target
            })
        
        return formatted
    
    def _analyze_errors(
        self,
        predictions: List[int],
        targets: List[int],
        questions: List[str],
        probs: List[List[float]]
    ) -> List[Dict[str, Any]]:
        """Analyze prediction errors."""
        errors = []
        
        for i, (pred, target, question, prob) in enumerate(zip(predictions, targets, questions, probs)):
            if pred != target:
                pred_answer = self._id_to_answer(pred)
                target_answer = self._id_to_answer(target)
                confidence = prob[pred] if pred < len(prob) else 0.0
                target_prob = prob[target] if target < len(prob) else 0.0
                
                errors.append({
                    "index": i,
                    "question": question if isinstance(question, str) else "",
                    "predicted_answer": pred_answer,
                    "target_answer": target_answer,
                    "confidence": confidence,
                    "target_confidence": target_prob,
                    "question_type": self._get_question_type(question) if isinstance(question, str) else "unknown"
                })
        
        # Sort by confidence (high confidence errors are more interesting)
        errors.sort(key=lambda x: x["confidence"], reverse=True)
        
        # Return top N errors
        return errors[:self.config.analysis.num_error_examples]
    
    def save_results(self, result: EvaluationResult, output_path: Optional[str] = None) -> None:
        """
        Save evaluation results to file.
        
        Includes resource metrics if monitoring was enabled.
        
        Args:
            result: Evaluation results
            output_path: Output file path
        """
        if output_path is None:
            os.makedirs(self.config.output_dir, exist_ok=True)
            output_path = os.path.join(self.config.output_dir, "evaluation_results.json")
        
        data = {
            "overall_metrics": result.overall_metrics,
            "per_type_metrics": result.per_type_metrics,
            "per_answer_metrics": result.per_answer_metrics,
            "metadata": result.metadata
        }
        
        # Add resource metrics if available
        if result.resource_metrics:
            data["resource_metrics"] = result.resource_metrics
        
        if result.predictions:
            data["num_predictions"] = len(result.predictions)
            # Save predictions separately if too large
            if len(result.predictions) > 1000:
                pred_path = output_path.replace(".json", "_predictions.json")
                with open(pred_path, "w", encoding="utf-8") as f:
                    json.dump(result.predictions, f, ensure_ascii=False, indent=2)
                data["predictions_file"] = pred_path
            else:
                data["predictions"] = result.predictions
        
        if result.errors:
            data["errors"] = result.errors
        
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Saved evaluation results to {output_path}")
    
    def print_summary(self, result: EvaluationResult) -> None:
        """Print evaluation summary."""
        print("\n" + "=" * 60)
        print("EVALUATION RESULTS")
        print("=" * 60)
        
        print("\nOverall Metrics:")
        print("-" * 40)
        for name, value in result.overall_metrics.items():
            print(f"  {name}: {value:.4f}")
        
        if result.per_type_metrics:
            print("\nPer Question Type:")
            print("-" * 40)
            for qtype, metrics in result.per_type_metrics.items():
                count = metrics.get("count", 0)
                acc = metrics.get("accuracy", 0)
                print(f"  {qtype}: acc={acc:.4f} (n={count})")
        
        print("\nMetadata:")
        print("-" * 40)
        for key, value in result.metadata.items():
            print(f"  {key}: {value}")
        
        print("=" * 60 + "\n")


# Metric calculator classes

class AccuracyCalculator:
    """Accuracy metric calculator."""
    
    def __init__(self, soft: bool = True):
        self.soft = soft
    
    def compute(self, predictions: List[str], targets: List[str]) -> float:
        if len(predictions) == 0:
            return 0.0
        
        if self.soft:
            # Soft accuracy: partial credit for similar answers
            correct = 0.0
            for pred, target in zip(predictions, targets):
                pred_lower = pred.lower().strip()
                target_lower = target.lower().strip()
                
                if pred_lower == target_lower:
                    correct += 1.0
                elif pred_lower in target_lower or target_lower in pred_lower:
                    correct += 0.5
            return correct / len(predictions)
        else:
            # Exact match
            correct = sum(1 for p, t in zip(predictions, targets) if p.lower().strip() == t.lower().strip())
            return correct / len(predictions)


class TopKAccuracyCalculator:
    """Top-K accuracy calculator."""
    
    def __init__(self, k: int = 5):
        self.k = k
    
    def compute(self, probs: List[List[float]], targets: List[int]) -> float:
        if len(probs) == 0:
            return 0.0
        
        correct = 0
        for prob, target in zip(probs, targets):
            top_k = sorted(range(len(prob)), key=lambda i: prob[i], reverse=True)[:self.k]
            if target in top_k:
                correct += 1
        
        return correct / len(probs)


class WUPSCalculator:
    """WUPS (Wu-Palmer Similarity) metric calculator."""
    
    def __init__(self, threshold: float = 0.9):
        self.threshold = threshold
    
    def compute(self, predictions: List[str], targets: List[str]) -> float:
        if len(predictions) == 0:
            return 0.0
        
        total_score = 0.0
        
        for pred, target in zip(predictions, targets):
            pred_lower = pred.lower().strip()
            target_lower = target.lower().strip()
            
            if pred_lower == target_lower:
                total_score += 1.0
            else:
                # Simple character-level similarity as fallback
                score = self._simple_similarity(pred_lower, target_lower)
                if score >= self.threshold:
                    total_score += score
        
        return total_score / len(predictions)
    
    def _simple_similarity(self, s1: str, s2: str) -> float:
        """Simple Jaccard similarity."""
        set1 = set(s1)
        set2 = set(s2)
        intersection = len(set1 & set2)
        union = len(set1 | set2)
        return intersection / union if union > 0 else 0.0


class F1Calculator:
    """F1 score calculator."""
    
    def compute(self, predictions: List[str], targets: List[str]) -> float:
        if len(predictions) == 0:
            return 0.0
        
        # Compute token-level F1
        total_f1 = 0.0
        
        for pred, target in zip(predictions, targets):
            pred_tokens = set(pred.lower().split())
            target_tokens = set(target.lower().split())
            
            if len(pred_tokens) == 0 and len(target_tokens) == 0:
                total_f1 += 1.0
            elif len(pred_tokens) == 0 or len(target_tokens) == 0:
                total_f1 += 0.0
            else:
                common = len(pred_tokens & target_tokens)
                precision = common / len(pred_tokens)
                recall = common / len(target_tokens)
                
                if precision + recall > 0:
                    f1 = 2 * precision * recall / (precision + recall)
                    total_f1 += f1
        
        return total_f1 / len(predictions)


class BLEUCalculator:
    """BLEU score calculator."""
    
    def compute(self, predictions: List[str], targets: List[str]) -> float:
        if len(predictions) == 0:
            return 0.0
        
        try:
            from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
            
            smoothing = SmoothingFunction().method1
            total_bleu = 0.0
            
            for pred, target in zip(predictions, targets):
                pred_tokens = pred.lower().split()
                target_tokens = [target.lower().split()]
                
                if len(pred_tokens) > 0:
                    score = sentence_bleu(target_tokens, pred_tokens, smoothing_function=smoothing)
                    total_bleu += score
            
            return total_bleu / len(predictions)
        
        except ImportError:
            logger.warning("NLTK not available for BLEU calculation")
            return 0.0


def create_evaluator(
    config: Optional[EvaluationConfig] = None,
    answer_vocabulary: Optional[List[str]] = None,
    **kwargs
) -> VQAEvaluator:
    """
    Factory function to create VQA evaluator.
    
    Args:
        config: Evaluation configuration
        answer_vocabulary: Answer vocabulary
        **kwargs: Additional arguments
        
    Returns:
        Initialized VQAEvaluator
    """
    return VQAEvaluator(
        config=config,
        answer_vocabulary=answer_vocabulary,
        **kwargs
    )
