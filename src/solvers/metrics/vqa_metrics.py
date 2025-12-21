"""
Metrics for VQA Evaluation
Provides evaluation metrics for Vietnamese VQA models.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Union, Tuple
from dataclasses import dataclass, field
from collections import defaultdict
import re

import torch
import numpy as np


@dataclass
class MetricResult:
    """
    Container for metric results.
    
    Attributes:
        value: Primary metric value
        per_class: Per-class values
        per_sample: Per-sample values
        metadata: Additional metadata
    """
    value: float
    per_class: Optional[Dict[str, float]] = None
    per_sample: Optional[List[float]] = None
    metadata: Optional[Dict[str, Any]] = None


class BaseMetric(ABC):
    """
    Abstract base class for metrics.
    """
    
    def __init__(self, name: str):
        """
        Initialize base metric.
        
        Args:
            name: Metric name
        """
        self.name = name
        self.reset()
    
    @abstractmethod
    def reset(self):
        """Reset metric state."""
        pass
    
    @abstractmethod
    def update(self, predictions: Any, targets: Any):
        """
        Update metric with new predictions.
        
        Args:
            predictions: Model predictions
            targets: Ground truth targets
        """
        pass
    
    @abstractmethod
    def compute(self) -> MetricResult:
        """
        Compute metric value.
        
        Returns:
            Metric result
        """
        pass


class VQAAccuracy(BaseMetric):
    """
    Standard VQA accuracy metric.
    Uses VQA v2 evaluation formula: min(#humans that provided that answer / 3, 1)
    """
    
    def __init__(
        self,
        name: str = 'vqa_accuracy',
        use_soft_accuracy: bool = True
    ):
        """
        Initialize VQA accuracy.
        
        Args:
            name: Metric name
            use_soft_accuracy: Use soft accuracy (VQA v2 style)
        """
        self.use_soft_accuracy = use_soft_accuracy
        super().__init__(name)
    
    def reset(self):
        """Reset metric state."""
        self.correct = 0.0
        self.total = 0
        self.per_sample_scores = []
    
    def update(
        self,
        predictions: torch.Tensor,
        targets: Union[torch.Tensor, List[Dict[str, float]]]
    ):
        """
        Update accuracy.
        
        Args:
            predictions: Predicted answer indices [batch] or logits [batch, num_classes]
            targets: Target indices [batch] or soft targets [batch, num_classes] or list of answer dicts
        """
        # Handle logits
        if predictions.dim() == 2:
            predictions = predictions.argmax(dim=-1)
        
        batch_size = predictions.size(0)
        
        if self.use_soft_accuracy and isinstance(targets, list):
            # Soft accuracy with multiple answers per question
            for i, (pred, target_dict) in enumerate(zip(predictions, targets)):
                pred_idx = pred.item()
                
                if pred_idx in target_dict:
                    # min(count / 3, 1)
                    score = min(target_dict[pred_idx] / 3.0, 1.0)
                else:
                    score = 0.0
                
                self.correct += score
                self.per_sample_scores.append(score)
        else:
            # Hard accuracy
            if isinstance(targets, list):
                targets = torch.tensor([list(t.keys())[0] for t in targets])
            
            correct = (predictions == targets).float()
            self.correct += correct.sum().item()
            self.per_sample_scores.extend(correct.tolist())
        
        self.total += batch_size
    
    def compute(self) -> MetricResult:
        """Compute accuracy."""
        if self.total == 0:
            accuracy = 0.0
        else:
            accuracy = self.correct / self.total
        
        return MetricResult(
            value=accuracy,
            per_sample=self.per_sample_scores,
            metadata={'total_samples': self.total}
        )


class TopKAccuracy(BaseMetric):
    """
    Top-K accuracy metric.
    """
    
    def __init__(
        self,
        k: int = 5,
        name: Optional[str] = None
    ):
        """
        Initialize top-k accuracy.
        
        Args:
            k: K value
            name: Metric name
        """
        self.k = k
        name = name or f'top{k}_accuracy'
        super().__init__(name)
    
    def reset(self):
        """Reset metric state."""
        self.correct = 0
        self.total = 0
        self.per_sample_scores = []
    
    def update(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor
    ):
        """
        Update top-k accuracy.
        
        Args:
            predictions: Prediction logits [batch, num_classes]
            targets: Target indices [batch]
        """
        # Get top-k predictions
        _, top_k_preds = predictions.topk(self.k, dim=-1)
        
        # Check if target is in top-k
        targets_expanded = targets.unsqueeze(-1).expand_as(top_k_preds)
        correct = (top_k_preds == targets_expanded).any(dim=-1).float()
        
        self.correct += correct.sum().item()
        self.total += targets.size(0)
        self.per_sample_scores.extend(correct.tolist())
    
    def compute(self) -> MetricResult:
        """Compute top-k accuracy."""
        if self.total == 0:
            accuracy = 0.0
        else:
            accuracy = self.correct / self.total
        
        return MetricResult(
            value=accuracy,
            per_sample=self.per_sample_scores,
            metadata={'k': self.k, 'total_samples': self.total}
        )


class WUPS(BaseMetric):
    """
    Wu-Palmer Similarity (WUPS) metric for VQA.
    Measures semantic similarity between predicted and ground truth answers.
    """
    
    def __init__(
        self,
        threshold: float = 0.9,
        name: str = 'wups'
    ):
        """
        Initialize WUPS metric.
        
        Args:
            threshold: WUPS threshold (typically 0.0 or 0.9)
            name: Metric name
        """
        self.threshold = threshold
        super().__init__(f'{name}@{threshold}')
        
        self._wn = None
    
    def _init_wordnet(self):
        """Initialize WordNet."""
        if self._wn is None:
            try:
                from nltk.corpus import wordnet as wn
                self._wn = wn
            except ImportError:
                raise ImportError("NLTK required for WUPS. Install with: pip install nltk")
    
    def reset(self):
        """Reset metric state."""
        self.scores = []
        self.total = 0
    
    def _wup_similarity(self, word1: str, word2: str) -> float:
        """
        Compute Wu-Palmer similarity between two words.
        
        Args:
            word1: First word
            word2: Second word
            
        Returns:
            WUP similarity score
        """
        self._init_wordnet()
        
        if word1 == word2:
            return 1.0
        
        synsets1 = self._wn.synsets(word1)
        synsets2 = self._wn.synsets(word2)
        
        if not synsets1 or not synsets2:
            return 0.0
        
        max_sim = 0.0
        for s1 in synsets1:
            for s2 in synsets2:
                sim = s1.wup_similarity(s2)
                if sim and sim > max_sim:
                    max_sim = sim
        
        return max_sim
    
    def _threshold_wups(self, similarity: float) -> float:
        """Apply threshold to WUPS score."""
        if similarity < self.threshold:
            return 0.0
        return similarity
    
    def update(
        self,
        predictions: List[str],
        targets: List[str]
    ):
        """
        Update WUPS.
        
        Args:
            predictions: Predicted answer strings
            targets: Target answer strings
        """
        for pred, target in zip(predictions, targets):
            sim = self._wup_similarity(pred.lower(), target.lower())
            score = self._threshold_wups(sim)
            self.scores.append(score)
        
        self.total += len(predictions)
    
    def compute(self) -> MetricResult:
        """Compute WUPS."""
        if not self.scores:
            return MetricResult(value=0.0)
        
        return MetricResult(
            value=np.mean(self.scores),
            per_sample=self.scores,
            metadata={'threshold': self.threshold, 'total_samples': self.total}
        )


class F1Score(BaseMetric):
    """
    F1 Score metric for multi-class classification.
    """
    
    def __init__(
        self,
        num_classes: int,
        average: str = 'macro',
        name: str = 'f1_score'
    ):
        """
        Initialize F1 score.
        
        Args:
            num_classes: Number of classes
            average: Averaging method (macro, micro, weighted)
            name: Metric name
        """
        self.num_classes = num_classes
        self.average = average
        super().__init__(name)
    
    def reset(self):
        """Reset metric state."""
        self.true_positives = torch.zeros(self.num_classes)
        self.false_positives = torch.zeros(self.num_classes)
        self.false_negatives = torch.zeros(self.num_classes)
    
    def update(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor
    ):
        """
        Update F1 score.
        
        Args:
            predictions: Predicted indices [batch] or logits [batch, num_classes]
            targets: Target indices [batch]
        """
        if predictions.dim() == 2:
            predictions = predictions.argmax(dim=-1)
        
        for pred, target in zip(predictions, targets):
            pred_idx = pred.item()
            target_idx = target.item()
            
            if pred_idx == target_idx:
                self.true_positives[pred_idx] += 1
            else:
                self.false_positives[pred_idx] += 1
                self.false_negatives[target_idx] += 1
    
    def compute(self) -> MetricResult:
        """Compute F1 score."""
        precision = self.true_positives / (self.true_positives + self.false_positives + 1e-10)
        recall = self.true_positives / (self.true_positives + self.false_negatives + 1e-10)
        f1 = 2 * precision * recall / (precision + recall + 1e-10)
        
        if self.average == 'macro':
            f1_value = f1.mean().item()
        elif self.average == 'micro':
            total_tp = self.true_positives.sum()
            total_fp = self.false_positives.sum()
            total_fn = self.false_negatives.sum()
            precision = total_tp / (total_tp + total_fp + 1e-10)
            recall = total_tp / (total_tp + total_fn + 1e-10)
            f1_value = (2 * precision * recall / (precision + recall + 1e-10)).item()
        else:  # weighted
            support = self.true_positives + self.false_negatives
            f1_value = (f1 * support).sum().item() / (support.sum().item() + 1e-10)
        
        per_class = {f'class_{i}': f1[i].item() for i in range(self.num_classes)}
        
        return MetricResult(
            value=f1_value,
            per_class=per_class,
            metadata={'average': self.average}
        )


class AnswerTypeAccuracy(BaseMetric):
    """
    Per-answer-type accuracy metric.
    Tracks accuracy for different question/answer types.
    """
    
    def __init__(
        self,
        answer_types: List[str],
        name: str = 'answer_type_accuracy'
    ):
        """
        Initialize answer type accuracy.
        
        Args:
            answer_types: List of answer types
            name: Metric name
        """
        self.answer_types = answer_types
        super().__init__(name)
    
    def reset(self):
        """Reset metric state."""
        self.correct_by_type = defaultdict(float)
        self.total_by_type = defaultdict(int)
    
    def update(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        answer_types: List[str]
    ):
        """
        Update per-type accuracy.
        
        Args:
            predictions: Predicted indices [batch]
            targets: Target indices [batch]
            answer_types: Answer types for each sample
        """
        if predictions.dim() == 2:
            predictions = predictions.argmax(dim=-1)
        
        correct = (predictions == targets).float()
        
        for i, ans_type in enumerate(answer_types):
            self.correct_by_type[ans_type] += correct[i].item()
            self.total_by_type[ans_type] += 1
    
    def compute(self) -> MetricResult:
        """Compute per-type accuracy."""
        accuracies = {}
        total_correct = 0
        total_samples = 0
        
        for ans_type in self.answer_types:
            if self.total_by_type[ans_type] > 0:
                acc = self.correct_by_type[ans_type] / self.total_by_type[ans_type]
            else:
                acc = 0.0
            
            accuracies[ans_type] = acc
            total_correct += self.correct_by_type[ans_type]
            total_samples += self.total_by_type[ans_type]
        
        overall = total_correct / total_samples if total_samples > 0 else 0.0
        
        return MetricResult(
            value=overall,
            per_class=accuracies,
            metadata={'total_by_type': dict(self.total_by_type)}
        )


class ExactMatchAccuracy(BaseMetric):
    """
    Exact match accuracy for open-ended VQA.
    """
    
    def __init__(
        self,
        normalize: bool = True,
        name: str = 'exact_match'
    ):
        """
        Initialize exact match accuracy.
        
        Args:
            normalize: Whether to normalize answers
            name: Metric name
        """
        self.normalize = normalize
        super().__init__(name)
    
    def reset(self):
        """Reset metric state."""
        self.correct = 0
        self.total = 0
        self.per_sample_scores = []
    
    def _normalize_answer(self, answer: str) -> str:
        """
        Normalize answer string.
        
        Args:
            answer: Raw answer string
            
        Returns:
            Normalized answer
        """
        if not self.normalize:
            return answer
        
        # Lowercase
        answer = answer.lower()
        
        # Remove punctuation
        answer = re.sub(r'[^\w\s]', '', answer)
        
        # Remove articles
        answer = re.sub(r'\b(a|an|the)\b', '', answer)
        
        # Remove extra whitespace
        answer = ' '.join(answer.split())
        
        return answer.strip()
    
    def update(
        self,
        predictions: List[str],
        targets: Union[List[str], List[List[str]]]
    ):
        """
        Update exact match.
        
        Args:
            predictions: Predicted answer strings
            targets: Target answer strings (single or multiple per question)
        """
        for pred, target in zip(predictions, targets):
            pred_norm = self._normalize_answer(pred)
            
            if isinstance(target, list):
                # Multiple acceptable answers
                target_norms = [self._normalize_answer(t) for t in target]
                match = pred_norm in target_norms
            else:
                target_norm = self._normalize_answer(target)
                match = pred_norm == target_norm
            
            score = 1.0 if match else 0.0
            self.correct += score
            self.per_sample_scores.append(score)
        
        self.total += len(predictions)
    
    def compute(self) -> MetricResult:
        """Compute exact match accuracy."""
        if self.total == 0:
            accuracy = 0.0
        else:
            accuracy = self.correct / self.total
        
        return MetricResult(
            value=accuracy,
            per_sample=self.per_sample_scores,
            metadata={'total_samples': self.total}
        )


class BLEUScore(BaseMetric):
    """
    BLEU score for generative VQA evaluation.
    """
    
    def __init__(
        self,
        n_gram: int = 4,
        name: str = 'bleu'
    ):
        """
        Initialize BLEU score.
        
        Args:
            n_gram: Maximum n-gram order
            name: Metric name
        """
        self.n_gram = n_gram
        super().__init__(name)
    
    def reset(self):
        """Reset metric state."""
        self.predictions = []
        self.references = []
    
    def update(
        self,
        predictions: List[str],
        references: List[List[str]]
    ):
        """
        Update BLEU.
        
        Args:
            predictions: Predicted strings
            references: Reference strings (multiple per sample)
        """
        self.predictions.extend(predictions)
        self.references.extend(references)
    
    def compute(self) -> MetricResult:
        """Compute BLEU score."""
        try:
            from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
        except ImportError:
            raise ImportError("NLTK required for BLEU. Install with: pip install nltk")
        
        # Tokenize
        tokenized_preds = [pred.split() for pred in self.predictions]
        tokenized_refs = [[ref.split() for ref in refs] for refs in self.references]
        
        # Compute BLEU
        smoothing = SmoothingFunction().method1
        
        weights = tuple([1.0 / self.n_gram] * self.n_gram)
        bleu = corpus_bleu(
            tokenized_refs,
            tokenized_preds,
            weights=weights,
            smoothing_function=smoothing
        )
        
        return MetricResult(
            value=bleu,
            metadata={'n_gram': self.n_gram, 'total_samples': len(self.predictions)}
        )


class MetricCollection:
    """
    Collection of metrics for comprehensive evaluation.
    """
    
    def __init__(self, metrics: List[BaseMetric]):
        """
        Initialize metric collection.
        
        Args:
            metrics: List of metric instances
        """
        self.metrics = {m.name: m for m in metrics}
    
    def reset(self):
        """Reset all metrics."""
        for metric in self.metrics.values():
            metric.reset()
    
    def update(self, predictions: Any, targets: Any, **kwargs):
        """
        Update all metrics.
        
        Args:
            predictions: Model predictions
            targets: Ground truth targets
            **kwargs: Additional arguments for specific metrics
        """
        for metric in self.metrics.values():
            try:
                metric.update(predictions, targets, **kwargs)
            except TypeError:
                # Metric doesn't accept kwargs
                metric.update(predictions, targets)
    
    def compute(self) -> Dict[str, MetricResult]:
        """
        Compute all metrics.
        
        Returns:
            Dictionary of metric results
        """
        return {name: metric.compute() for name, metric in self.metrics.items()}
    
    def __getitem__(self, name: str) -> BaseMetric:
        """Get metric by name."""
        return self.metrics[name]


def create_vqa_metrics(
    num_classes: int = 3000,
    answer_types: Optional[List[str]] = None
) -> MetricCollection:
    """
    Create standard VQA metric collection.
    
    Args:
        num_classes: Number of answer classes
        answer_types: List of answer types for per-type metrics
        
    Returns:
        Metric collection
    """
    metrics = [
        VQAAccuracy(use_soft_accuracy=True),
        TopKAccuracy(k=5),
        TopKAccuracy(k=10),
        F1Score(num_classes=num_classes, average='macro'),
    ]
    
    if answer_types:
        metrics.append(AnswerTypeAccuracy(answer_types))
    
    return MetricCollection(metrics)


__all__ = [
    'BaseMetric',
    'MetricResult',
    'VQAAccuracy',
    'TopKAccuracy',
    'WUPS',
    'F1Score',
    'AnswerTypeAccuracy',
    'ExactMatchAccuracy',
    'BLEUScore',
    'MetricCollection',
    'create_vqa_metrics'
]
