"""
Metrics for VQA Evaluation
Provides evaluation metrics for Vietnamese VQA models.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Union, Tuple
from dataclasses import dataclass, field
from collections import defaultdict, Counter
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


class METEORScore(BaseMetric):
    """
    METEOR (Metric for Evaluation of Translation with Explicit ORdering) score.
    Better handles synonyms and stemming compared to BLEU.
    """
    
    def __init__(self, name: str = 'meteor'):
        super().__init__(name)
    
    def reset(self):
        """Reset metric state."""
        self.predictions = []
        self.references = []
        self.per_sample_scores = []
    
    def update(
        self,
        predictions: List[str],
        references: List[List[str]]
    ):
        """
        Update METEOR.
        
        Args:
            predictions: Predicted strings
            references: Reference strings (multiple per sample)
        """
        self.predictions.extend(predictions)
        self.references.extend(references)
    
    def compute(self) -> MetricResult:
        """Compute METEOR score."""
        try:
            from nltk.translate.meteor_score import meteor_score
            import nltk
            try:
                nltk.data.find('corpora/wordnet')
            except LookupError:
                nltk.download('wordnet', quiet=True)
        except ImportError:
            raise ImportError("NLTK required for METEOR. Install with: pip install nltk")
        
        self.per_sample_scores = []
        for pred, refs in zip(self.predictions, self.references):
            # METEOR compares tokenized hypothesis with tokenized references
            pred_tokens = pred.split()
            refs_tokens = [ref.split() for ref in refs]
            
            # Get best score among all references
            best_score = 0.0
            for ref_tokens in refs_tokens:
                try:
                    score = meteor_score([ref_tokens], pred_tokens)
                    best_score = max(best_score, score)
                except Exception:
                    continue
            
            self.per_sample_scores.append(best_score)
        
        avg_score = np.mean(self.per_sample_scores) if self.per_sample_scores else 0.0
        
        return MetricResult(
            value=avg_score,
            per_sample=self.per_sample_scores,
            metadata={'total_samples': len(self.predictions)}
        )


class ROUGEScore(BaseMetric):
    """
    ROUGE (Recall-Oriented Understudy for Gisting Evaluation) score.
    Commonly used for summarization but applicable to VQA.
    """
    
    def __init__(
        self,
        rouge_type: str = 'rougeL',  # rouge1, rouge2, rougeL
        name: Optional[str] = None
    ):
        """
        Initialize ROUGE score.
        
        Args:
            rouge_type: Type of ROUGE (rouge1, rouge2, rougeL)
            name: Metric name
        """
        self.rouge_type = rouge_type
        name = name or rouge_type
        super().__init__(name)
    
    def reset(self):
        """Reset metric state."""
        self.predictions = []
        self.references = []
        self.per_sample_scores = []
    
    def update(
        self,
        predictions: List[str],
        references: List[List[str]]
    ):
        """
        Update ROUGE.
        
        Args:
            predictions: Predicted strings
            references: Reference strings (multiple per sample)
        """
        self.predictions.extend(predictions)
        self.references.extend(references)
    
    def _compute_rouge_l(self, pred: str, ref: str) -> float:
        """Compute ROUGE-L using Longest Common Subsequence."""
        pred_tokens = pred.lower().split()
        ref_tokens = ref.lower().split()
        
        if not pred_tokens or not ref_tokens:
            return 0.0
        
        # LCS computation
        m, n = len(pred_tokens), len(ref_tokens)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if pred_tokens[i-1] == ref_tokens[j-1]:
                    dp[i][j] = dp[i-1][j-1] + 1
                else:
                    dp[i][j] = max(dp[i-1][j], dp[i][j-1])
        
        lcs_length = dp[m][n]
        
        # Compute precision, recall, and F1
        precision = lcs_length / len(pred_tokens) if pred_tokens else 0.0
        recall = lcs_length / len(ref_tokens) if ref_tokens else 0.0
        
        if precision + recall == 0:
            return 0.0
        
        f1 = 2 * precision * recall / (precision + recall)
        return f1
    
    def _compute_rouge_n(self, pred: str, ref: str, n: int) -> float:
        """Compute ROUGE-N using n-gram overlap."""
        def get_ngrams(tokens, n):
            return [tuple(tokens[i:i+n]) for i in range(len(tokens)-n+1)]
        
        pred_tokens = pred.lower().split()
        ref_tokens = ref.lower().split()
        
        if len(pred_tokens) < n or len(ref_tokens) < n:
            return 0.0
        
        pred_ngrams = get_ngrams(pred_tokens, n)
        ref_ngrams = get_ngrams(ref_tokens, n)
        
        pred_counter = Counter(pred_ngrams)
        ref_counter = Counter(ref_ngrams)
        
        overlap = sum((pred_counter & ref_counter).values())
        
        precision = overlap / len(pred_ngrams) if pred_ngrams else 0.0
        recall = overlap / len(ref_ngrams) if ref_ngrams else 0.0
        
        if precision + recall == 0:
            return 0.0
        
        f1 = 2 * precision * recall / (precision + recall)
        return f1
    
    def compute(self) -> MetricResult:
        """Compute ROUGE score."""
        self.per_sample_scores = []
        
        for pred, refs in zip(self.predictions, self.references):
            # Get best score among all references
            best_score = 0.0
            for ref in refs:
                if self.rouge_type == 'rougeL':
                    score = self._compute_rouge_l(pred, ref)
                elif self.rouge_type == 'rouge1':
                    score = self._compute_rouge_n(pred, ref, 1)
                elif self.rouge_type == 'rouge2':
                    score = self._compute_rouge_n(pred, ref, 2)
                else:
                    score = self._compute_rouge_l(pred, ref)
                
                best_score = max(best_score, score)
            
            self.per_sample_scores.append(best_score)
        
        avg_score = np.mean(self.per_sample_scores) if self.per_sample_scores else 0.0
        
        return MetricResult(
            value=avg_score,
            per_sample=self.per_sample_scores,
            metadata={'rouge_type': self.rouge_type, 'total_samples': len(self.predictions)}
        )


class CIDErScore(BaseMetric):
    """
    CIDEr (Consensus-based Image Description Evaluation) score.
    Measures consensus between predicted and reference captions using TF-IDF.
    """
    
    def __init__(
        self,
        n_gram: int = 4,
        name: str = 'cider'
    ):
        """
        Initialize CIDEr score.
        
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
        self.per_sample_scores = []
    
    def update(
        self,
        predictions: List[str],
        references: List[List[str]]
    ):
        """
        Update CIDEr.
        
        Args:
            predictions: Predicted strings
            references: Reference strings (multiple per sample)
        """
        self.predictions.extend(predictions)
        self.references.extend(references)
    
    def _get_ngrams(self, sentence: str, n: int) -> Counter:
        """Get n-grams from a sentence."""
        tokens = sentence.lower().split()
        ngrams = [tuple(tokens[i:i+n]) for i in range(len(tokens)-n+1)]
        return Counter(ngrams)
    
    def _compute_document_frequency(self) -> Dict:
        """Compute document frequency for each n-gram."""
        df = {}
        num_docs = len(self.references)
        
        for n in range(1, self.n_gram + 1):
            df[n] = defaultdict(int)
            for refs in self.references:
                ngram_set = set()
                for ref in refs:
                    ngrams = self._get_ngrams(ref, n)
                    ngram_set.update(ngrams.keys())
                for ngram in ngram_set:
                    df[n][ngram] += 1
        
        return df, num_docs
    
    def _compute_tfidf(self, sentence: str, df: Dict, num_docs: int) -> Dict:
        """Compute TF-IDF vector for a sentence."""
        tfidf = {}
        
        for n in range(1, self.n_gram + 1):
            ngrams = self._get_ngrams(sentence, n)
            tfidf[n] = {}
            
            for ngram, count in ngrams.items():
                # TF: term frequency
                tf = count
                # IDF: inverse document frequency
                doc_freq = df[n].get(ngram, 0)
                if doc_freq > 0:
                    idf = np.log((num_docs + 1) / (doc_freq + 1))
                else:
                    idf = 0.0
                
                tfidf[n][ngram] = tf * idf
        
        return tfidf
    
    def _cosine_similarity(self, vec1: Dict, vec2: Dict) -> float:
        """Compute cosine similarity between two vectors."""
        dot_product = 0.0
        norm1 = 0.0
        norm2 = 0.0
        
        all_keys = set(vec1.keys()) | set(vec2.keys())
        
        for key in all_keys:
            v1 = vec1.get(key, 0.0)
            v2 = vec2.get(key, 0.0)
            dot_product += v1 * v2
            norm1 += v1 ** 2
            norm2 += v2 ** 2
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (np.sqrt(norm1) * np.sqrt(norm2))
    
    def compute(self) -> MetricResult:
        """Compute CIDEr score."""
        if not self.predictions:
            return MetricResult(value=0.0)
        
        df, num_docs = self._compute_document_frequency()
        self.per_sample_scores = []
        
        for pred, refs in zip(self.predictions, self.references):
            pred_tfidf = self._compute_tfidf(pred, df, num_docs)
            
            # Compute average similarity with all references
            similarities = []
            for ref in refs:
                ref_tfidf = self._compute_tfidf(ref, df, num_docs)
                
                # Average over n-gram orders
                sim_per_n = []
                for n in range(1, self.n_gram + 1):
                    sim = self._cosine_similarity(pred_tfidf[n], ref_tfidf[n])
                    sim_per_n.append(sim)
                
                avg_sim = np.mean(sim_per_n) if sim_per_n else 0.0
                similarities.append(avg_sim)
            
            # CIDEr uses mean over references
            cider_score = np.mean(similarities) if similarities else 0.0
            self.per_sample_scores.append(cider_score)
        
        # Scale by 10 (standard CIDEr scaling)
        avg_score = np.mean(self.per_sample_scores) * 10 if self.per_sample_scores else 0.0
        
        return MetricResult(
            value=avg_score,
            per_sample=[s * 10 for s in self.per_sample_scores],
            metadata={'n_gram': self.n_gram, 'total_samples': len(self.predictions)}
        )


class VQASoftAccuracy(BaseMetric):
    """
    Proper VQA v2 soft accuracy metric.
    Formula: accuracy = min(#humans_that_provided_answer / 3, 1)
    
    This metric compares the predicted answer against all annotator answers
    and computes accuracy based on how many annotators agreed with the prediction.
    """
    
    def __init__(
        self,
        id2answer: Optional[Dict[int, str]] = None,
        name: str = 'vqa_soft_accuracy'
    ):
        """
        Initialize VQA soft accuracy.
        
        Args:
            id2answer: Mapping from answer ID to answer string
            name: Metric name
        """
        self.id2answer = id2answer or {}
        super().__init__(name)
    
    def reset(self):
        """Reset metric state."""
        self.scores = []
        self.total = 0
    
    def update(
        self,
        predictions: torch.Tensor,
        answer_counts: List[Dict[int, int]]
    ):
        """
        Update VQA soft accuracy.
        
        Args:
            predictions: Predicted answer indices [batch]
            answer_counts: List of dicts mapping answer_id -> count for each sample
        """
        if predictions.dim() == 2:
            predictions = predictions.argmax(dim=-1)
        
        for pred, counts in zip(predictions, answer_counts):
            pred_id = pred.item()
            
            # VQA v2 formula: min(count / 3, 1)
            if pred_id in counts:
                score = min(counts[pred_id] / 3.0, 1.0)
            else:
                score = 0.0
            
            self.scores.append(score)
        
        self.total += len(answer_counts)
    
    def compute(self) -> MetricResult:
        """Compute VQA soft accuracy."""
        if not self.scores:
            return MetricResult(value=0.0)
        
        accuracy = np.mean(self.scores)
        
        return MetricResult(
            value=accuracy,
            per_sample=self.scores,
            metadata={'total_samples': self.total}
        )


class PrecisionRecallF1(BaseMetric):
    """
    Precision, Recall, and F1 metrics for VQA.
    Computes these at the answer level (word overlap).
    """
    
    def __init__(
        self,
        name: str = 'precision_recall_f1'
    ):
        super().__init__(name)
    
    def reset(self):
        """Reset metric state."""
        self.precisions = []
        self.recalls = []
        self.f1s = []
    
    def _normalize(self, text: str) -> set:
        """Normalize text and return set of words."""
        # Lowercase and remove punctuation
        text = re.sub(r'[^\w\s]', '', text.lower())
        return set(text.split())
    
    def update(
        self,
        predictions: List[str],
        references: List[List[str]]
    ):
        """
        Update Precision, Recall, F1.
        
        Args:
            predictions: Predicted answer strings
            references: Reference answer strings (multiple per sample)
        """
        for pred, refs in zip(predictions, references):
            pred_words = self._normalize(pred)
            
            # Compute against all references and take best F1
            best_precision = 0.0
            best_recall = 0.0
            best_f1 = 0.0
            
            for ref in refs:
                ref_words = self._normalize(ref)
                
                if not pred_words or not ref_words:
                    continue
                
                overlap = pred_words & ref_words
                
                precision = len(overlap) / len(pred_words) if pred_words else 0.0
                recall = len(overlap) / len(ref_words) if ref_words else 0.0
                
                if precision + recall > 0:
                    f1 = 2 * precision * recall / (precision + recall)
                else:
                    f1 = 0.0
                
                if f1 > best_f1:
                    best_precision = precision
                    best_recall = recall
                    best_f1 = f1
            
            self.precisions.append(best_precision)
            self.recalls.append(best_recall)
            self.f1s.append(best_f1)
    
    def compute(self) -> MetricResult:
        """Compute Precision, Recall, F1."""
        avg_precision = np.mean(self.precisions) if self.precisions else 0.0
        avg_recall = np.mean(self.recalls) if self.recalls else 0.0
        avg_f1 = np.mean(self.f1s) if self.f1s else 0.0
        
        return MetricResult(
            value=avg_f1,
            metadata={
                'precision': avg_precision,
                'recall': avg_recall,
                'f1': avg_f1,
                'total_samples': len(self.f1s)
            }
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
    answer_types: Optional[List[str]] = None,
    id2answer: Optional[Dict[int, str]] = None
) -> MetricCollection:
    """
    Create standard VQA metric collection.
    
    Args:
        num_classes: Number of answer classes
        answer_types: List of answer types for per-type metrics
        id2answer: Mapping from answer ID to answer string
        
    Returns:
        Metric collection
    """
    metrics = [
        VQAAccuracy(use_soft_accuracy=True),
        VQASoftAccuracy(id2answer=id2answer),
        TopKAccuracy(k=5),
        TopKAccuracy(k=10),
        F1Score(num_classes=num_classes, average='macro'),
        BLEUScore(n_gram=4),
        METEORScore(),
        ROUGEScore(rouge_type='rougeL'),
        CIDErScore(n_gram=4),
        PrecisionRecallF1(),
    ]
    
    if answer_types:
        metrics.append(AnswerTypeAccuracy(answer_types))
    
    return MetricCollection(metrics)


def create_comprehensive_vqa_metrics(
    id2answer: Optional[Dict[int, str]] = None
) -> Dict[str, BaseMetric]:
    """
    Create comprehensive VQA metrics for evaluation.
    
    Args:
        id2answer: Mapping from answer ID to answer string
        
    Returns:
        Dictionary of metric instances
    """
    return {
        'vqa_accuracy': VQASoftAccuracy(id2answer=id2answer),
        'exact_match': ExactMatchAccuracy(normalize=True),
        'bleu': BLEUScore(n_gram=4),
        'meteor': METEORScore(),
        'rouge_l': ROUGEScore(rouge_type='rougeL'),
        'rouge_1': ROUGEScore(rouge_type='rouge1'),
        'cider': CIDErScore(n_gram=4),
        'precision_recall_f1': PrecisionRecallF1(),
    }


__all__ = [
    'BaseMetric',
    'MetricResult',
    'VQAAccuracy',
    'VQASoftAccuracy',
    'TopKAccuracy',
    'WUPS',
    'F1Score',
    'AnswerTypeAccuracy',
    'ExactMatchAccuracy',
    'BLEUScore',
    'METEORScore',
    'ROUGEScore',
    'CIDErScore',
    'PrecisionRecallF1',
    'MetricCollection',
    'create_vqa_metrics',
    'create_comprehensive_vqa_metrics'
]
