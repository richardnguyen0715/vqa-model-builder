"""
VIVQA Evaluation Pipeline
Inference and evaluation pipeline for generative VQA models on VIVQA dataset.

Evaluates pre-trained models on VIVQA dataset using multiple metrics:
- Accuracy (exact match)
- Precision / Recall / F1
- BLEU, ROUGE-L, METEOR, CIDEr (NLG metrics)

Features:
- Load pre-trained model from checkpoint
- Batch inference with generation
- Comprehensive metric computation
- Detailed logging and results export
"""

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import json
import logging

from tqdm import tqdm
import numpy as np

from src.data.vivqa_dataset import VivqaDataset, create_vivqa_dataloader
from src.middleware.logger import data_process_logger
from src.core.pipeline_logger import PipelineLogger, get_pipeline_logger


# Try to import metrics and generation utilities
try:
    from src.solvers.metrics.vqa_metrics import (
        BaseMetric,
        ExactMatchAccuracy,
        BLEUScore,
        ROUGEScore,
        METEORScore,
        CIDErScore,
        PrecisionRecallF1
    )
    METRICS_AVAILABLE = True
except ImportError:
    METRICS_AVAILABLE = False
    data_process_logger.warning("VQA metrics module not fully available")


@dataclass
class EvaluationConfig:
    """Configuration for evaluation pipeline."""
    
    # Data
    csv_path: str = "data/vivqa/test.csv"
    images_dir: str = "data/vivqa/images"
    batch_size: int = 32
    num_workers: int = 4
    
    # Model
    checkpoint_path: str = ""
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Generation
    max_generate_length: int = 64
    num_beams: int = 1
    do_sample: bool = False
    temperature: float = 1.0
    top_p: float = 0.9
    
    # Output
    output_dir: str = "outputs/evaluation"
    save_predictions: bool = True
    save_metrics: bool = True
    
    # Logging
    num_samples_to_display: int = 10
    log_interval: int = 10


@dataclass
class EvaluationResult:
    """Container for evaluation results."""
    
    total_samples: int
    metrics: Dict[str, float]
    per_sample_metrics: List[Dict[str, Any]]
    predictions: List[Dict[str, Any]]
    summary: Dict[str, Any]
    evaluation_time: float


class VivqaEvaluationPipeline:
    """
    Evaluation pipeline for VIVQA dataset.
    
    Handles:
    - Loading pre-trained models
    - Generation on VIVQA dataset
    - Computing multiple metrics
    - Logging results
    """
    
    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        config: Optional[EvaluationConfig] = None,
        logger: Optional[PipelineLogger] = None,
        device: Optional[torch.device] = None
    ):
        """
        Initialize evaluation pipeline.
        
        Args:
            model: Pre-trained model instance
            tokenizer: Tokenizer for the model
            config: Evaluation configuration
            logger: Pipeline logger
            device: Device to run evaluation on
        """
        self.model = model
        self.tokenizer = tokenizer
        self.config = config or EvaluationConfig()
        self.logger = logger or get_pipeline_logger()
        self.device = device or torch.device(self.config.device)
        
        # Move model to device
        self.model.to(self.device)
        self.model.eval()
        
        # Create output directory
        Path(self.config.output_dir).mkdir(parents=True, exist_ok=True)
        
        self.logger.info(
            f"VivqaEvaluationPipeline initialized on device: {self.device}"
        )
    
    def evaluate(self) -> EvaluationResult:
        """Execute evaluation pipeline."""
        self.logger.start_stage("VIVQA EVALUATION")
        start_time = datetime.now()
        
        try:
            # Setup data
            self.logger.info("Setting up evaluation data")
            dataset = VivqaDataset(
                csv_path=self.config.csv_path,
                images_dir=self.config.images_dir,
                tokenizer=self.tokenizer,
                mode='inference'
            )
            
            dataloader = create_vivqa_dataloader(
                dataset=dataset,
                batch_size=self.config.batch_size,
                num_workers=self.config.num_workers,
                shuffle=False,
                pin_memory=True
            )
            
            self.logger.info(f"Loaded {len(dataset)} samples for evaluation")
            
            # Run inference
            predictions, references = self._inference_loop(dataloader)
            
            # Compute metrics
            self.logger.info("Computing evaluation metrics")
            metrics = self._compute_metrics(predictions, references)
            
            # Build results
            evaluation_time = (datetime.now() - start_time).total_seconds()
            
            results = EvaluationResult(
                total_samples=len(predictions),
                metrics=metrics,
                per_sample_metrics=[],
                predictions=predictions,
                summary=self._build_summary(metrics, len(predictions), evaluation_time),
                evaluation_time=evaluation_time
            )
            
            # Log and save results
            self._log_results(results)
            if self.config.save_predictions or self.config.save_metrics:
                self._save_results(results)
            
            self.logger.end_stage("EVALUATION COMPLETED SUCCESSFULLY")
            return results
        
        except Exception as e:
            self.logger.error(f"Evaluation pipeline failed: {e}")
            raise
    
    def _inference_loop(
        self,
        dataloader: DataLoader
    ) -> Tuple[List[Dict], List[Dict]]:
        """
        Run inference on dataset.
        
        Args:
            dataloader: Evaluation dataloader
            
        Returns:
            Tuple of (predictions, references) lists
        """
        self.logger.info("Starting inference loop")
        
        predictions = []
        references = []
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(dataloader, desc="Inference")):
                # Move batch to device
                batch = self._move_batch_to_device(batch)
                
                # Generate predictions
                generated_ids = self.model.generate(
                    input_ids=batch['input_ids'],
                    attention_mask=batch['attention_mask'],
                    images=batch['image'],
                    max_length=self.config.max_generate_length,
                    num_beams=self.config.num_beams,
                    do_sample=self.config.do_sample,
                    temperature=self.config.temperature if self.config.do_sample else 1.0,
                    top_p=self.config.top_p if self.config.do_sample else 1.0
                )
                
                # Decode predictions
                predictions_batch = self._decode_tokens(generated_ids)
                
                # Prepare predictions and references
                for i, (pred, ref, img_id, q) in enumerate(zip(
                    predictions_batch,
                    batch['answer'],
                    batch['img_id'],
                    batch['question']
                )):
                    predictions.append({
                        'prediction': pred,
                        'reference': ref,
                        'img_id': img_id,
                        'question': q
                    })
                    references.append(ref)
                
                # Log sample predictions
                if batch_idx % self.config.log_interval == 0:
                    self.logger.debug(
                        f"Batch {batch_idx}: Sample predictions\n"
                        f"  Question: {batch['question'][0]}\n"
                        f"  Prediction: {predictions_batch[0]}\n"
                        f"  Reference: {batch['answer'][0]}"
                    )
        
        self.logger.info(f"Inference completed for {len(predictions)} samples")
        return predictions, references
    
    def _decode_tokens(self, token_ids: torch.Tensor) -> List[str]:
        """
        Decode token IDs to text.
        
        Args:
            token_ids: Token IDs [batch, seq_len]
            
        Returns:
            List of decoded texts
        """
        texts = []
        for ids in token_ids:
            # Remove special tokens and padding
            ids_list = ids.tolist()
            
            # Find EOS token
            eos_id = getattr(self.tokenizer, 'eos_token_id', 2)
            if eos_id in ids_list:
                ids_list = ids_list[:ids_list.index(eos_id)]
            
            # Remove BOS and padding tokens
            bos_id = getattr(self.tokenizer, 'bos_token_id', 0)
            pad_id = getattr(self.tokenizer, 'pad_token_id', 1)
            ids_list = [t for t in ids_list if t not in [bos_id, pad_id]]
            
            # Decode
            text = self.tokenizer.decode(ids_list, skip_special_tokens=True)
            texts.append(text.strip())
        
        return texts
    
    def _compute_metrics(
        self,
        predictions: List[Dict],
        references: List[str]
    ) -> Dict[str, float]:
        """
        Compute evaluation metrics.
        
        Args:
            predictions: List of prediction dictionaries
            references: List of reference texts
            
        Returns:
            Dictionary of metrics
        """
        pred_texts = [p['prediction'] for p in predictions]
        
        metrics = {}
        
        # 1. Exact Match Accuracy
        self.logger.info("Computing exact match accuracy")
        exact_matches = sum(
            1 for pred, ref in zip(pred_texts, references)
            if self._normalize_answer(pred) == self._normalize_answer(ref)
        )
        metrics['accuracy'] = exact_matches / len(predictions) if predictions else 0.0
        
        # 2. Precision / Recall / F1 (token-level)
        self.logger.info("Computing precision, recall, F1")
        precision, recall, f1 = self._compute_precision_recall_f1(pred_texts, references)
        metrics['precision'] = precision
        metrics['recall'] = recall
        metrics['f1'] = f1
        
        # 3. NLG Metrics (BLEU, ROUGE-L, METEOR, CIDEr)
        if METRICS_AVAILABLE:
            self.logger.info("Computing NLG metrics (BLEU, ROUGE-L, METEOR, CIDEr)")
            
            # BLEU
            try:
                bleu_metric = BLEUScore(n_gram=4)
                bleu_metric.update(
                    predictions=pred_texts,
                    references=[[ref] for ref in references]
                )
                bleu_result = bleu_metric.compute()
                metrics['bleu'] = bleu_result.value
            except Exception as e:
                self.logger.warning(f"Failed to compute BLEU: {e}")
                metrics['bleu'] = 0.0
            
            # ROUGE-L
            try:
                rouge_metric = ROUGEScore()
                rouge_metric.update(
                    predictions=pred_texts,
                    references=[[ref] for ref in references]
                )
                rouge_result = rouge_metric.compute()
                metrics['rouge_l'] = rouge_result.value
            except Exception as e:
                self.logger.warning(f"Failed to compute ROUGE-L: {e}")
                metrics['rouge_l'] = 0.0
            
            # METEOR
            try:
                meteor_metric = METEORScore()
                meteor_metric.update(
                    predictions=pred_texts,
                    references=[[ref] for ref in references]
                )
                meteor_result = meteor_metric.compute()
                metrics['meteor'] = meteor_result.value
            except Exception as e:
                self.logger.warning(f"Failed to compute METEOR: {e}")
                metrics['meteor'] = 0.0
            
            # CIDEr
            try:
                cider_metric = CIDErScore()
                cider_metric.update(
                    predictions=pred_texts,
                    references=[[ref] for ref in references]
                )
                cider_result = cider_metric.compute()
                metrics['cider'] = cider_result.value
            except Exception as e:
                self.logger.warning(f"Failed to compute CIDEr: {e}")
                metrics['cider'] = 0.0
        else:
            self.logger.warning("NLG metrics not available")
            metrics['bleu'] = 0.0
            metrics['rouge_l'] = 0.0
            metrics['meteor'] = 0.0
            metrics['cider'] = 0.0
        
        return metrics
    
    def _normalize_answer(self, answer: str) -> str:
        """Normalize answer for comparison."""
        # Convert to lowercase and remove extra spaces
        answer = answer.lower().strip()
        # Remove common punctuation
        answer = answer.replace('.', '').replace(',', '')
        return ' '.join(answer.split())
    
    def _compute_precision_recall_f1(
        self,
        predictions: List[str],
        references: List[str]
    ) -> Tuple[float, float, float]:
        """
        Compute token-level precision, recall, and F1.
        
        Args:
            predictions: Predicted texts
            references: Reference texts
            
        Returns:
            Tuple of (precision, recall, f1)
        """
        total_precision = 0.0
        total_recall = 0.0
        total_f1 = 0.0
        
        for pred, ref in zip(predictions, references):
            # Tokenize
            pred_tokens = set(self._normalize_answer(pred).split())
            ref_tokens = set(self._normalize_answer(ref).split())
            
            # Compute intersection
            common_tokens = pred_tokens & ref_tokens
            
            # Precision
            if pred_tokens:
                precision = len(common_tokens) / len(pred_tokens)
            else:
                precision = 0.0 if ref_tokens else 1.0
            
            # Recall
            if ref_tokens:
                recall = len(common_tokens) / len(ref_tokens)
            else:
                recall = 0.0 if pred_tokens else 1.0
            
            # F1
            if precision + recall > 0:
                f1 = 2 * (precision * recall) / (precision + recall)
            else:
                f1 = 0.0
            
            total_precision += precision
            total_recall += recall
            total_f1 += f1
        
        n = len(predictions)
        return (
            total_precision / n if n > 0 else 0.0,
            total_recall / n if n > 0 else 0.0,
            total_f1 / n if n > 0 else 0.0
        )
    
    def _move_batch_to_device(self, batch: Dict) -> Dict:
        """Move batch tensors to device."""
        for key in batch:
            if isinstance(batch[key], torch.Tensor):
                batch[key] = batch[key].to(self.device)
        return batch
    
    def _log_results(self, results: EvaluationResult) -> None:
        """Log evaluation results."""
        self.logger.info("\n" + "=" * 80)
        self.logger.info("EVALUATION RESULTS SUMMARY")
        self.logger.info("=" * 80)
        
        self.logger.info(f"Total samples evaluated: {results.total_samples}")
        self.logger.info(f"Evaluation time: {results.evaluation_time:.2f} seconds")
        
        self.logger.info("\nMetrics:")
        for metric_name, metric_value in results.metrics.items():
            self.logger.info(f"  {metric_name.upper()}: {metric_value:.4f}")
        
        # Display sample predictions
        self.logger.info(f"\nSample Predictions (first {self.config.num_samples_to_display}):")
        for i, pred in enumerate(results.predictions[:self.config.num_samples_to_display]):
            self.logger.info(
                f"\nSample {i+1}:\n"
                f"  Question: {pred['question']}\n"
                f"  Prediction: {pred['prediction']}\n"
                f"  Reference: {pred['reference']}"
            )
        
        self.logger.info("\n" + "=" * 80)
    
    def _build_summary(
        self,
        metrics: Dict[str, float],
        total_samples: int,
        eval_time: float
    ) -> Dict[str, Any]:
        """Build results summary."""
        return {
            'timestamp': datetime.now().isoformat(),
            'total_samples': total_samples,
            'evaluation_time_seconds': eval_time,
            'metrics': metrics,
            'config': {
                'csv_path': self.config.csv_path,
                'batch_size': self.config.batch_size,
                'max_generate_length': self.config.max_generate_length
            }
        }
    
    def _save_results(self, results: EvaluationResult) -> None:
        """Save evaluation results to files."""
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Save metrics
        if self.config.save_metrics:
            metrics_path = output_dir / f"metrics_{timestamp}.json"
            with open(metrics_path, 'w', encoding='utf-8') as f:
                json.dump(results.summary, f, ensure_ascii=False, indent=2)
            self.logger.info(f"Metrics saved to: {metrics_path}")
        
        # Save predictions
        if self.config.save_predictions:
            predictions_path = output_dir / f"predictions_{timestamp}.json"
            with open(predictions_path, 'w', encoding='utf-8') as f:
                json.dump(results.predictions, f, ensure_ascii=False, indent=2)
            self.logger.info(f"Predictions saved to: {predictions_path}")
