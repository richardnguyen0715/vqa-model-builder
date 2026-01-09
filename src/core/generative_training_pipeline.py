"""
Generative VQA Training Pipeline
Training pipeline for sequence-to-sequence VQA models.
"""

import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, OneCycleLR
from torch.amp import autocast, GradScaler
from tqdm import tqdm
import numpy as np

from src.core.pipeline_logger import get_pipeline_logger, PipelineLogger
from src.modeling.meta_arch.generative_vqa_model import GenerativeVQAModel, GenerativeVQAConfig


@dataclass
class GenerativeTrainingConfig:
    """Configuration for generative VQA training."""
    
    # Training
    num_epochs: int = 20
    learning_rate: float = 5e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0
    
    # Mixed precision
    use_amp: bool = True
    
    # Early stopping
    early_stopping: bool = True
    patience: int = 5
    min_delta: float = 0.001
    
    # Checkpointing
    checkpoint_dir: str = "checkpoints"
    save_best: bool = True
    save_every_epoch: bool = False
    metric_for_best: str = "bleu"  # Use BLEU for generative models
    
    # Logging
    log_interval: int = 50
    eval_interval: int = 1
    num_samples_to_display: int = 5
    
    # Generation during eval
    max_generate_length: int = 64
    num_beams: int = 1
    do_sample: bool = False
    temperature: float = 1.0
    
    # Reproducibility
    seed: int = 42


@dataclass
class GenerativeTrainingOutput:
    """Output from generative training pipeline."""
    
    best_model_path: str
    best_metric: float
    final_metrics: Dict[str, float]
    training_history: List[Dict[str, float]]
    validation_history: List[Dict[str, float]]
    total_epochs: int
    training_time: float


class GenerativeTrainingPipeline:
    """
    Training pipeline for Generative VQA models.
    
    Handles:
    - Language modeling loss (CrossEntropy on token predictions)
    - Generation during evaluation
    - NLG metrics (BLEU, METEOR, ROUGE, CIDEr)
    """
    
    def __init__(
        self,
        model: GenerativeVQAModel,
        train_loader: DataLoader,
        val_loader: DataLoader,
        tokenizer: Any,
        config: Optional[GenerativeTrainingConfig] = None,
        logger: Optional[PipelineLogger] = None,
        device: Optional[torch.device] = None
    ):
        """
        Initialize training pipeline.
        
        Args:
            model: GenerativeVQAModel instance
            train_loader: Training dataloader
            val_loader: Validation dataloader
            tokenizer: Answer tokenizer for decoding
            config: Training configuration
            logger: Pipeline logger
            device: Training device
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.tokenizer = tokenizer
        self.config = config or GenerativeTrainingConfig()
        self.logger = logger or get_pipeline_logger()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Move model to device
        self.model.to(self.device)
        
        # Training state
        self.optimizer = None
        self.scheduler = None
        self.scaler = None
        
        self.current_epoch = 0
        self.global_step = 0
        self.best_metric = 0.0
        self.best_model_path = None
        
        # History
        self.training_history = []
        self.validation_history = []
        
        # Early stopping
        self.early_stop_counter = 0
        self.should_stop = False
    
    def run(self) -> GenerativeTrainingOutput:
        """Execute training pipeline."""
        self.logger.start_stage("GENERATIVE VQA TRAINING")
        start_time = time.time()
        
        try:
            # Setup
            self._set_seed()
            self._setup_optimizer()
            self._setup_scheduler()
            self._setup_amp()
            self._setup_checkpointing()
            
            # Training loop
            self._training_loop()
            
            # Final evaluation
            final_metrics = self._final_evaluation()
            
            training_time = time.time() - start_time
            self.logger.end_stage("GENERATIVE VQA TRAINING", success=True)
            
            return GenerativeTrainingOutput(
                best_model_path=self.best_model_path,
                best_metric=self.best_metric,
                final_metrics=final_metrics,
                training_history=self.training_history,
                validation_history=self.validation_history,
                total_epochs=self.current_epoch,
                training_time=training_time
            )
            
        except Exception as e:
            self.logger.error(f"Training failed: {e}")
            self.logger.end_stage("GENERATIVE VQA TRAINING", success=False)
            raise
    
    def _set_seed(self):
        """Set random seed."""
        import random
        seed = self.config.seed
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        self.logger.info(f"Random seed set to {seed}")
    
    def _setup_optimizer(self):
        """Setup optimizer with weight decay."""
        no_decay = ['bias', 'LayerNorm.weight', 'layer_norm.weight']
        
        optimizer_grouped_params = [
            {
                'params': [p for n, p in self.model.named_parameters() 
                          if not any(nd in n for nd in no_decay) and p.requires_grad],
                'weight_decay': self.config.weight_decay
            },
            {
                'params': [p for n, p in self.model.named_parameters() 
                          if any(nd in n for nd in no_decay) and p.requires_grad],
                'weight_decay': 0.0
            }
        ]
        
        self.optimizer = AdamW(
            optimizer_grouped_params,
            lr=self.config.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-8
        )
        
        self.logger.info(f"Optimizer: AdamW (lr={self.config.learning_rate})")
    
    def _setup_scheduler(self):
        """Setup learning rate scheduler."""
        total_steps = len(self.train_loader) * self.config.num_epochs
        warmup_steps = int(total_steps * self.config.warmup_ratio)
        
        self.scheduler = OneCycleLR(
            self.optimizer,
            max_lr=self.config.learning_rate,
            total_steps=total_steps,
            pct_start=self.config.warmup_ratio,
            anneal_strategy='cos'
        )
        
        self.logger.info(f"Scheduler: OneCycleLR (warmup={warmup_steps} steps)")
    
    def _setup_amp(self):
        """Setup automatic mixed precision."""
        if self.config.use_amp and self.device.type == 'cuda':
            self.scaler = GradScaler()
            self.logger.info("Mixed precision training enabled")
        else:
            self.scaler = None
    
    def _setup_checkpointing(self):
        """Setup checkpoint directory."""
        checkpoint_dir = Path(self.config.checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.logger.info(f"Checkpoint directory: {checkpoint_dir}")
    
    def _training_loop(self):
        """Main training loop."""
        self.logger.info(f"Starting training for {self.config.num_epochs} epochs")
        
        for epoch in range(self.config.num_epochs):
            self.current_epoch = epoch + 1
            self.logger.section(f"EPOCH {self.current_epoch}/{self.config.num_epochs}")
            
            # Train
            train_metrics = self._train_epoch()
            self.training_history.append(train_metrics)
            
            # Validate
            if self.current_epoch % self.config.eval_interval == 0:
                val_metrics = self._validate_epoch()
                self.validation_history.append(val_metrics)
                
                # Check improvement
                current_metric = val_metrics.get(self.config.metric_for_best, 0)
                is_best = current_metric > self.best_metric + self.config.min_delta
                
                if is_best:
                    self.best_metric = current_metric
                    self.early_stop_counter = 0
                    self.logger.success(f"New best {self.config.metric_for_best}: {self.best_metric:.4f}")
                else:
                    self.early_stop_counter += 1
                    self.logger.info(f"No improvement. Early stop: {self.early_stop_counter}/{self.config.patience}")
                
                # Save checkpoint
                self._save_checkpoint(is_best, val_metrics)
                
                # Early stopping
                if self.config.early_stopping and self.early_stop_counter >= self.config.patience:
                    self.logger.warning(f"Early stopping at epoch {self.current_epoch}")
                    break
            
            # Log summary
            self._log_epoch_summary(train_metrics, val_metrics if self.current_epoch % self.config.eval_interval == 0 else None)
        
        self.logger.success(f"Training completed after {self.current_epoch} epochs")
    
    def _train_epoch(self) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        
        total_loss = 0.0
        total_tokens = 0
        
        progress_bar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch} [Train]")
        
        for step, batch in enumerate(progress_bar):
            # Move to device
            batch = self._move_batch_to_device(batch)
            
            # Forward pass
            if self.scaler is not None:
                with autocast(device_type='cuda', dtype=torch.float16):
                    outputs = self.model(
                        pixel_values=batch['image'],
                        input_ids=batch['input_ids'],
                        attention_mask=batch['attention_mask'],
                        decoder_input_ids=batch['decoder_input_ids'],
                        decoder_attention_mask=batch['decoder_attention_mask'],
                        labels=batch['labels']
                    )
                    loss = outputs.loss / self.config.gradient_accumulation_steps
                
                self.scaler.scale(loss).backward()
            else:
                outputs = self.model(
                    pixel_values=batch['image'],
                    input_ids=batch['input_ids'],
                    attention_mask=batch['attention_mask'],
                    decoder_input_ids=batch['decoder_input_ids'],
                    decoder_attention_mask=batch['decoder_attention_mask'],
                    labels=batch['labels']
                )
                loss = outputs.loss / self.config.gradient_accumulation_steps
                loss.backward()
            
            total_loss += outputs.loss.item()
            
            # Count non-padding tokens
            num_tokens = (batch['labels'] != -100).sum().item()
            total_tokens += num_tokens
            
            # Gradient accumulation step
            if (step + 1) % self.config.gradient_accumulation_steps == 0:
                if self.scaler is not None:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
                    self.optimizer.step()
                
                self.optimizer.zero_grad()
                self.scheduler.step()
                self.global_step += 1
            
            # Update progress bar
            avg_loss = total_loss / (step + 1)
            perplexity = np.exp(min(avg_loss, 100))  # Cap to avoid overflow
            progress_bar.set_postfix({
                'loss': f'{avg_loss:.4f}',
                'ppl': f'{perplexity:.2f}',
                'lr': f'{self.scheduler.get_last_lr()[0]:.2e}'
            })
        
        avg_loss = total_loss / len(self.train_loader)
        perplexity = np.exp(min(avg_loss, 100))
        
        return {
            'loss': avg_loss,
            'perplexity': perplexity,
            'learning_rate': self.scheduler.get_last_lr()[0]
        }
    
    def _validate_epoch(self) -> Dict[str, float]:
        """Validate with generation and NLG metrics."""
        from src.solvers.metrics.vqa_metrics import (
            BLEUScore, METEORScore, ROUGEScore, CIDErScore, 
            PrecisionRecallF1, ExactMatchAccuracy
        )
        
        self.model.eval()
        
        total_loss = 0.0
        all_predictions = []
        all_references = []
        all_questions = []
        
        # Initialize metrics
        bleu_metric = BLEUScore(n_gram=4)
        meteor_metric = METEORScore()
        rouge_metric = ROUGEScore(rouge_type='rougeL')
        cider_metric = CIDErScore(n_gram=4)
        prf_metric = PrecisionRecallF1()
        exact_match = ExactMatchAccuracy(normalize=True)
        
        progress_bar = tqdm(self.val_loader, desc=f"Epoch {self.current_epoch} [Val]")
        
        with torch.no_grad():
            for batch in progress_bar:
                batch = self._move_batch_to_device(batch)
                
                # Compute loss
                outputs = self.model(
                    pixel_values=batch['image'],
                    input_ids=batch['input_ids'],
                    attention_mask=batch['attention_mask'],
                    decoder_input_ids=batch['decoder_input_ids'],
                    decoder_attention_mask=batch['decoder_attention_mask'],
                    labels=batch['labels']
                )
                total_loss += outputs.loss.item()
                
                # Generate predictions
                generated_ids = self.model.generate(
                    pixel_values=batch['image'],
                    input_ids=batch['input_ids'],
                    attention_mask=batch['attention_mask'],
                    max_length=self.config.max_generate_length,
                    num_beams=self.config.num_beams,
                    do_sample=self.config.do_sample,
                    temperature=self.config.temperature
                )
                
                # Decode predictions
                predictions = self._decode_tokens(generated_ids)
                references = batch['all_answers']  # List of lists
                questions = batch['question']
                
                all_predictions.extend(predictions)
                all_references.extend(references)
                all_questions.extend(questions)
                
                # Update progress bar
                progress_bar.set_postfix({'loss': f'{total_loss / (progress_bar.n + 1):.4f}'})
        
        # Compute metrics
        avg_loss = total_loss / len(self.val_loader)
        perplexity = np.exp(min(avg_loss, 100))
        
        # Update NLG metrics
        bleu_metric.update(all_predictions, all_references)
        rouge_metric.update(all_predictions, all_references)
        cider_metric.update(all_predictions, all_references)
        prf_metric.update(all_predictions, all_references)
        exact_match.update(all_predictions, all_references)
        
        # Try METEOR (may fail without NLTK data)
        try:
            meteor_metric.update(all_predictions, all_references)
            meteor_score = meteor_metric.compute().value
        except:
            meteor_score = 0.0
        
        bleu_result = bleu_metric.compute()
        rouge_result = rouge_metric.compute()
        cider_result = cider_metric.compute()
        prf_result = prf_metric.compute()
        exact_match_result = exact_match.compute()
        
        # Display samples
        self._display_samples(all_questions, all_predictions, all_references)
        
        # Log metrics
        metrics = {
            'loss': avg_loss,
            'perplexity': perplexity,
            'bleu': bleu_result.value,
            'meteor': meteor_score,
            'rouge_l': rouge_result.value,
            'cider': cider_result.value,
            'exact_match': exact_match_result.value,
            'precision': prf_result.metadata.get('precision', 0.0),
            'recall': prf_result.metadata.get('recall', 0.0),
            'f1': prf_result.value
        }
        
        self._log_metrics(metrics)
        
        return metrics
    
    def _decode_tokens(self, token_ids: torch.Tensor) -> List[str]:
        """Decode token IDs to strings."""
        predictions = []
        
        for ids in token_ids:
            # Remove special tokens
            ids_list = ids.tolist()
            
            # Find EOS and truncate
            eos_id = self.model.config.eos_token_id
            if eos_id in ids_list:
                ids_list = ids_list[:ids_list.index(eos_id)]
            
            # Remove BOS
            bos_id = self.model.config.bos_token_id
            if ids_list and ids_list[0] == bos_id:
                ids_list = ids_list[1:]
            
            # Remove PAD
            pad_id = self.model.config.pad_token_id
            ids_list = [i for i in ids_list if i != pad_id]
            
            # Decode
            if hasattr(self.tokenizer, 'decode'):
                text = self.tokenizer.decode(ids_list, skip_special_tokens=True)
            else:
                # Fallback for wrapped tokenizers
                text = self.tokenizer.tokenizer.decode(ids_list, skip_special_tokens=True)
            
            predictions.append(text.strip())
        
        return predictions
    
    def _display_samples(
        self, 
        questions: List[str], 
        predictions: List[str], 
        references: List[List[str]]
    ):
        """Display sample predictions."""
        self.logger.info("")
        self.logger.subsection("🔍 Sample Predictions (Generative)")
        
        num_samples = min(self.config.num_samples_to_display, len(questions))
        indices = np.random.choice(len(questions), num_samples, replace=False)
        
        for i, idx in enumerate(indices, 1):
            question = questions[idx]
            pred = predictions[idx]
            refs = references[idx]
            
            # Check if prediction matches any reference
            is_match = any(pred.lower().strip() == r.lower().strip() for r in refs)
            status = "✓" if is_match else "✗"
            
            self.logger.info(f"")
            self.logger.info(f"  [{status}] Sample {i}:")
            self.logger.info(f"      📌 Câu hỏi: \"{question}\"")
            self.logger.info(f"      🤖 Mô hình sinh: \"{pred}\"")
            self.logger.info(f"      ✅ Ground truth ({len(refs)}):")
            for j, ref in enumerate(refs[:3], 1):  # Show first 3
                match_indicator = "←" if pred.lower().strip() == ref.lower().strip() else ""
                self.logger.info(f"          {j}. \"{ref}\" {match_indicator}")
        
        self.logger.info("")
    
    def _log_metrics(self, metrics: Dict[str, float]):
        """Log metrics in formatted way."""
        self.logger.info("")
        self.logger.subsection("📊 Generative VQA Metrics")
        self.logger.info("=" * 60)
        
        self.logger.info("  📉 Loss Metrics:")
        self.logger.info(f"      • Loss: {metrics['loss']:.4f}")
        self.logger.info(f"      • Perplexity: {metrics['perplexity']:.2f}")
        
        self.logger.info("")
        self.logger.info("  📝 NLG Metrics:")
        self.logger.info(f"      • BLEU-4: {metrics['bleu']:.4f}")
        self.logger.info(f"      • METEOR: {metrics['meteor']:.4f}")
        self.logger.info(f"      • ROUGE-L: {metrics['rouge_l']:.4f}")
        self.logger.info(f"      • CIDEr: {metrics['cider']:.4f}")
        
        self.logger.info("")
        self.logger.info("  🎯 Matching Metrics:")
        self.logger.info(f"      • Exact Match: {metrics['exact_match']:.4f}")
        self.logger.info(f"      • Precision: {metrics['precision']:.4f}")
        self.logger.info(f"      • Recall: {metrics['recall']:.4f}")
        self.logger.info(f"      • F1: {metrics['f1']:.4f}")
        
        self.logger.info("=" * 60)
        self.logger.info("")
    
    def _move_batch_to_device(self, batch: Dict) -> Dict:
        """Move batch tensors to device."""
        return {
            k: v.to(self.device) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }
    
    def _save_checkpoint(self, is_best: bool, metrics: Dict[str, float]):
        """Save checkpoint."""
        checkpoint_dir = Path(self.config.checkpoint_dir)
        
        checkpoint = {
            'epoch': self.current_epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'best_metric': self.best_metric,
            'metrics': metrics,
            'config': self.model.config,
            'training_config': self.config
        }
        
        # Save epoch checkpoint
        if self.config.save_every_epoch:
            epoch_path = checkpoint_dir / f"checkpoint_epoch_{self.current_epoch}.pt"
            torch.save(checkpoint, epoch_path)
            self.logger.info(f"Checkpoint saved: {epoch_path}")
        
        # Save best model
        if is_best and self.config.save_best:
            best_path = checkpoint_dir / "best_generative_model.pt"
            torch.save(checkpoint, best_path)
            self.best_model_path = str(best_path)
            self.logger.success(f"Best model saved: {best_path}")
            self.logger.info(f"  {self.config.metric_for_best}: {self.best_metric:.4f}")
    
    def _log_epoch_summary(self, train_metrics: Dict, val_metrics: Optional[Dict]):
        """Log epoch summary."""
        self.logger.subsection(f"Epoch {self.current_epoch} Summary")
        
        self.logger.info("Training:")
        for k, v in train_metrics.items():
            self.logger.info(f"    {k}: {v:.4f}" if isinstance(v, float) else f"    {k}: {v}")
        
        if val_metrics:
            self.logger.info("Validation:")
            for k, v in val_metrics.items():
                self.logger.info(f"    {k}: {v:.4f}" if isinstance(v, float) else f"    {k}: {v}")
        
        self.logger.info(f"Best {self.config.metric_for_best}: {self.best_metric:.4f}")
        
        if self.device.type == 'cuda':
            mem = torch.cuda.memory_allocated(self.device) / 1024**3
            self.logger.info(f"GPU Memory: {mem:.2f} GB")
    
    def _final_evaluation(self) -> Dict[str, float]:
        """Run final evaluation."""
        self.logger.subsection("Final Evaluation")
        
        if self.best_model_path and os.path.exists(self.best_model_path):
            self.logger.info(f"Loading best model from: {self.best_model_path}")
            checkpoint = torch.load(self.best_model_path, map_location=self.device, weights_only=False)
            self.model.load_state_dict(checkpoint['model_state_dict'])
        
        final_metrics = self._validate_epoch()
        
        self.logger.success("Final Results:")
        for k, v in final_metrics.items():
            self.logger.info(f"    {k}: {v:.4f}" if isinstance(v, float) else f"    {k}: {v}")
        
        return final_metrics
