"""
Training Pipeline Module
Handles model training, evaluation, and checkpointing with comprehensive logging.
"""

import os
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from datetime import datetime

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from torch.cuda.amp import GradScaler
from torch.amp import autocast
from tqdm import tqdm

from src.core.pipeline_logger import get_pipeline_logger, PipelineLogger


@dataclass
class TrainingPipelineConfig:
    """Configuration for training pipeline."""
    
    # Training settings
    num_epochs: int = 20
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0
    
    # Optimizer
    optimizer_name: str = "adamw"  # adam, adamw, sgd
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    betas: Tuple[float, float] = (0.9, 0.999)
    
    # Scheduler
    scheduler_name: str = "cosine"  # cosine, linear, step
    warmup_ratio: float = 0.1
    warmup_steps: int = 0
    
    # Mixed precision
    use_amp: bool = True
    amp_dtype: str = "float16"  # float16, bfloat16
    
    # Early stopping
    early_stopping: bool = True
    patience: int = 5
    min_delta: float = 0.001
    
    # Checkpointing
    checkpoint_dir: str = "checkpoints"
    save_best: bool = True
    save_every_epoch: bool = True
    metric_for_best: str = "accuracy"
    
    # Logging
    log_interval: int = 50  # Log every N steps
    eval_interval: int = 1  # Evaluate every N epochs
    
    # Reproducibility
    seed: int = 42


@dataclass
class TrainingPipelineOutput:
    """Output from training pipeline."""
    
    best_model_path: str
    best_metric: float
    final_metrics: Dict[str, float]
    
    training_history: List[Dict[str, float]]
    validation_history: List[Dict[str, float]]
    
    total_epochs: int
    total_steps: int
    training_time: float


class TrainingPipeline:
    """
    Complete training pipeline for VQA.
    
    Handles:
    - Optimizer and scheduler setup
    - Training loop with gradient accumulation
    - Mixed precision training
    - Validation and metrics computation
    - Early stopping
    - Checkpointing
    - Comprehensive logging
    
    All steps are logged with detailed validation output.
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: Optional[TrainingPipelineConfig] = None,
        logger: Optional[PipelineLogger] = None,
        device: Optional[torch.device] = None,
        model_config: Optional[Any] = None,
        vocabulary: Optional[Dict[str, int]] = None,
        id2answer: Optional[Dict[int, str]] = None,
        num_samples_to_display: int = 5
    ):
        """
        Initialize training pipeline.
        
        Args:
            model: VQA model to train
            train_loader: Training data loader
            val_loader: Validation data loader
            config: Training pipeline configuration
            logger: Pipeline logger instance
            device: Training device
            model_config: Model configuration (for saving in checkpoint)
            vocabulary: Answer vocabulary (for saving in checkpoint)
            id2answer: Mapping from answer ID to answer text (for displaying predictions)
            num_samples_to_display: Number of sample predictions to display during validation
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config or TrainingPipelineConfig()
        self.logger = logger or get_pipeline_logger()
        self.model_config = model_config
        self.vocabulary = vocabulary
        self.id2answer = id2answer or {}
        self.num_samples_to_display = num_samples_to_display
        
        # Device
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
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
        
    def run(self) -> TrainingPipelineOutput:
        """
        Execute the complete training pipeline.
        
        Returns:
            TrainingPipelineOutput with training results
        """
        self.logger.start_stage("TRAINING PIPELINE")
        
        start_time = time.time()
        
        try:
            # Step 1: Set seed
            self._set_seed()
            
            # Step 2: Setup optimizer
            self._setup_optimizer()
            
            # Step 3: Setup scheduler
            self._setup_scheduler()
            
            # Step 4: Setup mixed precision
            self._setup_amp()
            
            # Step 5: Setup checkpointing
            self._setup_checkpointing()
            
            # Step 6: Log training configuration
            self._log_training_config()
            
            # Step 7: Training loop
            self._training_loop()
            
            # Step 8: Final evaluation
            final_metrics = self._final_evaluation()
            
            training_time = time.time() - start_time
            
            self.logger.end_stage("TRAINING PIPELINE", success=True)
            
            return TrainingPipelineOutput(
                best_model_path=self.best_model_path,
                best_metric=self.best_metric,
                final_metrics=final_metrics,
                training_history=self.training_history,
                validation_history=self.validation_history,
                total_epochs=self.current_epoch,
                total_steps=self.global_step,
                training_time=training_time
            )
            
        except Exception as e:
            self.logger.error(f"Training pipeline failed: {e}")
            self.logger.end_stage("TRAINING PIPELINE", success=False)
            raise
            
    def _set_seed(self):
        """Set random seed for reproducibility."""
        self.logger.subsection("Setting Random Seed")
        
        import random
        import numpy as np
        
        seed = self.config.seed
        
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            
        self.logger.key_value("Seed", seed)
        self.logger.success("Random seed set")
        
    def _setup_optimizer(self):
        """Setup optimizer."""
        self.logger.subsection("Setting Up Optimizer")
        
        # Get parameter groups (different LR for pretrained vs new layers)
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
        
        param_count = sum(len(g['params']) for g in optimizer_grouped_params)
        self.logger.key_value("Parameter groups", len(optimizer_grouped_params))
        self.logger.key_value("Total trainable tensors", param_count)
        
        # Create optimizer
        optimizer_name = self.config.optimizer_name.lower()
        
        if optimizer_name == "adamw":
            self.optimizer = torch.optim.AdamW(
                optimizer_grouped_params,
                lr=self.config.learning_rate,
                betas=self.config.betas,
                weight_decay=self.config.weight_decay
            )
        elif optimizer_name == "adam":
            self.optimizer = torch.optim.Adam(
                optimizer_grouped_params,
                lr=self.config.learning_rate,
                betas=self.config.betas
            )
        elif optimizer_name == "sgd":
            self.optimizer = torch.optim.SGD(
                optimizer_grouped_params,
                lr=self.config.learning_rate,
                momentum=0.9,
                weight_decay=self.config.weight_decay
            )
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_name}")
            
        self.logger.key_value("Optimizer", self.optimizer.__class__.__name__)
        self.logger.key_value("Learning rate", self.config.learning_rate)
        self.logger.key_value("Weight decay", self.config.weight_decay)
        self.logger.success("Optimizer configured")
        
    def _setup_scheduler(self):
        """Setup learning rate scheduler."""
        self.logger.subsection("Setting Up LR Scheduler")
        
        # Calculate total steps
        total_steps = len(self.train_loader) * self.config.num_epochs
        total_steps = total_steps // self.config.gradient_accumulation_steps
        
        # Warmup steps
        if self.config.warmup_steps > 0:
            warmup_steps = self.config.warmup_steps
        else:
            warmup_steps = int(total_steps * self.config.warmup_ratio)
            
        self.logger.key_value("Total training steps", total_steps)
        self.logger.key_value("Warmup steps", warmup_steps)
        
        scheduler_name = self.config.scheduler_name.lower()
        
        if scheduler_name == "cosine":
            from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, CosineAnnealingLR
            
            # Use linear warmup + cosine decay
            def lr_lambda(current_step):
                if current_step < warmup_steps:
                    return float(current_step) / float(max(1, warmup_steps))
                progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
                return max(0.0, 0.5 * (1.0 + torch.cos(torch.tensor(progress * 3.14159)).item()))
                
            self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)
            
        elif scheduler_name == "linear":
            def lr_lambda(current_step):
                if current_step < warmup_steps:
                    return float(current_step) / float(max(1, warmup_steps))
                return max(0.0, float(total_steps - current_step) / float(max(1, total_steps - warmup_steps)))
                
            self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)
            
        elif scheduler_name == "step":
            self.scheduler = torch.optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=total_steps // 3,
                gamma=0.1
            )
        else:
            self.scheduler = None
            self.logger.warning(f"Unknown scheduler: {scheduler_name}, no scheduler used")
            return
            
        self.logger.key_value("Scheduler", self.config.scheduler_name)
        self.logger.success("LR scheduler configured")
        
    def _setup_amp(self):
        """Setup automatic mixed precision."""
        self.logger.subsection("Setting Up Mixed Precision")
        
        if self.config.use_amp and self.device.type == "cuda":
            self.scaler = GradScaler()
            self.logger.key_value("AMP enabled", True)
            self.logger.key_value("AMP dtype", self.config.amp_dtype)
            self.logger.success("Mixed precision training enabled")
        else:
            self.scaler = None
            self.logger.info("Mixed precision training disabled")
            
    def _setup_checkpointing(self):
        """Setup checkpoint directory."""
        self.logger.subsection("Setting Up Checkpointing")
        
        checkpoint_dir = Path(self.config.checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger.key_value("Checkpoint directory", str(checkpoint_dir))
        self.logger.key_value("Save best only", self.config.save_best)
        self.logger.key_value("Save every epoch", self.config.save_every_epoch)
        self.logger.key_value("Best metric", self.config.metric_for_best)
        
        self.logger.success("Checkpointing configured")
        
    def _log_training_config(self):
        """Log full training configuration."""
        self.logger.subsection("Training Configuration Summary")
        
        # Training parameters
        self.logger.info("Training Parameters:")
        self.logger.key_value("Epochs", self.config.num_epochs)
        self.logger.key_value("Batch size", self.train_loader.batch_size)
        self.logger.key_value("Gradient accumulation", self.config.gradient_accumulation_steps)
        self.logger.key_value("Effective batch size", 
                            self.train_loader.batch_size * self.config.gradient_accumulation_steps)
        self.logger.key_value("Max grad norm", self.config.max_grad_norm)
        
        # Data info
        self.logger.info("Data Info:")
        self.logger.key_value("Training batches", len(self.train_loader))
        self.logger.key_value("Validation batches", len(self.val_loader))
        self.logger.key_value("Training samples", len(self.train_loader.dataset))
        self.logger.key_value("Validation samples", len(self.val_loader.dataset))
        
        # Early stopping
        self.logger.info("Early Stopping:")
        self.logger.key_value("Enabled", self.config.early_stopping)
        if self.config.early_stopping:
            self.logger.key_value("Patience", self.config.patience)
            self.logger.key_value("Min delta", self.config.min_delta)
            
    def _training_loop(self):
        """Main training loop."""
        self.logger.subsection("Starting Training Loop")
        
        for epoch in range(self.config.num_epochs):
            self.current_epoch = epoch + 1
            
            self.logger.section(f"EPOCH {self.current_epoch}/{self.config.num_epochs}")
            
            # Train one epoch
            train_metrics = self._train_epoch()
            self.training_history.append(train_metrics)
            
            # Validation
            if self.current_epoch % self.config.eval_interval == 0:
                val_metrics = self._validate_epoch()
                self.validation_history.append(val_metrics)
                
                # Check for improvement
                current_metric = val_metrics.get(self.config.metric_for_best, 0)
                is_best = current_metric > self.best_metric + self.config.min_delta
                
                if is_best:
                    self.best_metric = current_metric
                    self.early_stop_counter = 0
                    self.logger.success(f"New best {self.config.metric_for_best}: {self.best_metric:.4f}")
                else:
                    self.early_stop_counter += 1
                    self.logger.info(f"No improvement. Early stop counter: {self.early_stop_counter}/{self.config.patience}")
                    
                # Save checkpoint
                self._save_checkpoint(is_best=is_best, metrics=val_metrics)
                
                # Early stopping check
                if self.config.early_stopping and self.early_stop_counter >= self.config.patience:
                    self.logger.warning(f"Early stopping triggered after {self.current_epoch} epochs")
                    self.should_stop = True
                    break
                    
            # Log epoch summary
            self._log_epoch_summary(train_metrics, val_metrics if self.current_epoch % self.config.eval_interval == 0 else None)
            
        self.logger.success(f"Training completed after {self.current_epoch} epochs")
        
    def _train_epoch(self) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        
        progress_bar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch} [Train]")
        
        for step, batch in enumerate(progress_bar):
            # Move batch to device
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                    for k, v in batch.items()}
            
            # Forward pass
            if self.scaler is not None:
                with autocast(device_type='cuda', dtype=torch.float16, enabled=True):
                    outputs = self.model(
                        pixel_values=batch['image'],
                        input_ids=batch['input_ids'],
                        attention_mask=batch['attention_mask'],
                        labels=batch['label']
                    )
                    loss = outputs.loss
                    
                # Backward pass with scaling
                scaled_loss = loss / self.config.gradient_accumulation_steps
                self.scaler.scale(scaled_loss).backward()
                
            else:
                outputs = self.model(
                    pixel_values=batch['image'],
                    input_ids=batch['input_ids'],
                    attention_mask=batch['attention_mask'],
                    labels=batch['label']
                )
                loss = outputs.loss
                
                # Backward pass
                scaled_loss = loss / self.config.gradient_accumulation_steps
                scaled_loss.backward()
                
            # Accumulate metrics
            total_loss += loss.item()
            
            if hasattr(outputs, 'logits'):
                predictions = outputs.logits.argmax(dim=-1)
                total_correct += (predictions == batch['label']).sum().item()
                total_samples += batch['label'].size(0)
                
            # Gradient accumulation step
            if (step + 1) % self.config.gradient_accumulation_steps == 0:
                # Gradient clipping
                if self.scaler is not None:
                    self.scaler.unscale_(self.optimizer)
                    
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
                
                # Optimizer step
                if self.scaler is not None:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()
                    
                self.optimizer.zero_grad()
                
                # Scheduler step
                if self.scheduler is not None:
                    self.scheduler.step()
                    
                self.global_step += 1
                
            # Update progress bar
            current_lr = self.optimizer.param_groups[0]['lr']
            avg_loss = total_loss / (step + 1)
            accuracy = total_correct / total_samples if total_samples > 0 else 0
            
            progress_bar.set_postfix({
                'loss': f'{avg_loss:.4f}',
                'acc': f'{accuracy:.4f}',
                'lr': f'{current_lr:.2e}'
            })
            
            # Periodic logging
            if (step + 1) % self.config.log_interval == 0:
                self.logger.debug(f"Step {step+1}/{len(self.train_loader)} - Loss: {avg_loss:.4f}, Acc: {accuracy:.4f}, LR: {current_lr:.2e}")
                
        # Return epoch metrics
        return {
            'loss': total_loss / len(self.train_loader),
            'accuracy': total_correct / total_samples if total_samples > 0 else 0,
            'learning_rate': self.optimizer.param_groups[0]['lr']
        }
        
    def _validate_epoch(self, show_samples: bool = True) -> Dict[str, float]:
        """Validate for one epoch.
        
        Args:
            show_samples: Whether to display sample predictions
            
        Returns:
            Dictionary of validation metrics
        """
        self.model.eval()
        
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        
        all_predictions = []
        all_labels = []
        
        # Store sample predictions for display (separate correct and incorrect)
        correct_samples = []
        incorrect_samples = []
        max_samples_each = self.num_samples_to_display // 2 + 1
        
        progress_bar = tqdm(self.val_loader, desc=f"Epoch {self.current_epoch} [Val]")
        
        with torch.no_grad():
            for step, batch in enumerate(progress_bar):
                # Extract non-tensor data before moving to device
                questions = batch.get('question', [])
                ground_truths = batch.get('ground_truth', [])
                
                # Move batch to device (only tensors)
                batch_tensors = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                                for k, v in batch.items()}
                
                # Forward pass
                outputs = self.model(
                    pixel_values=batch_tensors['image'],
                    input_ids=batch_tensors['input_ids'],
                    attention_mask=batch_tensors['attention_mask'],
                    labels=batch_tensors['label']
                )
                
                loss = outputs.loss
                total_loss += loss.item()
                
                if hasattr(outputs, 'logits'):
                    predictions = outputs.logits.argmax(dim=-1)
                    total_correct += (predictions == batch_tensors['label']).sum().item()
                    total_samples += batch_tensors['label'].size(0)
                    
                    pred_ids = predictions.cpu().tolist()
                    label_ids = batch_tensors['label'].cpu().tolist()
                    
                    all_predictions.extend(pred_ids)
                    all_labels.extend(label_ids)
                    
                    # Collect sample predictions - try to get diverse samples (correct & incorrect)
                    if show_samples and self.id2answer:
                        for i in range(len(pred_ids)):
                            question = questions[i] if i < len(questions) else "N/A"
                            ground_truth = ground_truths[i] if i < len(ground_truths) else "N/A"
                            pred_answer = self.id2answer.get(pred_ids[i], f"<ID:{pred_ids[i]}>")
                            is_correct = pred_ids[i] == label_ids[i]
                            
                            # Mark if ground truth is not in vocabulary
                            gt_in_vocab = label_ids[i] != 0 or ground_truth == "<unk>"
                            
                            sample = {
                                'question': question,
                                'predicted': pred_answer,
                                'ground_truth': ground_truth,
                                'correct': is_correct,
                                'gt_in_vocab': gt_in_vocab
                            }
                            
                            # Prioritize samples where ground truth IS in vocabulary (more meaningful)
                            if label_ids[i] != 0:  # ground truth is in vocabulary
                                if is_correct and len(correct_samples) < max_samples_each:
                                    correct_samples.append(sample)
                                elif not is_correct and len(incorrect_samples) < max_samples_each:
                                    incorrect_samples.append(sample)
                            
                            # Stop collecting if we have enough of both
                            if len(correct_samples) >= max_samples_each and len(incorrect_samples) >= max_samples_each:
                                break
                        
                        # If we don't have enough samples with ground truth in vocab, 
                        # also collect some with <unk> ground truth
                        if len(correct_samples) + len(incorrect_samples) < self.num_samples_to_display:
                            for i in range(len(pred_ids)):
                                if label_ids[i] == 0:  # ground truth is <unk>
                                    question = questions[i] if i < len(questions) else "N/A"
                                    ground_truth = ground_truths[i] if i < len(ground_truths) else "N/A"
                                    pred_answer = self.id2answer.get(pred_ids[i], f"<ID:{pred_ids[i]}>")
                                    is_correct = pred_ids[i] == label_ids[i]
                                    
                                    sample = {
                                        'question': question,
                                        'predicted': pred_answer,
                                        'ground_truth': f"{ground_truth} (ngoài vocab)",
                                        'correct': is_correct,
                                        'gt_in_vocab': False
                                    }
                                    
                                    if is_correct and len(correct_samples) < max_samples_each:
                                        correct_samples.append(sample)
                                    elif not is_correct and len(incorrect_samples) < max_samples_each:
                                        incorrect_samples.append(sample)
                                    
                                    if len(correct_samples) + len(incorrect_samples) >= self.num_samples_to_display:
                                        break
                    
                # Update progress bar
                steps_done = step + 1
                avg_loss = total_loss / steps_done
                accuracy = total_correct / total_samples if total_samples > 0 else 0
                progress_bar.set_postfix({'loss': f'{avg_loss:.4f}', 'acc': f'{accuracy:.4f}'})
                
        # Calculate metrics
        avg_loss = total_loss / len(self.val_loader)
        accuracy = total_correct / total_samples if total_samples > 0 else 0
        
        # Display sample predictions (mix of correct and incorrect)
        if show_samples and (correct_samples or incorrect_samples):
            # Combine samples: show some correct, then some incorrect
            combined_samples = correct_samples[:max_samples_each] + incorrect_samples[:max_samples_each]
            self._display_sample_predictions(combined_samples[:self.num_samples_to_display])
        
        # Additional metrics
        metrics = {
            'loss': avg_loss,
            'accuracy': accuracy
        }
        
        return metrics
    
    def _display_sample_predictions(self, samples: List[Dict]):
        """Display sample predictions in a formatted way."""
        self.logger.info("")
        self.logger.subsection("Sample Predictions")
        
        for i, sample in enumerate(samples, 1):
            status = "✓" if sample['correct'] else "✗"
            self.logger.info(f"")
            self.logger.info(f"  [{status}] Sample {i}:")
            self.logger.info(f"      Câu hỏi: \"{sample['question']}\"")
            self.logger.info(f"      Mô hình trả lời: \"{sample['predicted']}\"")
            self.logger.info(f"      Đáp án đúng: \"{sample['ground_truth']}\"")
        
        correct_count = sum(1 for s in samples if s['correct'])
        self.logger.info(f"")
        self.logger.info(f"  Sample accuracy: {correct_count}/{len(samples)}")
        self.logger.info("")
        
    def _save_checkpoint(self, is_best: bool, metrics: Dict[str, float]):
        """Save model checkpoint."""
        checkpoint_dir = Path(self.config.checkpoint_dir)
        
        # Checkpoint data
        checkpoint = {
            'epoch': self.current_epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'best_metric': self.best_metric,
            'metrics': metrics,
            'config': self.config,
            'model_config': self.model_config,
            'vocabulary': self.vocabulary,
            'num_answers': len(self.vocabulary) if self.vocabulary else None
        }
        
        # Save latest
        if self.config.save_every_epoch:
            latest_path = checkpoint_dir / f"checkpoint_epoch_{self.current_epoch}.pt"
            torch.save(checkpoint, latest_path)
            self.logger.info(f"Checkpoint saved: {latest_path}")
            
        # Save best
        if is_best and self.config.save_best:
            best_path = checkpoint_dir / "best_model.pt"
            torch.save(checkpoint, best_path)
            self.best_model_path = str(best_path)
            self.logger.log_checkpoint(str(best_path), metrics, is_best=True)
            
    def _log_epoch_summary(self, train_metrics: Dict[str, float], val_metrics: Optional[Dict[str, float]]):
        """Log epoch summary."""
        self.logger.subsection(f"Epoch {self.current_epoch} Summary")
        
        self.logger.info("Training Metrics:")
        for name, value in train_metrics.items():
            self.logger.key_value(name, f"{value:.4f}" if isinstance(value, float) else value)
            
        if val_metrics:
            self.logger.info("Validation Metrics:")
            for name, value in val_metrics.items():
                self.logger.key_value(name, f"{value:.4f}" if isinstance(value, float) else value)
                
        self.logger.key_value("Best metric so far", f"{self.best_metric:.4f}")
        
        # GPU memory
        if self.device.type == "cuda":
            memory_allocated = torch.cuda.memory_allocated(self.device) / 1024**3
            self.logger.key_value("GPU memory", f"{memory_allocated:.2f} GB")
            
    def _final_evaluation(self) -> Dict[str, float]:
        """Run final evaluation."""
        self.logger.subsection("Final Evaluation")
        
        # Load best model if available
        if self.best_model_path and os.path.exists(self.best_model_path):
            self.logger.info(f"Loading best model from: {self.best_model_path}")
            checkpoint = torch.load(self.best_model_path, map_location=self.device, weights_only=False)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            
        # Final validation
        final_metrics = self._validate_epoch()
        
        self.logger.success("Final Evaluation Results:")
        for name, value in final_metrics.items():
            self.logger.key_value(name, f"{value:.4f}" if isinstance(value, float) else value)
            
        return final_metrics
