"""
Optimizers for VQA Training
Provides optimizer configurations and learning rate schedulers.
"""

from typing import Optional, List, Dict, Any, Union, Iterator
from dataclasses import dataclass, field
from enum import Enum
import math

import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler


class OptimizerType(Enum):
    """Enumeration of optimizer types."""
    ADAM = 'adam'
    ADAMW = 'adamw'
    SGD = 'sgd'
    RADAM = 'radam'
    LAMB = 'lamb'


class SchedulerType(Enum):
    """Enumeration of scheduler types."""
    CONSTANT = 'constant'
    LINEAR = 'linear'
    COSINE = 'cosine'
    COSINE_WARMUP = 'cosine_warmup'
    STEP = 'step'
    EXPONENTIAL = 'exponential'
    POLYNOMIAL = 'polynomial'
    ONE_CYCLE = 'one_cycle'


@dataclass
class OptimizerConfig:
    """
    Configuration for optimizer.
    
    Attributes:
        optimizer_type: Type of optimizer
        learning_rate: Initial learning rate
        weight_decay: Weight decay factor
        betas: Adam betas
        momentum: SGD momentum
        eps: Adam epsilon
        gradient_clip: Maximum gradient norm
        use_lookahead: Whether to use Lookahead wrapper
    """
    optimizer_type: str = 'adamw'
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    betas: tuple = (0.9, 0.999)
    momentum: float = 0.9
    eps: float = 1e-8
    gradient_clip: Optional[float] = 1.0
    use_lookahead: bool = False
    lookahead_k: int = 5
    lookahead_alpha: float = 0.5


@dataclass
class SchedulerConfig:
    """
    Configuration for learning rate scheduler.
    
    Attributes:
        scheduler_type: Type of scheduler
        num_warmup_steps: Number of warmup steps
        num_training_steps: Total training steps
        min_lr: Minimum learning rate
        num_cycles: Number of cosine cycles
        step_size: Step size for StepLR
        gamma: Decay factor
        power: Power for polynomial decay
    """
    scheduler_type: str = 'cosine_warmup'
    num_warmup_steps: int = 1000
    num_training_steps: int = 100000
    min_lr: float = 1e-7
    num_cycles: float = 0.5
    step_size: int = 10000
    gamma: float = 0.1
    power: float = 1.0


class WarmupScheduler(_LRScheduler):
    """
    Linear warmup scheduler.
    """
    
    def __init__(
        self,
        optimizer: Optimizer,
        num_warmup_steps: int,
        last_epoch: int = -1
    ):
        """
        Initialize warmup scheduler.
        
        Args:
            optimizer: Optimizer
            num_warmup_steps: Number of warmup steps
            last_epoch: Last epoch number
        """
        self.num_warmup_steps = num_warmup_steps
        super().__init__(optimizer, last_epoch)
    
    def get_lr(self):
        """Get learning rate."""
        if self.last_epoch < self.num_warmup_steps:
            warmup_factor = self.last_epoch / max(1, self.num_warmup_steps)
            return [base_lr * warmup_factor for base_lr in self.base_lrs]
        return self.base_lrs


class CosineWarmupScheduler(_LRScheduler):
    """
    Cosine annealing with warmup.
    """
    
    def __init__(
        self,
        optimizer: Optimizer,
        num_warmup_steps: int,
        num_training_steps: int,
        num_cycles: float = 0.5,
        min_lr: float = 0.0,
        last_epoch: int = -1
    ):
        """
        Initialize cosine warmup scheduler.
        
        Args:
            optimizer: Optimizer
            num_warmup_steps: Number of warmup steps
            num_training_steps: Total training steps
            num_cycles: Number of cosine cycles
            min_lr: Minimum learning rate
            last_epoch: Last epoch number
        """
        self.num_warmup_steps = num_warmup_steps
        self.num_training_steps = num_training_steps
        self.num_cycles = num_cycles
        self.min_lr = min_lr
        super().__init__(optimizer, last_epoch)
    
    def get_lr(self):
        """Get learning rate."""
        if self.last_epoch < self.num_warmup_steps:
            # Linear warmup
            warmup_factor = self.last_epoch / max(1, self.num_warmup_steps)
            return [base_lr * warmup_factor for base_lr in self.base_lrs]
        
        # Cosine annealing
        progress = (self.last_epoch - self.num_warmup_steps) / max(
            1, self.num_training_steps - self.num_warmup_steps
        )
        cosine_factor = 0.5 * (1.0 + math.cos(math.pi * self.num_cycles * 2.0 * progress))
        
        return [
            self.min_lr + (base_lr - self.min_lr) * cosine_factor
            for base_lr in self.base_lrs
        ]


class LinearWarmupScheduler(_LRScheduler):
    """
    Linear decay with warmup.
    """
    
    def __init__(
        self,
        optimizer: Optimizer,
        num_warmup_steps: int,
        num_training_steps: int,
        min_lr: float = 0.0,
        last_epoch: int = -1
    ):
        """
        Initialize linear warmup scheduler.
        
        Args:
            optimizer: Optimizer
            num_warmup_steps: Number of warmup steps
            num_training_steps: Total training steps
            min_lr: Minimum learning rate
            last_epoch: Last epoch number
        """
        self.num_warmup_steps = num_warmup_steps
        self.num_training_steps = num_training_steps
        self.min_lr = min_lr
        super().__init__(optimizer, last_epoch)
    
    def get_lr(self):
        """Get learning rate."""
        if self.last_epoch < self.num_warmup_steps:
            # Linear warmup
            warmup_factor = self.last_epoch / max(1, self.num_warmup_steps)
            return [base_lr * warmup_factor for base_lr in self.base_lrs]
        
        # Linear decay
        progress = (self.last_epoch - self.num_warmup_steps) / max(
            1, self.num_training_steps - self.num_warmup_steps
        )
        linear_factor = max(0.0, 1.0 - progress)
        
        return [
            self.min_lr + (base_lr - self.min_lr) * linear_factor
            for base_lr in self.base_lrs
        ]


class PolynomialWarmupScheduler(_LRScheduler):
    """
    Polynomial decay with warmup.
    """
    
    def __init__(
        self,
        optimizer: Optimizer,
        num_warmup_steps: int,
        num_training_steps: int,
        power: float = 1.0,
        min_lr: float = 0.0,
        last_epoch: int = -1
    ):
        """
        Initialize polynomial warmup scheduler.
        
        Args:
            optimizer: Optimizer
            num_warmup_steps: Number of warmup steps
            num_training_steps: Total training steps
            power: Polynomial power
            min_lr: Minimum learning rate
            last_epoch: Last epoch number
        """
        self.num_warmup_steps = num_warmup_steps
        self.num_training_steps = num_training_steps
        self.power = power
        self.min_lr = min_lr
        super().__init__(optimizer, last_epoch)
    
    def get_lr(self):
        """Get learning rate."""
        if self.last_epoch < self.num_warmup_steps:
            warmup_factor = self.last_epoch / max(1, self.num_warmup_steps)
            return [base_lr * warmup_factor for base_lr in self.base_lrs]
        
        # Polynomial decay
        progress = (self.last_epoch - self.num_warmup_steps) / max(
            1, self.num_training_steps - self.num_warmup_steps
        )
        poly_factor = (1.0 - progress) ** self.power
        
        return [
            self.min_lr + (base_lr - self.min_lr) * poly_factor
            for base_lr in self.base_lrs
        ]


class Lookahead(Optimizer):
    """
    Lookahead optimizer wrapper.
    Improves optimization by maintaining slow and fast weights.
    """
    
    def __init__(
        self,
        optimizer: Optimizer,
        k: int = 5,
        alpha: float = 0.5
    ):
        """
        Initialize Lookahead.
        
        Args:
            optimizer: Base optimizer
            k: Number of steps before slow weight update
            alpha: Interpolation factor
        """
        self.optimizer = optimizer
        self.k = k
        self.alpha = alpha
        self.param_groups = optimizer.param_groups
        
        # Initialize slow weights
        self.state = defaultdict(dict)
        for group in self.param_groups:
            for p in group['params']:
                if p.requires_grad:
                    self.state[p]['slow_buffer'] = p.data.clone()
        
        self._step_count = 0
    
    def step(self, closure=None):
        """Perform optimization step."""
        loss = self.optimizer.step(closure)
        self._step_count += 1
        
        if self._step_count % self.k == 0:
            # Update slow weights
            for group in self.param_groups:
                for p in group['params']:
                    if p.requires_grad:
                        slow = self.state[p]['slow_buffer']
                        slow.add_(p.data - slow, alpha=self.alpha)
                        p.data.copy_(slow)
        
        return loss
    
    def zero_grad(self):
        """Zero gradients."""
        self.optimizer.zero_grad()
    
    @property
    def defaults(self):
        """Get optimizer defaults."""
        return self.optimizer.defaults


from collections import defaultdict


class LayerWiseLearningRateDecay:
    """
    Layer-wise learning rate decay for transformer models.
    """
    
    def __init__(
        self,
        base_lr: float,
        decay_rate: float = 0.9,
        num_layers: int = 12
    ):
        """
        Initialize layer-wise LR decay.
        
        Args:
            base_lr: Base learning rate
            decay_rate: Decay rate per layer
            num_layers: Number of layers
        """
        self.base_lr = base_lr
        self.decay_rate = decay_rate
        self.num_layers = num_layers
    
    def get_layer_lr(self, layer_idx: int) -> float:
        """
        Get learning rate for layer.
        
        Args:
            layer_idx: Layer index (0 = bottom, num_layers-1 = top)
            
        Returns:
            Learning rate for layer
        """
        return self.base_lr * (self.decay_rate ** (self.num_layers - layer_idx - 1))
    
    def get_param_groups(
        self,
        model: nn.Module,
        weight_decay: float = 0.01
    ) -> List[Dict[str, Any]]:
        """
        Get parameter groups with layer-wise LR.
        
        Args:
            model: Model
            weight_decay: Weight decay
            
        Returns:
            List of parameter groups
        """
        param_groups = []
        
        # Group parameters by layer
        layer_params = defaultdict(list)
        other_params = []
        
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            
            # Try to extract layer index from name
            layer_idx = None
            for pattern in ['layer.', 'layers.', 'block.', 'blocks.']:
                if pattern in name:
                    try:
                        idx_str = name.split(pattern)[1].split('.')[0]
                        layer_idx = int(idx_str)
                        break
                    except (IndexError, ValueError):
                        pass
            
            if layer_idx is not None:
                layer_params[layer_idx].append(param)
            else:
                other_params.append(param)
        
        # Create param groups
        for layer_idx, params in sorted(layer_params.items()):
            lr = self.get_layer_lr(layer_idx)
            param_groups.append({
                'params': params,
                'lr': lr,
                'weight_decay': weight_decay
            })
        
        # Other parameters use base LR
        if other_params:
            param_groups.append({
                'params': other_params,
                'lr': self.base_lr,
                'weight_decay': weight_decay
            })
        
        return param_groups


def create_optimizer(
    model: nn.Module,
    config: OptimizerConfig,
    param_groups: Optional[List[Dict]] = None
) -> Optimizer:
    """
    Create optimizer from config.
    
    Args:
        model: Model
        config: Optimizer configuration
        param_groups: Optional custom parameter groups
        
    Returns:
        Optimizer instance
    """
    if param_groups is None:
        # Separate params with/without weight decay
        decay_params = []
        no_decay_params = []
        
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            
            # No weight decay for bias and normalization
            if 'bias' in name or 'norm' in name or 'ln' in name:
                no_decay_params.append(param)
            else:
                decay_params.append(param)
        
        param_groups = [
            {'params': decay_params, 'weight_decay': config.weight_decay},
            {'params': no_decay_params, 'weight_decay': 0.0}
        ]
    
    optimizer_type = config.optimizer_type.lower()
    
    if optimizer_type == 'adam':
        optimizer = torch.optim.Adam(
            param_groups,
            lr=config.learning_rate,
            betas=config.betas,
            eps=config.eps
        )
    
    elif optimizer_type == 'adamw':
        optimizer = torch.optim.AdamW(
            param_groups,
            lr=config.learning_rate,
            betas=config.betas,
            eps=config.eps
        )
    
    elif optimizer_type == 'sgd':
        optimizer = torch.optim.SGD(
            param_groups,
            lr=config.learning_rate,
            momentum=config.momentum,
            weight_decay=config.weight_decay
        )
    
    elif optimizer_type == 'radam':
        try:
            optimizer = torch.optim.RAdam(
                param_groups,
                lr=config.learning_rate,
                betas=config.betas,
                eps=config.eps
            )
        except AttributeError:
            # Fallback for older PyTorch versions
            optimizer = torch.optim.Adam(
                param_groups,
                lr=config.learning_rate,
                betas=config.betas,
                eps=config.eps
            )
    
    else:
        raise ValueError(f"Unknown optimizer type: {optimizer_type}")
    
    # Apply Lookahead if requested
    if config.use_lookahead:
        optimizer = Lookahead(
            optimizer,
            k=config.lookahead_k,
            alpha=config.lookahead_alpha
        )
    
    return optimizer


def create_scheduler(
    optimizer: Optimizer,
    config: SchedulerConfig
) -> _LRScheduler:
    """
    Create learning rate scheduler from config.
    
    Args:
        optimizer: Optimizer
        config: Scheduler configuration
        
    Returns:
        Scheduler instance
    """
    scheduler_type = config.scheduler_type.lower()
    
    if scheduler_type == 'constant':
        scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer, factor=1.0)
    
    elif scheduler_type == 'linear':
        scheduler = LinearWarmupScheduler(
            optimizer,
            num_warmup_steps=config.num_warmup_steps,
            num_training_steps=config.num_training_steps,
            min_lr=config.min_lr
        )
    
    elif scheduler_type == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config.num_training_steps,
            eta_min=config.min_lr
        )
    
    elif scheduler_type == 'cosine_warmup':
        scheduler = CosineWarmupScheduler(
            optimizer,
            num_warmup_steps=config.num_warmup_steps,
            num_training_steps=config.num_training_steps,
            num_cycles=config.num_cycles,
            min_lr=config.min_lr
        )
    
    elif scheduler_type == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=config.step_size,
            gamma=config.gamma
        )
    
    elif scheduler_type == 'exponential':
        scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer,
            gamma=config.gamma
        )
    
    elif scheduler_type == 'polynomial':
        scheduler = PolynomialWarmupScheduler(
            optimizer,
            num_warmup_steps=config.num_warmup_steps,
            num_training_steps=config.num_training_steps,
            power=config.power,
            min_lr=config.min_lr
        )
    
    elif scheduler_type == 'one_cycle':
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=optimizer.defaults['lr'] * 10,
            total_steps=config.num_training_steps,
            pct_start=config.num_warmup_steps / config.num_training_steps
        )
    
    else:
        raise ValueError(f"Unknown scheduler type: {scheduler_type}")
    
    return scheduler


def get_gradient_norm(model: nn.Module) -> float:
    """
    Compute total gradient norm.
    
    Args:
        model: Model
        
    Returns:
        Total gradient norm
    """
    total_norm = 0.0
    
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    
    return total_norm ** 0.5


def clip_gradients(
    model: nn.Module,
    max_norm: float,
    norm_type: float = 2.0
) -> float:
    """
    Clip gradients by norm.
    
    Args:
        model: Model
        max_norm: Maximum gradient norm
        norm_type: Type of norm
        
    Returns:
        Total gradient norm before clipping
    """
    return torch.nn.utils.clip_grad_norm_(
        model.parameters(),
        max_norm,
        norm_type=norm_type
    ).item()


__all__ = [
    'OptimizerType',
    'SchedulerType',
    'OptimizerConfig',
    'SchedulerConfig',
    'WarmupScheduler',
    'CosineWarmupScheduler',
    'LinearWarmupScheduler',
    'PolynomialWarmupScheduler',
    'Lookahead',
    'LayerWiseLearningRateDecay',
    'create_optimizer',
    'create_scheduler',
    'get_gradient_norm',
    'clip_gradients'
]
