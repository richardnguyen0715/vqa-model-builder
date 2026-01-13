"""
Data Augmentation Module
Provides image and text augmentation utilities for VQA training.
"""

import random
import logging
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torchvision import transforms

logger = logging.getLogger(__name__)


# =============================================================================
# Image Augmentation
# =============================================================================

class ImageAugmentation:
    """
    Image augmentation pipeline for VQA.
    
    Applies various augmentations to improve model generalization.
    Supports both training and evaluation modes.
    
    Attributes:
        mode: Augmentation mode (train, eval, or custom).
        image_size: Target image size.
        normalize_mean: Normalization mean values.
        normalize_std: Normalization std values.
    """
    
    def __init__(
        self,
        mode: str = "train",
        image_size: Tuple[int, int] = (224, 224),
        normalize_mean: Optional[List[float]] = None,
        normalize_std: Optional[List[float]] = None,
        augmentation_strength: str = "medium",
    ) -> None:
        """
        Initialize image augmentation.
        
        Args:
            mode: Augmentation mode (train, eval, custom).
            image_size: Target image size (height, width).
            normalize_mean: Normalization mean values.
            normalize_std: Normalization std values.
            augmentation_strength: Strength level (light, medium, strong).
        """
        self.mode = mode
        self.image_size = image_size
        self.normalize_mean = normalize_mean or [0.485, 0.456, 0.406]
        self.normalize_std = normalize_std or [0.229, 0.224, 0.225]
        self.augmentation_strength = augmentation_strength
        
        self.transform = self._build_transform()
    
    def _build_transform(self) -> transforms.Compose:
        """
        Build transformation pipeline based on mode.
        
        Returns:
            Composed transformation pipeline.
        """
        if self.mode == "eval":
            return self._build_eval_transform()
        elif self.mode == "train":
            return self._build_train_transform()
        else:
            return self._build_eval_transform()
    
    def _build_eval_transform(self) -> transforms.Compose:
        """
        Build evaluation transform (minimal augmentation).
        
        Returns:
            Evaluation transformation pipeline.
        """
        return transforms.Compose([
            transforms.Resize(self.image_size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=self.normalize_mean,
                std=self.normalize_std,
            ),
        ])
    
    def _build_train_transform(self) -> transforms.Compose:
        """
        Build training transform with augmentation.
        
        Returns:
            Training transformation pipeline.
        """
        # Base transforms
        transform_list = [
            transforms.Resize(
                (int(self.image_size[0] * 1.1), int(self.image_size[1] * 1.1))
            ),
            transforms.RandomCrop(self.image_size),
        ]
        
        # Add augmentations based on strength
        if self.augmentation_strength == "light":
            transform_list.extend([
                transforms.RandomHorizontalFlip(p=0.3),
                transforms.ColorJitter(
                    brightness=0.1,
                    contrast=0.1,
                    saturation=0.1,
                ),
            ])
        elif self.augmentation_strength == "medium":
            transform_list.extend([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(
                    brightness=0.2,
                    contrast=0.2,
                    saturation=0.2,
                    hue=0.1,
                ),
                transforms.RandomGrayscale(p=0.1),
            ])
        elif self.augmentation_strength == "strong":
            transform_list.extend([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.1),
                transforms.ColorJitter(
                    brightness=0.3,
                    contrast=0.3,
                    saturation=0.3,
                    hue=0.15,
                ),
                transforms.RandomGrayscale(p=0.2),
                transforms.RandomRotation(degrees=10),
                transforms.RandomAffine(
                    degrees=0,
                    translate=(0.1, 0.1),
                    scale=(0.9, 1.1),
                ),
            ])
        
        # Final transforms
        transform_list.extend([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=self.normalize_mean,
                std=self.normalize_std,
            ),
        ])
        
        return transforms.Compose(transform_list)
    
    def __call__(self, image: Any) -> torch.Tensor:
        """
        Apply transformation to image.
        
        Args:
            image: Input image (PIL Image or numpy array).
            
        Returns:
            Transformed image tensor.
        """
        return self.transform(image)


class RandomErasing:
    """
    Random erasing augmentation for images.
    
    Randomly selects a rectangle region and erases its pixels.
    Useful for improving model robustness to occlusion.
    """
    
    def __init__(
        self,
        probability: float = 0.5,
        scale: Tuple[float, float] = (0.02, 0.33),
        ratio: Tuple[float, float] = (0.3, 3.3),
        value: float = 0,
    ) -> None:
        """
        Initialize random erasing.
        
        Args:
            probability: Probability of applying erasing.
            scale: Range of proportion of erased area.
            ratio: Range of aspect ratio of erased area.
            value: Erasing value (0 for mean erasing).
        """
        self.probability = probability
        self.scale = scale
        self.ratio = ratio
        self.value = value
        
        self.erasing = transforms.RandomErasing(
            p=probability,
            scale=scale,
            ratio=ratio,
            value=value,
        )
    
    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Apply random erasing to tensor.
        
        Args:
            tensor: Input image tensor.
            
        Returns:
            Augmented tensor.
        """
        return self.erasing(tensor)


class MixUp:
    """
    MixUp augmentation for VQA.
    
    Creates virtual training examples by mixing pairs of samples.
    """
    
    def __init__(self, alpha: float = 0.4) -> None:
        """
        Initialize MixUp.
        
        Args:
            alpha: Beta distribution parameter for mixing ratio.
        """
        self.alpha = alpha
    
    def __call__(
        self,
        images: torch.Tensor,
        labels: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
        """
        Apply MixUp to batch.
        
        Args:
            images: Batch of images [B, C, H, W].
            labels: Batch of labels [B].
            
        Returns:
            Tuple of (mixed_images, labels_a, labels_b, lambda).
        """
        if self.alpha > 0:
            lam = random.betavariate(self.alpha, self.alpha)
        else:
            lam = 1.0
        
        batch_size = images.size(0)
        index = torch.randperm(batch_size)
        
        mixed_images = lam * images + (1 - lam) * images[index, :]
        labels_a, labels_b = labels, labels[index]
        
        return mixed_images, labels_a, labels_b, lam


class CutMix:
    """
    CutMix augmentation for VQA.
    
    Cuts and pastes patches among training images.
    """
    
    def __init__(self, alpha: float = 1.0) -> None:
        """
        Initialize CutMix.
        
        Args:
            alpha: Beta distribution parameter for mixing ratio.
        """
        self.alpha = alpha
    
    def _rand_bbox(
        self,
        size: Tuple[int, ...],
        lam: float,
    ) -> Tuple[int, int, int, int]:
        """
        Generate random bounding box.
        
        Args:
            size: Image size [B, C, H, W].
            lam: Lambda value for area calculation.
            
        Returns:
            Bounding box coordinates (x1, y1, x2, y2).
        """
        W = size[3]
        H = size[2]
        cut_rat = (1.0 - lam) ** 0.5
        cut_w = int(W * cut_rat)
        cut_h = int(H * cut_rat)
        
        cx = random.randint(0, W)
        cy = random.randint(0, H)
        
        x1 = max(0, cx - cut_w // 2)
        y1 = max(0, cy - cut_h // 2)
        x2 = min(W, cx + cut_w // 2)
        y2 = min(H, cy + cut_h // 2)
        
        return x1, y1, x2, y2
    
    def __call__(
        self,
        images: torch.Tensor,
        labels: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
        """
        Apply CutMix to batch.
        
        Args:
            images: Batch of images [B, C, H, W].
            labels: Batch of labels [B].
            
        Returns:
            Tuple of (mixed_images, labels_a, labels_b, lambda).
        """
        if self.alpha > 0:
            lam = random.betavariate(self.alpha, self.alpha)
        else:
            lam = 1.0
        
        batch_size = images.size(0)
        index = torch.randperm(batch_size)
        
        x1, y1, x2, y2 = self._rand_bbox(images.size(), lam)
        
        images[:, :, y1:y2, x1:x2] = images[index, :, y1:y2, x1:x2]
        
        # Adjust lambda based on actual cut area
        lam = 1 - ((x2 - x1) * (y2 - y1) / (images.size(-1) * images.size(-2)))
        
        labels_a, labels_b = labels, labels[index]
        
        return images, labels_a, labels_b, lam


# =============================================================================
# Text Augmentation
# =============================================================================

class TextAugmentation:
    """
    Text augmentation for Vietnamese VQA questions.
    
    Applies various text augmentations to improve model robustness.
    """
    
    def __init__(
        self,
        augmentation_probability: float = 0.3,
        enable_synonym_replacement: bool = True,
        enable_random_deletion: bool = True,
        enable_random_swap: bool = True,
    ) -> None:
        """
        Initialize text augmentation.
        
        Args:
            augmentation_probability: Probability of applying augmentation.
            enable_synonym_replacement: Enable synonym replacement.
            enable_random_deletion: Enable random word deletion.
            enable_random_swap: Enable random word swap.
        """
        self.augmentation_probability = augmentation_probability
        self.enable_synonym_replacement = enable_synonym_replacement
        self.enable_random_deletion = enable_random_deletion
        self.enable_random_swap = enable_random_swap
    
    def random_deletion(
        self,
        words: List[str],
        p: float = 0.1,
    ) -> List[str]:
        """
        Randomly delete words from sentence.
        
        Args:
            words: List of words.
            p: Probability of deleting each word.
            
        Returns:
            List of remaining words.
        """
        if len(words) <= 1:
            return words
        
        new_words = []
        for word in words:
            if random.random() > p:
                new_words.append(word)
        
        if len(new_words) == 0:
            return [random.choice(words)]
        
        return new_words
    
    def random_swap(
        self,
        words: List[str],
        n: int = 1,
    ) -> List[str]:
        """
        Randomly swap word positions.
        
        Args:
            words: List of words.
            n: Number of swaps.
            
        Returns:
            List of words with swapped positions.
        """
        if len(words) < 2:
            return words
        
        new_words = words.copy()
        
        for _ in range(n):
            idx1 = random.randint(0, len(new_words) - 1)
            idx2 = random.randint(0, len(new_words) - 1)
            new_words[idx1], new_words[idx2] = new_words[idx2], new_words[idx1]
        
        return new_words
    
    def __call__(self, text: str) -> str:
        """
        Apply text augmentation.
        
        Args:
            text: Input text.
            
        Returns:
            Augmented text.
        """
        if random.random() > self.augmentation_probability:
            return text
        
        words = text.split()
        
        if len(words) <= 1:
            return text
        
        # Apply augmentations
        augmentation_functions = []
        
        if self.enable_random_deletion:
            augmentation_functions.append(
                lambda w: self.random_deletion(w, p=0.1)
            )
        
        if self.enable_random_swap:
            augmentation_functions.append(
                lambda w: self.random_swap(w, n=1)
            )
        
        if augmentation_functions:
            aug_func = random.choice(augmentation_functions)
            words = aug_func(words)
        
        return " ".join(words)


# =============================================================================
# Dropout Utilities
# =============================================================================

class DropoutScheduler:
    """
    Dynamic dropout scheduler.
    
    Adjusts dropout rate during training based on schedule.
    """
    
    def __init__(
        self,
        initial_dropout: float = 0.1,
        final_dropout: float = 0.3,
        total_steps: int = 10000,
        warmup_steps: int = 1000,
        schedule: str = "linear",
    ) -> None:
        """
        Initialize dropout scheduler.
        
        Args:
            initial_dropout: Initial dropout rate.
            final_dropout: Final dropout rate.
            total_steps: Total training steps.
            warmup_steps: Warmup steps before increasing dropout.
            schedule: Schedule type (linear, cosine).
        """
        self.initial_dropout = initial_dropout
        self.final_dropout = final_dropout
        self.total_steps = total_steps
        self.warmup_steps = warmup_steps
        self.schedule = schedule
        self.current_step = 0
    
    def get_dropout(self, step: Optional[int] = None) -> float:
        """
        Get dropout rate for current step.
        
        Args:
            step: Step number (uses current_step if None).
            
        Returns:
            Current dropout rate.
        """
        if step is None:
            step = self.current_step
        
        if step < self.warmup_steps:
            return self.initial_dropout
        
        progress = (step - self.warmup_steps) / max(1, self.total_steps - self.warmup_steps)
        progress = min(1.0, progress)
        
        if self.schedule == "linear":
            return self.initial_dropout + progress * (self.final_dropout - self.initial_dropout)
        elif self.schedule == "cosine":
            import math
            return self.initial_dropout + 0.5 * (self.final_dropout - self.initial_dropout) * (1 - math.cos(math.pi * progress))
        else:
            return self.initial_dropout
    
    def step(self) -> float:
        """
        Advance step and get dropout rate.
        
        Returns:
            Updated dropout rate.
        """
        self.current_step += 1
        return self.get_dropout()
    
    def apply_to_model(self, model: nn.Module, dropout_rate: Optional[float] = None) -> None:
        """
        Apply dropout rate to all dropout layers in model.
        
        Args:
            model: Model to update.
            dropout_rate: Dropout rate to apply (uses current if None).
        """
        if dropout_rate is None:
            dropout_rate = self.get_dropout()
        
        for module in model.modules():
            if isinstance(module, nn.Dropout):
                module.p = dropout_rate


# =============================================================================
# Augmentation Factory
# =============================================================================

def create_image_augmentation(
    mode: str = "train",
    image_size: Tuple[int, int] = (224, 224),
    normalize_mean: Optional[List[float]] = None,
    normalize_std: Optional[List[float]] = None,
    augmentation_strength: str = "medium",
) -> ImageAugmentation:
    """
    Create image augmentation pipeline.
    
    Args:
        mode: Augmentation mode (train, eval).
        image_size: Target image size.
        normalize_mean: Normalization mean.
        normalize_std: Normalization std.
        augmentation_strength: Augmentation strength level.
        
    Returns:
        Configured ImageAugmentation instance.
    """
    return ImageAugmentation(
        mode=mode,
        image_size=image_size,
        normalize_mean=normalize_mean,
        normalize_std=normalize_std,
        augmentation_strength=augmentation_strength,
    )


def create_text_augmentation(
    augmentation_probability: float = 0.3,
) -> TextAugmentation:
    """
    Create text augmentation pipeline.
    
    Args:
        augmentation_probability: Probability of applying augmentation.
        
    Returns:
        Configured TextAugmentation instance.
    """
    return TextAugmentation(
        augmentation_probability=augmentation_probability,
    )


# =============================================================================
# Module Exports
# =============================================================================

__all__ = [
    "ImageAugmentation",
    "RandomErasing",
    "MixUp",
    "CutMix",
    "TextAugmentation",
    "DropoutScheduler",
    "create_image_augmentation",
    "create_text_augmentation",
]
