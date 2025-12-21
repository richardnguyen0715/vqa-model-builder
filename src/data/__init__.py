"""
Data Module
Provides data processing, loading, and augmentation utilities.
"""

from .augmentation import (
    ImageAugmentation,
    TextAugmentation,
    MixUpAugmentation,
    CutMixAugmentation,
    DropoutScheduler,
    get_augmentation_pipeline,
    create_train_augmentation,
    create_val_augmentation
)

__all__ = [
    # Image augmentation
    'ImageAugmentation',
    
    # Text augmentation
    'TextAugmentation',
    
    # Mixed sample augmentations
    'MixUpAugmentation',
    'CutMixAugmentation',
    
    # Dropout scheduling
    'DropoutScheduler',
    
    # Factory functions
    'get_augmentation_pipeline',
    'create_train_augmentation',
    'create_val_augmentation'
]