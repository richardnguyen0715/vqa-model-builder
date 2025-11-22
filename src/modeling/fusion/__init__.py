"""
Multimodal Fusion Module for VQA.
"""

from src.modeling.fusion.fusion_approaches import (
    BaseFusion,
    CrossAttentionFusion,
    QFormerFusion,
    SingleStreamFusion,
    create_fusion_model
)

__all__ = [
    'BaseFusion',
    'CrossAttentionFusion',
    'QFormerFusion',
    'SingleStreamFusion',
    'create_fusion_model'
]
