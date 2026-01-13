"""
Image and Text Representation Heads for VQA Models.
"""

from src.modeling.heads.image_representation import (
    BaseImageRepresentation,
    RegionBasedVisionEmbedding,
    VisionTransformerEmbedding,
    MultiResolutionFeatures,
    VisionTokenEmbedding,
    create_image_representation
)

from src.modeling.heads.text_representation import (
    BaseTextRepresentation,
    BERTTextEmbedding,
    RoBERTaTextEmbedding,
    DeBERTaV3TextEmbedding,
    GenericTransformerTextEmbedding,
    create_text_representation
)

__all__ = [
    # Image representations
    'BaseImageRepresentation',
    'RegionBasedVisionEmbedding',
    'VisionTransformerEmbedding',
    'MultiResolutionFeatures',
    'VisionTokenEmbedding',
    'create_image_representation',
    
    # Text representations
    'BaseTextRepresentation',
    'BERTTextEmbedding',
    'RoBERTaTextEmbedding',
    'DeBERTaV3TextEmbedding',
    'GenericTransformerTextEmbedding',
    'create_text_representation'
]
