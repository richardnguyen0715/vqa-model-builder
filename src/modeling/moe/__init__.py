"""
Mixture of Experts (MOE) Module for Vietnamese VQA
Provides expert routing and specialized processing for multimodal understanding.
"""

from src.modeling.moe.base_expert import BaseExpert, ExpertWithCapacity
from src.modeling.moe.router import (
    BaseRouter,
    TopKRouter,
    SoftRouter,
    NoisyTopKRouter,
    ExpertChoiceRouter,
    create_router
)
from src.modeling.moe.expert_types import (
    VisionExpert,
    TextExpert,
    MultimodalExpert,
    FeedForwardExpert,
    GatedLinearExpert,
    create_expert
)
from src.modeling.moe.specialized_experts import (
    SegmentationExpert,
    ObjectDetectionExpert,
    OCRExpert,
    SceneUnderstandingExpert,
    SpatialReasoningExpert,
    CountingExpert
)
from src.modeling.moe.moe_layer import (
    MOELayer,
    SparseMOELayer,
    HierarchicalMOE,
    VQAMOELayer
)
from src.modeling.moe.moe_config import MOEConfig, ExpertConfig, RouterConfig, VQAMOEConfig
from src.modeling.moe.moe_utils import (
    compute_expert_capacity,
    compute_load_balance_loss,
    compute_router_z_loss,
    get_expert_utilization,
    compute_expert_entropy,
    ExpertDropout,
    ExpertParallelWrapper,
    save_moe_checkpoint,
    load_moe_checkpoint,
    analyze_routing_patterns
)

__all__ = [
    # Base
    'BaseExpert',
    'ExpertWithCapacity',
    # Routers
    'BaseRouter',
    'TopKRouter', 
    'SoftRouter',
    'NoisyTopKRouter',
    'ExpertChoiceRouter',
    'create_router',
    # Expert Types
    'VisionExpert',
    'TextExpert',
    'MultimodalExpert',
    'FeedForwardExpert',
    'GatedLinearExpert',
    'create_expert',
    # Specialized Experts
    'SegmentationExpert',
    'ObjectDetectionExpert',
    'OCRExpert',
    'SceneUnderstandingExpert',
    'SpatialReasoningExpert',
    'CountingExpert',
    # MOE Layers
    'MOELayer',
    'SparseMOELayer',
    'HierarchicalMOE',
    'VQAMOELayer',
    # Config
    'MOEConfig',
    'ExpertConfig',
    'RouterConfig',
    'VQAMOEConfig',
    # Utils
    'compute_expert_capacity',
    'compute_load_balance_loss',
    'compute_router_z_loss',
    'get_expert_utilization',
    'compute_expert_entropy',
    'ExpertDropout',
    'ExpertParallelWrapper',
    'save_moe_checkpoint',
    'load_moe_checkpoint',
    'analyze_routing_patterns'
]
