"""
MOE Configuration Module
Defines configuration classes for Mixture of Experts components.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any


@dataclass
class ExpertConfig:
    """
    Configuration for individual expert.
    
    Attributes:
        expert_type: Type of expert (vision, text, multimodal, etc.)
        input_dim: Input feature dimension
        hidden_dim: Hidden layer dimension
        output_dim: Output feature dimension
        num_layers: Number of layers in expert
        dropout: Dropout rate
        activation: Activation function name
        use_layer_norm: Whether to use layer normalization
        expert_capacity: Maximum tokens each expert can process
    """
    expert_type: str = 'feedforward'
    input_dim: int = 768
    hidden_dim: int = 3072
    output_dim: int = 768
    num_layers: int = 2
    dropout: float = 0.1
    activation: str = 'gelu'
    use_layer_norm: bool = True
    expert_capacity: Optional[int] = None


@dataclass
class RouterConfig:
    """
    Configuration for expert router/gating mechanism.
    
    Attributes:
        router_type: Type of router (topk, soft, noisy_topk)
        num_experts: Total number of experts
        top_k: Number of experts to route to
        noise_std: Standard deviation for noisy routing
        load_balance_weight: Weight for load balancing loss
        capacity_factor: Factor to determine expert capacity
        use_aux_loss: Whether to use auxiliary load balancing loss
        jitter_noise: Whether to add jitter noise during training
    """
    router_type: str = 'topk'
    num_experts: int = 8
    top_k: int = 2
    noise_std: float = 1.0
    load_balance_weight: float = 0.01
    capacity_factor: float = 1.25
    use_aux_loss: bool = True
    jitter_noise: bool = True


@dataclass
class MOEConfig:
    """
    Main configuration for Mixture of Experts module.
    
    Attributes:
        input_dim: Input feature dimension
        hidden_dim: Hidden dimension for experts
        output_dim: Output feature dimension
        num_experts: Number of experts in the mixture
        num_experts_per_token: Experts activated per token (top_k)
        expert_configs: List of individual expert configurations
        router_config: Router configuration
        use_sparse_moe: Whether to use sparse MOE (memory efficient)
        expert_dropout: Dropout applied to expert outputs
        combine_method: Method to combine expert outputs (weighted_sum, concat)
        shared_expert: Whether to include a shared expert
        hierarchical: Whether to use hierarchical MOE
        num_hierarchical_levels: Number of levels in hierarchical MOE
    """
    input_dim: int = 768
    hidden_dim: int = 3072
    output_dim: int = 768
    num_experts: int = 8
    num_experts_per_token: int = 2
    expert_configs: Optional[List[ExpertConfig]] = None
    router_config: Optional[RouterConfig] = None
    use_sparse_moe: bool = True
    expert_dropout: float = 0.1
    combine_method: str = 'weighted_sum'
    shared_expert: bool = False
    hierarchical: bool = False
    num_hierarchical_levels: int = 2
    
    def __post_init__(self):
        """Initialize default configurations if not provided."""
        if self.router_config is None:
            self.router_config = RouterConfig(
                num_experts=self.num_experts,
                top_k=self.num_experts_per_token
            )
        
        if self.expert_configs is None:
            self.expert_configs = [
                ExpertConfig(
                    input_dim=self.input_dim,
                    hidden_dim=self.hidden_dim,
                    output_dim=self.output_dim
                )
                for _ in range(self.num_experts)
            ]


@dataclass
class VQAMOEConfig(MOEConfig):
    """
    Configuration for VQA-specific MOE with specialized experts.
    
    Attributes:
        vision_expert_indices: Indices of vision-specialized experts
        text_expert_indices: Indices of text-specialized experts
        multimodal_expert_indices: Indices of multimodal experts
        use_segmentation_expert: Whether to include SAM-based segmentation expert
        use_detection_expert: Whether to include object detection expert
        use_ocr_expert: Whether to include OCR expert for text in images
        use_scene_expert: Whether to include scene understanding expert
        use_spatial_expert: Whether to include spatial reasoning expert
        vietnamese_optimized: Whether experts are optimized for Vietnamese
    """
    vision_expert_indices: List[int] = field(default_factory=lambda: [0, 1])
    text_expert_indices: List[int] = field(default_factory=lambda: [2, 3])
    multimodal_expert_indices: List[int] = field(default_factory=lambda: [4, 5, 6, 7])
    use_segmentation_expert: bool = True
    use_detection_expert: bool = True
    use_ocr_expert: bool = True
    use_scene_expert: bool = True
    use_spatial_expert: bool = True
    vietnamese_optimized: bool = True
    
    def get_expert_type_for_index(self, index: int) -> str:
        """
        Get expert type based on index.
        
        Args:
            index: Expert index
            
        Returns:
            Expert type string
        """
        if index in self.vision_expert_indices:
            return 'vision'
        elif index in self.text_expert_indices:
            return 'text'
        elif index in self.multimodal_expert_indices:
            return 'multimodal'
        else:
            return 'feedforward'
