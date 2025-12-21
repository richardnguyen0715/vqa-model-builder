# Mixture of Experts (MOE) Module

## Overview

The MOE module implements a comprehensive Mixture of Experts architecture for Vietnamese Visual Question Answering (VQA). This module enables dynamic routing of tokens to specialized experts, allowing the model to leverage different expertise for different types of inputs.

## Architecture

### Core Components

```
Input Features
      |
      v
  [Router]  -----> Routing Weights & Expert Indices
      |
      v
  [Experts]
      |
      +-- Vision Expert (spatial attention, image understanding)
      +-- Text Expert (Vietnamese language processing)
      +-- Multimodal Expert (cross-modal reasoning)
      +-- Specialized Experts:
          +-- Segmentation Expert (SAM-inspired)
          +-- Object Detection Expert (DETR-inspired)
          +-- OCR Expert (Vietnamese text recognition)
          +-- Scene Understanding Expert
          +-- Spatial Reasoning Expert
          +-- Counting Expert
      |
      v
  [Weighted Combination]
      |
      v
Output Features
```

## Components

### 1. Routers

#### TopKRouter
- Routes each token to top-k experts based on learned gating scores
- Includes load balancing auxiliary loss

#### NoisyTopKRouter
- Adds trainable noise for better exploration
- Helps prevent expert collapse

#### SoftRouter
- Routes to all experts with soft weights
- No sparsity, suitable for smaller models

#### ExpertChoiceRouter
- Experts choose tokens instead of tokens choosing experts
- Ensures perfect load balancing

### 2. Expert Types

#### FeedForwardExpert
- Standard two-layer MLP
- GELU activation with residual connections

#### VisionExpert
- Spatial attention mechanism
- Multi-scale feature processing
- Position-aware transformations

#### TextExpert
- Self-attention for contextual understanding
- Optimized for Vietnamese language
- Positional encoding support

#### MultimodalExpert
- Cross-attention between modalities
- Modality-aware gating mechanism
- Fusion-oriented processing

### 3. Specialized Experts

#### SegmentationExpert
- SAM (Segment Anything Model) inspired
- Mask prediction with learnable tokens
- Boundary-aware feature extraction

#### ObjectDetectionExpert
- DETR-inspired architecture
- Learnable object queries
- Region proposal mechanism

#### OCRExpert
- Text region detection
- Vietnamese diacritics handling
- Reading order inference

#### SceneUnderstandingExpert
- Global scene representation
- Context distribution mechanism
- Scene classification support

#### SpatialReasoningExpert
- Spatial relationship modeling
- Graph attention mechanism
- Relative position encoding

### 4. MOE Layers

#### MOELayer
- Standard MOE implementation
- Configurable router and expert types
- Auxiliary loss computation

#### SparseMOELayer
- Memory-efficient implementation
- Capacity constraint per expert
- Token dispatch mechanism

#### HierarchicalMOE
- Two-level routing
- First routes to expert groups, then to individuals
- Better scalability for large models

#### VQAMOELayer
- VQA-specific configuration
- Combines all expert types
- Vietnamese language optimization

## Usage

### Basic Usage

```python
from src.modeling.moe import MOELayer, MOEConfig

# Configuration
config = MOEConfig(
    input_dim=768,
    hidden_dim=3072,
    output_dim=768,
    num_experts=8,
    num_experts_per_token=2
)

# Create MOE layer
moe_layer = MOELayer(config=config)

# Forward pass
x = torch.randn(batch_size, seq_len, 768)
output = moe_layer(x)

# Get auxiliary loss for training
aux_loss = moe_layer.get_aux_loss()
```

### VQA-specific MOE

```python
from src.modeling.moe import VQAMOELayer

vqa_moe = VQAMOELayer(
    input_dim=768,
    hidden_dim=3072,
    output_dim=768,
    num_vision_experts=2,
    num_text_experts=2,
    num_multimodal_experts=2,
    num_specialized_experts=2,
    top_k=2,
    vietnamese_optimized=True
)

output = vqa_moe(features)
```

### Hierarchical MOE

```python
from src.modeling.moe import HierarchicalMOE

hierarchical_moe = HierarchicalMOE(
    input_dim=768,
    hidden_dim=3072,
    output_dim=768,
    num_expert_groups=4,
    experts_per_group=4,
    top_k_groups=2,
    top_k_experts=1,
    expert_types=['vision', 'text', 'multimodal', 'feedforward']
)

output = hierarchical_moe(x)
```

## Configuration

### MOEConfig

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| input_dim | int | 768 | Input feature dimension |
| hidden_dim | int | 3072 | Hidden layer dimension |
| output_dim | int | 768 | Output feature dimension |
| num_experts | int | 8 | Number of experts |
| num_experts_per_token | int | 2 | Experts activated per token |
| use_sparse_moe | bool | True | Use sparse MOE |
| expert_dropout | float | 0.1 | Expert dropout rate |

### RouterConfig

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| router_type | str | 'topk' | Router type |
| top_k | int | 2 | Number of experts per token |
| noise_std | float | 1.0 | Noise standard deviation |
| load_balance_weight | float | 0.01 | Load balance loss weight |
| use_aux_loss | bool | True | Compute auxiliary loss |

## Training Considerations

### Load Balancing

The load balancing loss encourages uniform distribution of tokens across experts:

```python
loss = task_loss + moe_layer.get_aux_loss()
```

### Expert Dropout

Use ExpertDropout during training to improve robustness:

```python
from src.modeling.moe import ExpertDropout

expert_dropout = ExpertDropout(num_experts=8, drop_rate=0.1)
weights, indices = expert_dropout(routing_weights, expert_indices)
```

### Capacity Constraints

For sparse MOE, capacity limits prevent memory issues:

```python
capacity = int(capacity_factor * num_tokens * top_k / num_experts)
```

## Analysis Tools

### Expert Utilization

```python
from src.modeling.moe import get_expert_utilization

utilization = get_expert_utilization(expert_indices, num_experts)
```

### Routing Entropy

```python
from src.modeling.moe import compute_expert_entropy

entropy = compute_expert_entropy(router_probs)
```

### Pattern Analysis

```python
from src.modeling.moe import analyze_routing_patterns

analysis = analyze_routing_patterns(router_probs, expert_indices, num_experts)
```

## References

1. Shazeer et al. "Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer" (2017)
2. Fedus et al. "Switch Transformers: Scaling to Trillion Parameter Models" (2021)
3. Zhou et al. "Mixture-of-Experts with Expert Choice Routing" (2022)
4. Kirillov et al. "Segment Anything" (2023) - SAM
5. Carion et al. "End-to-End Object Detection with Transformers" (2020) - DETR
