# Image Representation Approaches for VQA

This module implements 4 different approaches for extracting visual features from images for Visual Question Answering tasks.

## 1. Region-Based Vision Embedding (`RegionBasedVisionEmbedding`)

**Approach**: Bottom-Up Attention using region proposals from object detection.

**Key Features**:
- Extracts features from detected object regions (default: 36 regions)
- Uses pretrained CNN backbone (ResNet50/101)
- Includes spatial features (bounding box coordinates)
- Inspired by: Anderson et al. "Bottom-Up and Top-Down Attention for VQA" (2018)

**Usage**:
```python
from src.modeling.heads import RegionBasedVisionEmbedding

model = RegionBasedVisionEmbedding(
    backbone_name='resnet101',
    num_regions=36,
    region_feat_dim=2048,
    output_dim=768,
    use_spatial_features=True
)

# Input: [batch_size, 3, 224, 224]
# Output: [batch_size, 36, 768]
features = model(images)
```

**Pros**:
- Rich object-level features
- Good for object-centric questions
- Interpretable (can visualize attended regions)

**Cons**:
- Requires region proposal network
- More computationally expensive
- Fixed number of regions

---

## 2. Vision Transformer Embedding (`VisionTransformerEmbedding`)

**Approach**: Patch-based vision transformer (ViT).

**Key Features**:
- Divides image into patches (default: 16x16)
- Uses transformer encoder with self-attention
- Includes CLS token for global representation
- Inspired by: Dosovitskiy et al. "An Image is Worth 16x16 Words" (2021)

**Usage**:
```python
from src.modeling.heads import VisionTransformerEmbedding

model = VisionTransformerEmbedding(
    image_size=224,
    patch_size=16,
    embed_dim=768,
    num_heads=12,
    num_layers=12,
    use_pretrained=True
)

# Input: [batch_size, 3, 224, 224]
# Output: [batch_size, 197, 768]  # 196 patches + 1 CLS token
features = model(images)
```

**Pros**:
- State-of-the-art performance
- Captures global context well
- Pretrained models available

**Cons**:
- Requires more training data
- Higher memory usage
- Fixed input resolution

---

## 3. Multi-Resolution Features (`MultiResolutionFeatures`)

**Approach**: Feature Pyramid Network (FPN) combining multiple scales.

**Key Features**:
- Extracts features from multiple CNN layers
- Combines low-level and high-level features
- Top-down pathway with lateral connections
- Inspired by: Lin et al. "Feature Pyramid Networks for Object Detection" (2017)

**Usage**:
```python
from src.modeling.heads import MultiResolutionFeatures

model = MultiResolutionFeatures(
    backbone_name='resnet50',
    output_dim=768,
    feature_levels=[2, 3, 4],  # Which ResNet stages to use
    fpn_dim=256
)

# Input: [batch_size, 3, 224, 224]
# Output: [batch_size, 1, 768]  # Global multi-scale feature
features = model(images)
```

**Pros**:
- Captures both fine details and semantic info
- Good for multi-scale reasoning
- Efficient fusion of features

**Cons**:
- Requires careful tuning of feature levels
- More complex architecture
- May need task-specific adaptation

---

## 4. Vision Token Embedding (`VisionTokenEmbedding`)

**Approach**: Learnable vision tokens with cross-attention (inspired by modern VLMs).

**Key Features**:
- Uses learnable query tokens that attend to image features
- Cross-attention mechanism for feature extraction
- Similar to Perceiver, BLIP-2, Flamingo approaches
- Flexible number of output tokens

**Usage**:
```python
from src.modeling.heads import VisionTokenEmbedding

model = VisionTokenEmbedding(
    backbone_name='resnet50',
    num_vision_tokens=32,
    token_dim=768,
    num_attention_heads=8,
    num_layers=4,
    use_token_type=True
)

# Input: [batch_size, 3, 224, 224]
# Output: [batch_size, 32, 768]  # 32 vision tokens
features = model(images)
```

**Pros**:
- Flexible output dimensionality
- Learns task-relevant features
- Can compress large images efficiently
- State-of-the-art in recent VLMs

**Cons**:
- Requires more training
- Less interpretable
- Token number needs tuning

---

## Factory Function

For easy model creation:

```python
from src.modeling.heads import create_image_representation

# Create any model type
model = create_image_representation(
    model_type='vision_token',  # or 'region_based', 'vit', 'multi_resolution'
    backbone_name='resnet50',
    output_dim=768,
    # ... other model-specific args
)
```

---

## Comparison Table

| Approach | Output Shape | Params | Speed | Best For |
|----------|--------------|--------|-------|----------|
| Region-Based | [B, 36, 768] | ~44M | Medium | Object-centric VQA |
| Vision Transformer | [B, 197, 768] | ~86M | Slow | General VQA |
| Multi-Resolution | [B, 1, 768] | ~25M | Fast | Multi-scale reasoning |
| Vision Token | [B, 32, 768] | ~40M | Medium | Efficient VLMs |

---

## Integration with VQA Model

```python
from src.modeling.heads import create_image_representation

# Choose your approach
vision_encoder = create_image_representation(
    model_type='vision_token',
    output_dim=768
)

# In VQA model forward pass
class VQAModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.vision_encoder = vision_encoder
        self.text_encoder = ...
        self.fusion_module = ...
        
    def forward(self, images, questions):
        # Extract visual features
        vision_features = self.vision_encoder(images)  # [B, N, D]
        
        # Extract text features
        text_features = self.text_encoder(questions)   # [B, M, D]
        
        # Multimodal fusion
        fused = self.fusion_module(vision_features, text_features)
        
        return fused
```

---

## References

1. Anderson et al., "Bottom-Up and Top-Down Attention for Image Captioning and Visual Question Answering", CVPR 2018
2. Dosovitskiy et al., "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale", ICLR 2021
3. Lin et al., "Feature Pyramid Networks for Object Detection", CVPR 2017
4. Alayrac et al., "Flamingo: a Visual Language Model for Few-Shot Learning", NeurIPS 2022
5. Li et al., "BLIP-2: Bootstrapping Language-Image Pre-training with Frozen Image Encoders", ICML 2023
