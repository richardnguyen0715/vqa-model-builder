# Multimodal Fusion Approaches for VQA

This module implements 3 different strategies for fusing vision and language features in Visual Question Answering tasks.

## Overview

All fusion models inherit from `BaseFusion` and support:
- **Flexible input dimensions**: Vision and text features can have different dimensions
- **Optional attention masks**: Handle variable-length sequences
- **Unified output**: All models produce consistent output format
- **Easy integration**: Factory function for simple model creation

---

## 1. Cross-Attention Fusion (`CrossAttentionFusion`)

**Approach**: Bidirectional cross-attention between vision and text modalities.

### Architecture

```
Vision Features  ←--Cross-Attn--→  Text Features
      ↓                                  ↓
 Self-Attention                   Self-Attention
      ↓                                  ↓
    Pool                               Pool
      ↓                                  ↓
      └─────────→  Fusion  ←────────────┘
                     ↓
              Fused Output
```

### Key Features

- **Bidirectional Attention**: 
  - Vision attends to Text (text-guided vision)
  - Text attends to Vision (vision-guided text)
- **Multi-layer processing**: Multiple cross-attention layers for deep interaction
- **Flexible fusion**: Supports concatenation, addition, or multiplication
- **Separate processing**: Each modality maintains its own representation stream

### Usage

```python
from src.modeling.fusion import CrossAttentionFusion

fusion = CrossAttentionFusion(
    vision_dim=768,
    text_dim=768,
    output_dim=768,
    num_attention_heads=8,
    num_layers=4,
    fusion_method='concat'  # 'concat', 'add', or 'multiply'
)

# Forward pass
vision_features = torch.randn(2, 36, 768)  # [batch, num_vision_tokens, vision_dim]
text_features = torch.randn(2, 20, 768)    # [batch, num_text_tokens, text_dim]

fused_output = fusion(vision_features, text_features)  # [batch, output_dim]
```

### Pros
- ✅ Explicit cross-modal interaction
- ✅ Maintains separate modality streams
- ✅ Flexible fusion strategies
- ✅ Well-established approach (ViLBERT, LXMERT)

### Cons
- ❌ Higher computational cost (separate streams)
- ❌ More parameters than single-stream
- ❌ Requires careful tuning of fusion method

### Best For
- Complex reasoning requiring explicit cross-modal alignment
- When you need interpretable attention weights
- Multi-task learning with modality-specific objectives

### References
- Lu et al., "ViLBERT: Pretraining Task-Agnostic Visiolinguistic Representations", NeurIPS 2019
- Tan & Bansal, "LXMERT: Learning Cross-Modality Encoder Representations from Transformers", EMNLP 2019

---

## 2. Q-Former Fusion (`QFormerFusion`)

**Approach**: Learnable query tokens extract relevant information from both modalities through cross-attention.

### Architecture

```
Learnable Query Tokens [Q1, Q2, ..., Qn]
         ↓
   Self-Attention
         ↓
         ├──→ Cross-Attn ──→ Vision Features
         └──→ Cross-Attn ──→ Text Features
         ↓
    Query Output (Fused)
```

### Key Features

- **Learnable Queries**: Fixed number of query tokens (typically 32) that learn to extract relevant features
- **Bottleneck Architecture**: Reduces computation by limiting information flow through queries
- **Bidirectional Querying**: Queries attend to both vision and text
- **Efficient**: Queries act as information bottleneck, reducing complexity

### Usage

```python
from src.modeling.fusion import QFormerFusion

fusion = QFormerFusion(
    vision_dim=768,
    text_dim=768,
    output_dim=768,
    num_query_tokens=32,
    num_attention_heads=8,
    num_layers=6
)

# Forward pass
vision_features = torch.randn(2, 36, 768)
text_features = torch.randn(2, 20, 768)

fused_output = fusion(vision_features, text_features)  # [batch, output_dim]
```

### How Q-Former Works

1. **Initialize**: Start with learnable query tokens `Q = [Q1, Q2, ..., Q32]`
2. **Self-Attention**: Queries attend to each other to share information
3. **Cross-Attention to Vision**: Queries extract visual information
   ```
   Q_vision = Attention(Q=Queries, K=Vision, V=Vision)
   ```
4. **Cross-Attention to Text**: Queries extract textual information
   ```
   Q_text = Attention(Q=Queries, K=Text, V=Text)
   ```
5. **Output**: Queries now contain fused multimodal information

### Pros
- ✅ Very efficient (queries act as bottleneck)
- ✅ Fixed output size regardless of input length
- ✅ Learns to extract most relevant features
- ✅ State-of-the-art approach (BLIP-2)
- ✅ Fewer parameters than cross-attention

### Cons
- ❌ Information bottleneck can lose details
- ❌ Query tokens need to be initialized well
- ❌ Less interpretable than explicit cross-attention

### Best For
- Large-scale pretraining with efficiency constraints
- When you need fixed-size output representation
- Connecting frozen pretrained encoders
- Real-time applications requiring fast inference

### References
- Li et al., "BLIP-2: Bootstrapping Language-Image Pre-training with Frozen Image Encoders and Large Language Models", ICML 2023
- Inspired by Perceiver (Jaegle et al., 2021) and Flamingo (Alayrac et al., 2022)

---

## 3. Single-Stream Fusion (`SingleStreamFusion`)

**Approach**: Concatenate vision and text tokens and process jointly in a unified transformer (ViLT-style).

### Architecture

```
Vision Tokens        Text Tokens
    [V1, V2, ...]        [T1, T2, ...]
         ↓                      ↓
    + Modality Type 0      + Modality Type 1
         ↓                      ↓
         └──────→ [CLS] ←───────┘
                    ↓
        [CLS, V1, V2, ..., T1, T2, ...]
                    ↓
         + Position Embeddings
                    ↓
          Unified Transformer
                    ↓
           Extract [CLS] Token
```

### Key Features

- **Unified Processing**: Vision and text tokens processed together in one transformer
- **Modality Type Embeddings**: Learned embeddings distinguish vision vs text tokens
- **Simple Architecture**: No separate cross-attention modules needed
- **Early Fusion**: Modalities interact from the very first layer
- **CLS Token**: Global representation extracted from special [CLS] token

### Usage

```python
from src.modeling.fusion import SingleStreamFusion

fusion = SingleStreamFusion(
    vision_dim=768,
    text_dim=768,
    output_dim=768,
    num_attention_heads=12,
    num_layers=6,
    max_vision_tokens=200,
    max_text_tokens=128
)

# Forward pass
vision_features = torch.randn(2, 36, 768)
text_features = torch.randn(2, 20, 768)

fused_output = fusion(vision_features, text_features)  # [batch, output_dim]
```

### How Single-Stream Works

1. **Project**: Project vision and text to common dimension
2. **Add Modality Types**: 
   - Vision tokens get type embedding 0
   - Text tokens get type embedding 1
3. **Concatenate**: `[CLS] + Vision + Text`
4. **Position Embeddings**: Add learnable position embeddings
5. **Unified Transformer**: Process all tokens together
6. **Extract Output**: Use [CLS] token as fused representation

### Pros
- ✅ Simplest architecture
- ✅ Fewer parameters than cross-attention approaches
- ✅ Early fusion allows deep interaction
- ✅ Can leverage pretrained transformers (e.g., BERT)
- ✅ Computationally efficient

### Cons
- ❌ Long sequence length (vision + text tokens)
- ❌ Quadratic attention complexity O((V+T)²)
- ❌ Less flexible than separate modality streams
- ❌ May struggle with very long sequences

### Best For
- Resource-constrained environments
- When simplicity is preferred
- Pretrained transformer backbone available
- Tasks requiring tight vision-language coupling

### References
- Kim et al., "ViLT: Vision-and-Language Transformer Without Convolution or Region Supervision", ICML 2021
- Inspired by BERT's approach to sentence pairs

---

## Comparison Table

| Feature | Cross-Attention | Q-Former | Single-Stream |
|---------|----------------|----------|---------------|
| **Architecture** | Dual-stream | Query-based | Unified stream |
| **Complexity** | O(V·T + V² + T²) | O(Q·V + Q·T + Q²) | O((V+T)²) |
| **Parameters** | High | Medium | Low |
| **Efficiency** | Medium | High | Medium |
| **Flexibility** | High | Medium | Low |
| **Interpretability** | High | Medium | Low |
| **Best For** | Complex reasoning | Large-scale pretraining | Simple tasks |

Where: V = vision tokens, T = text tokens, Q = query tokens (typically Q << V+T)

---

## Factory Function

Easy creation with a single function:

```python
from src.modeling.fusion import create_fusion_model

# Cross-Attention
fusion1 = create_fusion_model(
    'cross_attention',
    vision_dim=768,
    text_dim=768,
    num_layers=4
)

# Q-Former
fusion2 = create_fusion_model(
    'qformer',
    vision_dim=768,
    text_dim=768,
    num_query_tokens=32
)

# Single-Stream (ViLT)
fusion3 = create_fusion_model(
    'single_stream',
    vision_dim=768,
    text_dim=768
)
```

---

## Integration with VQA Model

```python
from src.modeling.heads import create_image_representation, create_text_representation
from src.modeling.fusion import create_fusion_model

class VQAModel(nn.Module):
    def __init__(self):
        super().__init__()
        
        # Vision encoder
        self.vision_encoder = create_image_representation(
            model_type='vision_token',
            num_vision_tokens=32,
            token_dim=768
        )
        
        # Text encoder
        self.text_encoder = create_text_representation(
            model_type='deberta-v3',
            pooling_strategy='all',  # Get all tokens
            output_dim=768
        )
        
        # Fusion module
        self.fusion = create_fusion_model(
            'qformer',  # or 'cross_attention', 'single_stream'
            vision_dim=768,
            text_dim=768,
            output_dim=768
        )
        
        # Answer prediction head
        self.answer_head = nn.Linear(768, num_answers)
    
    def forward(self, images, input_ids, attention_mask):
        # Extract features
        vision_features = self.vision_encoder(images)
        text_features = self.text_encoder(input_ids, attention_mask)
        
        # Fuse modalities
        fused_features = self.fusion(vision_features, text_features)
        
        # Predict answer
        logits = self.answer_head(fused_features)
        return logits
```

---

## Performance Tips

1. **Start Simple**: Begin with Single-Stream, upgrade if needed
2. **Use Q-Former**: For large-scale models or efficiency requirements
3. **Cross-Attention**: When you need explicit cross-modal reasoning
4. **Tune Query Tokens**: 16-64 queries work well for most tasks
5. **Layer Depth**: 4-6 layers usually sufficient
6. **Attention Heads**: Match encoder architecture (8 or 12 heads)

---

## Advanced Features

### Attention Masks

All fusion models support attention masks for variable-length sequences:

```python
# Create masks (True = valid token, False = padding)
vision_mask = torch.ones(batch_size, num_vision_tokens, dtype=torch.bool)
text_mask = torch.ones(batch_size, num_text_tokens, dtype=torch.bool)
text_mask[:, 15:] = False  # Mask padding tokens

# Forward with masks
output = fusion(vision_features, text_features, vision_mask, text_mask)
```

### Custom Fusion Methods (Cross-Attention)

```python
# Concatenation (default)
fusion = CrossAttentionFusion(fusion_method='concat')

# Element-wise addition
fusion = CrossAttentionFusion(fusion_method='add')

# Element-wise multiplication (Hadamard product)
fusion = CrossAttentionFusion(fusion_method='multiply')
```

### Variable Output Dimensions

```python
# Input: vision_dim=2048, text_dim=768
# Output: output_dim=512
fusion = create_fusion_model(
    'qformer',
    vision_dim=2048,
    text_dim=768,
    output_dim=512
)
```

---

## Choosing the Right Approach

### Use **Cross-Attention** if:
- You need explicit cross-modal reasoning
- Interpretability is important
- You have sufficient compute resources
- Multiple cross-modal interactions needed

### Use **Q-Former** if:
- Efficiency is critical
- Connecting frozen pretrained models
- Large-scale pretraining
- You need fixed-size representations

### Use **Single-Stream** if:
- Simplicity is preferred
- Limited compute budget
- Can leverage pretrained transformers
- Task requires tight vision-language coupling

---

## References

1. **Cross-Attention:**
   - Lu et al., "ViLBERT: Pretraining Task-Agnostic Visiolinguistic Representations for Vision-and-Language Tasks", NeurIPS 2019
   - Tan & Bansal, "LXMERT: Learning Cross-Modality Encoder Representations from Transformers", EMNLP 2019

2. **Q-Former:**
   - Li et al., "BLIP-2: Bootstrapping Language-Image Pre-training with Frozen Image Encoders and Large Language Models", ICML 2023
   - Jaegle et al., "Perceiver: General Perception with Iterative Attention", ICML 2021
   - Alayrac et al., "Flamingo: a Visual Language Model for Few-Shot Learning", NeurIPS 2022

3. **Single-Stream:**
   - Kim et al., "ViLT: Vision-and-Language Transformer Without Convolution or Region Supervision", ICML 2021
   - Devlin et al., "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding", NAACL 2019
