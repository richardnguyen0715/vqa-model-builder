# Text Representation Approaches for VQA

This module implements 4 different approaches for extracting textual features from questions for Visual Question Answering tasks.

## Overview

All text representation models are based on pretrained transformer models from HuggingFace and support:
- **Multiple pooling strategies**: CLS token, mean pooling, max pooling, or all tokens
- **Finetuning control**: Freeze or finetune pretrained weights
- **Projection layers**: Optional projection to match desired output dimension
- **Flexible integration**: Easy-to-use factory function

---

## 1. BERT Text Embedding (`BERTTextEmbedding`)

**Approach**: Bidirectional Encoder Representations from Transformers

**Key Features**:
- Bidirectional context understanding (reads left-to-right AND right-to-left)
- Masked Language Modeling (MLM) pretraining
- Next Sentence Prediction (NSP) task
- Strong baseline for many NLP tasks

**Usage**:
```python
from src.modeling.heads import BERTTextEmbedding

model = BERTTextEmbedding(
    model_name='bert-base-uncased',  # or 'bert-large-uncased'
    output_dim=768,
    pooling_strategy='cls',  # 'cls', 'mean', 'max', or 'all'
    finetune=True,
    add_projection=False
)

# Input: tokenized text
input_ids = torch.tensor([[101, 2054, 2003, ...]])  # [CLS] what is ...
attention_mask = torch.ones_like(input_ids)

# Output: text features
features = model(input_ids, attention_mask)  # [batch_size, 768]
```

**Pros**:
- Well-established and widely used
- Many pretrained variants available
- Good performance on diverse tasks

**Cons**:
- NSP task may not always be useful
- Training can be improved (see RoBERTa)

**Recommended for**: General VQA tasks, when you need a reliable baseline

---

## 2. RoBERTa Text Embedding (`RoBERTaTextEmbedding`)

**Approach**: Robustly Optimized BERT Approach

**Key Features**:
- Trained on 10x more data than BERT
- Removed Next Sentence Prediction (NSP) task
- Dynamic masking during training
- Larger batch sizes and longer training
- Better performance than BERT

**Usage**:
```python
from src.modeling.heads import RoBERTaTextEmbedding

model = RoBERTaTextEmbedding(
    model_name='roberta-base',  # or 'roberta-large'
    output_dim=768,
    pooling_strategy='mean',
    finetune=True,
    add_projection=False
)

features = model(input_ids, attention_mask)
```

**Pros**:
- Improved performance over BERT
- More robust training procedure
- Better representations

**Cons**:
- Slightly larger model size
- More computational cost

**Recommended for**: When you need better performance than BERT, production VQA systems

---

## 3. DeBERTa V3 Text Embedding (`DeBERTaV3TextEmbedding`)

**Approach**: Decoding-enhanced BERT with Disentangled Attention

**Key Features**:
- **Disentangled Attention**: Separates content and position representations
- **Enhanced Mask Decoder**: Better masked language modeling
- **ELECTRA-style pretraining**: Replaced token detection
- State-of-the-art performance on many benchmarks

**Usage**:
```python
from src.modeling.heads import DeBERTaV3TextEmbedding

model = DeBERTaV3TextEmbedding(
    model_name='microsoft/deberta-v3-base',  # or 'microsoft/deberta-v3-large'
    output_dim=768,
    pooling_strategy='cls',
    finetune=True,
    add_projection=False
)

features = model(input_ids, attention_mask, token_type_ids)
```

**Technical Details**:
- **Content vs Position**: Each token has separate vectors for:
  - Content: What the token means
  - Position: Where the token is located
- **Relative Position Encoding**: Better handling of long sequences
- **Enhanced Decoder**: Improves prediction quality

**Pros**:
- Best performance among the three
- Better long-range dependencies
- More parameter-efficient

**Cons**:
- Slightly more complex
- Newer (less community support than BERT/RoBERTa)

**Recommended for**: When you need state-of-the-art performance, complex reasoning tasks

---

## 4. Generic Transformer Text Embedding (`GenericTransformerTextEmbedding`)

**Approach**: Flexible wrapper for any HuggingFace transformer model

**Supported Models**:
- **DistilBERT**: Smaller, faster BERT (40% less parameters, 60% faster)
- **ALBERT**: Parameter-sharing BERT variant
- **ELECTRA**: Discriminator-based pretraining
- **XLNet**: Permutation-based language modeling
- **T5**: Text-to-text framework
- And many more from HuggingFace Model Hub

**Usage**:
```python
from src.modeling.heads import GenericTransformerTextEmbedding

# DistilBERT (fast and efficient)
model = GenericTransformerTextEmbedding(
    model_name='distilbert-base-uncased',
    output_dim=768,
    pooling_strategy='mean',
    finetune=True
)

# ALBERT (parameter efficient)
model = GenericTransformerTextEmbedding(
    model_name='albert-base-v2',
    output_dim=768
)

# ELECTRA (strong performance)
model = GenericTransformerTextEmbedding(
    model_name='google/electra-base-discriminator',
    output_dim=768
)

features = model(input_ids, attention_mask)
```

**Pros**:
- Maximum flexibility
- Access to all HuggingFace models
- Can use latest models

**Cons**:
- Need to understand each model's specifics
- Some models may have different APIs

**Recommended for**: Experimentation, using specialized models, accessing new releases

---

## Pooling Strategies

All models support multiple ways to aggregate token embeddings:

```python
# 1. CLS token (default for classification)
model = BERTTextEmbedding(pooling_strategy='cls')
# Output: [batch_size, hidden_size]

# 2. Mean pooling (average all tokens)
model = BERTTextEmbedding(pooling_strategy='mean')
# Output: [batch_size, hidden_size]

# 3. Max pooling (take maximum over tokens)
model = BERTTextEmbedding(pooling_strategy='max')
# Output: [batch_size, hidden_size]

# 4. All tokens (for attention mechanisms)
model = BERTTextEmbedding(pooling_strategy='all')
# Output: [batch_size, seq_len, hidden_size]
```

---

## Factory Function

Easy creation with a single function:

```python
from src.modeling.heads import create_text_representation

# BERT
bert = create_text_representation(
    model_type='bert',
    model_name='bert-base-uncased'
)

# RoBERTa
roberta = create_text_representation(
    model_type='roberta',
    model_name='roberta-large'
)

# DeBERTa V3
deberta = create_text_representation(
    model_type='deberta-v3',
    model_name='microsoft/deberta-v3-base'
)

# Generic (any HuggingFace model)
distilbert = create_text_representation(
    model_type='generic',
    model_name='distilbert-base-uncased'
)

# Or directly use model name as type
model = create_text_representation('sentence-transformers/all-MiniLM-L6-v2')
```

---

## Comparison Table

| Model | Size (Base) | Speed | Performance | Best For |
|-------|------------|-------|-------------|----------|
| **BERT** | 110M | Medium | Good | General baseline |
| **RoBERTa** | 125M | Medium | Better | Production systems |
| **DeBERTa V3** | 86M | Slower | Best | SOTA performance |
| **DistilBERT** | 66M | Fast | Good | Speed-critical apps |
| **ALBERT** | 12M | Fast | Good | Memory constraints |

---

## Integration with VQA Model

```python
from src.modeling.heads import create_text_representation, create_image_representation

class VQAModel(nn.Module):
    def __init__(self):
        super().__init__()
        
        # Text encoder
        self.text_encoder = create_text_representation(
            model_type='deberta-v3',
            model_name='microsoft/deberta-v3-base',
            pooling_strategy='all',  # Get all tokens for attention
            finetune=True
        )
        
        # Image encoder
        self.vision_encoder = create_image_representation(
            model_type='vision_token',
            num_vision_tokens=32
        )
        
        # Multimodal fusion
        self.fusion = MultimodalFusionModule(...)
        
    def forward(self, images, input_ids, attention_mask):
        # Extract features
        text_features = self.text_encoder(input_ids, attention_mask)
        vision_features = self.vision_encoder(images)
        
        # Fuse and predict
        output = self.fusion(vision_features, text_features)
        return output
```

---

## Advanced Features

### 1. Finetuning Control

```python
# Freeze encoder (feature extraction only)
model = BERTTextEmbedding(finetune=False)

# Finetune encoder (end-to-end training)
model = BERTTextEmbedding(finetune=True)
```

### 2. Dimension Projection

```python
# Add projection layer to match specific dimension
model = RoBERTaTextEmbedding(
    output_dim=1024,
    add_projection=True
)
```

### 3. Getting Tokenizer

```python
model = create_text_representation('bert')
tokenizer = model.get_tokenizer()

# Tokenize questions
questions = ["What color is the car?", "How many people?"]
encoded = tokenizer(
    questions,
    padding=True,
    truncation=True,
    max_length=128,
    return_tensors='pt'
)

# Forward
features = model(encoded['input_ids'], encoded['attention_mask'])
```

---

## Performance Tips

1. **Start with BERT**: Good baseline, well-documented
2. **Upgrade to RoBERTa**: Easy drop-in replacement, better performance
3. **Try DeBERTa V3**: If you need SOTA and have compute budget
4. **Use DistilBERT**: For real-time applications or edge devices
5. **Experiment with pooling**: 'cls' for simple tasks, 'mean' for robustness, 'all' for attention

---

## References

1. Devlin et al., "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding", NAACL 2019
2. Liu et al., "RoBERTa: A Robustly Optimized BERT Pretraining Approach", arXiv 2019
3. He et al., "DeBERTaV3: Improving DeBERTa using ELECTRA-Style Pre-Training with Gradient-Disentangled Embedding Sharing", ICLR 2023
4. Sanh et al., "DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter", NeurIPS 2019
