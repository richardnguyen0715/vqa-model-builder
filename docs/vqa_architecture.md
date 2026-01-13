# Vietnamese VQA Model Architecture

This document provides a comprehensive overview of the Vietnamese Visual Question Answering (VQA) model architecture with Mixture of Experts (MOE), Knowledge Base/RAG integration, and complete training/inference/evaluation pipelines.

## Table of Contents

1. [Overview](#overview)
2. [Architecture Components](#architecture-components)
3. [Module Structure](#module-structure)
4. [Configuration System](#configuration-system)
5. [Usage Examples](#usage-examples)

---

## Overview

The Vietnamese VQA system is designed specifically for answering questions about images in Vietnamese. Key features include:

- **Mixture of Experts (MOE)**: Dynamically routes inputs to specialized expert networks
- **Knowledge Base/RAG**: Retrieval-Augmented Generation for external knowledge
- **Vietnamese NLP**: Optimized for Vietnamese text with PhoBERT integration
- **Modular Design**: Easily swappable components and configurations

### System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    Vietnamese VQA Model                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────┐     ┌──────────────┐     ┌──────────────┐    │
│  │   Image      │     │   Question   │     │  Knowledge   │    │
│  │   Input      │     │   Input      │     │    Base      │    │
│  └──────┬───────┘     └──────┬───────┘     └──────┬───────┘    │
│         │                    │                    │              │
│         ▼                    ▼                    ▼              │
│  ┌──────────────┐     ┌──────────────┐     ┌──────────────┐    │
│  │   Visual     │     │    Text      │     │     RAG      │    │
│  │   Encoder    │     │   Encoder    │     │   Module     │    │
│  │  (ViT/CLIP)  │     │  (PhoBERT)   │     │  (Retriever) │    │
│  └──────┬───────┘     └──────┬───────┘     └──────┬───────┘    │
│         │                    │                    │              │
│         └────────────┬───────┴────────────┬──────┘              │
│                      │                    │                      │
│                      ▼                    ▼                      │
│              ┌──────────────────────────────────┐               │
│              │     Multimodal Fusion            │               │
│              │  (Cross-Attention / Concat)      │               │
│              └──────────────┬───────────────────┘               │
│                             │                                    │
│                             ▼                                    │
│              ┌──────────────────────────────────┐               │
│              │    Mixture of Experts (MOE)      │               │
│              │  ┌────┐ ┌────┐ ┌────┐ ┌────┐   │               │
│              │  │ E1 │ │ E2 │ │ E3 │ │ E4 │   │               │
│              │  └────┘ └────┘ └────┘ └────┘   │               │
│              └──────────────┬───────────────────┘               │
│                             │                                    │
│                             ▼                                    │
│              ┌──────────────────────────────────┐               │
│              │       Answer Head                │               │
│              │   (Classification / Generation)  │               │
│              └──────────────┬───────────────────┘               │
│                             │                                    │
│                             ▼                                    │
│                      ┌──────────┐                               │
│                      │  Answer  │                               │
│                      └──────────┘                               │
└─────────────────────────────────────────────────────────────────┘
```

---

## Architecture Components

### 1. Visual Encoder

Extracts visual features from input images.

**Supported Backbones:**
- `VIT_BASE`: Vision Transformer Base (default)
- `VIT_LARGE`: Vision Transformer Large
- `RESNET`: ResNet variants
- `CLIP`: CLIP visual encoder
- `SWIN`: Swin Transformer

```python
from src.modeling.meta_arch import VisualEncoderConfig, BackboneType

config = VisualEncoderConfig(
    backbone=BackboneType.VIT_BASE,
    pretrained=True,
    hidden_dim=768,
    freeze_backbone=False
)
```

### 2. Text Encoder

Encodes Vietnamese questions using pretrained language models.

**Supported Encoders:**
- `PHOBERT`: PhoBERT (vinai/phobert-base) - Recommended for Vietnamese
- `BERT`: Multilingual BERT
- `XLM_ROBERTA`: XLM-RoBERTa

```python
from src.modeling.meta_arch import TextEncoderConfig, TextEncoderType

config = TextEncoderConfig(
    encoder_type=TextEncoderType.PHOBERT,
    pretrained=True,
    hidden_dim=768,
    max_length=64
)
```

### 3. Multimodal Fusion

Combines visual and textual features.

**Fusion Types:**
- `CONCAT`: Simple concatenation
- `ELEMENT_WISE`: Element-wise multiplication
- `BILINEAR`: Bilinear pooling
- `CROSS_ATTENTION`: Cross-modal attention (recommended)
- `MCAN`: Deep modular co-attention

```python
from src.modeling.meta_arch import FusionConfig, FusionType

config = FusionConfig(
    fusion_type=FusionType.CROSS_ATTENTION,
    hidden_dim=768,
    num_heads=8,
    num_layers=2
)
```

### 4. Mixture of Experts (MOE)

Dynamically routes inputs to specialized expert networks.

```python
from src.modeling.meta_arch import MOEConfig

config = MOEConfig(
    use_moe=True,
    num_experts=8,
    top_k=2,
    capacity_factor=1.25,
    router_type="top_k",
    expert_type="ffn"
)
```

**Expert Types:**
- `ffn`: Feed-forward networks
- `sam`: SAM-based experts for segmentation
- `detection`: Detection-focused experts
- `ocr`: OCR-specialized experts

### 5. Knowledge Base / RAG

Retrieval-Augmented Generation for external knowledge.

```python
from src.modeling.meta_arch import KnowledgeConfig

config = KnowledgeConfig(
    use_knowledge=True,
    retriever_type="dense",
    num_retrieved=5,
    knowledge_dim=768
)
```

**Retriever Types:**
- `dense`: Dense retriever using embeddings
- `sparse`: BM25-based sparse retriever
- `hybrid`: Combination of dense and sparse

---

## Module Structure

```
src/
├── modeling/
│   ├── meta_arch/
│   │   ├── vqa_config.py      # Configuration dataclasses
│   │   ├── vqa_model.py       # Main VQA model
│   │   └── __init__.py
│   │
│   ├── inference/
│   │   ├── inference_config.py # Inference settings
│   │   ├── vqa_predictor.py   # Inference engine
│   │   └── __init__.py
│   │
│   ├── backbone/              # Visual backbones
│   ├── fusion/                # Fusion approaches
│   ├── heads/                 # Encoder heads
│   └── tokenizer/             # Tokenizers
│
├── pipeline/
│   ├── trainer/
│   │   ├── trainer_config.py  # Training configuration
│   │   ├── training_utils.py  # Training utilities
│   │   ├── vqa_trainer.py     # Main trainer
│   │   └── __init__.py
│   │
│   └── evaluator/
│       ├── evaluator_config.py # Evaluation configuration
│       ├── vqa_evaluator.py   # Evaluation engine
│       └── __init__.py
│
├── solvers/
│   ├── losses/                # Loss functions
│   ├── metrics/               # Evaluation metrics
│   └── optimizers/            # Optimizers and schedulers
│
└── knowledge_base/
    ├── vietnamese_processor.py # Vietnamese NLP
    ├── document_store.py      # Document storage
    ├── vector_store.py        # Vector databases
    ├── knowledge_encoder.py   # Encoding
    ├── retriever.py          # Retrievers
    ├── rag_module.py         # RAG integration
    └── kb_utils.py           # Utilities
```

---

## Configuration System

All components use dataclass-based configurations for type safety and IDE support.

### Model Configuration

```python
from src.modeling.meta_arch import VQAModelConfig, get_default_vietnamese_vqa_config

# Get default configuration optimized for Vietnamese
config = get_default_vietnamese_vqa_config()

# Or create custom configuration
config = VQAModelConfig(
    visual_encoder=VisualEncoderConfig(...),
    text_encoder=TextEncoderConfig(...),
    fusion=FusionConfig(...),
    moe=MOEConfig(...),
    knowledge=KnowledgeConfig(...),
    answer_head=AnswerHeadConfig(...)
)
```

### Training Configuration

```python
from src.pipeline.trainer import TrainingConfig, get_default_training_config

config = get_default_training_config()

# Key settings
config.num_epochs = 20
config.optimizer.lr = 2e-5
config.mixed_precision = MixedPrecisionMode.FP16
config.strategy = TrainingStrategy.GRADUAL_UNFREEZE
```

### Inference Configuration

```python
from src.modeling.inference import VQAInferenceConfig, get_default_inference_config

config = get_default_inference_config()

# Key settings
config.inference.use_fp16 = True
config.answer.decoding_strategy = DecodingStrategy.GREEDY
config.answer.return_top_n = 5
```

---

## Usage Examples

### Quick Start

```python
from src.modeling import create_vqa_model, get_default_vietnamese_vqa_config
from src.pipeline import create_trainer, get_default_training_config

# 1. Create model
model_config = get_default_vietnamese_vqa_config()
model = create_vqa_model(model_config)

# 2. Create trainer
train_config = get_default_training_config()
trainer = create_trainer(
    model=model,
    config=train_config,
    train_dataloader=train_loader,
    val_dataloader=val_loader
)

# 3. Train
results = trainer.train()

# 4. Inference
from src.modeling.inference import VQAPredictor

predictor = VQAPredictor(model)
result = predictor.predict(image, "Có bao nhiêu người trong hình?")
print(f"Answer: {result.answer}")
```

### With Knowledge Base

```python
from src.knowledge_base import (
    DocumentStore,
    FAISSVectorStore,
    DenseRetriever,
    RAGModule
)

# Setup knowledge base
doc_store = DocumentStore()
doc_store.add_documents([
    Document(text="Việt Nam có 54 dân tộc...", metadata={"topic": "vietnam"}),
    # ... more documents
])

# Create RAG module
rag = RAGModule(
    retriever=DenseRetriever(doc_store, vector_store),
    fusion_type="attention"
)

# Integrate with model
model_config.knowledge.use_knowledge = True
model = create_vqa_model(model_config, rag_module=rag)
```

### Evaluation

```python
from src.pipeline.evaluator import create_evaluator, EvaluationConfig

eval_config = EvaluationConfig(
    mode=EvaluationMode.DETAILED,
    metrics=MetricConfig(
        compute_accuracy=True,
        compute_wups=True,
        compute_per_type=True
    )
)

evaluator = create_evaluator(config=eval_config)
results = evaluator.evaluate(model, test_loader)
evaluator.print_summary(results)
```

---

## Key Features

### Vietnamese Language Support

- **PhoBERT Integration**: Native Vietnamese text encoding
- **Word Segmentation**: Using underthesea for Vietnamese tokenization
- **Vietnamese Question Types**: Automatic detection of question types:
  - Count (bao nhiêu, mấy)
  - Color (màu gì)
  - Location (ở đâu)
  - Yes/No (có phải, có không)
  - And more...

### Training Features

- **Mixed Precision**: FP16/BF16 training support
- **Gradient Accumulation**: For larger effective batch sizes
- **Gradient Checkpointing**: Memory-efficient training
- **Learning Rate Scheduling**: Cosine, linear, polynomial warmup
- **Early Stopping**: Automatic training termination
- **Checkpointing**: Save best and periodic checkpoints

### Inference Features

- **Multiple Decoding Strategies**: Greedy, top-k, top-p, beam search
- **Batch Inference**: Efficient batch processing
- **Attention Visualization**: Return attention weights for interpretation
- **Confidence Thresholding**: Filter low-confidence predictions

### Evaluation Features

- **Multiple Metrics**: Accuracy, WUPS, BLEU, F1
- **Per-Type Analysis**: Breakdown by question type
- **Error Analysis**: Detailed error examination
- **Visualization Support**: Generate analysis plots

---

## Requirements

```
torch>=2.0.0
transformers>=4.30.0
torchvision>=0.15.0
underthesea>=6.0.0
faiss-cpu>=1.7.0  # or faiss-gpu
chromadb>=0.4.0
pillow>=9.0.0
numpy>=1.24.0
```

---

## Citation

If you use this system, please cite:

```bibtex
@software{vietnamese_vqa,
  title = {Vietnamese VQA with MOE and RAG},
  year = {2024},
  description = {Visual Question Answering system optimized for Vietnamese}
}
```
