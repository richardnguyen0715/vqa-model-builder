# Pipeline Usage Guide

This document provides comprehensive instructions for using the AutoViVQA Pipeline system.

## Table of Contents

1. [Overview](#overview)
2. [Pipeline Components](#pipeline-components)
3. [Data Pipeline](#data-pipeline)
4. [Model Pipeline](#model-pipeline)
5. [Training Pipeline](#training-pipeline)
6. [CLI Reference](#cli-reference)
7. [Configuration Guide](#configuration-guide)
8. [Troubleshooting](#troubleshooting)

---

## Overview

The AutoViVQA Pipeline is a modular system consisting of three main stages:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          VQA PIPELINE ORCHESTRATOR                          │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐       │
│   │  DATA PIPELINE  │───▶│ MODEL PIPELINE  │───▶│TRAINING PIPELINE│       │
│   │                 │    │                 │    │                 │       │
│   │ • Load data     │    │ • Build model   │    │ • Train loop    │       │
│   │ • Validate      │    │ • Init weights  │    │ • Validation    │       │
│   │ • Split         │    │ • Move to GPU   │    │ • Checkpointing │       │
│   │ • Build vocab   │    │ • Validate      │    │ • Early stop    │       │
│   │ • Tokenize      │    │                 │    │                 │       │
│   │ • Transforms    │    │                 │    │                 │       │
│   │ • DataLoaders   │    │                 │    │                 │       │
│   └─────────────────┘    └─────────────────┘    └─────────────────┘       │
│           │                      │                      │                  │
│           ▼                      ▼                      ▼                  │
│   ┌─────────────────────────────────────────────────────────────────────┐  │
│   │                    COMPREHENSIVE LOGGING                             │  │
│   │   • Progress bars • Validation checks • Metrics • Architecture      │  │
│   └─────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Pipeline Components

### VQAPipeline (Orchestrator)

The main orchestrator that coordinates all pipeline stages.

**Location:** `src/core/vqa_pipeline.py`

```python
from src.core import VQAPipeline, VQAPipelineConfig

# Create configuration
config = VQAPipelineConfig(
    mode="train",
    output_dir="outputs",
    data=DataPipelineConfig(...),
    model=ModelPipelineConfig(...),
    training=TrainingPipelineConfig(...),
)

# Run pipeline
pipeline = VQAPipeline(config)
results = pipeline.run()
```

---

## Data Pipeline

### Overview

The Data Pipeline handles all data-related operations:

1. **Load Raw Data** - Load images and text annotations
2. **Validate Data** - Check for missing/corrupted data
3. **Compute Statistics** - Image shapes, question lengths, answer distribution
4. **Split Data** - Train/Val/Test split
5. **Build Vocabulary** - Answer-to-ID mapping
6. **Initialize Tokenizer** - PhoBERT tokenizer
7. **Build Transforms** - Image augmentation
8. **Create DataLoaders** - PyTorch DataLoaders

### Configuration

```python
from src.core import DataPipelineConfig

config = DataPipelineConfig(
    # Data paths
    images_dir="data/raw/images",
    text_file="data/raw/texts/evaluate_60k_data_balanced_preprocessed.csv",
    
    # Split ratios
    train_ratio=0.8,
    val_ratio=0.1,
    test_ratio=0.1,
    
    # DataLoader settings
    batch_size=32,
    eval_batch_size=64,
    num_workers=4,
    pin_memory=True,
    
    # Image settings
    image_size=(224, 224),
    
    # Augmentation: "light", "medium", "strong"
    augmentation_strength="medium",
    
    # Tokenizer
    tokenizer_name="vinai/phobert-base",
    max_seq_length=64,
    
    # Answer vocabulary
    min_answer_freq=5,  # Minimum frequency for answers
)
```

### Standalone Usage

```python
from src.core import DataPipeline, DataPipelineConfig

# Create config
config = DataPipelineConfig(
    images_dir="data/raw/images",
    batch_size=32,
)

# Run data pipeline
data_pipeline = DataPipeline(config)
output = data_pipeline.run()

# Access outputs
print(f"Train samples: {len(output.train_data)}")
print(f"Answer classes: {output.num_classes}")
print(f"Tokenizer vocab: {output.tokenizer.vocab_size}")

# Use dataloaders
for batch in output.train_loader:
    images = batch['image']
    input_ids = batch['input_ids']
    labels = batch['label']
    break
```

### Output Structure

```python
@dataclass
class DataPipelineOutput:
    train_data: List[OneSample]
    val_data: List[OneSample]
    test_data: List[OneSample]
    train_loader: DataLoader
    val_loader: DataLoader
    test_loader: DataLoader
    answer2id: Dict[str, int]
    id2answer: Dict[int, str]
    num_classes: int
    tokenizer: Any
    train_transform: transforms.Compose
    eval_transform: transforms.Compose
```

---

## Model Pipeline

### Overview

The Model Pipeline creates and configures the VQA model:

1. **Device Setup** - Auto-detect CUDA/CPU
2. **Build Configuration** - Create VQAModelConfig
3. **Create Model** - Instantiate VietnameseVQAModel
4. **Initialize Weights** - Xavier initialization for new layers
5. **Move to Device** - Transfer to GPU
6. **Log Architecture** - Parameter counts, module breakdown
7. **Validate Model** - Test forward pass

### Configuration

```python
from src.core import ModelPipelineConfig

config = ModelPipelineConfig(
    # Visual encoder
    visual_backbone="vit",  # "vit", "resnet", "clip", "swin"
    visual_model_name="openai/clip-vit-base-patch32",
    visual_output_dim=768,
    freeze_visual=False,
    
    # Text encoder
    text_encoder_type="phobert",  # "phobert", "bert"
    text_model_name="vinai/phobert-base",
    text_output_dim=768,
    text_max_length=64,
    freeze_text=False,
    
    # Fusion
    fusion_type="cross_attention",  # "cross_attention", "concat", "bilinear"
    fusion_hidden_dim=768,
    fusion_num_heads=8,
    fusion_num_layers=2,
    fusion_dropout=0.1,
    
    # Mixture of Experts (MOE)
    use_moe=False,
    moe_num_experts=8,
    moe_top_k=2,
    moe_hidden_dim=2048,
    
    # Knowledge Base / RAG
    use_knowledge=False,
    knowledge_num_contexts=5,
    
    # Answer head
    answer_hidden_dims=[768, 512],
    answer_dropout=0.3,
    
    # General
    embed_dim=768,
    dropout=0.1,
    device="auto",  # "auto", "cuda", "cpu"
)
```

### Supported Configurations

#### Visual Backbones

| Backbone | Model Name | Output Dim | Notes |
|----------|------------|------------|-------|
| `vit` | openai/clip-vit-base-patch32 | 768 | Default, CLIP-pretrained |
| `vit` | google/vit-base-patch16-224 | 768 | ImageNet pretrained |
| `resnet` | resnet50 | 2048 | Classic CNN |
| `clip` | openai/clip-vit-large-patch14 | 1024 | Larger CLIP |
| `swin` | microsoft/swin-base-patch4-window7-224 | 1024 | Swin Transformer |

#### Text Encoders

| Encoder | Model Name | Output Dim | Notes |
|---------|------------|------------|-------|
| `phobert` | vinai/phobert-base | 768 | Vietnamese BERT (default) |
| `phobert` | vinai/phobert-large | 1024 | Larger Vietnamese |
| `bert` | bert-base-multilingual-cased | 768 | Multilingual |

#### Fusion Types

| Type | Description | Best For |
|------|-------------|----------|
| `cross_attention` | Bidirectional cross-attention | Complex reasoning |
| `concat` | Simple concatenation + MLP | Speed, simplicity |
| `bilinear` | Bilinear interaction | Moderate complexity |

### Standalone Usage

```python
from src.core import ModelPipeline, ModelPipelineConfig

config = ModelPipelineConfig(
    visual_backbone="vit",
    text_encoder_type="phobert",
    use_moe=True,
)

# Run model pipeline
model_pipeline = ModelPipeline(config)
output = model_pipeline.run(num_answers=447)

# Access model
model = output.model
device = output.device

# Print architecture
print(f"Parameters: {output.total_params:,}")
print(f"Trainable: {output.trainable_params:,}")
```

---

## Training Pipeline

### Overview

The Training Pipeline handles the training loop:

1. **Set Random Seed** - Reproducibility
2. **Setup Optimizer** - AdamW with weight decay
3. **Setup Scheduler** - Cosine/Linear LR schedule
4. **Setup Mixed Precision** - AMP for faster training
5. **Training Loop** - Epoch-based training
6. **Validation** - Per-epoch evaluation
7. **Checkpointing** - Save best model
8. **Early Stopping** - Stop if no improvement

### Configuration

```python
from src.core import TrainingPipelineConfig

config = TrainingPipelineConfig(
    # Training settings
    num_epochs=20,
    gradient_accumulation_steps=2,
    max_grad_norm=1.0,
    
    # Optimizer
    optimizer_name="adamw",  # "adam", "adamw", "sgd"
    learning_rate=2e-5,
    weight_decay=0.01,
    
    # Scheduler
    scheduler_name="cosine",  # "cosine", "linear", "step"
    warmup_ratio=0.1,
    
    # Mixed precision
    use_amp=True,
    amp_dtype="float16",  # "float16", "bfloat16"
    
    # Early stopping
    early_stopping=True,
    patience=5,
    min_delta=0.001,
    
    # Checkpointing
    checkpoint_dir="checkpoints",
    save_best=True,
    save_every_epoch=True,
    metric_for_best="accuracy",
    
    # Logging
    log_interval=50,
    
    # Reproducibility
    seed=42,
)
```

### Training Loop Details

```
For each epoch:
    1. Training Phase
       ├── Forward pass (with AMP)
       ├── Compute loss
       ├── Backward pass (scaled gradients)
       ├── Gradient clipping
       ├── Optimizer step
       ├── Scheduler step
       └── Log metrics every N steps
    
    2. Validation Phase
       ├── Model eval mode
       ├── No gradient computation
       ├── Compute metrics
       └── Check early stopping
    
    3. Checkpointing
       ├── Save if best model
       └── Save epoch checkpoint
```

### Standalone Usage

```python
from src.core import TrainingPipeline, TrainingPipelineConfig

config = TrainingPipelineConfig(
    num_epochs=10,
    learning_rate=2e-5,
    use_amp=True,
)

# Requires model, train_loader, val_loader from previous pipelines
training_pipeline = TrainingPipeline(
    config=config,
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    device=device,
)

output = training_pipeline.run()

print(f"Best accuracy: {output.best_metric:.4f}")
print(f"Best model: {output.best_model_path}")
```

---

## CLI Reference

### Main Command

```bash
python -m src.core.vqa_pipeline [OPTIONS]
```

### Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--mode` | choice | train | `train`, `evaluate`, `inference` |
| `--config` | path | None | YAML configuration file |
| `--images-dir` | path | data/raw/images | Images directory |
| `--text-file` | path | data/raw/texts/*.csv | Text annotations |
| `--batch-size` | int | 32 | Training batch size |
| `--epochs` | int | 20 | Number of epochs |
| `--learning-rate` | float | 2e-5 | Learning rate |
| `--visual-backbone` | str | vit | Visual encoder type |
| `--text-encoder` | str | phobert | Text encoder type |
| `--use-moe` | flag | False | Enable MOE |
| `--use-knowledge` | flag | False | Enable RAG |
| `--output-dir` | path | outputs | Output directory |
| `--resume` | path | None | Resume from checkpoint |

### Examples

```bash
# Basic training
python -m src.core.vqa_pipeline --mode train --epochs 10

# Training with MOE
python -m src.core.vqa_pipeline --mode train --epochs 20 --use-moe

# Using config file
python -m src.core.vqa_pipeline --config configs/pipeline_config.yaml

# Resume training
python -m src.core.vqa_pipeline --mode train --resume checkpoints/checkpoint_epoch_5.pt

# Evaluation only
python -m src.core.vqa_pipeline --mode evaluate --resume checkpoints/best_model.pt
```

### Shell Scripts

```bash
# Main runner script
bash src/cli/run_pipeline.sh --mode train --epochs 10 --batch-size 16

# Clean output (no warnings)
bash src/cli/run_clean.sh --mode train --epochs 10

# Quick start (download + train)
bash src/cli/quick_start.sh

# Download data only
bash src/cli/download_data.sh
```

---

## Configuration Guide

### YAML Configuration

Create a YAML file with all settings:

```yaml
# my_config.yaml
mode: train
output_dir: outputs/my_experiment

data:
  images_dir: data/raw/images
  batch_size: 32
  augmentation_strength: medium

model:
  visual_backbone: vit
  text_encoder_type: phobert
  fusion_type: cross_attention
  use_moe: true
  moe_num_experts: 8

training:
  num_epochs: 20
  learning_rate: 2.0e-5
  use_amp: true
  early_stopping: true
```

Run with:
```bash
python -m src.core.vqa_pipeline --config my_config.yaml
```

### Environment Variables

```bash
# Set CUDA device
export CUDA_VISIBLE_DEVICES=0

# Disable warnings
export PYTHONWARNINGS="ignore::FutureWarning"

# Set number of threads
export OMP_NUM_THREADS=4
```

---

## Troubleshooting

### Common Issues

#### 1. CUDA Out of Memory

```
RuntimeError: CUDA out of memory
```

**Solutions:**
- Reduce batch size: `--batch-size 8`
- Enable gradient accumulation in config
- Use mixed precision: `use_amp: true`
- Freeze some layers: `freeze_visual: true`

#### 2. Memory Overflow (RAM)

```
MemoryOverflowException: RAM usage at 95%
```

**Solutions:**
- The pipeline uses lazy image loading by default
- Reduce `num_workers` in config
- Close other applications

#### 3. Slow Data Loading

**Solutions:**
- Increase `num_workers`: `num_workers: 8`
- Enable `pin_memory: true`
- Use SSD storage for data

#### 4. Poor Accuracy

**Solutions:**
- Increase epochs: `--epochs 30`
- Try different learning rates: `1e-5`, `5e-5`
- Enable MOE: `--use-moe`
- Try different fusion: `fusion_type: cross_attention`
- Increase augmentation: `augmentation_strength: strong`

#### 5. Training Instability

**Solutions:**
- Reduce learning rate: `--learning-rate 1e-5`
- Enable gradient clipping: `max_grad_norm: 0.5`
- Use warmup: `warmup_ratio: 0.1`
- Reduce batch size

### Debug Mode

Enable verbose logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Or in config
log_level: DEBUG
```

### GPU Monitoring

```bash
# Watch GPU usage
watch -n 1 nvidia-smi

# TensorBoard
tensorboard --logdir logs/tensorboard
```

---

## Best Practices

### 1. Start Small

```bash
# Test with 1 epoch first
python -m src.core.vqa_pipeline --mode train --epochs 1 --batch-size 8
```

### 2. Use Checkpointing

Always enable checkpointing to avoid losing progress:
```yaml
training:
  save_best: true
  save_every_epoch: true
```

### 3. Monitor Resources

The pipeline includes automatic resource monitoring. Check logs for warnings.

### 4. Reproducibility

Set seed for reproducible results:
```yaml
training:
  seed: 42
```

### 5. Experiment Tracking

Use different output directories for experiments:
```bash
python -m src.core.vqa_pipeline --output-dir outputs/exp_001
python -m src.core.vqa_pipeline --output-dir outputs/exp_002
```

---

## Next Steps

- Read [VQA Architecture](vqa_architecture.md) for model details
- Read [Fusion Approaches](fusion_approaches.md) for fusion strategies
- Read [MOE Approaches](moe_approaches.md) for expert routing
- Read [Knowledge Base](knowledge_base_approaches.md) for RAG integration
