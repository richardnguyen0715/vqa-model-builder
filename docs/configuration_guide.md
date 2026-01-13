# Configuration Guide

Complete guide for configuring AutoViVQA Model Builder.

---

## Table of Contents

- [Configuration Files](#configuration-files)
- [Pipeline Configuration](#pipeline-configuration)
- [Data Configuration](#data-configuration)
- [Model Configuration](#model-configuration)
- [Training Configuration](#training-configuration)
- [Resource Configuration](#resource-configuration)
- [Logging Configuration](#logging-configuration)
- [Examples](#examples)

---

## Configuration Files

The project uses YAML configuration files located in the `configs/` directory:

```
configs/
├── pipeline_config.yaml      # Main pipeline configuration
├── data_configs.yaml         # Data loading settings
├── model_configs.yaml        # Model architecture settings
├── training_configs.yaml     # Training hyperparameters
├── resource_configs.yaml     # Resource management
├── logging_configs.yaml      # Logging settings
├── checkpoint_configs.yaml   # Checkpointing settings
├── inference_configs.yaml    # Inference settings
└── warning_configs.yaml      # Warning suppression
```

---

## Pipeline Configuration

Main configuration file: `configs/pipeline_config.yaml`

```yaml
# =============================================================================
# PIPELINE CONFIGURATION
# =============================================================================

# Pipeline mode: train, evaluate, inference
mode: train

# Output directory for all artifacts
output_dir: outputs

# Random seed for reproducibility
seed: 42

# Debug mode (reduces dataset size for quick testing)
debug: false

# -----------------------------------------------------------------------------
# Data Pipeline Configuration
# -----------------------------------------------------------------------------
data:
  # Data directories
  data_dir: data/raw
  images_dir: data/raw/images
  
  # Data splitting ratios
  train_ratio: 0.8
  val_ratio: 0.1
  test_ratio: 0.1
  
  # DataLoader settings
  batch_size: 32
  num_workers: 4
  pin_memory: true
  
  # Preprocessing
  image_size: 224
  max_length: 128
  
  # Augmentation: none, light, medium, heavy
  augmentation_strength: medium
  
  # Vocabulary settings
  min_answer_freq: 5
  
  # Sampling (null for all data)
  max_samples: null

# -----------------------------------------------------------------------------
# Model Pipeline Configuration
# -----------------------------------------------------------------------------
model:
  # Visual backbone: vit, clip, resnet50
  visual_backbone: vit
  visual_pretrained: google/vit-base-patch16-224
  freeze_visual_encoder: true
  
  # Text encoder: phobert, bert
  text_encoder_type: phobert
  text_pretrained: vinai/phobert-base
  freeze_text_encoder: true
  
  # Fusion: concat, element_wise, cross_attention
  fusion_type: cross_attention
  
  # Model dimensions
  hidden_dim: 768
  num_heads: 8
  num_fusion_layers: 2
  dropout: 0.1
  
  # Mixture of Experts
  use_moe: false
  num_experts: 4
  top_k_experts: 2
  
  # Knowledge Base
  use_knowledge_base: false
  knowledge_dim: 768

# -----------------------------------------------------------------------------
# Training Pipeline Configuration
# -----------------------------------------------------------------------------
training:
  # Training duration
  num_epochs: 10
  
  # Optimization
  learning_rate: 2.0e-5
  weight_decay: 0.01
  max_grad_norm: 1.0
  
  # Optimizer: adam, adamw, sgd
  optimizer_type: adamw
  
  # Scheduler: linear, cosine, plateau
  scheduler_type: cosine
  warmup_ratio: 0.1
  
  # Mixed precision
  use_amp: true
  
  # Checkpointing
  save_best_only: true
  checkpoint_dir: checkpoints
  
  # Early stopping
  patience: 5
  
  # Resume training
  resume_from: null
```

---

## Data Configuration

### Data Directory Structure

Expected directory structure:
```
data/raw/
├── images/
│   ├── 000000000001.jpg
│   ├── 000000000002.jpg
│   └── ... (image files)
└── texts/
    └── evaluate_60k_data_balanced_preprocessed.csv
```

### CSV Format

The CSV file should have the following columns:
- `image_id`: Image filename (without extension)
- `question`: Vietnamese question text
- `answer`: Answer text

Example:
```csv
image_id,question,answer
000000000001,Đây là loại phương tiện gì?,xe máy
000000000002,Có bao nhiêu người trong ảnh?,2
```

### Data Splitting

Configure train/val/test ratios:

```yaml
data:
  train_ratio: 0.8   # 80% for training
  val_ratio: 0.1     # 10% for validation
  test_ratio: 0.1    # 10% for testing
```

**Note:** Ratios must sum to 1.0

### Augmentation Strength

| Level | Description | Use Case |
|-------|-------------|----------|
| `none` | No augmentation | Testing/debugging |
| `light` | Minimal transforms | Small datasets |
| `medium` | Standard augmentation | Default |
| `heavy` | Aggressive transforms | Large datasets |

```yaml
data:
  augmentation_strength: medium
```

### DataLoader Settings

```yaml
data:
  batch_size: 32      # Samples per batch
  num_workers: 4      # Parallel data loading workers
  pin_memory: true    # Pin memory for GPU transfer
```

**Recommendations:**
- `num_workers`: Set to number of CPU cores - 2
- `pin_memory`: Set `true` for GPU training
- `batch_size`: Adjust based on GPU memory

---

## Model Configuration

### Visual Backbones

| Backbone | Model ID | Parameters | Best For |
|----------|----------|------------|----------|
| `vit` | `google/vit-base-patch16-224` | 86M | General purpose |
| `clip` | `openai/clip-vit-base-patch32` | 86M | Multi-modal |
| `resnet50` | `resnet50` | 23M | Fast inference |

```yaml
model:
  visual_backbone: vit
  visual_pretrained: google/vit-base-patch16-224
  freeze_visual_encoder: true   # Freeze pretrained weights
```

### Text Encoders

| Encoder | Model ID | Language | Parameters |
|---------|----------|----------|------------|
| `phobert` | `vinai/phobert-base` | Vietnamese | 135M |
| `bert` | `bert-base-multilingual-cased` | Multilingual | 110M |

```yaml
model:
  text_encoder_type: phobert
  text_pretrained: vinai/phobert-base
  freeze_text_encoder: true
```

### Fusion Types

| Type | Description | Complexity |
|------|-------------|------------|
| `concat` | Simple concatenation | Low |
| `element_wise` | Element-wise operations | Medium |
| `cross_attention` | Cross-modal attention | High |

```yaml
model:
  fusion_type: cross_attention
  num_fusion_layers: 2
  num_heads: 8
```

### Mixture of Experts (MOE)

Enable MOE for specialized expert routing:

```yaml
model:
  use_moe: true
  num_experts: 4        # Number of expert networks
  top_k_experts: 2      # Experts activated per sample
```

### Knowledge Base

Enable knowledge-augmented reasoning:

```yaml
model:
  use_knowledge_base: true
  knowledge_dim: 768
```

---

## Training Configuration

### Basic Training Settings

```yaml
training:
  num_epochs: 20
  learning_rate: 2.0e-5
  weight_decay: 0.01
```

### Optimizer Configuration

**AdamW (recommended):**
```yaml
training:
  optimizer_type: adamw
  learning_rate: 2.0e-5
  weight_decay: 0.01
```

**SGD:**
```yaml
training:
  optimizer_type: sgd
  learning_rate: 0.01
  weight_decay: 0.0001
```

### Learning Rate Schedulers

**Cosine Annealing (recommended):**
```yaml
training:
  scheduler_type: cosine
  warmup_ratio: 0.1   # 10% of training for warmup
```

**Linear Decay:**
```yaml
training:
  scheduler_type: linear
  warmup_ratio: 0.1
```

**Reduce on Plateau:**
```yaml
training:
  scheduler_type: plateau
```

### Mixed Precision Training

Enable for faster training on modern GPUs:

```yaml
training:
  use_amp: true   # Automatic Mixed Precision
```

**Requirements:**
- NVIDIA GPU with Tensor Cores (Volta+)
- PyTorch 1.6+

### Gradient Clipping

Prevent exploding gradients:

```yaml
training:
  max_grad_norm: 1.0
```

### Early Stopping

Stop training when validation doesn't improve:

```yaml
training:
  patience: 5   # Epochs without improvement before stopping
```

### Checkpointing

```yaml
training:
  save_best_only: true       # Save only best model
  checkpoint_dir: checkpoints
```

### Resume Training

Resume from a checkpoint:

```yaml
training:
  resume_from: checkpoints/checkpoint_epoch_5.pt
```

---

## Resource Configuration

File: `configs/resource_configs.yaml`

```yaml
# =============================================================================
# RESOURCE MANAGEMENT CONFIGURATION
# =============================================================================

memory:
  # Maximum GPU memory to use (percentage)
  max_gpu_memory_percent: 90
  
  # Clear cache frequency
  clear_cache_frequency: 100  # batches

monitoring:
  # Enable resource monitoring
  enabled: true
  
  # Monitoring frequency (seconds)
  interval: 30
  
  # Alert thresholds
  gpu_memory_threshold: 0.9
  cpu_memory_threshold: 0.9

backup:
  # Enable automatic backups
  enabled: true
  
  # Backup frequency
  frequency: epoch  # batch, epoch, time
  
  # Keep last N backups
  keep_last: 3
```

---

## Logging Configuration

File: `configs/logging_configs.yaml`

```yaml
# =============================================================================
# LOGGING CONFIGURATION
# =============================================================================

logging:
  # Log level: DEBUG, INFO, WARNING, ERROR
  level: INFO
  
  # Log format
  format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
  
  # Console logging
  console:
    enabled: true
    colorized: true
  
  # File logging
  file:
    enabled: true
    directory: logs/pipeline
    max_size: 10MB
    backup_count: 5
  
  # TensorBoard
  tensorboard:
    enabled: true
    log_dir: logs/tensorboard
    
  # Metrics to log
  metrics:
    - loss
    - accuracy
    - learning_rate
    - gpu_memory
```

---

## Examples

### Example 1: Quick Debug Run

```yaml
# configs/debug_config.yaml
mode: train
debug: true

data:
  batch_size: 4
  max_samples: 100

model:
  visual_backbone: vit
  freeze_visual_encoder: true

training:
  num_epochs: 1
  use_amp: true
```

### Example 2: Full Training

```yaml
# configs/full_training.yaml
mode: train

data:
  batch_size: 32
  augmentation_strength: heavy

model:
  visual_backbone: clip
  text_encoder_type: phobert
  fusion_type: cross_attention
  use_moe: true

training:
  num_epochs: 50
  learning_rate: 1.0e-5
  patience: 10
  use_amp: true
```

### Example 3: Memory-Constrained GPU (8GB)

```yaml
# configs/low_memory.yaml
mode: train

data:
  batch_size: 8
  num_workers: 2

model:
  visual_backbone: vit
  freeze_visual_encoder: true
  hidden_dim: 512

training:
  num_epochs: 20
  use_amp: true
```

### Example 4: CPU-Only Training

```yaml
# configs/cpu_training.yaml
mode: train

data:
  batch_size: 4
  num_workers: 4
  pin_memory: false

model:
  visual_backbone: resnet50

training:
  num_epochs: 10
  use_amp: false   # No AMP on CPU
```

---

## Environment Variables

Override configuration via environment variables:

```bash
# Override batch size
export VQA_BATCH_SIZE=16

# Override learning rate
export VQA_LEARNING_RATE=1e-5

# Override output directory
export VQA_OUTPUT_DIR=/path/to/output
```

---

## Validation

The pipeline validates configuration on startup:

```
[CONFIG] Validating configuration...
[CONFIG] ✓ data.train_ratio + val_ratio + test_ratio = 1.0
[CONFIG] ✓ model.visual_backbone 'vit' is supported
[CONFIG] ✓ training.num_epochs > 0
[CONFIG] Configuration valid!
```

Common validation errors:
- Split ratios must sum to 1.0
- Unknown visual backbone type
- Resume checkpoint not found
- Invalid YAML syntax

---

## See Also

- [Getting Started](getting_started.md)
- [Pipeline Usage](pipeline_usage.md)
- [API Reference](api_reference.md)
