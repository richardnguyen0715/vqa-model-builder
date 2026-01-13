# API Reference

Complete API reference for AutoViVQA Model Builder.

---

## Table of Contents

- [Pipeline Classes](#pipeline-classes)
  - [VQAPipeline](#vqapipeline)
  - [DataPipeline](#datapipeline)
  - [ModelPipeline](#modelpipeline)
  - [TrainingPipeline](#trainingpipeline)
- [Configuration Classes](#configuration-classes)
- [Model Classes](#model-classes)
- [Data Classes](#data-classes)
- [Utility Functions](#utility-functions)

---

## Pipeline Classes

### VQAPipeline

The main orchestrator that coordinates all pipeline stages.

```python
from src.core import VQAPipeline, VQAPipelineConfig
```

#### Constructor

```python
VQAPipeline(config: VQAPipelineConfig)
```

**Parameters:**
| Parameter | Type | Description |
|-----------|------|-------------|
| `config` | `VQAPipelineConfig` | Pipeline configuration |

#### Methods

##### `run()`

Executes the complete VQA pipeline.

```python
results = pipeline.run()
```

**Returns:** `dict` - Training/evaluation results including:
- `best_metric`: Best validation accuracy
- `final_train_loss`: Final training loss
- `model_path`: Path to saved model
- `status`: "SUCCESS" or "FAILED"

**Example:**
```python
config = VQAPipelineConfig(
    mode="train",
    data=DataPipelineConfig(batch_size=32),
    model=ModelPipelineConfig(visual_backbone="vit"),
    training=TrainingPipelineConfig(num_epochs=10)
)
pipeline = VQAPipeline(config)
results = pipeline.run()
print(f"Best accuracy: {results['best_metric']:.4f}")
```

---

### DataPipeline

Handles data loading, validation, splitting, and preprocessing.

```python
from src.core.data_pipeline import DataPipeline, DataPipelineConfig
```

#### Constructor

```python
DataPipeline(config: DataPipelineConfig, output_dir: str = "outputs")
```

#### Methods

##### `run()`

Executes the complete data pipeline.

```python
data_outputs = data_pipeline.run()
```

**Returns:** `dict` containing:
- `train_loader`: PyTorch DataLoader for training
- `val_loader`: PyTorch DataLoader for validation
- `test_loader`: PyTorch DataLoader for testing
- `vocabulary`: Answer vocabulary
- `tokenizer`: Text tokenizer
- `data_stats`: Dataset statistics

**Pipeline Steps:**
1. Load raw data
2. Validate data integrity
3. Split into train/val/test
4. Build vocabulary
5. Initialize tokenizer
6. Create transforms
7. Build datasets
8. Create dataloaders
9. Run sanity checks

**Example:**
```python
config = DataPipelineConfig(
    batch_size=32,
    train_ratio=0.8,
    augmentation_strength="medium"
)
data_pipeline = DataPipeline(config)
outputs = data_pipeline.run()

train_loader = outputs['train_loader']
for batch in train_loader:
    images, questions, answers = batch['images'], batch['questions'], batch['answers']
```

---

### ModelPipeline

Builds and configures the VQA model.

```python
from src.core.model_pipeline import ModelPipeline, ModelPipelineConfig
```

#### Constructor

```python
ModelPipeline(
    config: ModelPipelineConfig,
    num_answers: int,
    output_dir: str = "outputs"
)
```

**Parameters:**
| Parameter | Type | Description |
|-----------|------|-------------|
| `config` | `ModelPipelineConfig` | Model configuration |
| `num_answers` | `int` | Number of answer classes |
| `output_dir` | `str` | Output directory |

#### Methods

##### `run()`

Builds and returns the configured model.

```python
model_outputs = model_pipeline.run()
```

**Returns:** `dict` containing:
- `model`: The VQA model (on device)
- `model_config`: Model configuration
- `model_info`: Model information (parameters, etc.)
- `device`: Training device (cuda/cpu)

**Example:**
```python
config = ModelPipelineConfig(
    visual_backbone="vit",
    text_encoder_type="phobert",
    fusion_type="cross_attention",
    hidden_dim=768
)
model_pipeline = ModelPipeline(config, num_answers=1000)
outputs = model_pipeline.run()

model = outputs['model']
print(f"Parameters: {outputs['model_info']['total_params']:,}")
```

---

### TrainingPipeline

Handles the training loop with mixed precision, validation, and checkpointing.

```python
from src.core.training_pipeline import TrainingPipeline, TrainingPipelineConfig
```

#### Constructor

```python
TrainingPipeline(
    config: TrainingPipelineConfig,
    model: torch.nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    model_config: dict,
    output_dir: str = "outputs"
)
```

#### Methods

##### `run()`

Executes the complete training loop.

```python
training_results = training_pipeline.run()
```

**Returns:** `dict` containing:
- `best_metric`: Best validation accuracy
- `final_train_loss`: Final training loss
- `final_train_accuracy`: Final training accuracy
- `final_val_loss`: Final validation loss
- `final_val_accuracy`: Final validation accuracy
- `best_epoch`: Epoch with best performance
- `total_epochs`: Total epochs trained
- `checkpoint_path`: Path to best checkpoint

**Example:**
```python
config = TrainingPipelineConfig(
    num_epochs=20,
    learning_rate=2e-5,
    use_amp=True,
    patience=5
)
training_pipeline = TrainingPipeline(
    config=config,
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    model_config=model_config
)
results = training_pipeline.run()
print(f"Best accuracy: {results['best_metric']:.4f} at epoch {results['best_epoch']}")
```

---

## Configuration Classes

### VQAPipelineConfig

Main configuration for the entire pipeline.

```python
@dataclass
class VQAPipelineConfig:
    mode: str = "train"                     # "train", "evaluate", "inference"
    output_dir: str = "outputs"             # Output directory
    data: DataPipelineConfig = None         # Data configuration
    model: ModelPipelineConfig = None       # Model configuration
    training: TrainingPipelineConfig = None # Training configuration
```

### DataPipelineConfig

Configuration for data loading and preprocessing.

```python
@dataclass
class DataPipelineConfig:
    # Data paths
    data_dir: str = "data/raw"
    images_dir: str = "data/raw/images"
    
    # Data splitting
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    test_ratio: float = 0.1
    
    # DataLoader settings
    batch_size: int = 32
    num_workers: int = 4
    pin_memory: bool = True
    
    # Preprocessing
    image_size: int = 224
    max_length: int = 128
    
    # Augmentation
    augmentation_strength: str = "medium"  # "none", "light", "medium", "heavy"
    
    # Limits
    max_samples: int = None  # None for all samples
    min_answer_freq: int = 5
```

### ModelPipelineConfig

Configuration for model architecture.

```python
@dataclass
class ModelPipelineConfig:
    # Visual backbone
    visual_backbone: str = "vit"           # "vit", "clip", "resnet50"
    visual_pretrained: str = "google/vit-base-patch16-224"
    freeze_visual_encoder: bool = True
    
    # Text encoder
    text_encoder_type: str = "phobert"     # "phobert", "bert"
    text_pretrained: str = "vinai/phobert-base"
    freeze_text_encoder: bool = True
    
    # Fusion
    fusion_type: str = "cross_attention"   # "concat", "element_wise", "cross_attention"
    
    # Model dimensions
    hidden_dim: int = 768
    num_heads: int = 8
    num_fusion_layers: int = 2
    dropout: float = 0.1
    
    # Advanced features
    use_moe: bool = False
    num_experts: int = 4
    use_knowledge_base: bool = False
```

### TrainingPipelineConfig

Configuration for training process.

```python
@dataclass
class TrainingPipelineConfig:
    # Training settings
    num_epochs: int = 10
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    
    # Optimizer
    optimizer_type: str = "adamw"          # "adam", "adamw", "sgd"
    
    # Scheduler
    scheduler_type: str = "cosine"         # "linear", "cosine", "plateau"
    warmup_ratio: float = 0.1
    
    # Mixed precision
    use_amp: bool = True
    
    # Checkpointing
    save_best_only: bool = True
    checkpoint_dir: str = "checkpoints"
    
    # Early stopping
    patience: int = 5
    
    # Gradient clipping
    max_grad_norm: float = 1.0
    
    # Resuming
    resume_from: str = None
```

---

## Model Classes

### VietnameseVQAModel

The main VQA model for Vietnamese.

```python
from src.modeling.meta_arch import VietnameseVQAModel, VQAModelConfig
```

#### Constructor

```python
VietnameseVQAModel(config: VQAModelConfig)
```

#### Forward Method

```python
def forward(
    self,
    images: torch.Tensor,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor = None,
    labels: torch.Tensor = None
) -> VQAOutput
```

**Parameters:**
| Parameter | Type | Shape | Description |
|-----------|------|-------|-------------|
| `images` | `Tensor` | `(B, C, H, W)` | Input images |
| `input_ids` | `Tensor` | `(B, L)` | Tokenized question IDs |
| `attention_mask` | `Tensor` | `(B, L)` | Attention mask |
| `labels` | `Tensor` | `(B,)` | Answer labels (optional) |

**Returns:** `VQAOutput` containing:
- `logits`: Output logits `(B, num_answers)`
- `loss`: Cross-entropy loss (if labels provided)
- `hidden_states`: Intermediate representations

**Example:**
```python
model = VietnameseVQAModel(config)
output = model(
    images=batch['images'],
    input_ids=batch['input_ids'],
    attention_mask=batch['attention_mask'],
    labels=batch['answers']
)
loss = output.loss
predictions = output.logits.argmax(dim=-1)
```

---

## Data Classes

### VQADataset

PyTorch Dataset for VQA data.

```python
from src.data.dataset import VQADataset
```

#### Constructor

```python
VQADataset(
    data: List[VQADataItem],
    tokenizer,
    transform = None,
    max_length: int = 128
)
```

#### Methods

##### `__getitem__(idx)`

Returns a single sample.

```python
sample = dataset[0]
# Returns dict with:
# - 'images': Tensor (C, H, W)
# - 'input_ids': Tensor (L,)
# - 'attention_mask': Tensor (L,)
# - 'answers': int
# - 'question': str
# - 'image_path': str
```

### VQADataItem

Data class for a single VQA sample.

```python
from src.schema.data_schema import VQADataItem

@dataclass
class VQADataItem:
    image_path: str      # Path to image file
    question: str        # Vietnamese question
    answer: str          # Answer text
    answer_idx: int      # Answer class index
    sample_id: str       # Unique sample ID
```

---

## Utility Functions

### Config Loading

```python
from utils.config_loader import load_config

config = load_config("configs/pipeline_config.yaml")
```

### Logger Setup

```python
from src.core.pipeline_logger import PipelineLogger

logger = PipelineLogger.get_logger("my_module")
logger.info("Hello world!")
```

### Resource Monitoring

```python
from src.resource_management import ResourceMonitor

monitor = ResourceMonitor()
stats = monitor.get_system_info()
print(f"GPU Memory: {stats['gpu_memory_used']}/{stats['gpu_memory_total']}")
```

---

## CLI Reference

### Command-Line Arguments

```bash
python -m src.core.vqa_pipeline [OPTIONS]
```

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--mode` | str | "train" | Pipeline mode (train/evaluate/inference) |
| `--config` | str | None | Path to YAML config file |
| `--output-dir` | str | "outputs" | Output directory |
| `--batch-size` | int | 32 | Batch size |
| `--epochs` | int | 10 | Number of epochs |
| `--learning-rate` | float | 2e-5 | Learning rate |
| `--visual-backbone` | str | "vit" | Visual backbone type |
| `--text-encoder` | str | "phobert" | Text encoder type |
| `--use-moe` | flag | False | Enable Mixture of Experts |
| `--use-knowledge` | flag | False | Enable Knowledge Base |
| `--resume` | str | None | Resume from checkpoint |
| `--debug` | flag | False | Enable debug mode |
| `--seed` | int | 42 | Random seed |

### Examples

```bash
# Basic training
python -m src.core.vqa_pipeline --mode train --epochs 20

# Training with specific config
python -m src.core.vqa_pipeline --config configs/pipeline_config.yaml

# Evaluate a model
python -m src.core.vqa_pipeline --mode evaluate --resume checkpoints/best_model.pt

# Debug mode with small batch
python -m src.core.vqa_pipeline --debug --batch-size 4 --epochs 1
```

---

## Type Hints

### Common Types

```python
from typing import Dict, List, Optional, Tuple, Union
import torch
from torch import Tensor
from torch.utils.data import DataLoader

# Common function signatures
def train_step(
    model: torch.nn.Module,
    batch: Dict[str, Tensor],
    optimizer: torch.optim.Optimizer,
    criterion: torch.nn.Module
) -> Tuple[float, float]:
    """Returns (loss, accuracy)"""
    pass

def collate_fn(
    batch: List[Dict[str, Union[Tensor, str, int]]]
) -> Dict[str, Tensor]:
    """Collates batch samples into tensors"""
    pass
```

---

## Error Handling

### Custom Exceptions

```python
from src.exception import DataValidationError, ModelConfigError

# Data validation
try:
    data_pipeline.run()
except DataValidationError as e:
    print(f"Data error: {e}")

# Model configuration
try:
    model_pipeline.run()
except ModelConfigError as e:
    print(f"Model error: {e}")
```

---

## See Also

- [Getting Started](getting_started.md)
- [Pipeline Usage](pipeline_usage.md)
- [VQA Architecture](vqa_architecture.md)
- [Configuration Guide](../configs/README.md)
