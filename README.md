# 🇻🇳 AutoViVQA Model Builder

<div align="center">

```
╔══════════════════════════════════════════════════════════════════════════════════════════╗
║                                                                                          ║
║  ██╗   ██╗ ██████╗  █████╗     ██████╗ ██╗██████╗ ███████╗██╗     ██╗███╗   ██╗███████╗  ║
║  ██║   ██║██╔═══██╗██╔══██╗    ██╔══██╗██║██╔══██╗██╔════╝██║     ██║████╗  ██║██╔════╝  ║
║  ██║   ██║██║   ██║███████║    ██████╔╝██║██████╔╝█████╗  ██║     ██║██╔██╗ ██║█████╗    ║
║  ╚██╗ ██╔╝██║▄▄ ██║██╔══██║    ██╔═══╝ ██║██╔═══╝ ██╔══╝  ██║     ██║██║╚██╗██║██╔══╝    ║
║   ╚████╔╝ ╚██████╔╝██║  ██║    ██║     ██║██║     ███████╗███████╗██║██║ ╚████║███████╗  ║
║    ╚═══╝   ╚══▀▀═╝ ╚═╝  ╚═╝    ╚═╝     ╚═╝╚═╝     ╚══════╝╚══════╝╚═╝╚═╝  ╚═══╝╚══════╝  ║
║                                                                                          ║
║                            Vietnamese Visual Question Answering                          ║
║                                  AutoViVQA Model Builder                                 ║
║                                                                                          ║
╚══════════════════════════════════════════════════════════════════════════════════════════╝
```

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.9+-ee4c2c.svg)](https://pytorch.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

**A comprehensive Vietnamese Visual Question Answering system with state-of-the-art features**

[Features](#-features) • [Installation](#-installation) • [Quick Start](#-quick-start) • [Documentation](#-documentation) • [Architecture](#-architecture)

</div>

---

## 📋 Table of Contents

- [Features](#-features)
- [Architecture Overview](#-architecture-overview)
- [Installation](#-installation)
- [Quick Start](#-quick-start)
- [Pipeline Usage](#-pipeline-usage)
  - [Training](#training)
  - [Evaluation](#evaluation)
  - [Inference](#inference)
- [Configuration](#-configuration)
- [Project Structure](#-project-structure)
- [Documentation](#-documentation)
- [Development](#-development)
- [License](#-license)

---

## ✨ Features

### 🎯 Core Capabilities

| Feature | Description |
|---------|-------------|
| **Vietnamese NLP** | Optimized for Vietnamese with PhoBERT integration |
| **Multiple Visual Backbones** | ViT, ResNet, CLIP, Swin Transformer |
| **Multimodal Fusion** | Cross-Attention, Q-Former, Bilinear fusion strategies |
| **Mixture of Experts (MOE)** | Dynamic routing to specialized expert networks |
| **Knowledge Base/RAG** | Retrieval-Augmented Generation for external knowledge |
| **Memory-Efficient** | Lazy image loading, AMP training |

### 🔧 Technical Features

- ✅ **End-to-end Pipeline** - Data loading → Model building → Training → Evaluation
- ✅ **Comprehensive Logging** - Detailed validation at every step
- ✅ **Resource Monitoring** - Real-time CPU, GPU, Memory tracking
- ✅ **Checkpointing** - Auto-save best models with early stopping
- ✅ **Mixed Precision** - FP16/BF16 training for faster performance
- ✅ **CLI & Python API** - Flexible usage options

---

## 🏗 Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                    Vietnamese VQA Model                          │
├─────────────────────────────────────────────────────────────────┤
│  ┌──────────────┐     ┌──────────────┐     ┌──────────────┐    │
│  │   Image      │     │   Question   │     │  Knowledge   │    │
│  │   Input      │     │   (Vietnamese)│     │    Base      │    │
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
│                      ▼                                          │
│              ┌──────────────────────────────────┐               │
│              │     Multimodal Fusion            │               │
│              │  (Cross-Attention / Q-Former)    │               │
│              └──────────────┬───────────────────┘               │
│                             ▼                                    │
│              ┌──────────────────────────────────┐               │
│              │    Mixture of Experts (MOE)      │               │
│              │  ┌────┐ ┌────┐ ┌────┐ ┌────┐   │               │
│              │  │ E1 │ │ E2 │ │ E3 │ │ E4 │   │               │
│              │  └────┘ └────┘ └────┘ └────┘   │               │
│              └──────────────┬───────────────────┘               │
│                             ▼                                    │
│              ┌──────────────────────────────────┐               │
│              │        Answer Head               │               │
│              └──────────────┬───────────────────┘               │
│                             ▼                                    │
│                      ┌──────────┐                               │
│                      │  Answer  │                               │
│                      └──────────┘                               │
└─────────────────────────────────────────────────────────────────┘
```

---

## 💻 Installation

### Prerequisites

- Python 3.11+
- CUDA 12.x (for GPU support)
- 8GB+ RAM (16GB+ recommended)
- NVIDIA GPU with 8GB+ VRAM (optional but recommended)

### Step 1: Clone Repository

```bash
git clone https://github.com/yourusername/AutovivqaModelBuilder.git
cd AutovivqaModelBuilder
```

### Step 2: Create Environment

```bash
# Using conda (recommended)
conda create -n vqa python=3.11
conda activate vqa

# Or using venv
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows
```

### Step 3: Install Dependencies

```bash
# Using poetry (recommended)
pip install poetry
poetry install

# Or using pip
pip install -r requirements.txt
```

### Step 4: Download Data

```bash
# Using CLI script
bash src/cli/download_data.sh

# Or using Python
python -m src.data.download_data
```

---

## 🚀 Quick Start

### Option 1: One-Command Training

```bash
# Download data + Train model (recommended for first time)
bash src/cli/quick_start.sh
```

### Option 2: CLI Training

```bash
# Basic training
python -m src.core.vqa_pipeline --mode train --epochs 10 --batch-size 16

# With all options
python -m src.core.vqa_pipeline \
    --mode train \
    --epochs 20 \
    --batch-size 32 \
    --learning-rate 2e-5 \
    --visual-backbone vit \
    --text-encoder phobert \
    --use-moe \
    --output-dir outputs
```

### Option 3: Using Config File

```bash
# Use YAML configuration
python -m src.core.vqa_pipeline --config configs/pipeline_config.yaml
```

### Option 4: Shell Script

```bash
# Using shell script with arguments
bash src/cli/run_pipeline.sh --mode train --epochs 10 --batch-size 16

# Or use clean mode (suppresses warnings)
bash src/cli/run_clean.sh --mode train --epochs 10
```

### Option 5: Python API

```python
from src.core import VQAPipeline, VQAPipelineConfig
from src.core import DataPipelineConfig, ModelPipelineConfig, TrainingPipelineConfig

# Configure pipeline
config = VQAPipelineConfig(
    mode="train",
    data=DataPipelineConfig(
        images_dir="data/raw/images",
        batch_size=16,
    ),
    model=ModelPipelineConfig(
        visual_backbone="vit",
        text_encoder_type="phobert",
        use_moe=False,
    ),
    training=TrainingPipelineConfig(
        num_epochs=10,
        learning_rate=2e-5,
    ),
)

# Run pipeline
pipeline = VQAPipeline(config)
results = pipeline.run()

print(f"Best accuracy: {results.training_output.best_metric:.4f}")
```

---

## 📖 Pipeline Usage

### Training

#### Basic Training

```bash
python -m src.core.vqa_pipeline --mode train --epochs 10
```

#### Advanced Training with MOE

```bash
python -m src.core.vqa_pipeline \
    --mode train \
    --epochs 20 \
    --batch-size 32 \
    --use-moe \
    --learning-rate 2e-5 \
    --output-dir outputs/moe_experiment
```

#### Training with Knowledge Base (RAG)

```bash
python -m src.core.vqa_pipeline \
    --mode train \
    --epochs 20 \
    --use-knowledge \
    --output-dir outputs/rag_experiment
```

#### Resume Training from Checkpoint

```bash
python -m src.core.vqa_pipeline \
    --mode train \
    --resume checkpoints/checkpoint_epoch_5.pt
```

### Evaluation

```bash
# Evaluate on test set
python -m src.core.vqa_pipeline --mode evaluate --resume checkpoints/best_model.pt
```

### Inference

```bash
# Run inference on new images
python -m src.core.vqa_pipeline --mode inference --resume checkpoints/best_model.pt
```

---

## ⚙️ Configuration

### Configuration Files

| File | Description |
|------|-------------|
| `configs/pipeline_config.yaml` | Complete pipeline configuration |
| `configs/model_configs.yaml` | Model architecture settings |
| `configs/training_configs.yaml` | Training hyperparameters |
| `configs/data_configs.yaml` | Data loading settings |
| `configs/resource_configs.yaml` | Resource monitoring thresholds |

### Key Configuration Options

```yaml
# configs/pipeline_config.yaml

# Data Configuration
data:
  images_dir: data/raw/images
  batch_size: 32
  num_workers: 4
  augmentation_strength: medium  # light, medium, strong

# Model Configuration
model:
  visual_backbone: vit           # vit, resnet, clip, swin
  text_encoder_type: phobert     # phobert, bert
  fusion_type: cross_attention   # cross_attention, concat, bilinear
  use_moe: false                 # Enable Mixture of Experts
  use_knowledge: false           # Enable Knowledge Base/RAG

# Training Configuration
training:
  num_epochs: 20
  learning_rate: 2.0e-5
  optimizer_name: adamw
  scheduler_name: cosine
  use_amp: true                  # Mixed precision training
  early_stopping: true
  patience: 5
```

### CLI Arguments Reference

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--mode` | str | train | Pipeline mode: train, evaluate, inference |
| `--config` | str | None | Path to YAML config file |
| `--epochs` | int | 20 | Number of training epochs |
| `--batch-size` | int | 32 | Training batch size |
| `--learning-rate` | float | 2e-5 | Learning rate |
| `--visual-backbone` | str | vit | Visual encoder: vit, resnet, clip, swin |
| `--text-encoder` | str | phobert | Text encoder: phobert, bert |
| `--use-moe` | flag | False | Enable Mixture of Experts |
| `--use-knowledge` | flag | False | Enable Knowledge Base/RAG |
| `--output-dir` | str | outputs | Output directory |
| `--resume` | str | None | Resume from checkpoint |

---

## 📁 Project Structure

```
AutovivqaModelBuilder/
├── 📁 configs/                    # Configuration files
│   ├── pipeline_config.yaml       # Main pipeline config
│   ├── model_configs.yaml         # Model architecture
│   ├── training_configs.yaml      # Training settings
│   ├── data_configs.yaml          # Data loading
│   └── resource_configs.yaml      # Resource monitoring
│
├── 📁 src/                        # Source code
│   ├── 📁 core/                   # Core pipeline modules
│   │   ├── vqa_pipeline.py        # Main orchestrator
│   │   ├── data_pipeline.py       # Data loading pipeline
│   │   ├── model_pipeline.py      # Model building pipeline
│   │   ├── training_pipeline.py   # Training loop pipeline
│   │   └── pipeline_logger.py     # Comprehensive logging
│   │
│   ├── 📁 cli/                    # Command-line interface
│   │   ├── run_pipeline.sh        # Main CLI script
│   │   ├── quick_start.sh         # Quick start script
│   │   ├── run_clean.sh           # Clean output script
│   │   └── download_data.sh       # Data download script
│   │
│   ├── 📁 data/                   # Data handling
│   │   ├── data_actions.py        # Data loading functions
│   │   ├── dataset.py             # PyTorch Dataset classes
│   │   ├── augmentation.py        # Image augmentation
│   │   └── download_data.py       # Kaggle data download
│   │
│   ├── 📁 modeling/               # Model architecture
│   │   ├── 📁 meta_arch/          # Main VQA model
│   │   │   ├── vqa_model.py       # VietnameseVQAModel
│   │   │   └── vqa_config.py      # Model configurations
│   │   ├── 📁 backbone/           # Visual encoders
│   │   ├── 📁 fusion/             # Multimodal fusion
│   │   ├── 📁 moe/                # Mixture of Experts
│   │   ├── 📁 knowledge_base/     # RAG module
│   │   └── 📁 tokenizer/          # Text tokenizers
│   │
│   ├── 📁 pipeline/               # Training utilities
│   │   ├── 📁 trainer/            # Training loops
│   │   └── 📁 evaluator/          # Evaluation metrics
│   │
│   ├── 📁 resource_management/    # Resource monitoring
│   │   ├── resource_monitor.py    # CPU/GPU/Memory monitor
│   │   └── resource_manager.py    # Auto-backup on thresholds
│   │
│   └── 📁 middleware/             # Utilities
│       ├── config_loader.py       # Config loading
│       ├── logger.py              # Logging setup
│       └── monitor.py             # Memory monitoring
│
├── 📁 data/                       # Data directory
│   └── 📁 raw/                    # Raw data
│       ├── 📁 images/             # Image files
│       └── 📁 texts/              # CSV/JSON annotations
│
├── 📁 docs/                       # Documentation
│   ├── vqa_architecture.md        # Architecture details
│   ├── fusion_approaches.md       # Fusion strategies
│   ├── moe_approaches.md          # MOE documentation
│   └── prepare_data.md            # Data preparation guide
│
├── 📁 checkpoints/                # Model checkpoints
├── 📁 outputs/                    # Training outputs
├── 📁 logs/                       # Log files
│
├── pyproject.toml                 # Project dependencies
├── README.md                      # This file
└── LICENSE                        # MIT License
```

---

## 📚 Documentation

### Core Documentation

| Document | Description |
|----------|-------------|
| [VQA Architecture](docs/vqa_architecture.md) | Complete system architecture |
| [Fusion Approaches](docs/fusion_approaches.md) | Multimodal fusion strategies |
| [MOE Approaches](docs/moe_approaches.md) | Mixture of Experts module |
| [Knowledge Base](docs/knowledge_base_approaches.md) | RAG integration |
| [Data Preparation](docs/prepare_data.md) | Data setup guide |

### Pipeline Output

After training, the pipeline generates:

```
outputs/
├── pipeline_summary.json       # Complete training summary
├── training_curves.png         # Loss/accuracy plots
└── predictions/                # Model predictions

checkpoints/
├── best_model.pt               # Best model checkpoint
├── checkpoint_epoch_1.pt       # Epoch checkpoints
└── checkpoint_epoch_N.pt

logs/
├── tensorboard/                # TensorBoard logs
└── pipeline/                   # Pipeline logs
```

### Logging Output Example

```
╔══════════════════════════════════════════════════════════════════════════════════════════╗
║                                   VQA PIPELINE                                           ║
╚══════════════════════════════════════════════════════════════════════════════════════════╝

================================================================================
============================= VQA PIPELINE STARTED =============================
================================================================================
    Mode: train
    Output directory: outputs
    Start time: 2026-01-10 00:01:28

-------------------- System Information --------------------
    Platform: Linux-6.14.0-37-generic-x86_64-with-glibc2.39
    Python version: 3.11.14
    PyTorch version: 2.9.1+cu128
    CUDA available: True
    GPU 0: NVIDIA GeForce RTX 3060 (11.6 GB)
    Total RAM: 31.1 GB

================================================================================
============================ STAGE 1: DATA PIPELINE ============================
================================================================================
✓ Loaded 37077 data samples
✓ Data split: 29661 train / 3707 val / 3709 test
✓ Built vocabulary with 447 answer classes
✓ DATA PIPELINE completed in 7.22s

================================================================================
=========================== STAGE 2: MODEL PIPELINE ============================
================================================================================
✓ Using CUDA: NVIDIA GeForce RTX 3060
    Total parameters: 243,163,583
    Trainable parameters: 243,163,583
    Model size (MB): 927.60
✓ MODEL PIPELINE completed in 6.97s

================================================================================
========================== STAGE 3: TRAINING PIPELINE ==========================
================================================================================
Epoch 1 [Train]: 100%|██████████| 1853/1853 [05:10<00:00, 5.97it/s, loss=1.88, acc=0.76]
Epoch 1 [Val]: 100%|██████████| 58/58 [00:13<00:00, 4.23it/s, loss=0.35, acc=0.96]
✓ New best accuracy: 0.9625
✓ Checkpoint saved [BEST]: checkpoints/best_model.pt

================================================================================
=============================== PIPELINE SUMMARY ===============================
================================================================================
    Status: SUCCESS
    Total execution time: 357.44s (6.0 min)
    Best metric: 0.9625
    Best model path: checkpoints/best_model.pt
```

---

## � Documentation

Comprehensive documentation is available in the `docs/` directory:

### Getting Started
| Document | Description |
|----------|-------------|
| [Getting Started Guide](docs/getting_started.md) | Installation and first run |
| [Pipeline Usage](docs/pipeline_usage.md) | Complete pipeline documentation |
| [Configuration Guide](docs/configuration_guide.md) | All configuration options |
| [API Reference](docs/api_reference.md) | Python API documentation |

### Architecture & Design
| Document | Description |
|----------|-------------|
| [VQA Architecture](docs/vqa_architecture.md) | System architecture overview |
| [Fusion Approaches](docs/fusion_approaches.md) | Multimodal fusion strategies |
| [MOE Approaches](docs/moe_approaches.md) | Mixture of Experts design |
| [Knowledge Base](docs/knowledge_base_approaches.md) | RAG implementation |

### Component Documentation
| Document | Description |
|----------|-------------|
| [Image Representation](docs/image_representation_approaches.md) | Visual encoders |
| [Text Representation](docs/text_representation_approaches.md) | Text encoders |
| [Data Preparation](docs/prepare_data.md) | Dataset preparation |

---

## �🔬 Development

### Running Tests

```bash
# Run all tests
pytest tests/

# Run specific test
pytest tests/test_pipeline.py -v
```

### Code Quality

```bash
# Format code
black src/

# Lint code
flake8 src/

# Type checking
mypy src/
```

### TensorBoard

```bash
# Launch TensorBoard
tensorboard --logdir logs/tensorboard

# Open in browser: http://localhost:6006
```

---

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 👤 Author

**Tuong Nguyen**
- Email: richardnguyen0715@gmail.com
- GitHub: [@tuong.nguyen](https://github.com/tuong.nguyen)

---

## 🙏 Acknowledgments

- [VinAI Research](https://vinai.io/) for PhoBERT
- [OpenAI](https://openai.com/) for CLIP
- [Hugging Face](https://huggingface.co/) for Transformers
- [PyTorch](https://pytorch.org/) team

---

<div align="center">

**⭐ Star this repo if you find it helpful!**

Made with ❤️ for Vietnamese NLP

</div>
