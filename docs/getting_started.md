# Getting Started Guide

This guide will help you get up and running with AutoViVQA Model Builder in just a few minutes.

## Prerequisites

Before you begin, ensure you have:

- **Python 3.11+** installed
- **NVIDIA GPU** with 8GB+ VRAM (recommended) or CPU
- **CUDA 12.x** (for GPU support)
- **16GB+ RAM** recommended

---

## Installation

### Step 1: Clone the Repository

```bash
git clone https://github.com/yourusername/AutovivqaModelBuilder.git
cd AutovivqaModelBuilder
```

### Step 2: Create Python Environment

Using **conda** (recommended):
```bash
conda create -n vqa python=3.11
conda activate vqa
```

Or using **venv**:
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows
```

### Step 3: Install Dependencies

Using **Poetry** (recommended):
```bash
pip install poetry
poetry install
```

Or using **pip**:
```bash
pip install torch torchvision transformers
pip install -r requirements.txt
```

### Step 4: Verify Installation

```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

Expected output:
```
PyTorch: 2.9.1+cu128
CUDA available: True
```

---

## Quick Start

### Option 1: One-Line Quick Start

Download data and start training with one command:

```bash
bash src/cli/quick_start.sh
```

This will:
1. Download the Vietnamese VQA dataset from Kaggle
2. Start training with default settings
3. Save the best model to `checkpoints/best_model.pt`

### Option 2: Step-by-Step

#### 1. Download Data

```bash
# Download from Kaggle
bash src/cli/download_data.sh
```

Or using Python:
```bash
python -m src.data.download_data
```

#### 2. Verify Data

Check that data is downloaded correctly:
```bash
ls -la data/raw/images/ | head -10
ls -la data/raw/texts/
```

Expected:
```
data/raw/
├── images/
│   ├── 000000000001.jpg
│   ├── 000000000002.jpg
│   └── ... (20,000+ images)
└── texts/
    └── evaluate_60k_data_balanced_preprocessed.csv
```

#### 3. Start Training

```bash
# Basic training (1 epoch for testing)
python -m src.core.vqa_pipeline --mode train --epochs 1 --batch-size 16

# Full training
python -m src.core.vqa_pipeline --mode train --epochs 20 --batch-size 32
```

#### 4. Monitor Progress

Training will show real-time progress:

```
================================================================================
============================= VQA PIPELINE STARTED =============================
================================================================================
    Mode: train
    Start time: 2026-01-10 00:01:28

-------------------- System Information --------------------
    PyTorch version: 2.9.1+cu128
    GPU 0: NVIDIA GeForce RTX 3060 (11.6 GB)

============================ STAGE 1: DATA PIPELINE ============================
✓ Loaded 37077 data samples
✓ Data split: 29661 train / 3707 val / 3709 test
✓ DATA PIPELINE completed in 7.22s

=========================== STAGE 2: MODEL PIPELINE ============================
✓ Total parameters: 243,163,583
✓ MODEL PIPELINE completed in 6.97s

========================== STAGE 3: TRAINING PIPELINE ==========================
Epoch 1 [Train]: 100%|██████████| 1853/1853 [05:10<00:00, loss=1.88, acc=0.76]
Epoch 1 [Val]: 100%|██████████| 58/58 [00:13<00:00, loss=0.35, acc=0.96]
✓ New best accuracy: 0.9625

=============================== PIPELINE SUMMARY ===============================
    Status: SUCCESS
    Best metric: 0.9625
```

---

## Configuration Options

### Using Command-Line Arguments

```bash
# All available options
python -m src.core.vqa_pipeline --help

# Common configurations
python -m src.core.vqa_pipeline \
    --mode train \
    --epochs 20 \
    --batch-size 32 \
    --learning-rate 2e-5 \
    --visual-backbone vit \
    --text-encoder phobert
```

### Using YAML Config File

Create or modify `configs/pipeline_config.yaml`:

```yaml
mode: train
output_dir: outputs

data:
  batch_size: 32
  augmentation_strength: medium

model:
  visual_backbone: vit
  text_encoder_type: phobert
  use_moe: false

training:
  num_epochs: 20
  learning_rate: 2.0e-5
  use_amp: true
```

Run with:
```bash
python -m src.core.vqa_pipeline --config configs/pipeline_config.yaml
```

### Using Python API

```python
from src.core import VQAPipeline, VQAPipelineConfig
from src.core import DataPipelineConfig, ModelPipelineConfig, TrainingPipelineConfig

config = VQAPipelineConfig(
    mode="train",
    data=DataPipelineConfig(batch_size=32),
    model=ModelPipelineConfig(visual_backbone="vit"),
    training=TrainingPipelineConfig(num_epochs=20),
)

pipeline = VQAPipeline(config)
results = pipeline.run()
```

---

## Training Configurations

### Recommended Settings by GPU

| GPU VRAM | Batch Size | Gradient Accum | Precision |
|----------|------------|----------------|-----------|
| 8 GB | 8 | 4 | float16 |
| 12 GB | 16 | 2 | float16 |
| 24 GB | 32 | 1 | float16 |
| 48 GB+ | 64 | 1 | float32 |

Example for 8GB GPU:
```bash
python -m src.core.vqa_pipeline \
    --mode train \
    --batch-size 8 \
    --epochs 20
```

### CPU Training

For CPU-only training:
```bash
python -m src.core.vqa_pipeline --mode train --batch-size 4 --epochs 10
```

Note: CPU training will be significantly slower.

---

## Outputs

After training, you'll find:

```
AutovivqaModelBuilder/
├── checkpoints/
│   ├── best_model.pt          # Best model checkpoint
│   ├── checkpoint_epoch_1.pt  # Epoch checkpoints
│   └── checkpoint_epoch_N.pt
│
├── outputs/
│   └── pipeline_summary.json  # Training summary
│
└── logs/
    ├── tensorboard/           # TensorBoard logs
    └── pipeline/              # Pipeline logs
```

### View TensorBoard

```bash
tensorboard --logdir logs/tensorboard
# Open http://localhost:6006 in browser
```

### Load Trained Model

```python
import torch
from src.modeling.meta_arch import VietnameseVQAModel, VQAModelConfig

# Load checkpoint
checkpoint = torch.load('checkpoints/best_model.pt', weights_only=False)

# Recreate model
config = checkpoint['model_config']
model = VietnameseVQAModel(config)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
```

---

## Troubleshooting

### Issue: CUDA Out of Memory

**Error:**
```
RuntimeError: CUDA out of memory
```

**Solution:**
Reduce batch size:
```bash
python -m src.core.vqa_pipeline --mode train --batch-size 8
```

### Issue: Data Not Found

**Error:**
```
FileNotFoundError: data/raw/images not found
```

**Solution:**
Download data first:
```bash
bash src/cli/download_data.sh
```

### Issue: Slow Training

**Solutions:**
1. Enable mixed precision (default):
   ```yaml
   training:
     use_amp: true
   ```

2. Increase workers:
   ```yaml
   data:
     num_workers: 8
   ```

3. Use GPU instead of CPU

### Issue: Import Errors

**Solution:**
Ensure all dependencies are installed:
```bash
poetry install
# or
pip install -r requirements.txt
```

---

## Next Steps

After completing this guide:

1. **Read the full documentation:**
   - [Pipeline Usage Guide](pipeline_usage.md)
   - [VQA Architecture](vqa_architecture.md)

2. **Try advanced features:**
   - Enable MOE: `--use-moe`
   - Enable Knowledge Base: `--use-knowledge`

3. **Experiment with configurations:**
   - Different visual backbones
   - Different fusion types
   - Hyperparameter tuning

4. **Run evaluation:**
   ```bash
   python -m src.core.vqa_pipeline --mode evaluate --resume checkpoints/best_model.pt
   ```

---

## Getting Help

- Check the [documentation](docs/)
- Open an issue on GitHub
- Contact: richardnguyen0715@gmail.com

---

Happy training! 🚀
