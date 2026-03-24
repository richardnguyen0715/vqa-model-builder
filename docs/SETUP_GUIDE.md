# VIVQA Pipeline Setup Script

Automated end-to-end setup script for the ViMoE-VQA model evaluation pipeline on the VIVQA (Vietnamese VQA) dataset.

## Overview

This script automates the complete workflow:

1. **Clone Repository** - Downloads the vqa-model-builder project from GitHub
2. **Setup Environment** - Creates Python virtual environment and installs dependencies via Poetry
3. **Download Data** - Fetches VIVQA dataset from Kaggle
4. **Download Model** - Downloads ViMoE-VQA pre-trained model from Hugging Face Hub
5. **Run Evaluation** - Executes model evaluation pipeline on VIVQA test set

## Prerequisites

### System Requirements

- **OS**: macOS, Linux, or WSL on Windows
- **Git**: For cloning the repository
- **Python 3.11+**: For running the project
- **pip**: For installing Poetry
- **Internet Connection**: For downloading data and models (>10 GB recommended)
- **Disk Space**: 
  - Repository: ~500 MB
  - Dataset: ~3-5 GB (VIVQA images + CSV)
  - Model: ~2-3 GB (ViMoE-VQA)
  - Total: ~6-9 GB recommended

### API Credentials

#### Kaggle API (Required for Dataset)

1. Go to https://www.kaggle.com/settings/account
2. Scroll to "API" section and click "Create New Token"
3. This downloads `kaggle.json`
4. Configure:
   ```bash
   mkdir -p ~/.kaggle
   mv ~/Downloads/kaggle.json ~/.kaggle/
   chmod 600 ~/.kaggle/kaggle.json
   ```

Or set environment variables:
```bash
export KAGGLE_USERNAME="your_username"
export KAGGLE_KEY="your_api_key"
```

#### Hugging Face Token (Optional, for private models)

```bash
export HF_TOKEN="your_huggingface_token"
```

Get token from: https://huggingface.co/settings/tokens

## Quick Start

### Full Automatic Setup

```bash
bash setup_vivqa_pipeline.sh
```

This runs all steps sequentially with interactive prompts.

### Skip Certain Steps

```bash
# Skip cloning if repository already exists
bash setup_vivqa_pipeline.sh --skip-clone

# Skip environment setup if Poetry is already configured
bash setup_vivqa_pipeline.sh --skip-clone --skip-env

# Only download and evaluate (skip all setup)
bash setup_vivqa_pipeline.sh --skip-clone --skip-env --skip-data --skip-model

# Run only evaluation (assumes all data and model are ready)
bash setup_vivqa_pipeline.sh --skip-clone --skip-env --skip-data --skip-model --skip-eval=false
```

### Verbose Mode

```bash
bash setup_vivqa_pipeline.sh --verbose
```

Shows detailed output of all commands.

### Show Help

```bash
bash setup_vivqa_pipeline.sh --help
```

## Usage Examples

### Example 1: Complete Fresh Setup

```bash
# From any directory
cd ~/projects
bash vqa-model-builder/setup_vivqa_pipeline.sh

# Follow prompts when asked about existing directories
```

**Expected Duration**: ~30-60 minutes (depending on internet speed)

### Example 2: Quick Re-evaluation

If you already have data and model downloaded:

```bash
cd vqa-model-builder
bash ../setup_vivqa_pipeline.sh --skip-clone --skip-env --skip-data --skip-model
```

### Example 3: Development Setup

For developers making code changes:

```bash
cd vqa-model-builder
source .venv/bin/activate
python src/core/vivqa_eval_cli.py --checkpoint checkpoints/RichardNguyen_ViMoE-VQA/pytorch_model.bin --help
```

## Step-by-Step Breakdown

### Step 1: Clone Repository

```bash
git clone https://github.com/richardnguyen0715/vqa-model-builder.git
cd vqa-model-builder
```

**What it does:**
- Clones the GitHub repository
- Checks if directory already exists (interactive)
- Creates local copy of entire project

**Time**: ~1-2 minutes

### Step 2: Setup Environment

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install poetry
poetry install
```

**What it does:**
- Creates isolated Python environment
- Installs Poetry (Python dependency manager)
- Installs all project dependencies (PyTorch, Transformers, etc.)
- Downloads ~2-3 GB of Python packages

**Time**: ~10-20 minutes (first time, ~2 minutes if cached)

**Note**: Large packages like PyTorch take time to download and install

### Step 3: Download VIVQA Dataset

```bash
bash src/cli/download_data_ver2.sh --dataset vivqa
```

**What it does:**
- Downloads VIVQA dataset from Kaggle
- Extracts and organizes files
- Consolidates images from train/test splits
- Places data in `data/vivqa/`

**Structure created:**
```
data/vivqa/
├── train.csv          # 11,999 samples
├── test.csv           # 3,001 samples
└── images/            # ~15,000 images
    ├── COCO_xxxxx.jpg
    ├── COCO_xxxxx.jpg
    └── ...
```

**Time**: ~5-15 minutes (depends on internet speed)

**Note**: Requires Kaggle credentials configured

### Step 4: Download Model

```bash
bash src/cli/download_model.sh
```

**What it does:**
- Downloads ViMoE-VQA model from Hugging Face Hub
- Saves model checkpoint (pytorch_model.bin or model.safetensors)
- Downloads supporting files (config.json, tokenizer files)

**Structure created:**
```
checkpoints/
└── RichardNguyen_ViMoE-VQA/
    ├── config.json
    ├── pytorch_model.bin        (~2-3 GB)
    ├── tokenizer.json
    └── ...
```

**Time**: ~5-10 minutes (depends on internet speed)

### Step 5: Run Evaluation

```bash
python src/core/vivqa_eval_cli.py \
    --checkpoint checkpoints/RichardNguyen_ViMoE-VQA/pytorch_model.bin \
    --csv-path data/vivqa/test.csv \
    --images-dir data/vivqa/images \
    --batch-size 32 \
    --device auto
```

**What it does:**
- Loads model checkpoint and tokenizer
- Processes 3,001 test samples
- Generates predictions for each question
- Computes 8 evaluation metrics:
  - Accuracy (exact match)
  - Precision, Recall, F1
  - BLEU, ROUGE-L, METEOR, CIDEr

**Output files:**
```
outputs/evaluation/
├── metrics_2026-03-24_20-30-45.json       # Evaluation metrics
└── predictions_2026-03-24_20-30-45.json   # Model predictions & references
```

**Time**: ~10-30 minutes (depends on GPU availability)

**Note**: Uses GPU if available, falls back to CPU

## Output Files

### After Completion

```
vqa-model-builder/
├── data/
│   └── vivqa/
│       ├── images/               # ~15,000 COCO images
│       ├── train.csv             # Training set
│       └── test.csv              # Test set
│
├── checkpoints/
│   └── RichardNguyen_ViMoE-VQA/
│       ├── pytorch_model.bin     # Model weights (~2.5 GB)
│       ├── config.json
│       └── ...
│
├── outputs/
│   └── evaluation/
│       ├── metrics_*.json        # Evaluation results
│       └── predictions_*.json    # Model predictions
│
└── logs/
    ├── data_loader/              # Dataset download logs
    ├── data_process/             # Processing logs
    └── tensorboard/              # TensorBoard events
```

### Evaluation Results

**metrics_*.json** contains:
```json
{
  "accuracy": 0.45,
  "precision": 0.52,
  "recall": 0.48,
  "f1": 0.50,
  "bleu": 0.35,
  "rouge_l": 0.42,
  "meteor": 0.38,
  "cider": 0.41,
  "total_samples": 3001,
  "evaluation_time": 1250.5
}
```

**predictions_*.json** contains:
```json
[
  {
    "id": "0",
    "question": "Cái gì là màu gì?",
    "reference": "mũ đỏ",
    "prediction": "mũ",
    "image_id": "COCO_000001.jpg"
  },
  ...
]
```

## Troubleshooting

### Kaggle Authentication Failed

```
Error: Could not read api key, are you sure your kaggle.json is configured?
```

**Solution:**
```bash
# Check if kaggle.json exists
ls ~/.kaggle/kaggle.json

# If not, download from https://www.kaggle.com/settings/account
# Then verify permissions
chmod 600 ~/.kaggle/kaggle.json

# Or set environment variables
export KAGGLE_USERNAME="your_username"
export KAGGLE_KEY="your_api_key"
```

### Hugging Face Download Failed

```
Error: Failed to download model from Hugging Face Hub
```

**Solution:**
```bash
# Check internet connection
ping huggingface.co

# Try manual download
cd vqa-model-builder
source .venv/bin/activate
python -c "from huggingface_hub import snapshot_download; snapshot_download('RichardNguyen/ViMoE-VQA', local_dir='checkpoints/RichardNguyen_ViMoE-VQA')"
```

### Out of Memory (OOM) During Evaluation

```
RuntimeError: CUDA out of memory
```

**Solution:**
```bash
# Use smaller batch size
python src/core/vivqa_eval_cli.py \
    --checkpoint checkpoints/RichardNguyen_ViMoE-VQA/pytorch_model.bin \
    --batch-size 16  # Reduced from 32

# Or use CPU
python src/core/vivqa_eval_cli.py \
    --checkpoint checkpoints/RichardNguyen_ViMoE-VQA/pytorch_model.bin \
    --device cpu
```

### Poetry Installation Issues

```
Error: Poetry command not found
```

**Solution:**
```bash
# Upgrade pip and reinstall poetry
pip install --upgrade pip setuptools wheel
pip install poetry --upgrade

# Or use system Python directly
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt  # If available
```

### PyTorch 2.6+ Checkpoint Loading Error

```
_pickle.UnpicklingError: Weights only load failed...
WeightsUnpickler error: Unsupported global: GLOBAL numpy._core.multiarray.scalar
```

**Solution:**
This is a PyTorch 2.6+ security feature. The issue has been fixed in the current version of the code.

Make sure you have the latest version:
```bash
cd vqa-model-builder
git pull origin main
```

If you still encounter it, update PyTorch:
```bash
pip install --upgrade torch
```

See [PyTorch 2.6 Compatibility Guide](docs/PYTORCH26_COMPATIBILITY.md) for more details.

### Slow Downloads

- Check internet connection: `speedtest-cli`
- Use wired connection instead of WiFi
- Close other bandwidth-consuming applications
- Try resuming: Scripts have resume capability built-in

## Manual Execution

If you want to run steps individually:

```bash
cd vqa-model-builder

# Activate environment
source .venv/bin/activate

# Download data
bash src/cli/download_data_ver2.sh --dataset vivqa

# Download model
bash src/cli/download_model.sh

# Run evaluation
python src/core/vivqa_eval_cli.py \
    --checkpoint checkpoints/RichardNguyen_ViMoE-VQA/pytorch_model.bin \
    --batch-size 32 \
    --output-dir outputs/evaluation
```

## Hardware Requirements

### Minimum

- CPU: Any modern processor
- RAM: 8 GB
- GPU: Not required (but slower)
- Storage: 15 GB

### Recommended

- CPU: Intel i7/AMD Ryzen 7 or better
- RAM: 16+ GB
- GPU: NVIDIA GPU with 6GB+ VRAM (CUDA compatible)
- Storage: 20+ GB SSD

### Optimal

- CPU: Intel i9/AMD Ryzen 9
- RAM: 32+ GB
- GPU: NVIDIA RTX 3090/4090 or H100
- Storage: 50+ GB NVMe SSD

## FAQs

**Q: How long does the complete setup take?**
A: 30-60 minutes for first-time setup, depending on internet speed and hardware.

**Q: Can I run this on Windows?**
A: Yes, use WSL2 (Windows Subsystem for Linux) or Git Bash.

**Q: Can I skip the automatic setup and do it manually?**
A: Yes, each step can be run independently. See Step-by-Step Breakdown section.

**Q: How do I update the model or dataset?**
A: Use `--skip-*` flags to skip unchanged steps, e.g., `--skip-data` to skip data download.

**Q: What if my internet connection drops?**
A: Most download scripts have resume capability. Re-run the script and it will continue from where it stopped.

**Q: How do I evaluate on custom data?**
A: Place your CSV and images in appropriate directories and modify evaluation command paths.

## Additional Resources

- **Documentation**: [docs/vivqa_usage_guide.md](docs/vivqa_usage_guide.md)
- **Examples**: [examples/vivqa_examples.py](examples/vivqa_examples.py)
- **Configuration**: [configs/vivqa_evaluation_configs.yaml](configs/vivqa_evaluation_configs.yaml)
- **Model Card**: https://huggingface.co/RichardNguyen/ViMoE-VQA
- **Dataset**: https://www.kaggle.com/datasets/dngtrungngha/vivqa

## Support & Issues

For issues, questions, or contributions:
- Check existing [documentation](docs/)
- Review [GitHub issues](https://github.com/richardnguyen0715/vqa-model-builder/issues)
- Create new issue with:
  - OS and Python version
  - Error message and logs
  - Steps to reproduce
  - Screenshot if applicable

## License

This project is licensed under the Apache License 2.0. See [LICENSE](LICENSE) file for details.

## Citation

If you use this pipeline in your research, please cite:

```bibtex
@software{vivqa_pipeline,
  title={ViMoE-VQA: Vietnamese Multimodal Mixture of Experts for Visual Question Answering},
  author={Nguyen, Richard},
  year={2026},
  url={https://github.com/richardnguyen0715/vqa-model-builder}
}
```

---

**Last Updated**: March 24, 2026  
**Version**: 1.0.0
