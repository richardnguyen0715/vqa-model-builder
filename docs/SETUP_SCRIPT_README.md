# Setup Script Implementation Summary

## Files Created

### 1. **setup_vivqa_pipeline.sh** (Main Script)
   - **Location**: Root directory
   - **Size**: ~650 lines
   - **Purpose**: Complete end-to-end automation
   - **Features**:
     - ✓ Full pipeline execution (clone → setup → download → evaluate)
     - ✓ Step-by-step progress with colored output
     - ✓ Dependency checking (git, python, pip)
     - ✓ Error handling and recovery
     - ✓ Interactive prompts for existing files
     - ✓ Skip options for individual steps
     - ✓ Verbose logging mode
     - ✓ Comprehensive summary at end

### 2. **SETUP_GUIDE.md** (Comprehensive Documentation)
   - **Location**: Root directory
   - **Size**: ~800 lines
   - **Purpose**: Detailed user guide
   - **Contents**:
     - Overview and prerequisites
     - Quick start examples
     - Step-by-step breakdown for each phase
     - Troubleshooting section
     - Hardware requirements
     - FAQs
     - Output file descriptions
     - Manual execution options

### 3. **QUICK_REFERENCE.sh** (Quick Checklist)
   - **Location**: Root directory
   - **Size**: ~150 lines
   - **Purpose**: Quick copy-paste commands
   - **Contents**:
     - Common usage patterns
     - Checklist for setup
     - Troubleshooting commands
     - Post-execution commands

## Execution Flow

```
setup_vivqa_pipeline.sh
├── Check Command Line Arguments
├── Check Dependencies
│   ├── git
│   ├── python3
│   ├── pip
│   └── curl/wget
│
├── Step 1: Clone Repository
│   ├── Check if already exists
│   ├── Prompt user if duplication
│   └── Clone if needed
│
├── Step 2: Setup Environment
│   ├── Create .venv
│   ├── Activate virtual environment
│   ├── Install Poetry
│   ├── poetry install (install dependencies)
│   └── Total: ~2-20 minutes
│
├── Step 3: Download VIVQA Dataset
│   ├── Check if already exists
│   ├── Run: src/cli/download_data_ver2.sh
│   ├── Organizes: data/vivqa/
│   └── Total: ~5-15 minutes
│
├── Step 4: Download Model
│   ├── Check if already exists
│   ├── Run: src/cli/download_model.sh
│   ├── Saves to: checkpoints/RichardNguyen_ViMoE-VQA/
│   └── Total: ~5-10 minutes
│
├── Step 5: Run Evaluation
│   ├── Check prerequisites
│   ├── Run: vivqa_eval_cli.py
│   ├── Outputs to: outputs/evaluation/
│   └── Total: ~10-30 minutes
│
├── Print Summary
│   ├── Check created directories
│   ├── Count files
│   └── Show disk usage
│
└── Print Next Steps
    └── Guide for post-execution
```

## Key Features

### 1. Dependency Checking
```bash
✓ Verifies git, python3, pip, curl/wget
✓ Shows versions found
✓ Exits gracefully if missing
```

### 2. Error Handling
```bash
✓ set -e: Exit on any error
✓ Try-catch patterns for recovery
✓ User-friendly error messages
✓ Suggestions for fixing issues
```

### 3. Interactive Prompts
```bash
✓ Asks before overwriting
✓ Asks before re-downloading
✓ Uses read commands for user input
✓ Respects user choices
```

### 4. Progress Tracking
```bash
✓ Color-coded output (Red/Green/Yellow/Blue/Cyan)
✓ Stage headers with dividers
✓ Step indicators (▶▶▶, ✓, ✗, ⚠, ℹ)
✓ Verbose logging option
```

### 5. Flexibility
```bash
✓ --skip-clone     : Skip repository cloning
✓ --skip-env       : Skip environment setup
✓ --skip-data      : Skip data download
✓ --skip-model     : Skip model download
✓ --skip-eval      : Skip evaluation
✓ --verbose        : Detailed output
✓ --help           : Show help screen
```

### 6. Resume Capability
```bash
✓ Checks if files already exist
✓ Asks user before re-downloading
✓ Allows skipping completed steps
✓ Supports partial re-execution
```

## Usage Patterns

### Pattern 1: Fresh Installation
```bash
bash setup_vivqa_pipeline.sh
# Runs all 5 steps automatically
# ~30-60 minutes total
```

### Pattern 2: Skip Already-Done Steps
```bash
bash setup_vivqa_pipeline.sh --skip-clone --skip-env
# Skips clone and environment setup
# Only downloads data, model, and runs evaluation
```

### Pattern 3: Re-run Evaluation Only
```bash
bash setup_vivqa_pipeline.sh \
  --skip-clone --skip-env --skip-data --skip-model
# Only runs evaluation with existing data and model
# ~15-30 minutes
```

### Pattern 4: Debugging with Verbose
```bash
bash setup_vivqa_pipeline.sh --verbose
# Shows every command execution
# Helpful for troubleshooting
```

### Pattern 5: Just Show Help
```bash
bash setup_vivqa_pipeline.sh --help
# Displays comprehensive help
# No execution performed
```

## Additional Scripts Used

The main script calls these existing support scripts:

1. **src/cli/download_data_ver2.sh**
   - Downloads VIVQA from Kaggle
   - Supports both VQA and VIVQA
   - Used as: `bash src/cli/download_data_ver2.sh --dataset vivqa`

2. **src/cli/download_model.sh**
   - Downloads from Hugging Face Hub
   - Auto-installs huggingface_hub if needed
   - Used as: `bash src/cli/download_model.sh`

3. **src/core/vivqa_eval_cli.py**
   - Runs evaluation on VIVQA dataset
   - Computes 8 metrics
   - Saves results to JSON
   - Used as: `python src/core/vivqa_eval_cli.py --checkpoint ...`

## Output Structure

After successful execution:

```
vqa-model-builder/
├── .venv/                                  # Virtual environment
├── vqa-model-builder/                      # Cloned repository (if not skip-clone)
├── data/
│   └── vivqa/
│       ├── train.csv                       # Training set (~11,999 samples)
│       ├── test.csv                        # Test set (~3,001 samples)
│       └── images/                         # ~15,000 COCO images
│           ├── COCO_val2014_000000000001.jpg
│           └── ...
├── checkpoints/
│   └── RichardNguyen_ViMoE-VQA/
│       ├── config.json
│       ├── pytorch_model.bin               # Model weights (~2.5 GB)
│       ├── tokenizer.json
│       └── ...
├── outputs/
│   └── evaluation/
│       ├── metrics_2026-03-24_HH-MM-SS.json
│       └── predictions_2026-03-24_HH-MM-SS.json
└── logs/
    ├── data_loader/                        # Dataset download logs
    ├── data_process/                       # Processing logs
    └── tensorboard/                        # TensorBoard events
```

## Estimated Timeline

| Step | Time | Notes |
|------|------|-------|
| Dependencies Check | 30s | Immediate |
| Repository Clone | 1-2 min | Depends on network |
| Environment Setup | 2-20 min | 20 min first-time (caching) |
| Data Download | 5-15 min | Kaggle speed |
| Model Download | 5-10 min | ~2-3 GB file |
| Evaluation | 10-30 min | GPU ~10min, CPU ~30min |
| **Total** | **30-90 min** | ~60 min typical |

## Hardware Tested On

- **macOS**: Apple Silicon M1/M2 (primary)
- **Linux**: NVIDIA GPU with CUDA
- **WSL2**: Windows Subsystem for Linux

## Dependencies Installed

The script installs:
- **Poetry**: Dependency management
- **PyTorch**: Deep learning framework
- **Transformers**: HuggingFace models
- **Kagglehub**: Dataset downloading
- **And 30+ other packages** (see pyproject.toml)

## Troubleshooting

### Issue: "Command not found: git"
**Solution**: Install Git
```bash
# macOS
brew install git

# Ubuntu/Debian
sudo apt-get install git

# Fedora
sudo dnf install git
```

### Issue: "Error: Python not found"
**Solution**: Install Python 3.11+
```bash
# macOS
brew install python@3.11

# Ubuntu/Debian
sudo apt-get install python3.11

# Check: python3 --version
```

### Issue: "Kaggle authentication failed"
**Solution**: Configure Kaggle API
```bash
# Download token from: https://www.kaggle.com/settings/account
mkdir -p ~/.kaggle
mv ~/Downloads/kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json
```

### Issue: "CUDA out of memory" during evaluation
**Solution**: Use smaller batch size or CPU
```bash
python src/core/vivqa_eval_cli.py \
  --checkpoint checkpoints/RichardNguyen_ViMoE-VQA/pytorch_model.bin \
  --batch-size 16 \
  --device cpu
```

## Next Steps After Setup

1. **Review Results**
   ```bash
   cat outputs/evaluation/metrics_*.json | python -m json.tool
   ```

2. **Analyze Predictions**
   ```bash
   cat outputs/evaluation/predictions_*.json | python -m json.tool | less
   ```

3. **Check Logs**
   ```bash
   tail -f logs/data_process/vivqa_evaluation.log
   ```

4. **Re-run with Different Parameters**
   ```bash
   source .venv/bin/activate
   python src/core/vivqa_eval_cli.py --help
   ```

## Advanced Usage

### Custom Model Evaluation
```bash
# Download different model
bash src/cli/download_model.sh --model-id openai/clip-vit-large-patch14

# Evaluate custom model
python src/core/vivqa_eval_cli.py \
  --checkpoint checkpoints/openai_clip-vit-large-patch14/pytorch_model.bin \
  --batch-size 64 \
  --num-beams 3
```

### Custom Dataset
```bash
# Place your CSV in: data/vivqa/test.csv
# Place images in: data/vivqa/images/
# Then run evaluation
python src/core/vivqa_eval_cli.py \
  --checkpoint checkpoints/RichardNguyen_ViMoE-VQA/pytorch_model.bin \
  --csv-path data/vivqa/test.csv
```

## Version Information

- **Script Version**: 1.0.0
- **Created**: March 24, 2026
- **Python Required**: 3.11 - 3.14
- **Poetry Version**: Latest
- **Compatible OS**: macOS, Linux, WSL2

## Support

For issues:
1. Check SETUP_GUIDE.md for detailed troubleshooting
2. Run with `--verbose` flag for debugging
3. See QUICK_REFERENCE.sh for common commands
4. Check project GitHub issues

---

**Created**: March 24, 2026  
**Last Updated**: March 24, 2026
