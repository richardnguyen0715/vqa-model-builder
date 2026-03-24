#!/bin/bash

##############################################################################
# Quick Reference - VIVQA Pipeline Setup
# 
# Save this file to use as a quick checklist
##############################################################################

# OPTION 1: Full Automatic Setup (Recommended)
# ============================================
cd ~/projects
bash vqa-model-builder/setup_vivqa_pipeline.sh


# OPTION 2: With Custom Settings
# ===============================

# Skip repository clone (already have it)
bash ../setup_vivqa_pipeline.sh --skip-clone

# Skip environment setup (poetry already installed)
bash ../setup_vivqa_pipeline.sh --skip-clone --skip-env

# Verbose mode (see all details)
bash ../setup_vivqa_pipeline.sh --verbose

# Only run evaluation (data & model ready)
bash ../setup_vivqa_pipeline.sh \
  --skip-clone \
  --skip-env \
  --skip-data \
  --skip-model


# OPTION 3: Manual Step-by-Step Execution
# ========================================

# Step 1: Clone
git clone https://github.com/richardnguyen0715/vqa-model-builder.git
cd vqa-model-builder

# Step 2: Setup environment
python3 -m venv .venv
source .venv/bin/activate
pip install poetry
poetry install

# Step 3: Download data
bash src/cli/download_data_ver2.sh --dataset vivqa

# Step 4: Download model
bash src/cli/download_model.sh

# Step 5: Run evaluation
python src/core/vivqa_eval_cli.py \
  --checkpoint checkpoints/RichardNguyen_ViMoE-VQA/pytorch_model.bin \
  --csv-path data/vivqa/test.csv \
  --images-dir data/vivqa/images \
  --batch-size 32 \
  --output-dir outputs/evaluation


# SETUP CHECKLIST
# ===============

[ ] Git installed (git --version)
[ ] Python 3.11+ installed (python3 --version)
[ ] pip available (pip --version)
[ ] 15+ GB disk space available (df -h)
[ ] Kaggle API credentials configured (~/.kaggle/kaggle.json)
[ ] Internet connection stable
[ ] HF_TOKEN set if using private models


# KAGGLE SETUP
# ============

# 1. Go to https://www.kaggle.com/settings/account
# 2. Click "Create New Token" in API section
# 3. Configure:
mkdir -p ~/.kaggle
mv ~/Downloads/kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json

# Or use environment variables:
export KAGGLE_USERNAME="your_username"
export KAGGLE_KEY="your_api_key"


# HUGGING FACE SETUP (Optional)
# =============================

export HF_TOKEN="your_token_from_https://huggingface.co/settings/tokens"


# TROUBLESHOOTING
# ===============

# Check Kaggle
kaggle --version
kaggle datasets list

# Check Poetry
poetry --version
poetry show

# Check Torch
python -c "import torch; print(f'Torch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"

# Manual model download
python << 'EOF'
from huggingface_hub import snapshot_download
snapshot_download(
    'RichardNguyen/ViMoE-VQA',
    local_dir='checkpoints/RichardNguyen_ViMoE-VQA'
)
EOF

# Manual data download
python << 'EOF'
import kagglehub
kagglehub.dataset_download("dngtrungngha/vivqa", path="data/vivqa")
EOF


# POST-EXECUTION
# ==============

# View results
cat outputs/evaluation/metrics_*.json | python -m json.tool

# View logs
tail -f logs/data_process/vivqa_evaluation.log

# View predictions
python -m json.tool outputs/evaluation/predictions_*.json | less

# Check directory size
du -sh . checkpoints data outputs
