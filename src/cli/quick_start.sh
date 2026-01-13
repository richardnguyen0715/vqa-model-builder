#!/bin/bash

# =============================================================================
# Vietnamese VQA Pipeline - Quick Start Script
# Downloads data and runs training with default configuration
# =============================================================================

set -e

# Suppress Python warnings
export PYTHONWARNINGS="ignore::FutureWarning,ignore::RuntimeWarning"

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}"
echo "╔══════════════════════════════════════════════════════════════════════════════╗"
echo "║                     Vietnamese VQA - Quick Start                             ║"
echo "╚══════════════════════════════════════════════════════════════════════════════╝"
echo -e "${NC}"

# Step 1: Download data
echo -e "${GREEN}[1/3]${NC} Downloading data from Kaggle..."
python -m src.data.download_data

# Step 2: Verify data
echo -e "${GREEN}[2/3]${NC} Verifying data..."
if [ ! -d "data/raw/images" ] || [ ! -f "data/raw/texts/evaluate_60k_data_balanced_preprocessed.csv" ]; then
    echo "Error: Data download incomplete. Please check the logs."
    exit 1
fi

IMAGE_COUNT=$(ls data/raw/images | wc -l)
echo "  Found $IMAGE_COUNT images"

# Step 3: Run training
echo -e "${GREEN}[3/3]${NC} Starting training pipeline..."
bash src/cli/run_pipeline.sh --mode train --batch-size 16 --epochs 10

echo ""
echo -e "${GREEN}Quick start completed!${NC}"
echo "Check outputs/ directory for trained model"
echo "Check logs/pipeline/ for detailed logs"
