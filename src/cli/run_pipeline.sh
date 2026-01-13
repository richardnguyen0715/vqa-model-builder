#!/bin/bash

# =============================================================================
# Vietnamese VQA Pipeline - Training Script
# AutoViVQA Model Builder
# =============================================================================

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Print functions
print_header() {
    echo -e "${BLUE}"
    echo "╔══════════════════════════════════════════════════════════════════════════════╗"
    echo "║                     Vietnamese VQA Pipeline - Training                       ║"
    echo "║                           AutoViVQA Model Builder                            ║"
    echo "╚══════════════════════════════════════════════════════════════════════════════╝"
    echo -e "${NC}"
}

print_step() {
    echo -e "${GREEN}[STEP]${NC} $1"
}

print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Default configuration
MODE="train"
IMAGES_DIR="data/raw/images"
TEXT_FILE="data/raw/texts/evaluate_60k_data_balanced_preprocessed.csv"
BATCH_SIZE=32
EPOCHS=20
LEARNING_RATE=2e-5
OUTPUT_DIR="outputs"
VISUAL_BACKBONE="vit"
TEXT_ENCODER="phobert"
USE_MOE=""
USE_KNOWLEDGE=""
RESUME=""
CONFIG_FILE=""

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --mode)
            MODE="$2"
            shift 2
            ;;
        --config)
            CONFIG_FILE="$2"
            shift 2
            ;;
        --images-dir)
            IMAGES_DIR="$2"
            shift 2
            ;;
        --text-file)
            TEXT_FILE="$2"
            shift 2
            ;;
        --batch-size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --epochs)
            EPOCHS="$2"
            shift 2
            ;;
        --learning-rate)
            LEARNING_RATE="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --visual-backbone)
            VISUAL_BACKBONE="$2"
            shift 2
            ;;
        --text-encoder)
            TEXT_ENCODER="$2"
            shift 2
            ;;
        --use-moe)
            USE_MOE="--use-moe"
            shift
            ;;
        --use-knowledge)
            USE_KNOWLEDGE="--use-knowledge"
            shift
            ;;
        --resume)
            RESUME="--resume $2"
            shift 2
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --mode MODE              Pipeline mode: train, evaluate, inference (default: train)"
            echo "  --config FILE            Path to YAML configuration file"
            echo "  --images-dir DIR         Path to images directory"
            echo "  --text-file FILE         Path to text CSV file"
            echo "  --batch-size SIZE        Batch size for training (default: 32)"
            echo "  --epochs N               Number of training epochs (default: 20)"
            echo "  --learning-rate LR       Learning rate (default: 2e-5)"
            echo "  --output-dir DIR         Output directory (default: outputs)"
            echo "  --visual-backbone NAME   Visual backbone: vit, resnet, clip (default: vit)"
            echo "  --text-encoder NAME      Text encoder: phobert, bert (default: phobert)"
            echo "  --use-moe                Enable Mixture of Experts"
            echo "  --use-knowledge          Enable Knowledge/RAG module"
            echo "  --resume PATH            Path to checkpoint to resume training"
            echo "  --help                   Show this help message"
            exit 0
            ;;
        *)
            print_error "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Main execution
print_header

print_step "Checking environment..."

# Check Python
if ! command -v python &> /dev/null; then
    print_error "Python is not installed or not in PATH"
    exit 1
fi

PYTHON_VERSION=$(python --version 2>&1)
print_info "Python version: $PYTHON_VERSION"

# Check if we're in the right directory
if [ ! -f "pyproject.toml" ]; then
    print_warning "pyproject.toml not found. Make sure you're in the project root directory."
fi

# Check data exists
print_step "Checking data..."
if [ ! -d "$IMAGES_DIR" ]; then
    print_warning "Images directory not found: $IMAGES_DIR"
    print_info "You may need to download the data first using: python -m src.data.download_data"
fi

if [ ! -f "$TEXT_FILE" ]; then
    print_warning "Text file not found: $TEXT_FILE"
fi

# Create output directory
print_step "Creating output directory..."
mkdir -p "$OUTPUT_DIR"
mkdir -p "logs/pipeline"

# Print configuration
print_step "Configuration:"
echo "  Mode: $MODE"
echo "  Images directory: $IMAGES_DIR"
echo "  Text file: $TEXT_FILE"
echo "  Batch size: $BATCH_SIZE"
echo "  Epochs: $EPOCHS"
echo "  Learning rate: $LEARNING_RATE"
echo "  Visual backbone: $VISUAL_BACKBONE"
echo "  Text encoder: $TEXT_ENCODER"
echo "  Output directory: $OUTPUT_DIR"
if [ -n "$USE_MOE" ]; then
    echo "  MOE: Enabled"
fi
if [ -n "$USE_KNOWLEDGE" ]; then
    echo "  Knowledge/RAG: Enabled"
fi
if [ -n "$CONFIG_FILE" ]; then
    echo "  Config file: $CONFIG_FILE"
fi

# Run pipeline
print_step "Starting VQA Pipeline..."
echo ""

# Set Python warnings environment variable
export PYTHONWARNINGS="ignore::FutureWarning,ignore::RuntimeWarning"

# Build command
CMD="python -m src.core.vqa_pipeline"

if [ -n "$CONFIG_FILE" ]; then
    CMD="$CMD --config $CONFIG_FILE"
else
    CMD="$CMD --mode $MODE"
    CMD="$CMD --images-dir $IMAGES_DIR"
    CMD="$CMD --text-file $TEXT_FILE"
    CMD="$CMD --batch-size $BATCH_SIZE"
    CMD="$CMD --epochs $EPOCHS"
    CMD="$CMD --learning-rate $LEARNING_RATE"
    CMD="$CMD --visual-backbone $VISUAL_BACKBONE"
    CMD="$CMD --text-encoder $TEXT_ENCODER"
    CMD="$CMD --output-dir $OUTPUT_DIR"
    CMD="$CMD $USE_MOE $USE_KNOWLEDGE $RESUME"
fi

print_info "Running: $CMD"
echo ""

# Execute
eval $CMD

# Check exit status
EXIT_STATUS=$?

echo ""
if [ $EXIT_STATUS -eq 0 ]; then
    print_step "Pipeline completed successfully!"
    print_info "Check $OUTPUT_DIR for results"
    print_info "Check logs/pipeline for detailed logs"
else
    print_error "Pipeline failed with exit code: $EXIT_STATUS"
    exit $EXIT_STATUS
fi
