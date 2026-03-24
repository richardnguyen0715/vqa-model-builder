#!/bin/bash

# Script to download Hugging Face models
# Supports downloading any model from Hugging Face Hub
# Default: RichardNguyen/ViMoE-VQA
# Usage:
#   ./src/cli/download_model.sh                                (downloads default model)
#   ./src/cli/download_model.sh --model-id MODEL_ID           (downloads specific model)
#   ./src/cli/download_model.sh --help                        (show help information)

set -e  # Exit on error

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default values
MODEL_ID="RichardNguyen/ViMoE-VQA"
OUTPUT_DIR="checkpoints"
CACHE_DIR=""
FORCE_DOWNLOAD=""
VERBOSE=""

# Help function
show_help() {
    cat << EOF
================================================================================
                    Hugging Face Model Downloader
================================================================================

Download pre-trained models from Hugging Face Hub to local directory.

USAGE:
    $(basename "$0") [OPTIONS]

OPTIONS:
    --model-id ID           Model ID from Hugging Face Hub
                           (default: RichardNguyen/ViMoE-VQA)
    --output-dir DIR        Output directory for models (default: checkpoints)
    --cache-dir DIR         Custom Hugging Face cache directory (optional)
    --force-download        Force download even if already cached
    --verbose               Enable verbose logging
    --info-only             Show model info without downloading
    --help                  Show this help message

EXAMPLES:
    # Download default model (RichardNguyen/ViMoE-VQA)
    ./src/cli/download_model.sh
    
    # Download specific model
    ./src/cli/download_model.sh --model-id meta-llama/Llama-2-7b
    
    # Download to custom directory
    ./src/cli/download_model.sh --output-dir ./models
    
    # Force re-download
    ./src/cli/download_model.sh --force-download
    
    # Show model info only
    ./src/cli/download_model.sh --info-only
    
    # Verbose mode with custom cache
    ./src/cli/download_model.sh --verbose --cache-dir /tmp/hf_cache

OUTPUT:
    Downloaded models are saved to: \$OUTPUT_DIR/\$MODEL_NAME/
    
    Example structure:
      checkpoints/
      └── RichardNguyen_ViMoE-VQA/
          ├── config.json
          ├── pytorch_model.bin (or model.safetensors)
          ├── tokenizer.json
          └── other_files...

ENVIRONMENT:
    HF_TOKEN: Set Hugging Face API token for private models
             export HF_TOKEN='your_token_here'

================================================================================
EOF
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --model-id)
            MODEL_ID="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --cache-dir)
            CACHE_DIR="$2"
            shift 2
            ;;
        --force-download)
            FORCE_DOWNLOAD="--force-download"
            shift
            ;;
        --verbose)
            VERBOSE="--verbose"
            shift
            ;;
        --info-only)
            INFO_ONLY="--info-only"
            shift
            ;;
        --help)
            show_help
            exit 0
            ;;
        *)
            echo -e "${RED}ERROR: Unknown option: $1${NC}"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Get the project root directory (2 levels up from this script)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

echo -e "${BLUE}=================================================================================${NC}"
echo -e "${BLUE}                    Hugging Face Model Downloader${NC}"
echo -e "${BLUE}=================================================================================${NC}"
echo ""
echo -e "${YELLOW}Configuration:${NC}"
echo "  Project root: $PROJECT_ROOT"
echo "  Model ID: $MODEL_ID"
echo "  Output directory: $OUTPUT_DIR"

if [ -n "$CACHE_DIR" ]; then
    echo "  Cache directory: $CACHE_DIR"
fi

if [ -n "$FORCE_DOWNLOAD" ]; then
    echo "  Mode: Force re-download"
fi

if [ -n "$VERBOSE" ]; then
    echo "  Logging: Verbose"
fi

if [ -n "$INFO_ONLY" ]; then
    echo "  Mode: Info only (no download)"
fi

echo ""

cd "$PROJECT_ROOT"

# Check if Python is available
if ! command -v python &> /dev/null; then
    echo -e "${RED}ERROR: Python is not installed or not in PATH${NC}"
    exit 1
fi

echo -e "${YELLOW}Checking Python environment...${NC}"
python --version

# Check if src module exists
if [ ! -f "src/__init__.py" ]; then
    echo -e "${YELLOW}Creating src/__init__.py${NC}"
    touch src/__init__.py
fi

if [ ! -f "src/data/__init__.py" ]; then
    echo -e "${YELLOW}Creating src/data/__init__.py${NC}"
    touch src/data/__init__.py
fi

echo ""
echo -e "${YELLOW}Checking dependencies...${NC}"

# Check if huggingface_hub is installed
python -c "import huggingface_hub" 2>/dev/null
if [ $? -ne 0 ]; then
    echo -e "${YELLOW}huggingface_hub not found. Installing...${NC}"
    python -m pip install huggingface_hub --quiet
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓${NC} huggingface_hub installed successfully"
    else
        echo -e "${RED}ERROR: Failed to install huggingface_hub${NC}"
        exit 1
    fi
else
    echo -e "${GREEN}✓${NC} huggingface_hub is available"
fi

echo ""
echo -e "${YELLOW}Executing download script...${NC}"
echo ""

# Build the command
CMD="python src/data/download_model.py"
CMD="$CMD --model-id '$MODEL_ID'"
CMD="$CMD --output-dir '$OUTPUT_DIR'"

if [ -n "$CACHE_DIR" ]; then
    CMD="$CMD --cache-dir '$CACHE_DIR'"
fi

if [ -n "$FORCE_DOWNLOAD" ]; then
    CMD="$CMD $FORCE_DOWNLOAD"
fi

if [ -n "$VERBOSE" ]; then
    CMD="$CMD $VERBOSE"
fi

if [ -n "$INFO_ONLY" ]; then
    CMD="$CMD $INFO_ONLY"
fi

# Execute the command
eval $CMD
EXIT_CODE=$?

echo ""

if [ $EXIT_CODE -eq 0 ]; then
    echo -e "${BLUE}=================================================================================${NC}"
    echo -e "${GREEN}Model Download Completed Successfully${NC}"
    echo -e "${BLUE}=================================================================================${NC}"
    echo ""
    
    # Extract model name from model ID
    MODEL_NAME=$(echo "$MODEL_ID" | sed 's/\//_/g')
    MODEL_PATH="$OUTPUT_DIR/$MODEL_NAME"
    
    echo -e "${GREEN}✓${NC} Model: $MODEL_ID"
    echo "  Location: $MODEL_PATH"
    echo ""
    
    if [ -d "$MODEL_PATH" ]; then
        echo "Checking model structure:"
        ls -lh "$MODEL_PATH" | grep -v "^total" | awk '{print "  " $9 " (" $5 ")"}'
    fi
    
    echo ""
    echo -e "${BLUE}=================================================================================${NC}"
    
    exit 0
else
    if [ $EXIT_CODE -eq 130 ]; then
        echo -e "${BLUE}=================================================================================${NC}"
        echo -e "${YELLOW}Download cancelled by user${NC}"
        echo -e "${BLUE}=================================================================================${NC}"
    else
        echo -e "${BLUE}=================================================================================${NC}"
        echo -e "${RED}ERROR: Model Download Failed${NC}"
        echo -e "${BLUE}Exit code: $EXIT_CODE${NC}"
        echo -e "${BLUE}=================================================================================${NC}"
        echo ""
        echo -e "${YELLOW}Troubleshooting:${NC}"
        echo "  1. Check your internet connection"
        echo "  2. Verify the model ID is correct"
        echo "  3. Check available disk space"
        echo "  4. Visit: https://huggingface.co/$MODEL_ID to verify model exists"
        echo ""
        
        if [ -z "$HF_TOKEN" ]; then
            echo "  5. If it's a private model, set HF_TOKEN environment variable:"
            echo "     export HF_TOKEN='your_huggingface_token'"
        fi
        
        echo ""
        echo "For more help, run with --verbose flag:"
        echo "  $0 --verbose"
        echo ""
    fi
    
    exit $EXIT_CODE
fi
