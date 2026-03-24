#!/bin/bash

# Script to download and organize VIVQA dataset from Kaggle
# Version 2: Enhanced version with support for both VQA and VIVQA datasets
# Usage: 
#   ./src/cli/download_data_ver2.sh              (downloads VIVQA by default)
#   ./src/cli/download_data_ver2.sh --dataset vqa (downloads VQA dataset)
#   ./src/cli/download_data_ver2.sh --help       (show help information)

set -e  # Exit on error

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default values
DATASET="vivqa"
OUTPUT_DIR="data/vivqa"
KAGGLE_DATASET="dngtrungngha/vivqa"

# Help function
show_help() {
    cat << EOF
================================================================================
                    Data Download Pipeline - Version 2
================================================================================

Download and organize datasets from Kaggle.

USAGE:
    $(basename "$0") [OPTIONS]

OPTIONS:
    --dataset DATASET       Dataset to download: 'vqa' or 'vivqa' (default: vivqa)
    --output-dir DIR        Output directory for VIVQA (default: data/vivqa)
    --kaggle-dataset ID     Kaggle dataset ID for VIVQA (default: dngtrungngha/vivqa)
    --help                  Show this help message

EXAMPLES:
    # Download VIVQA (default)
    ./src/cli/download_data_ver2.sh
    
    # Download VQA dataset
    ./src/cli/download_data_ver2.sh --dataset vqa
    
    # Download VIVQA to custom directory
    ./src/cli/download_data_ver2.sh --output-dir /path/to/vivqa
    
    # Download VIVQA from custom Kaggle dataset
    ./src/cli/download_data_ver2.sh --kaggle-dataset username/dataset_name

OUTPUT:
    VIVQA: data/vivqa/
        ├── images/       (merged images from train and test)
        ├── train.csv
        └── test.csv
    
    VQA: data/raw/
        ├── images/       (organized images)
        └── texts/        (CSV files)

================================================================================
EOF
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --dataset)
            DATASET="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --kaggle-dataset)
            KAGGLE_DATASET="$2"
            shift 2
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

# Validate dataset choice
if [[ "$DATASET" != "vqa" && "$DATASET" != "vivqa" ]]; then
    echo -e "${RED}ERROR: Invalid dataset choice: $DATASET${NC}"
    echo "Valid options are: vqa, vivqa"
    exit 1
fi

# Get the project root directory (2 levels up from this script)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

echo -e "${BLUE}=================================================================================${NC}"
echo -e "${BLUE}                    Data Download Pipeline - Version 2${NC}"
echo -e "${BLUE}=================================================================================${NC}"
echo ""
echo -e "${YELLOW}Configuration:${NC}"
echo "  Project root: $PROJECT_ROOT"
echo "  Dataset: $DATASET"
if [[ "$DATASET" == "vivqa" ]]; then
    echo "  Output directory: $OUTPUT_DIR"
    echo "  Kaggle dataset: $KAGGLE_DATASET"
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

# Check if required config files exist
echo -e "${YELLOW}Verifying configuration files...${NC}"

if [ ! -f "configs/data_configs.yaml" ]; then
    echo -e "${RED}ERROR: configs/data_configs.yaml not found${NC}"
    exit 1
fi
echo "  configs/data_configs.yaml: ${GREEN}OK${NC}"

if [ ! -f "configs/logging_configs.yaml" ]; then
    echo -e "${RED}ERROR: configs/logging_configs.yaml not found${NC}"
    exit 1
fi
echo "  configs/logging_configs.yaml: ${GREEN}OK${NC}"

echo ""

# Run the download script with appropriate arguments
echo -e "${YELLOW}Executing download script...${NC}"
echo ""

if [[ "$DATASET" == "vivqa" ]]; then
    python -m src.data.download_data \
        --dataset vivqa \
        --output-dir "$OUTPUT_DIR" \
        --kaggle-dataset "$KAGGLE_DATASET"
else
    python -m src.data.download_data \
        --dataset vqa
fi

EXIT_CODE=$?

echo ""
if [ $EXIT_CODE -eq 0 ]; then
    echo -e "${BLUE}=================================================================================${NC}"
    echo -e "${GREEN}Data Download Pipeline Completed Successfully${NC}"
    echo -e "${BLUE}=================================================================================${NC}"
    echo ""
    
    if [[ "$DATASET" == "vivqa" ]]; then
        echo -e "${GREEN}✓${NC} VIVQA dataset downloaded and organized"
        echo "  Location: $OUTPUT_DIR"
        echo "  Contents:"
        echo "    - images/   (merged from train + test sets)"
        echo "    - train.csv"
        echo "    - test.csv"
    else
        echo -e "${GREEN}✓${NC} VQA dataset downloaded and organized"
        echo "  Location: data/raw/"
        echo "  Contents:"
        echo "    - images/   (all images)"
        echo "    - texts/    (CSV files)"
    fi
    
    echo ""
    echo "Logs: logs/data_loader/"
    echo ""
    echo -e "${BLUE}=================================================================================${NC}"
    
    exit 0
else
    echo -e "${BLUE}=================================================================================${NC}"
    echo -e "${RED}ERROR: Data Download Pipeline Failed${NC}"
    echo -e "${BLUE}Exit code: $EXIT_CODE${NC}"
    echo -e "${BLUE}=================================================================================${NC}"
    echo ""
    echo "Please check logs in: logs/data_loader/"
    echo ""
    echo -e "${BLUE}=================================================================================${NC}"
    
    exit $EXIT_CODE
fi
