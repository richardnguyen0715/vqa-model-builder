#!/bin/bash

# Script to download and organize dataset from Kaggle
# Usage: ./src/cli/download_data.sh

set -e  # Exit on error

echo "=========================================="
echo "Starting Data Download Pipeline"
echo "=========================================="

# Get the project root directory (2 levels up from this script)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

echo "Project root: $PROJECT_ROOT"
cd "$PROJECT_ROOT"

# Check if Python is available
if ! command -v python &> /dev/null; then
    echo "ERROR: Python is not installed or not in PATH"
    exit 1
fi

# Check if required config files exist
if [ ! -f "configs/data_configs.yaml" ]; then
    echo "ERROR: configs/data_configs.yaml not found"
    exit 1
fi

if [ ! -f "configs/logging_configs.yaml" ]; then
    echo "ERROR: configs/logging_configs.yaml not found"
    exit 1
fi

# Run the download script
echo ""
echo "Running download_data.py..."
echo ""

python -m src.data.download_data

EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "Data Download Pipeline Completed Successfully"
    echo "=========================================="
    echo ""
    echo "Check logs in: logs/data_loader/"
    echo ""
else
    echo ""
    echo "=========================================="
    echo "ERROR: Data Download Pipeline Failed"
    echo "Exit code: $EXIT_CODE"
    echo "=========================================="
    echo ""
    echo "Check logs in: logs/data_loader/"
    echo ""
    exit $EXIT_CODE
fi
