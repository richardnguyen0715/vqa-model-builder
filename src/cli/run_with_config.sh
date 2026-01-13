#!/bin/bash

# =============================================================================
# Vietnamese VQA Pipeline - Run with Config File
# =============================================================================

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m'

# Default config path
CONFIG_FILE="${1:-configs/pipeline_config.yaml}"

echo -e "${BLUE}"
echo "╔══════════════════════════════════════════════════════════════════════════════╗"
echo "║                     Vietnamese VQA Pipeline                                  ║"
echo "║                    Running with Configuration File                           ║"
echo "╚══════════════════════════════════════════════════════════════════════════════╝"
echo -e "${NC}"

# Check config file exists
if [ ! -f "$CONFIG_FILE" ]; then
    echo -e "${RED}[ERROR]${NC} Configuration file not found: $CONFIG_FILE"
    echo "Usage: $0 [config_file]"
    echo "Default: configs/pipeline_config.yaml"
    exit 1
fi

echo -e "${GREEN}[INFO]${NC} Using configuration: $CONFIG_FILE"
echo ""

# Run pipeline with config
python -m src.core.vqa_pipeline --config "$CONFIG_FILE"

echo ""
echo -e "${GREEN}[DONE]${NC} Pipeline completed!"
