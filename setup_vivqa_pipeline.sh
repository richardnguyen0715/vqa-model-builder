#!/bin/bash

##############################################################################
# VIVQA Model Pipeline - Complete Setup Script
# 
# This script automates the entire setup and execution pipeline for
# ViMoE-VQA model evaluation on VIVQA dataset.
#
# Workflow:
# 1. Clone GitHub repository
# 2. Setup Python environment (poetry)
# 3. Download VIVQA dataset from Kaggle
# 4. Download pre-trained model from Hugging Face
# 5. Run model evaluation on VIVQA test set
#
# Usage:
#   bash setup_vivqa_pipeline.sh
#   bash setup_vivqa_pipeline.sh --skip-clone
#   bash setup_vivqa_pipeline.sh --help
#
# Author: VIVQA Team
# Date: 2026-03-24
##############################################################################

set -e  # Exit on any error

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Configuration
REPO_URL="https://github.com/richardnguyen0715/vqa-model-builder.git"
REPO_NAME="vqa-model-builder"
PROJECT_DIR=""
VENV_DIR=".venv"
PYTHON_VERSION="3.11"

# Flags
SKIP_CLONE=false
SKIP_ENV="false"
SKIP_DATA=false
SKIP_MODEL=false
SKIP_EVAL=false
VERBOSE=false

##############################################################################
# UTILITY FUNCTIONS
##############################################################################

print_header() {
    echo -e "${BLUE}═══════════════════════════════════════════════════════════════════════${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}═══════════════════════════════════════════════════════════════════════${NC}"
    echo ""
}

print_stage() {
    echo -e "${CYAN}▶▶▶ $1${NC}"
    echo ""
}

print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ ERROR: $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠ WARNING: $1${NC}"
}

print_info() {
    echo -e "${CYAN}ℹ $1${NC}"
}

show_help() {
    cat << EOF
${BLUE}════════════════════════════════════════════════════════════════════════${NC}
                    VIVQA Pipeline Setup Script
${BLUE}════════════════════════════════════════════════════════════════════════${NC}

Complete automation for ViMoE-VQA model setup and evaluation on VIVQA dataset.

USAGE:
    bash setup_vivqa_pipeline.sh [OPTIONS]

OPTIONS:
    --skip-clone        Skip repository cloning (use existing folder)
    --skip-env          Skip environment setup (poetry install)
    --skip-data         Skip VIVQA data download
    --skip-model        Skip model download from Hugging Face
    --skip-eval         Skip evaluation step
    --verbose           Enable verbose output
    --help              Show this help message

EXAMPLES:
    # Full setup from scratch
    bash setup_vivqa_pipeline.sh
    
    # Skip cloning if already cloned
    bash setup_vivqa_pipeline.sh --skip-clone
    
    # Only download data and model (skip environment setup)
    bash setup_vivqa_pipeline.sh --skip-clone --skip-env
    
    # Full setup with verbose logging
    bash setup_vivqa_pipeline.sh --verbose

WORKFLOW:
    1. Clone repository from GitHub
    2. Setup Python environment with Poetry
    3. Download VIVQA dataset from Kaggle
    4. Download ViMoE-VQA model from Hugging Face
    5. Run model evaluation on VIVQA test set

REQUIREMENTS:
    - git (for cloning repository)
    - Python 3.11+ (for project runtime)
    - pip (for installing Poetry)
    - Kaggle credentials (for dataset download)
        Set environment variables: KAGGLE_USERNAME, KAGGLE_KEY
        Or configure: ~/.kaggle/kaggle.json
    - Hugging Face credentials (for private models, if needed)
        Set environment variable: HF_TOKEN

OUTPUT:
    - logs/                          # Log files
    - data/vivqa/                    # Downloaded dataset
    - checkpoints/                   # Downloaded models
    - outputs/evaluation/            # Evaluation results

TROUBLESHOOTING:
    - If poetry install fails, try: pip install --upgrade poetry
    - If Kaggle fails, check: https://github.com/Kaggle/kaggle-api
    - If Hugging Face fails, check: https://huggingface.co/account/tokens

${BLUE}════════════════════════════════════════════════════════════════════════${NC}
EOF
}

##############################################################################
# DEPENDENCY CHECKING
##############################################################################

check_dependencies() {
    print_stage "Checking system dependencies"
    
    local missing_deps=()
    
    # Check git
    if ! command -v git &> /dev/null; then
        missing_deps+=("git")
        print_error "git is not installed"
    else
        GIT_VERSION=$(git --version | cut -d' ' -f3)
        print_success "git found (version $GIT_VERSION)"
    fi
    
    # Check Python
    if ! command -v python3 &> /dev/null; then
        missing_deps+=("python3")
        print_error "Python3 is not installed"
    else
        PYTHON_VERSION_FULL=$(python3 --version 2>&1 | cut -d' ' -f2)
        print_success "Python3 found (version $PYTHON_VERSION_FULL)"
    fi
    
    # Check pip
    if ! command -v pip &> /dev/null && ! command -v pip3 &> /dev/null; then
        missing_deps+=("pip")
        print_error "pip is not installed"
    else
        print_success "pip is available"
    fi
    
    # Check curl or wget (for downloads)
    if ! command -v curl &> /dev/null && ! command -v wget &> /dev/null; then
        missing_deps+=("curl or wget")
        print_error "Neither curl nor wget is installed"
    else
        print_success "Download utility found (curl/wget)"
    fi
    
    if [ ${#missing_deps[@]} -gt 0 ]; then
        echo ""
        print_error "Missing dependencies: ${missing_deps[*]}"
        print_info "Please install missing dependencies and try again"
        return 1
    fi
    
    echo ""
    print_success "All dependencies are available"
    return 0
}

##############################################################################
# STEP 1: CLONE REPOSITORY
##############################################################################

clone_repository() {
    print_stage "Step 1: Cloning GitHub repository"
    
    if [ "$SKIP_CLONE" = true ]; then
        print_warning "Skipping repository clone"
        if [ -d "$REPO_NAME" ]; then
            PROJECT_DIR="$REPO_NAME"
            print_success "Using existing directory: $PROJECT_DIR"
        else
            print_error "Directory $REPO_NAME not found and --skip-clone was used"
            return 1
        fi
    else
        if [ -d "$REPO_NAME" ]; then
            print_warning "Directory $REPO_NAME already exists"
            read -p "Do you want to remove and re-clone? (y/n) " -n 1 -r
            echo
            if [[ $REPLY =~ ^[Yy]$ ]]; then
                print_info "Removing existing directory..."
                rm -rf "$REPO_NAME"
            else
                print_info "Using existing directory"
                PROJECT_DIR="$REPO_NAME"
                return 0
            fi
        fi
        
        print_info "Cloning from $REPO_URL"
        if git clone "$REPO_URL"; then
            PROJECT_DIR="$REPO_NAME"
            print_success "Repository cloned successfully"
        else
            print_error "Failed to clone repository"
            return 1
        fi
    fi
    
    echo ""
    return 0
}

##############################################################################
# STEP 2: SETUP PYTHON ENVIRONMENT
##############################################################################

setup_environment() {
    print_stage "Step 2: Setting up Python environment with Poetry"
    
    if [ "$SKIP_ENV" = true ]; then
        print_warning "Skipping environment setup"
        echo ""
        return 0
    fi
    
    cd "$PROJECT_DIR"
    
    # Check if .venv already exists
    if [ -d "$VENV_DIR" ]; then
        print_warning "Virtual environment already exists at $VENV_DIR"
        read -p "Do you want to recreate it? (y/n) " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            print_success "Using existing virtual environment"
            cd ..
            echo ""
            return 0
        fi
        print_info "Removing existing environment..."
        rm -rf "$VENV_DIR"
    fi
    
    # Create virtual environment
    print_info "Creating Python virtual environment..."
    if python3 -m venv "$VENV_DIR"; then
        print_success "Virtual environment created"
    else
        print_error "Failed to create virtual environment"
        cd ..
        return 1
    fi
    
    # Activate virtual environment
    print_info "Activating virtual environment..."
    source "$VENV_DIR/bin/activate"
    
    # Upgrade pip
    print_info "Upgrading pip..."
    pip install --upgrade pip setuptools wheel --quiet
    
    # Install Poetry
    print_info "Installing Poetry..."
    if pip install poetry --quiet; then
        POETRY_VERSION=$(poetry --version 2>&1 | cut -d' ' -f3)
        print_success "Poetry installed (version $POETRY_VERSION)"
    else
        print_error "Failed to install Poetry"
        deactivate
        cd ..
        return 1
    fi
    
    # Install project dependencies
    print_info "Installing project dependencies with Poetry..."
    print_info "This may take several minutes..."
    if poetry install; then
        print_success "Project dependencies installed"
    else
        print_error "Failed to install project dependencies"
        deactivate
        cd ..
        return 1
    fi
    
    print_success "Environment setup completed"
    cd ..
    echo ""
    return 0
}

##############################################################################
# STEP 3: DOWNLOAD VIVQA DATA
##############################################################################

download_vivqa_data() {
    print_stage "Step 3: Downloading VIVQA dataset from Kaggle"
    
    if [ "$SKIP_DATA" = true ]; then
        print_warning "Skipping data download"
        echo ""
        return 0
    fi
    
    cd "$PROJECT_DIR"
    
    # Check if data already exists
    if [ -f "data/vivqa/train.csv" ] && [ -f "data/vivqa/test.csv" ] && [ -d "data/vivqa/images" ]; then
        TRAIN_COUNT=$(ls data/vivqa/images/*.jpg 2>/dev/null | wc -l)
        if [ $TRAIN_COUNT -gt 1000 ]; then
            print_warning "VIVQA data already exists (found ~$TRAIN_COUNT images)"
            read -p "Do you want to re-download? (y/n) " -n 1 -r
            echo
            if [[ ! $REPLY =~ ^[Yy]$ ]]; then
                print_success "Using existing VIVQA data"
                cd ..
                echo ""
                return 0
            fi
        fi
    fi
    
    # Activate virtual environment
    if [ -f "$VENV_DIR/bin/activate" ]; then
        source "$VENV_DIR/bin/activate"
    fi
    
    print_info "Starting VIVQA dataset download..."
    print_info "This requires Kaggle API credentials configured"
    print_info "Configuration: ~/.kaggle/kaggle.json or KAGGLE_USERNAME/KAGGLE_KEY"
    echo ""
    
    if bash ./src/cli/download_data_ver2.sh --dataset vivqa; then
        print_success "VIVQA data downloaded successfully"
    else
        print_error "Failed to download VIVQA data"
        print_info "Check your Kaggle credentials"
        print_info "Visit: https://www.kaggle.com/settings/account"
        cd ..
        return 1
    fi
    
    cd ..
    echo ""
    return 0
}

##############################################################################
# STEP 4: DOWNLOAD MODEL
##############################################################################

download_model() {
    print_stage "Step 4: Downloading ViMoE-VQA model from Hugging Face"
    
    if [ "$SKIP_MODEL" = true ]; then
        print_warning "Skipping model download"
        echo ""
        return 0
    fi
    
    cd "$PROJECT_DIR"
    
    # Check if model already exists
    if [ -f "checkpoints/RichardNguyen_ViMoE-VQA/config.json" ]; then
        print_warning "Model already exists at checkpoints/RichardNguyen_ViMoE-VQA/"
        read -p "Do you want to re-download? (y/n) " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            print_success "Using existing model"
            cd ..
            echo ""
            return 0
        fi
        print_info "Removing existing model..."
        rm -rf "checkpoints/RichardNguyen_ViMoE-VQA"
    fi
    
    # Activate virtual environment
    if [ -f "$VENV_DIR/bin/activate" ]; then
        source "$VENV_DIR/bin/activate"
    fi
    
    print_info "Starting model download from Hugging Face Hub..."
    print_info "Model: RichardNguyen/ViMoE-VQA"
    print_info "Size: ~2-3 GB (depending on format)"
    echo ""
    
    if bash ./src/cli/download_model.sh; then
        print_success "Model downloaded successfully"
    else
        print_error "Failed to download model"
        print_info "Check your internet connection and disk space"
        print_info "If it's a private model, ensure HF_TOKEN is set"
        cd ..
        return 1
    fi
    
    cd ..
    echo ""
    return 0
}

##############################################################################
# STEP 5: RUN EVALUATION
##############################################################################

run_evaluation() {
    print_stage "Step 5: Running model evaluation on VIVQA dataset"
    
    if [ "$SKIP_EVAL" = true ]; then
        print_warning "Skipping evaluation"
        echo ""
        return 0
    fi
    
    cd "$PROJECT_DIR"
    
    # Check if required files exist
    if [ ! -f "checkpoints/RichardNguyen_ViMoE-VQA/pytorch_model.bin" ] && \
       [ ! -f "checkpoints/RichardNguyen_ViMoE-VQA/model.safetensors" ]; then
        print_error "Model checkpoint not found"
        print_info "Please download the model first"
        cd ..
        return 1
    fi
    
    if [ ! -f "data/vivqa/test.csv" ]; then
        print_error "VIVQA test data not found"
        print_info "Please download the dataset first"
        cd ..
        return 1
    fi
    
    # Activate virtual environment
    if [ -f "$VENV_DIR/bin/activate" ]; then
        source "$VENV_DIR/bin/activate"
    fi
    
    print_info "Starting evaluation pipeline..."
    print_info "Output directory: outputs/evaluation/"
    print_info "This may take 10-30 minutes depending on hardware"
    echo ""
    
    # Run evaluation CLI
    if python src/core/vivqa_eval_cli.py \
        --checkpoint checkpoints/RichardNguyen_ViMoE-VQA/pytorch_model.bin \
        --csv-path data/vivqa/test.csv \
        --images-dir data/vivqa/images \
        --batch-size 32 \
        --device auto \
        --output-dir outputs/evaluation \
        --save-metrics \
        --save-predictions; then
        print_success "Evaluation completed successfully"
    else
        print_warning "Evaluation encountered an issue"
        print_info "Check logs in: logs/data_process/"
        print_info "Partial results may be available in: outputs/evaluation/"
    fi
    
    cd ..
    echo ""
    return 0
}

##############################################################################
# SUMMARY AND REPORT
##############################################################################

print_summary() {
    print_stage "Setup and Execution Summary"
    
    cd "$PROJECT_DIR"
    
    echo -e "${BLUE}Directory Structure:${NC}"
    echo ""
    
    # Check data
    if [ -d "data/vivqa" ]; then
        IMG_COUNT=$(ls data/vivqa/images/*.jpg 2>/dev/null | wc -l)
        echo -e "${GREEN}✓ VIVQA Dataset${NC}"
        echo "    Location: data/vivqa/"
        echo "    Images: ~$IMG_COUNT files"
        [ -f "data/vivqa/train.csv" ] && echo "    ✓ train.csv"
        [ -f "data/vivqa/test.csv" ] && echo "    ✓ test.csv"
    else
        echo -e "${YELLOW}○ VIVQA Dataset${NC} (not found)"
    fi
    
    echo ""
    
    # Check model
    if [ -f "checkpoints/RichardNguyen_ViMoE-VQA/config.json" ]; then
        SIZE=$(du -sh "checkpoints/RichardNguyen_ViMoE-VQA" 2>/dev/null | cut -f1)
        echo -e "${GREEN}✓ ViMoE-VQA Model${NC}"
        echo "    Location: checkpoints/RichardNguyen_ViMoE-VQA/"
        echo "    Size: $SIZE"
        [ -f "checkpoints/RichardNguyen_ViMoE-VQA/pytorch_model.bin" ] && echo "    ✓ pytorch_model.bin"
        [ -f "checkpoints/RichardNguyen_ViMoE-VQA/model.safetensors" ] && echo "    ✓ model.safetensors"
    else
        echo -e "${YELLOW}○ ViMoE-VQA Model${NC} (not found)"
    fi
    
    echo ""
    
    # Check evaluation results
    if [ -d "outputs/evaluation" ]; then
        METRICS=$(ls outputs/evaluation/metrics_*.json 2>/dev/null | wc -l)
        PREDS=$(ls outputs/evaluation/predictions_*.json 2>/dev/null | wc -l)
        echo -e "${GREEN}✓ Evaluation Results${NC}"
        echo "    Location: outputs/evaluation/"
        [ $METRICS -gt 0 ] && echo "    ✓ $METRICS metrics file(s)"
        [ $PREDS -gt 0 ] && echo "    ✓ $PREDS predictions file(s)"
    else
        echo -e "${YELLOW}○ Evaluation Results${NC} (not generated yet)"
    fi
    
    echo ""
    echo -e "${BLUE}Important Directories:${NC}"
    echo "    logs/               - Application logs"
    echo "    configs/            - Configuration files"
    echo "    src/                - Source code"
    echo "    docs/               - Documentation"
    
    echo ""
    
    cd ..
}

print_next_steps() {
    echo -e "${BLUE}════════════════════════════════════════════════════════════════════════${NC}"
    echo -e "${CYAN}Next Steps:${NC}"
    echo ""
    echo "1. Review evaluation results:"
    echo "   cat $PROJECT_DIR/outputs/evaluation/metrics_*.json"
    echo ""
    echo "2. Check detailed logs:"
    echo "   tail -f $PROJECT_DIR/logs/data_process/vivqa_evaluation.log"
    echo ""
    echo "3. Analyze predictions:"
    echo "   python -m json.tool $PROJECT_DIR/outputs/evaluation/predictions_*.json | less"
    echo ""
    echo "4. Re-run evaluation with different parameters:"
    echo "   cd $PROJECT_DIR"
    echo "   source $VENV_DIR/bin/activate"
    echo "   python src/core/vivqa_eval_cli.py --help"
    echo ""
    echo "5. For more information, see documentation:"
    echo "   cat $PROJECT_DIR/docs/vivqa_usage_guide.md"
    echo ""
    echo -e "${BLUE}════════════════════════════════════════════════════════════════════════${NC}"
}

##############################################################################
# MAIN SCRIPT LOGIC
##############################################################################

main() {
    print_header "VIVQA Model Pipeline - Complete Setup"
    
    # Parse command line arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            --skip-clone)
                SKIP_CLONE=true
                shift
                ;;
            --skip-env)
                SKIP_ENV=true
                shift
                ;;
            --skip-data)
                SKIP_DATA=true
                shift
                ;;
            --skip-model)
                SKIP_MODEL=true
                shift
                ;;
            --skip-eval)
                SKIP_EVAL=true
                shift
                ;;
            --verbose)
                VERBOSE=true
                set -x
                shift
                ;;
            --help)
                show_help
                exit 0
                ;;
            *)
                print_error "Unknown option: $1"
                show_help
                exit 1
                ;;
        esac
    done
    
    echo ""
    
    # Execute pipeline stages
    check_dependencies || exit 1
    
    clone_repository || exit 1
    
    setup_environment || exit 1
    
    download_vivqa_data || { print_warning "Data download failed, continuing..."; }
    
    download_model || { print_warning "Model download failed, continuing..."; }
    
    run_evaluation || { print_warning "Evaluation failed, continuing..."; }
    
    # Print summary and next steps
    print_summary
    
    print_header "Setup Completed!"
    print_next_steps
    
    echo -e "${GREEN}Pipeline execution finished!${NC}"
    echo ""
}

# Run main function
main "$@"
