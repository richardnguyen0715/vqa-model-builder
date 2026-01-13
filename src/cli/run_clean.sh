#!/bin/bash

# =============================================================================
# Clean Pipeline Runner
# Runs pipeline with all warnings suppressed for clean output
# =============================================================================

# Suppress warnings
export PYTHONWARNINGS="ignore::FutureWarning,ignore::RuntimeWarning,ignore::DeprecationWarning"

# Run pipeline with all arguments passed through
python -m src.core.vqa_pipeline "$@"
