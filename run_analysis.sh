#!/bin/bash
# Run multimodal nested CV analysis
# Usage: bash run_analysis.sh [N_PARALLEL]

set -euo pipefail

N_PARALLEL=${1:-10}

echo "=========================================="
echo "Multimodal Nested CV Analysis"
echo "Started: $(date)"
echo "Parallel workers: ${N_PARALLEL}"
echo "=========================================="

# Activate conda environment
# source ~/anaconda3/etc/profile.d/conda.sh
# conda activate multimodal

# Run main analysis
N_PARALLEL=${N_PARALLEL} python main_nested_cv.py

echo ""
echo "=========================================="
echo "Single-gene model comparison"
echo "=========================================="

# Run single-gene comparison
python single_gene.py

echo ""
echo "=========================================="
echo "Analysis complete!"
echo "Finished: $(date)"
echo "=========================================="
