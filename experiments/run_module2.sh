#!/bin/bash
# =============================================================================
# Module 2: RePOPE Ranking Shift
#
# Re-evaluates all 6 models on RePOPE (corrected annotations) and measures
# how model rankings change from POPE → RePOPE using Spearman ρ.
#
# Prerequisite: Module 1 must have been run first (POPE results required).
#
# Outputs:
#   results/predictions/*_repope_*.jsonl
#   reports/module2_ranking.json
#   figures/bump.png
# =============================================================================

set -e
cd "$(dirname "$0")/.."

IMAGE_DIR="data/raw/coco/val2014"
RESULTS_DIR="results/predictions"
REPORTS_DIR="reports"

echo "============================================================"
echo "Module 2: RePOPE Ranking Shift"
echo "============================================================"

# Step 1: Run inference on all 3 RePOPE splits
echo "[1/2] Running inference on RePOPE splits..."
python -m src.models.run_inference \
    --model all \
    --benchmark repope_adversarial \
    --image-dir "$IMAGE_DIR"

python -m src.models.run_inference \
    --model all \
    --benchmark repope_popular \
    --image-dir "$IMAGE_DIR"

python -m src.models.run_inference \
    --model all \
    --benchmark repope_random \
    --image-dir "$IMAGE_DIR"

# Step 2: Ranking shift analysis (Spearman rho)
echo "[2/2] Computing ranking shift (Spearman ρ)..."
python -m src.analysis.ranking_shift \
    --results-dir "$RESULTS_DIR" \
    --benchmarks pope_adversarial repope_adversarial \
    --output "$REPORTS_DIR/module2_ranking.json"

echo ""
echo "Module 2 complete."
echo "  Ranking report → $REPORTS_DIR/module2_ranking.json"
