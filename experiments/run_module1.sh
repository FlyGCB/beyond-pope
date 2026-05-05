#!/bin/bash
# =============================================================================
# Module 1: POPE Saturation + Yes-Bias Analysis
#
# Runs all 6 models on all 3 POPE splits (random / popular / adversarial),
# then computes CV-based saturation diagnostics and yes-bias distribution.
#
# Outputs:
#   results/predictions/*_pope_*.jsonl
#   reports/module1_saturation.json
#   reports/module1_bias.json
#   figures/bias_bar.png
# =============================================================================

set -e
cd "$(dirname "$0")/.."

IMAGE_DIR="data/raw/coco/val2014"
RESULTS_DIR="results/predictions"
REPORTS_DIR="reports"

echo "============================================================"
echo "Module 1: POPE Saturation + Yes-Bias"
echo "============================================================"

# Step 1: Run inference on all 3 POPE splits
echo "[1/3] Running inference on POPE splits..."
python -m src.models.run_inference \
    --model all \
    --benchmark pope_adversarial \
    --image-dir "$IMAGE_DIR"

python -m src.models.run_inference \
    --model all \
    --benchmark pope_popular \
    --image-dir "$IMAGE_DIR"

python -m src.models.run_inference \
    --model all \
    --benchmark pope_random \
    --image-dir "$IMAGE_DIR"

# Step 2: Saturation diagnostics (CV)
echo "[2/3] Computing saturation diagnostics..."
python -m src.analysis.saturation_diag \
    --results-dir "$RESULTS_DIR" \
    --benchmarks pope_adversarial pope_popular pope_random \
    --output "$REPORTS_DIR/module1_saturation.json"

# Step 3: Yes-bias analysis
echo "[3/3] Computing yes-bias distribution..."
python -m src.analysis.bias_analysis \
    --results-dir "$RESULTS_DIR" \
    --benchmarks pope_adversarial pope_popular pope_random \
    --output "$REPORTS_DIR/module1_bias.json"

echo ""
echo "Module 1 complete."
echo "  Saturation report → $REPORTS_DIR/module1_saturation.json"
echo "  Bias report       → $REPORTS_DIR/module1_bias.json"
