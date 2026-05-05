#!/bin/bash
# =============================================================================
# Module 3: DASH-B Discriminability Recovery
#
# Evaluates all 6 models on DASH-B (harder benchmark with cross-dataset
# negatives) and shows that CV recovers — benchmark difficulty restores
# the ability to distinguish between models.
#
# Prerequisite: DASH-B images must be in data/raw/benchmarks/dashb/images/
#               Run the following to prepare:
#
#   python -c "
#   from datasets import load_dataset
#   import json
#   from pathlib import Path
#   ds = load_dataset('YanNeu/DASH-B')['test']
#   # ... (see README for full conversion script)
#   "
#
# Outputs:
#   results/predictions/*_dashb.jsonl
#   reports/module3_saturation.json
#   reports/module3_ranking.json
# =============================================================================

set -e
cd "$(dirname "$0")/.."

IMAGE_DIR="data/raw/benchmarks/dashb/images"
RESULTS_DIR="results/predictions"
REPORTS_DIR="reports"

echo "============================================================"
echo "Module 3: DASH-B Discriminability Recovery"
echo "============================================================"

# Step 1: Run inference on DASH-B
echo "[1/3] Running inference on DASH-B..."
python -m src.models.run_inference \
    --model all \
    --benchmark dashb \
    --image-dir "$IMAGE_DIR"

# Step 2: Saturation diagnostics — compare CV on POPE vs DASH-B
echo "[2/3] Comparing saturation: POPE vs DASH-B..."
python -m src.analysis.saturation_diag \
    --results-dir "$RESULTS_DIR" \
    --benchmarks pope_adversarial dashb \
    --output "$REPORTS_DIR/module3_saturation.json"

# Step 3: Ranking shift POPE → DASH-B
echo "[3/3] Computing ranking shift (POPE → DASH-B)..."
python -m src.analysis.ranking_shift \
    --results-dir "$RESULTS_DIR" \
    --benchmarks pope_adversarial repope_adversarial dashb \
    --output "$REPORTS_DIR/module3_ranking.json"

echo ""
echo "Module 3 complete."
echo "  Saturation report → $REPORTS_DIR/module3_saturation.json"
echo "  Ranking report    → $REPORTS_DIR/module3_ranking.json"
