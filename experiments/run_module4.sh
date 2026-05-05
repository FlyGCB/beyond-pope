#!/bin/bash
# =============================================================================
# Module 4: X-POPE Three-Dimension Evaluation
#
# Builds the X-POPE dataset from Visual Genome + COCO, runs all 6 models
# on existence / attribute / relation splits, and computes H_total.
#
# Prerequisites:
#   data/raw/visual_genome/attributes.json
#   data/raw/visual_genome/relationships.json
#   data/raw/visual_genome/image_data.json
#   data/raw/coco/val2014/  (images)
#   data/raw/coco/annotations/instances_val2014.json
#
# Outputs:
#   data/processed/xpope_existence.jsonl
#   data/processed/xpope_attribute.jsonl
#   data/processed/xpope_relation.jsonl
#   results/predictions/*_xpope_*.jsonl
#   reports/module4_htotal.json
#   figures/radar.png
# =============================================================================

set -e
cd "$(dirname "$0")/.."

IMAGE_DIR="data/raw/coco/val2014"
RESULTS_DIR="results/predictions"
REPORTS_DIR="reports"

echo "============================================================"
echo "Module 4: X-POPE Three-Dimension Evaluation"
echo "============================================================"

# Step 1: Build X-POPE dataset
echo "[1/3] Building X-POPE dataset..."
python -m src.dataset.build_xpope \
    --vg-dir data/raw/visual_genome \
    --coco-dir data/raw/coco \
    --output-dir data/processed

# Step 2: Run inference on all 3 X-POPE splits
echo "[2/3] Running inference on X-POPE splits..."
python -m src.models.run_inference \
    --model all \
    --benchmark xpope_existence \
    --image-dir "$IMAGE_DIR"

python -m src.models.run_inference \
    --model all \
    --benchmark xpope_attribute \
    --image-dir "$IMAGE_DIR"

python -m src.models.run_inference \
    --model all \
    --benchmark xpope_relation \
    --image-dir "$IMAGE_DIR"

# Step 3: Compute H_total and generate radar chart
echo "[3/3] Computing H_total and generating figures..."
python -c "
import json, sys
sys.path.insert(0, 'src')
from pathlib import Path
from eval.h_total import rank_models_by_h_total

models = ['qwen2vl_7b','internvl2_8b','llava_ov_7b',
          'llama32v_11b','paligemma2_3b','phi35v_4b']
results = {}
for m in models:
    def get_acc(bm, m=m):
        p = Path(f'results/predictions/{m}_{bm}_summary.json')
        return json.load(open(p))['accuracy'] if p.exists() else None
    results[m] = {
        'existence_f1': get_acc('xpope_existence'),
        'attribute_f1': get_acc('xpope_attribute'),
        'relation_f1':  get_acc('xpope_relation'),
    }

ranked = rank_models_by_h_total(results)
output = {'rankings': ranked, 'model_scores': results}
Path('$REPORTS_DIR').mkdir(parents=True, exist_ok=True)
json.dump(output, open('$REPORTS_DIR/module4_htotal.json', 'w'), indent=2)

print('H_total Rankings:')
for r in ranked:
    print(f\"  #{r['rank']} {r['model']:<28} H={r['h_total']:.4f}\")
"

python -m src.viz.run_viz \
    --results "$RESULTS_DIR" \
    --out figures

echo ""
echo "Module 4 complete."
echo "  H_total report → $REPORTS_DIR/module4_htotal.json"
echo "  Radar chart    → figures/radar.png"
