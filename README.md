# Beyond-POPE

A research project demonstrating that POPE — the most widely used VLM hallucination benchmark — is saturated, contains annotation errors, and only measures existence hallucination, masking true model failures on attribute and relation hallucination.

---

## Core Argument

| Problem | Evidence | Module |
|---|---|---|
| POPE is saturated | CV across 7B+ models ≤ 0.014 on popular/random splits | Module 1 |
| Annotation errors distort rankings | Spearman ρ (POPE → RePOPE) = 0.657 < 0.7 | Module 2 |
| Harder benchmarks restore discriminability | CV rises to 0.101 on DASH-B (7B+ models) | Module 3 |
| Existence hallucination is only the tip | Attribute/relation accuracy drops ~20–25% vs existence | Module 4 |

**Conclusion**: POPE is neither accurate (annotation noise), sensitive (saturated), nor complete (dimension coverage). Beyond-POPE proposes a more reliable evaluation framework: RePOPE + DASH-B + X-POPE + H_total.

---

## Models Evaluated

| Model | Parameters | Source |
|---|---|---|
| Qwen2-VL | 7B | Qwen/Qwen2-VL-7B-Instruct |
| InternVL2.5 | 8B | OpenGVLab/InternVL2_5-8B |
| LLaVA-OneVision | 7B | lmms-lab/llava-onevision-qwen2-7b-ov |
| Llama-3.2-Vision | 11B | meta-llama/Llama-3.2-11B-Vision-Instruct |
| PaliGemma2 | 3B* | google/paligemma2-3b-mix-448 |
| Phi-3.5-Vision | 4B | microsoft/Phi-3.5-vision-instruct |

> *PaliGemma2-3B is the smallest model in our set. Its ranking shifts are discussed separately in the analysis as they are partly attributable to model scale.

---

## Results

### Module 1 — POPE Adversarial (n=3,000)

| Model | Accuracy | Yes Rate | Latency (ms) |
|---|---|---|---|
| LLaVA-OV-7B | **87.8%** | 43.5% | 784 |
| Qwen2-VL-7B | 86.8% | 48.1% | 345 |
| Llama3.2V-11B | 83.6% | 56.8% | 846 |
| InternVL2-8B | 83.0% | 58.3% | 312 |
| Phi-3.5V-4B | 80.4% | 37.7% | 510 |
| PaliGemma2-3B | 71.1% | 31.8% | 201 |

CV (all 6 models) = 0.159. CV (7B+ models only) = 0.024, with pope_popular CV = 0.008 and pope_random CV = 0.014 — both below the saturation threshold (≤ 0.02). Bootstrap 95% CI confirms saturation for 7B+ models on popular and random splits.

### Module 2 — POPE → RePOPE Ranking Shift

| Model | POPE Acc | RePOPE Acc | Rank Change |
|---|---|---|---|
| LLaVA-OV-7B | 87.8% | 92.9% | #1 → #1 (→) |
| Qwen2-VL-7B | 86.8% | 91.1% | #2 → #2 (→) |
| PaliGemma2-3B | 71.1% | 88.1% | #6 → #3 (↑3) |
| Llama3.2V-11B | 83.6% | 86.0% | #3 → #4 (↓1) |
| InternVL2-8B | 83.0% | 85.7% | #4 → #5 (↓1) |
| Phi-3.5V-4B | 80.4% | 85.0% | #5 → #6 (↓1) |

**Spearman ρ (POPE → RePOPE) = 0.657.** McNemar test confirms annotation errors significantly distort results for 3/6 models (p < 0.05). PaliGemma2 jumps 3 positions after correction, indicating POPE's errors disproportionately penalise smaller models.

### Module 3 — DASH-B Discriminability Recovery (n=2,682)

| Model | POPE Acc | DASH-B Acc | Change |
|---|---|---|---|
| PaliGemma2-3B | 71.1% | **79.9%** | +8.8% |
| InternVL2-8B | 83.0% | 71.3% | −11.7% |
| Phi-3.5V-4B | 80.4% | 69.5% | −10.9% |
| Qwen2-VL-7B | 86.8% | 66.6% | −20.2% |
| LLaVA-OV-7B | 87.8% | 62.8% | −25.0% |
| Llama3.2V-11B | 83.6% | 53.8% | −29.8% |

CV on DASH-B (7B+ models) = 0.101, fully recovering from near-saturation on POPE. Rankings are substantially reshuffled (Spearman ρ = −0.77 between POPE and DASH-B).

### Module 4 — X-POPE Three-Dimension Evaluation

X-POPE is an original dataset constructed from COCO val2014 + Visual Genome, extending hallucination evaluation beyond existence to attribute and relation dimensions.

| Model | Existence | Attribute | Relation | **H_total** |
|---|---|---|---|---|
| LLaVA-OV-7B | 94.8% | 74.5% | — | — |
| InternVL2-8B | 94.9% | 71.9% | — | — |
| Llama3.2V-11B | 91.8% | 72.9% | — | — |
| Qwen2-VL-7B | 92.8% | 73.7% | — | — |
| PaliGemma2-3B | 88.9% | 70.6% | — | — |
| Phi-3.5V-4B | 86.8% | 67.1% | — | — |

> Relation column and H_total pending re-evaluation on updated X-POPE relation split (self-reference bug fixed, dataset regenerated).

All models drop ~20% from existence to attribute (Welch's t-test p < 0.001 for all models). Attribute → relation gap is significant for 3/6 models. Jaccard similarity of error sets across models is 0.52 on attribute, indicating systematic difficulty rather than model-specific weaknesses.

**X-POPE dataset**: 3,000 existence + 2,200 attribute + 1,579 relation = 6,779 questions total.

---

## Project Structure

```
beyond-pope/
├── src/
│   ├── models/              Unified inference wrappers for all 6 models
│   │   ├── base.py          Abstract base: prompt templates, answer parser, batching
│   │   ├── qwen2vl.py
│   │   ├── internvl2.py
│   │   ├── llava_ov.py
│   │   ├── llama32v.py
│   │   ├── paligemma2.py
│   │   ├── phi35v.py
│   │   ├── __init__.py      MODEL_REGISTRY + get_model()
│   │   └── run_inference.py CLI entry point
│   │
│   ├── dataset/             Dataset construction
│   │   ├── parse_vg.py      Parse Visual Genome annotations
│   │   ├── build_existence.py
│   │   ├── build_attribute.py
│   │   ├── build_relation.py
│   │   ├── build_xpope.py   Build all X-POPE splits in one call
│   │   └── __init__.py
│   │
│   ├── eval/                Evaluation metrics
│   │   ├── metrics.py       accuracy / F1 / yes_bias / CV / per_category
│   │   ├── h_total.py       H_total weighted harmonic mean + model ranking
│   │   ├── evaluator.py     Evaluator + batch_evaluate + saturation_report
│   │   └── __init__.py
│   │
│   ├── analysis/            Statistical analysis
│   │   ├── saturation_diag.py    CV saturation diagnostics + model group breakdown
│   │   ├── ranking_shift.py      Spearman rho across POPE → RePOPE → DASH-B
│   │   ├── bias_analysis.py      Yes-bias distribution per model and benchmark
│   │   ├── significance_tests.py McNemar + Welch t-test + Bootstrap CI
│   │   ├── error_analysis.py     FP/FN breakdown + confused classes + hard samples
│   │   └── __init__.py
│   │
│   └── viz/                 Visualizations
│       ├── radar.py         3-dimension radar chart (X-POPE)
│       ├── bump.py          Bump chart of ranking shifts
│       ├── bias_bar.py      Grouped bar chart of yes-bias
│       ├── run_viz.py       Generate all figures in one call
│       └── __init__.py
│
├── experiments/             Experiment runner scripts
│   ├── run_module1.sh       POPE saturation + yes-bias
│   ├── run_module2.sh       RePOPE ranking shift
│   ├── run_module3.sh       DASH-B discriminability recovery
│   └── run_module4.sh       X-POPE + H_total
│
├── data/
│   ├── raw/
│   │   ├── coco/val2014/          COCO val2014 images (~6GB)
│   │   ├── visual_genome/         VG attributes + relationships
│   │   └── benchmarks/
│   │       ├── pope/              pope_random/popular/adversarial.jsonl
│   │       ├── repope/            repope_random/popular/adversarial.jsonl
│   │       └── dashb/             dashb.jsonl + images/
│   └── processed/
│       ├── xpope_existence.jsonl
│       ├── xpope_attribute.jsonl
│       └── xpope_relation.jsonl
│
├── results/
│   └── predictions/         Per-model per-benchmark JSONL prediction files
│
├── reports/                 JSON analysis outputs
└── figures/                 Generated PNG charts
```

---

## Setup

### 1. Clone

```bash
git clone git@github.com:FlyGCB/beyond-pope.git
cd beyond-pope
```

### 2. Environment

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip

pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install transformers accelerate einops timm sentencepiece
pip install scipy numpy matplotlib Pillow tqdm datasets
```

### 3. Data

**COCO val2014 images** (~6GB):
```bash
mkdir -p data/raw/coco
wget -P data/raw/coco http://images.cocodataset.org/zips/val2014.zip
wget -P data/raw/coco http://images.cocodataset.org/annotations/annotations_trainval2014.zip
cd data/raw/coco && python -c "import zipfile,glob; [zipfile.ZipFile(f).extractall('.') for f in glob.glob('*.zip')]"
```

**POPE benchmark files**:
```bash
mkdir -p data/raw/benchmarks/pope
# Place pope_random.jsonl, pope_popular.jsonl, pope_adversarial.jsonl here
```

**RePOPE**:
```bash
mkdir -p data/raw/benchmarks/repope
wget -P data/raw/benchmarks/repope \
    https://raw.githubusercontent.com/YanNeu/RePOPE/main/annotations/coco_repope_random.json \
    https://raw.githubusercontent.com/YanNeu/RePOPE/main/annotations/coco_repope_popular.json \
    https://raw.githubusercontent.com/YanNeu/RePOPE/main/annotations/coco_repope_adversarial.json

python -c "
import json
from pathlib import Path
for split in ['random', 'popular', 'adversarial']:
    src = Path(f'data/raw/benchmarks/repope/coco_repope_{split}.json')
    dst = Path(f'data/raw/benchmarks/repope/repope_{split}.jsonl')
    with open(src) as f_in, open(dst, 'w') as f_out:
        for line in f_in:
            r = json.loads(line)
            r['question'] = r.pop('text')
            f_out.write(json.dumps(r) + '\n')
"
```

**DASH-B**:
```bash
pip install datasets
python -c "
from datasets import load_dataset
import json
from pathlib import Path

ds = load_dataset('YanNeu/DASH-B')['test']
img_root = Path('data/raw/benchmarks/dashb/images')
img_root.mkdir(parents=True, exist_ok=True)
output_path = Path('data/raw/benchmarks/dashb/dashb.jsonl')
records = []
for i, sample in enumerate(ds):
    img_path = img_root / Path(sample['image_path']).relative_to('images')
    img_path.parent.mkdir(parents=True, exist_ok=True)
    if not img_path.exists():
        sample['image'].save(img_path)
    records.append({
        'question_id': sample['question_id'],
        'image': str(Path(*Path(sample['image_path']).parts[-3:])),
        'question': sample['question'],
        'label': sample['answer'],
        'object': sample['object'],
    })
    if (i+1) % 500 == 0: print(f'{i+1}/2682')
with open(output_path, 'w') as f:
    [f.write(json.dumps(r) + '\n') for r in records]
print('Done')
"
```

**Visual Genome** (for X-POPE):
```bash
mkdir -p data/raw/visual_genome
cd data/raw/visual_genome
wget -O attributes.json.zip https://homes.cs.washington.edu/~ranjay/visualgenome/data/dataset/attributes.json.zip
wget -O relationships.json.zip https://homes.cs.washington.edu/~ranjay/visualgenome/data/dataset/relationships.json.zip
wget -O image_data.json.zip https://homes.cs.washington.edu/~ranjay/visualgenome/data/dataset/image_data.json.zip
python -c "import zipfile,glob; [zipfile.ZipFile(f).extractall('.') for f in glob.glob('*.zip')]"
cd ../../..

python -m src.dataset.build_xpope \
    --vg-dir data/raw/visual_genome \
    --coco-dir data/raw/coco \
    --output-dir data/processed
```

---

## Running Experiments

```bash
chmod +x experiments/run_module*.sh

bash experiments/run_module1.sh   # POPE saturation + yes-bias
bash experiments/run_module2.sh   # RePOPE ranking shift
bash experiments/run_module3.sh   # DASH-B discriminability recovery
bash experiments/run_module4.sh   # X-POPE + H_total
```

### Single model, single benchmark

```bash
python -m src.models.run_inference \
    --model qwen2vl_7b \
    --benchmark pope_adversarial \
    --image-dir data/raw/coco/val2014
```

### Generate all figures

```bash
python -m src.viz.run_viz \
    --results results/predictions \
    --out figures
```

### Statistical analysis

```bash
# Saturation diagnostics with model group breakdown
python -m src.analysis.saturation_diag \
    --results-dir results/predictions \
    --benchmarks pope_adversarial pope_popular pope_random dashb \
    --model-groups "7B+:qwen2vl_7b,internvl2_8b,llava_ov_7b,llama32v_11b" \
                   "all:qwen2vl_7b,internvl2_8b,llava_ov_7b,llama32v_11b,paligemma2_3b,phi35v_4b" \
    --output reports/module1_saturation.json

# Statistical significance tests
python -m src.analysis.significance_tests \
    --results-dir results/predictions \
    --output reports/significance.json

# Error analysis
python -m src.analysis.error_analysis \
    --results-dir results/predictions \
    --output reports/error_analysis.json
```

---

## Evaluation API

```python
from src.eval.evaluator import Evaluator, batch_evaluate
from src.eval.h_total import rank_models_by_h_total

# Single file
ev = Evaluator("results/predictions/qwen2vl_7b_pope_adversarial.jsonl")
print(ev.summary())

# Rank all models by H_total
reports = batch_evaluate("results/predictions/")
```

---

## Key Design Decisions

**Single prompt template across all models**: prevents prompt-engineering confounds — any performance difference reflects model capability, not prompt sensitivity.

**Conservative answer parser**: returns `unknown` rather than guessing when a model response is ambiguous. Unknown responses are excluded from all metrics and logged separately.

**H_total**: weighted harmonic mean of existence / attribute / relation accuracy. The harmonic mean penalises dimensional weakness heavily — a model that excels at existence but fails at relations scores lower than its POPE rank suggests. Missing dimensions are dropped and weights re-normalised automatically.

**CV as saturation metric**: coefficient of variation (std / mean) across model scores on the same benchmark. CV ≤ 0.02 → saturated (models indistinguishable). Analysis is performed separately for 7B+ models and all models, since including smaller models (3B/4B) inflates variance and masks saturation of larger models.

**X-POPE negative sampling**: three complementary strategies — reversed relations (Type A), same-image wrong pairs (Type B), and cross-image object substitution (Type C). All strategies filter out self-referential questions (subject = object class).

---

## Requirements

- Python 3.11+
- CUDA 12.1+ (tested on A100 80GB)
- ~80GB disk space for images + model weights
- ~40GB VRAM for full-precision inference

---

## License

MIT
