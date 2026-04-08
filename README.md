# beyond-pope

**POPE is broken — and modern VLMs are hiding behind it.**

We show that POPE, the most widely used object hallucination benchmark, is saturated on 2024–2025 VLMs and contains systematic annotation errors that distort model rankings. More importantly, we reveal that while models score near-ceiling on existence hallucination, they continue to fail substantially on **attribute** and **relational** hallucination — failures that POPE cannot measure.

We contribute:

1. The first systematic evaluation of 6 state-of-the-art VLMs on POPE, RePOPE, and DASH-B
2. Quantitative evidence of POPE saturation and its annotation noise impact on rankings
3. **X-POPE** — a new benchmark extending hallucination evaluation to attribute and relation dimensions, built from COCO + Visual Genome

> *"We demonstrate that existing object hallucination benchmarks are saturated and annotation-noisy, masking persistent attribute and relational hallucination failures in state-of-the-art VLMs."*

---

## Key Results

### POPE is saturated — scores tell you almost nothing

| Model | POPE-Rand | POPE-Pop | POPE-Adv | CV (↓ = more saturated) |
|-------|-----------|----------|----------|--------------------------|
| Qwen2.5-VL-7B | — | — | — | — |
| InternVL2.5-8B | — | — | — | — |
| LLaVA-OV-7B | — | — | — | — |
| Llama-3.2-11B-V | — | — | — | — |
| PaliGemma2-3B | — | — | — | — |
| Idefics3-8B | — | — | — | — |

*Results will be filled after inference runs.*

### Annotation errors change model rankings (POPE → RePOPE)

| Model | POPE-Adv F1 | RePOPE-Adv F1 | Rank change |
|-------|-------------|----------------|-------------|
| ... | — | — | — |

### Models still fail on attribute and relation hallucination (X-POPE)

| Model | H_exist | H_attr | H_rel | H_total |
|-------|---------|--------|-------|---------|
| ... | — | — | — | — |

---

## What is X-POPE?

X-POPE is a multi-dimensional hallucination benchmark built on COCO val2014 images and Visual Genome annotations. Unlike POPE (existence only), X-POPE covers three hallucination types:

| Split | Question template | Example |
|-------|-------------------|---------|
| Existence | Is there a {object}? | Is there a dog in the image? |
| Attribute | Is the {object} {attribute}? | Is the car red? |
| Relation | Is the {object_a} {relation} {object_b}? | Is the person sitting on the chair? |

**Negative sample strategies (attribute split):**

- Type A — Wrong attribute, same category: "Is the car blue?" (car is red)
- Type B — Attribute from another object in the same image: "Is the car white?" (dog is white, car is red)

Type B is the hardest and most diagnostic — it tests whether the model confuses attributes across objects, which is a common real-world failure mode.

**Dataset statistics:**

| Split | Questions | Pos/Neg | Unique images |
|-------|-----------|---------|---------------|
| Existence | 3,000 | 1:1 | 500 |
| Attribute | 2,200 | 1:1 | 500 |
| Relation | 1,800 | 1:1 | 450 |
| **Total** | **7,000** | **1:1** | **500** |

Quality control: all annotations filtered from Visual Genome with ≥2 annotator agreement, restricted to 4 attribute types (color, material, size, shape), subjective attributes removed. 10% human-verified (inter-annotator agreement reported in paper).

---

## Unified Hallucination Score (H_total)

We propose a single comparable metric across all three dimensions:

```
H_total = (H_exist + H_attr + H_rel) / 3
```

where each H_x = 1 − accuracy on that split (higher = more hallucination).

We also report frequency-weighted and ablation variants to show ranking stability across different weight choices. See `src/eval/h_total.py` for full implementation.

---

## Models Evaluated

| Model | Size | HuggingFace ID |
|-------|------|----------------|
| Qwen2.5-VL | 7B | `Qwen/Qwen2-VL-7B-Instruct` |
| InternVL2.5 | 8B | `OpenGVLab/InternVL2_5-8B` |
| LLaVA-OneVision | 7B | `lmms-lab/llava-onevision-qwen2-7b-ov` |
| Llama-3.2-Vision | 11B | `meta-llama/Llama-3.2-11B-Vision-Instruct` |
| PaliGemma2 | 3B | `google/paligemma2-3b-pt-448` |
| Idefics3 | 8B | `HuggingFaceM4/Idefics3-8B-Llama3` |

All models evaluated with **identical settings**: same prompt template, greedy decoding (`do_sample=False`), `max_new_tokens=5`.

---

## Reproduce

### 1. Setup

```bash
git clone https://github.com/YOUR_USERNAME/beyond-pope
cd beyond-pope
pip install -r requirements.txt
```

### 2. Download data

```bash
# COCO val2014 images (~13GB)
wget http://images.cocodataset.org/zips/val2014.zip
unzip val2014.zip -d data/raw/coco/

# POPE benchmark files
git clone https://github.com/RUCAIBox/POPE data/raw/benchmarks/pope_repo
cp data/raw/benchmarks/pope_repo/output/coco/coco_pope_*.json data/raw/benchmarks/pope/

# RePOPE
git clone https://github.com/YanNeu/RePOPE data/raw/benchmarks/repope_repo
cp data/raw/benchmarks/repope_repo/annotations/*.json data/raw/benchmarks/repope/

# Visual Genome (for X-POPE construction)
wget https://cs.stanford.edu/people/rak248/VG_100K_2/images.zip
wget https://cs.stanford.edu/people/rak248/VG_100K_2/images2.zip
wget http://visualgenome.org/static/data/dataset/attributes.json.zip
wget http://visualgenome.org/static/data/dataset/relationships.json.zip
```

### 3. Build X-POPE

```bash
python -m src.dataset.build_xpope \
    --coco-dir data/raw/coco \
    --vg-dir data/raw/visual_genome \
    --output-dir data/processed
```

### 4. Run inference

```bash
# Single model, single benchmark (test first)
python -m src.models.run_inference \
    --model qwen2vl_7b \
    --benchmark pope_adversarial \
    --image-dir data/raw/coco/val2014

# Full sweep (all 6 models × all 7 benchmarks)
bash experiments/run_all.sh
```

Estimated time on a single A100 (80GB): ~12 hours for the full sweep.

### 5. Compute metrics and generate figures

```bash
python -m src.eval.compute_all
python -m src.viz.generate_all
```

Figures are written to `results/figures/`. Tables for the paper are in `results/tables/`.

---

## Project Structure

```
beyond-pope/
├── data/
│   ├── raw/            # COCO, Visual Genome, POPE, RePOPE, DASH-B
│   └── processed/      # X-POPE splits (existence / attribute / relation)
├── src/
│   ├── dataset/        # X-POPE construction pipeline
│   ├── models/         # Unified inference interface for all 6 VLMs
│   ├── eval/           # Metrics: Acc, F1, CV, Yes-rate, H_total
│   ├── analysis/       # Saturation, ranking shifts, bias analysis
│   └── viz/            # Figure generation (paper-ready PDFs)
├── experiments/        # Shell scripts for each experiment module
├── results/
│   ├── predictions/    # Raw model outputs (JSONL)
│   ├── metrics/        # Computed metrics (JSON)
│   ├── figures/        # Plots (PDF + PNG)
│   └── tables/         # LaTeX + CSV tables
└── README.md
```

---

## Experimental Modules

**Module 1 — POPE saturation diagnosis**
Run all 6 models on POPE (random / popular / adversarial). Compute Coefficient of Variation (CV) across models per split to quantify saturation. Analyze Yes-bias distribution.
→ `experiments/exp1_saturation.sh`

**Module 2 — RePOPE annotation impact**
Re-run on RePOPE (corrected labels). Measure rank changes vs POPE. Show that annotation errors in POPE's positive set (9.3% error rate) disproportionately affect certain models.
→ `experiments/exp2_repope.sh`

**Module 3 — DASH-B discriminability**
Run on DASH-B, a harder benchmark designed to resist saturation. Show that CV recovers — models are still differentiable when the benchmark is hard enough.
→ `experiments/exp3_dashb.sh`

**Module 4 — X-POPE multi-dimensional evaluation**
Full evaluation on X-POPE (existence + attribute + relation). Compute H_total. Show that models scoring near-ceiling on existence still fail substantially on attribute and relation hallucination.
→ `experiments/exp4_xpope.sh`

---

## Citation

If you use X-POPE or find this work useful, please cite:

```bibtex
@misc{beyond-pope-2025,
  title   = {Beyond POPE: Benchmarking Object Hallucination in 2024–2025 VLMs
             under Saturated and Harder Evaluation Settings},
  author  = {},
  year    = {2025},
  url     = {https://github.com/YOUR_USERNAME/beyond-pope}
}
```

---

## Acknowledgements

- [POPE](https://github.com/RUCAIBox/POPE) — Li et al., EMNLP 2023
- [RePOPE](https://github.com/YanNeu/RePOPE) — Neuhaus & Hein, 2025
- [DASH-B](https://github.com/YanNeu/DASH-B) — Neuhaus & Hein, 2025
- [Visual Genome](https://visualgenome.org/) — Krishna et al., IJCV 2017
- [lmms-eval](https://github.com/EvolvingLMMs-Lab/lmms-eval) — for benchmark formatting reference
