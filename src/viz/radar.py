"""
radar.py — Radar chart: 6 models × 3 hallucination dimensions.

Dimensions
----------
  Existence  : mean accuracy over xpope_existence
  Attribute  : mean accuracy over xpope_attribute
  Relation   : mean accuracy over xpope_relation

Data source: results/predictions/*_xpope_*.json
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

MODEL_LABELS = {
    "internvl2_8b":  "InternVL2-8B",
    "llama32v_11b":  "Llama3.2V-11B",
    "llava_ov_7b":   "LLaVA-OV-7B",
    "paligemma2_3b": "PaliGemma2-3B",
    "phi35v_4b":     "Phi-3.5V-4B",
    "qwen2vl_7b":    "Qwen2-VL-7B",
}

MODEL_COLORS = {
    "internvl2_8b":  "#636EFA",
    "llama32v_11b":  "#EF553B",
    "llava_ov_7b":   "#00CC96",
    "paligemma2_3b": "#AB63FA",
    "phi35v_4b":     "#FFA15A",
    "qwen2vl_7b":    "#19D3F3",
}

DIMS = ["Existence", "Attribute", "Relation"]
DIM_BENCHMARKS = {
    "Existence": "xpope_existence",
    "Attribute": "xpope_attribute",
    "Relation":  "xpope_relation",
}


def _load_scores(results_dir: Path) -> dict[str, dict[str, float]]:
    scores = {}
    for model in MODEL_LABELS:
        model_scores = {}
        for dim, bm in DIM_BENCHMARKS.items():
            p = results_dir / f"{model}_{bm}_summary.json"
            if p.exists():
                with open(p) as f:
                    model_scores[dim] = json.load(f)["accuracy"]
        if model_scores:
            scores[model] = model_scores
    return scores


def plot_radar(results_dir: Path, out: Path) -> None:
    scores = _load_scores(results_dir)

    N = len(DIMS)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]  # close

    fig, ax = plt.subplots(figsize=(7, 6), subplot_kw=dict(polar=True))

    for model, model_scores in scores.items():
        vals = [model_scores.get(d, 0.0) for d in DIMS]
        vals += vals[:1]
        color = MODEL_COLORS[model]
        ax.plot(angles, vals, color=color, linewidth=2.2, label=MODEL_LABELS[model])
        ax.fill(angles, vals, color=color, alpha=0.08)
        # markers
        ax.scatter(angles, vals, color=color, s=60, zorder=5)

    # Axis config
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(DIMS, fontsize=13, fontweight="bold")
    ax.set_ylim(0.5, 1.0)
    ax.set_yticks([0.6, 0.7, 0.8, 0.9, 1.0])
    ax.set_yticklabels(["0.60", "0.70", "0.80", "0.90", "1.00"], fontsize=8, color="gray")
    ax.yaxis.grid(True, color="gray", linestyle="--", alpha=0.4)
    ax.xaxis.grid(True, color="gray", linestyle="-", alpha=0.3)
    ax.spines["polar"].set_visible(False)

    ax.set_title(
        "Model Performance Across Hallucination Dimensions\n"
        "(Existence / Attribute / Relation — X-POPE)",
        fontsize=13, pad=20, fontweight="bold",
    )

    ax.legend(
        loc="upper right",
        bbox_to_anchor=(1.35, 1.15),
        fontsize=10,
        framealpha=0.9,
    )

    plt.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved → {out}")
