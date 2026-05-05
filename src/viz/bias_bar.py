"""
bias_bar.py — Grouped bar chart: yes-bias across benchmark families.

yes_bias = yes_rate - 0.5
  > 0 → yes-leaning  (warm color)
  < 0 → no-leaning   (cool color)

Data source: results/predictions/*_summary.json
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

MODEL_LABELS = {
    "internvl2_8b":  "InternVL2-8B",
    "llama32v_11b":  "Llama3.2V-11B",
    "llava_ov_7b":   "LLaVA-OV-7B",
    "paligemma2_3b": "PaliGemma2-3B",
    "phi35v_4b":     "Phi-3.5V-4B",
    "qwen2vl_7b":    "Qwen2-VL-7B",
}

# Benchmark families: name → list of benchmark keys to average
FAMILIES = {
    "POPE":   ["pope_adversarial", "pope_popular", "pope_random"],
    "RePOPE": ["repope_adversarial", "repope_popular", "repope_random"],
    "DASH-B": ["dashb"],
}

FAMILY_COLORS = {
    "POPE":   "#4C72B0",
    "RePOPE": "#55A868",
    "DASH-B": "#C44E52",
}


def _load_biases(results_dir: Path) -> dict[str, dict[str, float]]:
    """Return {family: {model: avg_yes_bias}}."""
    result = {}
    for family, benchmarks in FAMILIES.items():
        model_vals: dict[str, list[float]] = {m: [] for m in MODEL_LABELS}
        for bm in benchmarks:
            for model in MODEL_LABELS:
                p = results_dir / f"{model}_{bm}_summary.json"
                if p.exists():
                    with open(p) as f:
                        d = json.load(f)
                    if d.get("n_unknown", 0) < d.get("n_total", 1):
                        model_vals[model].append(d["yes_rate"] - 0.5)
        result[family] = {
            m: sum(v) / len(v) for m, v in model_vals.items() if v
        }
    return result


def plot_bias_bar(results_dir: Path, out: Path) -> None:
    biases = _load_biases(results_dir)

    model_keys = list(MODEL_LABELS.keys())
    model_display = [MODEL_LABELS[m] for m in model_keys]
    families = list(FAMILIES.keys())

    n_models  = len(model_keys)
    n_families = len(families)
    x = np.arange(n_models)
    width = 0.22
    offsets = np.linspace(-(n_families-1)/2, (n_families-1)/2, n_families) * width

    fig, ax = plt.subplots(figsize=(10, 5))

    for fam, offset in zip(families, offsets):
        vals = [biases[fam].get(m, 0.0) for m in model_keys]
        color = FAMILY_COLORS[fam]
        bars = ax.bar(x + offset, vals, width, label=fam,
                      color=color, alpha=0.85, edgecolor="white", linewidth=0.8)

        # Value labels
        for bar, val in zip(bars, vals):
            va = "bottom" if val >= 0 else "top"
            pad = 0.004 if val >= 0 else -0.004
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                val + pad,
                f"{val:+.2f}",
                ha="center", va=va, fontsize=7.5, color="#333",
            )

    # Zero line
    ax.axhline(0, color="black", linewidth=1.2, zorder=5)

    ax.set_xticks(x)
    ax.set_xticklabels(model_display, fontsize=10.5, rotation=10, ha="right")
    ax.set_ylabel("Yes Bias  (yes_rate − 0.5)", fontsize=11)
    ax.yaxis.grid(True, color="gray", linestyle="--", alpha=0.35)
    ax.set_axisbelow(True)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)

    # Bias direction annotations
    ax.text(0.01, 0.97, "← no-leaning", transform=ax.transAxes,
            fontsize=9, color="#1565C0", va="top")
    ax.text(0.01, 0.03, "yes-leaning →", transform=ax.transAxes,
            fontsize=9, color="#EF553B", va="bottom")

    ax.set_title(
        "Yes-Bias Distribution by Benchmark Family\n"
        "(yes_rate − 0.5 · positive = yes-leaning · negative = no-leaning)",
        fontsize=13, fontweight="bold", pad=12,
    )
    ax.legend(title="Benchmark", fontsize=10, title_fontsize=10,
              loc="upper right", framealpha=0.9)

    plt.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved → {out}")
