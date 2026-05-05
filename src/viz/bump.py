"""
bump.py — Bump chart: model rank changes across POPE → RePOPE → DASH-B.

Data source: results/predictions/*_summary.json
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from scipy.stats import spearmanr

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

STAGES = ["POPE\nadv", "RePOPE\nadv", "DASH-B"]
STAGE_BENCHMARKS = [
    "pope_adversarial",
    "repope_adversarial",
    "dashb",
]


def _get_ranking(results_dir: Path, benchmark: str) -> dict[str, int]:
    """Load accuracy for each model on a benchmark and return rank dict."""
    accs = {}
    for model in MODEL_LABELS:
        p = results_dir / f"{model}_{benchmark}_summary.json"
        if p.exists():
            with open(p) as f:
                d = json.load(f)
            # skip degenerate (all unknown)
            if d.get("n_unknown", 0) < d.get("n_total", 1):
                accs[model] = d["accuracy"]

    sorted_models = sorted(accs, key=accs.get, reverse=True)
    return {m: i + 1 for i, m in enumerate(sorted_models)}


def plot_bump(results_dir: Path, out: Path) -> None:
    rankings = [_get_ranking(results_dir, bm) for bm in STAGE_BENCHMARKS]
    n_stages = len(STAGES)
    n_models = len(MODEL_LABELS)

    # Spearman rho between consecutive stages
    rhos = []
    for i in range(n_stages - 1):
        a = [rankings[i].get(m, n_models) for m in MODEL_LABELS]
        b = [rankings[i+1].get(m, n_models) for m in MODEL_LABELS]
        rho, _ = spearmanr(a, b)
        rhos.append(rho)

    fig, ax = plt.subplots(figsize=(8, 5.5))

    xs = list(range(n_stages))

    for model in MODEL_LABELS:
        color = MODEL_COLORS[model]
        label = MODEL_LABELS[model]
        ys = [rankings[i].get(model, n_models) for i in range(n_stages)]

        # Line
        ax.plot(xs, ys, color=color, linewidth=2.8, solid_capstyle="round",
                path_effects=[pe.Stroke(linewidth=4.5, foreground="white"), pe.Normal()])

        # Circles with rank number
        for xi, yi in zip(xs, ys):
            ax.scatter(xi, yi, color=color, s=500, zorder=6,
                       edgecolors="white", linewidths=1.5)
            ax.text(xi, yi, str(yi), ha="center", va="center",
                    fontsize=9, fontweight="bold", color="white", zorder=7)

        # Model label on right
        ax.text(n_stages - 1 + 0.12, ys[-1], label,
                va="center", ha="left", fontsize=9.5, color=color, fontweight="bold")

    # Spearman rho annotations between stages
    for i, rho in enumerate(rhos):
        mid_x = (xs[i] + xs[i+1]) / 2
        ax.text(mid_x, 0.35, f"ρ = {rho:+.2f}",
                ha="center", va="center", fontsize=10, color="#444",
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#aaa", lw=1))

    ax.set_xticks(xs)
    ax.set_xticklabels(STAGES, fontsize=12, fontweight="bold")
    ax.set_yticks(range(1, n_models + 1))
    ax.set_yticklabels([f"#{r}" for r in range(1, n_models + 1)], fontsize=11)
    ax.invert_yaxis()
    ax.set_xlim(-0.4, n_stages - 1 + 1.3)
    ax.set_ylim(n_models + 0.5, 0.3)
    ax.yaxis.grid(True, color="gray", linestyle="--", alpha=0.3)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(left=False, bottom=False)

    ax.set_title(
        "Ranking Shifts: POPE → RePOPE → DASH-B\n"
        "(Adversarial split · Spearman ρ annotated between stages)",
        fontsize=13, fontweight="bold", pad=12,
    )

    plt.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved → {out}")
