"""
bias_analysis.py
----------------
Module 1 (supplementary): Yes-bias distribution across models and benchmarks.

yes_bias = yes_rate - 0.5
  > 0  → model over-predicts "yes"
  < 0  → model over-predicts "no"
  = 0  → perfectly calibrated

A well-designed benchmark should produce near-zero bias for all models.
Systematic bias (e.g. all models > +0.1) indicates benchmark imbalance
or prompt sensitivity.

Usage
-----
    python -m src.analysis.bias_analysis \
        --results-dir results/predictions \
        --output reports/module1_bias.json

    # Only POPE splits
    python -m src.analysis.bias_analysis \
        --results-dir results/predictions \
        --benchmarks pope_adversarial pope_popular pope_random \
        --output reports/module1_bias.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from collections import defaultdict

import numpy as np


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

def load_summaries(results_dir: Path) -> list[dict]:
    summaries = []
    for path in sorted(results_dir.glob("*_summary.json")):
        if path.name.startswith("_"):
            continue
        with open(path) as f:
            summaries.append(json.load(f))
    return summaries


# ---------------------------------------------------------------------------
# Core
# ---------------------------------------------------------------------------

def yes_bias(yes_rate: float) -> float:
    return round(yes_rate - 0.5, 4)


def bias_for_benchmark(records: list[dict]) -> dict:
    """
    Compute yes-bias stats for one benchmark across all models.

    Returns
    -------
    {
        benchmark,
        per_model: {model: {yes_rate, yes_bias, accuracy}},
        mean_bias, std_bias, max_abs_bias,
        systematic: bool,   # True if |mean_bias| > 0.1
        direction: str,     # "yes-leaning" / "no-leaning" / "balanced"
    }
    """
    per_model = {}
    for r in records:
        if r.get("n_unknown", 0) == r.get("n_total", 1):
            continue
        yr = r.get("yes_rate", 0.5)
        per_model[r["model"]] = {
            "yes_rate":  round(yr, 4),
            "yes_bias":  yes_bias(yr),
            "accuracy":  round(r.get("accuracy", 0), 4),
        }

    if not per_model:
        return {"benchmark": records[0]["benchmark"], "error": "no valid records"}

    biases = [v["yes_bias"] for v in per_model.values()]
    arr = np.array(biases)
    mean_bias = float(arr.mean())

    if mean_bias > 0.1:
        direction = "yes-leaning"
    elif mean_bias < -0.1:
        direction = "no-leaning"
    else:
        direction = "balanced"

    return {
        "benchmark":    records[0]["benchmark"],
        "per_model":    per_model,
        "mean_bias":    round(mean_bias, 4),
        "std_bias":     round(float(arr.std()), 4),
        "max_abs_bias": round(float(np.abs(arr).max()), 4),
        "systematic":   abs(mean_bias) > 0.1,
        "direction":    direction,
    }


# ---------------------------------------------------------------------------
# Cross-benchmark bias stability
# ---------------------------------------------------------------------------

def bias_stability(
    per_bm: dict[str, dict],
) -> dict[str, dict]:
    """
    For each model, compute bias variance across benchmarks.
    High variance → model's yes-bias is unstable across benchmarks.

    Returns {model: {biases_by_benchmark, mean_bias, std_bias}}
    """
    model_biases: dict[str, dict[str, float]] = defaultdict(dict)
    for bm, data in per_bm.items():
        for model, stats in data.get("per_model", {}).items():
            model_biases[model][bm] = stats["yes_bias"]

    result = {}
    for model, bm_biases in sorted(model_biases.items()):
        vals = list(bm_biases.values())
        arr  = np.array(vals)
        result[model] = {
            "biases_by_benchmark": bm_biases,
            "mean_bias":           round(float(arr.mean()), 4),
            "std_bias":            round(float(arr.std()), 4),
            "range":               round(float(arr.max() - arr.min()), 4),
        }
    return result


# ---------------------------------------------------------------------------
# Full report
# ---------------------------------------------------------------------------

def run(
    results_dir: Path,
    benchmarks: list[str] | None = None,
) -> dict:
    """
    Returns
    -------
    {
        "per_benchmark": {benchmark: bias_dict},
        "bias_stability": {model: stability_dict},
        "systematic_benchmarks": [str],
        "summary_table": [{benchmark, mean_bias, direction, systematic}],
    }
    """
    summaries = load_summaries(results_dir)

    groups: dict[str, list[dict]] = defaultdict(list)
    for s in summaries:
        groups[s["benchmark"]].append(s)

    if benchmarks:
        groups = {k: v for k, v in groups.items() if k in benchmarks}

    per_bm = {
        bm: bias_for_benchmark(records)
        for bm, records in sorted(groups.items())
    }

    stability = bias_stability(per_bm)

    systematic = [
        bm for bm, d in per_bm.items() if d.get("systematic")
    ]

    summary_table = [
        {
            "benchmark":  bm,
            "mean_bias":  d.get("mean_bias"),
            "std_bias":   d.get("std_bias"),
            "direction":  d.get("direction"),
            "systematic": d.get("systematic"),
        }
        for bm, d in per_bm.items()
    ]
    summary_table.sort(key=lambda x: abs(x["mean_bias"] or 0), reverse=True)

    return {
        "per_benchmark":          per_bm,
        "bias_stability":         stability,
        "systematic_benchmarks":  systematic,
        "summary_table":          summary_table,
    }


def print_report(report: dict):
    print(f"\n{'='*65}")
    print("YES-BIAS REPORT")
    print(f"{'='*65}")
    print(f"{'Benchmark':<30} {'Mean bias':>10} {'Std':>6} {'Direction':<15} Systematic?")
    print(f"{'-'*65}")
    for row in report["summary_table"]:
        sys_str = "⚠️  YES" if row["systematic"] else "✅ no"
        print(
            f"{row['benchmark']:<30} "
            f"{row['mean_bias']:>+10.4f} "
            f"{row['std_bias']:>6.4f} "
            f"{row['direction']:<15} "
            f"{sys_str}"
        )

    print(f"\n── Per-model bias stability (across benchmarks) ──")
    print(f"{'Model':<28} {'Mean bias':>10} {'Std':>6} {'Range':>6}")
    print(f"{'-'*55}")
    for model, s in report["bias_stability"].items():
        print(
            f"{model:<28} "
            f"{s['mean_bias']:>+10.4f} "
            f"{s['std_bias']:>6.4f} "
            f"{s['range']:>6.4f}"
        )

    print(f"\nSystematic bias benchmarks: {report['systematic_benchmarks']}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="results/predictions")
    parser.add_argument("--benchmarks",  nargs="*", default=None)
    parser.add_argument("--output",      default="reports/module1_bias.json")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    report = run(results_dir, args.benchmarks)
    print_report(report)

    with open(output_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nSaved → {output_path}")


if __name__ == "__main__":
    main()
