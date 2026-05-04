"""
saturation_diag.py
------------------
Module 1: Quantify POPE saturation using Coefficient of Variation (CV).

CV = std / mean across model accuracy/F1 scores on the same benchmark.
Low CV (≤ 0.02) → models cluster at ceiling → benchmark cannot rank them.

Usage
-----
    python -m src.analysis.saturation_diag \
        --results-dir results/predictions \
        --output reports/module1_saturation.json

    # Compare specific benchmarks
    python -m src.analysis.saturation_diag \
        --results-dir results/predictions \
        --benchmarks pope_adversarial pope_popular pope_random dashb \
        --output reports/module1_saturation.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from collections import defaultdict

import numpy as np

# CV threshold below which a benchmark is declared saturated
SATURATION_THRESHOLD = 0.02


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def load_summaries(results_dir: Path) -> list[dict]:
    """Load all *_summary.json files from results_dir."""
    summaries = []
    for path in sorted(results_dir.glob("*_summary.json")):
        if path.name.startswith("_"):
            continue
        with open(path) as f:
            summaries.append(json.load(f))
    return summaries


def group_by_benchmark(summaries: list[dict]) -> dict[str, list[dict]]:
    """Group summary records by benchmark name."""
    groups: dict[str, list[dict]] = defaultdict(list)
    for s in summaries:
        groups[s["benchmark"]].append(s)
    return dict(groups)


# ---------------------------------------------------------------------------
# Core computation
# ---------------------------------------------------------------------------

def cv(scores: list[float]) -> float:
    """Coefficient of Variation = std / mean (population std)."""
    arr = np.array(scores)
    mean = arr.mean()
    return float(arr.std() / mean) if mean > 0 else 0.0


def saturation_for_benchmark(
    records: list[dict],
    metric: str = "accuracy",
) -> dict:
    """
    Compute saturation diagnostics for one benchmark across all models.

    Parameters
    ----------
    records : list of summary dicts for the same benchmark
    metric  : 'accuracy' or 'yes_rate'

    Returns
    -------
    {
        benchmark, metric, n_models,
        scores: {model: score},
        mean, std, cv, max_gap,
        saturated: bool,
        ranking: [model sorted best→worst]
    }
    """
    scores = {r["model"]: r[metric] for r in records if metric in r}
    vals = list(scores.values())

    if len(vals) < 2:
        return {"benchmark": records[0]["benchmark"], "error": "too few models"}

    arr = np.array(vals)
    cv_val = cv(vals)

    ranking = sorted(scores, key=scores.get, reverse=True)

    return {
        "benchmark":  records[0]["benchmark"],
        "metric":     metric,
        "n_models":   len(scores),
        "scores":     scores,
        "mean":       round(float(arr.mean()), 4),
        "std":        round(float(arr.std()), 4),
        "cv":         round(cv_val, 4),
        "max_gap":    round(float(arr.max() - arr.min()), 4),
        "saturated":  cv_val <= SATURATION_THRESHOLD,
        "ranking":    ranking,
    }


# ---------------------------------------------------------------------------
# Full report
# ---------------------------------------------------------------------------

def run(
    results_dir: Path,
    benchmarks: list[str] | None = None,
    metric: str = "accuracy",
) -> dict:
    """
    Run saturation diagnostics across all benchmarks.

    Returns
    -------
    {
        "per_benchmark": {benchmark: saturation_dict},
        "summary_table": [{benchmark, cv, mean, max_gap, saturated}],
        "saturated_benchmarks": [str],
        "threshold": float,
    }
    """
    summaries = load_summaries(results_dir)
    groups = group_by_benchmark(summaries)

    if benchmarks:
        groups = {k: v for k, v in groups.items() if k in benchmarks}

    per_benchmark = {}
    for bm, records in sorted(groups.items()):
        per_benchmark[bm] = saturation_for_benchmark(records, metric)

    summary_table = [
        {
            "benchmark": bm,
            "cv":        d.get("cv"),
            "mean":      d.get("mean"),
            "std":       d.get("std"),
            "max_gap":   d.get("max_gap"),
            "saturated": d.get("saturated"),
        }
        for bm, d in per_benchmark.items()
    ]
    # Sort by CV ascending (most saturated first)
    summary_table.sort(key=lambda x: x["cv"] or 999)

    saturated = [d["benchmark"] for d in summary_table if d.get("saturated")]

    return {
        "per_benchmark":        per_benchmark,
        "summary_table":        summary_table,
        "saturated_benchmarks": saturated,
        "threshold_cv":         SATURATION_THRESHOLD,
        "metric":               metric,
    }


def print_report(report: dict):
    print(f"\n{'='*60}")
    print(f"SATURATION REPORT  (metric={report['metric']}, threshold CV≤{report['threshold_cv']})")
    print(f"{'='*60}")
    print(f"{'Benchmark':<30} {'CV':>6} {'Mean':>6} {'Gap':>6} {'Saturated'}")
    print(f"{'-'*60}")
    for row in report["summary_table"]:
        sat = "✅ YES" if row["saturated"] else "❌ no"
        print(
            f"{row['benchmark']:<30} "
            f"{row['cv']:>6.4f} "
            f"{row['mean']:>6.4f} "
            f"{row['max_gap']:>6.4f} "
            f"{sat}"
        )
    print(f"\nSaturated benchmarks: {report['saturated_benchmarks']}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="results/predictions")
    parser.add_argument("--benchmarks",  nargs="*", default=None,
                        help="Subset of benchmarks to analyse (default: all)")
    parser.add_argument("--metric",      default="accuracy",
                        choices=["accuracy", "yes_rate"])
    parser.add_argument("--output",      default="reports/module1_saturation.json")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    report = run(results_dir, args.benchmarks, args.metric)
    print_report(report)

    with open(output_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nSaved → {output_path}")


if __name__ == "__main__":
    main()
