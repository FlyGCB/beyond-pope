"""
saturation_diag.py
------------------
Module 1: Quantify POPE saturation using Coefficient of Variation (CV).

CV = std / mean across model accuracy scores on the same benchmark.
Low CV (≤ 0.02) → models cluster at ceiling → benchmark cannot rank them.

Key finding: POPE is saturated for 7B+ models (CV ≤ 0.014 on popular/random),
but this signal is masked when smaller models (3B/4B) are included.

Usage
-----
    python -m src.analysis.saturation_diag \
        --results-dir results/predictions \
        --benchmarks pope_adversarial pope_popular pope_random dashb \
        --model-groups "7B+:qwen2vl_7b,internvl2_8b,llava_ov_7b,llama32v_11b" \
                       "all:qwen2vl_7b,internvl2_8b,llava_ov_7b,llama32v_11b,paligemma2_3b,phi35v_4b" \
        --output reports/module1_saturation.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from collections import defaultdict

import numpy as np

SATURATION_THRESHOLD = 0.02


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def load_summaries(results_dir: Path) -> list[dict]:
    summaries = []
    for path in sorted(results_dir.glob("*_summary.json")):
        if path.name.startswith("_"):
            continue
        with open(path) as f:
            summaries.append(json.load(f))
    return summaries


def group_by_benchmark(summaries: list[dict]) -> dict[str, list[dict]]:
    groups: dict[str, list[dict]] = defaultdict(list)
    for s in summaries:
        groups[s["benchmark"]].append(s)
    return dict(groups)


def parse_model_groups(raw: list[str]) -> dict[str, list[str]]:
    """
    Parse --model-groups arguments.
    Format: "group_name:model1,model2,model3"
    """
    groups = {}
    for item in raw:
        name, models_str = item.split(":", 1)
        groups[name.strip()] = [m.strip() for m in models_str.split(",")]
    return groups


# ---------------------------------------------------------------------------
# Core computation
# ---------------------------------------------------------------------------

def cv(scores: list[float]) -> float:
    arr = np.array(scores)
    mean = arr.mean()
    return float(arr.std() / mean) if mean > 0 else 0.0


def saturation_for_benchmark(
    records: list[dict],
    metric: str = "accuracy",
    model_filter: list[str] | None = None,
) -> dict:
    scores = {
        r["model"]: r[metric]
        for r in records
        if metric in r
        and (model_filter is None or r["model"] in model_filter)
        and r.get("n_unknown", 0) < r.get("n_total", 1)
    }
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
    model_groups: dict[str, list[str]] | None = None,
) -> dict:
    summaries = load_summaries(results_dir)
    groups = group_by_benchmark(summaries)

    if benchmarks:
        groups = {k: v for k, v in groups.items() if k in benchmarks}

    # All-model analysis
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
    summary_table.sort(key=lambda x: x["cv"] or 999)
    saturated = [d["benchmark"] for d in summary_table if d.get("saturated")]

    # Per-group analysis
    per_group = {}
    group_summary_tables = {}

    if model_groups:
        for group_name, model_list in model_groups.items():
            group_per_bm = {}
            for bm, records in sorted(groups.items()):
                group_per_bm[bm] = saturation_for_benchmark(
                    records, metric, model_filter=model_list
                )
            per_group[group_name] = group_per_bm

            tbl = [
                {
                    "benchmark": bm,
                    "cv":        d.get("cv"),
                    "mean":      d.get("mean"),
                    "std":       d.get("std"),
                    "max_gap":   d.get("max_gap"),
                    "saturated": d.get("saturated"),
                    "models":    model_list,
                }
                for bm, d in group_per_bm.items()
            ]
            tbl.sort(key=lambda x: x["cv"] or 999)
            group_summary_tables[group_name] = tbl

    return {
        "per_benchmark":        per_benchmark,
        "per_group":            per_group,
        "summary_table":        summary_table,
        "group_summary_tables": group_summary_tables,
        "saturated_benchmarks": saturated,
        "threshold_cv":         SATURATION_THRESHOLD,
        "metric":               metric,
    }


def print_report(report: dict):
    thresh = report["threshold_cv"]
    metric = report["metric"]

    print(f"\n{'='*65}")
    print(f"SATURATION REPORT  (metric={metric}, threshold CV≤{thresh})")
    print(f"{'='*65}")

    print(f"\n── All models ──")
    print(f"{'Benchmark':<30} {'CV':>7} {'Mean':>7} {'Gap':>7} {'N':>3}  Saturated")
    print(f"{'-'*65}")
    for row in report["summary_table"]:
        sat = "✅ YES" if row["saturated"] else "❌ no"
        n = report["per_benchmark"].get(row["benchmark"], {}).get("n_models", "?")
        print(
            f"{row['benchmark']:<30} "
            f"{row['cv']:>7.4f} "
            f"{row['mean']:>7.4f} "
            f"{row['max_gap']:>7.4f} "
            f"{n:>3}  {sat}"
        )

    for group_name, tbl in report.get("group_summary_tables", {}).items():
        print(f"\n── Model group: {group_name} ──")
        print(f"{'Benchmark':<30} {'CV':>7} {'Mean':>7} {'Gap':>7} {'N':>3}  Saturated")
        print(f"{'-'*65}")
        for row in tbl:
            sat = "✅ YES" if row["saturated"] else "❌ no"
            n = len(row.get("models", []))
            print(
                f"{row['benchmark']:<30} "
                f"{row['cv']:>7.4f} "
                f"{row['mean']:>7.4f} "
                f"{row['max_gap']:>7.4f} "
                f"{n:>3}  {sat}"
            )

    print(f"\nSaturated benchmarks (all models): {report['saturated_benchmarks']}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="results/predictions")
    parser.add_argument("--benchmarks",  nargs="*", default=None)
    parser.add_argument("--metric",      default="accuracy",
                        choices=["accuracy", "yes_rate"])
    parser.add_argument(
        "--model-groups", nargs="*", default=None,
        metavar="NAME:model1,model2,...",
        help="Named model subsets. Format: 'GroupName:model1,model2,model3'",
    )
    parser.add_argument("--output", default="reports/module1_saturation.json")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    model_groups = parse_model_groups(args.model_groups) if args.model_groups else None

    report = run(results_dir, args.benchmarks, args.metric, model_groups)
    print_report(report)

    with open(output_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nSaved → {output_path}")


if __name__ == "__main__":
    main()
