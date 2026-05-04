"""
ranking_shift.py
----------------
Module 2 & 3: Quantify ranking instability across benchmarks using
Spearman rank correlation (ρ).

Low ρ between POPE and RePOPE → annotation errors distort rankings (Module 2)
CV recovery on DASH-B         → harder benchmark restores discriminability (Module 3)

Usage
-----
    python -m src.analysis.ranking_shift \
        --results-dir results/predictions \
        --output reports/module2_ranking.json

    # Custom benchmark sequence
    python -m src.analysis.ranking_shift \
        --results-dir results/predictions \
        --benchmarks pope_adversarial repope_adversarial dashb \
        --output reports/module23_ranking.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from collections import defaultdict
from itertools import combinations

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


def build_score_matrix(
    summaries: list[dict],
    metric: str = "accuracy",
) -> dict[str, dict[str, float]]:
    """
    Returns {benchmark: {model: score}}.
    Only includes (benchmark, model) pairs where the score is valid
    (i.e. not a 0.5 random-baseline result with unknown=n_total).
    """
    matrix: dict[str, dict[str, float]] = defaultdict(dict)
    for s in summaries:
        # Skip degenerate results (all unknown → random baseline)
        if s.get("n_unknown", 0) == s.get("n_total", 1):
            continue
        bm    = s["benchmark"]
        model = s["model"]
        if metric in s:
            matrix[bm][model] = s[metric]
    return dict(matrix)


# ---------------------------------------------------------------------------
# Spearman ρ
# ---------------------------------------------------------------------------

def spearman_rho(
    scores_a: dict[str, float],
    scores_b: dict[str, float],
) -> tuple[float, list[str]]:
    """
    Compute Spearman ρ between two {model: score} dicts.
    Only uses models present in both dicts.

    Returns (rho, common_models).
    """
    common = sorted(set(scores_a) & set(scores_b))
    if len(common) < 2:
        return float("nan"), common

    a = np.array([scores_a[m] for m in common])
    b = np.array([scores_b[m] for m in common])

    # Rank (1 = best)
    rank_a = len(a) + 1 - a.argsort().argsort()
    rank_b = len(b) + 1 - b.argsort().argsort()

    d2 = ((rank_a - rank_b) ** 2).sum()
    n  = len(common)
    rho = 1 - 6 * d2 / (n * (n**2 - 1))

    return round(float(rho), 4), common


def ranking_table(scores: dict[str, float]) -> list[tuple[int, str, float]]:
    """Return [(rank, model, score)] sorted best→worst."""
    sorted_models = sorted(scores, key=scores.get, reverse=True)
    return [(i+1, m, round(scores[m], 4)) for i, m in enumerate(sorted_models)]


# ---------------------------------------------------------------------------
# Rank change analysis
# ---------------------------------------------------------------------------

def rank_changes(
    scores_a: dict[str, float],
    scores_b: dict[str, float],
) -> dict[str, dict]:
    """
    For each model, compute rank in benchmark A vs B and the delta.

    Returns {model: {rank_a, rank_b, delta, score_a, score_b}}
    """
    common = sorted(set(scores_a) & set(scores_b))
    ranks_a = {m: r for r, m, _ in ranking_table({m: scores_a[m] for m in common})}
    ranks_b = {m: r for r, m, _ in ranking_table({m: scores_b[m] for m in common})}

    result = {}
    for m in common:
        delta = ranks_a[m] - ranks_b[m]   # positive = moved up in B
        result[m] = {
            "rank_a":  ranks_a[m],
            "rank_b":  ranks_b[m],
            "delta":   delta,
            "score_a": round(scores_a[m], 4),
            "score_b": round(scores_b[m], 4),
        }
    return result


# ---------------------------------------------------------------------------
# Full report
# ---------------------------------------------------------------------------

def run(
    results_dir: Path,
    benchmarks: list[str] | None = None,
    metric: str = "accuracy",
) -> dict:
    """
    Compute pairwise Spearman ρ and rank changes across benchmarks.

    Returns
    -------
    {
        "metric": str,
        "benchmarks_analysed": [str],
        "rankings": {benchmark: [(rank, model, score)]},
        "pairwise_rho": {
            "pope_adversarial__repope_adversarial": {
                rho, common_models, rank_changes: {model: {...}}
            },
            ...
        },
        "rho_summary": [{pair, rho, interpretation}],
    }
    """
    summaries  = load_summaries(results_dir)
    matrix     = build_score_matrix(summaries, metric)

    if benchmarks:
        matrix = {k: v for k, v in matrix.items() if k in benchmarks}

    bm_list = sorted(matrix.keys())

    # Per-benchmark rankings
    rankings = {
        bm: ranking_table(matrix[bm])
        for bm in bm_list
    }

    # Pairwise Spearman ρ
    pairwise = {}
    rho_summary = []

    for bm_a, bm_b in combinations(bm_list, 2):
        rho, common = spearman_rho(matrix[bm_a], matrix[bm_b])
        changes     = rank_changes(matrix[bm_a], matrix[bm_b])
        key         = f"{bm_a}__{bm_b}"

        # Interpretation
        if np.isnan(rho):
            interp = "insufficient data"
        elif rho >= 0.9:
            interp = "very stable ranking"
        elif rho >= 0.7:
            interp = "moderately stable"
        elif rho >= 0.5:
            interp = "unstable — annotation/difficulty effects present"
        else:
            interp = "highly unstable — rankings unreliable"

        pairwise[key] = {
            "benchmark_a":   bm_a,
            "benchmark_b":   bm_b,
            "rho":           rho,
            "n_common":      len(common),
            "common_models": common,
            "interpretation":interp,
            "rank_changes":  changes,
        }

        rho_summary.append({
            "pair":           key,
            "rho":            rho,
            "interpretation": interp,
        })

    rho_summary.sort(key=lambda x: x["rho"] if not np.isnan(x["rho"]) else 999)

    return {
        "metric":               metric,
        "benchmarks_analysed":  bm_list,
        "rankings":             {bm: [(r, m, s) for r, m, s in rankings[bm]]
                                 for bm in bm_list},
        "pairwise_rho":         pairwise,
        "rho_summary":          rho_summary,
    }


def print_report(report: dict):
    print(f"\n{'='*65}")
    print(f"RANKING SHIFT REPORT  (metric={report['metric']})")
    print(f"{'='*65}")

    for bm, rows in report["rankings"].items():
        print(f"\n── {bm} ──")
        for rank, model, score in rows:
            print(f"  #{rank}  {model:<28}  {score:.4f}")

    print(f"\n── Pairwise Spearman ρ ──")
    print(f"{'Pair':<50} {'ρ':>6}  Interpretation")
    print(f"{'-'*65}")
    for row in report["rho_summary"]:
        rho_str = f"{row['rho']:.4f}" if not np.isnan(row["rho"]) else "  nan"
        print(f"{row['pair']:<50} {rho_str:>6}  {row['interpretation']}")

    # Highlight biggest rank movers
    print(f"\n── Biggest rank changes (POPE adv → RePOPE adv) ──")
    key = "pope_adversarial__repope_adversarial"
    if key in report["pairwise_rho"]:
        changes = report["pairwise_rho"][key]["rank_changes"]
        sorted_changes = sorted(changes.items(), key=lambda x: abs(x[1]["delta"]), reverse=True)
        for model, c in sorted_changes:
            arrow = "↑" if c["delta"] > 0 else ("↓" if c["delta"] < 0 else "→")
            print(
                f"  {model:<28}  #{c['rank_a']} → #{c['rank_b']}  "
                f"({arrow}{abs(c['delta'])})  "
                f"{c['score_a']:.3f} → {c['score_b']:.3f}"
            )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="results/predictions")
    parser.add_argument(
        "--benchmarks", nargs="*",
        default=["pope_adversarial", "repope_adversarial", "dashb"],
        help="Benchmarks to include (in order)",
    )
    parser.add_argument("--metric", default="accuracy",
                        choices=["accuracy", "f1", "yes_rate"])
    parser.add_argument("--output", default="reports/module23_ranking.json")
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
