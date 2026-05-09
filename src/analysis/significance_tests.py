"""
significance_tests.py
---------------------
Statistical significance tests for beyond-POPE's core claims.

Test 1 — McNemar test: POPE vs RePOPE per-model accuracy shift
    H0: annotation corrections do not change a model's error pattern
    Reject H0 (p < 0.05) → annotation errors significantly distort results

Test 2 — Paired t-test: Existence vs Attribute vs Relation accuracy
    H0: the three X-POPE dimensions have equal per-sample accuracy
    Reject H0 (p < 0.05) → dimension gap is statistically significant

Test 3 — Bootstrap CI: POPE adversarial CV for 7B+ models
    Estimates 95% confidence interval for the CV saturation metric

Usage
-----
    python -m src.analysis.significance_tests \
        --results-dir results/predictions \
        --output reports/significance.json
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
from scipy import stats

random.seed(42)
np.random.seed(42)

MODELS = [
    "qwen2vl_7b", "internvl2_8b", "llava_ov_7b",
    "llama32v_11b", "paligemma2_3b", "phi35v_4b",
]
MODELS_7B_PLUS = [
    "qwen2vl_7b", "internvl2_8b", "llava_ov_7b", "llama32v_11b",
]


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def load_predictions(results_dir: Path, model: str, benchmark: str) -> list[dict]:
    path = results_dir / f"{model}_{benchmark}.jsonl"
    if not path.exists():
        return []
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def get_correct_array(preds: list[dict]) -> np.ndarray:
    """Return binary array: 1=correct, 0=wrong, skip unknowns."""
    return np.array([
        1 if r.get("correct") else 0
        for r in preds
        if r.get("correct") is not None
    ], dtype=int)


def align_by_question_id(
    preds_a: list[dict],
    preds_b: list[dict],
) -> tuple[np.ndarray, np.ndarray]:
    """
    Align two prediction lists by question_id so we compare the same questions.
    Returns (correct_a, correct_b) arrays of equal length.
    """
    map_a = {r["question_id"]: r for r in preds_a if r.get("correct") is not None}
    map_b = {r["question_id"]: r for r in preds_b if r.get("correct") is not None}
    common = sorted(set(map_a) & set(map_b))

    if not common:
        return np.array([]), np.array([])

    a = np.array([1 if map_a[qid]["correct"] else 0 for qid in common])
    b = np.array([1 if map_b[qid]["correct"] else 0 for qid in common])
    return a, b


# ---------------------------------------------------------------------------
# Test 1: McNemar test — POPE vs RePOPE
# ---------------------------------------------------------------------------

def mcnemar_test(
    correct_a: np.ndarray,
    correct_b: np.ndarray,
) -> dict:
    """
    McNemar test for paired binary outcomes.

    Contingency table:
                    RePOPE correct  RePOPE wrong
    POPE correct         a               b
    POPE wrong           c               d

    Statistic: (b - c)^2 / (b + c)  ~ chi2(1)

    b = correct on POPE, wrong on RePOPE
    c = wrong on POPE, correct on RePOPE

    Large c >> b → annotation corrections helped the model (was being
    unfairly penalised on POPE).
    """
    a = int(np.sum((correct_a == 1) & (correct_b == 1)))
    b = int(np.sum((correct_a == 1) & (correct_b == 0)))
    c = int(np.sum((correct_a == 0) & (correct_b == 1)))
    d = int(np.sum((correct_a == 0) & (correct_b == 0)))

    if (b + c) == 0:
        return {"statistic": 0.0, "p_value": 1.0, "b": b, "c": c,
                "a": a, "d": d, "significant": False,
                "note": "b+c=0, no discordant pairs"}

    # Continuity-corrected McNemar
    statistic = (abs(b - c) - 1) ** 2 / (b + c)
    p_value   = float(1 - stats.chi2.cdf(statistic, df=1))

    return {
        "contingency": {"a": a, "b": b, "c": c, "d": d},
        "b_pope_correct_repope_wrong": b,
        "c_pope_wrong_repope_correct": c,
        "net_gain_from_correction":    c - b,
        "statistic": round(statistic, 4),
        "p_value":   round(p_value, 6),
        "significant": p_value < 0.05,
    }


def run_mcnemar_pope_vs_repope(results_dir: Path) -> dict:
    """Test 1: for each model, McNemar on POPE adv vs RePOPE adv."""
    results = {}
    for model in MODELS:
        pope   = load_predictions(results_dir, model, "pope_adversarial")
        repope = load_predictions(results_dir, model, "repope_adversarial")

        if not pope or not repope:
            results[model] = {"error": "missing predictions"}
            continue

        # POPE and RePOPE don't share the same question IDs (different items)
        # so we compare overall correct arrays by position (same label distribution)
        # and use McNemar on the overlap where image is shared
        # Simpler: compare per-image accuracy using image_id alignment
        pope_by_img   = {r["image"]: r for r in pope   if r.get("correct") is not None}
        repope_by_img = {r["image"]: r for r in repope if r.get("correct") is not None}
        common_imgs   = sorted(set(pope_by_img) & set(repope_by_img))

        if len(common_imgs) < 10:
            results[model] = {"error": f"too few common images: {len(common_imgs)}"}
            continue

        corr_pope   = np.array([1 if pope_by_img[img]["correct"]   else 0 for img in common_imgs])
        corr_repope = np.array([1 if repope_by_img[img]["correct"] else 0 for img in common_imgs])

        test = mcnemar_test(corr_pope, corr_repope)
        test["n_common_images"] = len(common_imgs)
        test["pope_acc"]   = round(float(corr_pope.mean()),   4)
        test["repope_acc"] = round(float(corr_repope.mean()), 4)
        results[model] = test

    n_significant = sum(1 for v in results.values() if v.get("significant"))
    return {
        "test":          "McNemar (POPE adv vs RePOPE adv)",
        "hypothesis":    "H0: annotation corrections do not change error pattern",
        "per_model":     results,
        "n_significant": n_significant,
        "n_models":      len(MODELS),
    }


# ---------------------------------------------------------------------------
# Test 2: Paired t-test — Existence vs Attribute vs Relation
# ---------------------------------------------------------------------------

def run_paired_ttest_xpope(results_dir: Path) -> dict:
    """
    Test 2: For each model, paired t-test between dimension accuracy arrays.
    We compare per-sample correctness vectors across dimensions.
    Since the samples are different, we use an independent samples t-test
    (Welch's t-test) on the per-sample binary correctness.
    """
    pairs = [
        ("existence", "attribute"),
        ("attribute",  "relation"),
        ("existence",  "relation"),
    ]

    results = {}
    for model in MODELS:
        model_results = {}
        dim_arrays = {}

        for dim in ["existence", "attribute", "relation"]:
            preds = load_predictions(results_dir, model, f"xpope_{dim}")
            if preds:
                dim_arrays[dim] = get_correct_array(preds)

        for dim_a, dim_b in pairs:
            if dim_a not in dim_arrays or dim_b not in dim_arrays:
                model_results[f"{dim_a}_vs_{dim_b}"] = {"error": "missing data"}
                continue

            arr_a = dim_arrays[dim_a].astype(float)
            arr_b = dim_arrays[dim_b].astype(float)

            # Welch's t-test (unequal sample sizes / variances)
            t_stat, p_val = stats.ttest_ind(arr_a, arr_b, equal_var=False)

            model_results[f"{dim_a}_vs_{dim_b}"] = {
                f"{dim_a}_acc": round(float(arr_a.mean()), 4),
                f"{dim_b}_acc": round(float(arr_b.mean()), 4),
                "mean_diff":    round(float(arr_a.mean() - arr_b.mean()), 4),
                "t_statistic":  round(float(t_stat), 4),
                "p_value":      round(float(p_val), 8),
                "significant":  p_val < 0.05,
                f"n_{dim_a}":   len(arr_a),
                f"n_{dim_b}":   len(arr_b),
            }

        results[model] = model_results

    # Summary: how many (model, pair) combinations are significant
    all_tests = [
        v for m in results.values() for v in m.values()
        if isinstance(v, dict) and "significant" in v
    ]
    n_sig = sum(1 for t in all_tests if t["significant"])

    return {
        "test":        "Welch's t-test (X-POPE dimension pairs)",
        "hypothesis":  "H0: existence / attribute / relation accuracy are equal",
        "per_model":   results,
        "n_significant_pairs": n_sig,
        "n_total_pairs":       len(all_tests),
    }


# ---------------------------------------------------------------------------
# Test 3: Bootstrap CI for CV (saturation metric)
# ---------------------------------------------------------------------------

def bootstrap_cv(
    scores: list[float],
    n_bootstrap: int = 10000,
    ci: float = 0.95,
) -> dict:
    """Bootstrap 95% CI for CV = std/mean."""
    scores_arr = np.array(scores)
    boot_cvs = []
    for _ in range(n_bootstrap):
        sample = np.random.choice(scores_arr, size=len(scores_arr), replace=True)
        mean = sample.mean()
        if mean > 0:
            boot_cvs.append(sample.std() / mean)

    boot_cvs = np.array(boot_cvs)
    alpha = (1 - ci) / 2
    lower = float(np.percentile(boot_cvs, alpha * 100))
    upper = float(np.percentile(boot_cvs, (1 - alpha) * 100))
    observed = float(scores_arr.std() / scores_arr.mean())

    return {
        "observed_cv":  round(observed, 4),
        "ci_lower":     round(lower, 4),
        "ci_upper":     round(upper, 4),
        "ci_level":     ci,
        "n_bootstrap":  n_bootstrap,
        "saturated_with_high_confidence": upper <= 0.02,
    }


def run_bootstrap_cv(results_dir: Path) -> dict:
    """Test 3: Bootstrap CI for CV on POPE splits, 7B+ models only."""
    splits = ["pope_adversarial", "pope_popular", "pope_random"]
    results = {}

    for split in splits:
        scores = []
        for model in MODELS_7B_PLUS:
            summary_path = results_dir / f"{model}_{split}_summary.json"
            if summary_path.exists():
                with open(summary_path) as f:
                    d = json.load(f)
                scores.append(d["accuracy"])

        if len(scores) < 2:
            results[split] = {"error": "insufficient data"}
            continue

        results[split] = {
            "models": MODELS_7B_PLUS,
            "scores": scores,
            **bootstrap_cv(scores),
        }

    return {
        "test":       "Bootstrap CI for CV (7B+ models on POPE)",
        "hypothesis": "Is POPE saturation for 7B+ models statistically robust?",
        "per_split":  results,
    }


# ---------------------------------------------------------------------------
# Full report
# ---------------------------------------------------------------------------

def run(results_dir: Path) -> dict:
    print("Running Test 1: McNemar (POPE vs RePOPE)...")
    test1 = run_mcnemar_pope_vs_repope(results_dir)

    print("Running Test 2: Welch's t-test (X-POPE dimensions)...")
    test2 = run_paired_ttest_xpope(results_dir)

    print("Running Test 3: Bootstrap CI for CV...")
    test3 = run_bootstrap_cv(results_dir)

    return {
        "test1_mcnemar":    test1,
        "test2_ttest_xpope":test2,
        "test3_bootstrap_cv":test3,
    }


def print_report(report: dict):
    # Test 1
    t1 = report["test1_mcnemar"]
    print(f"\n{'='*65}")
    print(f"TEST 1: {t1['test']}")
    print(f"{'='*65}")
    print(f"{'Model':<28} {'p-value':>10} {'Sig':>5} {'net gain':>10} {'POPE':>7} {'RePOPE':>8}")
    print(f"{'-'*65}")
    for model, res in t1["per_model"].items():
        if "error" in res:
            print(f"{model:<28} {'ERROR':>10}")
            continue
        sig = "✅" if res["significant"] else "❌"
        print(
            f"{model:<28} "
            f"{res['p_value']:>10.4f} "
            f"{sig:>5} "
            f"{res['net_gain_from_correction']:>+10} "
            f"{res['pope_acc']:>7.4f} "
            f"{res['repope_acc']:>8.4f}"
        )
    print(f"\nSignificant: {t1['n_significant']}/{t1['n_models']} models")

    # Test 2
    t2 = report["test2_ttest_xpope"]
    print(f"\n{'='*65}")
    print(f"TEST 2: {t2['test']}")
    print(f"{'='*65}")
    for model, pairs in t2["per_model"].items():
        print(f"\n  {model}")
        for pair_name, res in pairs.items():
            if "error" in res:
                continue
            sig = "✅" if res["significant"] else "❌"
            print(
                f"    {pair_name:<30} "
                f"diff={res['mean_diff']:>+7.4f}  "
                f"p={res['p_value']:.2e}  {sig}"
            )

    # Test 3
    t3 = report["test3_bootstrap_cv"]
    print(f"\n{'='*65}")
    print(f"TEST 3: {t3['test']}")
    print(f"{'='*65}")
    print(f"{'Split':<25} {'CV':>7} {'95% CI':>20} {'Saturated?'}")
    print(f"{'-'*65}")
    for split, res in t3["per_split"].items():
        if "error" in res:
            continue
        sat = "✅ YES" if res["saturated_with_high_confidence"] else "❌ no"
        ci_str = f"[{res['ci_lower']:.4f}, {res['ci_upper']:.4f}]"
        print(f"{split:<25} {res['observed_cv']:>7.4f} {ci_str:>20}  {sat}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="results/predictions")
    parser.add_argument("--output",      default="reports/significance.json")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    report = run(results_dir)
    print_report(report)

    with open(output_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nSaved → {output_path}")


if __name__ == "__main__":
    main()
