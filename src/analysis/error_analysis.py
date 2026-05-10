"""
error_analysis.py
-----------------
Analyze error patterns in VLM predictions across benchmarks and dimensions.

Error types
-----------
  False Positive (FP) : model answers "yes" but ground truth is "no"
                        → model hallucinates an object/attribute/relation
  False Negative (FN) : model answers "no" but ground truth is "yes"
                        → model misses an existing object/attribute/relation

Key analyses
------------
  1. FP/FN breakdown per model per benchmark
  2. Most confused object classes (for existence)
  3. Most confused attribute types (for attribute)
  4. Most confused relation types (for relation)
  5. Per-model error consistency (do models fail on the same questions?)
  6. Hard samples: questions that ≥ N models answer wrong

Usage
-----
    python -m src.analysis.error_analysis \
        --results-dir results/predictions \
        --output reports/error_analysis.json

    # Only X-POPE dimensions
    python -m src.analysis.error_analysis \
        --results-dir results/predictions \
        --benchmarks xpope_existence xpope_attribute xpope_relation \
        --output reports/error_analysis_xpope.json
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

MODELS = [
    "qwen2vl_7b", "internvl2_8b", "llava_ov_7b",
    "llama32v_11b", "paligemma2_3b", "phi35v_4b",
]

DEFAULT_BENCHMARKS = [
    "pope_adversarial",
    "xpope_existence",
    "xpope_attribute",
    "xpope_relation",
]


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

def load_predictions(results_dir: Path, model: str, benchmark: str) -> list[dict]:
    path = results_dir / f"{model}_{benchmark}.jsonl"
    if not path.exists():
        return []
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


# ---------------------------------------------------------------------------
# Error type classification
# ---------------------------------------------------------------------------

def classify_errors(preds: list[dict]) -> dict:
    """
    Classify predictions into TP / FP / TN / FN.

    FP: predicted yes, label no  → hallucination
    FN: predicted no,  label yes → miss
    """
    tp = fp = tn = fn = 0
    fp_samples = []
    fn_samples = []

    for r in preds:
        pred  = r.get("answer", "unknown").lower()
        label = r.get("label",  r.get("ground_truth", "")).lower()

        if pred == "unknown":
            continue

        if pred == "yes" and label == "yes":
            tp += 1
        elif pred == "yes" and label == "no":
            fp += 1
            fp_samples.append(r)
        elif pred == "no" and label == "no":
            tn += 1
        elif pred == "no" and label == "yes":
            fn += 1
            fn_samples.append(r)

    total = tp + fp + tn + fn
    return {
        "TP": tp, "FP": fp, "TN": tn, "FN": fn,
        "total": total,
        "accuracy":     round((tp + tn) / total, 4) if total else 0,
        "fp_rate":      round(fp / (fp + tn), 4) if (fp + tn) else 0,
        "fn_rate":      round(fn / (fn + tp), 4) if (fn + tp) else 0,
        "hallucination_rate": round(fp / total, 4) if total else 0,
        "miss_rate":    round(fn / total, 4) if total else 0,
        "fp_samples":   fp_samples,
        "fn_samples":   fn_samples,
    }


# ---------------------------------------------------------------------------
# Analysis 1: FP/FN breakdown per model per benchmark
# ---------------------------------------------------------------------------

def fp_fn_breakdown(results_dir: Path, benchmarks: list[str]) -> dict:
    """Per-model FP/FN rates across benchmarks."""
    result = {}
    for model in MODELS:
        model_result = {}
        for bm in benchmarks:
            preds = load_predictions(results_dir, model, bm)
            if not preds:
                continue
            err = classify_errors(preds)
            model_result[bm] = {
                "accuracy":           err["accuracy"],
                "fp_rate":            err["fp_rate"],
                "fn_rate":            err["fn_rate"],
                "hallucination_rate": err["hallucination_rate"],
                "miss_rate":          err["miss_rate"],
                "TP": err["TP"], "FP": err["FP"],
                "TN": err["TN"], "FN": err["FN"],
            }
        result[model] = model_result
    return result


# ---------------------------------------------------------------------------
# Analysis 2: Most confused object/attribute/relation classes
# ---------------------------------------------------------------------------

def _count_field(samples: list[dict], field: str) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for s in samples:
        val = s.get(field, "unknown")
        if val:
            counts[val] += 1
    return dict(sorted(counts.items(), key=lambda x: -x[1]))


def confused_classes(results_dir: Path) -> dict:
    """
    For each benchmark, find which object/attribute/relation classes
    cause the most FP and FN errors across all models.
    """
    configs = {
        "xpope_existence": ("object",       "existence"),
        "xpope_attribute": ("attribute",    "attribute"),
        "xpope_relation":  ("relation",     "relation"),
        "pope_adversarial":("object",       "existence"),
    }

    result = {}
    for bm, (class_field, dim) in configs.items():
        all_fp: list[dict] = []
        all_fn: list[dict] = []

        for model in MODELS:
            preds = load_predictions(results_dir, model, bm)
            if not preds:
                continue
            err = classify_errors(preds)
            all_fp.extend(err["fp_samples"])
            all_fn.extend(err["fn_samples"])

        result[bm] = {
            "dimension":       dim,
            "class_field":     class_field,
            "top_fp_classes":  dict(list(_count_field(all_fp, class_field).items())[:15]),
            "top_fn_classes":  dict(list(_count_field(all_fn, class_field).items())[:15]),
            "total_fp":        len(all_fp),
            "total_fn":        len(all_fn),
        }

    return result


# ---------------------------------------------------------------------------
# Analysis 3: Hard samples (≥ N models wrong)
# ---------------------------------------------------------------------------

def hard_samples(
    results_dir: Path,
    benchmarks: list[str],
    min_models_wrong: int = 4,
) -> dict:
    """
    Find questions that at least `min_models_wrong` models answer incorrectly.
    These are the genuinely hard cases for all VLMs.
    """
    result = {}
    for bm in benchmarks:
        # question_id → list of (model, correct)
        q_results: dict[str, list] = defaultdict(list)

        for model in MODELS:
            preds = load_predictions(results_dir, model, bm)
            for r in preds:
                qid = r.get("question_id")
                if qid and r.get("correct") is not None:
                    q_results[qid].append({
                        "model":   model,
                        "correct": r["correct"],
                        "answer":  r.get("answer"),
                        "label":   r.get("label", r.get("ground_truth")),
                    })

        hard = []
        for qid, model_results in q_results.items():
            n_wrong = sum(1 for m in model_results if not m["correct"])
            if n_wrong >= min_models_wrong:
                # grab the question text from the first result
                sample_pred = next(
                    (r for model in MODELS
                     for r in load_predictions(results_dir, model, bm)
                     if r.get("question_id") == qid),
                    {}
                )
                hard.append({
                    "question_id":    qid,
                    "question":       sample_pred.get("question", ""),
                    "label":          sample_pred.get("label", sample_pred.get("ground_truth", "")),
                    "image":          sample_pred.get("image", ""),
                    "n_models_wrong": n_wrong,
                    "n_models_total": len(model_results),
                    "model_answers":  {m["model"]: m["answer"] for m in model_results},
                })

        hard.sort(key=lambda x: -x["n_models_wrong"])
        result[bm] = {
            "min_models_wrong": min_models_wrong,
            "n_hard_samples":   len(hard),
            "hard_samples":     hard[:50],  # top 50
        }

    return result


# ---------------------------------------------------------------------------
# Analysis 4: Error consistency across models
# ---------------------------------------------------------------------------

def error_consistency(results_dir: Path, benchmarks: list[str]) -> dict:
    """
    For each benchmark, compute the Jaccard similarity of error sets
    between each pair of models.

    High Jaccard → models fail on the same questions (systematic difficulty)
    Low Jaccard  → models fail on different questions (idiosyncratic errors)
    """
    result = {}
    for bm in benchmarks:
        # model → set of wrong question_ids
        model_errors: dict[str, set] = {}
        for model in MODELS:
            preds = load_predictions(results_dir, model, bm)
            if not preds:
                continue
            model_errors[model] = {
                r["question_id"]
                for r in preds
                if r.get("correct") is False and r.get("question_id")
            }

        models_with_data = list(model_errors.keys())
        pairwise = {}

        for i, m1 in enumerate(models_with_data):
            for m2 in models_with_data[i+1:]:
                s1, s2 = model_errors[m1], model_errors[m2]
                union = len(s1 | s2)
                jaccard = len(s1 & s2) / union if union > 0 else 0.0
                pairwise[f"{m1}__{m2}"] = round(jaccard, 4)

        # Average Jaccard across all pairs
        avg_jaccard = round(float(np.mean(list(pairwise.values()))), 4) if pairwise else 0.0

        result[bm] = {
            "pairwise_jaccard": pairwise,
            "avg_jaccard":      avg_jaccard,
            "interpretation":   (
                "high systematic difficulty" if avg_jaccard > 0.4
                else "mixed" if avg_jaccard > 0.25
                else "model-specific errors"
            ),
        }

    return result


# ---------------------------------------------------------------------------
# Full report
# ---------------------------------------------------------------------------

def run(
    results_dir: Path,
    benchmarks: list[str] | None = None,
    min_models_wrong: int = 4,
) -> dict:
    if benchmarks is None:
        benchmarks = DEFAULT_BENCHMARKS

    print("Analysis 1: FP/FN breakdown...")
    breakdown = fp_fn_breakdown(results_dir, benchmarks)

    print("Analysis 2: Confused classes...")
    confused = confused_classes(results_dir)

    print("Analysis 3: Hard samples...")
    hard = hard_samples(results_dir, benchmarks, min_models_wrong)

    print("Analysis 4: Error consistency...")
    consistency = error_consistency(results_dir, benchmarks)

    return {
        "benchmarks":       benchmarks,
        "models":           MODELS,
        "fp_fn_breakdown":  breakdown,
        "confused_classes": confused,
        "hard_samples":     hard,
        "error_consistency":consistency,
    }


def print_report(report: dict):
    # FP/FN summary table
    print(f"\n{'='*70}")
    print("FP/FN BREAKDOWN")
    print(f"{'='*70}")

    for bm in report["benchmarks"]:
        print(f"\n── {bm} ──")
        print(f"{'Model':<28} {'Acc':>6} {'FP%':>6} {'FN%':>6}  FP=hallucinate  FN=miss")
        print(f"{'-'*70}")
        for model in report["models"]:
            d = report["fp_fn_breakdown"].get(model, {}).get(bm)
            if not d:
                continue
            print(
                f"{model:<28} "
                f"{d['accuracy']:>6.4f} "
                f"{d['fp_rate']:>6.4f} "
                f"{d['fn_rate']:>6.4f}  "
                f"FP={d['FP']:>4}  FN={d['FN']:>4}"
            )

    # Confused classes
    print(f"\n{'='*70}")
    print("TOP CONFUSED CLASSES (aggregated across all models)")
    print(f"{'='*70}")
    for bm, data in report["confused_classes"].items():
        print(f"\n── {bm} ({data['dimension']}) ── field: {data['class_field']}")
        fp_top = list(data["top_fp_classes"].items())[:8]
        fn_top = list(data["top_fn_classes"].items())[:8]
        print(f"  FP (hallucinated): {', '.join(f'{k}({v})' for k, v in fp_top)}")
        print(f"  FN (missed):       {', '.join(f'{k}({v})' for k, v in fn_top)}")

    # Hard samples
    print(f"\n{'='*70}")
    print("HARD SAMPLES (≥ 4 models wrong)")
    print(f"{'='*70}")
    for bm, data in report["hard_samples"].items():
        print(f"\n── {bm}: {data['n_hard_samples']} hard samples ──")
        for s in data["hard_samples"][:5]:
            print(f"  [{s['n_models_wrong']}/6 wrong] {s['question'][:80]}")

    # Error consistency
    print(f"\n{'='*70}")
    print("ERROR CONSISTENCY (Jaccard similarity of error sets)")
    print(f"{'='*70}")
    print(f"{'Benchmark':<30} {'Avg Jaccard':>12}  Interpretation")
    print(f"{'-'*70}")
    for bm, data in report["error_consistency"].items():
        print(
            f"{bm:<30} "
            f"{data['avg_jaccard']:>12.4f}  "
            f"{data['interpretation']}"
        )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir",      default="results/predictions")
    parser.add_argument("--benchmarks",       nargs="*", default=None)
    parser.add_argument("--min-models-wrong", type=int,  default=4)
    parser.add_argument("--output",           default="reports/error_analysis.json")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    report = run(results_dir, args.benchmarks, args.min_models_wrong)
    print_report(report)

    # Strip raw samples from JSON output to keep file size manageable
    report_json = {
        k: v for k, v in report.items()
        if k != "hard_samples"
    }
    # Keep only top-20 hard samples per benchmark
    report_json["hard_samples"] = {
        bm: {
            "n_hard_samples": d["n_hard_samples"],
            "min_models_wrong": d["min_models_wrong"],
            "hard_samples": [
                {k: v for k, v in s.items() if k != "model_answers"}
                for s in d["hard_samples"][:20]
            ],
        }
        for bm, d in report["hard_samples"].items()
    }

    with open(output_path, "w") as f:
        json.dump(report_json, f, indent=2)
    print(f"\nSaved → {output_path}")


if __name__ == "__main__":
    main()
