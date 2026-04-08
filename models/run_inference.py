"""
Main inference script. Runs one model on one or more benchmarks.

Usage examples:

  # Single model, single benchmark
  python run_inference.py \
      --model qwen2vl_7b \
      --benchmark pope_adversarial \
      --image-dir data/raw/coco/val2014

  # Single model, all benchmarks
  python run_inference.py \
      --model internvl2_8b \
      --benchmark all \
      --image-dir data/raw/coco/val2014

  # All models, all benchmarks (full sweep)
  python run_inference.py \
      --model all \
      --benchmark all \
      --image-dir data/raw/coco/val2014

  # With 4-bit quantization (saves ~8GB VRAM)
  python run_inference.py \
      --model qwen2vl_7b \
      --benchmark xpope_attribute \
      --image-dir data/raw/coco/val2014 \
      --load-in-4bit
"""

import argparse
import json
import time
from pathlib import Path

from src.models import MODEL_REGISTRY, get_model


# ── Benchmark definitions ─────────────────────────────────────────────────────
# Maps benchmark name → JSONL file path (relative to project root)

BENCHMARKS = {
    # Existing benchmarks
    "pope_random":       "data/raw/benchmarks/pope/pope_random.jsonl",
    "pope_popular":      "data/raw/benchmarks/pope/pope_popular.jsonl",
    "pope_adversarial":  "data/raw/benchmarks/pope/pope_adversarial.jsonl",
    "repope_random":     "data/raw/benchmarks/repope/repope_random.jsonl",
    "repope_popular":    "data/raw/benchmarks/repope/repope_popular.jsonl",
    "repope_adversarial":"data/raw/benchmarks/repope/repope_adversarial.jsonl",
    "dashb":             "data/raw/benchmarks/dashb/dashb.jsonl",
    # Our new benchmark
    "xpope_existence":   "data/processed/xpope_existence.jsonl",
    "xpope_attribute":   "data/processed/xpope_attribute.jsonl",
    "xpope_relation":    "data/processed/xpope_relation.jsonl",
}


def run_one(
    model_name: str,
    benchmark_name: str,
    image_dir: Path,
    output_dir: Path,
    model_kwargs: dict,
) -> dict | None:
    """Run a single (model, benchmark) pair. Returns summary dict or None on skip."""

    benchmark_path = Path(BENCHMARKS[benchmark_name])
    output_path = output_dir / f"{model_name}_{benchmark_name}.jsonl"
    summary_path = output_dir / f"{model_name}_{benchmark_name}_summary.json"

    # Skip if already done
    if output_path.exists() and summary_path.exists():
        print(f"  [skip] {model_name} × {benchmark_name} (already exists)")
        with open(summary_path) as f:
            return json.load(f)

    if not benchmark_path.exists():
        print(f"  [warn] Benchmark file not found: {benchmark_path}")
        return None

    print(f"\n{'='*60}")
    print(f"  Model:     {model_name}")
    print(f"  Benchmark: {benchmark_name}")
    print(f"  Output:    {output_path}")
    print(f"{'='*60}")

    model = get_model(model_name, **model_kwargs)

    t0 = time.time()
    summary = model.evaluate_file(
        input_path=benchmark_path,
        image_dir=image_dir,
        output_path=output_path,
    )
    summary["wall_time_s"] = round(time.time() - t0, 1)

    # Save summary
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n  acc={summary['accuracy']:.3f}  "
          f"yes_rate={summary['yes_rate']:.3f}  "
          f"unknown={summary['n_unknown']}  "
          f"time={summary['wall_time_s']}s")

    # Free GPU memory between runs
    del model
    import torch, gc
    gc.collect()
    torch.cuda.empty_cache()

    return summary


def main():
    parser = argparse.ArgumentParser(description="VLM hallucination inference")

    parser.add_argument(
        "--model",
        default="qwen2vl_7b",
        help=f"Model name or 'all'. Available: {list(MODEL_REGISTRY.keys())}",
    )
    parser.add_argument(
        "--benchmark",
        default="pope_adversarial",
        help=f"Benchmark name or 'all'. Available: {list(BENCHMARKS.keys())}",
    )
    parser.add_argument(
        "--image-dir",
        default="data/raw/coco/val2014",
        help="Directory containing COCO images",
    )
    parser.add_argument(
        "--output-dir",
        default="results/predictions",
        help="Directory to write prediction JSONL files",
    )
    parser.add_argument(
        "--load-in-4bit",
        action="store_true",
        help="Load model in 4-bit quantization (saves VRAM)",
    )
    parser.add_argument(
        "--load-in-8bit",
        action="store_true",
        help="Load model in 8-bit quantization",
    )

    args = parser.parse_args()

    # Resolve model list
    models = list(MODEL_REGISTRY.keys()) if args.model == "all" else [args.model]

    # Resolve benchmark list
    benchmarks = list(BENCHMARKS.keys()) if args.benchmark == "all" else [args.benchmark]

    # Extra kwargs for model constructors
    model_kwargs = {}
    if args.load_in_4bit:
        model_kwargs["load_in_4bit"] = True
    if args.load_in_8bit:
        model_kwargs["load_in_8bit"] = True

    image_dir = Path(args.image_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nRunning {len(models)} model(s) × {len(benchmarks)} benchmark(s) "
          f"= {len(models) * len(benchmarks)} jobs\n")

    all_summaries = []
    for model_name in models:
        for benchmark_name in benchmarks:
            summary = run_one(
                model_name=model_name,
                benchmark_name=benchmark_name,
                image_dir=image_dir,
                output_dir=output_dir,
                model_kwargs=model_kwargs,
            )
            if summary:
                all_summaries.append(summary)

    # Print final table
    if all_summaries:
        print(f"\n{'='*60}")
        print(f"{'Model':<20} {'Benchmark':<25} {'Acc':>6} {'Yes%':>6} {'Unk':>5}")
        print(f"{'-'*20} {'-'*25} {'-'*6} {'-'*6} {'-'*5}")
        for s in all_summaries:
            print(
                f"{s['model']:<20} {s['benchmark']:<25} "
                f"{s['accuracy']:>6.3f} {s['yes_rate']:>6.3f} {s['n_unknown']:>5}"
            )
        print(f"{'='*60}\n")

        # Save combined summary
        combined_path = output_dir / "_all_summaries.json"
        with open(combined_path, "w") as f:
            json.dump(all_summaries, f, indent=2)
        print(f"Combined summary saved to {combined_path}")


if __name__ == "__main__":
    main()
