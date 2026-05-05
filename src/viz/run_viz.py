"""
run_viz.py — Generate all three paper figures.

Usage
-----
    python -m src.viz.run_viz
    python -m src.viz.run_viz --results results/predictions --out figures/

Outputs
-------
    figures/radar.png     — 6-model × 3-dimension radar chart (X-POPE)
    figures/bump.png      — POPE → RePOPE → DASH-B rank bump chart
    figures/bias_bar.png  — Yes-bias grouped bar chart
"""

from __future__ import annotations

import argparse
from pathlib import Path

from .radar    import plot_radar
from .bump     import plot_bump
from .bias_bar import plot_bias_bar


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate beyond-pope paper figures")
    parser.add_argument(
        "--results", default="results/predictions",
        help="Directory containing *_summary.json files",
    )
    parser.add_argument(
        "--out", default="figures",
        help="Output directory for PNG files",
    )
    args = parser.parse_args()

    results = Path(args.results)
    out     = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    print("\nGenerating figures...")
    plot_radar(   results, out / "radar.png")
    plot_bump(    results, out / "bump.png")
    plot_bias_bar(results, out / "bias_bar.png")
    print(f"\nDone — figures saved to ./{out}/")


if __name__ == "__main__":
    main()
