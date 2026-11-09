#!/usr/bin/env python3
"""Collect per-run JSON results into one CSV after the simulation jobs finish."""
import argparse
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".jupyter-sing/matplotlib"))

from sing_experiments.simulation_common import DEFAULT_OUTPUT, collect_results


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--run-label", default="default")
    args = parser.parse_args(argv)
    results = collect_results(args.output_dir, args.run_label)
    if results.empty:
        print("No results yet.")
        return 0
    destination = args.output_dir / f"metrics_{args.run_label}.csv"
    results.to_csv(destination, index=False)
    print(results.groupby(["model", "status"]).size().to_string())
    print(f"Saved {len(results)} runs to {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
