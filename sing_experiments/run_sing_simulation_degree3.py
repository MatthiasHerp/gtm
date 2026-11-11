#!/usr/bin/env python3
"""Revised degree-3 grid: maxiter 300, seeds 1/2, N=2000 first then N=500."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sing_experiments.run_sing_simulation import main as run_sing


def main(argv=None):
    return run_sing(argv, fixed_degree=3, defaults=dict(
        model_config=ROOT / "sing_experiments/sing_simulation_config_degree3.json",
        n_train=[2000, 500], seeds=[1, 2], run_label="degree3_maxiter300",
    ))


if __name__ == "__main__":
    raise SystemExit(main())
