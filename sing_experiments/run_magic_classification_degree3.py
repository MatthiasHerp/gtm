#!/usr/bin/env python3
"""Fit degree-3 MAGIC with all observations per derivative/likelihood call."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sing_experiments.run_magic_classification import main as run_magic


def main(argv=None):
    return run_magic(argv, fixed_order=3, defaults=dict(
        batch_size=None, logpdf_batch_size=None,
        output_dir=ROOT / 'sing_experiments/results/magic_classification_degree3_fullsample',
    ))


if __name__ == '__main__':
    raise SystemExit(main())
