#!/usr/bin/env python3
"""Fit only degree-3 SING MAGIC densities and save their classification analysis."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sing_experiments.run_magic_classification import main as run_magic


def main(argv=None):
    return run_magic(argv, fixed_order=3)


if __name__ == '__main__':
    raise SystemExit(main())
