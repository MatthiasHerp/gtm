#!/usr/bin/env python3
"""Run MAGIC SING degrees 2 then 3 after shared training-only Gaussianization."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sing_experiments.run_magic_classification import main as run_magic


def main(argv=None):
    return run_magic(argv, nonparanormal_default=True)


if __name__ == '__main__':
    raise SystemExit(main())
