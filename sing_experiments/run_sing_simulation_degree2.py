#!/usr/bin/env python3
"""Original degree-2 SING grid; reuse and skip its completed default-label fits."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sing_experiments.run_sing_simulation import main as run_sing


def main(argv=None):
    return run_sing(argv, fixed_degree=2)


if __name__ == "__main__":
    raise SystemExit(main())
