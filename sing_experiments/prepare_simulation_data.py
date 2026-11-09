#!/usr/bin/env python3
"""Optionally create/audit the paired dataset cache before fitting either model."""
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".jupyter-sing/matplotlib"))

from sing_experiments.simulation_common import common_parser, get_dataset, setup_runner


def main(argv=None):
    args = common_parser("Generate paired 10D R-vine train/test data, without PIT").parse_args(argv)
    study, cases = setup_runner(args)
    for case in cases:
        if args.list_only:
            print(case.name)
        else:
            _, metadata = get_dataset(args.output_dir, study, case)
            print(f"READY {case.name}: {metadata['dataset_hash'][:12]}", flush=True)
    print(f"{len(cases)} datasets; {study['n_test']} independent test observations per dataset.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
