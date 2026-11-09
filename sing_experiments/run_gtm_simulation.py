#!/usr/bin/env python3
"""Run the paper-configuration GTM comparison study on CPU or CUDA."""
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".jupyter-sing/matplotlib"))

from sing_experiments.simulation_common import (
    HERE, common_parser, ensure_config, read_json, run_case, setup_runner,
)


def main(argv=None):
    parser = common_parser(__doc__)
    parser.add_argument("--model-config", type=Path, default=HERE / "gtm_simulation_config.json")
    parser.add_argument("--device", default="auto", help="auto (CUDA when available), cpu, cuda, or cuda:N")
    parser.add_argument("--threads", type=int, default=1, help="Torch CPU thread count")
    args = parser.parse_args(argv)
    study, cases = setup_runner(args)
    config = read_json(args.model_config)
    if args.list_only:
        for case in cases:
            print(f"GTM: {case.name}")
        print(f"{len(cases)} GTM fits; {config['n_trials']} tuning trials per fit.")
        return 0
    # Lazy imports keep --help/--list-only available without a Torch environment.
    import torch
    from sing_experiments.simulation_gtm import fit_gtm, resolve_device, validate_gtm_config
    validate_gtm_config(config)
    if args.threads < 1:
        parser.error("threads must be positive")
    torch.set_num_threads(args.threads)
    device = resolve_device(args.device)
    profile = args.output_dir / "runs/gtm" / args.run_label
    ensure_config(profile / "config.json", dict(config, study=study, device=device, threads=args.threads))
    print(f"Selected {len(cases)} GTM fits on {device}; no PIT; threshold selection deferred.")
    failed = 0
    for case in cases:
        ok = run_case(profile / case.name, case, study, args, "GTM", config,
                      lambda data, meta, directory: fit_gtm(data, meta, directory, config, device))
        failed += not ok
    return int(failed > 0)


if __name__ == "__main__":
    raise SystemExit(main())
