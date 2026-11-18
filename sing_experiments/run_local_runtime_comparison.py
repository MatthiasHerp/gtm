#!/usr/bin/env python3
"""CPU runtime comparison: degree-2 SING versus the full tuned GTM protocol.

From the repository root, in the GTM Python environment:
    python sing_experiments/run_local_runtime_comparison.py
    python sing_experiments/run_local_runtime_comparison.py --seeds 1 2 3

SING defaults to .venv-sing/bin/python; GTM defaults to this Python. Override
with --sing-python /path/to/python and --gtm-python /path/to/python.
Only runtimes and identifying metadata are retained, in runtimes.csv. Shared
datasets are cached separately using the existing simulation-study machinery.
"""
from __future__ import annotations

import argparse
from contextlib import redirect_stderr, redirect_stdout
import csv
import fcntl
import os
from pathlib import Path
import random
import subprocess
import sys
import tempfile
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".jupyter-sing/matplotlib"))

FIELDS = ("model", "n_train", "marginals", "seed", "total_runtime_seconds",
          "status", "error", "device", "threads", "degree", "n_trials",
          "dataset_hash", "model_config_hash", "python")


def parser():
    result = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    result.add_argument("--seeds", type=int, nargs="+", default=[1])
    result.add_argument("--n-train", type=int, nargs="+", choices=[125, 500, 2000], default=[125, 500, 2000])
    result.add_argument("--models", nargs="+", choices=["sing", "gtm"], default=["sing", "gtm"])
    result.add_argument("--threads", type=int, default=1, help="BLAS/OpenMP and Torch CPU threads (default: 1)")
    result.add_argument("--sing-python", type=Path, default=ROOT / ".venv-sing/bin/python")
    result.add_argument("--gtm-python", type=Path, default=Path(sys.executable))
    result.add_argument("--sing-dir", type=Path, default=ROOT / ".external/SING")
    result.add_argument("--output-dir", type=Path, default=HERE / "results/local_runtime_comparison")
    result.add_argument("--data-dir", type=Path, default=HERE / "results/sing_gtm_comparison",
                        help="Existing paired simulation dataset cache; data preparation is not timed")
    result.add_argument("--list-only", action="store_true", help="Show jobs without importing model dependencies")
    result.add_argument("--worker", choices=["sing", "gtm"], help=argparse.SUPPRESS)
    return result


def append_result(path, row):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+", newline="") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        handle.seek(0, os.SEEK_END)
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        if handle.tell() == 0:
            writer.writeheader()
        writer.writerow(row)
        handle.flush()


def run_worker(args):
    """Prepare shared data and imports before timing the complete algorithm."""
    import numpy as np
    from sing_experiments.simulation_common import (
        DEFAULT_STUDY, Case, fingerprint, get_dataset, read_json, select_cases,
        validate_study,
    )

    study = read_json(DEFAULT_STUDY)
    validate_study(study)
    select_cases(study, args.n_train, ["gauss"], args.seeds)
    case = Case(args.n_train[0], "gauss", args.seeds[0])
    config = read_json(HERE / f"{args.worker}_simulation_config.json")
    row = dict(model="SING degree 2" if args.worker == "sing" else "GTM",
               n_train=case.n_train, marginals="gauss", seed=case.seed,
               device="cpu", threads=args.threads,
               degree=2 if args.worker == "sing" else "",
               n_trials=config["n_trials"] if args.worker == "gtm" else "",
               model_config_hash=fingerprint(config), python=sys.executable,
               dataset_hash="", total_runtime_seconds="", status="failed", error="")
    try:
        data, metadata = get_dataset(args.data_dir, study, case)
        row["dataset_hash"] = metadata["dataset_hash"]
        if args.worker == "sing":
            from sing_experiments.sing_helpers import FixedOrdering, load_sing
            # Load upstream modules before timing, as Torch/GTM imports also
            # occur before timing. All optimization and SING derivatives count.
            algorithm, _ = load_sing(
                batch_size=config["batch_size"], optimizer_derivatives=config["optimizer_derivatives"],
                optimizer_maxiter=config["optimizer_maxiter"], retain_density=False,
                progress=config.get("progress", True), sing_directory=args.sing_dir,
            )
            np.random.seed(metadata["seed_streams"]["fit"])
            random.seed(metadata["seed_streams"]["fit"])
        else:
            import torch
            from sing_experiments.simulation_gtm import fit_gtm, validate_gtm_config
            validate_gtm_config(config)
            torch.set_num_threads(args.threads)
        # Both libraries write optimizer diagnostics. Discard these with all
        # temporary GTM tuning files after the timed run; retain only the CSV.
        with tempfile.TemporaryDirectory(prefix="gtm-sing-runtime-") as temporary:
            scratch = Path(temporary)
            with (scratch / "run.log").open("w") as log, redirect_stdout(log), redirect_stderr(log):
                started = time.perf_counter()
                if args.worker == "sing":
                    algorithm(data["x_train"].copy(), 2, FixedOrdering(), config["delta"])
                else:
                    fit_gtm(data, metadata, scratch, config, "cpu", runtime_only=True)
                row.update(total_runtime_seconds=time.perf_counter() - started, status="ok")
    except (Exception, KeyboardInterrupt) as error:
        row["status"] = "interrupted" if isinstance(error, KeyboardInterrupt) else "failed"
        row["error"] = f"{type(error).__name__}: {error}"
        # A partial failed run must never look like a successful runtime.
        traceback.print_exc()
        if isinstance(error, KeyboardInterrupt):
            append_result(args.output_dir / "runtimes.csv", row)
            return 130
    append_result(args.output_dir / "runtimes.csv", row)
    duration = f"{row['total_runtime_seconds']:.2f}s" if row["status"] == "ok" else row["error"]
    print(f"{row['status'].upper()} {row['model']} N={case.n_train} seed={case.seed}: {duration}", flush=True)
    return int(row["status"] != "ok")


def main(argv=None):
    arg_parser = parser()
    args = arg_parser.parse_args(argv)
    if args.threads < 1 or any(seed < 1 or seed > 10 for seed in args.seeds):
        arg_parser.error("threads must be positive; seeds must be in the study grid 1–10")
    if any(len(values) != len(set(values)) for values in (args.seeds, args.n_train, args.models)):
        arg_parser.error("seeds, sample sizes, and models must not contain duplicates")
    if args.worker:
        if len(args.seeds) != 1 or len(args.n_train) != 1:
            arg_parser.error("an internal worker requires one seed and one sample size")
        return run_worker(args)
    jobs = [(n, seed, model) for n in args.n_train for seed in args.seeds for model in args.models]
    if args.list_only:
        for n, seed, model in jobs:
            print(f"{model.upper()} N={n} seed={seed} marginals=gauss device=cpu"
                  + (" degree=2" if model == "sing" else " trials=30 + pilots/refits"))
        print(f"{len(jobs)} sequential fits; {args.threads} thread(s); output: {args.output_dir / 'runtimes.csv'}")
        return 0

    # Enforce equal thread budgets before either Python imports numeric libraries.
    environment = dict(os.environ)
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                 "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS", "BLIS_NUM_THREADS"):
        environment[name] = str(args.threads)
    # Check both interpreters before launching a potentially long SING fit.
    for model in args.models:
        executable = getattr(args, f"{model}_python").absolute()
        imports = ("import numpy, pandas, scipy, sklearn, pyvinecopulib, TransportMaps" if model == "sing"
                   else "import torch, optuna, numpy, pandas, scipy, sklearn, pyvinecopulib; from gtm import GTM")
        try:
            check = subprocess.run([str(executable), "-c", imports], cwd=ROOT, env=environment,
                                   capture_output=True, text=True)
        except OSError as error:
            arg_parser.error(f"Cannot start {model} Python {executable}: {error}")
        if check.returncode:
            arg_parser.error(f"{model.upper()} dependencies unavailable in {executable}. "
                             f"Set --{model}-python to its environment's Python.\n{check.stderr}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    print("Timing fitting + training-data graph estimation; excluding data preparation/imports, "
          "test likelihoods and accuracy evaluation. Each invocation appends fresh measurements.", flush=True)
    failed = 0
    for n, seed, model in jobs:
        print(f"START {model.upper()} N={n} seed={seed}", flush=True)
        command = [str(getattr(args, f"{model}_python").absolute()), "-u", str(Path(__file__).resolve()),
                   "--worker", model, "--n-train", str(n), "--seeds", str(seed),
                   "--threads", str(args.threads), "--sing-dir", str(args.sing_dir.absolute()),
                   "--data-dir", str(args.data_dir.absolute()), "--output-dir", str(args.output_dir.absolute())]
        try:
            completed = subprocess.run(command, cwd=ROOT, env=environment)
        except KeyboardInterrupt:
            return 130
        if completed.returncode in (130, -2):
            return 130
        failed += completed.returncode != 0
    print(f"Saved: {args.output_dir / 'runtimes.csv'} ({failed} failed jobs)", flush=True)
    return int(failed > 0)


if __name__ == "__main__":
    raise SystemExit(main())
