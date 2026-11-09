#!/usr/bin/env python3
"""Run the CPU SING grid, sharing cached data with run_gtm_simulation.py."""
from __future__ import annotations

import os
import importlib.util
from pathlib import Path
import random
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".jupyter-sing/matplotlib"))

import numpy as np

from sing_experiments.sing_helpers import FixedOrdering, SING_COMMIT, graph_metrics, load_sing
from sing_experiments.simulation_common import (
    HERE, atomic_json, common_parser, ensure_config, likelihood_metrics,
    read_json, run_case, setup_runner, to_original_order, versions,
)


def fit_sing(data, metadata, directory, degree, config):
    np.random.seed(metadata["seed_streams"]["fit"])
    random.seed(metadata["seed_streams"]["fit"])
    started = time.perf_counter()
    sing, history = load_sing(
        batch_size=config["batch_size"], optimizer_derivatives=config["optimizer_derivatives"],
        optimizer_maxiter=config["optimizer_maxiter"], retain_density=True,
        progress=config.get("progress", True),
    )
    omega_model = sing(data["x_train"].copy(), degree, FixedOrdering(), config["delta"])
    fit_seconds = time.perf_counter() - started
    density = history[-1]["density"]

    def logpdf(points):
        batch = config["loglik_batch_size"]
        return np.concatenate([np.asarray(density.log_pdf(points[i:i + batch])).reshape(-1)
                               for i in range(0, len(points), batch)])

    likelihood, log_arrays = likelihood_metrics(logpdf(data["x_train"]), logpdf(data["x_test"]), data["scale"])
    order = data["variable_order"]
    omega = to_original_order(omega_model, order)
    initial_scores = to_original_order(history[0]["omega_raw"], order)
    final_scores = to_original_order(history[-1]["omega_raw"], order)
    adjacency = omega != 0
    np.fill_diagonal(adjacency, False)
    metrics = graph_metrics(data["truth"], adjacency, initial_scores)
    final_metrics = graph_metrics(data["truth"], adjacency, final_scores)
    diagnostics = [[dict(success=bool(log["success"]),
                        accepted_by_gradient=bool(log["accepted_by_gradient"]),
                        message=str(log["message"]), iterations=int(log["nit"]),
                        gradient_inf_norm=float(np.linalg.norm(log["jac"], ord=np.inf)))
                    for log in iteration["solver_logs"]] for iteration in history]
    atomic_json(directory / "solver_diagnostics.json", diagnostics)
    metrics.update(likelihood, device="cpu", degree=degree, sing_iterations=len(history),
                   fit_seconds=fit_seconds, score_metric="initial_dense_squared_hessian",
                   threshold_method="SING_delta_method", delta=config["delta"],
                   roc_auc_final_raw=final_metrics["roc_auc"],
                   versions=versions(["TransportMaps", "numpy", "scipy", "pyvinecopulib"]),
                   sing_commit=SING_COMMIT, dtype="float64")
    artifacts = dict(truth=data["truth"], scores=initial_scores, final_scores=final_scores,
                     adjacency=adjacency, thresholded_scores=omega,
                     variable_order=order, **log_arrays)
    return metrics, artifacts


def main(argv=None):
    parser = common_parser("SING comparison study (CPU; pinned upstream has no GPU backend)")
    parser.add_argument("--model-config", type=Path, default=HERE / "sing_simulation_config.json")
    parser.add_argument("--degrees", type=int, nargs="+", choices=(2, 3))
    args = parser.parse_args(argv)
    study, cases = setup_runner(args)
    config = read_json(args.model_config)
    degrees = study["sing_degrees"] if args.degrees is None else args.degrees
    if not set(degrees) <= set(study["sing_degrees"]):
        parser.error("Requested degrees are outside the study grid")
    for key in ("batch_size", "optimizer_maxiter", "loglik_batch_size"):
        if config[key] < 1:
            parser.error(f"{key} must be positive")
    print(f"Selected {len(cases) * len(degrees)} SING fits on CPU; no PIT/rank transformation.")
    if not args.list_only:
        if importlib.util.find_spec("TransportMaps") is None:
            parser.error("TransportMaps is missing; use the .venv-sing Python environment")
        if not (ROOT / ".external/SING/SING/SparsityIdentificationNonGaussian.py").exists():
            parser.error("Pinned upstream SING checkout is missing; see sing_experiments/README.md")
    failed = 0
    for degree in degrees:
        profile = args.output_dir / "runs/sing" / args.run_label / f"degree{degree}"
        if not args.list_only:
            ensure_config(profile / "config.json", dict(config, degree=degree, study=study))
        for case in cases:
            if args.list_only:
                print(f"SING degree {degree}: {case.name}")
                continue
            ok = run_case(profile / case.name, case, study, args, f"SING degree {degree}", config,
                          lambda data, meta, directory: fit_sing(data, meta, directory, degree, config))
            failed += not ok
    return int(failed > 0)


if __name__ == "__main__":
    raise SystemExit(main())
