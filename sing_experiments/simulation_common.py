"""Paired datasets, original-scale likelihoods, and resumable study outputs.

This module deliberately imports neither Torch nor TransportMaps, so the SING
and GTM runners can use separate environments and share identical cached data.
"""
from __future__ import annotations

import argparse
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import fcntl
import importlib.metadata
import itertools
import json
import os
from pathlib import Path
import platform
import tempfile
import time
import traceback

import numpy as np
import pandas as pd

from sing_experiments.sing_helpers import (
    ROOT, _marginal_ppf, graph_metrics, make_rvine, marginal_cdf,
    oracle_mixed_hessians, reference_graphs,
)

HERE = Path(__file__).resolve().parent
DEFAULT_OUTPUT = HERE / "results/sing_gtm_comparison"
DEFAULT_STUDY = HERE / "simulation_study.json"


@dataclass(frozen=True)
class Case:
    n_train: int
    marginals: str
    seed: int

    @property
    def name(self):
        return f"n{self.n_train}_{self.marginals}_seed{self.seed:02d}"


def read_json(path):
    return json.loads(Path(path).read_text())


def _json_default(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def json_safe(value):
    """Use JSON null for undefined metrics, never non-standard NaN/Infinity."""
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [json_safe(v) for v in value]
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False) as f:
        temporary = Path(f.name)
        json.dump(json_safe(value), f, indent=2, default=_json_default, allow_nan=False)
        f.write("\n")
    os.replace(temporary, path)


def atomic_npz(path, **arrays):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as f:
        temporary = Path(f.name)
        np.savez_compressed(f, **arrays)
    os.replace(temporary, path)


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=_json_default).encode()).hexdigest()


def validate_study(config):
    if config["schema_version"] != 1:
        raise ValueError("Unsupported simulation-study schema")
    if config["dimensionality"] != 10 or config["active_trees"] != 3:
        raise ValueError("This study fixes dimensionality=10 and active_trees=3")
    if config["nonparanormal_transform"] is not False or config["shuffle_variables"] is not True:
        raise ValueError("This study requires shuffled variables and no PIT/rank transformation")
    if config["standardization"] != "training_mean_std_ddof0":
        raise ValueError("Use training-mean/std standardization")
    for field in ("train_sizes", "seeds", "sing_degrees"):
        values = config[field]
        if not values or len(values) != len(set(values)):
            raise ValueError(f"{field} must be a nonempty list without duplicates")
        if any(isinstance(v, bool) or not isinstance(v, int) for v in values):
            raise ValueError(f"{field} must contain integers")
    if min(config["train_sizes"]) < 2 or min(config["seeds"]) < 0:
        raise ValueError("Training sizes must be >=2 and seeds nonnegative")
    if not set(config["sing_degrees"]) <= {2, 3}:
        raise ValueError("SING degrees must be 2 and/or 3")
    if (not config["marginals"] or len(set(config["marginals"])) != len(config["marginals"])
            or not set(config["marginals"]) <= {"gauss", "exp", "t"}):
        raise ValueError("Study marginals must be gauss, exp, and/or t, without duplicates")
    if isinstance(config["n_test"], bool) or not isinstance(config["n_test"], int) or config["n_test"] < 2:
        raise ValueError("n_test must be an integer >=2")


def ensure_config(path, config):
    """Never silently resume a directory with different experimental settings."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.with_name(path.name + ".lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if path.exists():
            if read_json(path) != json_safe(config):
                raise ValueError(f"Configuration differs from {path}; use a new output directory/run label")
        else:
            atomic_json(path, config)


def select_cases(study, n_train=None, marginals=None, seeds=None):
    selected = []
    for supplied, key in ((n_train, "train_sizes"), (marginals, "marginals"), (seeds, "seeds")):
        values = study[key] if supplied is None else supplied
        if not set(values) <= set(study[key]):
            raise ValueError(f"Requested {key} are outside the study grid")
        selected.append(values)
    return [Case(*values) for values in itertools.product(*selected)]


def seed_streams(seed):
    names = ("train", "test", "shuffle", "fit", "validation")
    streams = np.random.SeedSequence(seed).spawn(len(names))
    # pyvinecopulib 0.7.3 accepts signed 32-bit seeds, not arbitrary uint32.
    return {name: int(stream.generate_state(1)[0]) % (2**31 - 1)
            for name, stream in zip(names, streams)}


def standardize(train, other):
    """Fit the affine transform on train alone; preserve original column labels."""
    center, scale = train.mean(axis=0), train.std(axis=0, ddof=0)
    if not np.isfinite(scale).all() or np.any(scale <= 0):
        raise ValueError("Training columns must have finite, positive standard deviations")
    return (train - center) / scale, (other - center) / scale, center, scale


def to_original_order(matrix, variable_order):
    inverse = np.argsort(variable_order)
    return np.asarray(matrix)[np.ix_(inverse, inverse)]


def likelihood_metrics(train_logpdf_standardized, test_logpdf_standardized, scale):
    """Densities are for original observations: log p_X = log p_Z - sum(log std)."""
    adjustment = np.log(scale).sum()
    result = {}
    arrays = {}
    for split, values in (("train", train_logpdf_standardized), ("test", test_logpdf_standardized)):
        logpdf = np.asarray(values, dtype=float).reshape(-1) - adjustment
        if not len(logpdf) or not np.isfinite(logpdf).all():
            raise FloatingPointError(f"Nonfinite/empty {split} log densities")
        result[f"{split}_loglik_mean"] = float(logpdf.mean())
        result[f"{split}_loglik_sum"] = float(logpdf.sum())
        arrays[f"{split}_logpdf"] = logpdf
    return result, arrays


def audit_reference(vine, train, pairs):
    spec = pairs.attrs["marginal_spec"]
    u = marginal_cdf(train, spec)
    central = train[np.all((u > 0.025) & (u < 0.975), axis=1)][:64]
    if len(central) < 16:
        raise ValueError("Fewer than 16 central observations for the oracle reference audit")
    maxima = np.stack([np.abs(oracle_mixed_hessians(vine, central, step=h,
                       marginal_spec=spec)).max(axis=0) for h in (1e-3, 5e-4)])
    detected = np.all(maxima > 1e-4, axis=0)
    np.fill_diagonal(detected, False)
    _, truth = reference_graphs(pairs, dim=train.shape[1])
    if not np.array_equal(detected, truth):
        raise RuntimeError("Oracle reference audit inconclusive; dataset not used for scoring")
    return maxima


def get_dataset(output, study, case):
    """Serialize cache generation across CPU/GPU processes and environments."""
    directory = Path(output) / "datasets" / case.name
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / "dataset.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        return _get_dataset_unlocked(output, study, case)


def _get_dataset_unlocked(output, study, case):
    """Generate once; train/test sample seeds are independent of vine/shuffle seeds."""
    output = Path(output)
    ensure_config(output / "study.json", study)
    directory = output / "datasets" / case.name
    meta_path, array_path = directory / "metadata.json", directory / "data.npz"
    if meta_path.exists() and array_path.exists():
        metadata = read_json(meta_path)
        if metadata["study_hash"] != fingerprint(study):
            raise ValueError("Cached dataset has a different study configuration")
        if metadata["seed_streams"] != seed_streams(case.seed):
            raise ValueError("Cached dataset uses a different seed scheme; use a new output directory")
        with np.load(array_path) as archive:
            return {key: archive[key] for key in archive.files}, metadata
    streams = seed_streams(case.seed)
    vine, train, pairs = make_rvine(
        n=case.n_train, copula_seed=case.seed, sample_seed=streams["train"],
        dimensionality=study["dimensionality"], active_trees=study["active_trees"],
        tau_mean=study["tau_mean"], tau_range=study["tau_range"], marginals=case.marginals,
    )
    spec = pairs.attrs["marginal_spec"]
    test = _marginal_ppf(vine.simulate(study["n_test"], seeds=[streams["test"]]), spec)
    z_train, z_test, center, scale = standardize(train, test)
    order = np.random.default_rng(streams["shuffle"]).permutation(study["dimensionality"])
    study_graph, truth = reference_graphs(pairs, dim=study["dimensionality"])
    arrays = dict(train=train, test=test, x_train=z_train[:, order], x_test=z_test[:, order],
                  center=center, scale=scale, variable_order=order, truth=truth,
                  study_graph=study_graph)
    if study["audit_reference"]:
        arrays["oracle_max"] = audit_reference(vine, train, pairs)
    checksum = hashlib.sha256(vine.to_json().encode())
    for key in sorted(arrays):
        checksum.update(key.encode())
        checksum.update(np.ascontiguousarray(arrays[key]).tobytes())
    metadata = dict(case=case.__dict__, study_hash=fingerprint(study),
                    dataset_hash=checksum.hexdigest(), seed_streams=streams,
                    marginal_spec=spec, variable_order_1based=(order + 1).tolist(),
                    n_test=study["n_test"], reference_audited=study["audit_reference"],
                    nonparanormal_transform=False, standardization=study["standardization"],
                    generator_version=importlib.metadata.version("pyvinecopulib"),
                    generator_versions=versions(["pyvinecopulib", "numpy", "scipy"]))
    directory.mkdir(parents=True, exist_ok=True)
    atomic_json(directory / "vine.json", json.loads(vine.to_json()))
    pairs.to_csv(directory / "vine_pairs.csv", index=False)
    atomic_npz(array_path, **arrays)
    atomic_json(meta_path, metadata)
    return arrays, metadata


def versions(packages):
    result = dict(python=platform.python_version())
    for package in packages:
        try:
            result[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            result[package] = "not installed"
    return result


def common_parser(description):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--study-config", type=Path, default=DEFAULT_STUDY)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--n-train", type=int, nargs="+")
    parser.add_argument("--marginals", nargs="+", choices=("gauss", "exp", "t"))
    parser.add_argument("--seeds", type=int, nargs="+")
    parser.add_argument("--run-label", default="default")
    parser.add_argument("--list-only", action="store_true", help="List selected jobs without fitting")
    return parser


def setup_runner(args):
    if not args.run_label or Path(args.run_label).name != args.run_label or args.run_label in (".", ".."):
        raise ValueError("run-label must be a simple directory name")
    study = read_json(args.study_config)
    validate_study(study)
    cases = select_cases(study, args.n_train, args.marginals, args.seeds)
    return study, cases


def run_case(directory, case, study, args, model, model_config, fit):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / "run.lock").open("a+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print(f"BUSY {model}: {case.name} (another process owns this fit)", flush=True)
            return True
        return _run_case_unlocked(directory, case, study, args, model, model_config, fit)


def _run_case_unlocked(directory, case, study, args, model, model_config, fit):
    """Independent result JSONs avoid concurrent writers sharing one CSV."""
    directory = Path(directory)
    result_path = directory / "result.json"
    if result_path.exists() and read_json(result_path)["status"] == "ok":
        print(f"SKIP {model}: {case.name}", flush=True)
        return True
    directory.mkdir(parents=True, exist_ok=True)
    row = dict(model=model, run_label=args.run_label, n_train=case.n_train,
               marginals=case.marginals, seed=case.seed, n_test=study["n_test"],
               nonparanormal_transform=False, model_config=model_config,
               study_hash=fingerprint(study), started_utc=datetime.now(timezone.utc).isoformat(),
               status="running", loglik_scale="original_observations")
    atomic_json(result_path, row)
    started = time.perf_counter()
    try:
        with (directory / "run.log").open("w") as log, redirect_stdout(log), redirect_stderr(log):
            data, metadata = get_dataset(args.output_dir, study, case)
            metrics, artifacts = fit(data, metadata, directory)
            required = ["roc_auc", "train_loglik_mean", "test_loglik_mean"]
            if model.startswith("SING"):
                required += ["precision", "recall", "f1", "mcc"]
            if not all(np.isfinite(float(metrics[key])) for key in required):
                raise FloatingPointError("Run produced nonfinite required metrics")
            atomic_npz(directory / "evaluation.npz", **artifacts)
        row.update(metrics, status="ok", dataset_hash=metadata["dataset_hash"],
                   variable_order_1based=metadata["variable_order_1based"])
        (directory / "failure.txt").unlink(missing_ok=True)
    except KeyboardInterrupt:
        row.update(status="interrupted", error="KeyboardInterrupt",
                   elapsed_seconds=time.perf_counter() - started)
        atomic_json(result_path, row)
        print(f"INTERRUPTED {model}: {case.name}", flush=True)
        raise
    except Exception as error:
        row.update(status="failed", error=f"{type(error).__name__}: {error}")
        (directory / "failure.txt").write_text(traceback.format_exc())
    row["elapsed_seconds"] = time.perf_counter() - started
    atomic_json(result_path, row)
    print(f"{row['status'].upper()} {model}: {case.name} ({row['elapsed_seconds']:.1f}s)"
          + (f" — {row['error']}" if row["status"] == "failed" else ""), flush=True)
    return row["status"] == "ok"


def collect_results(output, run_label="default"):
    records = []
    for path in sorted(Path(output).glob(f"runs/*/{run_label}/**/result.json")):
        row = read_json(path)
        # Keep scalar columns in the comparison table; full settings stay in JSON.
        flat = {key: value for key, value in row.items() if not isinstance(value, (dict, list))}
        flat["result_path"] = str(path.resolve())
        records.append(flat)
    return pd.DataFrame(records)


def paired_comparisons(results, metric):
    """Only compare SING/GTM fitted to the exact same cached dataset."""
    if results.empty:
        return pd.DataFrame()
    good = results.loc[results.status.eq("ok")].copy()
    if good.empty:
        return pd.DataFrame()
    keys = ["n_train", "marginals", "seed"]
    if good.duplicated(keys + ["model"]).any():
        raise ValueError("Duplicate model/case results; select one run label")
    if (good.groupby(keys).dataset_hash.nunique() > 1).any():
        raise ValueError("Refusing to pair results from different datasets")
    table = good.pivot(index=keys, columns="model", values=metric)
    if "GTM" not in table.columns:
        return pd.DataFrame()
    rows = []
    for model in table.columns:
        if model == "GTM":
            continue
        complete = table[["GTM", model]].dropna()
        difference = complete["GTM"] - complete[model]
        for (n_train, marginal), values in difference.groupby(level=["n_train", "marginals"]):
            rows.append(dict(n_train=n_train, marginals=marginal, comparison=f"GTM minus {model}",
                             metric=metric, n_pairs=len(values), mean_difference=values.mean(),
                             sd_difference=values.std(ddof=1),
                             se_difference=values.std(ddof=1) / np.sqrt(len(values))))
    return pd.DataFrame(rows)
