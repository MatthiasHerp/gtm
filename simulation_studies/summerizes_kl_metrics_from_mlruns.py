#!/usr/bin/env python3
"""
CSV-only summary of KLD metrics from MLflow filesystem:

Reads:
  mlruns/<experiment_id>/<run_id>/metrics/<metric_name>

Takes the LAST logged value in each metric file and aggregates across runs.

Writes:
  - kld_metrics_long.csv                  (one row per run × metric)
  - kld_summary_by_experiment_split.csv   (mean/sd/n per experiment × split × model)
  - kld_summary_wide_test.csv             (one row per experiment; columns = models, values = mean±sd)
  - kld_summary_wide_train.csv
"""

from __future__ import annotations
import re
from pathlib import Path
from dataclasses import dataclass

import numpy as np
import pandas as pd


# =============================================================================
# CONFIG
# =============================================================================

@dataclass
class Config:
    mlruns_path: str = "/Users/franciscocapunay/Downloads/gtm_server_copy/mlruns"
    outdir: str = "/Users/franciscocapunay/Downloads/gtm_server_copy/kld_model_csv"
    verbose: bool = True

    want_metrics: tuple[str, ...] = (
        "kld_bgtm_test",
        "kld_gtm_test",
        "kld_gaussian_test",
        "kld_vine_oracle_test",
        "kld_bgtm_train",
        "kld_gtm_train",
        "kld_gaussian_train",
        "kld_vine_oracle_train",
    )


# =============================================================================
# Helpers
# =============================================================================

def _read_text(p: Path) -> str | None:
    try:
        return p.read_text().strip()
    except Exception:
        return None

def read_last_metric_value(metric_file: Path) -> float | None:
    """
    MLflow metric file lines:
        <timestamp> <value> <step>
    take last non-empty line, parse second token.
    """
    txt = _read_text(metric_file)
    if not txt:
        return None
    lines = [ln.strip() for ln in txt.splitlines() if ln.strip()]
    if not lines:
        return None
    parts = lines[-1].split()
    if len(parts) < 2:
        return None
    try:
        return float(parts[1])
    except Exception:
        return None

def read_experiment_name(exp_dir: Path) -> str:
    meta = exp_dir / "meta.yaml"
    txt = _read_text(meta)
    if not txt:
        return exp_dir.name
    m = re.search(r"^\s*name:\s*(?P<name>.+?)\s*$", txt, flags=re.MULTILINE)
    if m:
        return m.group("name").strip().strip("'").strip('"')
    return exp_dir.name

def is_run_dir(p: Path) -> bool:
    return p.is_dir() and (p / "metrics").is_dir()

def iter_run_dirs(mlruns_root: Path):
    ignore = {".trash", "models"}
    for exp_dir in sorted(mlruns_root.iterdir()):
        if not exp_dir.is_dir() or exp_dir.name in ignore:
            continue
        exp_id = exp_dir.name
        exp_name = read_experiment_name(exp_dir)
        for run_dir in sorted(exp_dir.iterdir()):
            if is_run_dir(run_dir):
                yield exp_id, exp_name, run_dir

def metric_to_model(metric: str) -> str:
    if "kld_bgtm_" in metric:
        return "BGTM"
    if "kld_gtm_" in metric:
        return "GTM"
    if "kld_gaussian_" in metric:
        return "Gaussian"
    if "kld_vine_oracle_" in metric:
        return "VineOracle"
    return metric

def metric_to_split(metric: str) -> str:
    if metric.endswith("_test"):
        return "test"
    if metric.endswith("_train"):
        return "train"
    return "unknown"

def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p

def mean_pm_sd(x: pd.Series) -> str:
    x = x.dropna().astype(float)
    if len(x) == 0:
        return ""
    mu = x.mean()
    sd = x.std(ddof=1) if len(x) > 1 else 0.0
    return f"{mu:.3f}±{sd:.3f}"


# =============================================================================
# Main
# =============================================================================

def main(cfg: Config):
    mlruns_root = Path(cfg.mlruns_path).expanduser().resolve()
    outdir = ensure_dir(Path(cfg.outdir).expanduser().resolve())

    if cfg.verbose:
        print(f"[INFO] mlruns_root={mlruns_root}")
        print(f"[INFO] outdir={outdir}")

    rows = []
    for exp_id, exp_name, run_dir in iter_run_dirs(mlruns_root):
        run_id = run_dir.name
        metrics_dir = run_dir / "metrics"

        for m in cfg.want_metrics:
            mf = metrics_dir / m
            if not mf.exists():
                continue
            v = read_last_metric_value(mf)
            if v is None:
                continue
            rows.append({
                "experiment_id": exp_id,
                "experiment": exp_name,
                "run_id": run_id,
                "metric": m,
                "split": metric_to_split(m),
                "model": metric_to_model(m),
                "kld": float(v),
            })

    if not rows:
        raise RuntimeError("No metric values found. Check want_metrics and mlruns_path.")

    df_long = pd.DataFrame(rows).sort_values(["experiment", "split", "model", "run_id"])
    df_long.to_csv(outdir / "kld_metrics_long.csv", index=False)

    # Long summary: mean/sd/n
    df_sum = (
        df_long
        .groupby(["experiment", "split", "model"], as_index=False)
        .agg(
            mean_kld=("kld", "mean"),
            sd_kld=("kld", "std"),
            n_runs=("kld", "count"),
        )
    )
    df_sum.to_csv(outdir / "kld_summary_by_experiment_split.csv", index=False)

    # Wide LaTeX-friendly: one row per experiment; columns are model mean±sd
    for split in ["test", "train"]:
        d = df_long[df_long["split"] == split].copy()
        wide = (
            d.groupby(["experiment", "model"])["kld"]
             .apply(mean_pm_sd)
             .unstack("model")
             .reset_index()
        )

        # Optional: stable column order
        order = ["experiment", "BGTM", "GTM", "Gaussian", "VineOracle"]
        for c in order:
            if c not in wide.columns:
                wide[c] = ""
        wide = wide[order]

        wide.to_csv(outdir / f"kld_summary_wide_{split}.csv", index=False)

    if cfg.verbose:
        print("[OK] Wrote:")
        print(f"  - {outdir / 'kld_metrics_long.csv'}")
        print(f"  - {outdir / 'kld_summary_by_experiment_split.csv'}")
        print(f"  - {outdir / 'kld_summary_wide_test.csv'}")
        print(f"  - {outdir / 'kld_summary_wide_train.csv'}")


if __name__ == "__main__":
    cfg = Config()
    main(cfg)
