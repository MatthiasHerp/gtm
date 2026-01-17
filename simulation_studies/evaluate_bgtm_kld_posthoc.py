#!/usr/bin/env python3
"""
Post-hoc evaluation of BGTM conditional-independence KLD from MLflow artifacts.

This version is "run-and-debug friendly":
- Hard-coded defaults in CONFIG (no CLI needed)
- Optional CLI override still supported
- Auto-detects mlruns/ by walking up from this script if MLRUNS_PATH=None
- Verbose prints and sanity checks

Run:
  python simulation_studies/evaluate_bgtm_kld_posthoc.py
"""

from __future__ import annotations

import os
import re
import sys
import argparse
from pathlib import Path
from dataclasses import dataclass

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# =============================================================================
# CONFIG (EDIT HERE)
# =============================================================================

@dataclass
class Config:
    # If None: auto-find by walking up from this script location
    mlruns_path: str | None = None
    # If None: will default to <project_root>/bgtm_kld_eval
    outdir: str | None = None
    experiment_name: str | None = None
    cred_level: float = 0.90
    eps: float = 1e-3
    require_truth: bool = True
    # Debug verbosity
    verbose: bool = True
    # Limit runs processed (None = all). Helpful for debugging.
    max_runs: int | None = None


# =============================================================================
# PATH DISCOVERY
# =============================================================================

def find_project_mlruns(start: Path, mlruns_dirname: str = "mlruns", max_up: int = 8) -> Path:
    """
    Walk up parent directories starting at `start` until we find `mlruns/`.
    """
    cur = start.resolve()
    for _ in range(max_up + 1):
        candidate = cur / mlruns_dirname
        if candidate.exists() and candidate.is_dir():
            return candidate
        cur = cur.parent

    raise FileNotFoundError(
        f"Could not find '{mlruns_dirname}/' by walking up from {start} (max_up={max_up}).\n"
        f"Fix: either (1) hardcode CONFIG.mlruns_path, or (2) run script from within the project."
    )


def default_outdir(project_root: Path, cfg: Config) -> Path:
    # format cred level nicely (e.g. 0.9 -> "0p90")
    cred_str = f"{cfg.cred_level:.2f}".replace(".", "p")
    out = project_root / "bgtm_kld_eval" / str(cfg.experiment_name) / cred_str
    return out


def clear_dir_contents(dir_path: Path, verbose: bool = False) -> None:
    """
    Deletes all files/folders inside dir_path, but keeps dir_path itself.
    """
    if not dir_path.exists():
        return
    for child in dir_path.iterdir():
        try:
            if child.is_file() or child.is_symlink():
                child.unlink()
                if verbose:
                    print(f"[CLEAN] removed file: {child}")
            elif child.is_dir():
                # recursive delete
                import shutil
                shutil.rmtree(child)
                if verbose:
                    print(f"[CLEAN] removed dir : {child}")
        except Exception as e:
            print(f"[WARN] Could not delete {child}: {e}")



# =============================================================================
# MLFLOW FILESYSTEM HELPERS
# =============================================================================

def find_run_dirs(
    mlruns_root: Path,
    experiment_name: str | None = None,
    verbose: bool = False,
) -> list[Path]:
    """
    Finds MLflow run directories under:
      mlruns/<experiment_id>/<run_id>/
      mlruns/<experiment_name>/<run_id>/

    Ignores:
      mlruns/.trash
      mlruns/models
    """
    mlruns_root = mlruns_root.resolve()
    ignore_names = {".trash", "models"}

    run_dirs: list[Path] = []

    # Choose which experiment directories to scan
    if experiment_name is not None:
        exp_dirs = [mlruns_root / experiment_name]
        if not exp_dirs[0].exists():
            raise FileNotFoundError(
                f"Experiment folder not found: {exp_dirs[0]}\n"
                f"Available folders in mlruns/: {[p.name for p in mlruns_root.iterdir() if p.is_dir()]}"
            )
    else:
        exp_dirs = [p for p in sorted(mlruns_root.iterdir()) if p.is_dir() and p.name not in ignore_names]

    for exp_dir in exp_dirs:
        if exp_dir.name in ignore_names:
            continue

        for run_dir in sorted(exp_dir.iterdir()):
            if not run_dir.is_dir():
                continue
            
            run_dirs.append(run_dir)
            if verbose:
                print(f"[find_run_dirs] {exp_dir.name} -> run: {run_dir.name}")

    return run_dirs

def find_ci_npz(run_dir: Path) -> Path | None:
    candidates = [
        # your current structure (confirmed)
        run_dir / "artifacts" / "gtm_bayes" / "ci_raw" / "bgtm_ci_raw_arrays.npz",

        # fallback (if in some runs you changed artifact_location)
        run_dir / "gtm_bayes" / "ci_raw" / "bgtm_ci_raw_arrays.npz",
    ]
    for c in candidates:
        if c.exists():
            return c
    return None

def find_truth_csv(run_dir: Path) -> Path | None:
    candidates = [
        run_dir / "artifacts" / "truth" / "true_structure.csv",
        run_dir / "truth" / "true_structure.csv",
    ]
    for c in candidates:
        if c.exists():
            return c
    return None

def _read_text_safely(p: Path) -> str | None:
    try:
        return p.read_text().strip()
    except Exception:
        return None

def _find_mlruns_root_from_run_dir(run_dir: Path) -> Path | None:
    """
    Walk up to find the parent folder named 'mlruns'.
    """
    cur = run_dir.resolve()
    for parent in [cur] + list(cur.parents):
        if parent.name == "mlruns":
            return parent
    return None

def _get_run_id_from_path(run_dir: Path) -> str:
    return run_dir.name

def _read_experiment_id_from_meta(meta_path: Path) -> str | None:
    """
    meta.yaml usually contains a line like: experiment_id: '441465781417630617'
    """
    txt = _read_text_safely(meta_path)
    if not txt:
        return None

    m = re.search(r"^\s*experiment_id:\s*'?(?P<eid>\d+)'?\s*$", txt, flags=re.MULTILINE)
    if m:
        return m.group("eid")
    return None

def _canonical_run_dir(run_dir: Path) -> Path | None:
    """
    Try to map a run_dir that might be under <experiment_name>/<run_id>
    to the canonical MLflow run dir: <experiment_id>/<run_id>.
    """
    run_dir = run_dir.resolve()
    run_id = _get_run_id_from_path(run_dir)

    # 1) If meta.yaml exists here, use it to get experiment_id
    meta_here = run_dir / "meta.yaml"
    mlruns_root = _find_mlruns_root_from_run_dir(run_dir)

    if meta_here.exists():
        eid = _read_experiment_id_from_meta(meta_here)
        if eid and mlruns_root:
            cand = mlruns_root / eid / run_id
            if cand.exists():
                return cand

    # 2) If no meta.yaml here, try to find canonical by scanning experiment_id dirs
    if mlruns_root:
        # common ignore dirs
        ignore = {".trash", "models"}
        for exp_dir in mlruns_root.iterdir():
            if not exp_dir.is_dir() or exp_dir.name in ignore:
                continue
            cand = exp_dir / run_id
            if cand.is_dir():
                # optional sanity: check meta.yaml exists
                if (cand / "meta.yaml").exists() or (cand / "tags").exists():
                    return cand

    return None

def read_tag(run_dir: Path, key: str) -> str | None:
    """
    Read MLflow tag value robustly even if run_dir is not the canonical location.
    """
    # A) direct (fast path)
    tag_path = run_dir / "tags" / key
    if tag_path.exists():
        return _read_text_safely(tag_path)

    # B) canonical location (experiment_id/run_id)
    canon = _canonical_run_dir(run_dir)
    if canon is not None:
        tag_path2 = canon / "tags" / key
        if tag_path2.exists():
            return _read_text_safely(tag_path2)

    # C) final fallback: brute search under mlruns_root
    mlruns_root = _find_mlruns_root_from_run_dir(run_dir)
    run_id = _get_run_id_from_path(run_dir)
    if mlruns_root:
        ignore = {".trash", "models"}
        for exp_dir in mlruns_root.iterdir():
            if not exp_dir.is_dir() or exp_dir.name in ignore:
                continue
            cand = exp_dir / run_id / "tags" / key
            if cand.exists():
                return _read_text_safely(cand)

    return None


def infer_seed(run_dir: Path) -> int | None:
    # 1) preferred: explicit seed tag
    s = read_tag(run_dir, "seed_value")
    if s:
        try:
            return int(float(s))
        except Exception:
            pass

    # 2) fallback: parse from runName
    rn = read_tag(run_dir, "mlflow.runName") or ""
    m = re.search(r"seed[_\- ](\d+)", rn)
    if m:
        return int(m.group(1))

    return None


def infer_experiment_name(run_dir: Path) -> str:
    en = read_tag(run_dir, "mlflow.experimentName")
    if en:
        return en
    return run_dir.parent.name  # exp_id fallback


# =============================================================================
# CORE COMPUTATIONS
# =============================================================================

def summarize_kld_samples(kld_samples: np.ndarray, cred_level: float, eps: float) -> dict[str, np.ndarray]:
    """
    kld_samples: [S_post, n_pairs]
    returns arrays length n_pairs
    """
    if kld_samples.ndim != 2:
        raise ValueError(f"kld_samples must be 2D [S_post, n_pairs], got shape {kld_samples.shape}")

    alpha = 1.0 - cred_level
    lo_q = alpha / 2.0
    hi_q = 1.0 - alpha / 2.0

    mean = np.nanmean(kld_samples, axis=0)
    lo = np.nanquantile(kld_samples, lo_q, axis=0)
    hi = np.nanquantile(kld_samples, hi_q, axis=0)

    # posterior mass near 0
    p_le_eps = np.mean(kld_samples <= eps, axis=0)

    # KLD-based CI decision (more meaningful than "0 in CI")
    lo_le_eps = (lo <= eps).astype(int)

    return {
        "kld_mean": mean,
        "kld_lo": lo,
        "kld_hi": hi,
        "p_kld_le_eps": p_le_eps,
        "lo_le_eps": lo_le_eps,
    }


def make_pair_label(var_row: int, var_col: int) -> str:
    return f"({int(var_row)},{int(var_col)})"


def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


# =============================================================================
# PLOTTING
# =============================================================================

def find_training_npz(run_dir: Path) -> Path | None:
    """
    Looks for ELBO training artifact in:
      <run_dir>/artifacts/gtm_bayes/training/*.npz
    Returns first npz found (sorted).
    """
    candidates = [
        run_dir / "artifacts" / "gtm_bayes" / "training",
        run_dir / "gtm_bayes" / "training",
    ]
    for folder in candidates:
        if folder.exists() and folder.is_dir():
            files = sorted([p for p in folder.iterdir() if p.suffix == ".npz"])
            if len(files) > 0:
                return files[0]
    return None


def load_elbo_from_npz(npz_path: Path) -> np.ndarray | None:
    try:
        z = np.load(npz_path, allow_pickle=False)
    except Exception:
        return None
    if "elbo" not in z.files:
        return None
    elbo = np.asarray(z["elbo"], dtype=float)
    if elbo.ndim != 1 or elbo.size == 0:
        return None
    return elbo


def summarize_elbo(elbo: np.ndarray, tail_window: int = 50) -> dict[str, float]:
    """
    Basic convergence diagnostics.
    """
    n = int(elbo.size)
    w = int(min(tail_window, n))
    tail = elbo[-w:]
    first = float(elbo[0])
    last = float(elbo[-1])
    tail_mean = float(np.nanmean(tail))
    tail_std = float(np.nanstd(tail))
    # last improvement compared to previous window mean (if exists)
    if n >= 2*w:
        prev = elbo[-2*w:-w]
        delta_tail = float(np.nanmean(tail) - np.nanmean(prev))
    else:
        delta_tail = float("nan")
    return {
        "elbo_len": float(n),
        "elbo_first": first,
        "elbo_last": last,
        "elbo_gain": float(last - first),
        "elbo_tail_mean": tail_mean,
        "elbo_tail_std": tail_std,
        "elbo_delta_tail_mean": delta_tail,
    }
    
def make_confusion_tables_from_counts(cm: pd.DataFrame) -> pd.DataFrame:
    """
    Input: cm with columns [experiment, seed, true_ci, pred_ci, count]
           where pred_ci contains only decided labels (independent/dependent).
    Output: one row per (experiment, seed) with TP/TN/FP/FN and derived rates.
    Convention:
      - positive class = dependent
      - negative class = independent
      TP: true dependent predicted dependent
      TN: true independent predicted independent
      FP: true independent predicted dependent
      FN: true dependent predicted independent
    """
    # pivot into 2x2 per group
    piv = (
        cm.pivot_table(
            index=["experiment", "seed", "true_ci"],
            columns="pred_ci",
            values="count",
            aggfunc="sum",
            fill_value=0,
        )
        .reset_index()
    )

    # ensure columns exist
    for c in ["dependent", "independent"]:
        if c not in piv.columns:
            piv[c] = 0

    out_rows = []
    for (exp, seed), g in piv.groupby(["experiment", "seed"]):
        # get counts by true class
        dep = g.loc[g["true_ci"] == "dependent"]
        indep = g.loc[g["true_ci"] == "independent"]

        TP = int(dep["dependent"].sum()) if not dep.empty else 0
        FN = int(dep["independent"].sum()) if not dep.empty else 0
        FP = int(indep["dependent"].sum()) if not indep.empty else 0
        TN = int(indep["independent"].sum()) if not indep.empty else 0

        total = TP + TN + FP + FN
        acc = (TP + TN) / total if total else np.nan
        tpr = TP / (TP + FN) if (TP + FN) else np.nan           # recall dependent
        tnr = TN / (TN + FP) if (TN + FP) else np.nan           # recall independent
        bal_acc = 0.5 * (tpr + tnr) if np.isfinite(tpr) and np.isfinite(tnr) else np.nan

        out_rows.append({
            "experiment": exp,
            "seed": seed,
            "TP_dep": TP,
            "FN_dep": FN,
            "FP_dep": FP,
            "TN_indep": TN,
            "n_decided": total,
            "accuracy_decided": acc,
            "recall_dependent": tpr,
            "recall_independent": tnr,
            "balanced_accuracy": bal_acc,
        })

    return pd.DataFrame(out_rows)

def pair_level_confusion(df_all: pd.DataFrame, thr: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Aggregate across seeds/runs to get ONE prediction per pair.
    pred rule:
      indep if mean p_kld_le_eps >= thr
      dep   if mean p_kld_le_eps <= 1-thr
      else uncertain
    Returns:
      - df_pair: per-pair summary + pred
      - cm_pair: confusion counts over decided pairs
    """
    df = df_all.copy()
    df["true_ci"] = np.where(df["dependence"] == 0, "independent", "dependent")

    df_pair = (
        df.groupby(["experiment", "pair", "true_ci"], dropna=False)
          .agg(
              mean_p=("p_kld_le_eps", "mean"),
              sd_p=("p_kld_le_eps", "std"),
              n=("p_kld_le_eps", "count"),
          )
          .reset_index()
    )

    hi = thr
    lo = 1.0 - thr
    df_pair["pred_ci_pair"] = "uncertain"
    df_pair.loc[df_pair["mean_p"] >= hi, "pred_ci_pair"] = "independent"
    df_pair.loc[df_pair["mean_p"] <= lo, "pred_ci_pair"] = "dependent"

    decided = df_pair[df_pair["pred_ci_pair"] != "uncertain"].copy()
    cm_pair = (
        decided.groupby(["experiment", "true_ci", "pred_ci_pair"])
               .size()
               .reset_index(name="count")
    )
    return df_pair, cm_pair


def plot_scatter_pairs_with_ci(df_all: pd.DataFrame, outpath: Path, eps: float) -> None:
    """
    Scatter of posterior mean KLD per (seed,pair) WITH credible interval error bars.
    Requires columns: pair, kld_mean, kld_lo, kld_hi (and optionally dependence).
    """
    df = df_all.copy()
    df["pair"] = df["pair"].astype(str)

    pairs_sorted = sorted(df["pair"].unique())
    pair_to_x = {p: i for i, p in enumerate(pairs_sorted)}
    df["x"] = df["pair"].map(pair_to_x)

    # error bars: asymmetric
    y = df["kld_mean"].to_numpy()
    yerr_low = (df["kld_mean"] - df["kld_lo"]).to_numpy()
    yerr_high = (df["kld_hi"] - df["kld_mean"]).to_numpy()
    yerr = np.vstack([yerr_low, yerr_high])

    plt.figure(figsize=(12, 4))

    # null band around 0 (optional but very helpful)
    if eps is not None and eps > 0:
        plt.axhspan(-eps, eps, alpha=0.15)
        plt.axhline(0.0, linewidth=1.0)

    if "dependence" in df.columns:
        for val, name in [(0, "independent"), (1, "dependent"), (-1, "unknown")]:
            sub = df.loc[df["dependence"].fillna(-1).astype(int) == val]
            if sub.empty:
                continue
            xs = sub["x"].to_numpy()
            ys = sub["kld_mean"].to_numpy()
            low = (sub["kld_mean"] - sub["kld_lo"]).to_numpy()
            high = (sub["kld_hi"] - sub["kld_mean"]).to_numpy()
            yerr_sub = np.vstack([low, high])

            plt.errorbar(xs, ys, yerr=yerr_sub, fmt="o", markersize=4, capsize=2, alpha=0.7, label=name)

        plt.legend()
    else:
        plt.errorbar(df["x"], df["kld_mean"], yerr=yerr, fmt="o", markersize=4, capsize=2, alpha=0.7)

    plt.xticks(range(len(pairs_sorted)), pairs_sorted, rotation=90)
    plt.ylabel("KLD mean (per run)")
    plt.title("Pairwise KLD means across seeds (with posterior CI)")
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()
    
def plot_heatmap_prob_from_pairs(df_pairs: pd.DataFrame, title: str, outpath: Path, col: str, vmin=None, vmax=None) -> None:
    """
    Heatmap for a probability-like column in [0,1], e.g. p_kld_le_eps.
    df_pairs must have: var_row, var_col, and `col`.
    """
    d = int(max(df_pairs["var_row"].max(), df_pairs["var_col"].max()) + 1)
    mat = np.full((d, d), np.nan)
    for _, r in df_pairs.iterrows():
        i, j = int(r["var_row"]), int(r["var_col"])
        mat[i, j] = r[col]
        mat[j, i] = r[col]

    plt.figure()
    plt.imshow(mat, aspect="equal", vmin=vmin, vmax=vmax)
    plt.colorbar(label=col)
    plt.title(title)
    plt.xlabel("var")
    plt.ylabel("var")
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()

def plot_boxplot_across_seeds(df_all: pd.DataFrame, outpath: Path, col: str) -> None:
    df = df_all.copy()
    df["pair"] = df["pair"].astype(str)

    pairs_sorted = sorted(df["pair"].unique())
    data = [df.loc[df["pair"] == p, col].values for p in pairs_sorted]

    plt.figure(figsize=(max(10, len(pairs_sorted) * 0.5), 5))
    plt.boxplot(data, labels=pairs_sorted, showfliers=False)
    if col == "p_kld_le_eps": plt.ylim(-0.05, 1.05)
    plt.xticks(rotation=90)
    plt.ylabel(
        ("KLD mean (per run)" if col == "kld_mean" else "p(KLD <= eps)") 
        )
    plt.title(
        ("KLD per pair across seeds" if col == "kld_mean" else "Posterior independence probability per pair across seeds")
        )
    
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()

def plot_scatter_pairs(
    df_all: pd.DataFrame,
    outpath: Path,
    col: str,
    *,
    thr: float = 0.95,
    eps: float | None = None,
) -> None:
    """
    Scatter plot of posterior practical-independence probability
    p_eps(i,j) = Pr(KLD_{i,j} <= eps | D) across seeds.

    - x-axis: variable pairs
    - y-axis: posterior probability
    - background bands: decision regions
    - color: truth
    - marker: prediction
    """

    df = df_all.copy()
    df["pair"] = df["pair"].astype(str)

    # order pairs by mean probability (clean readability)
    order = (
        df.groupby("pair")[col]
        .mean()
        .sort_values(ascending=True)
        .index.tolist()
    )
    df["x"] = df["pair"].map({p: i for i, p in enumerate(order)})

    plt.figure(figsize=(12, 4))
    rng = np.random.default_rng(0)

    # ---------- decision thresholds ----------
    hi_thr = thr
    lo_thr = 1.0 - thr

    # background decision bands
    plt.axhspan(hi_thr, 1.0, alpha=0.08)
    plt.axhspan(lo_thr, hi_thr, alpha=0.06)
    plt.axhspan(0.0, lo_thr, alpha=0.08)

    # threshold lines
    plt.axhline(hi_thr, linewidth=1.2)
    plt.axhline(lo_thr, linewidth=1.2, linestyle="--")
    plt.axhline(0.5, linewidth=0.8, linestyle=":", alpha=0.7)

    # ---------- scatter points ----------
    if "dependence" in df.columns and "pred_ci" in df.columns:
        # truth → color, prediction → marker
        truth_map = {0: "independent", 1: "dependent", -1: "unknown"}
        pred_mark = {
            "independent": "o",
            "dependent": "x",
            "uncertain": "s",   # visible marker (not '.')
        }

        for tval, tname in [(0, "independent"), (1, "dependent"), (-1, "unknown")]:
            sub = df.loc[df["dependence"].fillna(-1).astype(int) == tval]
            if sub.empty:
                continue

            for pname, marker in pred_mark.items():
                sub2 = sub.loc[sub["pred_ci"] == pname]
                if sub2.empty:
                    continue

                xj = (
                    sub2["x"].to_numpy(float)
                    + rng.uniform(-0.08, 0.08, size=len(sub2))
                )

                plt.scatter(
                    xj,
                    sub2[col],
                    s=28,
                    alpha=0.75,
                    marker=marker,
                    label=f"{tname} / pred={pname}",
                )
    else:
        # fallback (no truth/pred available)
        xj = df["x"].to_numpy(float) + rng.uniform(-0.08, 0.08, size=len(df))
        plt.scatter(xj, df[col], s=22, alpha=0.7)

    # ---------- legend (deduplicated) ----------
    handles, labels = plt.gca().get_legend_handles_labels()
    uniq = dict(zip(labels, handles))
    plt.legend(uniq.values(), uniq.keys(), fontsize=8, ncol=2)

    # ---------- axes & labels ----------
    plt.xticks(range(len(order)), order, rotation=90)
    plt.ylim(-0.05, 1.05)

    plt.ylabel(
        r"$p_{\varepsilon}(i,j)"
        r"=$p_{\varepsilon}(\mathrm{KLD} \leq \varepsilon)$"
    )

    title = "Posterior practical-independence probability per pair across seeds"
    if eps is not None:
        title += f" (ε={eps:g}, γ={thr:.2f})"
    plt.title(title)

    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()
    
def plot_cm_2x2(TN, FP, FN, TP, title, outpath):
    mat = np.array([[TN, FP],
                    [FN, TP]], dtype=float)

    plt.figure(figsize=(4, 3))
    plt.imshow(mat, aspect="equal")
    plt.xticks([0, 1], ["pred=indep", "pred=dep"])
    plt.yticks([0, 1], ["true=indep", "true=dep"])
    for (i, j), v in np.ndenumerate(mat):
        plt.text(j, i, f"{int(v)}", ha="center", va="center")
    plt.title(title)
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()


# =============================================================================
# CLI OVERRIDES (OPTIONAL)
# =============================================================================

def parse_args_optional() -> argparse.Namespace:
    ap = argparse.ArgumentParser(add_help=True)
    ap.add_argument("--mlruns", type=str, default=None, help="Override mlruns path")
    ap.add_argument("--outdir", type=str, default=None, help="Override output directory")
    ap.add_argument("--experiment_name", type=str, default=None)
    ap.add_argument("--cred_level", type=float, default=None)
    ap.add_argument("--eps", type=float, default=None)
    ap.add_argument("--require_truth", action="store_true", default=False)
    ap.add_argument("--quiet", action="store_true", default=False)
    ap.add_argument("--max_runs", type=int, default=None, help="Process only first N runs (debug)")
    return ap.parse_args()


def merge_config(cfg: Config, args: argparse.Namespace) -> Config:
    # CLI overrides only if provided
    if args.mlruns is not None:
        cfg.mlruns_path = args.mlruns
    if args.outdir is not None:
        cfg.outdir = args.outdir
    if args.experiment_name is not None:
        cfg.experiment_name = args.experiment_name
    if args.cred_level is not None:
        cfg.cred_level = args.cred_level
    if args.eps is not None:
        cfg.eps = args.eps
    if args.require_truth:
        cfg.require_truth = True
    if args.quiet:
        cfg.verbose = False
    if args.max_runs is not None:
        cfg.max_runs = args.max_runs
    return cfg


# =============================================================================
# MAIN
# =============================================================================

def main(cfg: Config, script_dir: Path = Path(__file__).resolve().parent) -> None:

    # Resolve mlruns root
    if cfg.mlruns_path is None:
        mlruns_root = find_project_mlruns(script_dir)
        project_root = mlruns_root.parent
    else:
        mlruns_root = Path(cfg.mlruns_path).expanduser().resolve()
        if not mlruns_root.exists():
            raise FileNotFoundError(f"mlruns path does not exist: {mlruns_root}")
        project_root = mlruns_root.parent

    # Resolve output directory
    if cfg.outdir is None:
        out_root = project_root / "bgtm_kld_eval"
    else:
        out_root = Path(cfg.outdir).expanduser().resolve()

    # Final output folder: out_root/experiment/cred_level/
    outdir = default_outdir(out_root.parent, cfg) if cfg.outdir is None else (
        out_root / str(cfg.experiment_name) / f"{cfg.cred_level:.2f}".replace(".", "p")
    )

    outdir = ensure_dir(outdir)
    # Clean previous results in that folder
    clear_dir_contents(outdir, verbose=cfg.verbose)

    plots_dir = ensure_dir(outdir / "plots")

    if cfg.verbose:
        print("=" * 80)
        print("BGTM KLD post-hoc evaluator")
        print(f"script_dir : {script_dir}")
        print(f"mlruns_root: {mlruns_root}")
        print(f"outdir     : {outdir}")
        print(f"cred_level : {cfg.cred_level}")
        print(f"eps        : {cfg.eps}")
        print(f"require_truth: {cfg.require_truth}")
        print(f"max_runs   : {cfg.max_runs}")
        print("=" * 80)

    run_dirs = find_run_dirs(
        mlruns_root,
        cfg.experiment_name,
        cfg.verbose
        )
    
    if len(run_dirs) == 0:
        raise RuntimeError(f"No MLflow run dirs found under: {mlruns_root}")

    if cfg.verbose:
        print(f"[INFO] Found {len(run_dirs)} MLflow run directories (before filtering).")

    rows_all = []
    rows_run_summary = []
    processed = 0
    
    trash_dir = (mlruns_root / ".trash").resolve()
    
    elbo_traces = {}  # run_id -> dict(seed=?, exp=?, elbo=np.array)


    for run_dir in sorted(run_dirs):
        #npz_path = run_dir / "artifacts" / "gtm_bayes" / "ci_raw" / "bgtm_ci_raw_arrays.npz"
        run_dir_resolved = run_dir.resolve()
        if trash_dir in run_dir_resolved.parents or run_dir_resolved == trash_dir:
            continue
        
        npz_path = find_ci_npz(run_dir)
        if  npz_path is None:
            if cfg.verbose:
                print(f"[SKIP] No ci_raw npz in run: {run_dir}")
            continue

        #truth_path = run_dir / "artifacts" / "truth" / "true_structure.csv"
        truth_path = find_truth_csv(run_dir)
        if cfg.require_truth and (truth_path is None or not truth_path.exists()):
            continue

        seed = infer_seed(run_dir)
        exp = infer_experiment_name(run_dir)
        run_id = run_dir.name

        if cfg.verbose:
            print(f"\n[RUN] exp={exp} seed={seed} run_id={run_id}")
            print(f"      npz: {npz_path}")

        try:
            npz = np.load(npz_path, allow_pickle=False)
        except Exception as e:
            print(f"[WARN] Failed to load NPZ: {npz_path} | {e}")
            continue

        if "kld" not in npz.files or "pair_index" not in npz.files:
            print(f"[WARN] Missing keys in {npz_path}. Found keys: {npz.files}")
            continue

        kld = np.asarray(npz["kld"], dtype=float)
        pair_index = np.asarray(npz["pair_index"], dtype=int)

        if cfg.verbose:
            print(f"      kld shape      : {kld.shape}")
            print(f"      pair_index shape: {pair_index.shape}")

        summ = summarize_kld_samples(kld, cred_level=cfg.cred_level, eps=cfg.eps)

        df_pairs = pd.DataFrame({
            "var_row": pair_index[:, 0],
            "var_col": pair_index[:, 1],
            "pair": [make_pair_label(a, b) for a, b in pair_index],
            "kld_mean": summ["kld_mean"],
            "kld_lo": summ["kld_lo"],
            "kld_hi": summ["kld_hi"],
            "p_kld_le_eps": summ["p_kld_le_eps"],
            "lo_le_eps": summ["lo_le_eps"],
            "seed": seed,
            "experiment": exp,
            "run_id": run_id,
        })
        
        df_pairs["kld_ci_width"] = df_pairs["kld_hi"] - df_pairs["kld_lo"]
        df_pairs["kld_pos_prob"] = (kld > 0).mean(axis=0)  # posterior P(KLD>0) per pair, debug

        thr_hi = cfg.cred_level
        thr_lo = 1.0 - cfg.cred_level

        df_pairs["pred_ci"] = "uncertain"
        df_pairs.loc[df_pairs["p_kld_le_eps"] >= thr_hi, "pred_ci"] = "independent"
        df_pairs.loc[df_pairs["p_kld_le_eps"] <= thr_lo, "pred_ci"] = "dependent"

        if truth_path.exists():
            try:
                truth = pd.read_csv(truth_path)
                truth["var_row"] = truth["var_row"].astype(int)
                truth["var_col"] = truth["var_col"].astype(int)
                df_pairs = df_pairs.merge(
                    truth[["var_row", "var_col", "dependence"]],
                    on=["var_row", "var_col"],
                    how="left",
                )
                if "dependence" in df_pairs.columns and df_pairs["dependence"].notna().any():
                    df_pairs["true_ci"] = np.where(df_pairs["dependence"] == 0, "independent", "dependent")
                    df_pairs["is_decided"] = df_pairs["pred_ci"] != "uncertain"
                    df_pairs["is_correct"] = (df_pairs["pred_ci"] == df_pairs["true_ci"]) & df_pairs["is_decided"]
                else:
                    df_pairs["true_ci"] = np.nan
                    df_pairs["is_decided"] = df_pairs["pred_ci"] != "uncertain"
                    df_pairs["is_correct"] = np.nan
                    
                if cfg.verbose:
                    print(f"      merged truth    : YES ({truth_path})")
            except Exception as e:
                print(f"[WARN] Failed to merge truth: {truth_path} | {e}")
        else:
            if cfg.verbose:
                print("      merged truth    : NO (truth file missing)")

        run_kld_mean_over_pairs = float(np.nanmean(df_pairs["kld_mean"].values))
        run_frac_lo_le_eps = float(np.mean(df_pairs["lo_le_eps"].values))
        run_mean_p_le_eps = float(np.nanmean(df_pairs["p_kld_le_eps"].values))
        
        # --- ELBO: load training trace (optional) ---
        elbo_npz = find_training_npz(run_dir)
        elbo = load_elbo_from_npz(elbo_npz) if elbo_npz is not None else None

        elbo_stats = {}
        if elbo is not None:
            elbo_stats = summarize_elbo(elbo, tail_window=50)


        rows_run_summary.append({
            "experiment": exp,
            "run_id": run_id,
            "seed": seed,
            "n_pairs": int(df_pairs.shape[0]),
            "kld_mean_over_pairs": run_kld_mean_over_pairs,
            "frac_pairs_lo_le_eps": run_frac_lo_le_eps,
            "mean_p_kld_le_eps": run_mean_p_le_eps,
            "npz_path": str(npz_path),
            
            "elbo_npz_path": str(elbo_npz) if elbo_npz is not None else None,
            "has_elbo": bool(elbo is not None),
            **elbo_stats,
        })
        
        if elbo is not None:
            elbo_traces[run_id] = {"seed": seed, "experiment": exp, "elbo": elbo}


        rows_all.append(df_pairs)

        # per-run heatmap
        #if seed is not None:
        #    heat_out = plots_dir / f"heatmap_kld_mean_seed_{seed}_run_{run_id[:8]}.png"
        #    plot_heatmap_prob_from_pairs(
        #        df_pairs,
        #        title=f"KLD mean heatmap | exp={exp} | seed={seed} | run={run_id[:8]}",
        #        outpath=heat_out,
        #        col="kld_mean",
        #        )
            
        #    prob_out = plots_dir / f"heatmap_p_kld_le_eps_seed_{seed}_run_{run_id[:8]}.png"
        #    plot_heatmap_prob_from_pairs(
        #        df_pairs,
        #        title=f"p(KLD<=eps) heatmap | exp={exp} | seed={seed} | run={run_id[:8]} | eps={cfg.eps}",
        #        outpath=prob_out,
        #        col="p_kld_le_eps",
        #        vmin=0.0,
        #        vmax=1.0
        #        )
            
        #    bin_out = plots_dir / f"heatmap_lo_le_eps_seed_{seed}_run_{run_id[:8]}.png"
        #    plot_heatmap_prob_from_pairs(
        #        df_pairs,
        #        title=f"CI decision: 1{chr(61)}(kld_lo<=eps) | exp={exp} | seed={seed} | run={run_id[:8]}",
        #        outpath=bin_out,
        #        col="lo_le_eps",
        #        vmin=0.0,
        #        vmax=1.0
        #        )

        processed += 1
        if cfg.max_runs is not None and processed >= cfg.max_runs:
            if cfg.verbose:
                print(f"[INFO] max_runs reached ({cfg.max_runs}). Stopping early.")
            break

    if len(rows_all) == 0:
        raise RuntimeError(
            "No runs with artifacts/gtm_bayes/ci_raw/bgtm_ci_raw_arrays.npz found.\n"
            "Check that mlruns_root is correct and that artifacts were copied."
        )

    df_all = pd.concat(rows_all, ignore_index=True)
    
    if "dependence" in df_all.columns and df_all["dependence"].notna().any():
        # build true_ci everywhere
        df_all["true_ci"] = np.where(df_all["dependence"] == 0, "independent", "dependent")
        

        # per-seed metrics
        df_seed_metrics = (
            df_all.groupby(["experiment", "seed"])
            .apply(lambda g: pd.Series({
                "n_pairs": len(g),
                "coverage_decided": (g["pred_ci"] != "uncertain").mean(),

                # accuracy among decided only
                "accuracy_on_decided": (
                    (g.loc[g["pred_ci"] != "uncertain", "pred_ci"] == g.loc[g["pred_ci"] != "uncertain", "true_ci"]).mean()
                    if (g["pred_ci"] != "uncertain").any() else np.nan
                ),

                # hit-rate with abstentions treated as wrong (optional but useful)
                "accuracy_with_abstain_as_wrong": (g["pred_ci"] == g["true_ci"]).mean(),

                # recalls among decided only
                "indep_recall_decided": (
                    (g.loc[(g["true_ci"]=="independent") & (g["pred_ci"]!="uncertain"), "pred_ci"] == "independent").mean()
                    if ((g["true_ci"]=="independent") & (g["pred_ci"]!="uncertain")).any() else np.nan
                ),
                "dep_recall_decided": (
                    (g.loc[(g["true_ci"]=="dependent") & (g["pred_ci"]!="uncertain"), "pred_ci"] == "dependent").mean()
                    if ((g["true_ci"]=="dependent") & (g["pred_ci"]!="uncertain")).any() else np.nan
                ),

                # abstentions
                "frac_uncertain": (g["pred_ci"] == "uncertain").mean(),
            }))
            .reset_index()
        )

        df_seed_metrics.to_csv(outdir / "seed_level_truth_comparison.csv", index=False)
        
        decided = df_all[df_all["pred_ci"] != "uncertain"].copy()

        cm = (
            decided.groupby(["experiment", "seed", "true_ci", "pred_ci"])
            .size()
            .reset_index(name="count")
        )
        cm.to_csv(outdir / "seed_level_confusion_counts.csv", index=False)
        
        seed_cm_table = make_confusion_tables_from_counts(cm)
        seed_cm_table.to_csv(outdir / "seed_level_confusion_table_2x2.csv", index=False)

        for _, r in seed_cm_table.iterrows():
            plot_cm_2x2(
                TN=r["TN_indep"], FP=r["FP_dep"], FN=r["FN_dep"], TP=r["TP_dep"],
                title=f"Confusion (decided only) | {r['experiment']} | seed={int(r['seed'])}",
                outpath=plots_dir / f"cm_decided_{r['experiment']}_seed_{int(r['seed'])}.png"
            )

        
        pred_counts = (
            df_all.groupby(["experiment","seed","pred_ci"])
            .size().reset_index(name="count")
        )
        pred_counts.to_csv(outdir / "seed_level_prediction_counts.csv", index=False)

    
    df_runs = pd.DataFrame(rows_run_summary)
    
    
    # Save ELBO run summary CSV (if any ELBO exists)
    if "has_elbo" in df_runs.columns and df_runs["has_elbo"].any():
        df_runs.to_csv(outdir / "bgtm_kld_run_summaries_with_elbo.csv", index=False)

        # --- Plot: all runs overlay ---
        plt.figure()
        for rid, rec in elbo_traces.items():
            plt.plot(rec["elbo"], alpha=0.7)
        plt.xlabel("Iteration")
        plt.ylabel("ELBO")
        plt.title(f"ELBO Convergence (all runs) | exp={cfg.experiment_name}")
        plt.tight_layout()
        plt.savefig(plots_dir / "elbo_convergence_all_runs.png", dpi=200)
        plt.close()

        # --- Plot: by seed (overlay traces per seed) ---
        #seeds_present = sorted({rec["seed"] for rec in elbo_traces.values() if rec["seed"] is not None})
        #for s in seeds_present:
        #    plt.figure()
        #    for rid, rec in elbo_traces.items():
        #        if rec["seed"] == s:
        #            plt.plot(rec["elbo"], alpha=0.7)
        #    plt.xlabel("Iteration")
        #    plt.ylabel("ELBO")
        #    plt.title(f"ELBO Convergence | seed={s} | exp={cfg.experiment_name}")
        #    plt.tight_layout()
        #    plt.savefig(plots_dir / f"elbo_convergence_seed_{s}.png", dpi=200)
        #    plt.close()

        # Optional: per-run plots (can be many files)
        # for rid, rec in elbo_traces.items():
        #     plt.figure()
        #     plt.plot(rec["elbo"])
        #     plt.xlabel("Iteration")
        #     plt.ylabel("ELBO")
        #     plt.title(f"ELBO | run={rid[:8]} | seed={rec['seed']}")
        #     plt.tight_layout()
        #     plt.savefig(plots_dir / f"elbo_convergence_run_{rid[:8]}.png", dpi=200)
        #     plt.close()   
    else:
        if cfg.verbose:
            print("[INFO] No ELBO traces found in training artifacts (gtm_bayes/training/*.npz).")


    if "dependence" in df_all.columns and df_all["dependence"].notna().any():
        df_pair, cm_pair = pair_level_confusion(df_all, thr=cfg.cred_level)
        df_pair.to_csv(outdir / "pair_level_predictions.csv", index=False)
        cm_pair.to_csv(outdir / "pair_level_confusion_counts.csv", index=False)

    # Save CSVs
    df_all.to_csv(outdir / "bgtm_kld_per_pair_per_run.csv", index=False)
    df_runs.to_csv(outdir / "bgtm_kld_run_summaries.csv", index=False)

    # Across-seed aggregation per pair
    agg_cols = ["pair"]
    if "dependence" in df_all.columns:
        agg_cols += ["dependence"]

    df_pair_across = (
        df_all
        .groupby(agg_cols, dropna=False)
        .agg(
            kld_mean_across_seeds=("kld_mean", "mean"),
            kld_sd_across_seeds=("kld_mean", "std"),
            kld_med_across_seeds=("kld_mean", "median"),
            frac_lo_le_eps=("lo_le_eps", "mean"),
            mean_p_kld_le_eps=("p_kld_le_eps", "mean"),
            n_runs=("kld_mean", "count"),
        )
        .reset_index()
    )
    df_pair_across.to_csv(outdir / "bgtm_kld_per_pair_across_seeds.csv", index=False)

    # Plots across seeds
    plot_boxplot_across_seeds(df_all, plots_dir / "boxplot_kld_per_pair_across_seeds.png", col="kld_mean")
    plot_boxplot_across_seeds(df_all, plots_dir / "boxplot_p_kld_le_eps_per_pair_across_seeds.png", col="p_kld_le_eps")
    plot_scatter_pairs(df_all, plots_dir / "scatter_kld_pairs_across_seeds.png", col="kld_mean")
    plot_scatter_pairs(df_all,plots_dir / "scatter_p_kld_le_eps_pairs_across_seeds.png",col="p_kld_le_eps", thr=cfg.cred_level,eps=cfg.eps)
    plot_scatter_pairs_with_ci(df_all, plots_dir / "scatter_kld_pairs_across_seeds_with_ci.png", eps=cfg.eps)

    # Run-level boxplot
    if df_runs["seed"].notna().any():
        plt.figure()
        plt.boxplot(df_runs["kld_mean_over_pairs"].values, showfliers=False)
        plt.ylabel("Mean KLD over pairs (per run)")
        plt.title("Run-level mean KLD over pairs (across seeds)")
        plt.tight_layout()
        plt.savefig(plots_dir / "boxplot_run_level_kld_mean_over_pairs.png", dpi=200)
        plt.close()

    print("\n[OK] Done.")
    print(f"Outputs: {outdir}")
    print(f"  - {outdir / 'bgtm_kld_per_pair_per_run.csv'}")
    print(f"  - {outdir / 'bgtm_kld_run_summaries.csv'}")
    print(f"  - {outdir / 'bgtm_kld_per_pair_across_seeds.csv'}")
    print(f"  - plots in: {plots_dir}")


if __name__ == "__main__":

    BASE = Config(
        mlruns_path="/Users/franciscocapunay/Downloads/gtm_server_copy/mlruns",
        outdir="/Users/franciscocapunay/Downloads/gtm_server_copy/bgtm_kld_eval",  # root folder
        eps=1e-1,
        require_truth=True,
        verbose=True,
        max_runs=None
    )

    # your experiments
    EXPERIMENTS = [
        "rine_5D_1000obs_bgtm_test_1",
        "rvine_5D_1000obs_bgtm",
        "cvine_5D_1000obs_bgtm",
        "dvine_5D_1000obs_bgtm",
    ]

    # your credibility levels
    CRED_LEVELS = [0.90, 0.95, 0.99]  # extend as you want

    # optional CLI overrides still work
    args = parse_args_optional()

    for exp_name in EXPERIMENTS:
        for cred in CRED_LEVELS:
            cfg = Config(**BASE.__dict__)
            cfg.experiment_name = exp_name
            cfg.cred_level = cred

            cfg = merge_config(cfg, args)

            print("\n" + "=" * 80)
            print(f"[EVAL] experiment={cfg.experiment_name} | cred_level={cfg.cred_level}")
            print("=" * 80)

            main(cfg)

