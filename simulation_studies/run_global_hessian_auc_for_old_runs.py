import mlflow
import mlflow.pytorch
import torch
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
import os
from pathlib import Path

from gtm import GTM
from simulation_studies.generate_synthetic_vine_data import generate_synthetic_vine_data

# ── Configuration ─────────────────────────────────────────────────────────────

mlruns_scratch_uri = "/Users/matthiasherp/Desktop/phd_github_repositories/gtm/mlruns_scratch"

# All experiments we want to patch
experiment_names = [
    # cvine normal
    "cvine_10D_125obs", "cvine_10D_500obs", "cvine_10D_2000obs",
    # cvine strong
    "cvine_strong_10D_125obs", "cvine_strong_10D_500obs", "cvine_strong_10D_2000obs",
    # dvine normal
    "dvine_10D_125obs", "dvine_10D_500obs", "dvine_10D_2000obs",
    # dvine strong
    "dvine_strong_10D_125obs", "dvine_strong_10D_500obs", "dvine_strong_10D_2000obs",
    # rvine normal
    "rvine_10D_125obs", "rvine_10D_500obs", "rvine_10D_2000obs",
    # rvine strong
    "rvine_strong_10D_125obs", "rvine_strong_10D_500obs", "rvine_strong_10D_2000obs",
]

# Parameters needed to reconstruct the data for each run
# (must match what was used in the original run_experiment call)
sample_size  = 10000
max_batchsize = 10000
min_val = -4
max_val = 4
copula_only = True

# ── Helper: check if metric already computed for a run ────────────────────────

def metric_already_computed(run_id, metric_key="auc_global_normed_hessian"):
    """
    Check if the metric was already logged for this run to allow
    restarting the script without recomputing everything.
    """
    run = mlflow.get_run(run_id)
    return metric_key in run.data.metrics

# ── Helper: load run tags needed to reconstruct data ─────────────────────────

def get_run_tags(run_row):
    """
    Extract the data generation tags from an mlflow run row.
    Columns from mlflow.search_runs() are named 'tags.seed_value' etc.
    """
    return {
        "seed_value"       : int(run_row["tags.seed_value"]),
        "seed_value_copula": int(run_row["tags.seed_value_copula"]) 
                             if run_row["tags.seed_value_copula"] not in [None, "None", ""] 
                             else None,
        "dimensionality"   : int(run_row["tags.dimensionality"]),
        "Independence_tree": int(run_row["tags.Independence_tree"]),
        "vine_type"        : run_row["tags.vine_type"],
        "N_train"          : int(run_row["tags.N_train"]),
        "N_validate"       : int(run_row["tags.N_validate"]),
        "N_test"           : int(run_row["tags.N_test"]),
        "tau_mean"         : float(run_row["tags.tau_mean"]),
        "tau_range"        : float(run_row["tags.tau_range"]),
        "negative_tau"     : run_row["tags.negative_tau"] == "True",
    }

# ── Main patching loop ────────────────────────────────────────────────────────

mlflow.set_tracking_uri(mlruns_scratch_uri)

# Collect all runs across all experiments first so tqdm shows total count
all_runs = []
print("Collecting runs across all experiments...")
for exp_name in experiment_names:
    experiment = mlflow.get_experiment_by_name(exp_name)
    if experiment is None:
        print(f"  WARNING: experiment '{exp_name}' not found — skipping.")
        continue
    df_runs = mlflow.search_runs([experiment.experiment_id])
    df_runs = df_runs[df_runs["status"] == "FINISHED"]   # only completed runs
    df_runs["_experiment_name"] = exp_name
    all_runs.append(df_runs)

df_all_runs = pd.concat(all_runs, ignore_index=True)
print(f"Found {len(df_all_runs)} finished runs across {len(experiment_names)} experiments.\n")

# ── Process each run ──────────────────────────────────────────────────────────

pbar = tqdm(df_all_runs.iterrows(), total=len(df_all_runs), desc="Patching runs")

n_success = 0
n_skipped = 0
n_failed  = 0
failed_runs = []

for _, run_row in pbar:

    run_id      = run_row["run_id"]
    exp_name    = run_row["_experiment_name"]
    run_name    = run_row["tags.mlflow.runName"]

    pbar.set_description(f"{exp_name} | {run_name}")

    # ── Skip if already computed ──────────────────────────────────────────────
    if metric_already_computed(run_id, metric_key="auc_global_normed_hessian"):
        tqdm.write(f"  SKIP (already computed): {run_name} [{run_id[:8]}]")
        n_skipped += 1
        continue

    try:
        # ── Extract tags to reconstruct the data ─────────────────────────────
        tags = get_run_tags(run_row)

        # ── Reconstruct the synthetic data (same seed → same data) ───────────
        synthetic_data_dict = generate_synthetic_vine_data(
            seed_value        = tags["seed_value"],
            seed_value_copula = tags["seed_value_copula"],
            dimensionality    = tags["dimensionality"],
            Independence_tree = tags["Independence_tree"],
            vine_type         = tags["vine_type"],
            N_train           = tags["N_train"],
            N_validate        = tags["N_validate"],
            N_test            = tags["N_test"],
            tau_mean          = tags["tau_mean"],
            tau_range         = tags["tau_range"],
            negative_tau      = tags["negative_tau"],
        )

        # ── Load model from mlflow artifact store → map to CPU ────────────────
        model_uri = f"runs:/{run_id}/model"
        model = mlflow.pytorch.load_model(model_uri, map_location=torch.device("cpu"))

        # Ensure model knows it is on CPU (some internal logic may use model.device)
        model.device = "cpu"
        model.to("cpu")
        #model.eval()

        # ── Compute global_normed_hessian for samples, train, val ─────────────
        # Samples from model
        ci_table_hessian_samples = model.compute_conditional_independence_table_local_relative_hessian(
            y                  = None,
            evaluation_data_type = "samples_from_model",
            sample_size        = sample_size,
            min_val            = min_val,
            max_val            = max_val,
            copula_only        = copula_only,
            batchsize          = max_batchsize,
        )

        # Train data
        ci_table_hessian_train = model.compute_conditional_independence_table_local_relative_hessian(
            y          = synthetic_data_dict["train_data"].detach(),
            copula_only = copula_only,
            batchsize   = max_batchsize,
        )

        # Validate data
        ci_table_hessian_val = model.compute_conditional_independence_table_local_relative_hessian(
            y          = synthetic_data_dict["validate_data"].detach(),
            copula_only = copula_only,
            batchsize   = max_batchsize,
        )

        # ── Merge with true structure to get dependence labels ────────────────
        df_structure = synthetic_data_dict["df_true_structure"]

        merged_samples = pd.merge(ci_table_hessian_samples, df_structure, on=["var_row", "var_col"])
        merged_train   = pd.merge(ci_table_hessian_train,   df_structure, on=["var_row", "var_col"])
        merged_val     = pd.merge(ci_table_hessian_val,     df_structure, on=["var_row", "var_col"])

        # Weighted joint data table (same weighting as original run_experiment)
        N_train    = tags["N_train"]
        N_validate = tags["N_validate"]
        portion_train = N_train    / (N_train + N_validate)
        portion_val   = N_validate / (N_train + N_validate)

        ci_table_hessian_data = ci_table_hessian_train.copy()
        ci_table_hessian_data["global_normed_hessian_abs_mean"] = (
            portion_train * ci_table_hessian_train["global_normed_hessian_abs_mean"]
            + portion_val * ci_table_hessian_val["global_normed_hessian_abs_mean"]
        )
        merged_data = pd.merge(ci_table_hessian_data, df_structure, on=["var_row", "var_col"])

        # ── Compute AUC scores ────────────────────────────────────────────────
        auc_global_nhess_samples = roc_auc_score(
            merged_samples["dependence"],
            merged_samples["global_normed_hessian_abs_mean"]
        )
        auc_global_nhess_train = roc_auc_score(
            merged_train["dependence"],
            merged_train["global_normed_hessian_abs_mean"]
        )
        auc_global_nhess_val = roc_auc_score(
            merged_val["dependence"],
            merged_val["global_normed_hessian_abs_mean"]
        )
        auc_global_nhess_data = roc_auc_score(
            merged_data["dependence"],
            merged_data["global_normed_hessian_abs_mean"]
        )

        # ── Log metrics back into the existing run ────────────────────────────
        with mlflow.start_run(run_id=run_id):
            mlflow.log_metric("auc_global_normed_hessian",       auc_global_nhess_samples)
            mlflow.log_metric("auc_global_normed_hessian_train", auc_global_nhess_train)
            mlflow.log_metric("auc_global_normed_hessian_val",   auc_global_nhess_val)
            mlflow.log_metric("auc_global_normed_hessian_data",  auc_global_nhess_data)

        tqdm.write(
            f"  OK  [{run_id[:8]}] {run_name} | "
            f"auc_samples={auc_global_nhess_samples:.4f} | "
            f"auc_data={auc_global_nhess_data:.4f}"
        )
        n_success += 1

    except Exception as e:
        tqdm.write(f"  FAIL [{run_id[:8]}] {run_name} | Reason: {e}")
        failed_runs.append({"run_id": run_id, "run_name": run_name, "exp": exp_name, "error": str(e)})
        n_failed += 1
        continue

# ── Final summary ─────────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print(f"Patching complete.")
print(f"  Success : {n_success}")
print(f"  Skipped : {n_skipped}  (metric already existed)")
print(f"  Failed  : {n_failed}")
if failed_runs:
    print(f"\nFailed runs:")
    for r in failed_runs:
        print(f"  [{r['run_id'][:8]}] {r['exp']} | {r['run_name']} | {r['error']}")
print(f"{'='*60}")