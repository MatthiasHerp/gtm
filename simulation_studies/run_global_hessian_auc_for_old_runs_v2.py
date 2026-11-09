import mlflow
import mlflow.pytorch
import torch
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
import os
import tempfile
from pathlib import Path

from gtm import GTM
from simulation_studies.generate_synthetic_vine_data import generate_synthetic_vine_data

# ── Configuration ─────────────────────────────────────────────────────────────

mlruns_scratch_uri = "/Users/matthiasherp/Desktop/phd_github_repositories/gtm/mlruns_scratch"

experiment_names = [
    "rvine_strong_50D_2000obs_maxpen"
    #"cvine_10D_125obs", "cvine_10D_500obs", "cvine_10D_2000obs",
    #"cvine_strong_10D_125obs", "cvine_strong_10D_500obs", "cvine_strong_10D_2000obs",
    #"dvine_10D_125obs", "dvine_10D_500obs", "dvine_10D_2000obs",
    #"dvine_strong_10D_125obs", "dvine_strong_10D_500obs", "dvine_strong_10D_2000obs",
    #"rvine_10D_125obs", "rvine_10D_500obs", "rvine_10D_2000obs",
    #"rvine_strong_10D_125obs", "rvine_strong_10D_500obs", "rvine_strong_10D_2000obs",
]

# The csv artifacts we want to patch with the new column
# maps: artifact filename → which ci_table result to merge in
csv_artifacts_to_patch = {
    "conditional_independence_table_model_samples.csv" : "samples",
    "conditional_independence_table_data.csv"          : "data",
    "conditional_independence_table_train.csv"         : "train",
    "conditional_independence_table_val.csv"           : "val",
}

new_column = "global_normed_hessian_abs_mean"

sample_size   = 10000
max_batchsize = 10000
min_val       = -100
max_val       = 100
copula_only   = True

# ── Helpers ───────────────────────────────────────────────────────────────────

def metric_already_computed(run_id, metric_key="auc_global_normed_hessian"):
    run = mlflow.get_run(run_id)
    return metric_key in run.data.metrics


def get_run_tags(run_row):
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


def download_and_patch_csv(run_id, artifact_filename, new_column_name, new_column_values_df, tmpdir):
    """
    Downloads an existing csv artifact, merges in the new column, and re-uploads it.

    Parameters
    ----------
    run_id              : str
    artifact_filename   : str   e.g. "conditional_independence_table_model_samples.csv"
    new_column_name     : str   e.g. "global_normed_hessian_abs_mean"
    new_column_values_df: pd.DataFrame with columns [var_row, var_col, new_column_name]
    tmpdir              : str   path to a local temp directory for downloading/uploading
    """
    # Download existing artifact to tmpdir
    local_path = mlflow.artifacts.download_artifacts(
        run_id=run_id,
        artifact_path=artifact_filename,
        dst_path=tmpdir,
    )

    # Load existing csv
    df_existing = pd.read_csv(local_path)

    # Drop column if it already exists (re-patching case)
    if new_column_name in df_existing.columns:
        df_existing = df_existing.drop(columns=[new_column_name])

    # Merge in the new column on var_row, var_col
    df_patched = pd.merge(
        df_existing,
        new_column_values_df[["var_row", "var_col", new_column_name]],
        on=["var_row", "var_col"],
        how="left",
    )

    # Save patched csv to tmpdir
    patched_path = os.path.join(tmpdir, artifact_filename)
    df_patched.to_csv(patched_path, index=False)

    # Re-upload to the same run — overwrites the artifact
    mlflow.log_artifact(patched_path)

    return df_patched

# ── Collect all runs ──────────────────────────────────────────────────────────

mlflow.set_tracking_uri(mlruns_scratch_uri)

all_runs = []
print("Collecting runs across all experiments...")
for exp_name in experiment_names:
    experiment = mlflow.get_experiment_by_name(exp_name)
    if experiment is None:
        print(f"  WARNING: experiment '{exp_name}' not found — skipping.")
        continue
    df_runs = mlflow.search_runs([experiment.experiment_id])
    df_runs = df_runs[df_runs["status"] == "FINISHED"]
    df_runs["_experiment_name"] = exp_name
    all_runs.append(df_runs)

df_all_runs = pd.concat(all_runs, ignore_index=True)
print(f"Found {len(df_all_runs)} finished runs across {len(experiment_names)} experiments.\n")

# ── Main patching loop ────────────────────────────────────────────────────────

pbar = tqdm(df_all_runs.iterrows(), total=len(df_all_runs), desc="Patching runs")

n_success = 0
n_skipped = 0
n_failed  = 0
failed_runs = []

for _, run_row in pbar:

    run_id   = run_row["run_id"]
    exp_name = run_row["_experiment_name"]
    run_name = run_row["tags.mlflow.runName"]

    pbar.set_description(f"{exp_name} | {run_name}")

    if metric_already_computed(run_id, metric_key="auc_global_normed_hessian"):
        tqdm.write(f"  SKIP (already computed): {run_name} [{run_id[:8]}]")
        n_skipped += 1
        continue

    try:
        tags = get_run_tags(run_row)

        # ── Reconstruct data ──────────────────────────────────────────────────
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

        # ── Load model → CPU ──────────────────────────────────────────────────
        model_uri = f"runs:/{run_id}/model"
        model = mlflow.pytorch.load_model(model_uri, map_location=torch.device("cpu"))
        model.device = "cpu"
        model.to("cpu")
        model.eval()

        # ── Compute global_normed_hessian for all data splits ─────────────────
        ci_table_hessian_samples = model.compute_conditional_independence_table_local_relative_hessian(
            y                    = None,
            evaluation_data_type = "samples_from_model",
            sample_size          = sample_size,
            min_val              = min_val,
            max_val              = max_val,
            copula_only          = copula_only,
            batchsize            = max_batchsize,
        )
        ci_table_hessian_train = model.compute_conditional_independence_table_local_relative_hessian(
            y           = synthetic_data_dict["train_data"].detach(),
            copula_only = copula_only,
            batchsize   = max_batchsize,
        )
        ci_table_hessian_val = model.compute_conditional_independence_table_local_relative_hessian(
            y           = synthetic_data_dict["validate_data"].detach(),
            copula_only = copula_only,
            batchsize   = max_batchsize,
        )
        # Weighted joint data table
        N_train    = tags["N_train"]
        N_validate = tags["N_validate"]
        portion_train = N_train    / (N_train + N_validate)
        portion_val   = N_validate / (N_train + N_validate)
        ci_table_hessian_data = ci_table_hessian_train.copy()
        ci_table_hessian_data[new_column] = (
            portion_train * ci_table_hessian_train[new_column]
            + portion_val * ci_table_hessian_val[new_column]
        )

        # Map artifact name → the right ci_table to merge in
        ci_table_map = {
            "conditional_independence_table_model_samples.csv" : ci_table_hessian_samples,
            "conditional_independence_table_data.csv"          : ci_table_hessian_data,
            "conditional_independence_table_train.csv"         : ci_table_hessian_train,
            "conditional_independence_table_val.csv"           : ci_table_hessian_val,
        }

        # ── Compute AUC scores ────────────────────────────────────────────────
        df_structure = synthetic_data_dict["df_true_structure"]

        merged_samples = pd.merge(ci_table_hessian_samples, df_structure, on=["var_row", "var_col"])
        merged_train   = pd.merge(ci_table_hessian_train,   df_structure, on=["var_row", "var_col"])
        merged_val     = pd.merge(ci_table_hessian_val,     df_structure, on=["var_row", "var_col"])
        merged_data    = pd.merge(ci_table_hessian_data,    df_structure, on=["var_row", "var_col"])

        auc_global_nhess_samples = roc_auc_score(merged_samples["dependence"], merged_samples[new_column])
        auc_global_nhess_train   = roc_auc_score(merged_train["dependence"],   merged_train[new_column])
        auc_global_nhess_val     = roc_auc_score(merged_val["dependence"],     merged_val[new_column])
        auc_global_nhess_data    = roc_auc_score(merged_data["dependence"],    merged_data[new_column])

        # ── Log metrics + patch csvs inside the existing run ──────────────────
        with tempfile.TemporaryDirectory() as tmpdir:
            with mlflow.start_run(run_id=run_id):

                # # Log AUC metrics
                mlflow.log_metric("auc_global_normed_hessian",       auc_global_nhess_samples)
                mlflow.log_metric("auc_global_normed_hessian_train", auc_global_nhess_train)
                mlflow.log_metric("auc_global_normed_hessian_val",   auc_global_nhess_val)
                mlflow.log_metric("auc_global_normed_hessian_data",  auc_global_nhess_data)

                # Patch each csv artifact with the new column
                for artifact_filename, ci_table in ci_table_map.items():
                    try:
                        download_and_patch_csv(
                            run_id               = run_id,
                            artifact_filename    = artifact_filename,
                            new_column_name      = new_column,
                            new_column_values_df = ci_table,
                            tmpdir               = tmpdir,
                        )
                        tqdm.write(f"    Patched csv: {artifact_filename}")
                    except Exception as csv_e:
                        tqdm.write(f"    WARNING: could not patch '{artifact_filename}': {csv_e}")

        tqdm.write(
            f"  OK  [{run_id[:8]}] {run_name} | "
            f"auc_samples={auc_global_nhess_samples:.4f} | "
            #f"auc_data={auc_global_nhess_data:.4f}"
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