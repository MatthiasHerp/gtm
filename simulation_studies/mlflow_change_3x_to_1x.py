"""
CLI script to convert an MLflow 3.x mlruns directory into an MLflow 1.30 compatible format.

Main fixes:
- Adds run_uuid (renamed to run_id in newer MLflow versions)
- Removes newer MLflow metadata fields unknown to MLflow 1.30
- Fixes artifact_uri paths after moving mlruns folders
"""

import os
import pathlib
from pathlib import Path
import yaml
import mlflow


def rewrite_yaml(metadata_file, updates=None, remove_keys=None):
    with open(metadata_file, "r") as f:
        y = yaml.safe_load(f)

    if updates:
        y.update(updates)

    if remove_keys:
        for key in remove_keys:
            y.pop(key, None)

    with open(metadata_file, "w") as f:
        yaml.dump(
            y,
            f,
            default_flow_style=False,
            sort_keys=False
        )


def fix_run_metadata(run_folder, fix_artifacts=True):
    metadata_file = run_folder / "meta.yaml"

    if not metadata_file.exists():
        return

    with open(metadata_file, "r") as f:
        y = yaml.safe_load(f)

    # MLflow 3.x uses run_id
    # MLflow 1.x requires run_uuid
    if "run_id" in y and "run_uuid" not in y:
        y["run_uuid"] = y["run_id"]

    # Optional: fix artifact path after moving mlruns folder
    if fix_artifacts:
        y["artifact_uri"] = f"file://{run_folder / 'artifacts'}"

    # Remove MLflow 2.x/3.x-only fields if they exist
    remove_keys = [
        "run_name",
        "source_type",
        "source_name",
    ]

    for key in remove_keys:
        y.pop(key, None)

    with open(metadata_file, "w") as f:
        yaml.dump(
            y,
            f,
            default_flow_style=False,
            sort_keys=False
        )


def fix_experiment_metadata(experiment_folder):

    metadata_file = experiment_folder / "meta.yaml"

    if not metadata_file.exists():
        return

    with open(metadata_file, "r") as f:
        y = yaml.safe_load(f)

    # MLflow 1.30 expects artifact_location
    if "artifact_location" in y:
        y["artifact_location"] = (
            f"file://{experiment_folder}"
        )

    # Remove fields introduced later
    remove_keys = [
        "last_update_time",
    ]

    for key in remove_keys:
        y.pop(key, None)

    with open(metadata_file, "w") as f:
        yaml.dump(
            y,
            f,
            default_flow_style=False,
            sort_keys=False
        )


def convert_mlruns(mlruns_path):

    mlruns_path = Path(mlruns_path).resolve()

    print(f"Converting: {mlruns_path}")

    for experiment_folder in mlruns_path.iterdir():

        if not experiment_folder.is_dir():
            continue

        if experiment_folder.name == ".DS_Store":
            continue

        print(f"Experiment: {experiment_folder.name}")

        # Fix experiment metadata
        fix_experiment_metadata(experiment_folder)

        # Fix runs
        for run_folder in experiment_folder.iterdir():

            if not run_folder.is_dir():
                continue

            fix_run_metadata(run_folder)

    print("Conversion finished.")


if __name__ == "__main__":

    os.chdir(
        os.path.dirname(
            os.path.dirname(
                os.path.abspath(__file__)
            )
        )
    )

    # Change this to your copied MLflow folder
    mlruns_folder = "mlruns_scratch"

    convert_mlruns(mlruns_folder)

    print("Done!")