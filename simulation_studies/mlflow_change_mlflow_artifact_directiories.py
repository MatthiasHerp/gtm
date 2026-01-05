"""CLI script to fix artifact paths when `mlruns` directory is moved.
This could happen if you rename a parent directory when running locally, or if
you download an `mlruns/` folder produced on a different machine/file system.
Args:
  path: Path to mlruns folder containing experiments.
"""
import pathlib
import mlflow
import yaml
from pathlib import Path
import argparse


def rewrite_artifact_path(metadata_file, pwd, artifact_path_key):
    with open(metadata_file, "r") as f:
        y = yaml.safe_load(f)
        y[artifact_path_key] = f"file://{pwd}"

    with open(metadata_file, "w") as f:
        #print(yaml.dump(y, default_flow_style=False, sort_keys=False))
        yaml.dump(y, f, default_flow_style=False, sort_keys=False)

def change_artifact_directory_experiment(experiment_id, absolute_path_to_project_root):

    absolute_path = Path(absolute_path_to_project_root + "/mlruns").resolve()

    experiment_folder = pathlib.Path(absolute_path, experiment_id)

    if ".DS_Store" not in str(experiment_folder):
        #print(experiment_folder)

        # Info: we do not fix the experiment metadata file as it is needed to store new runs on the server
        #       we only fix the run metadata files as these are checked for the ui

        #metadata_file = experiment_folder / "meta.yaml"

        # Fix experiment metadata
        #if metadata_file.exists():
        #    rewrite_artifact_path(metadata_file, experiment_folder, artifact_path_key='artifact_location')
        for run_folder in experiment_folder.iterdir():
            metadata_file = run_folder / "meta.yaml"
            #print(run_folder)
            if run_folder == "/Users/maherp/Desktop/Universitaet/Goettingen/5_Semester/master_thesis/mctm_pytorch/mlruns/868040026974936229/" \
                             "22326f5530214fc3995fa8d2bbbb356c":
                print("here")

            # Fix run metadata
            if metadata_file.exists():
                rewrite_artifact_path(metadata_file, run_folder / "artifacts", artifact_path_key='artifact_uri')


if __name__ == "__main__":
    
    import os 
    # change the working directory to root of the project
    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    absolute_path_to_project_root = os.getcwd()

    # Get the list of all experiments
    client = mlflow.tracking.MlflowClient()
    experiments = client.search_experiments()

    # Print the names of all experiments
    experiment_names = [experiment.name for experiment in experiments]
    #print(experiment_names)
    
    #experiment_names = ["rvinegmmsamecopula_dim_4_4000", "rvinegmmsamecopula3_dim_4_4000", "rvine_10_dim_2000obs", "rvine_10_dim",
    #                    "rvineposjoeonly_10_dim_4000obs", "rvineposjoeonly_6_2000obs", "rvineposjoeonly_10_2000obs",
    #                    "rvinegmm_5_dim_4000obs", "rvinegmm_4_dim_4000obs", "rvinegmm_10_dim_4000obs"] 
    
    for experiment_name in experiment_names:
        experiment = mlflow.get_experiment_by_name(experiment_name)
        change_artifact_directory_experiment(experiment.experiment_id, absolute_path_to_project_root)
    


    print("Done!")
