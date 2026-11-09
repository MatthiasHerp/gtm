import mlflow
import os

if __name__ == "__main__":
    # Optionally set mlflow tracking uri
    # os.environ["MLFLOW_TRACKING_URI"] = "file:/absolute/path/to/mlruns"
    
    mlflow.set_tracking_uri("file:mlruns")

    # Get all experiments
    experiments = mlflow.tracking.MlflowClient().search_experiments()
    
    if len(experiments) == 0:
        print("No experiments found in this mlruns folder.")
    else:
        for exp in experiments:
            print(f"Experiment name: {exp.name}")
            # Get all runs for this experiment
            runs = mlflow.search_runs([exp.experiment_id])
            print(f"  Number of runs: {len(runs)}")