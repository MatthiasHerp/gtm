from simulation_studies.run_experiment import run_experiment
import mlflow
from mlflow.tracking import MlflowClient
import torch

if __name__ == "__main__":

    TRACKING = "/mnt/mlruns"
    mlflow.set_tracking_uri(f"file:{TRACKING}")
    client = MlflowClient()

    data_type = "D-Vine"
    dimensionality = [5, 7, 10]
    independent_tree = [2, 2, 3]
    observations = 1000
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Sensitivity grid for the Gamma hyperprior uncertainty
    # (smaller cv = more confident / tighter prior; larger cv = more diffuse)
    cv_grid = [0.25, 0.5, 1.0, 2.0]

    n_seeds = 10

    for J, j_ind in zip(dimensionality, independent_tree):
        experiment_name = f"{data_type}_{J}D_ind_tree{j_ind}_{observations}obs_bgtm"
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment is None:
            experiment_id = mlflow.create_experiment(
                experiment_name,
                artifact_location=f"file:{TRACKING}/{experiment_name}"
            )
        else:
            experiment_id = experiment.experiment_id

        print("Tracking URI:", mlflow.get_tracking_uri())
        print("Experiment artifact_location:", client.get_experiment(experiment_id).artifact_location)

        for seed in range(n_seeds):
            # Keep the same seed across cv values (paired comparison)
            for cv in cv_grid:
                run_name = f"{experiment_name}_seed_{seed}_cv_{cv}"

                run_experiment(
                    run_name=run_name,
                    experiment_id=experiment_id,
                    seed_value=seed,
                    dimensionality=J,
                    Independence_tree=j_ind,
                    vine_type=data_type,
                    N_train=667,
                    N_validate=333,
                    N_test=20000,
                    number_transformation_layers=1,
                    number_decorrelation_layers=3,
                    degree_transformations=10,
                    degree_decorrelation=15,
                    spline_transformation="bspline",
                    spline_decorrelation="bspline",
                    transformation_spline_range=(-10, 10),
                    decorrelation_spline_range=(-10, 10),
                    device=device,

                    # penalties off (Bayesian smoothing instead)
                    penalty_decorrelation_ridge_param=None,
                    penalty_decorrelation_ridge_first_difference="sample",
                    penalty_decorrelation_ridge_second_difference="sample",
                    penalty_transformation_ridge_second_difference=None,
                    penalty_lasso_conditional_independence=None,
                    adaptive_lasso_weights_matrix=False,

                    optimizer_gtm="LBFGS",
                    learning_rate_gtm=1,

                    # <<< sensitivity knob >>>
                    cv=cv,

                    iterations=2000,
                    patience=5,
                    min_delta=1e-7,
                    seperate_copula_training=False,
                    max_batches_per_iter=False,
                    pretrained_transformation_layer=True,
                    n_trials=4,

                    # IMPORTANT: include cv to avoid collisions + keep artifacts separate
                    temp_folder=f"./temp_{data_type}_D{J}_seed{seed}_cv{cv}",

                    study_name=None,
                    posterior_sampling_size_bgtm=1024
                )

                print(f"{run_name} done")

        print("Experiment artifact_location:", client.get_experiment(experiment_id).artifact_location)
