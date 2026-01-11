import os
import torch
import mlflow

# Import your experiment runner
from simulation_studies.run_experiment import run_experiment


if __name__ == "__main__":
    # ---------------------------------------------------------------------
    # Make sure relative paths (e.g., ./temp) resolve from project root
    # (assumes this file lives in simulation_studies/ or a subfolder)
    # ---------------------------------------------------------------------
    this_file_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(this_file_dir)  # adjust if needed
    os.chdir(project_root)

    # ---------------------------------------------------------------------
    # MLflow experiment setup
    # ---------------------------------------------------------------------
    experiment_name = "rvine_5D_1000obs"
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        experiment_id = mlflow.create_experiment(experiment_name)
    else:
        experiment_id = experiment.experiment_id

    # ---------------------------------------------------------------------
    # Shared configuration
    # ---------------------------------------------------------------------
    device = "cuda" if torch.cuda.is_available() else "cpu"

    base_kwargs = dict(
        experiment_id=experiment_id,

        # Data generation
        dimensionality=5,
        Independence_tree=3,
        vine_type="R-Vine",
        N_train=667,
        N_validate=333,
        N_test=20000,

        # GTM architecture
        number_transformation_layers=1,
        number_decorrelation_layers=3,
        degree_transformations=10,
        degree_decorrelation=15,
        spline_transformation="bspline",
        spline_decorrelation="bspline",
        transformation_spline_range=(-10, 10),
        device=device,

        # Frequentist GTM tuning / training
        penalty_decorrelation_ridge_param=None,
        penalty_decorrelation_ridge_first_difference="sample",
        penalty_decorrelation_ridge_second_difference="sample",
        penalty_transformation_ridge_second_difference=None,
        penalty_lasso_conditional_independence=None,
        adaptive_lasso_weights_matrix=False,
        optimizer_gtm="LBFGS",          # <-- FIX: correct argument name
        learning_rate_gtm=1,
        iterations=2000,
        patience=5,
        min_delta=1e-7,
        seperate_copula_training=False,
        max_batches_per_iter=False,
        pretrained_transformation_layer=True,
        n_trials=4,
        temp_folder="./temp",
        study_name=None,

        # BGTM prior sensitivity
        cv=1,                           # sweep this if you want sensitivity

        # (Optional) BGTM CI-eval settings (keep defaults if you prefer)
        cred_level_bgtm=0.90,           # <-- FIX: correct spelling (was cred_leve_bgtm)
    )

    # ---------------------------------------------------------------------
    # Run a sweep over seeds
    # ---------------------------------------------------------------------
    n_seeds = 10
    for seed in range(n_seeds):
        run_experiment(
            run_name=f"{experiment_name}_seed_{seed}",
            seed_value=seed,
            **base_kwargs,
        )
