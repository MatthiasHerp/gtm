from simulation_studies.run_experiment import run_experiment
from simulation_studies.generate_synthetic_vine_data import generate_synthetic_vine_data
import mlflow
import os
import torch

if __name__ == "__main__":
    
    # change the working directory to root of the project
    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    experimental_name = "rine_10D_500obs_bootstrap_hydra_fixed_copula_smaller_penlasso_adpative"
    
    # check if experiment exists
    experiment = mlflow.get_experiment_by_name(experimental_name)
    if experiment is not None:
        experiment_id = experiment.experiment_id
    else:
        experiment_id = mlflow.create_experiment(experimental_name)

    
    for seed in range(22,100):
        
        experimental_name_nopen = "rine_10D_500obs_bootstrap_hydra_fixed_copula_smaller"
        
        # load model from run without lasso pen
        experiment_nopen = mlflow.get_experiment_by_name(experimental_name_nopen) 
        df_runs = mlflow.search_runs([experiment_nopen.experiment_id])
        # subset all runs seed smaller 32
        df_runs = df_runs[df_runs["tags.seed_value"].astype(int) == seed]
        bootstrap_bool = df_runs["tags.bootstrap_warpspeed"].str.contains("True")
        df_run_originals_only = df_runs[~bootstrap_bool]
        
        run_id = df_run_originals_only.iloc[0]["run_id"]

        # load_model
        model = mlflow.pytorch.load_model("runs:/{}/model".format(run_id), map_location=torch.device('cuda'))
        model.device = "cuda"
        model = model.to("cuda")

        dict_data = generate_synthetic_vine_data(seed_value=0,seed_value_copula=int(df_runs["tags.seed_value_copula"].iloc[0]))
        full_data = torch.concat((dict_data["train_data"],dict_data["validate_data"]),dim=0).to("cuda")

        pseudo_con_corr_matrix = model.compute_pseudo_conditional_correlation_matrix(full_data)

        # mean abs
        pseudo_con_corr_matrix = pseudo_con_corr_matrix.abs().mean(0)

        weights = 1 / pseudo_con_corr_matrix

        # set diagonal to zero (only pen off diagonals)
        weights = weights - torch.diag(torch.diag(weights))

                
        # First run on the data from the seed
        run_experiment(
            run_name="rine_10D_500obs_seed_{}".format(seed),
            experiment_id=experiment_id,
            # Tags
            seed_value=seed,
            seed_value_copula=0,
            dimensionality=10,
            Independence_tree=2,
            vine_type="R-Vine",
            N_train=400,
            N_validate=100,
            N_test=20000,
            # Parameters,
            number_transformation_layers = 1,
            number_decorrelation_layers= 3,
            degree_transformations = 10,
            degree_decorrelation = 15,
            spline_transformation = "bspline",
            spline_decorrelation = "bspline",
            transformation_spline_range = (-10, 10),
            decorrelation_spline_range = (-10, 10),
            device = "cuda",
            penalty_decorrelation_ridge_param = None,
            penalty_decorrelation_ridge_first_difference = "sample",
            penalty_decorrelation_ridge_second_difference = "sample",
            penalty_transformation_ridge_second_difference = None,
            penalty_lasso_conditional_independence = "sample",
            adaptive_lasso_weights_matrix=weights,
            optimizer="LBFGS",
            learning_rate=1,
            iterations=2000,
            patience=20,
            min_delta=1e-8,
            seperate_copula_training=False,
            max_batches_per_iter=False,
            pretrained_transformation_layer=True,
            n_trials=40,
            temp_folder="./temp"+experimental_name,
            study_name=None,
            bootstrap_warpspeed=False
        )
        
        ############################################################################################################################
        
        
        df_run_bootstrap_only = df_runs[bootstrap_bool]
        
        run_id = df_run_bootstrap_only.iloc[0]["run_id"]

        # load_model
        model = mlflow.pytorch.load_model("runs:/{}/model".format(run_id), map_location=torch.device('cuda'))
        model.device = "cuda"
        model = model.to("cuda")

        dict_data = generate_synthetic_vine_data(seed_value=0,seed_value_copula=int(df_runs["tags.seed_value_copula"].iloc[0]))
        full_data = torch.concat((dict_data["train_data"],dict_data["validate_data"]),dim=0).to("cuda")

        pseudo_con_corr_matrix = model.compute_pseudo_conditional_correlation_matrix(full_data)

        # mean abs
        pseudo_con_corr_matrix = pseudo_con_corr_matrix.abs().mean(0)

        weights = 1 / pseudo_con_corr_matrix

        # set diagonal to zero (only pen off diagonals)
        weights = weights - torch.diag(torch.diag(weights))
        
        
        # Second run on one bootstrap sample from the data of the seed
        run_experiment(
            run_name="rine_10D_500obs_bootstrap_seed_{}".format(seed),
            experiment_id=experiment_id,
            # Tags
            seed_value=seed,
            seed_value_copula=0,
            dimensionality=10,
            Independence_tree=2,
            vine_type="R-Vine",
            N_train=400,
            N_validate=100,
            N_test=20000,
            # Parameters,
            number_transformation_layers = 1,
            number_decorrelation_layers= 3,
            degree_transformations = 10,
            degree_decorrelation = 15,
            spline_transformation = "bspline",
            spline_decorrelation = "bspline",
            transformation_spline_range = (-10, 10),
            decorrelation_spline_range = (-10, 10),
            device = "cuda",
            penalty_decorrelation_ridge_param = None,
            penalty_decorrelation_ridge_first_difference = "sample",
            penalty_decorrelation_ridge_second_difference = "sample",
            penalty_transformation_ridge_second_difference = None,
            penalty_lasso_conditional_independence = "sample",
            adaptive_lasso_weights_matrix=weights,
            optimizer="LBFGS",
            learning_rate=1,
            iterations=2000,
            patience=20,
            min_delta=1e-8,
            seperate_copula_training=False,
            max_batches_per_iter=False,
            pretrained_transformation_layer=True,
            n_trials=40,
            temp_folder="./temp"+experimental_name,
            study_name=None,
            bootstrap_warpspeed=True
        )