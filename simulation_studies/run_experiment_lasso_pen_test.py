from simulation_studies.run_experiment import run_experiment
import mlflow
import os
import torch

if __name__ == "__main__":
    
    # change the working directory to root of the project
    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    experimental_name = "test"
    
    # check if experiment exists
    experiment = mlflow.get_experiment_by_name(experimental_name)
    if experiment is not None:
        experiment_id = experiment.experiment_id
    else:
        experiment_id = mlflow.create_experiment(experimental_name)
        
        
    experiment_nopen = mlflow.get_experiment_by_name("rvine_10D_125obs")
    df_runs_nopen = mlflow.search_runs([experiment_nopen.experiment_id])
    subset_nopen = df_runs_nopen
     ##################################################################################################################
    # "in" instaed of "not in"
    subset_nopen = subset_nopen[["bootstrap" not in subset_nopen.iloc[i]["tags.mlflow.runName"] for i in range(subset_nopen.shape[0])]]
        

    
    for seed in [0,1,2]: #range(0,100):
        
        
        subset = subset_nopen[subset_nopen["tags.seed_value_copula"].astype(int) == seed]
        
        artifact_uri = subset["artifact_uri"].iloc[0][7:]
        uri = artifact_uri + "/conditional_independence_table_data_train.csv"
        import pandas as pd
        conditional_independence_table = pd.read_csv(uri)
        weights = torch.zeros(10,10)
        weights[conditional_independence_table["var_row"].astype(int), conditional_independence_table["var_col"].astype(int)] = torch.FloatTensor(1 / conditional_independence_table["precision_abs_mean"].values)
        
        #weights = 1 / torch.abs(precision_matrix_train).mean(0)
        # Set all diagonal elements to zero
        #torch.fill_diagonal_(weights, 0)
        
        #weights = torch.FloatTensor(weights)
        weights = weights.to("cpu") # cuda
        
        
        
        # First run on the data from the seed
        run_experiment(
            run_name="rine_10D_500obs_seed_{}".format(seed),
            experiment_id=experiment_id,
            # Tags
            seed_value=0, ######## Fix ########
            seed_value_copula=seed, ######## Varying ########
            dimensionality=10,
            Independence_tree=2,
            vine_type="R-Vine",
            N_train=400,
            N_validate=100,
            N_test=20000,
            # Parameters,
            number_transformation_layers = 1,
            number_decorrelation_layers= 3,
            degree_transformations = 5,
            degree_decorrelation =  10,
            spline_transformation = "bspline",
            spline_decorrelation = "bspline",
            transformation_spline_range = (-5, 5),
            decorrelation_spline_range = (-10, 10),
            device = "cpu", # cuda
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
            n_trials=10, ## 40
            temp_folder="./temp"+experimental_name,
            study_name=None,
            bootstrap_warpspeed=False,
            sample_size = 100,##
            max_num_ci_sample_size = 10000,
            num_points_quad=5,##
            copula_only=True,##
            likelihood_ratio_metrics=False
        )