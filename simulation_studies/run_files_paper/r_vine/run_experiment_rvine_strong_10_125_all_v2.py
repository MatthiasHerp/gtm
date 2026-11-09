from simulation_studies.run_experiment import run_experiment
import mlflow
import os
import torch
import pandas as pd

if __name__ == "__main__":
    
    # change the working directory to root of the project
    os.chdir(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    
    experimental_name = "rvine_strong_10D_125obs"
    dimensionality = 10
    vine_type="R-Vine"
    Independence_tree=3
    N=125
    N_train=int(N*0.8)
    N_validate=int(N*0.2)
    tau_mean=0.5
    tau_range=0.2
    negative_tau=True
    
    
    
    # check if experiment exists
    experiment = mlflow.get_experiment_by_name(experimental_name)
    if experiment is not None:
        experiment_id = experiment.experiment_id
    else:
        experiment_id = mlflow.create_experiment(experimental_name)
        
    
    ##################################################################################################################
    # run model with lasso penalty
    ##################################################################################################################    
        
    
    for seed in range(76,100):

        run_experiment(
                run_name="Seed_{}_Lasso".format(seed),
                experiment_id=experiment_id,
                # Tags
                seed_value=seed, ######## Varying ########
                seed_value_copula=seed, ######## Varying ########
                dimensionality=dimensionality,
                Independence_tree=Independence_tree,
                vine_type=vine_type,
                N_train=N_train,
                N_validate=N_validate,
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
                device = "cuda",
                penalty_decorrelation_ridge_param = None,
                penalty_decorrelation_ridge_first_difference = "sample",
                penalty_decorrelation_ridge_second_difference = "sample",
                penalty_transformation_ridge_second_difference = None,
                penalty_lasso_conditional_independence = "sample",
                adaptive_lasso_weights_matrix=False,
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
                bootstrap_warpspeed=False,
                tau_mean=tau_mean,
                tau_range=tau_range,
                negative_tau=negative_tau
        )
 
        
    ##################################################################################################################
    # run model with adaptive lasso penalty
    ##################################################################################################################    
    
    ##################################################################################################################
    # weight matrix
    ##################################################################################################################
    experiment_no_lasso = mlflow.get_experiment_by_name(experimental_name)
    df_runs_no_lasso = mlflow.search_runs([experiment_no_lasso.experiment_id])
    subset_no_lasso = df_runs_no_lasso[["Lasso" not in df_runs_no_lasso.iloc[i]["tags.mlflow.runName"] for i in range(df_runs_no_lasso.shape[0])]]
    
    for seed in range(0,100):
         
        subset = subset_no_lasso[subset_no_lasso["tags.seed_value_copula"].astype(int) == seed]
            
        artifact_uri = subset["artifact_uri"].iloc[0][7:]
        uri = artifact_uri + "/conditional_independence_table_data.csv"
            
        conditional_independence_table = pd.read_csv(uri)
        weights = torch.zeros(10,10)
        weights[conditional_independence_table["var_row"].astype(int), conditional_independence_table["var_col"].astype(int)] = torch.FloatTensor(1 / conditional_independence_table["precision_abs_mean"].values)
            
        weights = weights.to("cuda")
            
        ##################################################################################################################
        ##################################################################################################################

        run_experiment(
                run_name="Seed_{}_Adaptive".format(seed),
                experiment_id=experiment_id,
                # Tags
                seed_value=seed, ######## Varying ########
                seed_value_copula=seed, ######## Varying ########
                dimensionality=dimensionality,
                Independence_tree=Independence_tree,
                vine_type=vine_type,
                N_train=N_train,
                N_validate=N_validate,
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
                device = "cuda",
                penalty_decorrelation_ridge_param = None,
                penalty_decorrelation_ridge_first_difference = "sample",
                penalty_decorrelation_ridge_second_difference = "sample",
                penalty_transformation_ridge_second_difference = None,
                penalty_lasso_conditional_independence = "sample",
                ##################################################################################################################
                adaptive_lasso_weights_matrix=weights,
                ##################################################################################################################
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
                bootstrap_warpspeed=False,
                tau_mean=tau_mean,
                tau_range=tau_range,
                negative_tau=negative_tau
        )