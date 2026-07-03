from dragon_application.run_experiment_dragon import run_experiment_dragon
import mlflow
import os

if __name__ == "__main__":
    
    # change the working directory to root of the project
    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    experimental_name = "dragon_application"
    
    # check if experiment exists
    experiment = mlflow.get_experiment_by_name(experimental_name)
    if experiment is not None:
        experiment_id = experiment.experiment_id
    else:
        experiment_id = mlflow.create_experiment(experimental_name)


    seed_value = 1
    
    run_experiment_dragon(
            run_name="dragon_no",
            experiment_id=experiment_id,
            # Tags
            seed_value=seed_value,
            dimensionality=100,
            train_portion=0.75,
            val_portion=0.15,
            test_portion=0.1,
            # Parameters,
            number_transformation_layers = 1,
            number_decorrelation_layers= 3,
            degree_transformations = 15,
            degree_decorrelation = 10,
            spline_transformation = "bspline",
            spline_decorrelation = "bspline",
            transformation_spline_range = (-5, 5),
        decorrelation_spline_range = (-10, 10),
            device = "cuda",
            penalty_decorrelation_ridge_param = None,
            penalty_decorrelation_ridge_first_difference = "sample",
            penalty_decorrelation_ridge_second_difference = "sample",
            penalty_transformation_ridge_second_difference = None,
            penalty_lasso_conditional_independence = None,
            adaptive_lasso_weights_matrix=False,
            optimizer="LBFGS",
            learning_rate=1,
            iterations=2000,
            patience=20,
            min_delta=1e-7,
            seperate_copula_training=False,
            max_batches_per_iter=False,
            pretrained_transformation_layer=True,
            n_trials=40,
            temp_folder="./temp",
            study_name=None,
            # Evaluation of Conditional Independence parameters
            #evaluation_data_type = "samples_from_model",
            num_processes=1,
            sample_size = 10000,
            max_batchsize=1000,
            num_points_quad=15,
            copula_only=True,
            min_val=-100, # fine to have large bounds as we dont do lr metrics
            max_val=100,
            bootstrap=False,
            likelihood_ratio_metrics=False)
    
    
    df_runs = mlflow.search_runs([experiment_id])
     ##################################################################################################################
    # "in" instaed of "not in"
    subset_no = df_runs[["no" in df_runs.iloc[i]["tags.mlflow.runName"] for i in range(df_runs.shape[0])]]
    subset_no = subset_no[["bootstrap" not in subset_no.iloc[i]["tags.mlflow.runName"] for i in range(subset_no.shape[0])]]
        
    artifact_uri = subset_no["artifact_uri"].iloc[0][7:]
    uri = artifact_uri + "/conditional_independence_table_model_samples.csv" #because we dont store train samples
    import pandas as pd
    import torch
    conditional_independence_table = pd.read_csv(uri)
    weights = torch.zeros(10,10)
    weights[conditional_independence_table["var_row"].astype(int), conditional_independence_table["var_col"].astype(int)] = torch.FloatTensor(1 / conditional_independence_table["precision_abs_mean"].values)
        
    #weights = 1 / torch.abs(precision_matrix_train).mean(0)
    # Set all diagonal elements to zero
    #torch.fill_diagonal_(weights, 0)
        
    #weights = torch.FloatTensor(weights)
    weights = weights.to("cuda") # cuda
    
    run_experiment_dragon(
            run_name="dragon_adaptive",
            experiment_id=experiment_id,
            # Tags
            seed_value=seed_value,
            dimensionality=100,
            train_portion=0.75,
            val_portion=0.15,
            test_portion=0.1,
            # Parameters,
            number_transformation_layers = 1,
            number_decorrelation_layers= 3,
            degree_transformations = 15,
            degree_decorrelation = 10,
            spline_transformation = "bspline",
            spline_decorrelation = "bspline",
            transformation_spline_range = (-5, 5),
            decorrelation_spline_range = (-10, 10),
            device = "cuda",
            penalty_decorrelation_ridge_param = None,
            penalty_decorrelation_ridge_first_difference = "sample",
            penalty_decorrelation_ridge_second_difference = "sample",
            penalty_transformation_ridge_second_difference = None,
            penalty_lasso_conditional_independence = None,
            adaptive_lasso_weights_matrix=weights,
            optimizer="LBFGS",
            learning_rate=1,
            iterations=2000,
            patience=20,
            min_delta=1e-7,
            seperate_copula_training=False,
            max_batches_per_iter=False,
            pretrained_transformation_layer=True,
            n_trials=40,
            temp_folder="./temp",
            study_name=None,
            # Evaluation of Conditional Independence parameters
            #evaluation_data_type = "samples_from_model",
            num_processes=1,
            sample_size = 10000,
            max_batchsize=1000,
            num_points_quad=15,
            copula_only=True,
            min_val=-100, # fine to have large bounds as we dont do lr metrics
            max_val=100,
            bootstrap=False,
            likelihood_ratio_metrics=False)
    
    
    
    