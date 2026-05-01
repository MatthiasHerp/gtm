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

    
    run_experiment_dragon(
            run_name="run_test",
            experiment_id=experiment_id,
            # Tags
            seed_value=1,
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
            iterations=5,######################################################################################################
            patience=5,
            min_delta=1e-7,
            seperate_copula_training=False,
            max_batches_per_iter=False,
            pretrained_transformation_layer=True,
            n_trials=4, ######################################################################################################
            temp_folder="./temp1",
            study_name=None,
            bootstrap_warpspeed=False,
            sample_size = 5,  ######################################################################################################
            max_num_ci_sample_size = 5,  ######################################################################################################
        )
    
    run_experiment_dragon(
            run_name="run_test_lasso_penalty",
            experiment_id=experiment_id,
            # Tags
            seed_value=1,
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
            adaptive_lasso_weights_matrix="sample",
            optimizer="LBFGS",
            learning_rate=1,
            iterations=5, ######################################################################################################
            patience=5,
            min_delta=1e-7,
            seperate_copula_training=False,
            max_batches_per_iter=False,
            pretrained_transformation_layer=True,
            n_trials=4, ######################################################################################################
            temp_folder="./temp1",
            study_name=None,
            bootstrap_warpspeed=False,
            sample_size = 5,  ######################################################################################################
            max_num_ci_sample_size = 5,  ######################################################################################################
        )