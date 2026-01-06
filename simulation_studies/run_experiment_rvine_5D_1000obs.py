from simulation_studies.run_experiment import run_experiment
import mlflow
import os

if __name__ == "__main__":
    
    # change the working directory to root of the project
    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    experimental_name = "rine_5D_1000obs"
    
    # chekc if experiment exists
    experiment = mlflow.get_experiment_by_name(experimental_name)
    if experiment is not None:
        experiment_id = experiment.experiment_id
    else:
        experiment_id = mlflow.create_experiment(experimental_name)

    
    for seed in [1,2,3]:
        
        run_experiment(
            run_name="rine_5D_1000obs_seed_{}".format(seed),
            experiment_id=experiment_id,
            # Tags
            seed_value=seed,
            dimensionality=5,
            Independence_tree=3,
            vine_type="R-Vine",
            N_train=667,
            N_validate=333,
            N_test=20000,
            # Parameters,
            number_transformation_layers = 1,
            number_decorrelation_layers= 3,
            degree_transformations = 10,
            degree_decorrelation = 15,
            spline_transformation = "bspline",
            spline_decorrelation = "bspline",
            transformation_spline_range = (-10, 10),
            device = "cpu",
            penalty_decorrelation_ridge_param = None,
            penalty_decorrelation_ridge_first_difference = "sample",
            penalty_decorrelation_ridge_second_difference = "sample",
            penalty_transformation_ridge_second_difference = None,
            penalty_lasso_conditional_independence = None,
            adaptive_lasso_weights_matrix=False,
            optimizer="LBFGS",
            learning_rate=1,
            iterations=2000,
            patience=5,
            min_delta=1e-7,
            seperate_copula_training=False,
            max_batches_per_iter=False,
            pretrained_transformation_layer=True,
            n_trials=4,
            temp_folder="./temp",
            study_name=None
        )