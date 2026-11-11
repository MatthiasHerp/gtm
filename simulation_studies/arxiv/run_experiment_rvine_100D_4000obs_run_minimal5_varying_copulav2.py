from simulation_studies.run_experiment import run_experiment
import mlflow
import os

if __name__ == "__main__":
    
    # change the working directory to root of the project
    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    experimental_name = "rine_100D_4000obs_bootstrap_hydra_varying_copula_minimal5"
    
    # check if experiment exists
    experiment = mlflow.get_experiment_by_name(experimental_name)
    if experiment is not None:
        experiment_id = experiment.experiment_id
    else:
        experiment_id = mlflow.create_experiment(experimental_name)

    
    for seed in range(0,100):
        
        # First run on the data from the seed
        run_experiment(
            run_name="rine_10D_4000obs_seed_{}".format(seed),
            experiment_id=experiment_id,
            # Tags
            seed_value=0, ######## Fix ########
            seed_value_copula=seed, ######## Varying ########
            dimensionality=100,
            Independence_tree=2,
            vine_type="R-Vine",
            N_train=3200,
            N_validate=800,
            N_test=1000, ############# 20000,
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
            hyperparameter_tune=False, ################################
            penalty_decorrelation_ridge_param = None,
            penalty_decorrelation_ridge_first_difference = 29.082814397626066,
            penalty_decorrelation_ridge_second_difference = 29.86738186261798, 
            penalty_transformation_ridge_second_difference = None,
            penalty_lasso_conditional_independence = None,
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
            num_processes=1,
            sample_size = 1000, ################################
            max_num_ci_sample_size = 1000,
            num_points_quad=5, ################################
            copula_only=True,################################
            min_val=-3,################################
            max_val=3################################
        )