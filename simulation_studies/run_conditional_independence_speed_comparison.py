from gtm import *

# Sample Copulas Package
import pyvinecopulib as pv
# Helpers to generate specfific vines and analyze there conditional independencies
from demos.pyvinecopulib_simulation_helpers import *

# Other Stuff
import numpy as np
import torch
from torch.utils.data import DataLoader
from demos.dataset_helpers import Generic_Dataset
from sklearn.metrics import roc_auc_score, roc_curve

if __name__ == "__main__":
    
    device = "cpu"
    
    set_seeds(1)
    
    D = 10  # dimension
    rvine_structure = pv.RVineStructure.simulate(10, seeds = [1])
    pair_copulas = sample_random_pair_copulas(D,Independence_tree=3)
    vine_model = pv.Vinecop.from_structure(structure=rvine_structure, pair_copulas=pair_copulas)
    df = compute_conditional_dependence_table(vine_model)
    
    # Train
    N_train = 2000
    simulated_data_uniform_train = vine_model.simulate(n=N_train)
    simulated_data_train = torch.distributions.Normal(0,1).icdf(torch.tensor(simulated_data_uniform_train)).float()

    # Validate
    N_validate = 2000
    simulated_data_uniform_validate = vine_model.simulate(n=N_validate)
    simulated_data_validate = torch.distributions.Normal(0,1).icdf(torch.tensor(simulated_data_uniform_validate)).float()

    # Test
    N_test = 20000
    simulated_data_uniform_test = vine_model.simulate(n=N_test)
    simulated_data_test = torch.distributions.Normal(0,1).icdf(torch.tensor(simulated_data_uniform_test)).float()
    
    
    # Create dataset and DataLoader
    dataset_train = Generic_Dataset(simulated_data_train)
    dataloader_train = DataLoader(dataset_train, batch_size=N_train)

    dataset_validate = Generic_Dataset(simulated_data_validate)
    dataloader_validate = DataLoader(dataset_validate, batch_size=N_validate)
    
    
    model = GTM(
        number_variables = 10,
        number_transformation_layers = 1,
        number_decorrelation_layers= 3,
        degree_transformations = 5,
        degree_decorrelation = 10,
        spline_transformation = "bspline",
        spline_decorrelation = "bspline",
        transformation_spline_range = (-5, 5),
        decorrelation_spline_range = (-10, 10),
        device = device)
    model.to(device=device)
    
    study = model.hyperparameter_tune_penalties( 
        train_dataloader = dataloader_train,
        validate_dataloader = dataloader_validate,
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
        n_trials=10,
        temp_folder=".",
        study_name=None)
        

    penalty_splines_params=torch.FloatTensor([
                                0, #study.best_params["penalty_decorrelation_ridge_param"],
                                study.best_params["penalty_decorrelation_ridge_first_difference"],
                                study.best_params["penalty_decorrelation_ridge_second_difference"],
                                0 #study.best_params["penalty_transformation_ridge_second_difference"]
                                ])
    penalty_splines_params = penalty_splines_params.to(device=device)
    adaptive_lasso_weights_matrix = False
    penalty_lasso_conditional_independence=False
    
    # pretrain the marginal transformations
    _ = model.pretrain_transformation_layer(dataloader_train, iterations=1000, max_batches_per_iter=False, penalty_splines_params=penalty_splines_params)

    # train the joint model
    _ = model.train(train_dataloader=dataloader_train, validate_dataloader=dataloader_validate, iterations=1000, optimizer="LBFGS",
                    penalty_splines_params=penalty_splines_params, adaptive_lasso_weights_matrix=adaptive_lasso_weights_matrix, penalty_lasso_conditional_independence=penalty_lasso_conditional_independence, 
                    max_batches_per_iter=False)
    
    
    y = None,
    evaluation_data_type = "samples_from_model"
    num_processes=1
    sample_size = 1000
    num_points_quad=10
    copula_only=True
    min_val=-5
    max_val=5
    batch_size=1000
    
    import time
    
    timer_start = time.time()
    conditional_independence_table_samples2 = model.compute_conditional_independence_table_v2(
                                            y = None,
                                            evaluation_data_type = evaluation_data_type,
                                            #num_processes=num_processes,
                                            sample_size = sample_size,
                                            num_points_quad=num_points_quad,
                                            copula_only=copula_only,
                                            min_val=min_val,
                                            max_val=max_val,
                                            batch_size=batch_size)
    timer_end = time.time()
    print(f"Time taken to compute conditional independence table v2 with {sample_size} samples: {timer_end - timer_start} seconds")
            
    timer_start = time.time()
    conditional_independence_table_samples = model.compute_conditional_independence_table(
                                            y = None,
                                            evaluation_data_type = evaluation_data_type,
                                            num_processes=num_processes,
                                            sample_size = sample_size,
                                            num_points_quad=num_points_quad,
                                            copula_only=copula_only,
                                            min_val=min_val,
                                            max_val=max_val)
    timer_end = time.time()
    print(f"Time taken to compute conditional independence table with {sample_size} samples: {timer_end - timer_start} seconds")            