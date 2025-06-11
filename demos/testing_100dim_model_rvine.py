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
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    D = 100  # dimension
    rvine_structure = pv.RVineStructure.simulate(D)
    pair_copulas = sample_random_pair_copulas(D,Independence_tree=3)
    vine_model = pv.Vinecop.from_structure(structure=rvine_structure, pair_copulas=pair_copulas)
    
    # Train
    N_train = 10000
    simulated_data_uniform_train = vine_model.simulate(n=N_train)
    simulated_data_train = torch.distributions.Normal(0,1).icdf(torch.tensor(simulated_data_uniform_train)).float()

    # Validate
    N_validate = 10000
    simulated_data_uniform_validate = vine_model.simulate(n=N_validate)
    simulated_data_validate = torch.distributions.Normal(0,1).icdf(torch.tensor(simulated_data_uniform_validate)).float()
    
    # Create dataset and DataLoader
    dataset_train = Generic_Dataset(simulated_data_train)
    dataloader_train = DataLoader(dataset_train, batch_size=N_train)

    dataset_validate = Generic_Dataset(simulated_data_validate)
    dataloader_validate = DataLoader(dataset_validate, batch_size=N_validate)
    
    model = GTM(
        number_variables = 100,
        number_transformation_layers = 1,
        number_decorrelation_layers = 3,
        degree_transformations = 10,
        degree_decorrelation = 20,
        spline_transformation = "bspline",
        spline_decorrelation = "bspline",
        transformation_spline_range = (-15, 15),
        device = device)
                
    model.to(device)                       
                
    #study = model.hyperparameter_tune_penalties( 
    #                            train_dataloader=dataloader_train, 
    #                            validate_dataloader=dataloader_validate, 
    #                            penalty_decorrelation_ridge_param = 0,
    #                            penalty_decorrelation_ridge_first_difference = "sample",
    #                            penalty_decorrelation_ridge_second_difference = "sample",
    #                            penalty_transformation_ridge_second_difference = "sample",
    #                            penalty_lasso_conditional_independence = 0,
    #                            adaptive_lasso_weights_matrix=False,
    #                            iterations=1000, 
    #                            patience=20, 
    #                            min_delta=1e-8, 
    #                            optimizer='LBFGS', 
    #                            pretrained_transformation_layer=True,
    #                            n_trials=40)
    
    penalty_params=torch.FloatTensor([
                            0, #study.best_params["penvalueridge"],
                            0, #study.best_params["penfirstridge"],
                            0, #study.best_params["pensecondridge"],
                            0, #study.best_params["ctm_pensecondridge"]
                            ])

    # pretrain the marginal transformations
    _ = model.pretrain_transformation_layer(dataloader_train, iterations=1000)
    
    # train the joint model
    _ = model.train(train_dataloader=dataloader_train, validate_dataloader=dataloader_validate, 
                        iterations=1000, optimizer="LBFGS",
                        penalty_splines_params=penalty_params)
    
                