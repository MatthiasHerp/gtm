from gtm import *

# Sample Copulas Package
import pyvinecopulib as pv
# Helpers to generate specfific vines and analyze there conditional independencies
from pyvinecopulib_simulation_helpers import *
# Other Stuff
import numpy as np
import torch


def generate_synthetic_vine_data(seed_value=1, 
                                 dimensionality=10,
                                 Independence_tree=3,
                                 vine_type="R-Vine",
                                 N_train=2000, 
                                 N_validate=2000, 
                                 N_test=20000):
    """
    Generate Synthetic Data from a Randomly Sampled Vine Copula with Gaussian Marginals.

    Args:
        - seed_value (int): Seed for reproducibility.
        - dimensionality (int): Number of dimensions for the vine copula.
        - Independence_tree (int): Tree level from which to set independence copulas to have full conditional independencies in the related pairs.
        - vine_type (str): Type of vine to generate ("R-Vine", "C-Vine", or "D-Vine").
        - N_train (int): Number of training samples.
        - N_validate (int): Number of validation samples.
        - N_test (int): Number of test samples.
        
    Returns:
        dict: A dictionary containing:
            - 'train_data': Training data tensor.
            - 'validate_data': Validation data tensor.
            - 'test_data': Test data tensor.
            - 'loglik_true_train': True log-likelihoods for training data.
            - 'loglik_true_validate': True log-likelihoods for validation data.
            - 'loglik_true_test': True log-likelihoods for test data.
            - 'loglik_true_est_train': Estimated log-likelihoods with the true vine structure using estimated parameters (oracle) for training data.
            - 'loglik_true_est_validate': Estimated log-likelihoods with the true vine structure using estimated parameters (oracle) for validation data.
            - 'loglik_true_est_test': Estimated log-likelihoods with the true vine structure using estimated parameters (oracle) for test data.
            - 'df_true_structure': DataFrame detailing the true dependence structure of the vine copula.
    """
    
    # Sets seeds across packages for reproducibility
    set_seeds(seed_value)
    
    ### 1. Sample Synthetic Copula Data and Compute Likelihoods
    # We sample data from a R-Vine-Copula and add Gaussian marginals. 
    # To create the Vine we use the functions `sample_random_pair_copulas`and `compute_conditional_dependence_table` from our helpers. 
    # Theses allow use to set independence copulas upon Tree 3 to add full conditional independencies as detailed in our Paper. The `df`contains a tables with the copulas, 
    # there parameters and there tree positions in a manner comparable to the conditional independence table later generated with the GTM.
    if vine_type == "R-Vine":
        vine_structure = pv.RVineStructure.simulate(dimensionality)
    elif vine_type == "C-Vine":
        vine_structure = pv.CVineStructure.simulate(dimensionality)
    elif vine_type == "D-Vine":
        vine_structure = pv.DVineStructure.simulate(dimensionality)
    else:
        raise ValueError("vine_type must be one of 'R-Vine', 'C-Vine', or 'D-Vine'")
    
    pair_copulas = sample_random_pair_copulas(dimensionality,Independence_tree=Independence_tree)
    vine_model = pv.Vinecop.from_structure(structure=vine_structure, pair_copulas=pair_copulas)
    df = compute_conditional_dependence_table(vine_model)

    # creating a table to compare the true dependence structure later on to what the gtm learned
    df_true_structure = df[["tree","edge","conditioned variables", "conditioned variable 1", "conditioned variable 2", "dependence", "var_row", "var_col"]]
    df_true_structure_sub = df_true_structure[["var_row", "var_col", "dependence"]]
    df_true_structure_sub.loc[:, "var_row"] = df_true_structure_sub["var_row"] - 1
    df_true_structure_sub.loc[:, "var_col"] = df_true_structure_sub["var_col"] - 1

    
    # Train
    simulated_data_uniform_train = vine_model.simulate(n=N_train)
    simulated_data_train = torch.distributions.Normal(0,1).icdf(torch.tensor(simulated_data_uniform_train)).float()

    # Validate
    simulated_data_uniform_validate = vine_model.simulate(n=N_validate)
    simulated_data_validate = torch.distributions.Normal(0,1).icdf(torch.tensor(simulated_data_uniform_validate)).float()

    # Test
    simulated_data_uniform_test = vine_model.simulate(n=N_test)
    simulated_data_test = torch.distributions.Normal(0,1).icdf(torch.tensor(simulated_data_uniform_test)).float()
    
    
    # Notice we use Sklars Theorem to compute the density of the joint copula and Gaussian marginals density.
    loglik_copula = np.log(vine_model.pdf(simulated_data_uniform_train))
    log_marginals = torch.distributions.Normal(0,1).log_prob(simulated_data_train).sum(1)
    loglik_true_train = torch.tensor(loglik_copula) + log_marginals

    loglik_copula = np.log(vine_model.pdf(simulated_data_uniform_validate))
    log_marginals = torch.distributions.Normal(0,1).log_prob(simulated_data_validate).sum(1)
    loglik_true_validate = torch.tensor(loglik_copula) + log_marginals

    loglik_copula = np.log(vine_model.pdf(simulated_data_uniform_test))
    log_marginals = torch.distributions.Normal(0,1).log_prob(simulated_data_test).sum(1)
    loglik_true_test = torch.tensor(loglik_copula) + log_marginals
    
    # We further estimate the copula on the synthetic data to get an oracle denisity estimator. 
    # Hence an estimator that knows the true underlying structure and merely estiamtes the model parameters.
    copula_pv_est = vine_model
    copula_pv_est.fit(simulated_data_uniform_train)
    means = simulated_data_train.mean(0)
    vars = simulated_data_train.var(0)
    loglik_copula = np.log(copula_pv_est.pdf(simulated_data_uniform_train))
    log_marginals = torch.distributions.Normal(means,vars).log_prob(simulated_data_train).sum(1)
    loglik_true_est_train = torch.tensor(loglik_copula) + log_marginals

    loglik_copula = np.log(copula_pv_est.pdf(simulated_data_uniform_validate))
    log_marginals = torch.distributions.Normal(means,vars).log_prob(simulated_data_validate).sum(1)
    loglik_true_est_validate = torch.tensor(loglik_copula) + log_marginals

    loglik_copula = np.log(copula_pv_est.pdf(simulated_data_uniform_test))
    log_marginals = torch.distributions.Normal(means,vars).log_prob(simulated_data_test).sum(1)
    loglik_true_est_test = torch.tensor(loglik_copula) + log_marginals
    
    
    return {
        "train_data": simulated_data_train,
        "validate_data": simulated_data_validate,
        "test_data": simulated_data_test,
        "loglik_true_train": loglik_true_train,
        "loglik_true_validate": loglik_true_validate,
        "loglik_true_test": loglik_true_test,
        "loglik_true_est_train": loglik_true_est_train,
        "loglik_true_est_validate": loglik_true_est_validate,
        "loglik_true_est_test": loglik_true_est_test,
        "df_true_structure": df_true_structure_sub   
    }