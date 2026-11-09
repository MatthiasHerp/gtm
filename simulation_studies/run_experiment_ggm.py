from gtm import *

# Sample Copulas Package
import pyvinecopulib as pv
# Helpers to generate specfific vines and analyze there conditional independencies
from demos.pyvinecopulib_simulation_helpers import *
from simulation_studies.generate_synthetic_vine_data import generate_synthetic_vine_data
from simulation_studies.mlflow_plot_storage_helpers import log_mlflow_plot, create_temp_folder, clear_temp_folder
from simulation_studies.tail_assesment import kld_tails_assessment

# Other Stuff
import mlflow as mlflow
import numpy as np
import torch
from torch.utils.data import DataLoader
from demos.dataset_helpers import Generic_Dataset
from sklearn.metrics import roc_auc_score

import mlflow
import torch
import matplotlib.pyplot as plt

import pandas as pd
import math

from inverse_covariance import QuicGraphLassoCV, QuicGraphicalLassoCV, QuicGraphicalLasso
from sklearn.preprocessing import StandardScaler

def p_to_corr(matrix):

            diag_sqrt = np.diag(matrix).reshape(matrix.shape[0],1) ** 0.5
            matrix_std_multiplied = diag_sqrt @ diag_sqrt.T
            
            return -1 * matrix / matrix_std_multiplied
        
        
def run_graph_lasso(seed_value=1,
                    seed_value_copula=1,
        dimensionality=10,
        Independence_tree=3,
        vine_type="R-Vine",
        N_train=100,
        N_validate=25,
        N_test=10000,
        tau_mean=0.3,
        tau_range=0.2,
        negative_tau=True,
        gauss_copula_only=False):

    synthetic_data_dict = generate_synthetic_vine_data(
        seed_value=seed_value,
        seed_value_copula=seed_value_copula,
        dimensionality=dimensionality,
        Independence_tree=Independence_tree,
        vine_type=vine_type,
        N_train=N_train,
        N_validate=N_validate,
        N_test=N_test,
        tau_mean=tau_mean, 
        tau_range=tau_range, 
        negative_tau=negative_tau,
        gauss_copula_only=gauss_copula_only
    )
    
    y_train = synthetic_data_dict["train_data"]
    y_validate = synthetic_data_dict["validate_data"]
    
    
    y_all = torch.vstack([y_train, y_validate])
        
    Y_all = y_all.numpy()
    # Normalize the data
    scaler = StandardScaler()
    Y_all_norm = scaler.fit_transform(Y_all)
    
    
    ###########################################################################
    ############ GGM
    ###########################################################################
    
    cv_model = QuicGraphicalLassoCV().fit(Y_all_norm)
    cov = QuicGraphicalLasso(lam=cv_model.lam_).fit(Y_all)
        #lam=int or np.ndarray,  # (optional) Initial penalty (scalar or matrix) 
        #lams=int or list,       # If int, determines the number of grid points 
                                # per refinement
        #n_refinements=int,      # Number of times the grid is refined
        #cv=int,                 # Number of folds or sklearn CV object
        #score_metric=str,       # One of 'log_likelihood' (default), 'frobenius', 
                                #        'kl', or 'quadratic'
        #init_method=str,        # Initial covariance estimate: 'corrcoef' or 'cov'
        # # X is data matrix (n_samples, n_features) 
        
    p_matrix = cov.precision_ #emp_cov
    
    p_matrix = np.around(p_matrix, decimals=3) #get_precision(), decimals=3)
    
    # Get the lower triangular indices, including the diagonal
    lower_triangular_indices = np.tril_indices_from(p_matrix, k=-1)
    
    # results in the precision matrix elements being the conditional correlations
    p_matrix = p_matrix
    c_matrix = p_to_corr(p_matrix)

    # Extract the corresponding entries from the precision matrix
    entries_abs = np.abs(p_matrix[lower_triangular_indices])
    entries_abs_c = np.abs(c_matrix[lower_triangular_indices])
    
    ###########################################################################
    ############ GM
    ###########################################################################
    
    # Plain Gaussian: use empirical covariance
    emp_cov = np.cov(Y_all_norm, rowvar=False)
    gm_precision = np.linalg.inv(emp_cov)
    gm_precision = np.around(gm_precision, decimals=3)

    gm_c_matrix = p_to_corr(gm_precision)
    
    lower_triangular_indices = np.tril_indices_from(gm_precision, k=-1)
    gm_entries_abs = np.abs(gm_precision[lower_triangular_indices])
    gm_entries_abs_c = np.abs(gm_c_matrix[lower_triangular_indices])
    
    ###########################################################################
    
    # Create a list of tuples (row_number, col_number, precision_matrix_entry)
    data = list(zip(lower_triangular_indices[0], lower_triangular_indices[1], entries_abs, entries_abs_c,
                    gm_entries_abs, gm_entries_abs_c))

    # Convert the list of tuples into a Pandas DataFrame
    df = pd.DataFrame(data, columns=['var_row', 'var_col', 'ggm_p_matrix_abs', 'ggm_c_matrix_abs', 'gm_p_matrix_abs', 'gm_c_matrix_abs'])
    
    true_ci_graph = synthetic_data_dict["df_true_structure"]
    df = pd.merge(df, true_ci_graph, on=['var_row', 'var_col'])
    
    return df



def run_experiment_ggm(
    run_name,
    experiment_id,
    # Tags
    seed_value=1,
    seed_value_copula=None,
    dimensionality=10,
    Independence_tree=3,
    vine_type="R-Vine",
    N_train=2000,
    N_validate=2000,
    N_test=20000,
    # Parameters,
    device = "cpu",
    temp_folder="./temp",
    # Evaluation of Conditional Independence parameters
    #evaluation_data_type = "samples_from_model",
    sample_size = 10000,
    max_num_ci_sample_size = 10000,
    bootstrap_warpspeed=False,
    threshhold_kld_tails = np.linspace(0, 5, 500),
    tau_mean=0.3,
    tau_range=0.2,
    negative_tau=True,
    gauss_copula_only=False
):
    """
    Run a GGM experiment on synthetic vine copula data and store results using mlflow.

    """
    
    create_temp_folder(temp_folder)
    
    # Start run, name it, add to correct experiment and add tags which we define as the parameters of the data generation
    run = mlflow.start_run(
            run_name="{}".format(run_name),
            experiment_id=experiment_id,
            tags={"seed_value": seed_value,
                  "seed_value_copula": seed_value_copula,
                  "dimensionality": dimensionality,
                  "Independence_tree": Independence_tree,
                  "vine_type": vine_type,
                  "N_train": N_train,
                  "N_validate": N_validate,
                  "N_test": N_test,
                  "bootstrap_warpspeed": bootstrap_warpspeed,
                  "tau_mean": tau_mean,
                  "tau_range": tau_range,
                  "negative_tau": negative_tau})

    df = run_graph_lasso(seed_value=seed_value,
                            seed_value_copula=seed_value_copula,
                            dimensionality=dimensionality,
                            Independence_tree=Independence_tree,
                            vine_type=vine_type,
                            N_train=N_train,
                            N_validate=N_validate,
                            N_test=N_test,
                            tau_mean=tau_mean,
                            tau_range=tau_range,
                            negative_tau=negative_tau,
                            gauss_copula_only=gauss_copula_only)
    
    df.to_csv(temp_folder+"/conditional_independence_table.csv", index=False)
    mlflow.log_artifact(temp_folder+"/conditional_independence_table.csv")   

    auc_p = roc_auc_score(df["dependence"], df["ggm_p_matrix_abs"])
    auc_c = roc_auc_score(df["dependence"], df["ggm_c_matrix_abs"])
    auc_p_gm = roc_auc_score(df["dependence"], df["gm_p_matrix_abs"])
    auc_c_gm = roc_auc_score(df["dependence"], df["gm_c_matrix_abs"])
    
    mlflow.log_metric(key="auc_p", value=auc_p)
    mlflow.log_metric(key="auc_c", value=auc_c)
    mlflow.log_metric(key="auc_p_gm", value=auc_p_gm)
    mlflow.log_metric(key="auc_c_gm", value=auc_c_gm)
    
    mlflow.end_run()
    
    clear_temp_folder(temp_folder)
    





if __name__ == "__main__":
    
    import os
    
    # change the working directory to root of the project
    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    experimental_name = "test"
    
    # check if experiment exists
    experiment = mlflow.get_experiment_by_name(experimental_name)
    if experiment is not None:
        experiment_id = experiment.experiment_id
    else:
        experiment_id = mlflow.create_experiment(experimental_name)
    
    
    run_experiment_ggm(
    run_name="test",
    experiment_id=experiment_id,
    # Tags
    seed_value=1,
    seed_value_copula=1,
    dimensionality=10,
    Independence_tree=3,
    vine_type="R-Vine",
    N_train=2000,
    N_validate=2000,
    N_test=20000,
    # Parameters,
    device = "cpu",
    temp_folder="./temp",
    # Evaluation of Conditional Independence parameters
    #evaluation_data_type = "samples_from_model",
    sample_size = 10000,
    max_num_ci_sample_size = 10000,
    bootstrap_warpspeed=False,
    threshhold_kld_tails = np.linspace(0, 5, 500),
    tau_mean=0.3,
    tau_range=0.2,
    negative_tau=True
)