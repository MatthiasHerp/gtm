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

def run_gaussian(seed_value=1,
                 seed_value_copula=1,
                 dimensionality=10,
                 Independence_tree=3,
                 vine_type="R-Vine",
                 N_train=100,
                 N_validate=25,
                 N_test=10000,
                 tau_mean=0.3,
                 tau_range=0.2,
                 negative_tau=True):

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
        negative_tau=negative_tau
    )
    
    y_train = synthetic_data_dict["train_data"]
    y_validate = synthetic_data_dict["validate_data"]
    
    y_all = torch.vstack([y_train, y_validate])
    Y_all = y_all.numpy()

    # Normalize the data
    scaler = StandardScaler()
    Y_all_norm = scaler.fit_transform(Y_all)


    # Plain Gaussian: use empirical covariance
    emp_cov = np.cov(Y_all_norm, rowvar=False)
    gm_precision = np.linalg.inv(emp_cov)
    gm_precision = np.around(gm_precision, decimals=3)
    
    # Conditional correlations
    def p_to_corr(matrix):
        diag_sqrt = np.diag(matrix).reshape(matrix.shape[0],1) ** 0.5
        matrix_std_multiplied = diag_sqrt @ diag_sqrt.T
        return -1 * matrix / matrix_std_multiplied

    gm_c_matrix = p_to_corr(gm_precision)
    
    lower_triangular_indices = np.tril_indices_from(gm_precision, k=-1)
    gm_entries_abs = np.abs(gm_precision[lower_triangular_indices])
    gm_entries_abs_c = np.abs(gm_c_matrix[lower_triangular_indices])

    data = list(zip(
        lower_triangular_indices[0], 
        lower_triangular_indices[1], 
        gm_entries_abs, gm_entries_abs_c
    ))
    df = pd.DataFrame(
        data, 
        columns=[
            'var_row', 'var_col', 
            'gm_p_matrix_abs', 'gm_c_matrix_abs'
        ]
    )
    
    true_ci_graph = synthetic_data_dict["df_true_structure"]
    df = pd.merge(df, true_ci_graph, on=['var_row', 'var_col'])
    
    return df


def run_experiment_gaussian(
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
    device="cpu",
    temp_folder="./temp",
    sample_size=10000,
    max_num_ci_sample_size=10000,
    bootstrap_warpspeed=False,
    threshhold_kld_tails=np.linspace(0, 5, 500),
    tau_mean=0.3,
    tau_range=0.2,
    negative_tau=True
):
    """
    Run a plain Gaussian experiment on synthetic vine copula data and store results using mlflow.
    """
    create_temp_folder(temp_folder)
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
              "negative_tau": negative_tau}
    )

    df = run_gaussian(
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
        negative_tau=negative_tau
    )
    
    df.to_csv(temp_folder+"/conditional_independence_table.csv", index=False)
    mlflow.log_artifact(temp_folder+"/conditional_independence_table.csv")   

    auc_p = roc_auc_score(df["dependence"], df["gm_p_matrix_abs"])
    auc_c = roc_auc_score(df["dependence"], df["gm_c_matrix_abs"])
    
    mlflow.log_metric(key="auc_p", value=auc_p)
    mlflow.log_metric(key="auc_c", value=auc_c)
    
    mlflow.end_run()
    clear_temp_folder(temp_folder)
    

if __name__ == "__main__":
    import os
    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    experimental_name = "test_gaussian"

    experiment = mlflow.get_experiment_by_name(experimental_name)
    if experiment is not None:
        experiment_id = experiment.experiment_id
    else:
        experiment_id = mlflow.create_experiment(experimental_name)
    
    run_experiment_gaussian(
        run_name="test_gaussian",
        experiment_id=experiment_id,
        seed_value=1,
        seed_value_copula=1,
        dimensionality=10,
        Independence_tree=3,
        vine_type="R-Vine",
        N_train=2000,
        N_validate=2000,
        N_test=20000,
        device="cpu",
        temp_folder="./temp",
        sample_size=10000,
        max_num_ci_sample_size=10000,
        bootstrap_warpspeed=False,
        threshhold_kld_tails=np.linspace(0, 5, 500),
        tau_mean=0.3,
        tau_range=0.2,
        negative_tau=True
    )