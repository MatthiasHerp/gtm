from gtm import *

# Sample Copulas Package
import pyvinecopulib as pv
# Helpers to generate specfific vines and analyze there conditional independencies
from demos.pyvinecopulib_simulation_helpers import *
from simulation_studies.generate_synthetic_vine_data import generate_synthetic_vine_data
from simulation_studies.mlflow_plot_storage_helpers import log_mlflow_plot, create_temp_folder, clear_temp_folder

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
import numpy as np
import math

def run_experiment_dragon(
    run_name,
    experiment_id,
    # Tags
    seed_value=1,
    seed_value_copula=None,
    dimensionality=100,
    Independence_tree=3,
    vine_type="R-Vine",
    N_train=2000,
    N_validate=2000,
    N_test=20000,
    # Parameters,
    number_transformation_layers = 1,
    number_decorrelation_layers= 3,
    degree_transformations = 15,
    degree_decorrelation = 30,
    spline_transformation = "bspline",
    spline_decorrelation = "bspline",
    transformation_spline_range = (-10, 10),
    decorrelation_spline_range = (-10, 10),
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
    n_trials=30,
    temp_folder="./temp",
    study_name=None,
    # Evaluation of Conditional Independence parameters
    #evaluation_data_type = "samples_from_model",
    num_processes=1,
    sample_size = 10000,
    max_num_ci_sample_size = 10000,
    num_points_quad=15,
    copula_only=False,
    min_val=-6,
    max_val=6,
    bootstrap_warpspeed=False
):
    """
    Run a GTM experiment on synthetic vine copula data and store results using mlflow.

    Args:
        - seed_value (int): Seed for reproducibility.
        - seed_value_copula (int or None): Seed for fixing the vine copula model (structure, pair copulas, params) that are sampled. If None then each seed_value has a different vine copula.
        - dimensionality (int): Number of dimensions for the vine copula.
        - Independence_tree (int): Tree level from which to set independence copulas to have full conditional independencies in the related pairs.
        - vine_type (str): Type of vine to generate ("R-Vine", "C-Vine", or "D-Vine").
        - N_train (int): Number of training samples.
        - N_validate (int): Number of validation samples.
        - N_test (int): Number of test samples.
        - batch_size (int): Batch size for training.
        - learning_rate (float): Learning rate for the optimizer.
        - num_epochs (int): Maximum number of epochs for training.
        - hidden_layers (list): List defining the hidden layer sizes of the GTM.
        - weight_decay (float): Weight decay for the optimizer.
        - patience (int): Patience for early stopping.
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
                  "bootstrap_warpspeed": bootstrap_warpspeed})
    
    # have to set seed here as we dont do it in simualated data sampling
    set_seeds(seed_value)
    
    data = pd.read_csv("dragon_application/dragon_analysis_dataset_subset.csv")
    shuffled_data = data.sample(frac=1).reset_index(drop=True)
    N = shuffled_data.shape[0]
    D = shuffled_data.shape[1]
    N_train = int(0.75 * N)
    N_validate = int(0.15 * N)
    N_test = N - N_train - N_validate
    data_train = shuffled_data.iloc[:N_train, :]
    data_validate = shuffled_data.iloc[N_train:N_train + N_validate, :]
    data_test = shuffled_data.iloc[N_train + N_validate:N_train + N_validate + N_test, :]

    data_train_numpy = data_train.to_numpy()
    data_validate_numpy = data_validate.to_numpy()
    data_test_numpy = data_test.to_numpy()

    data_train_numpy_log = np.log(data_train_numpy + 0.01)
    data_validate_numpy_log = np.log(data_validate_numpy + 0.01)
    data_test_numpy_log = np.log(data_test_numpy + 0.01)

    data_train_numpy_standardized = ((data_train_numpy_log - data_train_numpy_log.min(axis=0)) / (data_train_numpy_log.max(axis=0) - data_train_numpy_log.min(axis=0)) - 0.5) * 8
    data_validate_numpy_standardized = ((data_validate_numpy_log - data_train_numpy_log.min(axis=0)) / (data_train_numpy_log.max(axis=0) - data_train_numpy_log.min(axis=0)) - 0.5) * 8
    data_test_numpy_standardized = ((data_test_numpy_log - data_train_numpy_log.min(axis=0)) / (data_train_numpy_log.max(axis=0) - data_train_numpy_log.min(axis=0)) - 0.5) * 8
       
    synthetic_data_dict = {}
    synthetic_data_dict['train_data'] = torch.from_numpy(data_train_numpy_standardized).float() 
    synthetic_data_dict['validate_data'] = torch.from_numpy(data_validate_numpy_standardized).float() 
    synthetic_data_dict['test_data'] = torch.from_numpy(data_test_numpy_standardized).float() 
    
    #synthetic_data_dict = generate_synthetic_vine_data(
    #    seed_value=seed_value,
    #    seed_value_copula=seed_value_copula,
    #    dimensionality=dimensionality,
    #    Independence_tree=Independence_tree,
    #    vine_type=vine_type,
    #    N_train=N_train,
    #    N_validate=N_validate,
    #    N_test=N_test
    #)
    #
    ## Create dataset and DataLoader, if bootstrapped note that
    #if bootstrap_warpspeed:
    #    #merge train and validate data for warpspeed bootstrap
    #    combined_data = torch.cat((synthetic_data_dict['train_data'], synthetic_data_dict['validate_data']), dim=0)
    #    
    #    # bootstrap sample with replacement (mit zurücklegen)
    #    indices = torch.randint(0, combined_data.size(0), (N_train + N_validate,))
    #    bootstrapped_data = combined_data[indices]
    #    
    #    # save bootstrapp indices for easy reproducibility
    #    np.save(temp_folder+"/bootstrap_indices.npy", np.array(indices.detach().cpu()))
    #    mlflow.log_artifact(temp_folder+"/bootstrap_indices.npy")
    #    
    #    # Split back into train and validate sets
    #    synthetic_data_dict['train_data'] = bootstrapped_data[:N_train]
    #    synthetic_data_dict['validate_data'] = bootstrapped_data[N_train:]
    #    
    #    # Create dataset and DataLoader
    #    dataset_train = Generic_Dataset(synthetic_data_dict['train_data'])
    #    dataloader_train = DataLoader(dataset_train, batch_size=N_train)
#
    #    dataset_validate = Generic_Dataset(synthetic_data_dict['validate_data'])
    #    dataloader_validate = DataLoader(dataset_validate, batch_size=N_validate)
    #else:
    #    dataset_train = Generic_Dataset(synthetic_data_dict['train_data'])
    #    dataloader_train = DataLoader(dataset_train, batch_size=N_train)
#
    #    dataset_validate = Generic_Dataset(synthetic_data_dict['validate_data'])
    #    dataloader_validate = DataLoader(dataset_validate, batch_size=N_validate)
        
    dataset_train = Generic_Dataset(synthetic_data_dict['train_data'])
    dataloader_train = DataLoader(dataset_train, batch_size=N_train)

    dataset_validate = Generic_Dataset(synthetic_data_dict['validate_data'])
    dataloader_validate = DataLoader(dataset_validate, batch_size=N_validate)
        
    # Run GTM Model Training
    # first store all training parameters
    mlflow.log_param(key="number_transformation_layers", value=number_transformation_layers)
    mlflow.log_param(key="number_decorrelation_layers", value=number_decorrelation_layers)
    mlflow.log_param(key="degree_transformations", value=degree_transformations)
    mlflow.log_param(key="degree_decorrelation", value=degree_decorrelation)
    mlflow.log_param(key="spline_transformation", value=spline_transformation)
    mlflow.log_param(key="spline_decorrelation", value=spline_decorrelation)
    mlflow.log_param(key="transformation_spline_range", value=transformation_spline_range)
    mlflow.log_param(key="decorrelation_spline_range", value=decorrelation_spline_range)
    mlflow.log_param(key="device", value=device)
    mlflow.log_param(key="penalty_decorrelation_ridge_param" , value=penalty_decorrelation_ridge_param)
    mlflow.log_param(key="penalty_decorrelation_ridge_first_difference", value=penalty_decorrelation_ridge_first_difference)
    mlflow.log_param(key="penalty_decorrelation_ridge_second_difference", value=penalty_decorrelation_ridge_second_difference)
    mlflow.log_param(key="penalty_transformation_ridge_second_difference", value=penalty_transformation_ridge_second_difference)
    mlflow.log_param(key="penalty_lasso_conditional_independence", value=penalty_lasso_conditional_independence)
    if adaptive_lasso_weights_matrix is not False:
        mlflow.log_param(key="adaptive_lasso_weights_matrix", value="matrix_passed")
    else:
        mlflow.log_param(key="adaptive_lasso_weights_matrix", value=adaptive_lasso_weights_matrix)
    mlflow.log_param(key="optimizer", value=optimizer)
    mlflow.log_param(key="learning_rate", value=learning_rate)
    mlflow.log_param(key="iterations", value=iterations)
    mlflow.log_param(key="patience", value=patience)
    mlflow.log_param(key="min_delta", value=min_delta)
    mlflow.log_param(key="seperate_copula_training", value=seperate_copula_training)
    mlflow.log_param(key="max_batches_per_iter", value=max_batches_per_iter)
    mlflow.log_param(key="pretrained_transformation_layer", value=pretrained_transformation_layer)
    mlflow.log_param(key="n_trials", value=n_trials)
    mlflow.log_param(key="temp_folder", value=temp_folder)
    mlflow.log_param(key="study_name", value=study_name)   
    
    model = GTM(
        number_variables = dimensionality,
        number_transformation_layers = number_transformation_layers,
        number_decorrelation_layers= number_decorrelation_layers,
        degree_transformations = degree_transformations,
        degree_decorrelation = degree_decorrelation,
        spline_transformation = spline_transformation,
        spline_decorrelation = spline_decorrelation,
        transformation_spline_range = transformation_spline_range,
        decorrelation_spline_range = decorrelation_spline_range,
        device = device)
    model.to(device=device)


    study = model.hyperparameter_tune_penalties( 
        train_dataloader = dataloader_train,
        validate_dataloader = dataloader_validate,
        penalty_decorrelation_ridge_param = penalty_decorrelation_ridge_param,
        penalty_decorrelation_ridge_first_difference = penalty_decorrelation_ridge_first_difference,
        penalty_decorrelation_ridge_second_difference = penalty_decorrelation_ridge_second_difference,
        penalty_transformation_ridge_second_difference = penalty_transformation_ridge_second_difference,
        penalty_lasso_conditional_independence = penalty_lasso_conditional_independence,
        adaptive_lasso_weights_matrix=adaptive_lasso_weights_matrix,
        optimizer=optimizer,
        learning_rate=learning_rate,
        iterations=iterations,
        patience=patience,
        min_delta=min_delta,
        seperate_copula_training=seperate_copula_training,
        max_batches_per_iter=max_batches_per_iter,
        pretrained_transformation_layer=pretrained_transformation_layer,
        n_trials=n_trials,
        temp_folder=temp_folder,
        study_name=study_name)
        
    # for every penalty if we pass a none the set penalty to zero
    if penalty_decorrelation_ridge_param is None:
        penalty_decorrelation_ridge_param_chosen = 0
    elif penalty_decorrelation_ridge_param is float:
        penalty_decorrelation_ridge_param_chosen = penalty_decorrelation_ridge_param
    else:
        penalty_decorrelation_ridge_param_chosen = study.best_params["penalty_decorrelation_ridge_param"]

    if penalty_decorrelation_ridge_first_difference is None:
        penalty_decorrelation_ridge_first_difference_chosen = 0
    elif penalty_decorrelation_ridge_first_difference is float:
        penalty_decorrelation_ridge_first_difference_chosen = penalty_decorrelation_ridge_first_difference
    else:
        penalty_decorrelation_ridge_first_difference_chosen = study.best_params["penalty_decorrelation_ridge_first_difference"]

    if penalty_decorrelation_ridge_second_difference is None:
        penalty_decorrelation_ridge_second_difference_chosen = 0
    elif penalty_decorrelation_ridge_second_difference is float:
        penalty_decorrelation_ridge_second_difference_chosen = penalty_decorrelation_ridge_second_difference
    else:
        penalty_decorrelation_ridge_second_difference_chosen = study.best_params["penalty_decorrelation_ridge_second_difference"]
    
    if penalty_transformation_ridge_second_difference is None:
        penalty_transformation_ridge_second_difference_chosen = 0
    elif penalty_transformation_ridge_second_difference is float:
        penalty_transformation_ridge_second_difference_chosen = penalty_transformation_ridge_second_difference
    else:
        penalty_transformation_ridge_second_difference_chosen = study.best_params["penalty_transformation_ridge_second_difference"]

    penalty_splines_params_chosen=torch.FloatTensor([
                                penalty_decorrelation_ridge_param_chosen,
                                penalty_decorrelation_ridge_first_difference_chosen,
                                penalty_decorrelation_ridge_second_difference_chosen,
                                penalty_transformation_ridge_second_difference_chosen
                                ])
    penalty_splines_params_chosen = penalty_splines_params_chosen.to(device=device)
    
    
    if penalty_lasso_conditional_independence is None:
        penalty_lasso_conditional_independence_chosen = False
    elif penalty_lasso_conditional_independence is float:
        penalty_lasso_conditional_independence_chosen = penalty_lasso_conditional_independence
    else:
        penalty_lasso_conditional_independence_chosen = study.best_params["penalty_lasso_conditional_independence"]
        penalty_lasso_conditional_independence_chosen = torch.FloatTensor([penalty_lasso_conditional_independence_chosen])
        penalty_lasso_conditional_independence_chosen = penalty_lasso_conditional_independence_chosen.to(device=device)
    
    # here we store the adaptive lasso weights matrix as an artifact
    # the same way plots can also be stored for each run    
    if adaptive_lasso_weights_matrix is not False:
        np.save(temp_folder+"/adaptive_lasso_weights_matrix.npy", np.array(adaptive_lasso_weights_matrix.detach().cpu()))
        mlflow.log_artifact(temp_folder+"/adaptive_lasso_weights_matrix.npy")

    # pretrain the marginal transformations
    _ = model.pretrain_transformation_layer(dataloader_train, iterations=iterations, max_batches_per_iter=max_batches_per_iter, penalty_splines_params=penalty_splines_params_chosen, temp_folder=temp_folder)
    
    
    # train the joint model
    training_dict = model.train(train_dataloader=dataloader_train, validate_dataloader=dataloader_validate, iterations=iterations, optimizer=optimizer, learning_rate=learning_rate, patience=patience, min_delta=min_delta,
                penalty_splines_params=penalty_splines_params_chosen, adaptive_lasso_weights_matrix=adaptive_lasso_weights_matrix, penalty_lasso_conditional_independence=penalty_lasso_conditional_independence_chosen, 
                max_batches_per_iter=max_batches_per_iter, temp_folder=temp_folder)
    
    # plot training curves
    plt.plot(training_dict["loss_list_training"], label="Training Loss")
    if "loss_list_validation" in training_dict:
        plt.plot(training_dict["loss_list_validation"], label="Validation Loss")
    plt.xlabel("Iteration")
    plt.ylabel("Negative Log Likelihood")
    plt.title("GTM Training Curve")
    plt.legend()
    fig_train = plt.gcf()
    plt.close()
    # log the plot as an mlflow artifact
    log_mlflow_plot(fig_train, 'training_curves.png', temporary_storage_directory=temp_folder)
    
    # Log trained model
    _ = mlflow.pytorch.log_model(model, "model")
    # can be loaded (to cpu) via: model = mlflow.pytorch.load_model("runs:/{}/model".format(run_id), map_location=torch.device('cpu'))
    
    # store all trained parameters
    mlflow.log_param(key="penalty_decorrelation_ridge_param_chosen", value=penalty_decorrelation_ridge_param_chosen)
    mlflow.log_param(key="penalty_decorrelation_ridge_first_difference_chosen", value=penalty_decorrelation_ridge_first_difference_chosen)
    mlflow.log_param(key="penalty_decorrelation_ridge_second_difference_chosen", value=penalty_decorrelation_ridge_second_difference_chosen)
    mlflow.log_param(key="penalty_transformation_ridge_second_difference_chosen", value=penalty_transformation_ridge_second_difference_chosen)
    if penalty_lasso_conditional_independence_chosen is not False:
        mlflow.log_param(key="penalty_lasso_conditional_independence_chosen", value=penalty_lasso_conditional_independence_chosen.detach().cpu().item())
    else:
        mlflow.log_param(key="penalty_lasso_conditional_independence_chosen", value=penalty_lasso_conditional_independence_chosen)

    ### 4. Compare to Benchmarks

    # We compare the learned GTM to a Gaussian Approximation and the Oracle Model. We expect the GTM to lie between these two in terms of approximation the true underlying distribution.
    # We measure this by means of the Kullback Leibler Divergence which we approximate on the test set which is equivalent to the log likelihood ratio between the true distribution and an approximation of it..
    log_likelihood_train_gtm = model.log_likelihood(synthetic_data_dict['train_data']).detach().mean().cpu()
    log_likelihood_validate_gtm = model.log_likelihood(synthetic_data_dict['validate_data']).detach().mean().cpu()
    log_likelihood_test_gtm = model.log_likelihood(synthetic_data_dict['test_data']).detach().mean().cpu()
    
    mlflow.log_metric(key="log_likelihood_train_gtm", value=log_likelihood_train_gtm)
    mlflow.log_metric(key="log_likelihood_validate_gtm", value=log_likelihood_validate_gtm)
    mlflow.log_metric(key="log_likelihood_test_gtm", value=log_likelihood_test_gtm)
    
    # estimate the Multivariate Normal Distribution as Model
    mean_mvn_model = synthetic_data_dict['train_data'].mean(0)
    cov_mvn_model = synthetic_data_dict['train_data'].T.cov()
    mvn_model = torch.distributions.MultivariateNormal(loc=mean_mvn_model, covariance_matrix=cov_mvn_model)
    log_likelihood_train_gaussian = mvn_model.log_prob(synthetic_data_dict['train_data'])
    log_likelihood_validate_gaussian = mvn_model.log_prob(synthetic_data_dict['validate_data'])
    log_likelihood_test_gaussian = mvn_model.log_prob(synthetic_data_dict['test_data'])

    mlflow.log_metric(key="log_likelihood_train_gaussian", value=log_likelihood_train_gaussian.mean().detach().cpu())
    mlflow.log_metric(key="log_likelihood_validate_gaussian", value=log_likelihood_validate_gaussian.mean().detach().cpu())
    mlflow.log_metric(key="log_likelihood_test_gaussian", value=log_likelihood_test_gaussian.mean().detach().cpu())

    ### 5. Evaluate and Plot GTM Results
    # We evaluate the model further by showing how to generate synthetic samples, plot the conditional correlation patterns as well as identify conditional independencies.
    # To Do so we start by evaluate the conditional indepenedence_relationships. By computing the table we compute pseudo correlation matrix based conditional independence metrics 
    # and likelihood based conditional independence metrics such as the iae and the kld. For more details on these see the paper. This computation may take some time as it computes quadratures under the hood, see the algorithm 1 in the paper.
    
    #mlflow.log_param(key="evaluation_data_type", value=evaluation_data_type)
    mlflow.log_param(key="num_processes", value=num_processes)
    mlflow.log_param(key="sample_size", value=sample_size)
    mlflow.log_param(key="max_num_ci_sample_size", value=max_num_ci_sample_size)
    mlflow.log_param(key="num_points_quad", value=num_points_quad)
    mlflow.log_param(key="copula_only", value=copula_only)
    mlflow.log_param(key="min_val", value=min_val)
    mlflow.log_param(key="max_val", value=max_val)
    
    import time
    
    timer_start = time.time()
    conditional_independence_table_samples2 = model.compute_conditional_independence_table_v2(
                                            y = None,
                                            evaluation_data_type = "samples_from_model",
                                            #num_processes=num_processes,
                                            sample_size = 1000,
                                            num_points_quad=num_points_quad,
                                            copula_only=copula_only,
                                            min_val=min_val,
                                            max_val=max_val)
    timer_end = time.time()
    print(f"Time taken to compute conditional independence table v2 with {sample_size} samples: {timer_end - timer_start} seconds")
    
    timer_start = time.time()
    conditional_independence_table_samples = model.compute_conditional_independence_table(
                                            y = None,
                                            evaluation_data_type = "samples_from_model",
                                            num_processes=num_processes,
                                            sample_size = 1000,
                                            num_points_quad=num_points_quad,
                                            copula_only=copula_only,
                                            min_val=min_val,
                                            max_val=max_val)
    timer_end = time.time()
    print(f"Time taken to compute conditional independence table with {sample_size} samples: {timer_end - timer_start} seconds")
    
    
    
    ##
    ##
    ##
    ##if max_num_ci_sample_size < sample_size:
    ##    # Compute number of chunks
    ##    n_chunks = math.ceil(sample_size / max_num_ci_sample_size)
    ##    print("needs to chunk synthetic sample ci computation, uses {} chunks of size ".format(n_chunks) + str(max_num_ci_sample_size))
    ##    print("is be more then sample_size if sample_size / max_num_ci_sample_size not an integer")
##
    ##    result_tables = []
    ##    for i in range(n_chunks):
    ##        print(f"Processing chunk {i+1}/{n_chunks} ({max_num_ci_sample_size} samples)...")
    ##        chunk_table = model.compute_conditional_independence_table(
    ##            y=None,
    ##            evaluation_data_type="samples_from_model",
    ##            num_processes=num_processes,
    ##            sample_size=max_num_ci_sample_size,
    ##            num_points_quad=num_points_quad,
    ##            copula_only=copula_only,
    ##            min_val=min_val,
    ##            max_val=max_val
    ##        )
    ##        result_tables.append(chunk_table)
##
    ##    # Concatenate all results into one dataframe
    ##    all_results = pd.concat(result_tables, ignore_index=True)
##
    ##    # Aggregate (average) using groupby:
    ##    conditional_independence_table_samples = (
    ##        all_results
    ##        .groupby(["var_row", "var_col"], as_index=False)
    ##        .agg({
    ##            "precision_abs_mean": "mean",
    ##            "precision_square_mean": "mean",
    ##            "cond_correlation_abs_mean": "mean",
    ##            "cond_correlation_square_mean": "mean",
    ##            "kld": "mean",
    ##            "iae": "mean"
    ##        })
    ##    )
    ##else:  
    ##    conditional_independence_table_samples = model.compute_conditional_independence_table(
    ##                                        y = None,
    ##                                        evaluation_data_type = "samples_from_model",
    ##                                        num_processes=num_processes,
    ##                                        sample_size = sample_size,
    ##                                        num_points_quad=num_points_quad,
    ##                                        copula_only=copula_only,
    ##                                        min_val=min_val,
    ##                                        max_val=max_val)
    ##
##
    ##conditional_independence_table_data_train = model.compute_conditional_independence_table(
    ##                                    y = synthetic_data_dict['train_data'].detach(),
    ##                                    evaluation_data_type = "data",
    ##                                    num_processes=num_processes,
    ##                                    sample_size = sample_size,
    ##                                    num_points_quad=num_points_quad,
    ##                                    copula_only=copula_only,
    ##                                    min_val=min_val,
    ##                                    max_val=max_val)
    ##
##
    ##conditional_independence_table_data_val = model.compute_conditional_independence_table(
    ##                                    y = synthetic_data_dict['validate_data'].detach(),
    ##                                    evaluation_data_type = "data",
    ##                                    num_processes=num_processes,
    ##                                    sample_size = sample_size,
    ##                                    num_points_quad=num_points_quad,
    ##                                    copula_only=copula_only,
    ##                                    min_val=min_val,
    ##                                    max_val=max_val)
    ##
    ##### 6. Identifying the Conditional Independence Graph 
    ### We compare the true known conditional independence Graph to the one learned by the GTM. To do so we first merge the true structure table with our learned one.
    ##
    ###merged_ci_tables_samples = pd.merge(
    ###    conditional_independence_table_samples,
    ###    synthetic_data_dict["df_true_structure"],
    ###    on=["var_row", "var_col"]
    ###)
###
    ###merged_ci_tables_data_train = pd.merge(
    ###    conditional_independence_table_data_train,
    ###    synthetic_data_dict["df_true_structure"],
    ###    on=["var_row", "var_col"]
    ###)
    ##
    ##merged_ci_tables_samples = conditional_independence_table_samples
    ##merged_ci_tables_data_train = conditional_independence_table_data_train
    ##
##
    ### the iae makes no sense when using the true data, and the kld is the log likelihood ratio so we del iae and rename kld into ll_diff
    ##del merged_ci_tables_data_train["iae"]
    ##merged_ci_tables_data_train["ll_diff"] = merged_ci_tables_data_train["kld"]
    ##del merged_ci_tables_data_train["kld"]
    ##
    ###merged_ci_tables_data_val = pd.merge(
    ###    conditional_independence_table_data_val,
    ###    synthetic_data_dict["df_true_structure"],
    ###    on=["var_row", "var_col"]
    ###)
    ##del merged_ci_tables_data_val["iae"]
    ##merged_ci_tables_data_val["ll_diff"] = merged_ci_tables_data_val["kld"]
    ##del merged_ci_tables_data_val["kld"]
    ##
    ##merged_ci_tables_data_val = conditional_independence_table_data_val
    ##
    ### store the merged table as an artifact
    ##merged_ci_tables_samples.to_csv(temp_folder+"/conditional_independence_table_model_samples.csv", index=False)
    ##mlflow.log_artifact(temp_folder+"/conditional_independence_table_model_samples.csv")   
    ##
    ##conditional_independence_table_data_train.to_csv(temp_folder+"/conditional_independence_table_data_train.csv", index=False)
    ##mlflow.log_artifact(temp_folder+"/conditional_independence_table_data_train.csv")   
    ##
    ##conditional_independence_table_data_val.to_csv(temp_folder+"/conditional_independence_table_data_validate.csv", index=False)
    ##mlflow.log_artifact(temp_folder+"/conditional_independence_table_data_validate.csv")   
##
    mlflow.end_run()
    
    clear_temp_folder(temp_folder)
    
