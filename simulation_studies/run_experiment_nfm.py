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
import numpy as np
import math
import normflows as nf
from tqdm import tqdm
from gtm.gtm_plots_analysis.compute_conditional_independence_kld_v2 import compute_conditional_independence_kld_v2, setup_quadrature, compute_single_var_marginals, compute_pair_metrics, compute_two_var_marginal


# Set up model
def create_spline_flow(latent_size=10, 
                       number_of_layers=3,
                       hidden_layers=2, 
                       hidden_units=10, 
                       device="cpu"):
    torch.manual_seed(0)
    K = number_of_layers

    latent_size = latent_size
    hidden_units = hidden_units
    hidden_layers = hidden_layers

    flows = []
    for i in range(K):
        
        flows += [nf.flows.AutoregressiveRationalQuadraticSpline(latent_size, hidden_layers, hidden_units)]
        flows += [nf.flows.LULinearPermute(latent_size)]

    # Set base distribuiton
    q0 = nf.distributions.DiagGaussian(latent_size, trainable=False)
        
    # Construct flow model
    nfm = nf.NormalizingFlow(q0=q0, flows=flows)

    # Move model on GPU if available
    #enable_cuda = True
    #device = torch.device('cuda' if torch.cuda.is_available() and enable_cuda else 'cpu')
    nfm = nfm.to(device)
    
    return nfm


def create_maf_flow(latent_size=10, 
                    number_of_layers=3, 
                    hidden_layers=2, 
                    hidden_units=10, 
                    device="cpu"):
    torch.manual_seed(0)
    
    flows = []
    for i in range(number_of_layers):
              
        # Construct flow model
        flows += [nf.flows.MaskedAffineAutoregressive(
            features=latent_size,
            hidden_features=hidden_units,
            #context_features=None,
            num_blocks=hidden_layers,
            use_residual_blocks=False,
            #random_mask=False,
            #activation=F.relu,
            #dropout_probability=0.0,
            use_batch_norm=True,
        )]
        flows += [nf.flows.LULinearPermute(latent_size)]
        
    # Set base distribuiton
    q0 = nf.distributions.DiagGaussian(latent_size, trainable=False)
        
    # Construct flow model
    nfm = nf.NormalizingFlow(q0=q0, flows=flows)

    # Move model on GPU if available
    #enable_cuda = True
    #device = torch.device('cuda' if torch.cuda.is_available() and enable_cuda else 'cpu')
    nfm = nfm.to(device)
    
    return nfm


# Train model

def train_nf(nfm, simulated_data_train, simulated_data_validate,
             early_stopping_patience =  30,
             max_iter = 2000,
             lr=1e-4, 
             weight_decay=1e-6,
             min_delta=1e-7
             ):


    loss_hist = np.array([])
    loss_hist_val = np.array([])

    val_loss_min = torch.inf
    early_stopper=0
    optimizer = torch.optim.Adam(nfm.parameters(), lr=lr, weight_decay=weight_decay)
    for it in tqdm(range(max_iter)):
        optimizer.zero_grad()
        
        ## Get training samples
        #x_np, _ = make_moons(num_samples, noise=0.1)
        #x = torch.tensor(x_np).float().to(device)
        
        # Compute loss
        loss = nfm.forward_kld(simulated_data_train)
        
        # validation set based early stopping
        with torch.no_grad():
            val_loss_current = nfm.forward_kld(simulated_data_validate)
            if val_loss_current + min_delta < val_loss_min:
                val_loss_min = val_loss_current
                best_model_dict = nfm.state_dict().copy()
                early_stopper = 0
            else:
                early_stopper += 1
                if early_stopper >= early_stopping_patience:
                    print(f"Early stopping at iteration {it} with validation loss {val_loss_min.item():.4f}")
                    break

        # Log loss
        loss_hist = np.append(loss_hist, loss.to('cpu').data.numpy())
        loss_hist_val = np.append(loss_hist_val, val_loss_current.to('cpu').data.numpy())
        
        loss.backward()
        optimizer.step()
        
    nfm.load_state_dict(best_model_dict)
    
    return nfm, loss_hist, loss_hist_val


def hyperparameter_search(simulated_data_train, simulated_data_validate, latent_size, 
                          early_stopping_patience, max_iter, lr, min_delta, 
                          weight_decay_list=[1e-5, 1e-6, 1e-7],number_of_layers_list=[3,4,5], hidden_layers_list=[2,3,4], hidden_units_list=[10,20,30], nf_model="maf"):
    results = []
    for number_of_layers in number_of_layers_list:
        for hidden_layer in hidden_layers_list:
            for hidden_unit in hidden_units_list:
                for weight_decay in weight_decay_list:
                    print(f"Training {nf_model.upper()} with {number_of_layers} layers, {hidden_layer} hidden layers {hidden_unit} hidden units and weight decay {weight_decay}.")
                    if nf_model == "maf":
                        nfm = create_maf_flow(number_of_layers=number_of_layers, latent_size=latent_size, hidden_layers=hidden_layer, hidden_units=hidden_unit)
                    elif nf_model == "snf":
                        nfm = create_spline_flow(number_of_layers=number_of_layers, latent_size=latent_size, hidden_layers=hidden_layer, hidden_units=hidden_unit)
                    nfm_trained, loss_hist, loss_hist_val = train_nf(nfm, simulated_data_train, simulated_data_validate,
                                                                    early_stopping_patience =  early_stopping_patience,
                                                                    max_iter = max_iter,
                                                                    lr=lr, 
                                                                    weight_decay=weight_decay,
                                                                    min_delta=min_delta)
                results.append({
                    "number_of_layers": number_of_layers,
                    "hidden_layer": hidden_layer,
                    "hidden_unit": hidden_unit,
                    "weight_decay": weight_decay,
                    "model": nfm_trained,
                    "loss_val": loss_hist_val[-1],
                    "loss_hist": loss_hist,
                    "loss_hist_val": loss_hist_val
                })
    return results



def compute_single_var_marginals(model, data, quad_x, quad_w, batch_size=None):

    N, D = data.shape
    Q = quad_x.shape[0]
    single_var_marginals = torch.zeros(D, N, device=model.device)

    # log weights computed once
    log_w = torch.log(quad_w)  # shape (Q,)

    with tqdm(total=D, desc="Single var marginals", position=1, leave=False) as pbar:
        for col_idx in range(D):

            if batch_size is None:
                batch_size_eff = N
            else:
                batch_size_eff = batch_size

            marginal_col = torch.zeros(N, device=model.device)

            for start_idx in range(0, N, batch_size_eff):
                end_idx = min(start_idx + batch_size_eff, N)
                batch_n = end_idx - start_idx

                # build chunk on the fly
                data_chunk = data[start_idx:end_idx]                                          # shape (batch_n, D)
                expanded_data_chunk = data_chunk.unsqueeze(1).expand(
                    batch_n, Q, D
                ).reshape(batch_n * Q, D).clone()                                             # shape (batch_n*Q, D)

                # expand quad points and log weights
                # (Q,) -> (batch_n, Q) -> (batch_n*Q,)
                expanded_points     = quad_x.unsqueeze(0).expand(batch_n, Q).reshape(batch_n * Q)   # shape (batch_n*Q,)
                expanded_log_w      = log_w.unsqueeze(0).expand(batch_n, Q).reshape(batch_n * Q)    # shape (batch_n*Q,)

                # replace column
                expanded_data_chunk[:, col_idx] = expanded_points

                # log likelihood
                ll_chunk = model.log_prob(
                    expanded_data_chunk
                )                                                                              # shape (batch_n*Q,)

                # log-sum-exp over Q points
                # ll_chunk + log_w: shape (batch_n*Q,) -> (batch_n, Q)
                log_integrand = (ll_chunk + expanded_log_w).view(batch_n, Q)                  # shape (batch_n, Q)
                marginal_col[start_idx:end_idx] = torch.logsumexp(log_integrand, dim=1)       # shape (batch_n,)

            single_var_marginals[col_idx] = marginal_col
            pbar.update(1)

    return single_var_marginals  # shape (D, N) - already in log space


def compute_two_var_marginal(model, data, col_idx_1, col_idx_2, 
                             #quad_x, quad_w, 
                             quad_x_2d, log_quad_w_2d,
                             batch_size=None):

    N, D = data.shape
    Q2 = quad_x_2d.shape[0]

    ## construct 2D grid once
    #points_i = quad_x.unsqueeze(1).expand(Q, Q).reshape(-1)           # shape (Q²,)
    #points_j = quad_x.unsqueeze(0).expand(Q, Q).reshape(-1)           # shape (Q²,)
    #points   = torch.stack([points_i, points_j], dim=1)                # shape (Q², 2)
#
    ## 2D log weights via outer sum in log space
    #log_w    = torch.log(quad_w)                                       # shape (Q,)
    #log_w_2d = (log_w.unsqueeze(1) + log_w.unsqueeze(0)).reshape(-1)  # shape (Q²,)
    
    points = quad_x_2d.to(model.device)  # shape (Q², 2)
    log_w_2d = log_quad_w_2d.to(model.device)

    if batch_size is None:
        batch_size_eff = N
    else:
        batch_size_eff = batch_size

    marginal = torch.zeros(N, device=model.device)

    for start_idx in range(0, N, batch_size_eff):
        end_idx = min(start_idx + batch_size_eff, N)
        batch_n = end_idx - start_idx

        # build chunk on the fly
        data_chunk = data[start_idx:end_idx]                                               # shape (batch_n, D)
        expanded_data_chunk = data_chunk.unsqueeze(1).expand(
            batch_n, Q2, D
        ).reshape(batch_n * Q2, D).clone()                                                 # shape (batch_n*Q², D)

        # expand points and log weights
        expanded_points  = points.unsqueeze(0).expand(
            batch_n, Q2, 2
        ).reshape(batch_n * Q2, 2)                                                         # shape (batch_n*Q², 2)
        expanded_log_w   = log_w_2d.unsqueeze(0).expand(
            batch_n, Q2
        ).reshape(batch_n * Q2)                                                            # shape (batch_n*Q²,)

        # replace columns
        expanded_data_chunk[:, col_idx_1] = expanded_points[:, 0]
        expanded_data_chunk[:, col_idx_2] = expanded_points[:, 1]

        # log likelihood
        ll_chunk = model.log_prob(
            expanded_data_chunk
        )                                                                                   # shape (batch_n*Q²,)

        # log-sum-exp over Q² points
        log_integrand = (ll_chunk + expanded_log_w).view(batch_n, Q2)                     # shape (batch_n, Q²)
        marginal[start_idx:end_idx] = torch.logsumexp(log_integrand, dim=1)               # shape (batch_n,)

    return marginal  # shape (N,) - already in log space


def evaluate_conditional_independence(device, num_points_quad, min_val, max_val, model, evaluation_data, evaluation_data_type):
    N, D = evaluation_data.shape

    # move data to device once
    evaluation_data = evaluation_data.to(device)

    pairs = [(i, j) for i in range(D) for j in range(i + 1, D)] # all pairs
    # -----------------------------------------------------------------
    # Step 1: Setup quadrature points and weights once
    # -----------------------------------------------------------------
    #with tqdm(total=D, desc="Setup", position=0) as setup_pbar:
    #setup_pbar.set_description("Setting up quadrature points")
    print("Setting up quadrature points")
    quad_x, quad_w, quad_x_2d, log_quad_w_2d = setup_quadrature(
        num_points=num_points_quad,
        min_val=min_val,
        max_val=max_val,
        device=device,
    )

    # quad_x: shape (Q,)
    # quad_w: shape (Q,)
    #setup_pbar.update(1)
    # -----------------------------------------------------------------
    # Step 2: Compute model log likelihood on data once
    # -----------------------------------------------------------------
    #setup_pbar.set_description("Computing model log likelihood")
    print("Computing model log likelihood")
    model_ll = model.log_prob(evaluation_data)  # shape (N,)
    #setup_pbar.update(1)
    # -----------------------------------------------------------------
    # Step 3: Compute all D single variable marginals once
    # -----------------------------------------------------------------
    #setup_pbar.set_description("Computing single variable marginals")
    print(("Computing single variable marginals"))
    single_var_marginals = compute_single_var_marginals(
        model=model,
        data=evaluation_data,
        quad_x=quad_x,
        quad_w=quad_w,
        #batch_size=batch_size,
    )
    # single_var_marginals: shape (D, N)
    #setup_pbar.update(1)
    #setup_pbar.set_description("Setup complete")
    #setup_pbar.update(1)
    # -----------------------------------------------------------------
    # Step 4: Loop over pairs, compute two var marginal and metrics
    # -----------------------------------------------------------------
    results = []
    results = [
        {
            "var_col": col_idx_1,
            "var_row": col_idx_2,
            **dict(zip(["kld", "iae"], compute_pair_metrics(
                model_ll=model_ll,
                single_var_marginals=single_var_marginals,
                two_var_marginal=compute_two_var_marginal(
                    model=model,
                    data=evaluation_data,
                    col_idx_1=col_idx_1,
                    col_idx_2=col_idx_2,
                    quad_x_2d=quad_x_2d,
                    log_quad_w_2d=log_quad_w_2d,
                    batch_size=None,
                ),
                col_idx_1=col_idx_1,
                col_idx_2=col_idx_2,
                evaluation_data_type=evaluation_data_type,
            )))
        }
        for col_idx_1, col_idx_2 in tqdm(pairs, desc="Computing pair metrics", position=0, leave=True)
    ]
    # -----------------------------------------------------------------
    # Step 5: Assemble results
    # -----------------------------------------------------------------
    results_df = pd.DataFrame(results)

    return results_df




from gtm.gtm_plots_analysis.compute_normalised_hessian_metric import pairwise_blockwise_nuclear_normalize_vectorised
from gtm.gtm_plots_analysis.compute_precision_matrix_summary_statistics import compute_precision_matrix_summary_statistics

def compute_local_loglikelihood_hessian(model,
                              y):
        
        samples = y.detach().clone().requires_grad_(True)  # (N, D)
        N, D = samples.shape

        score = model.log_prob(samples).sum()
        grad = torch.autograd.grad(score, samples, create_graph=True)[0]  # Shape: (1, D)

        hessian_vmap = [torch.autograd.grad(grad.sum(0)[d], samples, create_graph=True)[0] for d in range(D)]  # Shape: (1, D)

        hessian_vmap = torch.stack(hessian_vmap).permute(1, 0, 2)
        
        return hessian_vmap
    
    
def compute_conditional_independence_table_local_relative_hessian(model,y=None,
                                                                  evaluation_data_type="data",
                                                                  sample_size=1000,
                                                                  min_val=-torch.inf,
                                                                  max_val=+torch.inf):
    
    if evaluation_data_type == "data":
        if y==None:
            print("if evaluation_data_type=data then you need to pass data using arguement y.")
        else:
            evaluation_data=y
    elif evaluation_data_type == "samples_from_model":
        evaluation_data = model.sample(sample_size)[0].detach()
        # only data within the bound otherwise drop datapoints
        bool_mask = (evaluation_data >= min_val) & (evaluation_data <= max_val)
        if bool_mask.all(dim=1).sum() < sample_size:
            print(f"Warning: Only {bool_mask.all(dim=1).sum().item()} samples are within the specified bounds. Others are dropped.")
        evaluation_data = evaluation_data[bool_mask.all(dim=1)]
    
    hessians = compute_local_loglikelihood_hessian(model,
                              evaluation_data)
    
    normed_hessian = pairwise_blockwise_nuclear_normalize_vectorised(hessians, eps=1e-12)
    
    table = compute_precision_matrix_summary_statistics(normed_hessian.detach())
    
    table["normed_hessian_abs_mean"] = table["abs_mean"]
    
    table = table[[
            "var_row",
            "var_col",
            "normed_hessian_abs_mean",
        ]]
    
    return table





def run_experiment(
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
    nfm_name="maf", #snf
    device = "cpu",
    number_of_layers_list=[3,4,5], 
    hidden_layers_list=[2,3,4], 
    hidden_units_list=[10,20,30],
    weight_decay_list=[1e-5, 1e-6, 1e-7],
    optimizer="Adam",
    learning_rate=1e-3,
    iterations=2000,
    patience=30,
    min_delta=1e-7,
    temp_folder="./temp",
    # Evaluation of Conditional Independence parameters
    #evaluation_data_type = "samples_from_model",
    sample_size = 10000,
    max_num_ci_sample_size = 10000,
    num_points_quad=15,
    copula_only=False,
    min_val=-6,
    max_val=6,
    bootstrap_warpspeed=False,
    threshhold_kld_tails = np.linspace(0, 5, 500)
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
    
    
    synthetic_data_dict = generate_synthetic_vine_data(
        seed_value=seed_value,
        seed_value_copula=seed_value_copula,
        dimensionality=dimensionality,
        Independence_tree=Independence_tree,
        vine_type=vine_type,
        N_train=N_train,
        N_validate=N_validate,
        N_test=N_test
    )
    
    # Create dataset and DataLoader, if bootstrapped note that
    if bootstrap_warpspeed:
        #merge train and validate data for warpspeed bootstrap
        combined_data = torch.cat((synthetic_data_dict['train_data'], synthetic_data_dict['validate_data']), dim=0)
        
        # bootstrap sample with replacement (mit zurücklegen)
        indices = torch.randint(0, combined_data.size(0), (N_train + N_validate,))
        bootstrapped_data = combined_data[indices]
        
        # save bootstrapp indices for easy reproducibility
        np.save(temp_folder+"/bootstrap_indices.npy", np.array(indices.detach().cpu()))
        mlflow.log_artifact(temp_folder+"/bootstrap_indices.npy")
        
        # Split back into train and validate sets
        synthetic_data_dict['train_data'] = bootstrapped_data[:N_train]
        synthetic_data_dict['validate_data'] = bootstrapped_data[N_train:]
        
        # Create dataset and DataLoader
        dataset_train = Generic_Dataset(synthetic_data_dict['train_data'])
        dataloader_train = DataLoader(dataset_train, batch_size=N_train)

        dataset_validate = Generic_Dataset(synthetic_data_dict['validate_data'])
        dataloader_validate = DataLoader(dataset_validate, batch_size=N_validate)
    else:
        dataset_train = Generic_Dataset(synthetic_data_dict['train_data'])
        dataloader_train = DataLoader(dataset_train, batch_size=N_train)

        dataset_validate = Generic_Dataset(synthetic_data_dict['validate_data'])
        dataloader_validate = DataLoader(dataset_validate, batch_size=N_validate)
        
    # Run NFM Model Training
    # first store all training parameters
    mlflow.log_param(key="nfm_name", value=nfm_name)
    mlflow.log_param(key="device", value=device)
    mlflow.log_param(key="number_of_layers_list" , value=number_of_layers_list)
    mlflow.log_param(key="hidden_layers_list", value=hidden_layers_list)
    mlflow.log_param(key="hidden_units_list", value=hidden_units_list)
    mlflow.log_param(key="weight_decay_list", value=weight_decay_list)
    mlflow.log_param(key="optimizer", value=optimizer)
    mlflow.log_param(key="learning_rate", value=learning_rate)
    mlflow.log_param(key="iterations", value=iterations)
    mlflow.log_param(key="patience", value=patience)
    mlflow.log_param(key="min_delta", value=min_delta)
    mlflow.log_param(key="temp_folder", value=temp_folder)  
    

    results = hyperparameter_search(simulated_data_train=synthetic_data_dict['train_data'], 
                                    simulated_data_validate=synthetic_data_dict['validate_data'], 
                                    latent_size=dimensionality, 
                                    early_stopping_patience=patience, 
                                    max_iter=iterations, 
                                    lr=learning_rate, 
                                    weight_decay_list=weight_decay_list,
                                    number_of_layers_list=number_of_layers_list, 
                                    hidden_layers_list=hidden_layers_list, 
                                    hidden_units_list=hidden_units_list, 
                                    min_delta=min_delta,
                                    nf_model=nfm_name)
    
    model = pd.DataFrame(results).sort_values("loss_val")["model"][0]
    model.device = device
    model = model.to(device)
    number_of_layers_opt = pd.DataFrame(results).sort_values("loss_val")["number_of_layers"][0]
    hidden_layer_opt = pd.DataFrame(results).sort_values("loss_val")["hidden_layer"][0]
    hidden_unit_opt = pd.DataFrame(results).sort_values("loss_val")["hidden_unit"][0]
    weight_decay_opt = pd.DataFrame(results).sort_values("loss_val")["weight_decay"][0]
    loss_list_training = pd.DataFrame(results).sort_values("loss_val")["loss_hist"][0]
    loss_list_validation = pd.DataFrame(results).sort_values("loss_val")["loss_hist_val"][0]
    
    # plot training curves
    plt.plot(loss_list_training, label="Training Loss")
    if loss_list_validation is not None:
        plt.plot(loss_list_validation, label="Validation Loss")
    plt.xlabel("Iteration")
    plt.ylabel("Negative Log Likelihood")
    plt.title("Training Curve")
    plt.legend()
    fig_train = plt.gcf()
    plt.close()
    # log the plot as an mlflow artifact
    log_mlflow_plot(fig_train, 'training_curves.png', temporary_storage_directory=temp_folder)
    
    # Log trained model
    _ = mlflow.pytorch.log_model(model, "model")
    # can be loaded (to cpu) via: model = mlflow.pytorch.load_model("runs:/{}/model".format(run_id), map_location=torch.device('cpu'))
    
    # store all trained parameters
    mlflow.log_param(key="number_of_layers_opt", value=number_of_layers_opt)
    mlflow.log_param(key="hidden_layer_opt", value=hidden_layer_opt)
    mlflow.log_param(key="hidden_unit_opt", value=hidden_unit_opt)
    mlflow.log_param(key="weight_decay_opt", value=weight_decay_opt)
    ### 4. Compare to Benchmarks

    # We compare the learned GTM to a Gaussian Approximation and the Oracle Model. We expect the GTM to lie between these two in terms of approximation the true underlying distribution.
    # We measure this by means of the Kullback Leibler Divergence which we approximate on the test set which is equivalent to the log likelihood ratio between the true distribution and an approximation of it..
    log_likelihood_train = model.log_prob(synthetic_data_dict['train_data']).detach().cpu()
    log_likelihood_validate = model.log_prob(synthetic_data_dict['validate_data']).detach().cpu()
    log_likelihood_test_gtm = model.log_prob(synthetic_data_dict['test_data']).detach().cpu()

    # estimate the Multivariate Normal Distribution as Model
    mean_mvn_model = synthetic_data_dict['train_data'].mean(0)
    cov_mvn_model = synthetic_data_dict['train_data'].T.cov()
    mvn_model = torch.distributions.MultivariateNormal(loc=mean_mvn_model, covariance_matrix=cov_mvn_model)
    log_likelihood_train_gaussian = mvn_model.log_prob(synthetic_data_dict['train_data'])
    log_likelihood_validate_gaussian = mvn_model.log_prob(synthetic_data_dict['validate_data'])
    log_likelihood_test_gaussian = mvn_model.log_prob(synthetic_data_dict['test_data'])

    kld_gtm_train = np.round(torch.mean(synthetic_data_dict["loglik_true_train"] - log_likelihood_train).item(),4)
    kld_gaussian_train = np.round(torch.mean(synthetic_data_dict["loglik_true_train"] - log_likelihood_train_gaussian).item(),4)
    kld_vine_oracle_train = np.round(torch.mean(synthetic_data_dict["loglik_true_train"] - synthetic_data_dict["loglik_true_est_train"]).item(),4)
    rel_kld_gtm_train = np.round((kld_gtm_train - kld_vine_oracle_train) / (kld_gaussian_train - kld_vine_oracle_train),4)
   
    kld_gtm_test = np.round(torch.mean(synthetic_data_dict["loglik_true_test"] - log_likelihood_test_gtm).item(),4)
    kld_gaussian_test = np.round(torch.mean(synthetic_data_dict["loglik_true_test"] - log_likelihood_test_gaussian).item(),4)
    kld_vine_oracle_test = np.round(torch.mean(synthetic_data_dict["loglik_true_test"] - synthetic_data_dict["loglik_true_est_test"]).item(),4)
    rel_kld_gtm_test = np.round((kld_gtm_test - kld_vine_oracle_test) / (kld_gaussian_test - kld_vine_oracle_test),4)
    
    # store log likelihood metrics
    mlflow.log_metric(key="kld_gtm_train", value=kld_gtm_train)
    mlflow.log_metric(key="kld_gaussian_train", value=kld_gaussian_train)
    mlflow.log_metric(key="kld_vine_oracle_train", value=kld_vine_oracle_train)
    mlflow.log_metric(key="relative_kld_gtm_train", value=rel_kld_gtm_train)

    mlflow.log_metric(key="kld_gtm_test", value=kld_gtm_test)
    mlflow.log_metric(key="kld_gaussian_test", value=kld_gaussian_test)
    mlflow.log_metric(key="kld_vine_oracle_test", value=kld_vine_oracle_test)
    mlflow.log_metric(key="relative_kld_gtm_test", value=rel_kld_gtm_test)
    
     # store test set log likelihood e.g. kld for the tails    
    df_gtm_tail_kld = kld_tails_assessment(synthetic_data_dict['test_data'],
                             synthetic_data_dict["loglik_true_test"],
                             log_likelihood_test_gtm,
                             thresholds = threshhold_kld_tails)

    df_ggm_tail_kld = kld_tails_assessment(synthetic_data_dict['test_data'],
                                synthetic_data_dict["loglik_true_test"],
                                log_likelihood_test_gaussian,
                                thresholds = threshhold_kld_tails)
    
    df_gtm_tail_kld.to_csv(temp_folder+"/nf_tail_kld_table.csv", index=False)
    mlflow.log_artifact(temp_folder+"/nf_tail_kld_table.csv")  
    
    df_ggm_tail_kld.to_csv(temp_folder+"/ggm_tail_kld_table.csv", index=False)
    mlflow.log_artifact(temp_folder+"/ggm_tail_kld_table.csv")  

    ### 5. Evaluate and Plot GTM Results
    # We evaluate the model further by showing how to generate synthetic samples, plot the conditional correlation patterns as well as identify conditional independencies.
    # To Do so we start by evaluate the conditional indepenedence_relationships. By computing the table we compute pseudo correlation matrix based conditional independence metrics 
    # and likelihood based conditional independence metrics such as the iae and the kld. For more details on these see the paper. This computation may take some time as it computes quadratures under the hood, see the algorithm 1 in the paper.
    
    #mlflow.log_param(key="evaluation_data_type", value=evaluation_data_type)
    mlflow.log_param(key="sample_size", value=sample_size)
    mlflow.log_param(key="max_num_ci_sample_size", value=max_num_ci_sample_size)
    mlflow.log_param(key="num_points_quad", value=num_points_quad)
    mlflow.log_param(key="copula_only", value=copula_only)
    mlflow.log_param(key="min_val", value=min_val)
    mlflow.log_param(key="max_val", value=max_val)
    
    
    if max_num_ci_sample_size < sample_size:
        # Compute number of chunks
        n_chunks = math.ceil(sample_size / max_num_ci_sample_size)
        print("needs to chunk synthetic sample ci computation, uses {} chunks of size ".format(n_chunks) + str(max_num_ci_sample_size))
        print("is be more then sample_size if sample_size / max_num_ci_sample_size not an integer")

        result_tables = []
        for i in range(n_chunks):
            print(f"Processing chunk {i+1}/{n_chunks} ({max_num_ci_sample_size} samples)...")
            chunk_table = evaluate_conditional_independence(
                device=device,
                num_points_quad=num_points_quad,
                min_val=min_val,
                max_val=max_val,
                model=model,
                evaluation_data=model.sample(min(max_num_ci_sample_size, sample_size - i*max_num_ci_sample_size))[0],
                evaluation_data_type="samples_from_model"
            )
            result_tables.append(chunk_table)

        # Concatenate all results into one dataframe
        all_results = pd.concat(result_tables, ignore_index=True)

        # Aggregate (average) using groupby:
        conditional_independence_table_samples = (
            all_results
            .groupby(["var_row", "var_col"], as_index=False)
            .agg({
                "precision_abs_mean": "mean",
                "precision_square_mean": "mean",
                "cond_correlation_abs_mean": "mean",
                "cond_correlation_square_mean": "mean",
                "kld": "mean",
                "iae": "mean"
            })
        )
    else:  
        conditional_independence_table_samples = evaluate_conditional_independence(
            device=device,
            num_points_quad=num_points_quad,
            min_val=min_val,
            max_val=max_val,
            model=model,
            evaluation_data=model.sample(sample_size)[0],
            evaluation_data_type="samples_from_model"
        )
    

    conditional_independence_table_train = evaluate_conditional_independence(
        device=device,
        num_points_quad=num_points_quad,
        min_val=min_val,
        max_val=max_val,
        model=model,
        evaluation_data=synthetic_data_dict['train_data'].detach(),
        evaluation_data_type="data"
    )
    

    conditional_independence_table_val = evaluate_conditional_independence(
        device=device,
        num_points_quad=num_points_quad,
        min_val=min_val,
        max_val=max_val,
        model=model,
        evaluation_data=synthetic_data_dict['validate_data'].detach(),
        evaluation_data_type="data"
    )
    
    
    # creating the joint data train and validation evaluation
    portion_val = N_validate / (N_train+N_validate)
    portion_train = N_train / (N_train+N_validate)
    
    conditional_independence_table_data = conditional_independence_table_train
    conditional_independence_table_data["kld"]                       = portion_train * conditional_independence_table_train["kld"] + portion_val * conditional_independence_table_val["kld"]

    
    # Relative Hessian Metric
    ci_table_hessian_samples = compute_conditional_independence_table_local_relative_hessian(model,
                                                                  y=None,
                                                                  evaluation_data_type="samples_from_model",
                                                                  sample_size=1000,
                                                                  min_val=min_val,
                                                                  max_val=max_val)
    
    
    ci_table_relative_hessian_train = compute_conditional_independence_table_local_relative_hessian(model,
                                                                  y=synthetic_data_dict['train_data'].detach(),
                                                                  evaluation_data_type="data",
                                                                  sample_size=1000,
                                                                  min_val=min_val,
                                                                  max_val=max_val)
    
    ci_table_relative_hessian_val = compute_conditional_independence_table_local_relative_hessian(model,
                                                                  y=synthetic_data_dict['validate_data'].detach(),
                                                                  evaluation_data_type="data",
                                                                  sample_size=1000,
                                                                  min_val=min_val,
                                                                  max_val=max_val)
    
    ci_table_relative_hessian_data = ci_table_relative_hessian_train
    ci_table_relative_hessian_data["auc_normed_hessian"] = portion_train * ci_table_relative_hessian_train["normed_hessian_abs_mean"] + portion_val * ci_table_relative_hessian_val["normed_hessian_abs_mean"]
    
    
    
    ### 6. Identifying the Conditional Independence Graph 
    # We compare the true known conditional independence Graph to the one learned by the GTM. To do so we first merge the true structure table with our learned one.
    
    #################### Synth
    merged_ci_tables_samples = pd.merge(
        conditional_independence_table_samples,
        synthetic_data_dict["df_true_structure"],
        on=["var_row", "var_col"]
    )
    merged_ci_tables_samples = pd.merge(
        merged_ci_tables_samples,
        ci_table_hessian_samples,
        on=["var_row", "var_col"]
    )
    ####################

    #################### Joint Data
    merged_ci_tables_data = pd.merge(
        conditional_independence_table_data,
        synthetic_data_dict["df_true_structure"],
        on=["var_row", "var_col"]
    )
    merged_ci_tables_data = pd.merge(
        merged_ci_tables_data,
        ci_table_relative_hessian_data,
        on=["var_row", "var_col"]
    )
    # the iae makes no sense when using the true data, and the kld is the log likelihood ratio so we del iae and rename kld into ll_diff
    del merged_ci_tables_data["iae"]
    merged_ci_tables_data["ll_diff"] = merged_ci_tables_data["kld"]
    del merged_ci_tables_data["kld"]
    ####################
    
    #################### Train Data
    merged_ci_tables_train = pd.merge(
        conditional_independence_table_train,
        synthetic_data_dict["df_true_structure"],
        on=["var_row", "var_col"]
    )
    merged_ci_tables_train = pd.merge(
        merged_ci_tables_train,
        ci_table_relative_hessian_train,
        on=["var_row", "var_col"]
    )
    # the iae makes no sense when using the true data, and the kld is the log likelihood ratio so we del iae and rename kld into ll_diff
    del merged_ci_tables_train["iae"]
    merged_ci_tables_train["ll_diff"] = merged_ci_tables_train["kld"]
    del merged_ci_tables_train["kld"]
    ####################
    
    #################### Validate Data
    merged_ci_tables_val = pd.merge(
        conditional_independence_table_val,
        synthetic_data_dict["df_true_structure"],
        on=["var_row", "var_col"]
    )
    merged_ci_tables_val = pd.merge(
        merged_ci_tables_val,
        ci_table_relative_hessian_val,
        on=["var_row", "var_col"]
    )
    # the iae makes no sense when using the true data, and the kld is the log likelihood ratio so we del iae and rename kld into ll_diff
    del merged_ci_tables_val["iae"]
    merged_ci_tables_val["ll_diff"] = merged_ci_tables_val["kld"]
    del merged_ci_tables_val["kld"]
    ####################
    
    
    # store the merged table as an artifact
    merged_ci_tables_samples.to_csv(temp_folder+"/conditional_independence_table_model_samples.csv", index=False)
    mlflow.log_artifact(temp_folder+"/conditional_independence_table_model_samples.csv")   
    
    conditional_independence_table_data.to_csv(temp_folder+"/conditional_independence_table_data.csv", index=False)
    mlflow.log_artifact(temp_folder+"/conditional_independence_table_data.csv")   
    
    conditional_independence_table_train.to_csv(temp_folder+"/conditional_independence_table_train.csv", index=False)
    mlflow.log_artifact(temp_folder+"/conditional_independence_table_train.csv")   
    
    conditional_independence_table_val.to_csv(temp_folder+"/conditional_independence_table_val.csv", index=False)
    mlflow.log_artifact(temp_folder+"/conditional_independence_table_val.csv")   

    # Store metrics based on synthetic samples
    auc_iae = roc_auc_score(merged_ci_tables_samples["dependence"], merged_ci_tables_samples["iae"])
    auc_kld = roc_auc_score(merged_ci_tables_samples["dependence"], merged_ci_tables_samples["kld"])
    auc_nhess = roc_auc_score(merged_ci_tables_samples["dependence"], merged_ci_tables_samples["normed_hessian_abs_mean"])
    
    mlflow.log_metric(key="auc_iae", value=auc_iae)
    mlflow.log_metric(key="auc_kld", value=auc_kld)
    mlflow.log_metric(key="auc_normed_hessian", value=auc_nhess)

    # Store metrics based on true data joint
    auc_ll_diff_data = roc_auc_score(merged_ci_tables_data["dependence"], merged_ci_tables_data["ll_diff"])
    auc_nhess_data = roc_auc_score(merged_ci_tables_data["dependence"], merged_ci_tables_data["normed_hessian_abs_mean"])

    mlflow.log_metric(key="auc_loglik_diff_data", value=auc_ll_diff_data)
    mlflow.log_metric(key="auc_normed_hessian_data", value=auc_nhess_data)
    
    
    # Store metrics based on training data
    auc_ll_diff_train = roc_auc_score(merged_ci_tables_train["dependence"], merged_ci_tables_train["ll_diff"])
    auc_nhess_train = roc_auc_score(merged_ci_tables_train["dependence"], merged_ci_tables_train["normed_hessian_abs_mean"])

    mlflow.log_metric(key="auc_loglik_diff_train", value=auc_ll_diff_train)
    mlflow.log_metric(key="auc_normed_hessian_train", value=auc_nhess_train)
    
    
    # Store metrics based on true data joint
    auc_ll_diff_val = roc_auc_score(merged_ci_tables_val["dependence"], merged_ci_tables_val["ll_diff"])
    auc_nhess_val = roc_auc_score(merged_ci_tables_val["dependence"], merged_ci_tables_val["normed_hessian_abs_mean"])

    mlflow.log_metric(key="auc_loglik_diff_val", value=auc_ll_diff_val)
    mlflow.log_metric(key="auc_normed_hessian_val", value=auc_nhess_val)
    
    mlflow.end_run()
    
    clear_temp_folder(temp_folder)
    

    
    
if __name__ == "__main__":
    
    run_experiment(
        run_name="test_run",
        experiment_id=0,
        seed_value=1,
        seed_value_copula=1,
        dimensionality=10,
        Independence_tree=3,
        vine_type="R-Vine",
        N_train=500,
        N_validate=125,
        N_test=20000,
        nfm_name="snf", #snf
        device = "cpu",
        number_of_layers_list=[3], 
        hidden_layers_list=[2], 
        hidden_units_list=[10], 
        weight_decay_list=[1e-6],
        optimizer="Adam",
        learning_rate=1e-3,
        iterations=2000,
        patience=10,
        min_delta=1e-7,
        temp_folder="./temp",
        sample_size = 100,
        max_num_ci_sample_size = 100,
        num_points_quad=5,
        copula_only=False,
        min_val=-6,
        max_val=6,
    )


    run_experiment(
        run_name="test_run",
        experiment_id=0,
        seed_value=1,
        seed_value_copula=1,
        dimensionality=10,
        Independence_tree=3,
        vine_type="R-Vine",
        N_train=500,
        N_validate=125,
        N_test=20000,
        nfm_name="maf", #snf
        device = "cpu",
        number_of_layers_list=[3], 
        hidden_layers_list=[2], 
        hidden_units_list=[10], 
        weight_decay_list=[1e-6],
        optimizer="Adam",
        learning_rate=1e-3,
        iterations=2000,
        patience=10,
        min_delta=1e-7,
        temp_folder="./temp",
        sample_size = 100,
        max_num_ci_sample_size = 100,
        num_points_quad=5,
        copula_only=False,
        min_val=-6,
        max_val=6,
    )
