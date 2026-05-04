import multiprocessing
import pickle
import time
import warnings

import pandas as pd
import torch

#from gtm.gtm_plots_analysis.compute_precision_matrix_summary_statistics import *
#from gtm.gtm_plots_analysis.independence_kld_process_row import *


import os

import numpy as np
# from run_simulation_study import run_simulation_study, log_mlflow_plot
import pandas as pd
import torch

from tqdm import tqdm

from gtm.gtm_plots_analysis.nd_quad import (  # https://github.com/harisankar95/nd-quadrature
    getQuad, integrateQuad, transformQuad)

from gtm.gtm_plots_analysis.compute_precision_matrix_summary_statistics import *



def setup_quadrature(num_points, min_val, max_val, device):
    """
    Compute 1D quadrature points and weights once for reuse across all dimensions.
    
    Args:
        num_points: number of quadrature points
        min_val: lower integration limit (same for all dimensions)
        max_val: upper integration limit (same for all dimensions)
        device: torch device
    
    Returns:
        x: 1D quadrature points, shape (Q,)
        w: 1D quadrature weights, shape (Q,)
    """
    x, w = getQuad(num_points, ndim=1)
    x, w = transformQuad(x, w, [[min_val, max_val]])
    
    x = torch.FloatTensor(x).squeeze().to(device)  # shape (Q,)
    w = torch.FloatTensor(w).squeeze().to(device)  # shape (Q,)
    
    return x, w


def build_expanded_data(data, col_indices, quad_x, quad_w):
    """
    Build expanded dataset for numerical integration.
    For 2 variables, 2D grid is constructed via outer product of 1D points.
    
    Args:
        data:        shape (N, D)
        col_indices: list of column indices to integrate out
                     [col] for single var -> Q points
                     [col_i, col_j] for two var -> Q² points
        quad_x:      1D quadrature points, shape (Q,)
        quad_w:      1D quadrature weights, shape (Q,)
    
    Returns:
        expanded_data:    shape (N * Q, D) or (N * Q², D)
        expanded_weights: shape (N * Q,)   or (N * Q²,)
    """
    N, D = data.shape
    Q = quad_x.shape[0]

    if len(col_indices) == 1:
        # Single variable case
        # points and weights are just the 1D quad points
        points = quad_x                    # shape (Q,)
        weights = quad_w                   # shape (Q,)

    elif len(col_indices) == 2:
        # Two variable case
        # construct 2D grid via outer product
        # points_i, points_j: shape (Q, Q) -> (Q², )
        points_i = quad_x.unsqueeze(1).expand(Q, Q).reshape(-1)  # shape (Q²,)
        points_j = quad_x.unsqueeze(0).expand(Q, Q).reshape(-1)  # shape (Q²,)
        points = torch.stack([points_i, points_j], dim=1)         # shape (Q², 2)
        
        # 2D weights are outer product of 1D weights
        weights = (quad_w.unsqueeze(1) * quad_w.unsqueeze(0)).reshape(-1)  # shape (Q²,)

    num_points = points.shape[0]  # Q or Q²

    # Expand data: (N, D) -> (N*Q, D) or (N*Q², D)
    expanded_data = data.unsqueeze(1).expand(N, num_points, D).reshape(N * num_points, D).clone()

    # Expand weights: (Q,) or (Q²,) -> (N*Q,) or (N*Q²,)
    expanded_weights = weights.unsqueeze(0).expand(N, num_points).reshape(N * num_points)

    # Replace columns with quadrature points
    if len(col_indices) == 1:
        expanded_data[:, col_indices[0]] = points.unsqueeze(0).expand(N, Q).reshape(N * Q)
    elif len(col_indices) == 2:
        expanded_data[:, col_indices[0]] = points[:, 0].unsqueeze(0).expand(N, num_points).reshape(N * num_points)
        expanded_data[:, col_indices[1]] = points[:, 1].unsqueeze(0).expand(N, num_points).reshape(N * num_points)

    return expanded_data, expanded_weights


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
                ll_chunk = model.log_likelihood(
                    expanded_data_chunk, return_lambda_matrix=False
                )                                                                              # shape (batch_n*Q,)

                # log-sum-exp over Q points
                # ll_chunk + log_w: shape (batch_n*Q,) -> (batch_n, Q)
                log_integrand = (ll_chunk + expanded_log_w).view(batch_n, Q)                  # shape (batch_n, Q)
                marginal_col[start_idx:end_idx] = torch.logsumexp(log_integrand, dim=1)       # shape (batch_n,)

            single_var_marginals[col_idx] = marginal_col
            pbar.update(1)

    return single_var_marginals  # shape (D, N) - already in log space


def compute_two_var_marginal(model, data, col_idx_1, col_idx_2, quad_x, quad_w, batch_size=None):

    N, D = data.shape
    Q = quad_x.shape[0]
    Q2 = Q * Q

    # construct 2D grid once
    points_i = quad_x.unsqueeze(1).expand(Q, Q).reshape(-1)           # shape (Q²,)
    points_j = quad_x.unsqueeze(0).expand(Q, Q).reshape(-1)           # shape (Q²,)
    points   = torch.stack([points_i, points_j], dim=1)                # shape (Q², 2)

    # 2D log weights via outer sum in log space
    log_w    = torch.log(quad_w)                                       # shape (Q,)
    log_w_2d = (log_w.unsqueeze(1) + log_w.unsqueeze(0)).reshape(-1)  # shape (Q²,)

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
        ll_chunk = model.log_likelihood(
            expanded_data_chunk, return_lambda_matrix=False
        )                                                                                   # shape (batch_n*Q²,)

        # log-sum-exp over Q² points
        log_integrand = (ll_chunk + expanded_log_w).view(batch_n, Q2)                     # shape (batch_n, Q²)
        marginal[start_idx:end_idx] = torch.logsumexp(log_integrand, dim=1)               # shape (batch_n,)

    return marginal  # shape (N,) - already in log space

def compute_pair_metrics(
    model_ll,
    single_var_marginals,
    two_var_marginal,
    col_idx_1,
    col_idx_2,
    evaluation_data_type="data",
):
    """
    Compute CI metrics for a single pair (col_idx_1, col_idx_2).
    Pure tensor operations, no model calls.
    Called immediately after compute_two_var_marginal, result is scalar metrics.
    
    Args:
        model_ll:             log likelihood of data under model, shape (N,)
        single_var_marginals: log marginals for all columns,      shape (D, N)
        two_var_marginal:     log marginal for this pair,         shape (N,)
        col_idx_1:            first column index
        col_idx_2:            second column index
        evaluation_data_type: "data", "samples_from_model", or "uniform_random_samples"
    
    Returns:
        kld: scalar KL divergence metric for this pair
        iae: scalar integrated absolute error metric for this pair
    """

    # p1 = log f(Y) - full model log likelihood
    p1 = model_ll                                # shape (N,)

    # p2 = log f(Y_{/ij}) - marginal with both cols integrated out
    p2 = two_var_marginal                        # shape (N,)

    # p4 = log f(Y_i | Y_{/ij}) 
    #    = log f(Y_i, Y_{/ij}) - log f(Y_{/ij})
    #    = log marginal(col_j integrated out) - p2
    p4 = single_var_marginals[col_idx_2] - p2   # shape (N,)

    # p5 = log f(Y_j | Y_{/ij})
    #    = log f(Y_j, Y_{/ij}) - log f(Y_{/ij})
    #    = log marginal(col_i integrated out) - p2
    p5 = single_var_marginals[col_idx_1] - p2   # shape (N,)

    # p3 = log f(Y_i, Y_j | Y_{/ij})
    #    = log f(Y) - log f(Y_{/ij})
    p3 = p1 - p2                                 # shape (N,)

    # actual log conditional joint distribution
    actual_log          = p3                     # shape (N,)

    # log conditional joint under CI assumption
    # log f(Y_i|Y_{/ij}) + log f(Y_j|Y_{/ij})
    under_ci_log        = p4 + p5               # shape (N,)

    # full data log likelihood under CI assumption
    # log f(Y_i|Y_{/ij}) + log f(Y_j|Y_{/ij}) + log f(Y_{/ij})
    #under_ci_log_full   = p4 + p5 + p2          # shape (N,)

    if evaluation_data_type in ["data", "samples_from_model"]:

        # KLD metric
        # E[ log f(Y_i,Y_j|Y_{/ij}) - log f(Y_i|Y_{/ij}) f(Y_j|Y_{/ij}) ]
        ll_dev = actual_log - under_ci_log       # shape (N,)
        # handle numerical instability
        ll_dev = ll_dev.nan_to_num(nan=-torch.inf, posinf=-torch.inf, neginf=-torch.inf)
        ll_dev = ll_dev[ll_dev > -torch.inf]
        ll_dev = ll_dev[ll_dev.abs() < ll_dev.abs().quantile(0.99)]
        kld    = ll_dev.mean()                   # scalar

        # IAE metric
        # E[ |f(Y_i,Y_j|Y_{/ij}) - f(Y_i|Y_{/ij}) f(Y_j|Y_{/ij})| * f(Y_{/ij}) / f(Y) ]
        # weights = f(Y_{/ij}) / f(Y) in log space
        weights = torch.exp(p2 - model_ll) #under_ci_log_full - model_ll)  # shape (N,)
        ll_dev2 = torch.abs(
            torch.exp(actual_log) - torch.exp(under_ci_log)
        )                                                   # shape (N,)
        ll_dev2 = ll_dev2.nan_to_num(nan=-torch.inf, posinf=-torch.inf, neginf=-torch.inf)
        ll_dev2 = ll_dev2[ll_dev2 > -torch.inf]
        weights = weights[ll_dev2 > -torch.inf]
        iae     = ll_dev2 * weights                         # shape (N,)
        iae     = iae[iae < iae.quantile(0.99)]
        iae     = iae.mean() / 2                            # scalar

    elif evaluation_data_type == "uniform_random_samples":

        # KLD metric weighted by actual density
        ll_dev = torch.exp(actual_log) * (actual_log - under_ci_log)  # shape (N,)
        ll_dev = ll_dev.nan_to_num(nan=-torch.inf, posinf=-torch.inf, neginf=-torch.inf)
        ll_dev = ll_dev[ll_dev > -torch.inf]
        ll_dev = ll_dev[ll_dev.abs() < ll_dev.abs().quantile(0.99)]
        kld    = ll_dev.mean()                                         # scalar

        # IAE metric
        ll_dev2 = torch.abs(
            torch.exp(actual_log) - torch.exp(under_ci_log)
        )                                                              # shape (N,)
        ll_dev2 = ll_dev2.nan_to_num(nan=-torch.inf, posinf=-torch.inf, neginf=-torch.inf)
        ll_dev2 = ll_dev2[ll_dev2 > -torch.inf]
        iae     = ll_dev2[ll_dev2 < ll_dev2.quantile(0.99)]
        iae     = iae.mean() / 2                                       # scalar

    return kld.item(), iae.item()


def compute_conditional_independence_kld_v2(
    model,
    data,
    num_points_quad=20,
    min_val=-5,
    max_val=5,
    batch_size=None,
    evaluation_data_type="data",
    sample_size=10000,
    copula_only=False
):
    """
    Orchestrates the full conditional independence KLD computation.
    
    Args:
        model:                model with log_likelihood method
        data:                 shape (N, D)
        pairs:                list of (col_idx_1, col_idx_2) tuples to evaluate
                              e.g. [(0,1), (0,2), (1,2)] for D=3
        num_points_quad:      number of quadrature points per dimension
        min_val:              lower integration limit
        max_val:              upper integration limit
        batch_size:           number of rows N to process at once (None = all at once)
        evaluation_data_type: "data", "samples_from_model", or "uniform_random_samples"
    
    Returns:
        results_df: dataframe with columns:
                    [var_col_1, var_col_2, kld, iae]
    """
    with torch.no_grad():
        # in case of gpu cuda compute
        if evaluation_data_type == "data":
            evaluation_data = data[:sample_size]  # Adjust this based on your needs
            if copula_only == True:
                evaluation_data = model.after_transformation(evaluation_data)
        elif evaluation_data_type == "uniform_random_samples":
            evaluation_data = torch.distributions.Uniform(min_val, max_val).sample(
                [sample_size, model.y_train.size(1)]
            )   
        elif evaluation_data_type == "samples_from_model":
            evaluation_data = model.sample(sample_size).detach()
            if copula_only == True:
                evaluation_data = model.after_transformation(evaluation_data).detach()
            # only data within the bound otherwise drop datapoints
            bool_mask = (evaluation_data >= min_val) & (evaluation_data <= max_val)
            if bool_mask.all(dim=1).sum() < sample_size:
                print(f"Warning: Only {bool_mask.all(dim=1).sum().item()} samples are within the specified bounds. Others are dropped.")
            evaluation_data = evaluation_data[bool_mask.all(dim=1)]

        if copula_only == True:
            model.num_trans_layers = 0
            
            
        precision_matrix = (
            model.compute_pseudo_precision_matrix(evaluation_data).detach().cpu()
        )
        correlation_matrix = (
            model.compute_pseudo_conditional_correlation_matrix(evaluation_data)
            .detach()
            .cpu()
        )

        precision_matrix_summary_statistics = compute_precision_matrix_summary_statistics(
            precision_matrix
        )
        
        correlation_matrix_summary_statistics = compute_precision_matrix_summary_statistics(
            correlation_matrix
        )
        

        N, D = evaluation_data.shape
        device = model.device

        # move data to device once
        evaluation_data = evaluation_data.to(device)
        
        pairs = [(i, j) for i in range(D) for j in range(i + 1, D)] # all pairs

        # -----------------------------------------------------------------
        # Step 1: Setup quadrature points and weights once
        # -----------------------------------------------------------------
        #with tqdm(total=D, desc="Setup", position=0) as setup_pbar:

        #setup_pbar.set_description("Setting up quadrature points")
        print("Setting up quadrature points")
        quad_x, quad_w = setup_quadrature(
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
        model_ll = model.log_likelihood(evaluation_data, return_lambda_matrix=False)  # shape (N,)
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
            batch_size=batch_size,
        )
        # single_var_marginals: shape (D, N)
        #setup_pbar.update(1)
        #setup_pbar.set_description("Setup complete")
        #setup_pbar.update(1)

        # -----------------------------------------------------------------
        # Step 4: Loop over pairs, compute two var marginal and metrics
        # -----------------------------------------------------------------
        results = []

        with tqdm(
            total=len(pairs),
            desc="Computing pair metrics",
            position=0,
            leave=True,
        ) as pair_pbar:

            for pair_num, (col_idx_1, col_idx_2) in enumerate(pairs):

                pair_pbar.set_description(
                    f"Pair {pair_num+1}/{len(pairs)}: ({col_idx_1}, {col_idx_2})"
                )

                # compute two var marginal for this pair
                # built and consumed chunk by chunk
                two_var_marginal = compute_two_var_marginal(
                    model=model,
                    data=evaluation_data,
                    col_idx_1=col_idx_1,
                    col_idx_2=col_idx_2,
                    quad_x=quad_x,
                    quad_w=quad_w,
                    batch_size=batch_size,
                )
                # shape (N,)

                # immediately compute metrics and discard two_var_marginal
                kld, iae = compute_pair_metrics(
                    model_ll=model_ll,
                    single_var_marginals=single_var_marginals,
                    two_var_marginal=two_var_marginal,
                    col_idx_1=col_idx_1,
                    col_idx_2=col_idx_2,
                    evaluation_data_type=evaluation_data_type,
                )

                # store only scalars
                results.append({
                    "var_col": col_idx_1,
                    "var_row": col_idx_2,
                    "kld":       kld,
                    "iae":       iae,
                })

                # two_var_marginal goes out of scope here and is freed
                del two_var_marginal

                # update progress bar with current metrics
                pair_pbar.set_postfix({
                    "kld": f"{kld:.4f}",
                    "iae": f"{iae:.4f}",
                })
                pair_pbar.update(1)

        # -----------------------------------------------------------------
        # Step 5: Assemble results
        # -----------------------------------------------------------------
        results_df = pd.DataFrame(results)
        
        precision_matrix_summary_statistics = pd.merge(precision_matrix_summary_statistics,results_df, on=["var_col", "var_row"], how="inner")
        precision_matrix_summary_statistics["precision_abs_mean"] = np.array(precision_matrix_summary_statistics["abs_mean"])  
        precision_matrix_summary_statistics["cond_correlation_abs_mean"] = np.array(correlation_matrix_summary_statistics["abs_mean"])   
        sub_kld_summary_statistics = precision_matrix_summary_statistics[
            [
                "var_row",
                "var_col",
                "precision_abs_mean",
                #"precision_square_mean",
                "cond_correlation_abs_mean",
                #"cond_correlation_square_mean",
                "kld",
                "iae",
            ]
        ]
    return sub_kld_summary_statistics