import time
import warnings
import torch

import pandas as pd
import torch
from typing import TYPE_CHECKING, Optional
from torch.nn.utils.stateless import _reparametrize_module

from gtm.gtm_plots_analysis.compute_precision_matrix_summary_statistics import *
from gtm.gtm_plots_analysis.independence_kld_process_row import *

if TYPE_CHECKING:

    from gtm_model.gtm import GTM
    from gtm_training.training_bayes.variational_model_estimator import TauPack, VI_Model

def compute_conditional_independence_kld(
    self,
    y=None,
    x=False,
    evaluation_data_type="data",
    num_processes=10,          # kept for API, but no longer used
    sample_size=1000,
    num_points_quad=20,
    optimized=False,
    copula_only=False,
    min_val=-5,
    max_val=5,
    likelihood_based_metrics=True,
):

    # ------------------------------------------------------------------
    # 1. Put evaluation_data on the same device as the model
    # ------------------------------------------------------------------
    try:
        device = next(self.parameters()).device
    except Exception:
        # fallback if GTM doesn't behave like nn.Module
        device = torch.device("cpu")

    if evaluation_data_type == "data":
        evaluation_data = y[:sample_size].to(device)
        if copula_only:
            evaluation_data = self.after_transformation(evaluation_data)
    elif evaluation_data_type == "uniform_random_samples":
        evaluation_data = torch.distributions.Uniform(min_val, max_val).sample(
            [sample_size, self.y_train.size(1)]
        ).to(device)
    elif evaluation_data_type == "samples_from_model":
        evaluation_data = self.sample(sample_size).to(device)
        if copula_only:
            evaluation_data = self.after_transformation(evaluation_data)

    if copula_only:
        self.num_trans_layers = 0

    # ------------------------------------------------------------------
    # 2. Compute likelihood + pseudo precision / correlation
    #    Keep heavy stuff on device, only move what we need to CPU.
    # ------------------------------------------------------------------
    with torch.no_grad():
        ll_evaluation_data = self.log_likelihood(evaluation_data).detach()  # [N]
        precision_matrix = (
            self.compute_pseudo_precision_matrix(evaluation_data)
            .detach()
            .cpu()
        )  # [S, d, d] on CPU
        correlation_matrix = (
            self.compute_pseudo_conditional_correlation_matrix(evaluation_data)
            .detach()
            .cpu()
        )  # [S, d, d] on CPU

    ll_evaluation_data_cpu = ll_evaluation_data.cpu()

    precision_matrix_summary_statistics = compute_precision_matrix_summary_statistics(
        precision_matrix
    )

    # ------------------------------------------------------------------
    # 3. Compute row-wise KLD / IAE WITHOUT multiprocessing
    # ------------------------------------------------------------------
    if likelihood_based_metrics:
        actual_log_distribution_glq_list = []
        under_ci_assumption_log_distribution_glq_list = []
        under_ci_assumption_log_distribution_glq_full_data_list = []

        start = time.time()

        # single-process, but still on GPU/CPU according to `device`
        with torch.no_grad():
            for row_num in range(precision_matrix_summary_statistics.shape[0]):
                (
                    actual_log_distribution_glq,
                    under_ci_assumption_log_distribution_glq,
                    under_ci_assumption_log_distribution_glq_full_data,
                ) = independence_kld_process_row(
                    row_num,
                    precision_matrix_summary_statistics,
                    evaluation_data,      # already on correct device
                    self,
                    num_points_quad,
                    optimized,
                    min_val,
                    max_val,
                )

                # keep tensors on CPU for later scalar operations
                actual_log_distribution_glq_list.append(
                    actual_log_distribution_glq.cpu()
                )
                under_ci_assumption_log_distribution_glq_list.append(
                    under_ci_assumption_log_distribution_glq.cpu()
                )
                under_ci_assumption_log_distribution_glq_full_data_list.append(
                    under_ci_assumption_log_distribution_glq_full_data.cpu()
                )

        end = time.time()
        print(f"Time taken (single-process, device={device}): {end - start:.2f}s")
        print("All rows processed.")
    else:
        actual_log_distribution_glq_list = []
        under_ci_assumption_log_distribution_glq_list = []
        under_ci_assumption_log_distribution_glq_full_data_list = []

    # ------------------------------------------------------------------
    # 4. Summaries (precision, correlation, KLD, IAE) on CPU
    # ------------------------------------------------------------------
    precision_abs_mean_list = []
    precision_square_mean_list = []
    cond_correlation_abs_mean_list = []
    cond_correlation_square_mean_list = []

    if likelihood_based_metrics:
        kld_list = []
        iae_list = []

    for row_num in range(precision_matrix_summary_statistics.shape[0]):
        var_row_num = int(precision_matrix_summary_statistics.iloc[row_num]["var_row"])
        var_col_num = int(precision_matrix_summary_statistics.iloc[row_num]["var_col"])

        precision_abs_mean = precision_matrix[:, var_row_num, var_col_num].abs().mean()
        precision_square_mean = (
            precision_matrix[:, var_row_num, var_col_num].square().mean()
        )

        cond_correlation_abs_mean = (
            correlation_matrix[:, var_row_num, var_col_num].abs().mean()
        )
        cond_correlation_square_mean = (
            correlation_matrix[:, var_row_num, var_col_num].square().mean()
        )

        if likelihood_based_metrics:
            actual_log_distribution_glq = actual_log_distribution_glq_list[row_num]
            under_ci_assumption_log_distribution_glq = (
                under_ci_assumption_log_distribution_glq_list[row_num]
            )
            under_ci_assumption_log_distribution_glq_full_data = (
                under_ci_assumption_log_distribution_glq_full_data_list[row_num]
            )

            if evaluation_data_type in ("data", "samples_from_model"):
                ll_dev = (
                    actual_log_distribution_glq
                    - under_ci_assumption_log_distribution_glq
                )
                ll_dev = ll_dev[~torch.isnan(ll_dev)]
                ll_dev = ll_dev[~torch.isinf(ll_dev)]
                ll_dev = ll_dev[ll_dev.abs() < ll_dev.abs().quantile(0.99)]
                kld = ll_dev.mean()

                actual_conditioning_set_log_distribution_glq = (
                    under_ci_assumption_log_distribution_glq_full_data
                    - under_ci_assumption_log_distribution_glq
                )
                weights = torch.exp(
                    actual_conditioning_set_log_distribution_glq
                    - ll_evaluation_data_cpu
                )
                ll_dev2 = torch.abs(
                    torch.exp(actual_log_distribution_glq)
                    - torch.exp(under_ci_assumption_log_distribution_glq)
                )
                ll_dev2 = ll_dev2[~torch.isnan(ll_dev2)]
                ll_dev2 = ll_dev2[~torch.isinf(ll_dev2)]
                iae = ll_dev2 * weights
                iae = iae[iae < iae.quantile(0.99)]
                iae = iae.mean() / 2

            elif evaluation_data_type == "uniform_random_samples":
                ll_dev = torch.exp(actual_log_distribution_glq) * (
                    actual_log_distribution_glq
                    - under_ci_assumption_log_distribution_glq
                )
                ll_dev = ll_dev[~torch.isnan(ll_dev)]
                ll_dev = ll_dev[~torch.isinf(ll_dev)]
                kld = ll_dev.mean()

                ll_dev2 = torch.abs(
                    torch.exp(actual_log_distribution_glq)
                    - torch.exp(under_ci_assumption_log_distribution_glq)
                )
                ll_dev2 = ll_dev2[~torch.isnan(ll_dev2)]
                ll_dev2 = ll_dev2[~torch.isinf(ll_dev2)]
                iae = ll_dev2
                iae = iae[iae < iae.quantile(0.99)]
                iae = iae.mean() / 2

            kld_list.append(kld.item())
            iae_list.append(iae.item())

        precision_abs_mean_list.append(precision_abs_mean.item())
        precision_square_mean_list.append(precision_square_mean.item())
        cond_correlation_abs_mean_list.append(cond_correlation_abs_mean.item())
        cond_correlation_square_mean_list.append(cond_correlation_square_mean.item())

    precision_matrix_summary_statistics["precision_abs_mean"] = precision_abs_mean_list
    precision_matrix_summary_statistics["precision_square_mean"] = (
        precision_square_mean_list
    )
    precision_matrix_summary_statistics["cond_correlation_abs_mean"] = (
        cond_correlation_abs_mean_list
    )
    precision_matrix_summary_statistics["cond_correlation_square_mean"] = (
        cond_correlation_square_mean_list
    )

    if likelihood_based_metrics:
        precision_matrix_summary_statistics["kld"] = kld_list
        precision_matrix_summary_statistics["iae"] = iae_list

        sub_kld_summary_statistics = precision_matrix_summary_statistics[
            [
                "var_row",
                "var_col",
                "precision_abs_mean",
                "precision_square_mean",
                "cond_correlation_abs_mean",
                "cond_correlation_square_mean",
                "kld",
                "iae",
            ]
        ]
    else:
        sub_kld_summary_statistics = precision_matrix_summary_statistics[
            [
                "var_row",
                "var_col",
                "precision_abs_mean",
                "precision_square_mean",
                "cond_correlation_abs_mean",
                "cond_correlation_square_mean",
            ]
        ]

    if copula_only:
        self.num_trans_layers = 1

    sub_kld_summary_statistics.reset_index(inplace=True)
    return sub_kld_summary_statistics

def compute_conditional_independence_kld_bayesian(
    self: "GTM",
    vi_model: "VI_Model",
    tau_nodes: Optional["TauPack"] = None,
    y=None,
    evaluation_data_type="data",
    num_processes=1,          # Not Used, tbignored
    sample_size=1000,
    num_points_quad=20,
    optimized=False,
    copula_only=False,
    min_val=-5.0,
    max_val=5.0,
    likelihood_based_metrics=True,
    S_posterior: int = 32,
    cred_level: float = 0.95,
):
    """
    Bayesian version:
      - Draw S_posterior samples from q(θ, τ)
      - For each draw, compute the conditional independence table (metrics per pair)
      - Aggregate over draws to get posterior means and credible intervals
    """

    # 0) Setup ------------------------------------------------------------
    device = next(self.parameters()).device

    # evaluation data (same for all posterior draws)
    evaluation_data = get_evaluation_data(self, y, evaluation_data_type, sample_size, copula_only, min_val, max_val, device)

    if copula_only:
        old_num_trans_layers = self.num_trans_layers
        self.num_trans_layers = 0
    else:
        old_num_trans_layers = None

    decor_present = not (self.number_decorrelation_layers == 0 or self.transform_only)

    # We'll need the list of pairs once. We get it by calling the existing
    # precision summary function on one "reference" parameter setting.
    
    with torch.no_grad():
        precision_ref = (
            self.compute_pseudo_precision_matrix(evaluation_data)
            .detach()
            .cpu()
        )  # [S_eval, d, d]
        
    precision_summary = compute_precision_matrix_summary_statistics(precision_ref)
    n_pairs = precision_summary.shape[0]

    # Arrays to store metrics across posterior samples:
    metrics_precision_sq      = []
    metrics_precision_abs     = []
    metrics_corr_abs          = []
    metrics_corr_sq           = []
    metrics_kld               = [] if likelihood_based_metrics else None
    metrics_iae               = [] if likelihood_based_metrics else None

    # 1) Draw S_posterior samples from q(θ, τ) ---------------------------
    thetas = vi_model.sample_theta(S_posterior, antithetic=True)  # [S, D]
    tau4_vec, tau1_vec, tau2_vec = sample_tau_vectors(self, vi_model, tau_nodes, S_posterior, device, decor_present)

    alpha = 1.0 - cred_level

    # 2) Loop over posterior draws ---------------------------------------
    for s in range(S_posterior):
        theta_s = thetas[s]
        params_s = vi_model._theta_to_state_dict(theta_s)

        t4 = tau4_vec[s]
        t1 = tau1_vec[s]
        t2 = tau2_vec[s]

        # Instantiate model with (θ_s, τ_s)
        with _reparametrize_module(self, params_s):
            # write τ into self.hyperparameter so log_likelihood etc use them
            self.hyperparameter["transformation"]["tau"] = float(t4.item())
            if decor_present:
                self.hyperparameter["decorrelation"]["tau_1"] = float(t1.item())
                self.hyperparameter["decorrelation"]["tau_2"] = float(t2.item())

            with torch.no_grad():
                ll_evaluation_data = self.log_likelihood(evaluation_data).detach()
                precision_matrix = (
                    self.compute_pseudo_precision_matrix(evaluation_data)
                    .detach()
                    .cpu()
                )  # [S_eval, d, d]
                correlation_matrix = (
                    self.compute_pseudo_conditional_correlation_matrix(evaluation_data)
                    .detach()
                    .cpu()
                )  # [S_eval, d, d]

        ll_eval_cpu = ll_evaluation_data.cpu()

        # Pairwise summaries for this posterior draw
        prec_abs_s  = []
        prec_sq_s   = []
        corr_abs_s  = []
        corr_sq_s   = []
        kld_s       = [] if likelihood_based_metrics else None
        iae_s       = [] if likelihood_based_metrics else None

        for row_num in range(n_pairs):
            var_row = int(precision_summary.iloc[row_num]["var_row"])
            var_col = int(precision_summary.iloc[row_num]["var_col"])

            # precision / correlation summaries over evaluation data
            prec_vals = precision_matrix[:, var_row, var_col]
            corr_vals = correlation_matrix[:, var_row, var_col]

            prec_abs_s.append(float(prec_vals.abs().mean()))
            prec_sq_s.append(float(prec_vals.square().mean()))
            corr_abs_s.append(float(corr_vals.abs().mean()))
            corr_sq_s.append(float(corr_vals.square().mean()))

            if likelihood_based_metrics:
                # For each pair and posterior draw, compute the KLD/IAE row via your existing helper
                (
                    actual_log_distribution_glq,
                    under_ci_assumption_log_distribution_glq,
                    under_ci_assumption_log_distribution_glq_full_data,
                ) = independence_kld_process_row(
                    row_num,
                    precision_summary,
                    evaluation_data,  # still on device
                    self,
                    num_points_quad,
                    optimized,
                    min_val,
                    max_val,
                )

                # move to CPU for safe manipulations
                actual = actual_log_distribution_glq.cpu()
                under  = under_ci_assumption_log_distribution_glq.cpu()
                under_full = under_ci_assumption_log_distribution_glq_full_data.cpu()

                if evaluation_data_type in ("data", "samples_from_model"):
                    ll_dev = actual - under
                    ll_dev = ll_dev[~torch.isnan(ll_dev)]
                    ll_dev = ll_dev[~torch.isinf(ll_dev)]
                    ll_dev = ll_dev[ll_dev.abs() < ll_dev.abs().quantile(0.99)]
                    kld = ll_dev.mean()

                    actual_cond = under_full - under
                    weights = torch.exp(actual_cond - ll_eval_cpu)
                    ll_dev2 = torch.abs(torch.exp(actual) - torch.exp(under))
                    ll_dev2 = ll_dev2[~torch.isnan(ll_dev2)]
                    ll_dev2 = ll_dev2[~torch.isinf(ll_dev2)]
                    iae_vals = ll_dev2 * weights
                    iae_vals = iae_vals[iae_vals < iae_vals.quantile(0.99)]
                    iae = iae_vals.mean() / 2.0

                elif evaluation_data_type == "uniform_random_samples":
                    ll_dev = torch.exp(actual) * (actual - under)
                    ll_dev = ll_dev[~torch.isnan(ll_dev)]
                    ll_dev = ll_dev[~torch.isinf(ll_dev)]
                    kld = ll_dev.mean()

                    ll_dev2 = torch.abs(torch.exp(actual) - torch.exp(under))
                    ll_dev2 = ll_dev2[~torch.isnan(ll_dev2)]
                    ll_dev2 = ll_dev2[~torch.isinf(ll_dev2)]
                    iae_vals = ll_dev2
                    iae_vals = iae_vals[iae_vals < iae_vals.quantile(0.99)]
                    iae = iae_vals.mean() / 2.0
                else:
                    raise ValueError("unknown evaluation_data_type in Bayesian KLD")

                kld_s.append(float(kld))
                iae_s.append(float(iae))

        # append this draw's metrics
        metrics_precision_abs.append(prec_abs_s)
        metrics_precision_sq.append(prec_sq_s)
        metrics_corr_abs.append(corr_abs_s)
        metrics_corr_sq.append(corr_sq_s)
        if likelihood_based_metrics:
            metrics_kld.append(kld_s)
            metrics_iae.append(iae_s)

    # 3) Convert to tensors & aggregate over posterior draws -------------
    metrics_precision_abs = torch.tensor(metrics_precision_abs)  # [S, n_pairs]
    metrics_precision_sq  = torch.tensor(metrics_precision_sq)
    metrics_corr_abs      = torch.tensor(metrics_corr_abs)
    metrics_corr_sq       = torch.tensor(metrics_corr_sq)

    if likelihood_based_metrics:
        metrics_kld = torch.tensor(metrics_kld)
        metrics_iae = torch.tensor(metrics_iae)

    # Posterior means
    prec_abs_mean  = metrics_precision_abs.mean(dim=0).tolist()
    prec_sq_mean   = metrics_precision_sq.mean(dim=0).tolist()
    corr_abs_mean  = metrics_corr_abs.mean(dim=0).tolist()
    corr_sq_mean   = metrics_corr_sq.mean(dim=0).tolist()
    if likelihood_based_metrics:
        kld_mean = metrics_kld.mean(dim=0).tolist()
        iae_mean = metrics_iae.mean(dim=0).tolist()

    # Credible intervals per metric & pair
    q_lo = alpha / 2.0
    q_hi = 1.0 - alpha / 2.0

    prec_abs_lo, prec_abs_hi = ci_from_samples(metrics_precision_abs, q_lo, q_hi)
    corr_abs_lo, corr_abs_hi = ci_from_samples(metrics_corr_abs, q_lo, q_hi)
    
    if likelihood_based_metrics:
        kld_lo, kld_hi = ci_from_samples(metrics_kld, q_lo, q_hi)
        iae_lo, iae_hi = ci_from_samples(metrics_iae, q_lo, q_hi)

    # 4) Build final DataFrame similar to the frequentist version --------
    df = precision_summary.copy()  # has columns var_row, var_col, etc.

    df["precision_abs_mean"] = prec_abs_mean
    df["precision_abs_lo"]   = prec_abs_lo
    df["precision_abs_hi"]   = prec_abs_hi

    df["precision_square_mean"] = prec_sq_mean
    df["cond_correlation_abs_mean"] = corr_abs_mean
    df["cond_correlation_abs_lo"]   = corr_abs_lo
    df["cond_correlation_abs_hi"]   = corr_abs_hi
    df["cond_correlation_square_mean"] = corr_sq_mean

    if likelihood_based_metrics:
        df["kld_mean"] = kld_mean
        df["kld_lo"]   = kld_lo
        df["kld_hi"]   = kld_hi
        df["iae_mean"] = iae_mean
        df["iae_lo"]   = iae_lo
        df["iae_hi"]   = iae_hi

    if copula_only and old_num_trans_layers is not None:
        self.num_trans_layers = old_num_trans_layers

    df.reset_index(inplace=True)
    return df

def sample_tau_vectors(self: "GTM", vi_model: "VI_Model", tau_nodes, S_posterior, device, decor_present):
    if tau_nodes is not None:
        tau4_vec, tau1_vec, tau2_vec, _ = tau_nodes.sample_many(
            S=S_posterior,
            decor_present=decor_present,
            generator=vi_model._rng,
        )
    else:
        # fall back to fixed hyperparameters on the model
        tau4_vec = torch.full(
            (S_posterior,),
            float(self.hyperparameter["transformation"]["tau"]),
            device=device,
            dtype=torch.float32,
        )
        if decor_present:
            tau1_vec = torch.full(
                (S_posterior,),
                float(self.hyperparameter["decorrelation"]["tau_1"]),
                device=device,
                dtype=torch.float32,
            )
            tau2_vec = torch.full(
                (S_posterior,),
                float(self.hyperparameter["decorrelation"]["tau_2"]),
                device=device,
                dtype=torch.float32,
            )
        else:
            tau1_vec = torch.zeros((S_posterior,), device=device)
            tau2_vec = torch.zeros((S_posterior,), device=device)
    return tau4_vec,tau1_vec,tau2_vec

def get_evaluation_data(self: "GTM", y, evaluation_data_type, sample_size, copula_only, min_val, max_val, device):
    if evaluation_data_type == "data":
        if y is None:
            raise ValueError("y must be provided when evaluation_data_type='data'")
        evaluation_data = y[:sample_size].to(device)
        if copula_only:
            evaluation_data = self.after_transformation(evaluation_data)

    elif evaluation_data_type == "uniform_random_samples":
        evaluation_data = torch.distributions.Uniform(min_val, max_val).sample(
            [sample_size, self.y_train.size(1)]
        ).to(device)

    elif evaluation_data_type == "samples_from_model":
        evaluation_data = self.sample(sample_size).to(device)
        if copula_only:
            evaluation_data = self.after_transformation(evaluation_data)
    else:
        raise ValueError(f"evaluation_data_type {evaluation_data_type} not understood.")
    return evaluation_data

def ci_from_samples(t, q_lo=0.025, q_hi=0.975):
        qs = torch.quantile(t, torch.tensor([q_lo, q_hi], device=t.device), dim=0)
        return qs[0].tolist(), qs[1].tolist()
