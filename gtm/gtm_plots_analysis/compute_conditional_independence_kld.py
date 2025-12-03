import time
import warnings

import pandas as pd
import torch

from gtm.gtm_plots_analysis.compute_precision_matrix_summary_statistics import *
from gtm.gtm_plots_analysis.independence_kld_process_row import *


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
