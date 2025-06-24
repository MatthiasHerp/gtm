import multiprocessing
import pickle
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
    num_processes=10,
    sample_size=1000,
    num_points_quad=20,
    optimized=False,
    copula_only=False,
    min_val=-5,
    max_val=5,
    likelihood_based_metrics=True,
):
    # in case of gpu cuda compute
    if evaluation_data_type == "data":
        evaluation_data = y[:sample_size]  # Adjust this based on your needs
        if copula_only == True:
            evaluation_data = self.after_transformation(evaluation_data)
    elif evaluation_data_type == "uniform_random_samples":
        evaluation_data = torch.distributions.Uniform(min_val, max_val).sample(
            [sample_size, self.y_train.size(1)]
        )
    elif evaluation_data_type == "samples_from_model":
        evaluation_data = self.sample(sample_size).detach()
        if copula_only == True:
            evaluation_data = self.after_transformation(evaluation_data).detach()

    if copula_only == True:
        self.num_trans_layers = 0
    ll_evaluation_data = self.log_likelihood(evaluation_data).detach().cpu()

    precision_matrix = (
        self.compute_pseudo_precision_matrix(evaluation_data).detach().cpu()
    )
    correlation_matrix = (
        self.compute_pseudo_conditional_correlation_matrix(evaluation_data)
        .detach()
        .cpu()
    )

    precision_matrix_summary_statistics = compute_precision_matrix_summary_statistics(
        precision_matrix
    )

    if likelihood_based_metrics == True:
        actual_log_distribution_glq_list = []
        under_ci_assumption_log_distribution_glq_list = []

        start = time.time()

        # Using Pool from the multiprocessing module
        if num_processes > 1:
            with multiprocessing.Pool(processes=num_processes) as pool:
                results = pool.starmap(
                    independence_kld_process_row,
                    [
                        (
                            row_num,
                            precision_matrix_summary_statistics,
                            evaluation_data,
                            self,
                            num_points_quad,
                            optimized,
                            min_val,
                            max_val,
                        )
                        for row_num in range(
                            precision_matrix_summary_statistics.shape[0]
                        )
                    ],
                )
            # Unpacking the results
            (
                actual_log_distribution_glq_list,
                under_ci_assumption_log_distribution_glq_list,
                under_ci_assumption_log_distribution_glq_full_data_list,
            ) = zip(*results)
        else:
            results = [
                independence_kld_process_row(
                    row_num,
                    precision_matrix_summary_statistics,
                    evaluation_data,
                    self,
                    num_points_quad,
                    optimized,
                    min_val,
                    max_val,
                )
                for row_num in range(precision_matrix_summary_statistics.shape[0])
            ]

            (
                actual_log_distribution_glq_list,
                under_ci_assumption_log_distribution_glq_list,
                under_ci_assumption_log_distribution_glq_full_data_list,
            ) = zip(*results)

        end = time.time()

        print(f"Time taken: {end - start}")

        print("All rows processed.")

    precision_abs_mean_list = []
    precision_square_mean_list = []
    cond_correlation_abs_mean_list = []
    cond_correlation_square_mean_list = []
    if likelihood_based_metrics == True:
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

        if likelihood_based_metrics == True:
            actual_log_distribution_glq = actual_log_distribution_glq_list[row_num]
            under_ci_assumption_log_distribution_glq = (
                under_ci_assumption_log_distribution_glq_list[row_num]
            )
            under_ci_assumption_log_distribution_glq_full_data = (
                under_ci_assumption_log_distribution_glq_full_data_list[row_num]
            )

            # in case of gpu cuda compute
            if (
                evaluation_data_type == "data"
                or evaluation_data_type == "samples_from_model"
            ):
                ll_dev = (
                    actual_log_distribution_glq
                    - under_ci_assumption_log_distribution_glq
                )
                ll_dev = ll_dev[~torch.isnan(ll_dev)]
                ll_dev = ll_dev[~torch.isinf(ll_dev)]
                ll_dev = ll_dev[ll_dev.abs() < ll_dev.abs().quantile(0.99)]
                kld = ll_dev.mean()

                # p4_glq + p5_glq + p2_glq - (p4_glq+p5_glq) = p2_glq
                # p2_glq: # compute p2 = log(f(Y_{/ij}))
                actual_conditioning_set_log_distribution_glq = (
                    under_ci_assumption_log_distribution_glq_full_data
                    - under_ci_assumption_log_distribution_glq
                )
                weights = torch.exp(
                    actual_conditioning_set_log_distribution_glq - ll_evaluation_data
                )  # p4_glq + p5_glq + p2_glq - (p4_glq+p5_glq) = p2_glq
                ll_dev2 = torch.abs(
                    torch.exp(actual_log_distribution_glq)
                    - torch.exp(under_ci_assumption_log_distribution_glq)
                )
                ll_dev2 = ll_dev2[~torch.isnan(ll_dev2)]
                ll_dev2 = ll_dev2[~torch.isinf(ll_dev2)]
                # correct
                iae = ll_dev2 * weights
                iae = iae[iae < iae.quantile(0.99)]
                iae = iae.mean() / 2

                # old
                # ll_dev2 = torch.abs(torch.exp(actual_log_distribution_glq) - torch.exp(under_ci_assumption_log_distribution_glq)) / torch.exp(actual_log_distribution_glq)
                # ll_dev2 = ll_dev2[~torch.isnan(ll_dev2)]
                # ll_dev2 = ll_dev2[~torch.isinf(ll_dev2)]
                # ll_dev2 = ll_dev2[ll_dev2.abs() < ll_dev2.abs().quantile(0.98)]
                # iae = ll_dev2.mean()

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

        # print("Finished row_num: ", row_num)

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

    if likelihood_based_metrics == True:
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

    if copula_only == True:
        self.num_trans_layers = 1

    sub_kld_summary_statistics.reset_index(inplace=True)

    return sub_kld_summary_statistics
