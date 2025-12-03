import warnings
from itertools import combinations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch

from gtm.gtm_splines.bernstein_prediction_vectorized import \
    bernstein_prediction_vectorized
# from gtm.gtm_splines.bspline_prediction_old import bspline_prediction
from gtm.gtm_splines.bspline_prediction_vectorized import \
    bspline_prediction_vectorized


def plot_splines(
    layer, covariate_exists=False, affine=False, storage=None, show_plot=True
):

    # num_variables = layer.number_variables
    # num_splines = int(num_variables * (num_variables-1) / 2)
    # num_splines = layer.params.size()[1]
    # num_variables = layer.number_variables

    if layer.type == "transformation":

        spline_range = convert_to_float_tensor(layer)

        poly_min = spline_range[0, :]
        poly_max = spline_range[1, :]

        num_variables = layer.number_variables
        num_splines = num_variables

    elif layer.type == "decorrelation":

        spline_range = convert_to_float_tensor(layer)

        poly_min = spline_range[0, :]
        poly_max = spline_range[1, :]

        num_variables = layer.number_variables
        num_splines = int(num_variables * (num_variables - 1) / 2)

        lower_tri_indices = np.tril_indices(num_splines, k=-1)
        output_splines = torch.zeros(
            (1000, num_splines), device=poly_min.device, dtype=torch.float32
        )

    device = poly_min.device  # convenience

    data_span_vec = torch.zeros(
        (1000, num_variables), dtype=torch.float32, device=device
    )
    for i in range(num_variables):
        data_span_vec[:, i] = torch.linspace(
            poly_min[i], poly_max[i], 1000, device=device
        )

    if layer.type == "transformation":
        results = pd.DataFrame(
            columns=[
                "y",
                "y_estimated",
                "z_tilde",
                "z_tilde_derivativ",
                "covariate",
                "spline_num",
            ]
        )
        if covariate_exists is True:
            for cov_value in [0.0, 0.25, 0.5, 0.75, 1.0]:
                covariate_value = torch.full(
                    (1000,), cov_value, device=device, dtype=torch.float32
                )
                return_dict = layer.forward(
                    data_span_vec, covariate=covariate_value, return_log_d=True
                )
                z_tilde = return_dict["output"]
                log_d = return_dict["log_d"]
                z_tilde_derivativ = torch.exp(log_d)
                return_dict_inverse = layer.forward(
                    z_tilde, covariate=covariate_value, inverse=True
                )
                data_span_vec_estimated = return_dict_inverse["output"]

                for spline_num in range(num_variables):
                    results = results._append(
                        pd.DataFrame(
                            {
                                "y": data_span_vec.detach()
                                .cpu()
                                .numpy()[:, spline_num],
                                "y_estimated": data_span_vec_estimated.detach()
                                .cpu()
                                .numpy()[:, spline_num],
                                "z_tilde": z_tilde.detach()
                                .cpu()
                                .numpy()[:, spline_num],
                                "z_tilde_derivativ": z_tilde_derivativ.detach()
                                .cpu()
                                .numpy()[:, spline_num],
                                "covariate": cov_value,
                                "spline_num": spline_num,
                            }
                        ),
                        ignore_index=True,
                    )

        else:
            results = results.drop(("covariate"), axis=1)
            return_dict = layer.forward(
                data_span_vec, covariate=False, return_log_d=True
            )
            z_tilde = return_dict["output"]
            log_d = return_dict["log_d"]

            z_tilde_derivativ = torch.exp(log_d)
            return_dict_inverse = layer.forward(
                z_tilde, covariate=False, inverse=True
            )
            data_span_vec_estimated = return_dict_inverse["output"]

            for spline_num in range(num_variables):
                results = results._append(
                    pd.DataFrame(
                        {
                            "y": data_span_vec.detach()
                            .cpu()
                            .numpy()[:, spline_num],
                            "y_estimated": data_span_vec_estimated.detach()
                            .cpu()
                            .numpy()[:, spline_num],
                            "z_tilde": z_tilde.detach()
                            .cpu()
                            .numpy()[:, spline_num],
                            "z_tilde_derivativ": z_tilde_derivativ.detach()
                            .cpu()
                            .numpy()[:, spline_num],
                            "spline_num": spline_num,
                        }
                    ),
                    ignore_index=True,
                )

    elif layer.type == "decorrelation":
        z_tilde = data_span_vec.clone()
        results = pd.DataFrame(
            columns=[
                "y",
                "y_estimated",
                "z_tilde",
                "z_tilde_derivativ",
                "covariate",
                "spline_num",
            ]
        )

        # Precompute CPU versions for spline evaluation  # CHANGED
        spline_range_cpu_1d = spline_range.detach().cpu()[:, 0]
        knots_cpu = layer.knots.detach().cpu()

        if covariate_exists is True:
            for cov_value in [0.0, 0.25, 0.5, 0.75, 1.0]:
                covariate_value = torch.full(
                    (1000,), cov_value, device=device, dtype=torch.float32
                )
                for spline_num in range(num_splines):

                    if num_splines == 1:
                        col_indices = 0
                    else:
                        col_indices = lower_tri_indices[1][spline_num]

                    # --- all spline inputs to CPU ---  # CHANGED
                    params_cpu = (
                        layer.params[:, spline_num]
                        .detach()
                        .cpu()
                        .unsqueeze(1)
                    )
                    input_cpu = (
                        data_span_vec[:, col_indices]
                        .detach()
                        .cpu()
                        .unsqueeze(1)
                    )

                    if layer.spline == "bspline":
                        output_cpu = bspline_prediction_vectorized(
                            params_cpu,
                            input_cpu,
                            knots_cpu,
                            layer.degree,
                            spline_range_cpu_1d,
                            derivativ=0,
                            return_penalties=False,
                            varying_degrees=False,
                        )
                    elif layer.spline == "bernstein":
                        output_cpu = bernstein_prediction_vectorized(
                            params_cpu,
                            input_cpu,
                            layer.degree,
                            spline_range_cpu_1d,
                            derivativ=0,
                            return_penalties=False,
                            binom_n=layer.binom_n,
                            binom_n1=layer.binom_n1,
                            binom_n2=layer.binom_n2,
                        )

                    # Move back to original device  # CHANGED
                    output_splines[:, spline_num] = output_cpu.to(device)

                    results = results._append(
                        pd.DataFrame(
                            {
                                "y": data_span_vec.detach()
                                .cpu()
                                .numpy()[:, col_indices],
                                "z_tilde": output_splines.detach()
                                .cpu()
                                .numpy()[:, spline_num],
                                "covariate": cov_value,
                                "spline_num": spline_num,
                            }
                        ),
                        ignore_index=True,
                    )
        else:
            results = results.drop(("covariate"), axis=1)

            for spline_num in range(num_splines):
                if num_splines == 1:
                    col_indices = 0
                else:
                    col_indices = lower_tri_indices[1][spline_num]

                if affine == True:
                    params = layer.params_multiplier[:, spline_num]
                    degree = layer.degree_multi
                else:
                    params = layer.params[:, spline_num]
                    degree = layer.degree

                # --- all spline inputs to CPU ---  # CHANGED
                params_cpu = params.detach().cpu().unsqueeze(1)
                input_cpu = (
                    data_span_vec[:, col_indices]
                    .detach()
                    .cpu()
                    .unsqueeze(1)
                )

                if layer.spline == "bspline":
                    output_cpu = bspline_prediction_vectorized(
                        params_cpu,
                        input_cpu,
                        knots_cpu,
                        degree,
                        spline_range_cpu_1d,
                        derivativ=0,
                        return_penalties=False,
                        varying_degrees=False,
                    )
                elif layer.spline == "bernstein":
                    output_cpu = bernstein_prediction_vectorized(
                        params_cpu,
                        input_cpu,
                        degree,
                        spline_range_cpu_1d,
                        derivativ=0,
                        return_penalties=False,
                        binom_n=layer.binom_n,
                        binom_n1=layer.binom_n1,
                        binom_n2=layer.binom_n2,
                    )

                # back to original device for storage  # CHANGED
                output_splines[:, spline_num] = output_cpu.to(device)

                results = results._append(
                    pd.DataFrame(
                        {
                            "y": data_span_vec.detach()
                            .cpu()
                            .numpy()[:, col_indices],
                            "z_tilde": output_splines.detach()
                            .cpu()
                            .numpy()[:, spline_num],
                            "spline_num": spline_num,
                        }
                    ),
                    ignore_index=True,
                )

    if num_splines > 1:
        number_rows = int(np.ceil(num_splines / 3))
        number_cols = 3
        fig, axs = plt.subplots(
            nrows=number_rows,
            ncols=number_cols,
            squeeze=False,
            figsize=(15, 5 * number_rows),
            gridspec_kw={"wspace": 0.25, "hspace": 0.2},
            sharey=False,
            sharex=False,
        )

        for spline_num in range(num_splines):
            subset_results = results[results["spline_num"] == spline_num]
            row = int(spline_num // 3)
            col = int(spline_num % 3)
            if covariate_exists is True:
                sns.lineplot(
                    x="y",
                    y="z_tilde",
                    hue="covariate",
                    data=subset_results,
                    ax=axs[row, col],
                    color="blue",
                )
                if layer.type == "transformation":
                    sns.lineplot(
                        x="y",
                        y="z_tilde_derivativ",
                        hue="covariate",
                        data=subset_results,
                        ax=axs[row, col],
                        color="orange",
                    )
                    sns.lineplot(
                        x="y_estimated",
                        y="z_tilde",
                        hue="covariate",
                        linestyle="--",
                        data=subset_results,
                        ax=axs[row, col],
                        color="green",
                    )
            else:
                sns.lineplot(x="y", y="z_tilde", data=subset_results, ax=axs[row, col])
                if layer.type == "transformation":
                    sns.lineplot(
                        x="y",
                        y="z_tilde_derivativ",
                        data=subset_results,
                        ax=axs[row, col],
                        color="orange",
                    )
                    sns.lineplot(
                        x="y_estimated",
                        y="z_tilde",
                        linestyle="--",
                        data=subset_results,
                        ax=axs[row, col],
                        color="green",
                    )

                    if spline_num == 0:
                        axs[row, col].plot([], [], color="blue", label="spline")
                        axs[row, col].plot([], [], color="orange", label="derivative")
                        axs[row, col].plot(
                            [], [], color="green", linestyle="--", label="inverse"
                        )
                        axs[row, col].legend()

            if layer.type == "transformation":
                axs[row, col].set_ylim(
                    subset_results["z_tilde"].min(), subset_results["z_tilde"].max()
                )
                axs[row, col].set_xlim(
                    subset_results["y"].min(), subset_results["y"].max()
                )

                axs[row, col].set_xlabel("y_" + str(spline_num))
                axs[row, col].set_ylabel("z_tilde" + str(spline_num))

            elif layer.type == "decorrelation":
                axs[row, col].set_ylim(
                    subset_results["z_tilde"].min(), subset_results["z_tilde"].max()
                )
                axs[row, col].set_xlim(
                    subset_results["y"].min(), subset_results["y"].max()
                )

                axs[row, col].set_xlabel("z_tilde")
                axs[row, col].set_ylabel("lambda_" + str(spline_num))

        plt.subplots_adjust(wspace=0.05)

    else:
        fig, ax = plt.subplots(figsize=(6, 6))
        if covariate_exists is True:
            sns.lineplot(
                x="y", y="z_tilde", hue="covariate", data=results, ax=ax, color="blue"
            )
        else:
            sns.lineplot(x="y", y="z_tilde", data=results, ax=ax)
            if layer.type == "transformation":
                sns.lineplot(
                    x="y", y="z_tilde_derivativ", data=results, ax=ax, color="orange"
                )
                sns.lineplot(
                    x="y_estimated",
                    y="z_tilde",
                    linestyle="--",
                    data=results,
                    ax=ax,
                    color="green",
                )
        ax.set_ylim(results["z_tilde"].min(), results["z_tilde"].max())
        ax.set_xlim(results["y"].min(), results["y"].max())

        ax.set_xlabel("z_tilde")
        ax.set_ylabel("lambda_" + str(0))
        if layer.type == "transformation":
            ax.plot([], [], color="blue", label="spline")
            ax.plot([], [], color="orange", label="derivative")
            ax.plot([], [], color="green", linestyle="--", label="inverse")
            ax.legend()

    if storage:
        plt.savefig(storage, bbox_inches="tight")
    if show_plot == True:
        plt.show()


def convert_to_float_tensor(layer):
    """
    Make sure spline_range is a float tensor.
    If it's already a tensor, keep its device and just cast dtype.
    Otherwise, infer device from layer.params (first element) or CPU.
    """
    if torch.is_tensor(layer.spline_range):
        spline_range = layer.spline_range.to(dtype=torch.float32)
    else:
        try:
            if isinstance(layer.params, torch.nn.ParameterList):
                target_device = layer.params[0].device
            else:
                target_device = layer.params.device
        except Exception:
            target_device = torch.device("cpu")

        spline_range = torch.tensor(
            layer.spline_range, dtype=torch.float32, device=target_device
        )

    return spline_range
