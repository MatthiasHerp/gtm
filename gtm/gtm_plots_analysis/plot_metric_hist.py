from itertools import combinations

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch


def plot_metric_hist(metric, covariate=False, bins=20, storage=None, show_plot=True):
    if torch.is_tensor(metric):
        metric = metric.detach().numpy()

    if covariate is not False:
        # here 0.33 to have less groups for better overview in plots
        covariate = torch.round(covariate / 0.33) * 0.33
        covariate = covariate.detach().numpy()
        # numbers_covariates = torch.unique(covariate).size(0)
        # covariate_values = torch.unique(covariate)
        # sns.color_palette("rocket")

    num_cols = metric.shape[1]
    num_combinations = int(num_cols * (num_cols - 1) / 2)

    if num_combinations > 1:
        number_rows = int(np.ceil(num_combinations / 3))
        number_cols = 3
        fig, axs = plt.subplots(
            nrows=number_rows,
            ncols=number_cols,
            squeeze=False,
            figsize=(15, 5 * number_rows),
            gridspec_kw={"wspace": 0.25, "hspace": 0.2},
        )
        a = 0
        for i, j in combinations(range(num_cols), 2):
            if i != j:
                row = int(a // 3)  # Get the row index
                col = int(a % 3)  # Get the column index

                if covariate is not False:
                    # warnings.warn("Covariate is not supported for 3d data yet")
                    sns.histplot(
                        x=metric[:, i, j],
                        hue=covariate,
                        ax=axs[row, col],
                        palette="rocket",
                        bins=bins,
                        multiple="dodge",
                        stat="density",
                        common_norm=False,
                    )
                else:
                    # palette from here:
                    # https: // seaborn.pydata.org / tutorial / color_palettes.html
                    sns.histplot(x=metric[:, i, j], ax=axs[row, col], bins=bins)

                axs[row, col].set_xlabel("y_" + str(i))
                axs[row, col].set_ylabel("y_" + str(j))

                a += 1
    else:
        fig, ax = plt.subplots(figsize=(6, 6))
        # palette from here:
        # https: // seaborn.pydata.org / tutorial / color_palettes.html
        sns.histplot(x=metric[:, 0, 1], ax=ax)

        ax.set_xlabel("y_" + str(0))
        ax.set_ylabel("y_" + str(1))

    if storage:
        plt.savefig(storage, bbox_inches="tight")
    if show_plot:
        plt.show()
    else:
        plt.close(fig)
