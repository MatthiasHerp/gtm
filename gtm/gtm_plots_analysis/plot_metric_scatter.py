import torch
import seaborn as sns
import matplotlib.pyplot as plt
from itertools import combinations
import warnings
import numpy as np

#kl_divergence_scatter
def plot_metric_scatter(data, metric, covariate=False, x_lim=None, y_lim=None, metric_type="kl_divergence"):
    # Ensures that by default all points are in the plot and axis have the same span (not distortion, can see distribution clearly)
    if x_lim is None:
        x_lim = [data.min(), data.max()]
    if y_lim is None:
        y_lim = [data.min(), data.max()]

    if torch.is_tensor(data):
        data = data.detach().numpy()
    if torch.is_tensor(metric):
        metric = metric.detach().numpy()

    num_cols = data.shape[1]
    num_combinations = int(num_cols * (num_cols - 1) / 2)

    if num_combinations > 1:
        number_rows = int(np.ceil(num_combinations / 3))
        number_cols = 3
        fig, axs = plt.subplots(nrows=number_rows, ncols=number_cols, squeeze=False,
                                figsize=(15, 5 * number_rows),
                                gridspec_kw={'wspace': 0.25, 'hspace': 0.2})
        a = 0
        for i, j in combinations(range(num_cols), 2):
            if i != j:
                row = int(a // 3)  # Get the row index
                col = int(a % 3)  # Get the column index
                if covariate is not False:
                    warnings.warn("Covariate is not supported for 3d data yet")

                max_deviance = np.max(abs(metric))  # ensures that the colorbar is centered
                norm = plt.Normalize(-max_deviance, max_deviance)

                # palette from here:
                # https: // seaborn.pydata.org / tutorial / color_palettes.html
                if metric_type == "kl_divergence":
                    sns.scatterplot(x=data[:, i], y=data[:, j], hue=metric, ax=axs[row,col], palette='icefire')
                elif metric_type == "precision_matrix":
                    sns.scatterplot(x=data[:, i], y=data[:, j], hue=metric[:,i,j], hue_norm=norm, ax=axs[row,col], palette='icefire')

                # Create a scalar mappable to show the legend
                # https://stackoverflow.com/questions/62884183/trying-to-add-a-colorbar-to-a-seaborn-scatterplot
                sm = plt.cm.ScalarMappable(cmap="icefire", norm=norm)
                sm.set_array([])

                axs[row,col].set_xlim(x_lim)
                axs[row,col].set_ylim(y_lim)

                axs[row,col].set_xlabel("y_" + str(i))
                axs[row,col].set_ylabel("y_" + str(j))

                # Remove the legend and add a colorbar
                axs[row,col].get_legend().remove()

                a += 1

        #axs[0,number_cols-1].figure.colorbar(sm)

    else:
        fig, ax = plt.subplots(figsize=(6, 6))
        
        max_deviance = np.max(abs(metric))  # ensures that the colorbar is centered
        norm = plt.Normalize(-max_deviance, max_deviance)

        # palette from here:
        #https: // seaborn.pydata.org / tutorial / color_palettes.html
        if metric_type == "kl_divergence":
            sns.scatterplot(x=data[:,0], y=data[:,1], hue=metric, hue_norm=norm, ax = ax, palette='icefire')
        elif metric_type == "precision_matrix":
            sns.scatterplot(x=data[:, 0], y=data[:, 1], hue=metric[:, 0, 1], hue_norm=norm, ax=ax, palette='icefire')
        #sns.scatterplot(x=data[:, 0], y=data[:, 1], hue=metric, ax=ax, palette='icefire')

        # Create a scalar mappable to show the legend
        # https://stackoverflow.com/questions/62884183/trying-to-add-a-colorbar-to-a-seaborn-scatterplot
        max_deviance = np.max(abs(metric)) # ensures that the colorbar is centered
        norm = plt.Normalize(-max_deviance, max_deviance)
        sm = plt.cm.ScalarMappable(cmap="icefire", norm=norm)
        sm.set_array([])

        ax.set_xlabel("y_" + str(0))
        ax.set_ylabel("y_" + str(1))

        # Remove the legend and add a colorbar
        ax.get_legend().remove()
        #ax.figure.colorbar(sm)

    return fig