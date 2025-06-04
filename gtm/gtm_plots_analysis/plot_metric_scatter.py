import torch
import seaborn as sns
import matplotlib.pyplot as plt
from itertools import combinations
import warnings
import numpy as np

def plot_metric_scatter(data, metric, covariate=False, x_lim=None, y_lim=None, metric_type="matrix", 
                        pairs=False, strength_value=False, strength_name="", show_colorbar=True, hide_axis_info=False, sub_title_fontsize=10,
                        after_marginal_transformation=False, label_metric="Pseudo Conditional Correlation"):
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
    if pairs is not False:
        num_combinations = len(pairs)
    else:
        num_combinations = int(num_cols * (num_cols - 1) / 2)

    if num_combinations > 1:
        number_rows = int(np.ceil(num_combinations / 4))
        number_cols = 4
        fig, axs = plt.subplots(nrows=number_rows, ncols=number_cols, squeeze=False,
                                figsize=(15, 5 * number_rows),
                                gridspec_kw={'wspace': 0.25, 'hspace': 0.2})
        a = 0
        if pairs is not False:
            combis = pairs
        else:
            combis = combinations(range(num_cols), 2)
            
        for i, j in combis: #combinations(range(num_cols), 2):
            #if pairs is not False:
            #    if [i,j] not in pairs:
            #            continue
            if i != j:
                row = int(a // 4)  # Get the row index
                col = int(a % 4)  # Get the column index

                if covariate is not False:
                    warnings.warn("Covariate is not supported for 3d data yet")

                max_deviance = np.max(abs(metric))  # ensures that the colorbar is centered
                norm = plt.Normalize(-max_deviance, max_deviance)

                if metric_type == "vector":
                    hb = axs[row, col].hexbin(data[:, i], data[:, j], C=metric, gridsize=60, cmap='icefire', norm=norm, reduce_C_function=np.mean)
                elif metric_type == "matrix":
                    hb = axs[row, col].hexbin(data[:, i], data[:, j], C=metric[:, i, j], gridsize=60, cmap='icefire', norm=norm, reduce_C_function=np.mean)

                if hide_axis_info is not True:
                    axs[row, col].set_xlim(x_lim)
                    axs[row, col].set_ylim(y_lim)
                    axs[row, col].set_xlabel(f"y_{i}")
                    axs[row, col].set_ylabel(f"y_{j}")
                else:
                    axs[row, col].set_xticks([])  # Remove x-axis ticks
                    axs[row, col].set_yticks([])  # Remove y-axis ticks
                    
                    
                if strength_value is not False:
                    if after_marginal_transformation == False:
                        axs[row, col].set_title(f"({a+1}) $Y_{i}$,$Y_{j}$ " + strength_name.upper() + ": " + str(np.round(strength_value[a],3)),fontsize=sub_title_fontsize)
                    elif after_marginal_transformation == True:
                        axs[row, col].set_title(r"(%d) $\tilde{Z}_{%d}, \tilde{Z}_{%d}$ " % (a+1,i, j) + strength_name.upper() + ": " + str(np.round(strength_value[a],3)),fontsize=sub_title_fontsize)
                    
                    

                a += 1

        # Add a colorbar to the last plot
        if show_colorbar==True:
            cb = fig.colorbar(hb, ax=axs, orientation='vertical')
            cb.set_label(label=label_metric, fontsize=sub_title_fontsize) 

    else:
        fig, ax = plt.subplots(figsize=(6, 6))
        
        max_deviance = np.max(abs(metric))  # ensures that the colorbar is centered
        norm = plt.Normalize(-max_deviance, max_deviance)

        if metric_type == "vector":
            hb = ax.hexbin(data[:, 0], data[:, 1], C=metric, gridsize=30, cmap='icefire', norm=norm, reduce_C_function=np.mean)
        elif metric_type == "matrix":
            hb = ax.hexbin(data[:, 0], data[:, 1], C=metric[:, 0, 1], gridsize=30, cmap='icefire', norm=norm, reduce_C_function=np.mean)

        if hide_axis_info is not True:
            ax.set_xlabel(f"y_{0}")
            ax.set_ylabel(f"y_{1}")
            ax.set_xlim(x_lim)
            ax.set_ylim(y_lim)
            
        if strength_value is not False:
            if after_marginal_transformation == False:
                ax.set_title(f"($Y_{0}$,$Y_{1}$)   "+strength_name +str(np.round(strength_value[0],3)), fontsize=sub_title_fontsize)
            elif after_marginal_transformation == True:
                ax.set_title(r"($\tilde{Z}_{0}, \tilde{Z}_{1}$)" + strength_name + str(np.round(strength_value[0], 3)), fontsize=sub_title_fontsize)

        # Add a colorbar
        if show_colorbar==True:
            cb = fig.colorbar(hb, ax=ax, orientation='vertical')
            cb.set_label(label=label_metric, fontsize=sub_title_fontsize) 

    # Remove the last (9th) empty plot
    #fig.delaxes(axs[2,2])

    return fig