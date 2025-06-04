import torch
import seaborn as sns
import matplotlib.pyplot as plt
from itertools import combinations
import warnings
import numpy as np


def plot_densities(data,covariate=False,x_lim=None,y_lim=None,density_plot=True):

    # Ensures that by default all points are in the plot and axis have the same span (not distortion, can see distribution clearly)
    if x_lim is None:
        x_lim = [data.min(),data.max()]
    if y_lim is None:
        y_lim = [data.min(),data.max()]

    num_cols = data.shape[1]
    num_combinations = int(num_cols * (num_cols - 1) / 2)
    if covariate is False:
        numbers_covariates = 1
    else:

        #binning the covariates
        # as explained here: https://discuss.pytorch.org/t/binning-tensor-values/89998
        covariate = torch.round(covariate/0.2)*0.2
        numbers_covariates = torch.unique(covariate).size(0)
        covariate_values = torch.unique(covariate)

    if num_combinations > 1 :
        if covariate is False:
            number_rows = int(np.ceil(num_combinations / 3))
            number_cols = 3
        else:
            number_rows = num_combinations
            number_cols = 6
        fig, axs = plt.subplots(nrows=number_rows, ncols=number_cols, squeeze=False,
                                figsize=(5*number_cols,5*number_rows),
                                gridspec_kw={'wspace': 0.25, 'hspace': 0.2})
        a=0
        for i, j in combinations(range(num_cols), 2):
            if i != j:
                if covariate is False:
                    row = int(a // 3)  # Get the row index
                    col = int(a % 3)  # Get the column index
                    sns.scatterplot(x=data[:,i], y=data[:,j], alpha=0.6, color="k", ax=axs[row,col])
                    #TODO:  got ValueError: Contour levels must be increasing
                    # suggestion on whre it came from: https://stackoverflow.com/questions/62233779/valueerror-contour-levels-must-be-increasing-contour-plot-in-python
                    if density_plot==True:
                        #clean_data = data[~np.isnan(data).any(axis=1) & ~np.isinf(data).any(axis=1)]
                        mask = ~(torch.isnan(data).any(dim=1) | torch.isinf(data).any(dim=1))
                        clean_data = data[mask]
                        sns.kdeplot(x=clean_data[:, i], y=clean_data[:, j], fill=True, alpha=0.9, ax=axs[row,col])

                    axs[row, col].set_xlabel("y_" + str(i))
                    axs[row, col].set_ylabel("y_" + str(j))

                    axs[row, col].set_xlim(x_lim)
                    axs[row, col].set_ylim(y_lim)

                else:
                    #warnings.warn("multivariate and covariate combination not implemented as grid yet")
                    for c in range(numbers_covariates):
                        sub_data = data[covariate == covariate_values[c]]
                        sub_covariate = covariate[covariate == covariate_values[c]]
                        sns.scatterplot(x=sub_data[:, i], y=sub_data[:, j], hue=sub_covariate, alpha=0.6, color="k", ax=axs[a,c])

                        if density_plot == True:
                            sns.kdeplot(x=sub_data[:,i], y=sub_data[:,j], fill=True, alpha=0.9, ax=axs[a,c])

                        #no labels

                        axs[a,c].set_xlim(x_lim)
                        axs[a,c].set_ylim(y_lim)

                a += 1

        plt.subplots_adjust(wspace=0.05)

    else:
        fig, ax = plt.subplots(nrows=numbers_covariates, figsize=(6, 6*numbers_covariates))
        if covariate is False:
            sns.scatterplot(x=data[:,0], y=data[:,1], alpha=0.6, color="k", ax=ax)
            sns.kdeplot(x=data[:, 0], y=data[:, 1], fill=True, alpha=0.9, ax=ax)

            ax.set_xlabel(r"$Y_1$", fontsize=14)
            ax.set_ylabel(r"$Y_2$", fontsize=14)
        else:
            for c in range(numbers_covariates):
                sub_data = data[covariate == covariate_values[c]]
                sub_covariate = covariate[covariate == covariate_values[c]]
                sns.scatterplot(x=sub_data[:, 0], y=sub_data[:, 1], hue=sub_covariate, alpha=0.6, color="k",
                                ax=ax[c])
                sns.kdeplot(x=sub_data[:, 0], y=sub_data[:, 1], fill=True, alpha=0.9, ax=ax[c,])

                ax[c,].set_xlabel(r"$Y_1$", fontsize=14)
                ax[c,].set_ylabel(r"$Y_2$", fontsize=14)

        if x_lim is not None:
            ax.set_xlim(x_lim)
        if y_lim is not None:
            ax.set_ylim(y_lim)

    return fig