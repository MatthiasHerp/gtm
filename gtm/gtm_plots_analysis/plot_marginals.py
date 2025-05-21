import torch
import seaborn as sns
import matplotlib.pyplot as plt
from itertools import combinations
import warnings
import numpy as np

def plot_marginals(data,covariate=False,names=False, y_lim=False):
    """
    function takes data and creates a grid of histograms of each column in the data tensor

    :param data: torch.tensor
    :return:
    """

    if covariate is not False:
        #here 0.33 to have less groups for better overview in plots
        covariate = torch.round(covariate / 0.33) * 0.33
        covariate = covariate.detach().numpy()
        #numbers_covariates = torch.unique(covariate).size(0)
        #covariate_values = torch.unique(covariate)
        #sns.color_palette("rocket")

    #data = data.numpy()
    num_rows = int(np.ceil(data.shape[1]/3))
    num_cols = 3

    fig, axs = plt.subplots(nrows=num_rows, ncols=num_cols, squeeze=False,
                            figsize=(15, 5 * num_rows),
                            gridspec_kw={'wspace': 0.25, 'hspace': 0.2}, sharey=False, sharex=False)  # dont want to share all x (sharex) and y axis (sharey)

    for y_num in range(data.shape[1]):
        row = int(y_num // 3)  # Get the row index
        col = int(y_num % 3)  # Get the column index

        if covariate is not False:
            sns.histplot(x=data[:,y_num], hue=covariate, ax=axs[row,col], palette="rocket", bins=8, multiple="dodge",stat="density", common_norm=False)
        else:
            sns.histplot(x=data[:,y_num], ax=axs[row,col])
        if names == False:
            axs[row, col].set_xlabel("y_" + str(y_num))
        else:
            axs[row, col].set_xlabel(names[y_num])
        
        if y_lim is not False:
            axs[row, col].set_ylim(y_lim)

    return fig