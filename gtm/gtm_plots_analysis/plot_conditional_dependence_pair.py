import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib import gridspec


def plot_conditional_dependence_pair(
    loaded_model,
    sample_indices,
    resampled_samples,
    show_colorbar=True,
    title=None,
    storage=None,
    show_plot=True,
    show_ticks=False,  # <-- New argument
):
    # Define figure and custom gridspec
    fig = plt.figure(figsize=(10, 8))
    spec = gridspec.GridSpec(nrows=2, ncols=3, width_ratios=[1, 1, 0.05], wspace=0.3)

    axes = np.empty((2, 2), dtype=object)
    axes[0, 0] = fig.add_subplot(spec[0, 0])
    axes[0, 1] = fig.add_subplot(spec[0, 1])
    axes[1, 0] = fig.add_subplot(spec[1, 0])
    axes[1, 1] = fig.add_subplot(spec[1, 1])
    cax = fig.add_subplot(spec[:, 2])  # This spans both rows

    norm = plt.Normalize(-1, 1)

    # ----------- Top row: Y -----------
    sns.kdeplot(
        x=resampled_samples[:, sample_indices[0]],
        y=resampled_samples[:, sample_indices[1]],
        fill=True,
        alpha=0.9,
        ax=axes[0, 0],
    )

    conditional_correlation_matrix_resampled_samples = (
        loaded_model.compute_pseudo_conditional_correlation_matrix(
            resampled_samples
        ).detach()
    )

    axes[0, 1].hexbin(
        resampled_samples[:, sample_indices[0]].numpy(),
        resampled_samples[:, sample_indices[1]].numpy(),
        C=conditional_correlation_matrix_resampled_samples[
            :, sample_indices[0], sample_indices[1]
        ].detach(),
        gridsize=30,
        cmap="icefire",
        reduce_C_function=np.mean,
        norm=norm,
    )

    # ----------- Bottom row: \tilde{Z} -----------
    resampled_samples_after_transformation = loaded_model.after_transformation(
        resampled_samples
    ).detach()

    sns.kdeplot(
        x=resampled_samples_after_transformation[:, sample_indices[0]],
        y=resampled_samples_after_transformation[:, sample_indices[1]],
        fill=True,
        alpha=0.9,
        ax=axes[1, 0],
    )

    hb = axes[1, 1].hexbin(
        resampled_samples_after_transformation[:, sample_indices[0]].numpy(),
        resampled_samples_after_transformation[:, sample_indices[1]].numpy(),
        C=conditional_correlation_matrix_resampled_samples[
            :, sample_indices[0], sample_indices[1]
        ].detach(),
        gridsize=30,
        cmap="icefire",
        reduce_C_function=np.mean,
        norm=norm,
    )

    # ----------- Formatting -----------
    for i in range(2):
        for j in range(2):
            ax = axes[i, j]

            if not show_ticks:
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_xlabel("")
                ax.set_ylabel("")

            x0, x1 = ax.get_xlim()
            y0, y1 = ax.get_ylim()
            max_range = max(x1 - x0, y1 - y0)
            x_center = (x1 + x0) / 2
            y_center = (y1 + y0) / 2
            ax.set_xlim([x_center - max_range / 2, x_center + max_range / 2])
            ax.set_ylim([y_center - max_range / 2, y_center + max_range / 2])
            ax.set_aspect("equal")

    # ----------- Colorbar -----------
    if show_colorbar:
        fig.colorbar(
            hb, cax=cax, orientation="vertical", label="Conditional Correlation"
        )

    # ----------- Axis labels -----------
    axes[0, 0].set_title("Density", fontsize=13)
    axes[0, 1].set_title("Conditional Correlation", fontsize=13)
    axes[0, 0].set_ylabel("Y", fontsize=13)
    axes[0, 0].set_xlabel(" ", fontsize=13)
    axes[1, 0].set_ylabel(r"$\tilde{Z}$", fontsize=13)
    axes[1, 0].set_xlabel(" ", fontsize=13)

    # ----------- Title and saving -----------
    if title is not None:
        fig.suptitle(title)

    plt.tight_layout(rect=[0, 0, 0.98, 0.96])

    if storage:
        plt.savefig(storage, bbox_inches="tight")

    if show_plot == True:
        plt.show()
    else:
        plt.close(fig)
