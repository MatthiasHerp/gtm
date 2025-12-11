import torch
import numpy as np
import matplotlib.pyplot as plt

def plot_theta_ci(mu, lower, upper, order=None, title="θ credible intervals"):
    """
    Plot marginal credible intervals for θ.

    Args
    ----
    mu    : 1D tensor [D]
    lower : 1D tensor [D]
    upper : 1D tensor [D]
    order : optional permutation (1D tensor of indices) to reorder coefficients
    """
    # apply ordering if provided
    if order is not None:
        mu = mu[order]
        lower = lower[order]
        upper = upper[order]

    # move to CPU + numpy
    mu_np    = mu.detach().cpu().numpy()
    lower_np = lower.detach().cpu().numpy()
    upper_np = upper.detach().cpu().numpy()

    D = mu_np.shape[0]
    x = np.arange(D)

    # asymmetrical errors in x-direction (since θ is on x-axis)
    err_low = mu_np - lower_np
    err_hi  = upper_np - mu_np
    xerr = np.vstack([err_low, err_hi])  # shape [2, D]

    plt.figure(figsize=(8, max(4, D * 0.15)))
    plt.errorbar(
        mu_np,
        x,
        xerr=xerr,
        fmt="o",
        capsize=3,
    )
    plt.axvline(0, color="black", linewidth=1)
    plt.gca().invert_yaxis()

    plt.xlabel("θ value")
    plt.ylabel("parameter index")
    plt.title(title)
    plt.tight_layout()
    plt.show()


def plot_theta_ci_group(ci: dict, group_name: str, max_params: int = 200):
    """
    ci: {"mu": Tensor, "lower": Tensor, "upper": Tensor}
    Plots horizontal errorbars for up to `max_params` parameters.
    """
    mu    = ci["mu"][:max_params].numpy()
    lower = ci["lower"][:max_params].numpy()
    upper = ci["upper"][:max_params].numpy()

    D = mu.shape[0]
    idx = np.arange(D)

    err_low = mu - lower
    err_hi  = upper - mu
    xerr = np.vstack([err_low, err_hi])

    plt.figure(figsize=(8, max(4, D * 0.1)))
    plt.errorbar(mu, idx, xerr=xerr, fmt=".", capsize=2)
    plt.axvline(0.0, linestyle="--", linewidth=1)
    plt.gca().invert_yaxis()
    plt.xlabel("θ value")
    plt.ylabel("parameter index (within group)")
    plt.title(f"θ credible intervals – {group_name}")
    plt.tight_layout()
    plt.show()
    

def plot_tau_nodes(tau_info, title="τ Hyperparameter Credible Intervals"):
    """
    tau_info: list of tuples or dicts:
        [
            ("τ4", mean, lower, upper),
            ("τ1", mean, lower, upper),
            ("τ2", mean, lower, upper),
        ]
    """

    names = [t[0] for t in tau_info]
    means = np.array([t[1] for t in tau_info], dtype=float)
    lows  = np.array([t[2] for t in tau_info], dtype=float)
    highs = np.array([t[3] for t in tau_info], dtype=float)

    # asymmetric y-error: [lower_error, upper_error]
    err_low = means - lows
    err_hi  = highs - means
    yerr = np.vstack([err_low, err_hi])   # shape (2, K)

    x = np.arange(len(names))

    plt.figure(figsize=(7, 4))
    plt.errorbar(
        x,
        means,
        yerr=yerr,
        fmt="o",
        capsize=4,
        color="black",
    )
    plt.xticks(x, names)
    plt.ylabel("τ value")
    plt.title(title)
    plt.tight_layout()
    plt.show()


def plot_theta_ci_block(ci_block: dict, title: str, max_params: int = 200):
    """
    ci_block: dict with keys 'mu', 'lower', 'upper'
    """
    mu    = ci_block["mu"][:max_params].numpy()
    lower = ci_block["lower"][:max_params].numpy()
    upper = ci_block["upper"][:max_params].numpy()

    D = mu.shape[0]
    idx = np.arange(D)

    err_low = mu - lower
    err_hi  = upper - mu
    xerr = np.vstack([err_low, err_hi])

    plt.figure(figsize=(8, max(3, D * 0.12)))
    plt.errorbar(mu, idx, xerr=xerr, fmt=".", capsize=2)
    plt.axvline(0.0, linestyle="--", linewidth=1)
    plt.gca().invert_yaxis()
    plt.xlabel("θ value")
    plt.ylabel("parameter index (within block)")
    plt.title(title)
    plt.tight_layout()
    plt.show()
