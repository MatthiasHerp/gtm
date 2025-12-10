import torch
import math

from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from gtm_training.training_bayes.variational_model_estimator import GammaTauNode, VI_Model



def theta_credible_intervals(vi_model: "VI_Model", level=0.95, sort_by_magnitude=False):
    """
    Compute marginal credible intervals for θ under a mean-field Gaussian VI model.

    Returns:
        mu       : [D]
        lower_ci : [D]
        upper_ci : [D]
        order    : permutation index (if sorting is requested)
    """
    mu = vi_model.mu.detach()
    sigma = vi_model.sigma.detach()

    # Standard normal quantile
    alpha = 1.0 - level
    z = torch.distributions.Normal(0.0, 1.0).icdf(
        torch.tensor(1.0 - alpha / 2.0, device=mu.device)
    )

    lower = mu - z * sigma
    upper = mu + z * sigma

    if sort_by_magnitude:
        # Sort by |μ| or |μ|/σ → both are common
        order = torch.argsort(mu.abs(), descending=True)
        return {
            "mu": mu[order], 
            "lower": lower[order],
            "upper": upper[order],
            "order":order
            }

    return {
        "mu": mu,
        "lower": lower,
        "upper": upper,
        "order": None
        }



def tau_ci_from_node(node: "GammaTauNode", level=0.95, n_mc: int = 50_000):
    """
    Compute credible interval for a Gamma(a_hat, b_hat) variational distribution.

    Returns:
        mean, lower, upper (Python floats)
    """
    a_hat = node.a_hat.detach()
    b_hat = node.b_hat.detach()

    tau_samples, _, _ = node.sample_tau(n_mc)  # [n_mc]
    tau_samples = tau_samples.detach()
    
    
    alpha = 1.0 - level
    qs = torch.quantile(
        tau_samples,
        torch.tensor([alpha / 2.0, 1.0 - alpha / 2.0], device=tau_samples.device),
        dim=0,
    )
    lo, hi = qs[0], qs[1]

    return {
        "mean": float(a_hat / b_hat),
        "lower": float(lo),
        "upper": float(hi)
        }

