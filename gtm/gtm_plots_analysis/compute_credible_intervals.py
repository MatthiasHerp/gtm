import torch
from typing import Dict


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
    
    
def shrinkage_groups(self):
    """
    Return indices for transformation vs decorrelation parameters
    based on the schema keys used when flattening.
    """
    trans_idx = []
    decor_idx = []

    offset = 0
    for key, shape in self._schema:
        n = int(torch.tensor(shape).prod().item())
        idx = torch.arange(offset, offset + n, device=self.device)
        if key.startswith("transformation."):
            trans_idx.append(idx)
        elif key.startswith("decorrelation_layers."):
            decor_idx.append(idx)
        offset += n
    ...
    return trans_idx, decor_idx


def theta_ci_by_group(
    vi_model: "VI_Model",
    level: float = 0.95,
    sort_within_group: bool = False,
) -> Dict[str, dict]:
    """
    Compute marginal CIs for θ and split them into
    - 'transformation'
    - 'decorrelation'

    Returns dict:
      {
        "transformation": {"mu": ..., "lower": ..., "upper": ...} or None,
        "decorrelation":  {"mu": ..., "lower": ..., "upper": ...} or None,
      }
    """
    mu = vi_model.mu.detach()
    sigma = vi_model.sigma.detach()

    alpha = 1.0 - level
    z = torch.distributions.Normal(0.0, 1.0).icdf(
        torch.tensor(1.0 - alpha / 2.0, device=mu.device)
    )

    lower = mu - z * sigma
    upper = mu + z * sigma

    trans_idx, decor_idx = vi_model.shrinkage_groups()

    def _subset(idx: torch.Tensor):
        if idx is None or idx.numel() == 0:
            return None
        mu_g    = mu[idx]
        lower_g = lower[idx]
        upper_g = upper[idx]

        if sort_within_group:
            order = torch.argsort(mu_g.abs(), descending=True)
            mu_g    = mu_g[order]
            lower_g = lower_g[order]
            upper_g = upper_g[order]

        return {
            "mu":    mu_g.cpu(),
            "lower": lower_g.cpu(),
            "upper": upper_g.cpu(),
        }

    return {
        "transformation": _subset(trans_idx),
        "decorrelation":  _subset(decor_idx),
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

