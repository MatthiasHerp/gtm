### BAYESIAN APPROACH
import torch
import math

from torch.distributions import Normal
from torch import nn, Tensor
from typing import TYPE_CHECKING

from gtm.gtm_layers.layer_utils import bayesian_splines
from gtm.gtm_model import gtm

if TYPE_CHECKING:
    from ..gtm_model.gtm import GTM # type-only; no runtime import


def _flatten_state_dict(sd):
    """Flatten a (detached) state_dict with only tensor leaves into a single vector
    and keep a schema to reconstruct the dict later."""
    keys = []
    shapes = []
    flats = []
    for k, v in sd.items():
        if not torch.is_tensor(v):
            continue
        keys.append(k)
        shapes.append(v.shape)
        flats.append(v.reshape(-1))
    flat = torch.cat(flats) if flats else torch.empty(0)
    schema = list(zip(keys, shapes))
    return flat, schema


def _unflatten_to_state_dict(flat: Tensor, schema):
    """Inverse of _flatten_state_dict."""
    out = {}
    idx = 0
    for k, shape in schema:
        n = int(torch.tensor(shape).prod().item())
        out[k] = flat[idx:idx+n].reshape(shape)
        idx += n
    return out


class VI_Model(nn.Module):
    def __init__(
        self,
        model: "GTM",
        init_scale: float = 0.05,
        learn_scale: bool = True,
        device: torch.device | str = "cpu",
        ):
        
        super().__init__()
        
        self.model = model
        self.device = model.device
        
        
        # Snapshot an initial state dict to define θ's dimension and schema.
        with torch.no_grad():
            base_sd = {k: v.detach().to(self.device) for k, v in model.state_dict().items() if torch.is_tensor(v)}
            theta0, self._schema = _flatten_state_dict(base_sd)

        D = theta0.numel()

        # Variational parameters: μ and ρ with σ = softplus(ρ)
        self.mu = nn.Parameter(theta0.clone())
        self.rho = nn.Parameter(torch.full((D,), math.log(math.exp(init_scale) - 1.0)))
        self.learn_scale = learn_scale
        if not learn_scale:
            self.rho.requires_grad_(False)

        self._normal0 = torch.distributions.Normal(
            torch.zeros(D, device=self.device),
            torch.ones(D, device=self.device),
        )
    
    @property
    def sigma(self) -> Tensor:
        return nn.functional.softplus(self.rho)
        
        
    def sample_theta(self, num_samples: int = 1) -> Tensor:
        """Reparameterized samples θ = μ + σ ⊙ ε, ε ~ N(0, I). Shape: [S, D]."""
        eps = self._normal0.sample((num_samples,))
        return self.mu + self.sigma * eps

    def log_q(self, theta: Tensor) -> Tensor:
        """log q_phi(θ) under mean-field Normal. theta shape [S, D] or [D]. Returns [S]."""
        mu = self.mu
        sigma = self.sigma
        # compute per-sample log prob
        if theta.dim() == 1:
            theta = theta.unsqueeze(0)
        S, D = theta.shape
        # Normal log-density per dimension then sum
        log_det = torch.sum(torch.log(sigma))
        quad = 0.5 * torch.sum(((theta - mu) / sigma) ** 2, dim=1)
        const = 0.5 * D * math.log(2 * math.pi)
        return -(const + log_det + quad)

    def _theta_to_state_dict(self, theta_1d: Tensor):
        return _unflatten_to_state_dict(theta_1d, self._schema)

    @torch.no_grad()
    def set_model_params(self, theta_1d: Tensor):
        """Load θ back into the GTM model."""
        sd_new = self._theta_to_state_dict(theta_1d)
        # Use existing state_dict to preserve buffers that are not in θ
        full_sd = self.model.state_dict()
        for k, v in sd_new.items():
            full_sd[k] = v
        self.model.load_state_dict(full_sd, strict=False)

    def step(
        self,
        #samples: Tensor,
        #hyperparameter_transformation,
        #hyperparameter_decorrelation,
        #objective_fn,
        log_p_tilde_vals,
        #objective_type = "negloglik",
        mcmc_samples: int = 1,
        seed: int | None = None,
    ) :
        """
        One stochastic-ELBO step (no optimizer step).
        Returns dict with 'loss' and components for logging.
        """
        if seed is not None:
            torch.manual_seed(seed)

        # Sample θ ~ q
        thetas = self.sample_theta(mcmc_samples)  # [S, D]
        log_q_vals = self.log_q(thetas)         # [S]

        #log_p_tilde_vals = []  # log unnormalized posterior per sample

        for s in range(mcmc_samples):
            theta_s = thetas[s]
            # Push θ into model
            self.set_model_params(theta_s)

            # Use your provided objective to compute: posterior = NLL + priors
            """out = objective_fn(
                model=self.model,
                samples=samples,
                hyperparameter_transformation=hyperparameter_transformation,
                hyperparameter_decorrelation=hyperparameter_decorrelation,
                objective_type=objective_type,
                vi_model=self,            # optional: in case you want it
                sample_size=1,
                seed=seed if seed is not None else 11041998,
            )
            # Your function returns a POSITIVE objective (NLL + priors).
            # log \tilde p(θ, y) = - (NLL + priors)
            neglogpost = out["posterior"]
            log_p_tilde = -neglogpost
            log_p_tilde_vals.append(log_p_tilde.reshape(()))

        log_p_tilde_vals = torch.stack(log_p_tilde_vals)  # [S]
"""
        # Monte-Carlo KL(q || p) estimate: E_q[log q - log p̃]
        # (Note: additive constant log p(y) cancels in optimization)
        loss = torch.mean(log_q_vals - log_p_tilde_vals)

        return {
            "loss": loss,
            "mean_log_q": torch.mean(log_q_vals).detach(),
            "mean_log_p_tilde": torch.mean(log_p_tilde_vals).detach(),
            "sigma_mean": self.sigma.mean().detach(),
            "sigma_max": self.sigma.max().detach(),
            "sigma_min": self.sigma.min().detach(),
        }