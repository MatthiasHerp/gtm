### BAYESIAN APPROACH
import torch
import math
from torch.distributions import Normal
from torch import nn, Tensor
from typing import TYPE_CHECKING

from gtm.gtm_layers.layer_utils import bayesian_splines
from torch.nn.utils.stateless import _reparametrize_module
from gtm.gtm_model import gtm

if TYPE_CHECKING:
    from ..gtm_model.gtm import GTM # type-only; no runtime import


def _flatten_state_dict(sd, key_filter= None):
    """Flatten a (detached) state_dict with only tensor leaves into a single vector
    and keep a schema to reconstruct the dict later. Optionally filter by key."""
    keys = []
    shapes = []
    flats = []
    for k, v in sd.items():
        if not torch.is_tensor(v):
            continue
        if key_filter is not None and not key_filter(k):
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
        key_filter= None,
        device: torch.device | str = "cpu",
        ):
        
        super().__init__()
        
        self.model = model
        self.device = model.device
        
        
        # Snapshot an initial state dict to define θ's dimension and schema.
        with torch.no_grad():
            base_sd = {k: v.detach().to(self.device) for k, v in model.state_dict().items() if torch.is_tensor(v)}
            theta0, self._schema = _flatten_state_dict(base_sd, key_filter=key_filter)

        D = theta0.numel()
        
        if D == 0:
            raise RuntimeError("Key filter selected zero parameters; check your include/exclude patterns.")
                
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
        return 1e-6 + nn.functional.softplus(self.rho) #to avoid softplus near-zero stickiness 1e-6
        
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
        samples: Tensor,
        hyperparameter_transformation,
        hyperparameter_decorrelation,
        model: "GTM",
        sample_size,
        mcmc_samples: int = 100,
        seed: int | None = None,
    ):
        """
        One stochastic-ELBO step (no optimizer step).
        Returns dict with 'loss' and components for logging.
        """
        if seed is not None:
            torch.manual_seed(seed)

        # Sample θ ~ q
        thetas = self.sample_theta(mcmc_samples)  # [S, D]
        log_q_vals = self.log_q(thetas)         # [S]

        log_p_tilde_vals = []  # log unnormalized posterior per sample
        neg_likelihood_list = []
        prior_dec_list = []
        prior_trans_list = []
        qf_neg_prior_list = []
        qf_sum_list=[]
        neg_log_post_list=[]
        
        for s in range(mcmc_samples):
            theta_s = thetas[s]
            params_s = self._theta_to_state_dict(theta_s)  # tensors keep graph to (mu, rho)

            with _reparametrize_module(self.model, params_s):
            # Use your provided objective to compute: posterior = NLL + priors
                out = model.__bayesian_training_objective__(
                    samples=samples,
                    hyperparameters_transformation=hyperparameter_transformation,
                    hyperparameters_decorrelation=hyperparameter_decorrelation,
                    sample_size=sample_size
                )
            
            # Your function returns a POSITIVE objective (NLL + priors).
            # log \tilde p(θ, y) = - (NLL + priors)
            neglogpost = out['neg_posterior']
            log_p_tilde = -neglogpost
            log_p_tilde_vals.append(log_p_tilde.reshape(()))
            
            #Tracking
            neg_log_post_list.append(neglogpost.reshape(()))
            neg_likelihood_list.append(out['negative_log_lik'].reshape(()))
            prior_dec_list.append(out['negative_decorrelation_prior'].reshape(()))
            prior_trans_list.append(out['negative_transformation_prior']['neg_log_prior_total'].reshape(()))
            qf_neg_prior_list.append(out['negative_transformation_prior']['neg_log_prior_qf'].reshape(()))
            qf_sum_list.append(out["negative_transformation_prior"]["qf_sum"].reshape(()))
            

        log_p_tilde_vals = torch.stack(log_p_tilde_vals)  # [S]
        # Monte-Carlo KL(q || p) estimate: E_q[log q - log p̃]
        # (Note: additive constant log p(y) cancels in optimization)
        elbo_loss = torch.mean(log_q_vals - log_p_tilde_vals) #-ELBO (i.e., E_q[log q - log p̃])

        neg_log_post_list=torch.stack(neg_log_post_list)
        neg_likelihood_list= torch.stack(neg_likelihood_list)
        prior_dec_list = torch.stack(prior_dec_list)
        prior_trans_list = torch.stack(prior_trans_list)
        qf_neg_prior_list = torch.stack(qf_neg_prior_list)
        qf_sum_list= torch.stack(qf_sum_list)
        
        return {
            "loss": elbo_loss,
            "neg_log_posterior": torch.mean(neg_log_post_list).detach(),
            "mean_log_q": torch.mean(log_q_vals).detach(),
            "mean_log_p_tilde": torch.mean(log_p_tilde_vals).detach(),
            "sigma_mean": self.sigma.mean().detach(),
            "sigma_max": self.sigma.max().detach(),
            "sigma_min": self.sigma.min().detach(),
            "neg_log_likelihood": torch.mean(neg_likelihood_list).detach(),
            "neg_prior_decorrelation": torch.mean(prior_dec_list).detach(),
            "neg_prior_transformation": torch.mean(prior_trans_list).detach(),
            "transformation_neg_log_prior_df": torch.mean(qf_neg_prior_list).detach(),  #= E[0.5 τ qf]
            "transformation_sum_qf": torch.mean(qf_sum_list).detach()#qf
        }