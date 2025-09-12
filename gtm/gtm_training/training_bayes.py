### BAYESIAN APPROACH
import torch
import math

from torch.distributions import Normal
from torch import nn, Tensor
from typing import TYPE_CHECKING

from gtm.gtm_layers.layer_utils import bayesian_splines

if TYPE_CHECKING:
    from ..gtm_model.gtm import GTM # type-only; no runtime import


class variational_inference(nn.Module):
    def __init__(
        self,
        mean_field: bool= True,
        start_mu: Tensor = None,
        start_sigma: Tensor = None,
        hyperparameter_transformation: dict[str, float] = None,
        hyperparameter_decorrelation: dict[str, float] = None
        
        ):
        super().__init__()
        
        
        self.hyperparameter_transformation: dict[str, float] = hyperparameter_transformation
        self.hyperparameter_decorrelation: dict[str, float]= hyperparameter_decorrelation
        self.mu: Tensor = start_mu
        self.sigma: Tensor = start_sigma
        self.mean_field: bool = mean_field
        self.distribution_for_aprox = Normal(self.mu, self.sigma)
    
    def kl_standard_normal(self, mu, std):
    # 0.5 * (mu^2 + std^2 - 1 - log std^2)
        return 0.5 * (mu.pow(2) + std.pow(2) - 1.0 - 2.0*torch.log(std + 1e-8))
    
    def elbo(self, x, y, mc_samples=1):
        # Reparameterized samples from q
        std_w = softplus(rho_w)
        std_b = softplus(rho_b)

        # Monte Carlo estimate of E_q[log p(y|x, w, b)]
        loglik = 0.0
        for _ in range(mc_samples):
            eps_w = torch.randn_like(mu_w)
            eps_b = torch.randn_like(mu_b)
            w = mu_w + std_w * eps_w
            b = mu_b + std_b * eps_b

            mean = x * w + b
            ll = self.distribution_for_aprox.log_prob(y).sum()  # sum over data
            loglik = loglik + ll

        loglik = loglik / mc_samples

        # KL terms (sum over dimensions; here each is scalar)
        kl = self.kl_standard_normal(mu_w, std_w) + self.kl_standard_normal(mu_b, std_b)
        kl = kl.sum()

        # ELBO
        return loglik - kl
    
        
    def forward(
        self,
        mu,
        sigma,
        ):
        
        
        elbo = self.elbo(x,y)
        
        return self.distribution_for_aprox(mu, sigma).sample(1)