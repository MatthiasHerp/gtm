### BAYESIAN APPROACH
import torch
import math

from torch.distributions import Normal


from ..gtm_model.gtm import GTM


class variational_inference:
    def __init__(self):
        pass
    
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
            ll = Normal(mean, sigma_obs).log_prob(y).sum()  # sum over data
            loglik = loglik + ll

        loglik = loglik / mc_samples

        # KL terms (sum over dimensions; here each is scalar)
        kl = kl_standard_normal(mu_w, std_w) + kl_standard_normal(mu_b, std_b)
        kl = kl.sum()

        # ELBO
        return loglik - kl