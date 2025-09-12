from dataclasses import dataclass
from typing import Mapping

import torch
import numpy as np

from torch.distributions import InverseGamma, Laplace
from torch import Tensor
from torch.linalg import matrix_rank

def generate_diagonal_matrix(n):
    matrix = torch.eye(n, dtype=torch.float32)  # Create an identity matrix of size n
    return torch.flip(matrix, dims=[0])  # Flip the matrix along the vertical axis

class bayesian_splines:
    @staticmethod
    def difference_penalty_matrix(order, n_params) -> Tensor:
        D = np.eye(n_params)
        for _ in range(order):
            D = np.diff(D, n=1, axis=0)
        return torch.tensor(D.T @ D, dtype=torch.float32) # K = DᵀD
    
    @staticmethod
    def gamma_hyperprior_distribution(a,b) -> InverseGamma:
        return InverseGamma(concentration=a, rate=b)
    
    @staticmethod
    def laplace_prior_distribution(a,b) -> Laplace:
        return Laplace(loc=a, scale=b)
    
    @staticmethod
    def log_gamma_prior(a: Tensor,b: Tensor, x:Tensor) -> Tensor:
        return -(a + 1) * torch.log(x) - b / x
    
    @staticmethod
    def log_normal_distr(K: Tensor, alpha_2: Tensor, gamma: Tensor) -> Tensor:
        
        if K.shape[0] != gamma.shape[1]:
            raise KeyError('Dimensions does not coincide')
            
            
        r: int = matrix_rank(K).item()
        cov: Tensor = (gamma.T @ K @ gamma) / (2*alpha_2)
        return -0.5 * r * torch.log(alpha_2) - cov

@dataclass(frozen=True)
class BayesianPriors:
    sigma_a: Tensor
    sigma_b: Tensor
    alpha_a: Tensor
    alpha_b: Tensor
    order_prior_diff: int
    K_prior: Tensor
    hyperprior_sigma_dist: InverseGamma
    hyperprior_alpha_dist: InverseGamma
    prior_gamma_given_tau : callable


class BayesianInitializer:
    """Builds Bayesian prior tensors/distributions for a model instance."""

    REQUIRED_KEYS = ("sigma_a", "sigma_b", "tau_a", "tau_b")
    
    @staticmethod
    def build(
        model,
        hyperparameter: Mapping[str, float],
        n_params: int
        ) -> BayesianPriors:
        # Guard rails
        if getattr(model, "spline", None) == "bernstein":
            raise NotImplementedError(
                "Bayesian Inference is not implemented for Bernstein polynomials"
            )
            
        if missing := [k for k in BayesianInitializer.REQUIRED_KEYS if k not in hyperparameter]:
            raise KeyError(f"Missing hyperparameter(s): {', '.join(missing)}")
        
        device: torch.device = model.device
        dtype: torch.dtype = torch.float32

        # Scalars as 0-dim tensors on the right device/dtype
        sigma_a: Tensor = torch.as_tensor(hyperparameter["sigma_a"], dtype=dtype, device=device)
        sigma_b: Tensor = torch.as_tensor(hyperparameter["sigma_b"], dtype=dtype, device=device)
        alpha_a: Tensor = torch.as_tensor(hyperparameter["tau_a"],   dtype=dtype, device=device)
        alpha_b: Tensor = torch.as_tensor(hyperparameter["tau_b"],   dtype=dtype, device=device)

        # Difference penalty matrix K = DᵀD
        order_prior_diff = 2
        #n_params = int(model.padded_params.shape[0])
        K_prior: Tensor = bayesian_splines.difference_penalty_matrix(order=order_prior_diff, n_params=n_params).to(device=device)

        # Priors (note: torch.distributions.InverseGamma expects concentration/rate)
        hyperprior_sigma_dist: InverseGamma = bayesian_splines.gamma_hyperprior_distribution(a=sigma_a, b=sigma_b)
        hyperprior_alpha_dist: InverseGamma = bayesian_splines.gamma_hyperprior_distribution(a=alpha_a, b=alpha_b)
        
        
        log_norm_distr_for_gamma_given_tau = bayesian_splines.log_normal_distr
        
        
        return BayesianPriors(
            sigma_a=sigma_a,
            sigma_b=sigma_b,
            alpha_a=alpha_a,
            alpha_b=alpha_b,
            order_prior_diff=order_prior_diff,
            K_prior=K_prior,
            hyperprior_sigma_dist=hyperprior_sigma_dist,
            hyperprior_alpha_dist=hyperprior_alpha_dist,
            prior_gamma_given_tau= log_norm_distr_for_gamma_given_tau
        )
    