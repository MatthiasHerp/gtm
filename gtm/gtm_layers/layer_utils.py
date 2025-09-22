from dataclasses import dataclass
from typing import Mapping, TYPE_CHECKING

import torch
import numpy as np

from torch.distributions import InverseGamma, Laplace
from torch import Tensor
from torch.nn.functional import softplus
from torch.linalg import matrix_rank


if TYPE_CHECKING:
    from gtm_model.gtm import GTM 

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
        
        r: int = matrix_rank(K).item()
        cov: Tensor = (gamma @ K @ gamma.T) / (2*alpha_2) #TODO CHECK DIMENSIONS
        
        return -0.5 * r * torch.log(alpha_2) - cov
    
    @staticmethod
    def log_mvn_zero_mean_prec(K: Tensor, alpha2: Tensor, gamma: Tensor, eps: float = 1e-6) -> Tensor:
        """
        log N(gamma | 0, (alpha2*K)^(-1)) with precision alpha2 * (K + eps I).
        """
        if gamma.dim() == 1:
            gamma = gamma.unsqueeze(0)  # [1, Kdim]
        Kdim = K.shape[0]
        Kreg = K + eps * torch.eye(Kdim, device=K.device, dtype=K.dtype)
        # quadratic term γ^T K γ
        quad = torch.einsum('bi,ij,bj->b', gamma, Kreg, gamma)
        # log|K| term (constant wrt gamma, but helps calibration)
        logdet_K = torch.logdet(Kreg)
        r = Kdim
        return 0.5 * r * torch.log(alpha2) + 0.5 * logdet_K - 0.5 * alpha2 * quad #shape [B]

    # ---------- NEW Utilities ----------
    @staticmethod
    def invgamma_logpdf(x: Tensor, a: Tensor, b: Tensor, include_const: bool = False) -> Tensor:
        """
        log p(x | a,b) for InverseGamma(a,b) with rate parametrization.
        include_const=False drops a*log b - lgamma(a), which does not affect gradients wrt x or θ.
        """
        logp = -(a + 1.0) * torch.log(x) - b / x
        if include_const:
            logp = logp + (a * torch.log(b) - torch.lgamma(a))
        return logp

    @staticmethod
    def log_prior_gamma_ridge(gamma: Tensor, K: Tensor, tau2: Tensor, sigma2: Tensor, eps: float = 1e-6) -> Tensor:
        """
        gamma: [Kdim, M]  (columns independent)
        K:     [Kdim, Kdim] (difference penalty base)
        tau2, sigma2: scalar Tensors (>0)
        returns scalar log p(Γ | tau2, sigma2, K) (without constants independent of Γ)
        """
        Kdim = K.shape[0]
        I = torch.eye(Kdim, device=K.device, dtype=K.dtype)
        #Q = tau2 * (K + eps * I) + (1.0 / sigma2) * I           # [Kdim,Kdim] PD
        K = K / (K.abs().max() + 1e-8)
        Q = (1.0 / tau2) * K + (1.0 / sigma2) * I
        
        #L = torch.linalg.cholesky(Q)                             # stable logdet + quad
        L = torch.linalg.cholesky(Q + eps * I) 
        logdetQ = 2.0 * torch.sum(torch.log(torch.diag(L)))     # scalar

        Z = L @ gamma                                           # [Kdim, M]
        quad = (Z * Z).sum()                                    # Σ_m γ_m^T Q γ_m

        M = gamma.shape[1]
        return 0.5 * (M * logdetQ - quad) 
    
    
    @staticmethod
    def defining_prior(
        model: "GTM",
        hyperparameter,
        is_transformation = False
        ):
        
        def _invgamma_mean(a, b):
            # mean exists if a>1; fallback otherwise
            a = float(a)
            return b / (a - 1.0) if a > 1.0 else b / (a + 1.0)
        
        sub_model = model.transformation if is_transformation else model.decorrelation_layers
        total_logp = torch.zeros((), device=model.device, dtype=torch.float32)
        
        if not is_transformation:
            
            """
            Returns NEGATIVE log prior (scalar) to be added to NLL.
            Sums over all decorrelation layers and columns.
            """
            # Plug-in (stable) alpha^2; or make alpha2 a learnable parameter and add its log-prior below.
            a_tau = torch.as_tensor(hyperparameter['tau_a'], device=model.device, dtype=torch.float32)
            b_tau = torch.as_tensor(hyperparameter['tau_b'], device=model.device, dtype=torch.float32)
            
            a_sigma = torch.as_tensor(hyperparameter['sigma_a'], device=model.device, dtype=torch.float32)
            b_sigma = torch.as_tensor(hyperparameter['sigma_a'], device=model.device, dtype=torch.float32)
            
            
            alpha2_hat   = torch.as_tensor(_invgamma_mean(a_tau, b_tau),   device=model.device)
            sigma_2_hat = torch.as_tensor(_invgamma_mean(a_sigma, b_sigma),   device=model.device)

            
            for layer in model.decorrelation_layers:
                # K built once per layer; ensure it's on device
                K = layer.priors.K_prior.to(device=model.device, dtype=torch.float32)
                gamma = layer.params  # [Kdim, M]
                
                total_logp = total_logp + bayesian_splines.log_prior_gamma_ridge(gamma, K, alpha2_hat, sigma_2_hat)

            # NEGATIVE log prior to add onto NLL in your objective
            
        else: 
            a_tau = torch.as_tensor(hyperparameter['tau_a'],  device=sub_model.device, dtype=torch.float32)
            b_tau = torch.as_tensor(hyperparameter['tau_b'],  device=sub_model.device, dtype=torch.float32)
            a_sig = torch.as_tensor(hyperparameter['sigma_a'],device=sub_model.device, dtype=torch.float32)
            b_sig = torch.as_tensor(hyperparameter['sigma_b'],device=sub_model.device, dtype=torch.float32)

            tau2_hat   = torch.as_tensor(_invgamma_mean(a_tau, b_tau),   device=sub_model.device)
            sigma2_hat = torch.as_tensor(_invgamma_mean(a_sig, b_sig),   device=sub_model.device)

            K = sub_model.priors.K_prior.to(device=sub_model.device, dtype=torch.float32)
            gammas = sub_model.padded_params
            gammas = bayesian_splines._restrict_parameters_(
                params_a=gammas,
                covariate=False,
                degree=sub_model.max_degree,
                monotonically_increasing=sub_model.monotonically_increasing,
                device=sub_model.device
            )
            total_logp = bayesian_splines.log_prior_gamma_ridge(gammas, K, tau2_hat, sigma2_hat)
            
        return -total_logp
            
    @staticmethod
    def _restrict_parameters_(
        params_a: Tensor,
        covariate: bool,
        degree: int,
        monotonically_increasing: bool,
        device: torch.device=None
    ):
        
        if not monotonically_increasing:
            return params_a.clone()

        params_restricted: Tensor = params_a.clone().T  # [B, K]
        B, K = params_restricted.shape

        params_restricted[:, 1:] = softplus(params_restricted[:, 1:])
        
        # Create upper triangular summing matrix: [K, K]
        sum_matrix: Tensor = torch.triu(input=torch.ones(K, K, device=device))  # [K, K]
        
        # Apply cumulative sum: [B, K] x [K, K]ᵗ = [B, K]
        params_restricted: Tensor = torch.matmul(input=params_restricted, other=sum_matrix)

        return params_restricted.T
    
    
    

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
        
        
        
        return BayesianPriors(
            sigma_a=sigma_a,
            sigma_b=sigma_b,
            alpha_a=alpha_a,
            alpha_b=alpha_b,
            order_prior_diff=order_prior_diff,
            K_prior=K_prior,
            hyperprior_sigma_dist=hyperprior_sigma_dist,
            hyperprior_alpha_dist=hyperprior_alpha_dist
        )
    