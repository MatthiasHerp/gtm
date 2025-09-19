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

    
    
    @staticmethod
    def defining_prior(
        model: "GTM",
        hyperparameter,
        gammas= None,
        is_init: bool= False,
        is_transformation = False
        ):
        
        
        sub_model = model.transformation if is_transformation else model.decorrelation_layers
        
        
        if not is_transformation:
            samples_deco_sigma = [
                layer.priors.hyperprior_sigma_dist  # distribution objects
                for layer in sub_model #model.decorrelation_layers
                ]
            
            samples_deco_alpha = [
                layer.priors.hyperprior_alpha_dist  # distribution objects
                for layer in sub_model#model.decorrelation_layers
                ]
            
            sigma2_deco_samples: list[Tensor]= [d.sample(()) for d in samples_deco_sigma] 
            alpha2_deco_samples: list[Tensor] = [d.sample(()) for d in samples_deco_alpha]     
            
            # stack per-sample results into a single tensor
            sigmas_prior = torch.stack([
            bayesian_splines.log_gamma_prior(
                hyperparameter['sigma_a'],
                hyperparameter['sigma_b'],
                s
            )
            for s in sigma2_deco_samples
            ])
            
            log_prior_sigma: Tensor = -1 * sigmas_prior.sum()
            
            alphas_prior = torch.stack([
            bayesian_splines.log_gamma_prior(
                hyperparameter['tau_a'],
                hyperparameter['tau_b'],
                s
            )
            for s in alpha2_deco_samples
            ])
            
            log_prior_alpha: Tensor = -1 * alphas_prior.sum()
            
            Ks: list[Tensor] = [layer.priors.K_prior for layer in model.decorrelation_layers] 
            
            if is_init:
                gammas: list[Tensor] = [layer.params for layer in model.decorrelation_layers]
                
            total_log_gamma = torch.zeros((), device=model.device, dtype=torch.float32)
            
            for K, alpha_prior, gamma in zip(Ks, alphas_prior, gammas):
                pieces: list = []
                M = gamma.shape[1]
                for m in range(M):
                    gamma_m = gamma[:, m].unsqueeze(0)  # shape (1, K) if your function expects batch
                    # log N(0, (alpha_prior^{-1} Q^{-1}))  == log MVN with precision (alpha_prior * Q)
                    logp = bayesian_splines.log_normal_distr(
                        K=K,
                        alpha_2=alpha_prior,   # see note below about meaning
                        gamma=gamma_m          # the K-vector for this edge
                        # optionally pass Q or its Cholesky if your function takes it
                        )
                    pieces.append(logp)
                
                total_log_gamma: Tensor = total_log_gamma + torch.stack(pieces).sum()
            
            
            decorrelation_prior: Tensor = total_log_gamma + log_prior_sigma + log_prior_alpha
            
            
            return -1*decorrelation_prior

        else: 
            
            sigma_samples: Tensor = sub_model.priors.hyperprior_sigma_dist.sample(())
            sigmas_prior: Tensor = bayesian_splines.log_gamma_prior(a=hyperparameter['sigma_a'], b=hyperparameter['sigma_b'], x=sigma_samples)
            
            
            
            alpha_samples: Tensor = sub_model.priors.hyperprior_alpha_dist.sample(())
            alpha_prior: Tensor = bayesian_splines.log_gamma_prior(a=hyperparameter['tau_a'], b=hyperparameter['tau_b'],x=alpha_samples)
            
            
            K: Tensor = sub_model.priors.K_prior
            
            if is_init:
                gammas = sub_model.padded_params
            
            gammas = bayesian_splines._restrict_parameters_(
                    params_a= gammas,
                    covariate=False,
                    degree=sub_model.max_degree,
                    monotonically_increasing=sub_model.monotonically_increasing,
                    device=sub_model.device
                )
            
            pieces = []
            M = gammas.shape[1]
            
            for m in range(M):
                
                gamma_m = gammas[:, m].unsqueeze(0)
                logp: Tensor = bayesian_splines.log_normal_distr(
                        K=K,
                        alpha_2=alpha_prior,   # see note below about meaning
                        gamma=gamma_m          # the K-vector for this edge
                        # optionally pass Q or its Cholesky if your function takes it
                        )
                pieces.append(logp)
                
            transformation_prior= torch.stack(pieces).sum()
            
            return -1*transformation_prior
    
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
    