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
        D = torch.eye(n_params)
        for _ in range(order):
            D = D[1:] - D[:-1]
        return D.T @ D # K = DᵀD
    
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
    def log_mvn_zero_mean_prec_ck(
        theta: torch.Tensor,   # [Kdim, M]  columns = independent coefficient blocks
        K: torch.Tensor,       # [Kdim, Kdim] RW2 precision base: D2ᵀD2 (psd)
        lam: torch.Tensor,     # scalar λ > 0 (precision)
        eps: float = 1e-6,
    ) -> torch.Tensor:
        """
        CK prior: θ ~ N(0, (λ K)^(-1)), with intrinsic RW2 K.
        Returns log p(Γ | λ, K) up to θ-independent constants (we keep the useful ones).
        """
        # symmetrize
        K = 0.5 * (K + K.T)
        Kdim = K.shape[0]

        # rank r for RW2 (null space: {1, x})
        with torch.no_grad():
            r = torch.linalg.matrix_rank(K)

        # stable alternative: log|K+εI| - (Kdim-r) log ε  (≈ pseudo-logdet)
        Kreg = K + eps * torch.eye(Kdim, device=K.device, dtype=K.dtype)
        L = torch.linalg.cholesky(Kreg)
        logdet_Kreg = 2.0 * torch.log(torch.diag(L)).sum()
        log_pdet_K = logdet_Kreg - (Kdim - r) * torch.log(torch.as_tensor(eps, device=K.device, dtype=K.dtype))

        # log |λK|_* = r log λ + log|K|_*   (pseudo-determinant)
        logdet_Q = r * torch.log(lam) + log_pdet_K

        # quadratic term = λ * Σ_m θ_mᵀ K θ_m  (nullspace not penalized)
        quad = -1*lam * theta.T @ lam @theta #torch.einsum('im,ij,jm->', theta, K, theta)   # scalar

        return 0.5 * (quad)


    @staticmethod
    def log_prior_gamma_ridge(
        theta: Tensor,
        K1: Tensor,
        K2: Tensor,
        kappa1: Tensor,
        kappa2: Tensor,
        sigma2: Tensor,
        eps: float = 1e-6
        ) -> Tensor:
        
        K1 = 0.5*(K1 + K1.T)
        K2 = 0.5*(K2 + K2.T)
        Kdim = K1.shape[0]
        
        I = torch.eye(Kdim, device=K1.device, dtype=K1.dtype)
        
        Q = kappa1*K1 + kappa2*K2 #+ (1.0/sigma2)*I
        
        L = torch.linalg.cholesky(Q + eps*I)
        
        logdetQ = 2.0 * torch.log(torch.diag(L)).sum()     # scalar
        
        Z = L @ theta                                          # [Kdim, M]
        quad = (Z * Z).sum()                                    # Σ_m γ_m^T Q γ_m

        M = theta.shape[1]
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
        def _gamma_mean(a, b):  # shape a, rate b
                return float(a) / float(b)
        
        sub_model = model.transformation if is_transformation else model.decorrelation_layers
        
        total_logp = torch.zeros((), device=model.device, dtype=torch.float32)
        
        if not is_transformation:
            
            # --- Decorrelation layer prior (RW1 + RW2) ---------------------------
            # Paper §3.3: ridge on first & second differences to pull toward linear
            # (Gaussian-copula baseline) and to smooth curvature. :contentReference[oaicite:0]{index=0}
            
            # Plug-in (stable) alpha^2; or make alpha2 a learnable parameter and add its log-prior below.
            
            # Penalization Term
            # controls shrinkage toward linear behavior
            pen_term1 = hyperparameter.get('RW1', {})
            a_kappa_1 = torch.as_tensor(pen_term1['tau_a'], device=model.device, dtype=torch.float32)
            b_kappa_1 = torch.as_tensor(pen_term1['tau_b'], device=model.device, dtype=torch.float32)
            
            # controls smoothness
            pen_term2 = hyperparameter.get('RW2', {})
            a_kappa_2 = torch.as_tensor(pen_term2['tau_a'], device=model.device, dtype=torch.float32)
            b_kappa_2 = torch.as_tensor(pen_term2['tau_b'], device=model.device, dtype=torch.float32)
            
            # Ridge General Variation
            a_sigma = torch.as_tensor(hyperparameter['sigma_a'], device=model.device, dtype=torch.float32)
            b_sigma = torch.as_tensor(hyperparameter['sigma_b'], device=model.device, dtype=torch.float32)
            
            kappa_hat_1   = torch.as_tensor(_gamma_mean(a_kappa_1, b_kappa_1),   device=model.device)
            kappa_hat_2   = torch.as_tensor(_gamma_mean(a_kappa_2, b_kappa_2),   device=model.device)
            
            sigma_hat = torch.as_tensor(_invgamma_mean(a_sigma, b_sigma),   device=model.device)
            
            for layer in model.decorrelation_layers:
                # K built once per layer; ensure it's on device
                K_RW1 = layer.priors.K_prior_RW1.to(device=model.device, dtype=torch.float32)
                K_RW2 = layer.priors.K_prior_RW2.to(device=model.device, dtype=torch.float32)
                
                theta_D = layer.params  # [Kdim, M]
                
                # log_prior_gamma_ridge is assumed to implement: -0.5/sigma2 * sum_c theta_c^T K theta_c (+ const)
                total_logp = total_logp + bayesian_splines.log_prior_gamma_ridge(
                    theta_D,
                    K_RW1,
                    K_RW2,
                    kappa_hat_1,
                    kappa_hat_2,
                    sigma_hat
                    )
            
            
        else:
            # --- Transformation (marginal) layer prior (RW2 only) ----------------
            # Paper §3.3: monotone splines; first-diff penalty not needed; use RW2-only
            # smoothing (λ4 in the paper). :contentReference[oaicite:1]{index=1}
            
            # Penalization Term
            pen_term2 = hyperparameter.get('RW2', {})
            a_lambda = torch.as_tensor(pen_term2['tau_a'], device=model.device, dtype=torch.float32)
            b_lambda = torch.as_tensor(pen_term2['tau_b'], device=model.device, dtype=torch.float32)
            
            lambda_hat   = torch.distributions.Uniform(0,100000).sample()##torch.as_tensor(_gamma_mean(a_lambda, b_lambda),   device=model.device)
            
            K_Prior_RW2 = sub_model.priors.K_prior_RW2.to(device=sub_model.device, dtype=torch.float32)
            
            K_Prior_RW2 = 0.5*(K_Prior_RW2+ K_Prior_RW2.T)
            
            # Use the *restricted* (monotone) coefficients θ for the P-spline prior (paper §2.1). :contentReference[oaicite:2]{index=2}
            #varphi = sub_model.padded_params
            
            #theta_T, logJ = bayesian_splines._restrict_parameters_(
            #    params_a=varphi,
            #    monotonically_increasing=sub_model.monotonically_increasing,
            #    #use_softplus=False,
            #    device=sub_model.device
            #)
            
            theta_T = torch.vstack([
                torch.nn.functional.pad(p, (0, K_Prior_RW2.shape[0] - p.numel()))
                for p in sub_model.padded_params]).T
            
            
            
            total_logp = (
                bayesian_splines.log_mvn_zero_mean_prec_ck(theta_T, K_Prior_RW2, lambda_hat)
                + logJ.sum()
                )
            
        # NEGATIVE log prior to add onto NLL in your objective    
        return -1*total_logp
    
    @staticmethod
    def _restrict_parameters_(
        params_a: Tensor,
        monotonically_increasing: bool,
        use_softplus: bool = True,
        device: torch.device=None
    ):
        
        if not monotonically_increasing:
            return params_a.clone()

        #Prepropressing
        a =params_a.T
        
        ## Jaccobian
        tail_pre= a[: ,1:]
        logJ = torch.nn.functional.logsigmoid(tail_pre).sum(dim=1) if use_softplus else tail_pre.sum(dim=1)# sum log σ(a_j), j>=2

        # Param Restriction
        params_restricted: Tensor = a.clone()  # [B, K]
        B, K = params_restricted.shape
        params_restricted[:, 1:] = softplus(params_restricted[:, 1:]) if use_softplus else torch.exp(params_restricted[:, 1:]) 
        #params_restricted[:, 1:] = torch.exp(params_restricted[:, 1:])
        
        # Create upper triangular summing matrix: [K, K]
        sum_matrix: Tensor = torch.triu(input=torch.ones(K, K, device=device))  # [K, K]
        
        # Apply cumulative sum: [B, K] x [K, K]ᵗ = [B, K]
        params_restricted: Tensor = torch.matmul(input=params_restricted, other=sum_matrix)

        
        return params_restricted.T, logJ
    
@dataclass(frozen=True)
class BayesianPriors:
    sigma_a: Tensor
    sigma_b: Tensor
    RW1_alpha_a: Tensor
    RW1_alpha_b: Tensor
    RW2_alpha_a: Tensor
    RW2_alpha_b: Tensor
    K_prior_RW1: Tensor
    K_prior_RW2: Tensor
    #hyperprior_sigma_dist: InverseGamma
    #hyperprior_alpha_dist: InverseGamma


class BayesianInitializer:
    """Builds Bayesian prior tensors/distributions for a model instance."""
    @staticmethod
    def build(
        model,
        hyperparameter: Mapping[str, float],
        n_params: int,
        is_transformation,
        ) -> BayesianPriors:
        # Guard rails
        if getattr(model, "spline", None) == "bernstein":
            raise NotImplementedError(
                "Bayesian Inference is not implemented for Bernstein polynomials"
            )
            
        #if missing := [k for k in BayesianInitializer.REQUIRED_KEYS if k not in hyperparameter]:
        #    raise KeyError(f"Missing hyperparameter(s): {', '.join(missing)}")
        
        device: torch.device = model.device
        dtype: torch.dtype = torch.float32

        # Scalars as 0-dim tensors on the right device/dtype
        sigma_a: Tensor = torch.as_tensor(hyperparameter["sigma_a"], dtype=dtype, device=device)
        sigma_b: Tensor = torch.as_tensor(hyperparameter["sigma_b"], dtype=dtype, device=device)
        
        # Penalization Term
        pen_term1 = hyperparameter['RW1']
        a_tau_1 = torch.as_tensor(pen_term1['tau_a'], device=model.device, dtype=torch.float32)
        b_tau_1 = torch.as_tensor(pen_term1['tau_b'], device=model.device, dtype=torch.float32)
        
        pen_term2 = hyperparameter['RW2']
        a_tau_2 = torch.as_tensor(pen_term2['tau_a'], device=model.device, dtype=torch.float32)
        b_tau_2 = torch.as_tensor(pen_term2['tau_b'], device=model.device, dtype=torch.float32)
        
        #order_prior_diff = 1
        K_prior_RW1: Tensor = bayesian_splines.difference_penalty_matrix(order=1, n_params=n_params).to(device=device)
        
        #order_prior_diff = 2
        #n_params = int(model.padded_params.shape[0])
        K_prior_RW2: Tensor = bayesian_splines.difference_penalty_matrix(order=2, n_params=n_params).to(device=device)
        
        # normalize once so τ² has a consistent meaning
        K_prior_RW1 = K_prior_RW1 / (K_prior_RW1.abs().max() + 1e-12)
        K_prior_RW2 = K_prior_RW2 / (K_prior_RW2.abs().max() + 1e-12)
        # Priors (note: torch.distributions.InverseGamma expects concentration/rate)
        #hyperprior_sigma_dist: InverseGamma = bayesian_splines.gamma_hyperprior_distribution(a=sigma_a, b=sigma_b)
        #hyperprior_alpha_dist: InverseGamma = bayesian_splines.gamma_hyperprior_distribution(a=alpha_a, b=alpha_b)
        
        
        return BayesianPriors(
            sigma_a=sigma_a,
            sigma_b=sigma_b,
            RW1_alpha_a=a_tau_1,
            RW1_alpha_b=b_tau_1,
            RW2_alpha_a=a_tau_2,
            RW2_alpha_b=b_tau_2,
            K_prior_RW1=K_prior_RW1,
            K_prior_RW2=K_prior_RW2,
        )
    