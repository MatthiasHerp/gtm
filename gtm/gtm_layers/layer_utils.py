from dataclasses import dataclass
from typing import Mapping, TYPE_CHECKING

import torch
import re

from torch.distributions import InverseGamma, Laplace
from torch import Tensor
from torch.nn.functional import softplus


if TYPE_CHECKING:
    from gtm_model.gtm import GTM 

_TRANSFORM_PARAM_RE = re.compile(r"^transformation\.params\.(\d+)$")

def generate_diagonal_matrix(n):
    matrix = torch.eye(n, dtype=torch.float32)  # Create an identity matrix of size n
    return torch.flip(matrix, dims=[0])  # Flip the matrix along the vertical axis

def get_transformation_param_matrix_live(model):
    name_to_param = dict(model.named_parameters())
    cols = []
    for i in range(9999):  # or discover i’s via a regex on names
        key = f"transformation.params.{i}"
        if key not in name_to_param: break
        cols.append(name_to_param[key].reshape(-1))   # 1-D view is fine
    if not cols:
        raise RuntimeError("No transformation.params.* found.")
    varphi = torch.stack(cols, dim=1).contiguous()    # [K, M], contiguous
    M = varphi.shape[1]
    return varphi, M

class bayesian_splines:
    @staticmethod
    def difference_penalty_matrix(order, n_params, device=None, dtype=None) -> Tensor:
        D = torch.eye(n_params, device=device, dtype=dtype)
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
        gamma: torch.Tensor,   # [Kdim, M]  columns = independent coefficient blocks
        K: torch.Tensor,       # [Kdim, Kdim] RW2 precision base: D2ᵀD2 (psd)
        tau2: torch.Tensor,     # scalar λ > 0 (precision)
        eps: float = 1e-6,
    ) -> torch.Tensor:
        
        # symmetrize
        K = 0.5 * (K + K.T)
        gamma = gamma.T.contiguous()
        dgamma = gamma[:, 1:] - gamma[:, :-1]
        safe_dgamma = (dgamma + eps).clamp_min(1e-8)
        
        # β(γ): β1=γ1; βk=log(Δγ_k), k>=2
        beta1 = gamma[:, :1].contiguous()                        # [..., 1]
        beta_tail = torch.log(safe_dgamma)                       # [..., D-1]
        beta = torch.cat([beta1, beta_tail], dim=-1).contiguous()# [..., D]
        
        # Eigenvalue Decomposition
        evals, evecs = torch.linalg.eigh(K)
        mask = evals > 1e-10
        #SAFETY
        if mask.sum() == 0:
            raise RuntimeError("RW2 K has no positive eigenvalues with given tol.")
        
        Q = evecs[:, mask].contiguous()
        Lsqrt = (Q * evals[mask].sqrt()).unsqueeze(0)      # shape [D, r]
        Z=beta @ Lsqrt
        qf = torch.sum(Z*Z, dim=-1)          # [batch]
        
        # Jacobian term: -sum log(Δγ_k) per batch
        log_jac = -torch.sum(torch.log(safe_dgamma), dim=-1).reshape(-1)    #[B]
        
        neg_log_prior_qf  = 0.5 * tau2 * qf  # depends on tau
        neg_log_prior_jac = -1*log_jac          # independent of tau
        
        #nlp = (neg_log_prior_qf + neg_log_prior_jac).sum()
        
        out = (
            neg_log_prior_qf    #(0.5 * tau2 * qf)   dependent on tau
            + neg_log_prior_jac  # log_jac independent on tau
            ).sum()        
        
        return {
            "neg_log_prior_total" : out,
            "neg_log_prior_qf" : neg_log_prior_qf.sum(), #Important for Sanity Check
            "neg_log_prior_jac": neg_log_prior_jac.sum(),
            "qf_sum": qf.sum(),                             ## sum over margins
            "qf_mean": qf.mean()                           #mean over margins
            } 

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
        
                # symmetrize
        K1 = 0.5 * (K1 + K1.T)
        K2 = 0.5 * (K2 + K2.T)

        # Quadratic forms per block (sum over columns M)
        qf1 = torch.einsum('im,ij,jm->', theta, K1, theta)   # scalar
        qf2 = torch.einsum('im,ij,jm->', theta, K2, theta)   # scalar

        neg_log_prior_qf = 0.5 * (kappa1 * qf1 + kappa2 * qf2) # scalar
        
        return {
            "neg_log_prior_total": neg_log_prior_qf,
            "qf1": qf1, 
            "qf2": qf2
            }
    
    @staticmethod
    def defining_prior(
        model: "GTM",
        hyperparameter,
        is_transformation = False
        ):
        
        sub_model = model.transformation if is_transformation else model.decorrelation_layers
        sub_model = model.transformation if is_transformation else model.decorrelation_layers
        
        sub_model = model.transformation if is_transformation else model.decorrelation_layers        
        
        total_logp = torch.zeros((), device=model.device, dtype=torch.float32)
        
        if not is_transformation:
            
            # --- Decorrelation layer prior (RW1 + RW2) ---------------------------
            # Paper §3.3: ridge on first & second differences to pull toward linear
            # (Gaussian-copula baseline) and to smooth curvature. :contentReference[oaicite:0]{index=0}
            
            # Plug-in (stable) alpha^2; or make alpha2 a learnable parameter and add its log-prior below.
            
            # Penalization Term
            # controls shrinkage toward linear behavior
            
            total = {
                "neg_log_prior_total": torch.zeros((), device=model.device),
                "qf1": torch.zeros((), device=model.device),
                "qf2": torch.zeros((), device=model.device)
                }
            
            tau_1 = torch.as_tensor(hyperparameter["tau_1"], device=model.device)
            tau_2 = torch.as_tensor(hyperparameter["tau_2"], device=model.device)
            
            sigma_hat=1
            
            for layer in sub_model:
                # K built once per layer; ensure it's on device
                K_RW1 = layer.priors.K_prior_RW1.to(device=model.device, dtype=torch.float32)
                K_RW2 = layer.priors.K_prior_RW2.to(device=model.device, dtype=torch.float32)
                
                theta_D = layer.params  # [Kdim, M]
                
                part = bayesian_splines.log_prior_gamma_ridge(theta_D, K_RW1, K_RW2, tau_1, tau_2, sigma_hat)
                
                for k in total:
                    total[k] = total[k] + part[k]
            
            return total
            
        else:
            # --- Transformation (marginal) layer prior (RW2 only) ----------------
            # Paper §3.3: monotone splines; first-diff penalty not needed; use RW2-only
            # smoothing (λ4 in the paper). :contentReference[oaicite:1]{index=1}
            
            tau_hat   = hyperparameter
            
            K_Prior_RW2 = sub_model.priors.K_prior_RW2.to(sub_model.device)
            K_Prior_RW2 = 0.5*(K_Prior_RW2+ K_Prior_RW2.T)
            
            # Use the *restricted* (monotone) coefficients θ for the P-spline prior (paper §2.1). :contentReference[oaicite:2]{index=2}
            varphi, M = get_transformation_param_matrix_live(model)  # [K, M], live params
            varphi = varphi.to(sub_model.device)
            
            # If you need to pad rows to match K_Prior_RW2 (rare):
            Kdim = K_Prior_RW2.shape[0]
            if varphi.shape[0] < Kdim:
                pad_rows = Kdim - varphi.shape[0]
                varphi = torch.nn.functional.pad(varphi, (0, 0, 0, pad_rows))
            
            theta_T = bayesian_splines._restrict_parameters_(
                params_a=varphi,
                monotonically_increasing=sub_model.monotonically_increasing,
                #use_softplus=False,
                device=sub_model.device
            )
            
            total_logp = (
                bayesian_splines.log_mvn_zero_mean_prec_ck(theta_T, K_Prior_RW2, tau_hat)
                )
            
            return total_logp    
        
    
    @staticmethod
    def _restrict_parameters_(
        params_a: Tensor,
        monotonically_increasing: bool,
        use_softplus: bool = True,
        device: torch.device=None
    ):
        
        if not monotonically_increasing:
            # Return a fresh tensor to avoid any aliasing surprises
            return params_a.contiguous()

        # Work in [B, K] = [M, K] by transposing and making contiguous
        a = params_a.T.contiguous()                                  # [M, K]

        head = a[:, :1]                                              # [M, 1]
        tail = a[:, 1:]                                              # [M, K-1]

        if use_softplus:
            tail_pos = softplus(tail) + 1e-6                       # [M, K-1]
        else:
            tail_pos = torch.exp(tail)

        eps = torch.cat([head, tail_pos], dim=1)                     # [M, K]
        eps_cum = torch.cumsum(eps, dim=1)                           # [M, K]

        out = eps_cum.T.contiguous()                                 # [K, M]
        return out
    
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
        K_prior_RW1: Tensor = bayesian_splines.difference_penalty_matrix(order=1, n_params=n_params, device=device, dtype=dtype)
        
        #order_prior_diff = 2
        #n_params = int(model.padded_params.shape[0])
        K_prior_RW2: Tensor = bayesian_splines.difference_penalty_matrix(order=2, n_params=n_params, device=device, dtype=dtype)
        
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
    