### BAYESIAN APPROACH
import torch
import math
import os
from torch.distributions import Normal
from torch.special import digamma as ψ
from torch import nn, Tensor
from torch.amp import autocast
from torch.nn import functional as F

try: torch.set_float32_matmul_precision("high")
except: pass

from typing import TYPE_CHECKING

from torch.nn.utils.stateless import _reparametrize_module

if TYPE_CHECKING:
    from gtm_model.gtm import GTM # type-only; no runtime import


def _finite(name, x):
    ok = torch.isfinite(x).all()
    if not ok:
        bad = x[~torch.isfinite(x)]
        raise FloatingPointError(
            f"[NON-FINITE] {name}: shape={tuple(x.shape)} "
            f"min={x.nan_to_num().min().item():.3e} "
            f"max={x.nan_to_num().max().item():.3e} "
            f"examples={bad[:5].detach().cpu().tolist()}"
        )
        
def _finite_vmap_safe(x, name):
    torch._assert(
        torch.isfinite(x).all(),
        f"Non-finite detected in {name}"
    )


def _finite_scalar(name, x):
    _finite(name, x.reshape(-1))


def _logmeanexp(x: torch.Tensor, dim: int = 0) -> torch.Tensor:
    """Stable log-mean-exp along dim."""
    m, _ = torch.max(x, dim=dim, keepdim=True)
    return m.squeeze(dim) + torch.log(torch.mean(torch.exp(x - m), dim=dim))

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
        mv_block_keys: list[str] | None = None,
        device: torch.device | str = "cpu",
        full_mvn: bool = False,
        mu_init: Tensor | dict | None = None,
        ):
        
        super().__init__()
        
        self.model = model
        self.device = model.device
        
        self._rng = None  # optional; can be set from outside
        
        # Snapshot an initial state dict to define θ's dimension and schema.
        with torch.no_grad():
            base_sd = {k: v.detach().to(self.device) 
                       for k, v in model.state_dict().items() 
                       if torch.is_tensor(v)
                       }
            
            theta0, self._schema = _flatten_state_dict(base_sd, key_filter=key_filter)

        D = theta0.numel()
        
        if D == 0:
            raise RuntimeError("Key filter selected zero parameters; check your include/exclude patterns.")
        
        if mu_init is not None:
            if isinstance(mu_init, dict):
                flat_init, schema_init = _flatten_state_dict(mu_init, key_filter=key_filter)
                schema_keys = [k for k, _ in self._schema]
                init_keys   = [k for k, _ in schema_init]
                if schema_keys != init_keys:
                    raise ValueError("mu_init state_dict has different keys/order than model.state_dict().")
                if flat_init.numel() != D:
                    raise ValueError("mu_init flatten size mismatch.")
                theta0 = flat_init.to(self.device)
            else:
                flat = mu_init.to(self.device)
                if flat.numel() != D:
                    raise ValueError(f"mu_init has {flat.numel()} params but expected {D}.")
                theta0 = flat.reshape_as(theta0)
        
        
        # Mean Vector
        self.mu = nn.Parameter(theta0.clone())
        
        # --- Build block structure ---------------------------------------------------
        #self.rho = nn.Parameter(torch.full((D,), math.log(math.exp(init_scale) - 1.0)))
        self._build_block_structure(D, mv_block_keys, full_mvn=full_mvn)
        
        # scalear -> softplus(ρ) = init_scale
        rho0 = math.log(math.exp(init_scale)-1.0)
        
        if self.num_diag > 0:
            self.rho = nn.Parameter(
                torch.full(
                    (self.num_diag,),
                    rho0,
                    device=self.device,
                )
            )
        else:
            # keep attributes to avoid attributes errors
            self.rho = nn.Parameter(
                torch.empty(
                    0, 
                    device = self.device
                    )
                )
        
        self.learn_scale = learn_scale
        if not learn_scale:
            self.rho.requires_grad_(False)

        # Multivariate blocks: unconstrained Cholesky params concatenated
        if self.block_sizes:
            total_tris = sum(
                k * (k + 1) // 2 
                for k in self.block_sizes
                )
            
            L_unscontrained = torch.zeros(
                total_tris, device=self.device
            )
            
            # initialize diagonals so softplus(diag) = init_scale
            offset = 0 
            for K in self.block_sizes:
                for i in range(K):
                    pos = offset + (i * (i + 1))// 2 + i
                    L_unscontrained[pos] = rho0
                offset += K * (K + 1) // 2
            
            self.L_unconstrained = nn.Parameter(L_unscontrained)
        else: 
            self.L_unconstrained = nn.Parameter(
                torch.empty(0, device=self.device)
            )
        
        self._normal0 = Normal(
            torch.zeros(D, device=self.device),
            torch.ones(D, device=self.device),
        )
        
        with torch.no_grad():
            sigma0 = self.sigma.detach().clone()
        self.register_buffer("sigma_init", sigma0)
        
        try:
            self._single_objective_terms = torch.compile(
                self._single_objective_terms, 
                mode="max-autotune"
                )

        except Exception:
            pass

    
    def set_rng(self, gen: torch.Generator | None):
        self._rng = gen
        
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

        if trans_idx:
            trans_idx = torch.cat(trans_idx)
        else:
            trans_idx = torch.empty(0, dtype=torch.long, device=self.device)

        if decor_idx:
            decor_idx = torch.cat(decor_idx)
        else:
            decor_idx = torch.empty(0, dtype=torch.long, device=self.device)

        return trans_idx, decor_idx

    
    def _build_block_structure(
        self,
        D: int,
        mv_block_keys: list[str]| None,
        full_mvn: bool = False,
        ):
        """
        Decide which flattened parameters belong to full-covariance blocks.

        For each (key, shape) in self._schema, if key contains any substring in
        mv_block_keys, that whole tensor becomes one MVN block (e.g. one spline).
        All remaining dims stay independent (diagonal).
        """
        
        self.block_indices: list[torch.Tensor] = []
        self.block_sizes: list[torch.Tensor] = []
        
        if full_mvn:
            # --- ONE HUGE BLOCK: everything is multivariate ---
            idx_all = torch.arange(D, device=self.device)
            self.block_indices = [idx_all]
            self.block_sizes = [D]

            block_mask = torch.ones(D, dtype=torch.bool, device=self.device)
            self.block_mask = block_mask
            self.diag_mask = ~block_mask      # all False
            self.diag_indices = torch.nonzero(self.diag_mask, as_tuple=True)[0]
            self.num_diag = int(self.diag_indices.numel())
            return
        
        block_mask = torch.zeros(D, dtype= torch.bool, device = self.device)
        
        if mv_block_keys:
            offset = 0
            for key, shape in self._schema:
                n = int(torch.tensor(shape).prod().item())
                idx = torch.arange(offset, offset + n, device=self.device)
                
                if any(sub in key for sub in mv_block_keys):
                    self.block_indices.append(idx)
                    self.block_sizes.append(n)
                    block_mask[idx] = True

                offset += n
        
        self.block_mask = block_mask
        self.diag_mask = ~block_mask
        
        self.diag_indices = torch.nonzero(self.diag_mask, as_tuple=True)[0]
        self.num_diag = int(self.diag_indices.numel())
        


    @property
    def sigma_diag(self) -> Tensor:
        """Std for diagonal (independent) dims only."""
        if self.num_diag == 0:
            return torch.empty(0, device= self.device)
        
        return 1e-6 + F.softplus(self.rho) #to avoid softplus near-zero stickiness 1e-6
    
    def _build_L_blocks(self) -> list[Tensor]:
        """
        Turn self.L_unconstrained into a list of lower-triangular matrices
        with positive diagonal via softplus.
        """
        L_blocks: list[Tensor] = []
        offset = 0

        for K in self.block_sizes:
            n_elem = K * (K + 1) // 2

            tri = torch.zeros((K, K), device=self.device)
            tril_idx = torch.tril_indices(K, K, offset=0)

            # FIX 1: use the correctly named attribute
            tri[tril_idx[0], tril_idx[1]] = self.L_unconstrained[offset:offset + n_elem]

            # enforce positive diagonal
            diag_raw = torch.diagonal(tri)
            diag_pos = F.softplus(diag_raw) + 1e-6
            tri = tri.clone()
            tri[torch.arange(K, device=self.device),
                torch.arange(K, device=self.device)] = diag_pos

            L_blocks.append(tri)
            offset += n_elem

        return L_blocks
    
    @property
    def sigma(self) -> Tensor:
        """
        Approximate per-parameter std dev = sqrt(diag(Σ)) combining
        block and diagonal parts. Used only for monitoring.
        """
        
        D = self.mu.numel()
        sigma2 = torch.zeros(D, device = self.device)
        
        # diag dims 
        if self.num_diag >0:
            sig_d = self.sigma_diag
            sigma2[self.diag_indices] = sig_d ** 2
        
        # block dims
        if self.block_sizes:
            L_blocks = self._build_L_blocks()
            
            for idx, L in zip(self.block_indices, L_blocks):
                # diag(Σ_b) for Σ_b = L L^T is sum over squares of each row
                cov_diag = (L**2).sum(dim=1)
                sigma2[idx] = cov_diag
        
        return torch.sqrt(1e-6 + sigma2) 
        
        
    def sample_theta(self, num_samples: int = 1, antithetic = False) -> Tensor:
        """
        Reparameterized samples θ = μ + T ε, ε ~ N(0, I).
        Block-diagonal T: full-cov blocks for splines, diagonal for the rest.
        Shape: [S, D].
        """
        
        gen = self._rng
        D = self.mu.numel()
        
        # global base noise, so antithetic pairing is for entire θ
        if antithetic and num_samples >= 2:
            half = num_samples // 2
            eps_half = torch.randn((half, D), device=self.mu.device, generator=gen)
            eps = torch.cat([eps_half, -eps_half], dim=0)
            if eps.shape[0] < num_samples:
                extra = torch.randn((1, D), device=self.mu.device, generator=gen)
                eps = torch.cat([eps, extra], dim=0)
        else:
            eps = torch.randn((num_samples, D), device=self.mu.device, generator=gen)

        theta = self.mu.expand_as(eps).clone()

        # Diagonal part: θ_i = μ_i + σ_i ε_i
        if self.num_diag > 0:
            sig_d = self.sigma_diag  # [D_diag]
            theta[:, self.diag_indices] += eps[:, self.diag_indices] * sig_d

        # Block parts: θ_b = μ_b + L_b ε_b
        if self.block_sizes:
            L_blocks = self._build_L_blocks()
            for idx, L in zip(self.block_indices, L_blocks):
                eps_block = eps[:, idx]       # [S, K]
                theta[:, idx] += eps_block @ L.T  # [S, K]

        return theta
        

#    def log_q(self, theta: Tensor) -> Tensor:
#        """log q_phi(θ) under mean-field Normal. theta shape [S, D] or [D]. Returns [S]."""
#        mu = self.mu
#        sigma = self.sigma
#        # compute per-sample log prob
#        if theta.dim() == 1:
#            theta = theta.unsqueeze(0)
#        S, D = theta.shape
#        # Normal log-density per dimension then sum
#        log_det = torch.sum(torch.log(sigma))
#        quad = 0.5 * torch.sum(((theta - mu) / sigma) ** 2, dim=1)
#        const = 0.5 * D * math.log(2 * math.pi)
#        return -(const + log_det + quad)
    
    def log_q(self, theta: Tensor) -> Tensor:
        """
        log q_phi(θ) for block-diagonal MVN:
        - full-cov blocks for splines
        - diagonal for the rest
        Input: θ shape [S, D] or [D]; returns [S].
        """
        if theta.dim() == 1:
            theta = theta.unsqueeze(0)
        S, D = theta.shape
        mu = self.mu

        log_q_val = torch.zeros(S, device=theta.device, dtype=theta.dtype)

        # Diagonal contribution
        if self.num_diag > 0:
            sig_d = self.sigma_diag  # [D_diag]
            mu_d = mu[self.diag_indices]  # [D_diag]
            th_d = theta[:, self.diag_indices]  # [S, D_diag]

            diff = (th_d - mu_d) / sig_d  # [S, D_diag]
            quad = (diff ** 2).sum(dim=1)  # [S]
            log_det = torch.sum(torch.log(sig_d))  # scalar

            const = self.num_diag * math.log(2.0 * math.pi)
            log_q_diag = -0.5 * (const + 2.0 * log_det + quad)
            log_q_val += log_q_diag

        # Block contributions
        if self.block_sizes:
            L_blocks = self._build_L_blocks()
            for bi, (idx, L) in enumerate(zip(self.block_indices, L_blocks)):
                K = idx.numel()
                th_b = theta[:, idx]        # [S, K]
                mu_b = mu[idx]              # [K]
                diff = th_b - mu_b          # [S, K]

                d = torch.diagonal(L)
                _finite(f"diag(L)[block={bi}]", d)
                
                d_clamped = d.clamp_min(1e-6)
                
                cond_proxy = (d_clamped.max() / d_clamped.min()).item()
                if cond_proxy > 1e8:
                    print(f"WARNING ill-conditioned L: cond_proxy≈{cond_proxy:.2e} (block={bi}, K={K})")


                # y = L^{-1} (θ - μ)  (whitening)
                # torch.linalg.solve_triangular solves A x = B with B shape [..., K]
                y = torch.linalg.solve_triangular(
                    L,
                    diff.T,
                    upper=False,
                ).T  # [S, K]
                
                _finite(f"y (whitened)[block={bi}]", y)
                
                quad = (y ** 2).sum(dim=1)  # [S]
                log_det = 2.0 * torch.log(d_clamped).sum() # log |Σ_b|

                const = K * math.log(2.0 * math.pi)
                log_q_block = -0.5 * (const + log_det + quad)
                log_q_val += log_q_block
                
        _finite("log_q_val", log_q_val)
        return log_q_val



    def _theta_to_state_dict(self, theta_1d: Tensor):
        return _unflatten_to_state_dict(theta_1d, self._schema)

    def _device_type_from(self):
        if isinstance(self.device, torch.device):
            return self.device.type       # "cuda" / "cpu"
        return str(self.device)
    
    @torch.no_grad()
    def set_model_params(self, theta_1d: Tensor):
        """Load θ back into the GTM model."""
        sd_new = self._theta_to_state_dict(theta_1d)
        # Use existing state_dict to preserve buffers that are not in θ
        full_sd = self.model.state_dict()
        for k, v in sd_new.items():
            full_sd[k] = v
        self.model.load_state_dict(full_sd, strict=False)
        
    def _single_objective_terms(
        self,
        theta_1d: Tensor,
        samples: Tensor,
        model: "GTM",
        tau4_s: Tensor,   # scalar tensor on device
        tau1_s: Tensor,   # scalar tensor on device (can be 0.0 if no decor)
        tau2_s: Tensor,   # scalar tensor on device (can be 0.0 if no decor)
        sample_size_total: int,
    ):
        # Reparametrize model with θ_s (pure functional call)
        params_s = self._theta_to_state_dict(theta_1d)

        with _reparametrize_module(model, params_s):
            with autocast(device_type=self._device_type_from(), dtype=torch.float16):
                out = model.__bayesian_training_objective__(
                    samples=samples,
                    hyperparameters_transformation=tau4_s,
                    hyperparameters_decorrelation={"tau_1": tau1_s, "tau_2": tau2_s},
                    N_total=sample_size_total,
                    B=samples.shape[0],
                )
                
        #for k in ["neg_posterior", "negative_log_lik", "nll_batch"]:
        #    _finite(f"out[{k}]", out[k].reshape(-1))

        #dp = out["negative_decorrelation_prior"]
        #tp = out["negative_transformation_prior"]
        #_finite("out[decor.neg_log_prior_total]", dp["neg_log_prior_total"].reshape(-1))
        #_finite("out[decor.qf1]", dp["qf1"].reshape(-1))
        #_finite("out[decor.qf2]", dp["qf2"].reshape(-1))
        #_finite("out[trans.neg_log_prior_total]", tp["neg_log_prior_total"].reshape(-1))
        #_finite("out[trans.neg_log_prior_qf]", tp["neg_log_prior_qf"].reshape(-1))
        
        
        # Extract tensors needed by ELBO/monitors (all scalars)
        neg_post   = out["neg_posterior"]
        neg_lik    = out["negative_log_lik"]
        nll_batch  = out["nll_batch"]

        ndp_total  = out["negative_decorrelation_prior"]["neg_log_prior_total"]
        qf1_sum    = out["negative_decorrelation_prior"]["qf1"]
        qf2_sum    = out["negative_decorrelation_prior"]["qf2"]

        ntp_total  = out["negative_transformation_prior"]["neg_log_prior_total"]
        ntp_qf     = out["negative_transformation_prior"]["neg_log_prior_qf"]
        qf_sum_T   = out["negative_transformation_prior"]["qf_sum"]
        qf_mean_T  = out["negative_transformation_prior"]["qf_mean"]

        # Return a flat tuple of tensors so vmap can stack to [S, ...]
        return (neg_post, neg_lik, nll_batch,
                ndp_total, qf1_sum, qf2_sum,
                ntp_total, ntp_qf, qf_sum_T, qf_mean_T)

    def step(
        self,
        samples: Tensor,
        hyperparameter_transformation,
        hyperparameter_decorrelation,
        model: "GTM",
        sample_size_total,
        mcmc_samples: int = 100,
        seed: int | None = None,
        beta_kl=1.0,
        beta_logp=1.0,
        tau_pack: 'TauPack'=None,
        beta_tau_kl=1.0
    ):
        """
        One stochastic-ELBO step (no optimizer step).
        Returns dict with 'loss' and components for logging.
        """
        
        # Sample θ ~ q
        thetas = self.sample_theta(mcmc_samples, antithetic=True)  # [S, D]
        
        self.check_values_and_sanity(thetas)
        
        log_q_vals = self.log_q(thetas)         # [S]

        log_p_tilde_vals = []   # log unnormalized posterior per sample

        # --- prepare τ vectors (size S) and τ-KL per-sample ---
        decor_present_flag = not (model.number_decorrelation_layers == 0 or model.transform_only)

        if tau_pack is not None:
            tau4_vec, tau1_vec, tau2_vec, kl_vec = tau_pack.sample_many(
                S=mcmc_samples, decor_present=decor_present_flag, generator=self._rng
            )
            # Fallback to fixed τ if a node is None
            if tau4_vec is None:
                tau4_vec = torch.full((mcmc_samples,), float(hyperparameter_transformation["tau"]),
                                      device=self.device, dtype=torch.float32)
            if decor_present_flag:
                if tau1_vec is None:
                    tau1_vec = torch.full((mcmc_samples,), float(hyperparameter_decorrelation.get("tau_1", 0.0)),
                                          device=self.device, dtype=torch.float32)
                if tau2_vec is None:
                    tau2_vec = torch.full((mcmc_samples,), float(hyperparameter_decorrelation.get("tau_2", 0.0)),
                                          device=self.device, dtype=torch.float32)
            else:
                tau1_vec = torch.zeros((mcmc_samples,), device=self.device, dtype=torch.float32)
                tau2_vec = torch.zeros((mcmc_samples,), device=self.device, dtype=torch.float32)
        else:
            # EB/CAVI fixed τ’s broadcasted to S
            tau4_vec = torch.full((mcmc_samples,), float(hyperparameter_transformation["tau"]),
                                  device=self.device, dtype=torch.float32)
            if decor_present_flag:
                tau1_vec = torch.full((mcmc_samples,), float(hyperparameter_decorrelation.get("tau_1", 0.0)),
                                      device=self.device, dtype=torch.float32)
                tau2_vec = torch.full((mcmc_samples,), float(hyperparameter_decorrelation.get("tau_2", 0.0)),
                                      device=self.device, dtype=torch.float32)
            else:
                tau1_vec = torch.zeros((mcmc_samples,), device=self.device, dtype=torch.float32)
                tau2_vec = torch.zeros((mcmc_samples,), device=self.device, dtype=torch.float32)
            kl_vec = None  # no τ KL term

        # --- vectorized objective over S samples ---
        # thetas: [S, D]; tau*_vec: [S]
        results = torch.vmap(
            lambda θ, t4, t1, t2: self._single_objective_terms(
                θ, samples, model, t4, t1, t2, sample_size_total
            )
        )(thetas, tau4_vec, tau1_vec, tau2_vec)

        # Unpack stacked results, each is [S]
        (neg_post_vec, neg_lik_vec, nll_batch_vec,
         ndp_vec, qf1_vec, qf2_vec,
         ntp_vec, ntp_qf_vec, qf_sum_T_vec, qf_mean_T_vec) = results
        
        _finite("neg_post_vec", neg_post_vec)
        _finite("neg_lik_vec", neg_lik_vec)
        _finite("nll_batch_vec", nll_batch_vec)

        log_p_tilde_vals = -neg_post_vec  # [S]
        elbo_core = torch.mean(beta_kl * log_q_vals - beta_logp * log_p_tilde_vals)
        _finite_scalar("elbo_core", elbo_core)

        if kl_vec is not None:
            _finite("kl_vec", kl_vec)
            elbo_loss = elbo_core + beta_tau_kl * torch.mean(kl_vec)
        else:
            elbo_loss = elbo_core
        
        _finite_scalar("elbo_loss", elbo_loss)

        # Monitors (means over S)
        return {
            "elbo_loss": elbo_loss,
            "mean_log_q": torch.mean(log_q_vals).detach(),
            "mean_log_p_tilde": torch.mean(log_p_tilde_vals).detach(),
            "log_likelihhod_batch": float(_logmeanexp(-nll_batch_vec, dim=0).detach()),

            "sigma_mean": self.sigma.mean().detach(),
            "sigma_max": self.sigma.max().detach(),
            "sigma_min": self.sigma.min().detach(),

            "neg_prior_decorrelation": torch.mean(ndp_vec).detach(),
            "qf1_decorrelation": torch.mean(qf1_vec).detach(),
            "qf2_decorrelation": torch.mean(qf2_vec).detach(),

            "neg_prior_transformation": torch.mean(ntp_vec).detach(),
            "transformation_neg_log_prior_df": torch.mean(ntp_qf_vec).detach(),
            "transformation_sum_qf": torch.mean(qf_sum_T_vec).detach(),
            "transformation_mean_qf": torch.mean(qf_mean_T_vec).detach(),
        }

    def check_values_and_sanity(self, thetas):
        _finite("thetas", thetas)
        _finite("mu", self.mu)
        _finite("rho", self.rho)
        _finite("L_unconstrained", self.L_unconstrained)

        sig_d = self.sigma_diag
        _finite("sigma_diag", sig_d)

        # check L blocks + diagonals
        if self.block_sizes:
            L_blocks = self._build_L_blocks()
            for j, L in enumerate(L_blocks):
                _finite(f"L_block[{j}]", L)
                d = torch.diagonal(L)
                _finite(f"diag(L_block[{j}])", d)
                if (d <= 0).any():
                    raise FloatingPointError(f"[BAD DIAG] L_block[{j}] has non-positive diagonal!")

        
    @torch.no_grad()
    def predictive_loglik_sum_batch(
        self,
        y_batch: torch.Tensor,
        model: "GTM",
        hyperparameter_transformation,
        hyperparameter_decorrelation,
        sample_size: int,
        S: int = 8,
        batch_size=1,
        seed: int | None = None,
        tau_nodes: "TauPack"=None,
        use_tau_vi_now: bool = False,
    ) -> float:
        """For Validation used the ELPD Approach."""
    
        thetas = self.sample_theta(S, antithetic=True)  # [S, D]

        # either VI τ or fixed τ
        if use_tau_vi_now and (tau_nodes is not None):
            tau4_vec, tau1_vec, tau2_vec, _ = self._tau_vectors(
                S=S,
                model=model,
                hyper_T=hyperparameter_transformation,
                hyper_D=hyperparameter_decorrelation,
                tau_nodes=tau_nodes,
            )
        else:
            tau4_vec, tau1_vec, tau2_vec, _ = self._tau_vectors(
                S=S,
                model=model,
                hyper_T=hyperparameter_transformation,
                hyper_D=hyperparameter_decorrelation,
                tau_nodes=None,
            )

        # vmap a function that returns the batch NLL for each θ_s
        def _single_nll(theta_1d, t4, t1, t2):
            params_s = self._theta_to_state_dict(theta_1d)
            with _reparametrize_module(model, params_s):
                with autocast(device_type=self._device_type_from(), dtype=torch.float16):
                    out = model.__bayesian_training_objective__(
                        samples=y_batch,
                        hyperparameters_transformation=t4,
                        hyperparameters_decorrelation={"tau_1": t1, "tau_2": t2},
                        N_total=sample_size,
                        B=batch_size,
                    )
            return out["negative_log_lik"].reshape(())

        nll_vec = torch.vmap(_single_nll)(thetas, tau4_vec, tau1_vec, tau2_vec)  # [S]
        ll_vec = -nll_vec
        return float(_logmeanexp(ll_vec, dim=0))
    
    
    ##### HELPER FOR BRECHMARKING AND REVERSE-SAMPLING ######
    
    def _tau_vectors(
        self,
        S: int,
        model: "GTM",
        hyper_T,
        hyper_D,
        tau_nodes: "TauPack",
    ):
        """
        Helper: produce tau4_vec, tau1_vec, tau2_vec, kl_vec for S MC samples.

        - Uses TauPack.sample_many if tau_nodes is not None.
        - Falls back to fixed hyperparameters if node is missing.
        - Handles 'no decorrelation layer' case.
        """
        device = self.device
        decor_present_flag = not (model.number_decorrelation_layers == 0 or model.transform_only)

        if tau_nodes is not None:
            tau4_vec, tau1_vec, tau2_vec, kl_vec = tau_nodes.sample_many(
                S=S,
                decor_present=decor_present_flag,
                generator=self._rng,
            )
            # Fallbacks if some nodes are None
            if tau4_vec is None:
                tau4_vec = torch.full(
                    (S,),
                    float(hyper_T["tau"]),
                    device=device,
                    dtype=torch.float32,
                )
            if decor_present_flag:
                if tau1_vec is None:
                    tau1_vec = torch.full(
                        (S,),
                        float(hyper_D.get("tau_1", 0.0)),
                        device=device,
                        dtype=torch.float32,
                    )
                if tau2_vec is None:
                    tau2_vec = torch.full(
                        (S,),
                        float(hyper_D.get("tau_2", 0.0)),
                        device=device,
                        dtype=torch.float32,
                    )
            else:
                tau1_vec = torch.zeros((S,), device=device, dtype=torch.float32)
                tau2_vec = torch.zeros((S,), device=device, dtype=torch.float32)
        else:
            # Pure EB / fixed τ case
            tau4_vec = torch.full(
                (S,),
                float(hyper_T["tau"]),
                device=device,
                dtype=torch.float32,
            )
            decor_present_flag = not (model.number_decorrelation_layers == 0 or model.transform_only)
            if decor_present_flag:
                tau1_vec = torch.full(
                    (S,),
                    float(hyper_D.get("tau_1", 0.0)),
                    device=device,
                    dtype=torch.float32,
                )
                tau2_vec = torch.full(
                    (S,),
                    float(hyper_D.get("tau_2", 0.0)),
                    device=device,
                    dtype=torch.float32,
                )
            else:
                tau1_vec = torch.zeros((S,), device=device, dtype=torch.float32)
                tau2_vec = torch.zeros((S,), device=device, dtype=torch.float32)
            kl_vec = None

        return tau4_vec, tau1_vec, tau2_vec, kl_vec
    
    
    @torch.no_grad()
    def predictive_log_prob(
        self,
        y: torch.Tensor,                # [N, d]
        model: "GTM",
        hyperparameter_transformation,
        hyperparameter_decorrelation,
        tau_nodes: "TauPack" = None,
        S: int = 32,
    ) -> torch.Tensor:
        """
        Monte Carlo estimate of log p(y) under the Bayesian predictive,
        via log-mean-exp over (θ, τ):

            log p(y_n) ≈ log (1/S ∑_s p(y_n | θ_s, τ_s))

        Returns: [N] tensor of log predictive densities for each observation.
        """
        device = self.device
        N = y.shape[0]

        # θ ~ q
        thetas = self.sample_theta(S, antithetic=True)  # [S, D]

        # τ vectors
        tau4_vec, tau1_vec, tau2_vec, _ = self._tau_vectors(
            S=S,
            model=model,
            hyper_T=hyperparameter_transformation,
            hyper_D=hyperparameter_decorrelation,
            tau_nodes=tau_nodes,
        )

        logps = []

        for s in range(S):
            theta_s = thetas[s]
            params_s = self._theta_to_state_dict(theta_s)

            with _reparametrize_module(model, params_s):
                # write τ into the model's hyperparameter dict
                model.hyperparameter["transformation"]["tau"] = float(tau4_vec[s].item())
                if model.number_decorrelation_layers > 0 and not model.transform_only:
                    model.hyperparameter["decorrelation"]["tau_1"] = float(tau1_vec[s].item())
                    model.hyperparameter["decorrelation"]["tau_2"] = float(tau2_vec[s].item())

                # your GTM API: returns [N] log-likelihood
                ll_s = model.log_likelihood(y)  # [N]
                logps.append(ll_s)

        logps = torch.stack(logps, dim=0)  # [S, N]

        # log-mean-exp over S
        m, _ = torch.max(logps, dim=0, keepdim=True)      # [1, N]
        log_mean = m.squeeze(0) + torch.log(torch.mean(torch.exp(logps - m), dim=0))  # [N]

        return log_mean
    
    @torch.no_grad()
    def predictive_sample(
        self,
        model: "GTM",
        hyperparameter_transformation,
        hyperparameter_decorrelation,
        tau_nodes: "TauPack" = None,
        n_samples: int = 10_000,
        S: int = 32,
    ) -> torch.Tensor:
        """
        Draw samples from the Bayesian predictive

            p(y) ≈ (1/S) ∑_s p(y | θ_s, τ_s)

        by:
          - sampling θ_s ~ q(θ)
          - sampling τ_s ~ q(τ) (or using hyperparameters)
          - for each s, drawing ≈ n_samples / S from model.sample()

        Returns: [n_samples, d] tensor of draws.
        """
        device = self.device
        S_eff = min(S, n_samples)
        n_per = math.ceil(n_samples / S_eff)

        # θ ~ q
        thetas = self.sample_theta(S_eff, antithetic=True)  # [S_eff, D]

        # τ vectors
        tau4_vec, tau1_vec, tau2_vec, _ = self._tau_vectors(
            S=S_eff,
            model=model,
            hyper_T=hyperparameter_transformation,
            hyper_D=hyperparameter_decorrelation,
            tau_nodes=tau_nodes,
        )

        samples_all = []

        for s in range(S_eff):
            theta_s = thetas[s]
            params_s = self._theta_to_state_dict(theta_s)

            with _reparametrize_module(model, params_s):
                model.hyperparameter["transformation"]["tau"] = float(tau4_vec[s].item())
                if model.number_decorrelation_layers > 0 and not model.transform_only:
                    model.hyperparameter["decorrelation"]["tau_1"] = float(tau1_vec[s].item())
                    model.hyperparameter["decorrelation"]["tau_2"] = float(tau2_vec[s].item())

                y_s = model.sample(n_per)  # [n_per, d]
                samples_all.append(y_s)

        y_cat = torch.cat(samples_all, dim=0)
        return y_cat[:n_samples]

class VariationalGamma:
    """q(tau) = Gamma(a_hat, b_hat) (shape-rate) with analytics you need for ELBO."""
    def __init__(
        self,
        a0: float,
        b0: float,
        rank_total: int,
        init_from_prior: bool = True
        ):
        
        
        self.a0 = float(a0)
        self.b0 = float(b0)
        self.rank_total = int(rank_total)
        
        if init_from_prior:
            # start exactly at the prior
            self.a_hat = self.a0
            self.b_hat = self.b0
        else:
            # previous behavior (more aggressive)
            self.a_hat = self.a0 + 0.5 * self.rank_total
            self.b_hat = self.b0 + 1e-6

    @property
    def mean(self) -> float:
        return float(self.a_hat / max(self.b_hat, 1e-12))

    @property
    def log_mean(self) -> float:
        # E_q[log tau] = ψ(a_hat) - log(b_hat)
        return float(ψ(torch.tensor(self.a_hat)).item() - math.log(self.b_hat))

    def update_from_E_qf_total(self, E_qf_total: float):
        """CAVI-style update using E_q[βᵀKβ] (can be MC-estimated)."""
        self.a_hat = self.a0 + 0.5 * self.rank_total
        self.b_hat = self.b0 + 0.5 * float(E_qf_total)

    def kl_to_prior(self) -> float:
        """
        KL(q||p) for Gamma(shape-rate).
        KL = (a_hat - a0)*ψ(a_hat) - log Γ(a_hat) + log Γ(a0)
             + a0*(log b_hat - log b0) + a_hat*(b0/b_hat - 1)
        """
        a_hat, b_hat, a0, b0 = self.a_hat, self.b_hat, self.a0, self.b0
        # use torch for stability but return float
        a_hat_t = torch.tensor(a_hat)
        term = (
            (a_hat - a0) * ψ(a_hat_t)
            - torch.lgamma(a_hat_t)
            + torch.lgamma(torch.tensor(a0))
            + a0 * (math.log(b_hat) - math.log(b0))
            + a_hat * (b0 / b_hat - 1.0)
        )
        return float(term)

class GammaTauNode(nn.Module):
    """
    Variational node for a single precision parameter τ with:
      prior:        Gamma(a, b)  (shape a, rate b)
      variational:  Gamma(a_hat, b_hat) reparameterized

    a_hat = exp(mu)        (shape)
    b_hat = exp(log_sigma) (rate)

    Initialisation ensures:
       E_q[τ] = mean_init
       CV     = cv_init
    """
    def __init__(
        self,
        a: float,
        b: float,
        mean_init: float,
        cv_init: float=0.5,
        eps_floor: float=1e-3,
        device="cpu"
        ):
        
        super().__init__()
        
        # prior hyper-parameters (buffers, not learmed)
        self.register_buffer("a", torch.as_tensor(float(a), dtype=torch.float32, device=device))
        self.register_buffer("b", torch.as_tensor(float(b), dtype=torch.float32, device=device))
        
        self.eps_floor = float(eps_floor)
        
        # choose initial (a_hat, b_hat) such that:
        #   E_q[τ] = mean_init
        #   CV(τ)  = sqrt(Var)/E ≈ cv_init
        #
        # For Gamma(a_hat, b_hat): mean = a_hat / b_hat, var = a_hat / b_hat^2
        # => CV^2 = 1 / a_hat  => a_hat = 1 / cv^2,    b_hat = a_hat / mean
        
        mean_init = max(float(mean_init), 1e-6)
        cv_init = max(float(cv_init), 1e-6)
        
        a_hat0 = 1.0/(cv_init **2)
        b_hat0 = a_hat0/mean_init

        self.mu         = nn.Parameter(torch.log(torch.as_tensor(a_hat0, dtype=torch.float32, device=device)))
        self.log_sigma  = nn.Parameter(torch.log(torch.as_tensor(b_hat0, dtype=torch.float32, device=device)))

    
    @property # shape parameter of q(τ)
    def a_hat(self): return torch.exp(self.mu).clamp_min(1e-8)
    
    
    @property # rate parameters of q(τ)
    def b_hat(self): return torch.exp(self.log_sigma).clamp_min(1e-8)
    
    def _q_dist(self): return torch.distributions.Gamma(self.a_hat, self.b_hat)
    
    def _p_dist(self): return torch.distributions.Gamma(self.a, self.b)
    
    
    def sample_tau(self, n_samples: int, generator: torch.Generator | None = None):
        """
        Returns (tau[S], logp_tau[S], logq_tau[S]) with:
          q(τ) = Gamma(a_hat, b_hat)
          p(τ) = Gamma(a, b)
        """
        
        
        q = self._q_dist()
        p = self._p_dist()

        try:
            tau = q.rsample((n_samples,), generator=generator)      # [S]
        except TypeError:
            # older torch Gamma.rsample has no generator arg
            if generator is not None:
                # draw base noise via generator, then inverse-CDF
                # (fallback: just ignore generator)
                tau = q.rsample((n_samples,))
            else:
                tau = q.rsample((n_samples,))

        tau = tau + self.eps_floor
        logq_tau = q.log_prob(tau)          # [S]
        logp_tau = p.log_prob(tau)          # [S]

        return tau, logp_tau, logq_tau

    def kl_mc(self, S: int = 8, generator: torch.Generator | None = None):
        """Monte Carlo KL = E_q[log q - log p]."""
        _, logp, logq = self.sample_tau(S, generator=generator)
        return (logq - logp).mean()

    def mean_and_var_tau_mc(self, S: int = 128, generator: torch.Generator | None = None):
        """
        We actually have closed form, so we don't need MC:
            mean = a_hat / b_hat
            var  = a_hat / b_hat^2
        Kept the name *_mc for API compatibility.
        """
        a_hat = self.a_hat
        b_hat = self.b_hat
        mean = a_hat / b_hat
        var = a_hat / (b_hat ** 2)
        return float(mean), float(var)

    def clamp_sigma(self, min_val: float=1e-3, max_val: float=3.0):
        with torch.no_grad():
            lo = torch.log(torch.tensor(min_val))
            hi = torch.log(torch.tensor(max_val))
            self.log_sigma.clamp_(min=lo, max=hi)
            

class TauPack:
    """
    Holds up to three tau nodes (τ4 for transformation, τ1/τ2 for decor),
    samples them on demand, and returns:
      - hyper_T / hyper_D dictionaries for the model call
      - (logq - logp) for the τ KL contribution (per-sample, to add to ELBO)
    """
    def __init__(self, node4: GammaTauNode | None, node1: GammaTauNode | None, node2: GammaTauNode | None):
        self.node4 = node4
        self.node1 = node1
        self.node2 = node2

    def sample_once(self, decor_present: bool, generator=None):
        logq_minus_logp = 0.0
        
        # τ4
        if self.node4 is not None:
            tau4, logp4, logq4 = self.node4.sample_tau(1, generator=generator)
            tau4 = tau4.squeeze(0)
            logq_minus_logp = logq_minus_logp + (logq4 - logp4).squeeze(0) #KL Divergenve for τ4
            hyper_T = {"tau": tau4}
        else:
            hyper_T = {}

        # τ1, τ2
        if decor_present and (self.node1 is not None) and (self.node2 is not None):
            t1, p1, q1 = self.node1.sample_tau(1, generator=generator)
            t2, p2, q2 = self.node2.sample_tau(1, generator=generator)
            t1, t2 = t1.squeeze(0), t2.squeeze(0)
            logq_minus_logp = logq_minus_logp + (q1 - p1).squeeze(0) + (q2 - p2).squeeze(0) #KL Divergenve for τ1, τ2
            hyper_D = {"tau_1": t1, "tau_2": t2}
        else:
            hyper_D = {}
        return hyper_T, hyper_D, logq_minus_logp
    
    def tau_param_group(self, lr: float, **extras):
        """Convenience param-group for optimizer; keeps your naming."""
        params = []
        if self.node4 is not None:
            params += [self.node4.mu, self.node4.log_sigma]
        if self.node1 is not None:
            params += [self.node1.mu, self.node1.log_sigma]
        if self.node2 is not None:
            params += [self.node2.mu, self.node2.log_sigma]
        return {"params": params, "lr": lr, "name": "tau_nodes", **extras}
    
    def sample_many(self, S: int, decor_present: bool, generator=None):
        """
        Vectorized τ sampling for S MC samples.

        Returns:
            tau4_vec: [S] (or None)
            tau1_vec: [S] (or None)
            tau2_vec: [S] (or None)
            kl_vec:   [S]  (per-sample KL = (log q - log p) sum over present τ's)
        """
        device = (self.node4.mu.device if self.node4 is not None
                  else (self.node1.mu.device if self.node1 is not None else "cpu"))
        kl = torch.zeros((S,), device=device)

        tau4_vec = None
        tau1_vec = None
        tau2_vec = None

        if self.node4 is not None:
            t4, p4, q4 = self.node4.sample_tau(S, generator=generator)  # [S]
            tau4_vec = t4
            kl = kl + (q4 - p4)

        if decor_present and (self.node1 is not None) and (self.node2 is not None):
            t1, p1, q1 = self.node1.sample_tau(S, generator=generator)
            t2, p2, q2 = self.node2.sample_tau(S, generator=generator)
            tau1_vec, tau2_vec = t1, t2
            kl = kl + (q1 - p1) + (q2 - p2)

        return tau4_vec, tau1_vec, tau2_vec, kl


