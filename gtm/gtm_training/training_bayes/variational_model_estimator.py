### BAYESIAN APPROACH
import torch
import math
import os
from torch.distributions import Normal
from torch.special import digamma as ψ
from torch import nn, Tensor
from torch.amp import autocast


try: torch.set_float32_matmul_precision("high")
except: pass

from typing import TYPE_CHECKING

from torch.nn.utils.stateless import _reparametrize_module

if TYPE_CHECKING:
    from gtm_model.gtm import GTM # type-only; no runtime import

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
        device: torch.device | str = "cpu",
        ):
        
        super().__init__()
        
        self.model = model
        self.device = model.device
        
        self._rng = None  # optional; can be set from outside
        
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
        
        try:
            self._single_objective_terms = torch.compile(self._single_objective_terms, mode="max-autotune")
        except Exception:
            pass

    
    def set_rng(self, gen: torch.Generator | None):
        self._rng = gen

    
    @property
    def sigma(self) -> Tensor:
        return 1e-6 + nn.functional.softplus(self.rho) #to avoid softplus near-zero stickiness 1e-6
        
    def sample_theta(self, num_samples: int = 1, antithetic = False) -> Tensor:
        """Reparameterized samples θ = μ + σ ⊙ ε, ε ~ N(0, I). Shape: [S, D]."""
        #eps = self._normal0.sample((num_samples,))
        #return self.mu + self.sigma * eps
        gen = self._rng
        if antithetic and num_samples >= 2:
            half = num_samples // 2
            eps = torch.randn((half, self.mu.numel()), device=self.mu.device, generator=gen)
            thetas = torch.cat([self.mu + self.sigma * eps,
                                self.mu - self.sigma * eps], dim=0)
            if thetas.shape[0] < num_samples:
                # one extra draw if odd
                extra = torch.randn((1, self.mu.numel()), device=self.mu.device, generator=gen)
                thetas = torch.cat([thetas, self.mu + self.sigma * extra], dim=0)
            return thetas
        else:
            eps = torch.randn((num_samples, self.mu.numel()), device=self.mu.device, generator=gen)
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
            with autocast(device_type=str(self.device), dtype=torch.float16):
                out = model.__bayesian_training_objective__(
                    samples=samples,
                    hyperparameters_transformation=tau4_s,
                    hyperparameters_decorrelation={"tau_1": tau1_s, "tau_2": tau2_s},
                    N_total=sample_size_total,
                    B=samples.shape[0],
                )

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

        log_p_tilde_vals = -neg_post_vec  # [S]
        elbo_core = torch.mean(beta_kl * log_q_vals - beta_logp * log_p_tilde_vals)

        if kl_vec is not None:
            elbo_loss = elbo_core + beta_tau_kl * torch.mean(kl_vec)
        else:
            elbo_loss = elbo_core

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

        decor_present_flag = not (model.number_decorrelation_layers == 0 or model.transform_only)

        if use_tau_vi_now and (tau_nodes is not None):
            tau4_vec, tau1_vec, tau2_vec, _kl = tau_nodes.sample_many(
                S=S, decor_present=decor_present_flag, generator=self._rng
            )
            if tau4_vec is None:
                tau4_vec = torch.full((S,), float(hyperparameter_transformation["tau"]),
                                      device=self.device, dtype=torch.float32)
            if decor_present_flag:
                if tau1_vec is None:
                    tau1_vec = torch.full((S,), float(hyperparameter_decorrelation.get("tau_1", 0.0)),
                                          device=self.device, dtype=torch.float32)
                if tau2_vec is None:
                    tau2_vec = torch.full((S,), float(hyperparameter_decorrelation.get("tau_2", 0.0)),
                                          device=self.device, dtype=torch.float32)
            else:
                tau1_vec = torch.zeros((S,), device=self.device, dtype=torch.float32)
                tau2_vec = torch.zeros((S,), device=self.device, dtype=torch.float32)
        else:
            tau4_vec = torch.full((S,), float(hyperparameter_transformation["tau"]),
                                  device=self.device, dtype=torch.float32)
            if decor_present_flag:
                tau1_vec = torch.full((S,), float(hyperparameter_decorrelation.get("tau_1", 0.0)),
                                      device=self.device, dtype=torch.float32)
                tau2_vec = torch.full((S,), float(hyperparameter_decorrelation.get("tau_2", 0.0)),
                                      device=self.device, dtype=torch.float32)
            else:
                tau1_vec = torch.zeros((S,), device=self.device, dtype=torch.float32)
                tau2_vec = torch.zeros((S,), device=self.device, dtype=torch.float32)

        # vmap a function that returns the batch NLL for each θ_s
        def _single_nll(theta_1d, t4, t1, t2):
            params_s = self._theta_to_state_dict(theta_1d)
            with _reparametrize_module(model, params_s):
                with autocast(str(self.device), dtype=torch.float16):
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
        eps_floor: float=1e-5,
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


