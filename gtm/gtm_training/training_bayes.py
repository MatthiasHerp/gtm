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


def _flatten_state_dict(sd):
    """Flatten a (detached) state_dict with only tensor leaves into a single vector
    and keep a schema to reconstruct the dict later."""
    keys = []
    shapes = []
    flats = []
    for k, v in sd.items():
        if not torch.is_tensor(v):
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
        device: torch.device | str = "cpu",
        ):
        
        super().__init__()
        
        self.model = model
        self.device = model.device
        
        
        # Snapshot an initial state dict to define θ's dimension and schema.
        with torch.no_grad():
            base_sd = {k: v.detach().to(self.device) for k, v in model.state_dict().items() if torch.is_tensor(v)}
            theta0, self._schema = _flatten_state_dict(base_sd)

        D = theta0.numel()

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
    ) :
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
        
        for s in range(mcmc_samples):
            theta_s = thetas[s]
            # Push θ into model
            #self.set_model_params(theta_s)
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
            neg_likelihood_list.append(out['negative_log_lik'].reshape(()))
            prior_dec_list.append(out['negative_decorrelation_prior'].reshape(()))
            prior_trans_list.append(out['negative_transformation_prior'].reshape(()))

        log_p_tilde_vals = torch.stack(log_p_tilde_vals)  # [S]
        # Monte-Carlo KL(q || p) estimate: E_q[log q - log p̃]
        # (Note: additive constant log p(y) cancels in optimization)
        loss = torch.mean(log_q_vals - log_p_tilde_vals) #ELBO

        
        neg_likelihood_list= torch.stack(neg_likelihood_list)
        prior_dec_list = torch.stack(prior_dec_list)
        prior_trans_list = torch.stack(prior_trans_list)
        
        return {
            "loss": loss,
            "mean_log_q": torch.mean(log_q_vals).detach(),
            "mean_log_p_tilde": torch.mean(log_p_tilde_vals).detach(),
            "sigma_mean": self.sigma.mean().detach(),
            "sigma_max": self.sigma.max().detach(),
            "sigma_min": self.sigma.min().detach(),
            "neg_log_likelihood": torch.mean(neg_likelihood_list).detach(),
            "neg_prior_decorrelation": torch.mean(prior_dec_list).detach(),
            "neg_prior_transformation": torch.mean(prior_trans_list).detach(),
        }
        
        
    # --- NEW: Full-Bayes VI over tau's (and optional AR(1)) ---
class VI_Model_FullBayes(VI_Model):
    """
    Extends your VI over theta to include variational factors for:
      - tau4^2 (transformation layer)
      - tau_dec^2 for decorrelation blocks (list)
      - (optional) AR(1): sigma^2 and rho (disabled by default)

    Parametrization:
      s_tau = log(tau^2)  with  q(s_tau) = Normal(m, softplus(r))
      Prior on tau^2 ~ IG(a, b)  =>  log p(s_tau) = const + (-a)*s_tau - b * exp(-s_tau)
    """
    def __init__(
        self,
        model: "GTM",
        tau4_prior: tuple[float, float],                # (a4, b4) for IG
        tau_dec_priors: list[tuple[float, float]] | None = None,  # list of (a,b) per decor block
        use_ar1: bool = False,
        ar1_sigma_prior: tuple[float, float] | None = None,       # IG(a_sigma, b_sigma)
        ar1_rho_prior: tuple[float, float] | None = None,         # Beta(a_rho, b_rho) on u=(rho+1)/2
        init_scale_logtau: float = 0.0,               # start tau^2 near exp(0)=1.0
        device: torch.device | str = "cpu",
    ):
        super().__init__(model=model, init_scale=0.05, learn_scale=True, device=device)

        self.use_ar1 = bool(use_ar1)

        # ---- Priors (kept as buffers) ----
        a4, b4 = tau4_prior
        self.register_buffer("a4", torch.tensor(float(a4), device=self.device))
        self.register_buffer("b4", torch.tensor(float(b4), device=self.device))

        self.tau_dec_priors = tau_dec_priors or []
        A = torch.tensor([ab[0] for ab in self.tau_dec_priors], device=self.device, dtype=torch.float32)
        B = torch.tensor([ab[1] for ab in self.tau_dec_priors], device=self.device, dtype=torch.float32)
        self.register_buffer("a_dec", A)   # [J] (possibly empty)
        self.register_buffer("b_dec", B)   # [J]

        # ---- Variational params for s = log(tau^2) ----
        # Transformation layer
        self.m_s_tau4 = nn.Parameter(torch.tensor(float(init_scale_logtau), device=self.device))
        self.r_s_tau4 = nn.Parameter(torch.tensor(math.log(math.exp(0.5) - 1.0), device=self.device))

        # Decorrelation blocks
        J = len(self.tau_dec_priors)
        if J > 0:
            self.m_s_tau_dec = nn.Parameter(torch.full((J,), float(init_scale_logtau), device=self.device))
            self.r_s_tau_dec = nn.Parameter(torch.full((J,), math.log(math.exp(0.5) - 1.0), device=self.device))
        else:
            # register "empty" tensors to keep code simple
            self.register_buffer("m_s_tau_dec", torch.empty(0, device=self.device))
            self.register_buffer("r_s_tau_dec", torch.empty(0, device=self.device))

        # ---- Optional AR(1) layer (off by default) ----
        if self.use_ar1:
            if ar1_sigma_prior is None or ar1_rho_prior is None:
                raise ValueError("AR(1) selected but priors not provided.")
            aS, bS = ar1_sigma_prior
            aR, bR = ar1_rho_prior
            self.register_buffer("a_sigma", torch.tensor(float(aS), device=self.device))
            self.register_buffer("b_sigma", torch.tensor(float(bS), device=self.device))
            self.register_buffer("a_rho", torch.tensor(float(aR), device=self.device))
            self.register_buffer("b_rho", torch.tensor(float(bR), device=self.device))

            # log(sigma^2) variational params
            self.m_s_sig = nn.Parameter(torch.tensor(0.0, device=self.device))
            self.r_s_sig = nn.Parameter(torch.tensor(math.log(math.exp(0.5) - 1.0), device=self.device))
            # rho via tanh(eta), eta ~ N
            self.m_eta = nn.Parameter(torch.tensor(0.0, device=self.device))
            self.r_eta = nn.Parameter(torch.tensor(math.log(math.exp(0.5) - 1.0), device=self.device))

        # standard normals for reparam sampling
        self._stdN1 = Normal(torch.tensor(0.0, device=self.device), torch.tensor(1.0, device=self.device))

    # ----- helpers -----
    @staticmethod
    def _softplus(x: Tensor) -> Tensor:
        return nn.functional.softplus(x) + 1e-6

    def _sample_logtau(self, m: Tensor, r: Tensor, S: int) -> tuple[Tensor, Tensor]:
        """Sample s=log(tau^2) and tau^2 under q(s)=N(m, softplus(r)^2). Returns (s, tau2) with shape [S, *]."""
        std = self._softplus(r)
        eps = self._stdN1.sample((S,) + m.shape)
        s = m + std * eps
        return s, torch.exp(s)

    def _logq_normal(self, x: Tensor, m: Tensor, r: Tensor) -> Tensor:
        """Elementwise log q for Normal(m, softplus(r)). Broadcast over batch dimension."""
        std = self._softplus(r)
        var = std * std
        return -0.5 * (math.log(2 * math.pi) + torch.log(var) + ((x - m) ** 2) / var)

    def _log_prior_IG_in_logspace(self, s: Tensor, a: Tensor | float, b: Tensor | float) -> Tensor:
        """
        For tau^2 ~ IG(a,b), with s=log(tau^2): log p(s) = const + (-a)*s - b*exp(-s).
        (const drops in ELBO, so we only keep the s-dependent part.)
        Shapes broadcast over s.
        """
        return (-torch.as_tensor(a, device=s.device, dtype=s.dtype)) * s - torch.as_tensor(b, device=s.device, dtype=s.dtype) * torch.exp(-s)

    def _inject_taus_into_hparams(self, hT: dict, hD: dict, tau4_2: float | Tensor, tau_dec_2_list: list[float | Tensor]):
        """
        Return DEEP COPIES of hyperparam dicts with the current sampled tau-values inserted.
        Adapt the keys here to match your model's structure if needed.
        Expected:
            hT["RW2"]["tau"] = scalar
            hD has blocks, e.g. hD["RW1"]["tau"], hD["RW2"]["tau"], ...
        """
        import copy
        T = copy.deepcopy(hT) if hT is not None else {}
        D = copy.deepcopy(hD) if hD is not None else {}

        # Transformation layer tau
        if "RW2" in T:
            T["RW2"]["tau"] = tau4_2
        elif "rw2" in T:
            T["rw2"]["tau"] = tau4_2

        # Decorrelation layer taus (map in order of appearance)
        # We walk through D's subdicts and fill 'tau' found keys
        dec_blocks = [k for k in D.keys()]
        j = 0
        for k in dec_blocks:
            if isinstance(D[k], dict) and "tau" in D[k] and j < len(tau_dec_2_list):
                D[k]["tau"] = tau_dec_2_list[j]
                j += 1
        return T, D

    # ----- main step -----
    def step(
        self,
        samples: Tensor,
        hyperparameter_transformation: dict,
        hyperparameter_decorrelation: dict,
        model: "GTM",
        sample_size: int,
        mcmc_samples: int = 32,
        seed: int | None = None,
    ):
        """
        One stochastic-ELBO step with FULL Bayes over tau's (and optional AR(1) if enabled).
        We compute:
          loss = mean_s [ log q(theta,tau,...) - log p(y,theta,tau,...) ]  (up to const)
               = mean_s [ (log q_theta + log q_latents) - ( -neg_posterior(model|taus) + log p(tau) + ... ) ]
        """
        if seed is not None:
            torch.manual_seed(seed)

        # ===== sample θ ~ q =====
        thetas = self.sample_theta(mcmc_samples)      # [S, D]
        logq_theta = self.log_q(thetas)               # [S]

        # ===== sample tau's under q(s)=Normal =====
        s_tau4, tau4_2 = self._sample_logtau(self.m_s_tau4, self.r_s_tau4, mcmc_samples)  # [S], [S]
        logq_s_tau4 = self._logq_normal(s_tau4, self.m_s_tau4, self.r_s_tau4).reshape(-1)  # [S]
        logp_s_tau4 = self._log_prior_IG_in_logspace(s_tau4, self.a4, self.b4).reshape(-1) # [S]

        J = len(self.tau_dec_priors)
        if J > 0:
            s_tau_dec, tau_dec_2 = self._sample_logtau(self.m_s_tau_dec, self.r_s_tau_dec, mcmc_samples)  # [S,J], [S,J]
            logq_s_tau_dec = self._logq_normal(s_tau_dec, self.m_s_tau_dec, self.r_s_tau_dec).sum(-1)     # [S]
            # elementwise IG prior in log-space, sum across J
            logp_s_tau_dec = self._log_prior_IG_in_logspace(s_tau_dec, self.a_dec, self.b_dec).sum(-1)    # [S]
        else:
            tau_dec_2 = torch.empty((mcmc_samples, 0), device=self.device)
            logq_s_tau_dec = torch.zeros(mcmc_samples, device=self.device)
            logp_s_tau_dec = torch.zeros(mcmc_samples, device=self.device)

        # ===== optional AR(1) =====
        if self.use_ar1:
            # sigma^2
            s_sig, sig2 = self._sample_logtau(self.m_s_sig, self.r_s_sig, mcmc_samples)     # [S], [S]
            logq_s_sig = self._logq_normal(s_sig, self.m_s_sig, self.r_s_sig).reshape(-1)
            logp_s_sig = self._log_prior_IG_in_logspace(s_sig, self.a_sigma, self.b_sigma).reshape(-1)

            # rho = tanh(eta), eta~N
            std_eta = self._softplus(self.r_eta)
            eps_eta = self._stdN1.sample((mcmc_samples,))
            eta = self.m_eta + std_eta * eps_eta                 # [S]
            rho = torch.tanh(eta)                                # [S] in (-1,1)
            # q(eta) normal
            logq_eta = self._logq_normal(eta, self.m_eta, self.r_eta).reshape(-1)
            # prior on rho: Beta(a_rho,b_rho) on u=(rho+1)/2 with Jacobian
            u = 0.5 * (rho + 1.0)
            logp_rho = (
                (self.a_rho - 1.0) * torch.log(u.clamp_min(1e-8)) +
                (self.b_rho - 1.0) * torch.log((1 - u).clamp_min(1e-8)) -
                math.log(2.0)  # Jacobian du/drho = 1/2
            )
        else:
            sig2 = None
            rho  = None
            logq_s_sig = logp_s_sig = logq_eta = logp_rho = torch.zeros(mcmc_samples, device=self.device)

        # ===== evaluate model (likelihood + priors on theta | tau) =====
        log_p_tilde_vals = []        # -neg_posterior + hyperpriors
        nll_list, ndec_list, ntrans_list = [], [], []

        for s in range(mcmc_samples):
            # patch sampled taus into the hyperparameters dicts
            tau4_val = tau4_2[s].item() if tau4_2.dim()==1 else float(tau4_2[s])
            tau_dec_vals = [float(tau_dec_2[s, j]) for j in range(tau_dec_2.shape[1])] if J>0 else []

            hT_s, hD_s = self._inject_taus_into_hparams(
                hyperparameter_transformation, hyperparameter_decorrelation,
                tau4_val, tau_dec_vals
            )

            # (optional) inject AR(1) params if your model reads them from hyperparams
            # e.g., hD_s["AR1"] = {"sigma2": float(sig2[s]), "rho": float(rho[s])}

            # reparametrize model weights with current theta sample
            theta_s = thetas[s]
            params_s = self._theta_to_state_dict(theta_s)

            with _reparametrize_module(self.model, params_s):
                out = model.__bayesian_training_objective__(
                    samples=samples,
                    hyperparameters_transformation=hT_s,
                    hyperparameters_decorrelation=hD_s,
                    sample_size=sample_size
                )

            neglogpost_theta_given_tau = out["neg_posterior"]       # NLL + penalty(θ|τ,...) (+ consts internal to your model)
            # add *hyperprior* contributions for tau's (and AR if active)
            # remember: we operate in log-space priors already computed above.
            extra = logp_s_tau4[s] + logp_s_tau_dec[s]
            if self.use_ar1:
                extra = extra + logp_s_sig[s] + logp_rho[s]

            log_p_tilde = -neglogpost_theta_given_tau + extra
            log_p_tilde_vals.append(log_p_tilde.reshape(()))

            # bookkeeping
            nll_list.append(out["negative_log_lik"].reshape(()))
            ndec_list.append(out["negative_decorrelation_prior"].reshape(()))
            ntrans_list.append(out["negative_transformation_prior"].reshape(()))

        log_p_tilde_vals = torch.stack(log_p_tilde_vals)  # [S]

        # ===== total log q over all factors =====
        logq_total = logq_theta + logq_s_tau4 + logq_s_tau_dec
        if self.use_ar1:
            logq_total = logq_total + logq_s_sig + logq_eta

        # ===== ELBO (negated) =====
        loss = torch.mean(logq_total - log_p_tilde_vals)

        # tracking
        nll = torch.stack(nll_list).mean().detach()
        ndp = torch.stack(ndec_list).mean().detach()
        ntp = torch.stack(ntrans_list).mean().detach()

        return {
            "loss": loss,
            "mean_log_q": torch.mean(logq_total).detach(),
            "mean_log_p_tilde": torch.mean(log_p_tilde_vals).detach(),
            "sigma_mean": self.sigma.mean().detach(),
            "neg_log_likelihood": nll,
            "neg_prior_decorrelation": ndp,
            "neg_prior_transformation": ntp,
            # optional: expose current E_q[tau^2] for monitoring
            "E_tau4_2": torch.exp(self.m_s_tau4 + 0.5 * self._softplus(self.r_s_tau4)**2).detach(),
            "E_tau_dec_2": (
                torch.exp(self.m_s_tau_dec + 0.5 * self._softplus(self.r_s_tau_dec)**2).detach()
                if len(self.tau_dec_priors) > 0 else torch.tensor([], device=self.device)
            ),
        }
