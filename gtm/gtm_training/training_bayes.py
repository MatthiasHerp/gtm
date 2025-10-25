### BAYESIAN APPROACH
import torch
import math
from torch.distributions import Normal
from torch.special import digamma as ψ
from torch import nn, Tensor
from typing import TYPE_CHECKING

from gtm.gtm_layers.layer_utils import bayesian_splines
from torch.nn.utils.stateless import _reparametrize_module
from gtm.gtm_model import gtm

if TYPE_CHECKING:
    from ..gtm_model.gtm import GTM # type-only; no runtime import

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
        beta_logp=1.0
    ):
        """
        One stochastic-ELBO step (no optimizer step).
        Returns dict with 'loss' and components for logging.
        """
        
        # Sample θ ~ q
        thetas = self.sample_theta(mcmc_samples, antithetic=True)  # [S, D]
        log_q_vals = self.log_q(thetas)         # [S]

        log_p_tilde_vals = []  # log unnormalized posterior per sample
        neg_likelihood_list = []
        ll_list = []
        
        #Decorrelation
        prior_dec_list = []
        qf1_sum_dec_list = []
        qf2_sum_dec_list = []
        
        #Transformation
        prior_trans_list = []
        qf_neg_prior_trans_list = []
        qf_sum_trans_list=[]
        qf_mean_trans_list=[]
        
        for s in range(mcmc_samples):
            theta_s = thetas[s]
            params_s = self._theta_to_state_dict(theta_s)  # tensors keep graph to (mu, rho)

            with _reparametrize_module(self.model, params_s):
                # Pick one key from the schema that you know we optimize
                name0, _ = self._schema[0]
                live = dict(self.model.named_parameters())[name0]
                want = self._theta_to_state_dict(theta_s)[name0]
                if not torch.allclose(live, want, atol=1e-6, rtol=1e-6):
                    raise RuntimeError(f"Reparam not visible for {name0}")
                
            # Use your provided objective to compute: posterior = NLL + priors
                out = model.__bayesian_training_objective__(
                    samples=samples,
                    hyperparameters_transformation=hyperparameter_transformation,
                    hyperparameters_decorrelation=hyperparameter_decorrelation,
                    N_total=sample_size_total,
                    B=samples.shape[0]
                )
            
            # Your function returns a POSITIVE objective (NLL + priors).
            # log \tilde p(θ, y) = - (NLL + priors)
            neglogpost = out['neg_posterior']
            log_p_tilde = -neglogpost
            log_p_tilde_vals.append(log_p_tilde.reshape(()))
            
            #Likelihood
            neg_likelihood_list.append(out['negative_log_lik'].reshape(()))
            ll_list.append(-out['nll_batch'].reshape(()))
            
            
            #Transformation Layer
            prior_trans_list.append(out['negative_transformation_prior']['neg_log_prior_total'].reshape(()))
            qf_neg_prior_trans_list.append(out['negative_transformation_prior']['neg_log_prior_qf'].reshape(()))
            qf_sum_trans_list.append(out["negative_transformation_prior"]["qf_sum"].reshape(()))
            qf_mean_trans_list.append(out['negative_transformation_prior']['qf_mean'].reshape(()))
            
            #Decorrelation Layer
            prior_dec_list.append(out['negative_decorrelation_prior']['neg_log_prior_total'].reshape(()))
            qf1_sum_dec_list.append(out['negative_decorrelation_prior']['qf1'])
            qf2_sum_dec_list.append(out['negative_decorrelation_prior']['qf2'])

        log_p_tilde_vals = torch.stack(log_p_tilde_vals)  # [S]
        # Monte-Carlo KL(q || p) estimate: E_q[log q - log p̃]
        # (Note: additive constant log p(y) cancels in optimization)
        elbo_loss = torch.mean(beta_kl*log_q_vals - beta_logp*log_p_tilde_vals) #-ELBO (i.e., E_q[log q - log p̃])

        neg_likelihood_list= torch.stack(neg_likelihood_list)
        ll_list=torch.stack(ll_list)
        
        # Decorrelation Layer
        prior_dec_list = torch.stack(prior_dec_list)
        qf1_sum_dec_list = torch.stack(qf1_sum_dec_list)
        qf2_sum_dec_list = torch.stack(qf2_sum_dec_list)
        
        # Transformation Layer
        prior_trans_list = torch.stack(prior_trans_list)
        qf_neg_prior_trans_list = torch.stack(qf_neg_prior_trans_list)
        qf_sum_trans_list= torch.stack(qf_sum_trans_list)
        qf_mean_trans_list = torch.stack(qf_mean_trans_list)
        
        return {
            "elbo_loss": elbo_loss,
            "mean_log_q": torch.mean(log_q_vals).detach(),
            "mean_log_p_tilde": torch.mean(log_p_tilde_vals).detach(),
            "log_likelihhod_batch": float(_logmeanexp(ll_list, dim=0).detach()),
            
            # Variance Tracking
            "sigma_mean": self.sigma.mean().detach(),
            "sigma_max": self.sigma.max().detach(),
            "sigma_min": self.sigma.min().detach(),
            
            #Decorrelation Layer
            "neg_prior_decorrelation": torch.mean(prior_dec_list).detach(),
            "qf1_decorrelation": torch.mean(qf1_sum_dec_list).detach(),
            "qf2_decorrelation": torch.mean(qf2_sum_dec_list).detach(),
            
            #Transformation Layer
            "neg_prior_transformation": torch.mean(prior_trans_list).detach(),
            "transformation_neg_log_prior_df": torch.mean(qf_neg_prior_trans_list).detach(),  #= E[0.5 τ qf]
            "transformation_sum_qf": torch.mean(qf_sum_trans_list).detach(), #qf sum
            "transformation_mean_qf": torch.mean(qf_mean_trans_list).detach()
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
        seed: int | None = None
    ) -> float:
        """For Validation used the ELPD Approach."""
    
        thetas = self.sample_theta(S, antithetic=True)  # [S, D]
        ll_list = []
        for s in range(S):
            
            theta_s = thetas[s]
            params_s = self._theta_to_state_dict(theta_s)
            
            with _reparametrize_module(model, params_s):
                
                out = model.__bayesian_training_objective__(
                    samples=y_batch,
                    hyperparameters_transformation=hyperparameter_transformation,
                    hyperparameters_decorrelation=hyperparameter_decorrelation,
                    N_total=sample_size,
                    B=batch_size
                )
                
                nll = out["negative_log_lik"].reshape(())
                
                ll_list.append(-nll)               # <<< THE MISSING MINUS
        ll = torch.stack(ll_list)                  # [S], each is SUM log-lik for the batch
        return float(_logmeanexp(ll, dim=0))       # SUM log predictive for the batch


class VariationalGamma:
    """q(tau) = Gamma(a_hat, b_hat) (shape-rate) with analytics you need for ELBO."""
    def __init__(self, a0: float, b0: float, rank_total: int):
        self.a0 = float(a0)
        self.b0 = float(b0)
        self.rank_total = int(rank_total)
        # initialize with prior (or a mild update)
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
