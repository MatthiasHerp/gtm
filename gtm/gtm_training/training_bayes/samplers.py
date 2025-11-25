import torch
import math
from torch.nn.utils.stateless import _reparametrize_module

def sample_bayesian_predictive(
    model,
    VI,
    tau_nodes,
    hyper_T,
    hyper_D,
    n_samples: int,
    S: int = 32,
    decor_present: bool = True,
):
    """
    Draw samples from the Bayesian predictive:
        p(y) ≈ (1/S) ∑_s p(y | θ_s, τ_s)
    by:
      - sampling θ_s ~ q(θ)
      - sampling τ_s ~ q(τ) (if tau_nodes is not None)
      - for each s, drawing n_samples/S from model with those parameters
    """
    device = model.device
    S = min(S, n_samples)
    n_per = math.ceil(n_samples / S)

    theta_samples = VI.sample_theta(S, antithetic=True)  # [S, D]

    # sample taus for each s (or fall back to hyperparameters)
    if tau_nodes is not None:
        tau4_vec, tau1_vec, tau2_vec, _ = tau_nodes.sample_many(
            S=S,
            decor_present=decor_present,
            generator=VI._rng,
        )
    else:
        tau4_vec = torch.full((S,), float(hyper_T["tau"]),  device=device)
        if decor_present:
            tau1_vec = torch.full((S,), float(hyper_D["tau_1"]), device=device)
            tau2_vec = torch.full((S,), float(hyper_D["tau_2"]), device=device)
        else:
            tau1_vec = torch.zeros((S,), device=device)
            tau2_vec = torch.zeros((S,), device=device)

    samples_all = []

    for s in range(S):
        theta_s = theta_samples[s]

        params_s = VI._theta_to_state_dict(theta_s)

        with _reparametrize_module(model, params_s):
            # if your sample() does not take taus, you can instead
            # stash them into model.hyperparameter[...] before calling sample
            model.hyperparameter["transformation"]["tau"] = float(tau4_vec[s].item())
            if decor_present:
                model.hyperparameter["decorrelation"]["tau_1"] = float(tau1_vec[s].item())
                model.hyperparameter["decorrelation"]["tau_2"] = float(tau2_vec[s].item())

            y_s = model.sample(n_per)  # shape [n_per, d]
            samples_all.append(y_s)

    y_cat = torch.cat(samples_all, dim=0)
    return y_cat[:n_samples]


def log_prob_bayesian_mixture(
    model,
    VI,
    tau_nodes,
    hyper_T,
    hyper_D,
    y: torch.Tensor, # [N, d]
    S: int = 32,
    decor_present: bool = True,
):
    """
    Compute log p(y) under the Bayesian predictive via log-mean-exp over θ, τ.
    """
    device = model.device
    N = y.shape[0]

    theta_samples = VI.sample_theta(S, antithetic=True)  # [S, D]

    if tau_nodes is not None:
        tau4_vec, tau1_vec, tau2_vec, _ = tau_nodes.sample_many(
            S=S,
            decor_present=decor_present,
            generator=VI._rng,
        )
    else:
        tau4_vec = torch.full((S,), float(hyper_T["tau"]),  device=device)
        if decor_present:
            tau1_vec = torch.full((S,), float(hyper_D["tau_1"]), device=device)
            tau2_vec = torch.full((S,), float(hyper_D["tau_2"]), device=device)
        else:
            tau1_vec = torch.zeros((S,), device=device)
            tau2_vec = torch.zeros((S,), device=device)

    logps = []

    for s in range(S):
        theta_s = theta_samples[s]
        params_s = VI._theta_to_state_dict(theta_s)

        with _reparametrize_module(model, params_s):
            model.hyperparameter["transformation"]["tau"] = float(tau4_vec[s].item())
            if decor_present:
                model.hyperparameter["decorrelation"]["tau_1"] = float(tau1_vec[s].item())
                model.hyperparameter["decorrelation"]["tau_2"] = float(tau2_vec[s].item())

            # adapt this to your GTM's density API
            # e.g. if you have model.log_prob(y):
            ll_s = model.log_prob(y)     # [N]
            logps.append(ll_s)

    logps = torch.stack(logps, dim=0)   # [S, N]

    # log mean exp over S
    m, _ = torch.max(logps, dim=0, keepdim=True)
    log_mean = m.squeeze(0) + torch.log(torch.mean(torch.exp(logps - m), dim=0))  # [N]

    return log_mean  # [N]
