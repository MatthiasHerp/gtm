# --- keep your existing imports ---
from __future__ import annotations
import numpy as np
import torch
from torch.utils.data import DataLoader
from dataclasses import dataclass
from typing import Dict, Tuple, List
import pyvinecopulib as pv

from gtm import GTM
from dataset_helpers import Generic_Dataset

# ---------------------
# Utilities & helpers
# ---------------------

def set_seed(seed: int = 0):
    np.random.seed(seed)
    torch.manual_seed(seed)

def to_normal(u: np.ndarray) -> torch.Tensor:
    return torch.distributions.Normal(0, 1).icdf(torch.tensor(u, dtype=torch.float32))

def make_dataloader(x: torch.Tensor, batch_size: int | None = None) -> DataLoader:
    if batch_size is None:
        batch_size = len(x)
    return DataLoader(Generic_Dataset(x), batch_size=batch_size, shuffle=False)

@dataclass
class FitOutputs:
    model: GTM
    history: Dict

# -----------------------------
# Model factories (transform-only)
# -----------------------------

def make_freq_transform_only(device: str = "cpu") -> GTM:
    model = GTM(
        number_variables=2,
        number_transformation_layers=1,
        number_decorrelation_layers=0,
        degree_transformations=10,
        spline_transformation="bspline",
        transformation_spline_range=(-10, 10),
        device=device,
    )
    model.transform_only = True
    return model

def make_bayes_transform_only(device: str = "cpu") -> GTM:
    model = GTM(
        number_variables=2,
        number_transformation_layers=1,
        number_decorrelation_layers=0,
        degree_transformations=10,
        spline_transformation="bspline",
        transformation_spline_range=(-10, 10),
        device=device,
        inference="bayesian",
        hyperparameter={
            "transformation": {
                "sigma_a": 2.1, "sigma_b": 1e6,
                "RW2": {"tau_a": 0.8, "tau_b": 0.3},
                "RW1": {"tau_a": 1.5, "tau_b": 0.01},
            },
            "decorrelation": {
                "sigma_a": 2.1, "sigma_b": 1e6,
                "RW2": {"tau_a": 1.5, "tau_b": 0.01},
                "RW1": {"tau_a": 1.5, "tau_b": 0.01},
            },
        },
    )
    model.transform_only = True
    return model

# ----------------------------------
# Training wrappers (freq and bayes)
# ----------------------------------

def train_frequentist_transform_only(
    model: GTM,
    dl_train: DataLoader,
    dl_val: DataLoader,
    n_trials: int = 30,
    iterations_pretrain: int = 1000,
) -> FitOutputs:
    study = model.hyperparameter_tune_penalties(
        train_dataloader=dl_train,
        validate_dataloader=dl_val,
        penalty_decorrelation_ridge_param=None,
        penalty_decorrelation_ridge_first_difference="sample",
        penalty_decorrelation_ridge_second_difference="sample",
        penalty_transformation_ridge_second_difference=None,
        adaptive_lasso_weights_matrix=False,
        optimizer="LBFGS",
        learning_rate=1.0,
        iterations=2000,
        patience=5,
        min_delta=1e-7,
        seperate_copula_training=False,
        max_batches_per_iter=False,
        pretrained_transformation_layer=True,
        n_trials=n_trials,
        temp_folder=".",
        study_name=None,
    )

    penalty_splines_params = torch.FloatTensor([
        0.0,
        study.best_params["penalty_decorrelation_ridge_first_difference"],
        study.best_params["penalty_decorrelation_ridge_second_difference"],
        0.0,
    ])

    out = model.pretrain_transformation_layer(
        dl_train,
        iterations=iterations_pretrain,
        max_batches_per_iter=False,
        penalty_splines_params=penalty_splines_params,
    )
    return FitOutputs(model=model, history={"study": study, "pretrain": out})

def train_bayesian_transform_only(
    model: GTM,
    dl_train: DataLoader,
    dl_val: DataLoader,
    iterations_pretrain: int = 100,
) -> FitOutputs:
    out = model.pretrain_transformation_layer(
        train_dataloader=dl_train,
        validate_dataloader=dl_val,
        hyperparameters=None,
        iterations=iterations_pretrain,
        learning_rate=0.01,
        mcmc_sample_train=1,
        mcmc_sample_val=4,
        mc_ramp_every=10,
        mc_ramp_max=64,
        patience=15,
        min_delta=1e-3,
        rho_lr_multiplier=2.0,
        sched_factor=0.5,
        sched_patience=6,
        sched_threshold=1e-4,
    )
    return FitOutputs(model=model, history={"pretrain": out})

# -------------------------------------------------
# Extractors: coefficients & VI posterior parameters
# -------------------------------------------------

def get_transformation_params_freq(model: GTM) -> torch.Tensor:
    sd = model.state_dict()
    keys = [k for k in sd.keys() if k.startswith("transformation.params.")]
    if not keys:
        if hasattr(model, "transformation") and hasattr(model.transformation, "params"):
            p = model.transformation.params
            if isinstance(p, torch.Tensor):
                return p.detach().reshape(-1)
            elif isinstance(p, (list, tuple)):
                return torch.cat([pi.detach().reshape(-1) for pi in p])
        raise RuntimeError("No 'transformation.params.*' found in state_dict().")
    keys = sorted(keys, key=lambda s: int(s.rsplit(".", 1)[-1]))
    return torch.cat([sd[k].detach().reshape(-1) for k in keys])

def get_vi_posterior_params_from_output(vi_out: dict) -> Tuple[torch.Tensor, torch.Tensor]:
    if not isinstance(vi_out, dict) or "vi_model" not in vi_out:
        raise RuntimeError("Expected vi_out to be a dict with key 'vi_model'.")
    vi = vi_out["vi_model"]
    if isinstance(vi_out.get("mu", None), torch.Tensor):
        mu_flat = vi_out["mu"].detach().reshape(-1)
    elif hasattr(vi, "mu") and isinstance(vi.mu, torch.Tensor):
        mu_flat = vi.mu.detach().reshape(-1)
    else:
        raise RuntimeError("VI mean vector (mu) not found in output or vi_model.")
    softplus = lambda x: torch.log1p(torch.exp(x))
    if isinstance(vi_out.get("rho", None), torch.Tensor):
        std_flat = softplus(vi_out["rho"]).detach().reshape(-1)
    elif isinstance(vi_out.get("log_sigma", None), torch.Tensor):
        std_flat = vi_out["log_sigma"].exp().detach().reshape(-1)
    elif isinstance(vi_out.get("sigma", None), torch.Tensor):
        std_flat = vi_out["sigma"].abs().detach().reshape(-1)
    elif hasattr(vi, "rho") and isinstance(vi.rho, torch.Tensor):
        std_flat = softplus(vi.rho).detach().reshape(-1)
    elif hasattr(vi, "log_sigma") and isinstance(vi.log_sigma, torch.Tensor):
        std_flat = vi.log_sigma.exp().detach().reshape(-1)
    elif hasattr(vi, "sigma") and isinstance(vi.sigma, torch.Tensor):
        std_flat = vi.sigma.abs().detach().reshape(-1)
    else:
        raise RuntimeError("VI dispersion (rho/log_sigma/sigma) not found in output or vi_model.")

    sd_mu  = vi._theta_to_state_dict(mu_flat)
    sd_std = vi._theta_to_state_dict(std_flat)

    keys = [k for k in sd_mu.keys() if k.startswith("transformation.params.")]
    if not keys:
        if "transformation.params" in sd_mu:
            keys = ["transformation.params"]
        else:
            raise RuntimeError(f"No 'transformation.params.*' in VI schema keys: {list(sd_mu.keys())[:20]} ...")

    def parse_idx(s):
        try:
            return int(s.rsplit(".", 1)[-1])
        except ValueError:
            return 0
    keys = sorted(keys, key=parse_idx)

    mu_t  = torch.cat([sd_mu[k].detach().reshape(-1)  for k in keys])
    std_t = torch.cat([sd_std[k].detach().reshape(-1) for k in keys])
    return mu_t, std_t

# -----------------------------------
# Posterior sanity check (freq vs VI)
# -----------------------------------

def posterior_match_score(theta_freq: torch.Tensor, vi_mu: torch.Tensor, vi_std: torch.Tensor) -> Dict:
    theta_freq = theta_freq.detach().reshape(-1)
    vi_mu = vi_mu.detach().reshape(-1)
    vi_std = vi_std.detach().reshape(-1)
    if not (theta_freq.numel() == vi_mu.numel() == vi_std.numel()):
        raise ValueError("Parameter vector lengths mismatch between frequentist and VI posterior.")
    z = (theta_freq - vi_mu) / (vi_std + 1e-12)
    md2 = torch.dot(z, z).item()
    frac_within_1 = (z.abs() <= 1.0).float().mean().item()
    frac_within_2 = (z.abs() <= 2.0).float().mean().item()
    return {
        "theta_freq": theta_freq,
        "vi_mu": vi_mu,
        "mah_dist_sq": md2,
        "n_params": z.numel(),
        "frac_within_1sigma": frac_within_1,
        "frac_within_2sigma": frac_within_2,
        "z_abs_median": z.abs().median().item(),
    }

# >>> NEW: (1) Analytic HPD mass at θ̂ (plus two-sided χ² p-value)
# >>> replace your chi2_hpd_summary with this
def _chi2_quantile(alpha: float, df: int) -> float:
    # Try SciPy (exact)
    try:
        from scipy.stats import chi2 as _scipy_chi2
        return float(_scipy_chi2.ppf(alpha, df))
    except Exception:
        pass

    # Fallback: Wilson–Hilferty approximation (accurate & fast)
    # q_alpha ≈ df * [1 - 2/(9df) + z_alpha * sqrt(2/(9df))]^3,  z_alpha = Φ^{-1}(alpha)
    z_alpha = torch.distributions.Normal(0.0, 1.0).icdf(torch.tensor(alpha))
    t = 1.0 - 2.0 / (9.0 * df) + z_alpha * torch.sqrt(torch.tensor(2.0 / (9.0 * df)))
    return float((df * (t ** 3)).item())


def chi2_hpd_summary(theta_hat: torch.Tensor, mu: torch.Tensor, sigma: torch.Tensor) -> dict:
    # Mahalanobis radius^2
    z = (theta_hat - mu) / (sigma + 1e-12)
    z2 = torch.dot(z, z)
    d = theta_hat.numel()

    # Percentile under Chi^2_d (cdf is implemented)
    chi2 = torch.distributions.Chi2(df=d)
    u = chi2.cdf(z2)                            # scalar tensor in [0,1]

    # Two-sided p-value for "too small or too large" z^2
    p_two = 2.0 * torch.minimum(u, 1.0 - u)

    # 95% cutoff via robust quantile helper
    cut95 = _chi2_quantile(0.95, d)

    return {
        "z2": float(z2),
        "percentile": float(u),
        "p_two_sided": float(p_two),
        "cut95": float(cut95),
        "in_95_hpd": bool(float(z2) <= cut95),
    }


# >>> NEW: (2) Per-coordinate sign probabilities and joint “all signs match”
def posterior_sign_probs(mu: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
    std_norm = torch.distributions.Normal(0.0, 1.0)
    z = mu / (sigma + 1e-12)
    return std_norm.cdf(z)                          # P(theta_j > 0) = Φ(μ/σ)

def prob_all_same_sign(theta_hat: torch.Tensor, mu: torch.Tensor, sigma: torch.Tensor) -> Dict:
    p_pos = posterior_sign_probs(mu, sigma)         # length d
    mask_pos = (theta_hat > 0)
    mask_neg = (theta_hat < 0)
    # ambiguous zeros are ignored in the product
    p_all = torch.prod(p_pos[mask_pos]).item() * torch.prod((1.0 - p_pos)[mask_neg]).item()
    kept = int(mask_pos.sum() + mask_neg.sum())
    skipped = int(theta_hat.numel() - kept)
    return {"prob_all_same_sign": float(p_all), "coords_used": kept, "coords_skipped_zero": skipped}

# >>> NEW: (3) General event probability under q via Monte Carlo
def posterior_event_prob(mu: torch.Tensor, sigma: torch.Tensor, predicate, S: int = 5000, seed: int | None = 0) -> float:
    """
    Monte Carlo P(predicate(theta)) under mean-field q: theta = mu + sigma * eps, eps ~ N(0,I).
    Works on CPU/GPU, no reliance on randn_like(generator=...).
    """
    # Optional reproducibility
    gen = None
    if seed is not None:
        gen = torch.Generator(device=mu.device)
        gen.manual_seed(seed)

    cnt = 0
    for _ in range(S):
        # randn supports 'generator'; reshape to mu
        eps = torch.randn(mu.numel(), device=mu.device, dtype=mu.dtype, generator=gen).reshape_as(mu)
        th = mu + sigma * eps
        if predicate(th):
            cnt += 1
    return cnt / S
# -----------------------------------
# Distributional diagnostics (identity)
# -----------------------------------

def normality_scores(x: torch.Tensor) -> Dict:
    x = x.detach().cpu()
    means = x.mean(dim=0)
    stds = x.std(dim=0, unbiased=True)
    skews = ((x - means)**3).mean(dim=0) / (stds**3 + 1e-12)
    kurts = ((x - means)**4).mean(dim=0) / (stds**4 + 1e-12) - 3.0
    return {
        "mean": means.numpy().tolist(),
        "std": stds.numpy().tolist(),
        "skew": skews.numpy().tolist(),
        "excess_kurtosis": kurts.numpy().tolist(),
    }

def transform_batch(model: GTM, x: torch.Tensor) -> torch.Tensor:
    if hasattr(model, "transformation_layer") and hasattr(model.transformation_layer, "forward"):
        return model.transformation_layer.forward(x)
    return model.forward(x)['output']

# -----------------
# Main experiment
# -----------------

def run():
    set_seed(42)
    device = "cpu"

    # (A) Independent 2D standard normals
    N_train = 2000
    N_val = 2000
    x_train_A = torch.randn(N_train, 2)
    x_val_A = torch.randn(N_val, 2)
    dl_train_A = make_dataloader(x_train_A, batch_size=N_train)
    dl_val_A = make_dataloader(x_val_A, batch_size=N_val)

    mF_A = make_freq_transform_only(device)
    fitF_A = train_frequentist_transform_only(mF_A, dl_train_A, dl_val_A)

    mB_A = make_bayes_transform_only(device)
    fitB_A = train_bayesian_transform_only(mB_A, dl_train_A, dl_val_A)

    with torch.no_grad():
        yF_A = transform_batch(fitF_A.model, x_val_A)
        yB_A = transform_batch(fitB_A.model, x_val_A)
    diag_F_A = normality_scores(yF_A)
    diag_B_A = normality_scores(yB_A)

    thetaF_A = get_transformation_params_freq(fitF_A.model)
    muB_A, stdB_A = get_vi_posterior_params_from_output(fitB_A.history['pretrain'])
    post_A = posterior_match_score(thetaF_A, muB_A, stdB_A)

    # >>> NEW: compute the 3 posterior probability diagnostics for (A)
    hpd_A = chi2_hpd_summary(post_A["theta_freq"], post_A["vi_mu"], stdB_A)
    sign_probs_A = posterior_sign_probs(muB_A, stdB_A)
    sign_summary_A = {
        "min": float(sign_probs_A.min()),
        "median": float(sign_probs_A.median()),
        "max": float(sign_probs_A.max()),
    }
    prob_same_sign_A = prob_all_same_sign(post_A["theta_freq"], muB_A, stdB_A)
    prob_all_small_A = posterior_event_prob(muB_A, stdB_A, lambda th: (th.abs() <= 0.1).all(), S=5000, seed=123)
    prob_sup_close_A = posterior_event_prob(
        muB_A, stdB_A, lambda th: (th - post_A["theta_freq"]).abs().max() <= 0.1, S=5000, seed=123
    )

    # (B) Joe copula (rot=90) -> probit marginals
    copula = pv.Bicop(family=pv.BicopFamily.joe, parameters=np.array([[2.5]]), rotation=90)
    u_train = copula.simulate(n=N_train)
    u_val = copula.simulate(n=N_val)
    x_train_B = to_normal(u_train)
    x_val_B = to_normal(u_val)
    dl_train_B = make_dataloader(x_train_B, batch_size=N_train)
    dl_val_B = make_dataloader(x_val_B, batch_size=N_val)

    mF_B = make_freq_transform_only(device)
    fitF_B = train_frequentist_transform_only(mF_B, dl_train_B, dl_val_B)

    mB_B = make_bayes_transform_only(device)
    fitB_B = train_bayesian_transform_only(mB_B, dl_train_B, dl_val_B)

    with torch.no_grad():
        yF_B = transform_batch(fitF_B.model, x_val_B)
        yB_B = transform_batch(fitB_B.model, x_val_B)
    diag_F_B = normality_scores(yF_B)
    diag_B_B = normality_scores(yB_B)

    thetaF_B = get_transformation_params_freq(fitF_B.model)
    muB_B, stdB_B = get_vi_posterior_params_from_output(fitB_B.history['pretrain'])
    post_B = posterior_match_score(thetaF_B, muB_B, stdB_B)

    # >>> NEW: compute the 3 posterior probability diagnostics for (B)
    hpd_B = chi2_hpd_summary(post_B["theta_freq"], post_B["vi_mu"], stdB_B)
    sign_probs_B = posterior_sign_probs(muB_B, stdB_B)
    sign_summary_B = {
        "min": float(sign_probs_B.min()),
        "median": float(sign_probs_B.median()),
        "max": float(sign_probs_B.max()),
    }
    prob_same_sign_B = prob_all_same_sign(post_B["theta_freq"], muB_B, stdB_B)
    prob_all_small_B = posterior_event_prob(muB_B, stdB_B, lambda th: (th.abs() <= 0.1).all(), S=5000, seed=456)
    prob_sup_close_B = posterior_event_prob(
        muB_B, stdB_B, lambda th: (th - post_B["theta_freq"]).abs().max() <= 0.1, S=5000, seed=456
    )

    # Pairwise closeness of learned transforms (F vs B): L2 on a grid
    grid = torch.linspace(-3.5, 3.5, 181).reshape(-1, 1)
    grid2 = torch.cat([grid, torch.zeros_like(grid)], dim=1)
    with torch.no_grad():
        gF1 = transform_batch(fitF_A.model, grid2)[:, 0]
        gB1 = transform_batch(fitB_A.model, grid2)[:, 0]
    l2_grid_A_dim1 = torch.mean((gF1 - gB1)**2).item()

    # Summaries
    def fmt(diag):
        return {k: [float(f"{diag[k][i]:+.3f}") for i in range(len(diag[k]))] for k in diag}

    print("=== (A) Independent N(0,1)^2 ===")
    print("FREQ normality:", fmt(diag_F_A))
    print("VI   normality:", fmt(diag_B_A))
    print("Posterior match:", {k: v for k, v in post_A.items() if k != "theta_freq" and k != "vi_mu"})
    print("HPD summary:", hpd_A)
    print("Sign probs summary:", sign_summary_A)
    print("P(all signs match θ̂):", prob_same_sign_A)
    print("P(all |θ_j| ≤ 0.1):", float(prob_all_small_A))
    print("P(max_j |θ_j-θ̂_j| ≤ 0.1):", float(prob_sup_close_A))
    print("Grid L2 (dim1, F vs VI):", l2_grid_A_dim1)

    print("\n=== (B) Joe copula (rot=90), probit marginals ===")
    print("FREQ normality:", fmt(diag_F_B))
    print("VI   normality:", fmt(diag_B_B))
    print("Posterior match:", {k: v for k, v in post_B.items() if k != "theta_freq" and k != "vi_mu"})
    print("HPD summary:", hpd_B)
    print("Sign probs summary:", sign_summary_B)
    print("P(all signs match θ̂):", prob_same_sign_B)
    print("P(all |θ_j| ≤ 0.1):", float(prob_all_small_B))
    print("P(max_j |θ_j-θ̂_j| ≤ 0.1):", float(prob_sup_close_B))

if __name__ == "__main__":
    run()
