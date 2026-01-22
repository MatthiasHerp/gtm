# ============================================================
#  Drop-in optimized Bayesian KLD computation (copy/paste)
#  Main changes:
#   1) Cache quadrature grids (no repeated getQuad/transformQuad)
#   2) Chunk pairs and evaluate many pairs per likelihood call
#   3) Vectorize KLD trimming (0.99 abs-quantile) across pairs
# ============================================================

from __future__ import annotations
import time
from typing import Optional, Tuple, Dict

import numpy as np
import torch
from tqdm import tqdm
from typing import TYPE_CHECKING, Optional
from torch.nn.utils.stateless import _reparametrize_module


from gtm.gtm_plots_analysis.nd_quad import getQuad, transformQuad  # you already import these
from gtm.gtm_plots_analysis.compute_precision_matrix_summary_statistics import *
from gtm.gtm_plots_analysis.independence_kld_process_row import *

if TYPE_CHECKING:

    from gtm_model.gtm import GTM
    from gtm_training.training_bayes.variational_model_estimator import TauPack, VI_Model


# -------------------------
# Quadrature cache (global)
# -------------------------
_QUAD_CACHE: Dict[tuple, Tuple[torch.Tensor, torch.Tensor]] = {}

def _get_quad_tensors(
    *,
    num_points_quad: int,
    min_val: float,
    max_val: float,
    ndim: int,
    device: torch.device,
    dtype: torch.dtype,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Returns:
      X: [Q, ndim]
      W: [Q]
    Cached by (ndim, npoints, min, max, device, dtype).
    """
    key = (ndim, int(num_points_quad), float(min_val), float(max_val), str(device), str(dtype))
    if key in _QUAD_CACHE:
        return _QUAD_CACHE[key]

    limits = [[min_val, max_val]] * ndim
    x, w = getQuad(num_points_quad, ndim=ndim)
    x, w = transformQuad(x, w, limits)

    X = torch.as_tensor(x, device=device, dtype=dtype)  # [Q, ndim]
    W = torch.as_tensor(w, device=device, dtype=dtype)  # [Q]
    _QUAD_CACHE[key] = (X, W)
    return X, W


# --------------------------------------
# Vectorized GLQ integrals for many pairs
# --------------------------------------
@torch.no_grad()
def _log_int_out_two_cols_many_pairs(
    model,
    data: torch.Tensor,          # [N, D]
    col_i: torch.Tensor,         # [P] (int64)
    col_j: torch.Tensor,         # [P] (int64)
    X2: torch.Tensor,            # [Q2, 2]
    W2: torch.Tensor,            # [Q2]
) -> torch.Tensor:
    """
    Computes log ∫∫ f(y with cols i,j replaced by grid) dy_i dy_j
    for many pairs simultaneously.

    Returns:
      log_integral: [P, N]
    """
    # shapes
    N, D = data.shape
    P = col_i.numel()
    Q2 = X2.shape[0]

    # Base: [P, N, Q2, D]
    base = data[None, :, None, :].expand(P, N, Q2, D).clone()

    # Fill varying columns with X2
    # base[p, n, q, col_i[p]] = X2[q,0]
    # base[p, n, q, col_j[p]] = X2[q,1]
    base.scatter_(3, col_i[:, None, None, None].expand(P, N, Q2, 1), X2[None, None, :, 0:1].expand(P, N, Q2, 1))
    base.scatter_(3, col_j[:, None, None, None].expand(P, N, Q2, 1), X2[None, None, :, 1:2].expand(P, N, Q2, 1))

    flat = base.reshape(P * N * Q2, D)
    ll = model.log_likelihood(flat)           # [P*N*Q2]
    dens = torch.exp(ll).reshape(P, N, Q2)    # [P,N,Q2]

    # weighted integral over Q2
    integral = (dens * W2[None, None, :]).sum(dim=2)  # [P,N]
    # numerical safety
    integral = torch.clamp(integral, min=torch.finfo(integral.dtype).tiny)
    return torch.log(integral)


@torch.no_grad()
def _log_int_out_one_col_many_pairs(
    model,
    data: torch.Tensor,          # [N, D]
    col_k: torch.Tensor,         # [P] (int64)
    X1: torch.Tensor,            # [Q1, 1]
    W1: torch.Tensor,            # [Q1]
) -> torch.Tensor:
    """
    Computes log ∫ f(y with col k replaced by grid) dy_k
    for many columns (here: one per pair) simultaneously.

    Returns:
      log_integral: [P, N]
    """
    N, D = data.shape
    P = col_k.numel()
    Q1 = X1.shape[0]

    base = data[None, :, None, :].expand(P, N, Q1, D).clone()
    base.scatter_(3, col_k[:, None, None, None].expand(P, N, Q1, 1), X1[None, None, :, 0:1].expand(P, N, Q1, 1))

    flat = base.reshape(P * N * Q1, D)
    ll = model.log_likelihood(flat)
    dens = torch.exp(ll).reshape(P, N, Q1)

    integral = (dens * W1[None, None, :]).sum(dim=2)  # [P,N]
    integral = torch.clamp(integral, min=torch.finfo(integral.dtype).tiny)
    return torch.log(integral)


@torch.no_grad()
def _ci_log_dists_many_pairs_glq(
    model,
    data: torch.Tensor,          # [N,D]
    col_i: torch.Tensor,         # [P]
    col_j: torch.Tensor,         # [P]
    X1: torch.Tensor, W1: torch.Tensor,
    X2: torch.Tensor, W2: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Vectorized version of compute_ci_probability_deviance_two_dim_glq for many pairs.
    Returns:
      actual_log_distribution_glq: [P, N]
      under_ci_assumption_log_distribution_glq: [P, N]
    """
    # p1 = log f(Y) (same for all pairs)
    p1 = model.log_likelihood(data)  # [N]

    # p2 = log f(Y_{/ij}) by integrating out i and j
    p2 = _log_int_out_two_cols_many_pairs(model, data, col_i, col_j, X2, W2)  # [P,N]

    # p3 = log f(Y_i, Y_j | Y_{/ij})
    p3 = p1[None, :] - p2  # [P,N]

    # p4 = log f(Y_i | Y_{/ij}) = log f(Y_i, Y_{/ij}) - log f(Y_{/ij})
    #      where log f(Y_i, Y_{/ij}) integrates out Y_j => integrate out col_j
    p4_num = _log_int_out_one_col_many_pairs(model, data, col_j, X1, W1)  # [P,N]
    p4 = p4_num - p2

    # p5 = log f(Y_j | Y_{/ij}) integrates out col_i
    p5_num = _log_int_out_one_col_many_pairs(model, data, col_i, X1, W1)  # [P,N]
    p5 = p5_num - p2

    actual = p3
    under  = p4 + p5
    return actual, under


# --------------------------------------
# Evaluation data helper (your same logic)
# --------------------------------------
def get_evaluation_data(
    self: "GTM",
    VI_Model: "VI_Model",
    y,
    evaluation_data_type,
    sample_size,
    copula_only,
    min_val,
    max_val,
    tau_nodes,
    VI_predective_sampling,
    device
):
    if evaluation_data_type == "data":
        if y is None:
            raise ValueError("y must be provided when evaluation_data_type='data'")
        evaluation_data = y[:sample_size].to(device)
        if copula_only:
            evaluation_data = self.after_transformation(evaluation_data)

    elif evaluation_data_type == "uniform_random_samples":
        evaluation_data = torch.distributions.Uniform(min_val, max_val).sample(
            [sample_size, self.y_train.size(1)]
        ).to(device)

    elif evaluation_data_type == "samples_from_model":
        evaluation_data = VI_Model.predictive_sample(
            model=self,
            hyperparameter_decorrelation=self.hyperparameter["decorrelation"],
            hyperparameter_transformation=self.hyperparameter["transformation"],
            n_samples=sample_size,
            tau_nodes=tau_nodes,
            S=VI_predective_sampling
        )
    else:
        raise ValueError(f"evaluation_data_type {evaluation_data_type} not understood.")
    return evaluation_data


def ci_from_samples(t, q_lo=0.025, q_hi=0.975):
    qs = torch.quantile(t, torch.tensor([q_lo, q_hi], device=t.device), dim=0)
    return qs[0].tolist(), qs[1].tolist()


# ---------------------------------------------------------
# Main: optimized Bayesian KLD conditional independence table
# ---------------------------------------------------------
def compute_conditional_independence_kld_bayesian(
    self: "GTM",
    vi_model: "VI_Model",
    y=None,
    evaluation_data_type="data",
    num_processes=1,          # Not used
    tau_nodes: Optional["TauPack"] = None,
    sample_size=1000,
    num_points_quad=20,
    optimized=True,           # now means "use vectorized+cached"
    copula_only=False,
    min_val=-5.0,
    max_val=5.0,
    likelihood_based_metrics=True,
    S_posterior: int = 32,
    S_posterior_predictive_sampling: int = 32,
    cred_level: float = 0.95,
    pair_chunk_size: int = 16,     # <-- NEW: chunk pairs
):
    """
    Bayesian version (optimized):
      - Cache quadrature grids
      - Chunk pairs and evaluate KLD for many pairs per likelihood call
      - Vectorize trimming for KLD (0.99 abs-quantile) across pairs
    """

    device = next(self.parameters()).device
    dtype = torch.float32  # use model/data dtype if you prefer

    # 0) Setup evaluation data
    evaluation_data = get_evaluation_data(
        self,
        vi_model,
        y,
        evaluation_data_type,
        sample_size,
        copula_only,
        min_val,
        max_val,
        tau_nodes=tau_nodes,
        VI_predective_sampling=S_posterior_predictive_sampling,
        device=device
    ).contiguous()

    old_num_trans_layers = None
    if copula_only:
        old_num_trans_layers = self.num_trans_layers
        self.num_trans_layers = 0

    # Build pair list once
    with torch.no_grad():
        precision_ref = self.compute_pseudo_precision_matrix(evaluation_data).detach().cpu()
    precision_summary = compute_precision_matrix_summary_statistics(precision_ref)
    n_pairs = precision_summary.shape[0]

    pair_index = torch.as_tensor(
        precision_summary[["var_row", "var_col"]].to_numpy(dtype=np.int64),
        device=device,
        dtype=torch.long
    )  # [n_pairs, 2]

    # Cache quadrature tensors once
    X1, W1 = _get_quad_tensors(
        num_points_quad=num_points_quad,
        min_val=min_val, max_val=max_val,
        ndim=1, device=device, dtype=dtype
    )
    X2, W2 = _get_quad_tensors(
        num_points_quad=num_points_quad,
        min_val=min_val, max_val=max_val,
        ndim=2, device=device, dtype=dtype
    )

    # Storage: preallocate [S_posterior, n_pairs]
    metrics_kld = None
    if likelihood_based_metrics:
        metrics_kld = torch.empty((S_posterior, n_pairs), dtype=torch.float32, device="cpu")

    alpha = 1.0 - cred_level
    q_lo = alpha / 2.0
    q_hi = 1.0 - alpha / 2.0

    # Draw posterior thetas
    thetas = vi_model.sample_theta(S_posterior, antithetic=True)

    start_total = time.time()
    bar = tqdm(
        range(S_posterior),
        total=S_posterior,
        desc="[Bayesian CI] posterior draws",
        unit="draw",
        dynamic_ncols=True,
        smoothing=0.05,
        leave=True,
    )

    kld_running_sum = 0.0

    for s in bar:
        draw_t0 = time.time()
        theta_s = thetas[s]
        params_s = vi_model._theta_to_state_dict(theta_s)

        # Reparametrize model parameters for this draw
        with _reparametrize_module(self, params_s):

            if likelihood_based_metrics:
                # We fill a GPU tensor of size [n_pairs] then copy once to CPU
                kld_this_draw = torch.empty((n_pairs,), device=device, dtype=torch.float32)

                # process pairs in chunks
                for start in range(0, n_pairs, pair_chunk_size):
                    end = min(start + pair_chunk_size, n_pairs)
                    cols = pair_index[start:end]        # [P,2]
                    col_i = cols[:, 0]
                    col_j = cols[:, 1]

                    # actual/under: [P, N]
                    actual, under = _ci_log_dists_many_pairs_glq(
                        self,
                        evaluation_data,
                        col_i, col_j,
                        X1, W1,
                        X2, W2,
                    )

                    if evaluation_data_type in ("data", "samples_from_model"):
                        ll_dev = actual - under  # [P,N]

                        # remove non-finite safely using NaNs
                        ll_dev = ll_dev.to(torch.float32)
                        finite = torch.isfinite(ll_dev)
                        ll_dev = torch.where(finite, ll_dev, torch.tensor(float("nan"), device=device))

                        # per-pair 0.99 quantile of abs deviation
                        abs_dev = ll_dev.abs()
                        q = torch.nanquantile(abs_dev, 0.99, dim=1)  # [P]

                        keep = abs_dev < q[:, None]
                        keep = keep & torch.isfinite(ll_dev)

                        # mean with masking: sum / count (avoid empty)
                        count = keep.sum(dim=1).clamp_min(1)
                        val = torch.where(keep, ll_dev, torch.zeros_like(ll_dev)).sum(dim=1) / count
                        kld_chunk = torch.nan_to_num(val, nan=0.0, posinf=0.0, neginf=0.0)

                    elif evaluation_data_type == "uniform_random_samples":
                        # KLD ≈ E[ exp(actual) * (actual-under) ]
                        # (your existing formula)
                        ll_dev = torch.exp(actual) * (actual - under)
                        finite = torch.isfinite(ll_dev)
                        ll_dev = torch.where(finite, ll_dev, torch.tensor(float("nan"), device=device))
                        kld_chunk = torch.nanmean(ll_dev, dim=1)
                        kld_chunk = torch.nan_to_num(kld_chunk, nan=0.0, posinf=0.0, neginf=0.0)

                    else:
                        raise ValueError("unknown evaluation_data_type in Bayesian KLD")

                    kld_this_draw[start:end] = kld_chunk

                # copy once to CPU storage
                metrics_kld[s] = kld_this_draw.detach().cpu()

                # progress stats
                kld_draw_mean = float(metrics_kld[s].mean().item())
                kld_running_sum += kld_draw_mean

        dt = time.time() - draw_t0
        postfix = {
            "device": str(device),
            "pairs": n_pairs,
            "sec/draw": f"{dt:.2f}",
        }
        if likelihood_based_metrics:
            postfix["KLD(draw)"] = f"{kld_draw_mean:.4g}"
            postfix["KLD(avg)"]  = f"{(kld_running_sum / (s+1)):.4g}"
        bar.set_postfix(postfix)

    total_sec = time.time() - start_total
    print(
        f"[Bayesian CI] Time taken (device={device}, "
        f"S_post={S_posterior}, pairs={n_pairs}): "
        f"{total_sec:.2f}s  (~{total_sec / float(S_posterior):.2f}s per posterior draw)"
    )

    # Aggregate
    df = precision_summary.copy()

    raw = {}
    if likelihood_based_metrics:
        # metrics_kld: [S, n_pairs] on CPU
        kld_mean = metrics_kld.mean(dim=0).tolist()

        kld_t = metrics_kld.to(torch.float32)  # CPU tensor
        # torch.quantile works on CPU too
        qs = torch.quantile(kld_t, torch.tensor([q_lo, q_hi]), dim=0)
        kld_lo = qs[0].tolist()
        kld_hi = qs[1].tolist()

        df["kld_mean"] = kld_mean
        df["kld_lo"]   = kld_lo
        df["kld_hi"]   = kld_hi

        raw["kld"] = metrics_kld.numpy()

    raw["pair_index"] = precision_summary[["var_row", "var_col"]].to_numpy(dtype=np.int64)

    if copula_only and old_num_trans_layers is not None:
        self.num_trans_layers = old_num_trans_layers

    df.reset_index(inplace=True)
    return df, raw
