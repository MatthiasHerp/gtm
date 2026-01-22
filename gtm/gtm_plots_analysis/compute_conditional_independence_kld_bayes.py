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


# ============================================================
# Memory-safe GLQ integrals (stream over quadrature points)
# ============================================================

@torch.no_grad()
def _log_int_out_one_col_stream(
    model,
    data: torch.Tensor,      # [N, D]
    col_k: torch.Tensor,     # [P] int64
    X1: torch.Tensor,        # [Q1, 1]
    W1: torch.Tensor,        # [Q1]
    q_microbatch: int = 64,  # number of quadrature points per microbatch
) -> torch.Tensor:
    """
    Computes for each pair p and each row n:
      log ∫ f( y with col_k[p] replaced by x ) dx
    without allocating [P,N,Q,D].

    Returns: [P, N]
    """
    N, D = data.shape
    P = col_k.numel()
    Q1 = X1.shape[0]

    # replicate base for P pairs: [P,N,D] (manageable)
    base = data.unsqueeze(0).expand(P, N, D).contiguous()
    flat_base = base.reshape(P * N, D)  # [P*N, D]

    # accumulate integral in probability space
    integral = torch.zeros((P * N,), device=data.device, dtype=torch.float32)

    # micro-batch quadrature points to control peak memory
    for s in range(0, Q1, q_microbatch):
        e = min(s + q_microbatch, Q1)
        xb = X1[s:e, 0]         # [B]
        wb = W1[s:e]            # [B]
        B = xb.numel()

        # expand to [B, P*N, D] as a view then materialize only [B*P*N, D]
        xrep = xb[:, None].expand(B, P * N)  # [B, P*N]
        xrep_flat = xrep.reshape(-1)         # [B*P*N]

        y = flat_base.unsqueeze(0).expand(B, P * N, D).reshape(B * P * N, D).clone()
        # set varying column (different per pair): need per-row mapping of which column to set
        # We do it by indexing per pair block.
        # Create an index vector of length P*N telling which pair each row belongs to:
        pair_id = torch.arange(P, device=data.device).repeat_interleave(N)  # [P*N]
        col_for_row = col_k[pair_id]                                        # [P*N]
        col_for_row = col_for_row.unsqueeze(0).expand(B, P * N).reshape(-1) # [B*P*N]

        y[torch.arange(B * P * N, device=data.device), col_for_row] = xrep_flat

        ll = model.log_likelihood(y)               # [B*P*N]
        dens = torch.exp(ll).to(torch.float32)     # [B*P*N]

        # weight each quadrature point
        wrep = wb[:, None].expand(B, P * N).reshape(-1)  # [B*P*N]
        integral += (dens * wrep).reshape(B, P * N).sum(dim=0)

        # free temp
        del y, ll, dens

    integral = torch.clamp(integral, min=torch.finfo(torch.float32).tiny)
    return torch.log(integral).reshape(P, N)


@torch.no_grad()
def _log_int_out_two_cols_stream(
    model,
    data: torch.Tensor,      # [N, D]
    col_i: torch.Tensor,     # [P]
    col_j: torch.Tensor,     # [P]
    X2: torch.Tensor,        # [Q2, 2]
    W2: torch.Tensor,        # [Q2]
    q_microbatch: int = 16,  # for Q2=400, 16/32 is typical
) -> torch.Tensor:
    """
    Computes:
      log ∫∫ f( y with cols i,j replaced by (x1,x2) ) dx1 dx2
    for many pairs, streaming over quadrature points.

    Returns: [P, N]
    """
    N, D = data.shape
    P = col_i.numel()
    Q2 = X2.shape[0]

    base = data.unsqueeze(0).expand(P, N, D).contiguous()
    flat_base = base.reshape(P * N, D)  # [P*N, D]

    integral = torch.zeros((P * N,), device=data.device, dtype=torch.float32)

    pair_id = torch.arange(P, device=data.device).repeat_interleave(N)  # [P*N]
    col_i_row = col_i[pair_id]                                          # [P*N]
    col_j_row = col_j[pair_id]                                          # [P*N]

    for s in range(0, Q2, q_microbatch):
        e = min(s + q_microbatch, Q2)
        Xb = X2[s:e, :]  # [B,2]
        Wb = W2[s:e]     # [B]
        B = Xb.shape[0]

        x1 = Xb[:, 0]
        x2 = Xb[:, 1]

        x1rep = x1[:, None].expand(B, P * N).reshape(-1)  # [B*P*N]
        x2rep = x2[:, None].expand(B, P * N).reshape(-1)

        y = flat_base.unsqueeze(0).expand(B, P * N, D).reshape(B * P * N, D).clone()

        # expand column indices to [B*P*N]
        col_i_rep = col_i_row.unsqueeze(0).expand(B, P * N).reshape(-1)
        col_j_rep = col_j_row.unsqueeze(0).expand(B, P * N).reshape(-1)

        idx = torch.arange(B * P * N, device=data.device)
        y[idx, col_i_rep] = x1rep
        y[idx, col_j_rep] = x2rep

        ll = model.log_likelihood(y)
        dens = torch.exp(ll).to(torch.float32)

        wrep = Wb[:, None].expand(B, P * N).reshape(-1)
        integral += (dens * wrep).reshape(B, P * N).sum(dim=0)

        del y, ll, dens

    integral = torch.clamp(integral, min=torch.finfo(torch.float32).tiny)
    return torch.log(integral).reshape(P, N)


@torch.no_grad()
def _ci_log_dists_many_pairs_glq_stream(
    model,
    data: torch.Tensor,      # [N,D]
    col_i: torch.Tensor,     # [P]
    col_j: torch.Tensor,     # [P]
    X1: torch.Tensor, W1: torch.Tensor,
    X2: torch.Tensor, W2: torch.Tensor,
    qmb_1d: int = 64,
    qmb_2d: int = 16,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Returns:
      actual: [P,N]
      under : [P,N]
    """
    p1 = model.log_likelihood(data)  # [N]

    p2 = _log_int_out_two_cols_stream(model, data, col_i, col_j, X2, W2, q_microbatch=qmb_2d)  # [P,N]

    p3 = p1[None, :] - p2

    p4_num = _log_int_out_one_col_stream(model, data, col_j, X1, W1, q_microbatch=qmb_1d)      # [P,N]
    p4 = p4_num - p2

    p5_num = _log_int_out_one_col_stream(model, data, col_i, X1, W1, q_microbatch=qmb_1d)      # [P,N]
    p5 = p5_num - p2

    actual = p3
    under = p4 + p5
    return actual, under


# --------------------------------------
# Your evaluation data helper (unchanged)
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


# ---------------------------------------------------------
# Main function: memory-safe chunked+streaming version
# ---------------------------------------------------------
def compute_conditional_independence_kld_bayesian(
    self: "GTM",
    vi_model: "VI_Model",
    y=None,
    evaluation_data_type="data",
    num_processes=1,
    tau_nodes: Optional["TauPack"] = None,
    sample_size=1000,
    num_points_quad=20,
    optimized=True,
    copula_only=False,
    min_val=-5.0,
    max_val=5.0,
    likelihood_based_metrics=True,
    S_posterior: int = 32,
    S_posterior_predictive_sampling: int = 32,
    cred_level: float = 0.95,
    pair_chunk_size: int = 4,      # <<< IMPORTANT: keep small
    q_microbatch_1d: int = 64,     # 1D quadrature microbatch
    q_microbatch_2d: int = 8,      # 2D quadrature microbatch (smaller!)
):
    """
    OOM-safe Bayesian CI KLD:
      - cache quadrature
      - chunk pairs
      - stream over quadrature points (no [P,N,Q,D] tensor)
    """

    device = next(self.parameters()).device
    dtype = torch.float32

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

    with torch.no_grad():
        precision_ref = self.compute_pseudo_precision_matrix(evaluation_data).detach().cpu()
    precision_summary = compute_precision_matrix_summary_statistics(precision_ref)
    n_pairs = precision_summary.shape[0]

    pair_index = torch.as_tensor(
        precision_summary[["var_row", "var_col"]].to_numpy(dtype=np.int64),
        device=device,
        dtype=torch.long
    )  # [n_pairs, 2]

    # cache quadrature
    X1, W1 = _get_quad_tensors(
        num_points_quad=num_points_quad, min_val=min_val, max_val=max_val,
        ndim=1, device=device, dtype=dtype
    )
    X2, W2 = _get_quad_tensors(
        num_points_quad=num_points_quad, min_val=min_val, max_val=max_val,
        ndim=2, device=device, dtype=dtype
    )

    alpha = 1.0 - cred_level
    q_lo = alpha / 2.0
    q_hi = 1.0 - alpha / 2.0

    metrics_kld = None
    if likelihood_based_metrics:
        metrics_kld = torch.empty((S_posterior, n_pairs), dtype=torch.float32, device="cpu")

    thetas = vi_model.sample_theta(S_posterior, antithetic=True)

    start_total = time.time()
    bar = tqdm(range(S_posterior), total=S_posterior, desc="[Bayesian CI] posterior draws", unit="draw")

    kld_running_sum = 0.0

    for s in bar:
        t0 = time.time()
        theta_s = thetas[s]
        params_s = vi_model._theta_to_state_dict(theta_s)

        with _reparametrize_module(self, params_s):
            if likelihood_based_metrics:
                kld_draw = torch.empty((n_pairs,), device=device, dtype=torch.float32)

                for start in range(0, n_pairs, pair_chunk_size):
                    end = min(start + pair_chunk_size, n_pairs)
                    cols = pair_index[start:end]
                    col_i = cols[:, 0]
                    col_j = cols[:, 1]

                    actual, under = _ci_log_dists_many_pairs_glq_stream(
                        self,
                        evaluation_data,
                        col_i, col_j,
                        X1, W1,
                        X2, W2,
                        qmb_1d=q_microbatch_1d,
                        qmb_2d=q_microbatch_2d,
                    )  # [P,N]

                    if evaluation_data_type in ("data", "samples_from_model"):
                        ll_dev = (actual - under).to(torch.float32)  # [P,N]
                        finite = torch.isfinite(ll_dev)
                        ll_dev = torch.where(finite, ll_dev, torch.tensor(float("nan"), device=device))

                        abs_dev = ll_dev.abs()
                        q = torch.nanquantile(abs_dev, 0.99, dim=1)  # [P]
                        keep = (abs_dev < q[:, None]) & torch.isfinite(ll_dev)

                        count = keep.sum(dim=1).clamp_min(1)
                        val = torch.where(keep, ll_dev, torch.zeros_like(ll_dev)).sum(dim=1) / count
                        kld_chunk = torch.nan_to_num(val, nan=0.0, posinf=0.0, neginf=0.0)

                    elif evaluation_data_type == "uniform_random_samples":
                        ll_dev = torch.exp(actual) * (actual - under)
                        kld_chunk = torch.nan_to_num(torch.nanmean(ll_dev, dim=1), nan=0.0, posinf=0.0, neginf=0.0)

                    else:
                        raise ValueError("unknown evaluation_data_type in Bayesian KLD")

                    kld_draw[start:end] = kld_chunk

                metrics_kld[s] = kld_draw.detach().cpu()
                kld_draw_mean = float(metrics_kld[s].mean().item())
                kld_running_sum += kld_draw_mean

        dt = time.time() - t0
        bar.set_postfix({
            "pairs": n_pairs,
            "pair_chunk": pair_chunk_size,
            "qmb2d": q_microbatch_2d,
            "sec/draw": f"{dt:.2f}",
            "KLD(draw)": f"{kld_draw_mean:.4g}",
            "KLD(avg)": f"{(kld_running_sum/(s+1)):.4g}",
        })

    total_sec = time.time() - start_total
    print(
        f"[Bayesian CI] Time taken (device={device}, S_post={S_posterior}, pairs={n_pairs}): "
        f"{total_sec:.2f}s (~{total_sec/float(S_posterior):.2f}s per draw)"
    )

    df = precision_summary.copy()
    raw = {}

    if likelihood_based_metrics:
        kld_mean = metrics_kld.mean(dim=0).tolist()
        qs = torch.quantile(metrics_kld.to(torch.float32), torch.tensor([q_lo, q_hi]), dim=0)
        df["kld_mean"] = kld_mean
        df["kld_lo"] = qs[0].tolist()
        df["kld_hi"] = qs[1].tolist()
        raw["kld"] = metrics_kld.numpy()

    raw["pair_index"] = precision_summary[["var_row", "var_col"]].to_numpy(dtype=np.int64)

    if copula_only and old_num_trans_layers is not None:
        self.num_trans_layers = old_num_trans_layers

    df.reset_index(inplace=True)
    return df, raw