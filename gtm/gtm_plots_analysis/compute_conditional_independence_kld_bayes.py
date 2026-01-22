from __future__ import annotations
import time
from typing import Optional, Tuple, Dict, TYPE_CHECKING

import numpy as np
import torch
from tqdm import tqdm
from torch.nn.utils.stateless import _reparametrize_module

from gtm.gtm_plots_analysis.nd_quad import getQuad, transformQuad
from gtm.gtm_plots_analysis.compute_precision_matrix_summary_statistics import *

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

    X = torch.as_tensor(x, device=device, dtype=dtype)
    W = torch.as_tensor(w, device=device, dtype=dtype)
    _QUAD_CACHE[key] = (X, W)
    return X, W


# ============================================================
# Streaming integrals with precomputed indices (faster)
# ============================================================

@torch.no_grad()
def _log_int_out_two_cols_stream_fast(
    model,
    flat_base: torch.Tensor,   # [P*N, D] base replicated (already)
    P: int,
    N: int,
    D: int,
    col_i_row: torch.Tensor,   # [P*N] column index for each row
    col_j_row: torch.Tensor,   # [P*N]
    X2: torch.Tensor,          # [Q2,2]
    W2: torch.Tensor,          # [Q2]
    q_microbatch: int,
    use_amp: bool = False,
    amp_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """
    Returns log integral: [P,N]
    """
    Q2 = X2.shape[0]
    device = flat_base.device

    integral = torch.zeros((P * N,), device=device, dtype=torch.float32)

    # constants reused
    row_idx = torch.arange(P * N, device=device)  # for scatter on the flattened P*N axis

    for s in range(0, Q2, q_microbatch):
        e = min(s + q_microbatch, Q2)
        Xb = X2[s:e, :]  # [B,2]
        Wb = W2[s:e]     # [B]
        B = Xb.shape[0]

        # Create [B, P*N, D] view then materialize [B*P*N, D] once
        y = flat_base.unsqueeze(0).expand(B, P * N, D).reshape(B * P * N, D).clone()

        # Repeat per-microbatch column indices
        col_i_rep = col_i_row.unsqueeze(0).expand(B, P * N).reshape(-1)
        col_j_rep = col_j_row.unsqueeze(0).expand(B, P * N).reshape(-1)

        # Values to insert
        x1rep = Xb[:, 0].unsqueeze(1).expand(B, P * N).reshape(-1)
        x2rep = Xb[:, 1].unsqueeze(1).expand(B, P * N).reshape(-1)

        idx = torch.arange(B * P * N, device=device)
        y[idx, col_i_rep] = x1rep
        y[idx, col_j_rep] = x2rep

        if use_amp and y.is_cuda:
            with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=True):
                ll = model.log_likelihood(y)
            dens = torch.exp(ll.float())
        else:
            ll = model.log_likelihood(y)
            dens = torch.exp(ll).to(torch.float32)

        # weights: [B, P*N]
        wrep = Wb.unsqueeze(1).expand(B, P * N)
        integral += (dens.reshape(B, P * N) * wrep).sum(dim=0)

        del y, ll, dens

    integral = torch.clamp(integral, min=torch.finfo(torch.float32).tiny)
    return torch.log(integral).reshape(P, N)


@torch.no_grad()
def _log_int_out_one_col_stream_fast(
    model,
    flat_base: torch.Tensor,   # [P*N, D]
    P: int,
    N: int,
    D: int,
    col_k_row: torch.Tensor,   # [P*N]
    X1: torch.Tensor,          # [Q1,1]
    W1: torch.Tensor,          # [Q1]
    q_microbatch: int,
    use_amp: bool = False,
    amp_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """
    Returns log integral: [P,N]
    """
    Q1 = X1.shape[0]
    device = flat_base.device

    integral = torch.zeros((P * N,), device=device, dtype=torch.float32)

    for s in range(0, Q1, q_microbatch):
        e = min(s + q_microbatch, Q1)
        xb = X1[s:e, 0]   # [B]
        wb = W1[s:e]      # [B]
        B = xb.numel()

        y = flat_base.unsqueeze(0).expand(B, P * N, D).reshape(B * P * N, D).clone()

        col_rep = col_k_row.unsqueeze(0).expand(B, P * N).reshape(-1)
        xrep = xb.unsqueeze(1).expand(B, P * N).reshape(-1)

        idx = torch.arange(B * P * N, device=device)
        y[idx, col_rep] = xrep

        if use_amp and y.is_cuda:
            with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=True):
                ll = model.log_likelihood(y)
            dens = torch.exp(ll.float())
        else:
            ll = model.log_likelihood(y)
            dens = torch.exp(ll).to(torch.float32)

        wrep = wb.unsqueeze(1).expand(B, P * N)
        integral += (dens.reshape(B, P * N) * wrep).sum(dim=0)

        del y, ll, dens

    integral = torch.clamp(integral, min=torch.finfo(torch.float32).tiny)
    return torch.log(integral).reshape(P, N)


@torch.no_grad()
def _log_int_out_two_one_cols_fused_stream_fast(
    model,
    flat_base: torch.Tensor,     # [P*N, D]
    P: int,
    N: int,
    D: int,
    col_a_row: torch.Tensor,     # [P*N] (for p4_num)
    col_b_row: torch.Tensor,     # [P*N] (for p5_num)
    X1: torch.Tensor,            # [Q1,1]
    W1: torch.Tensor,            # [Q1]
    q_microbatch: int,
    use_amp: bool = False,
    amp_dtype: torch.dtype = torch.bfloat16,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Fused computation of two separate 1D integrals in ONE likelihood call stream:
      int_a: integrate out col_a_row
      int_b: integrate out col_b_row

    Returns:
      log_int_a: [P,N]
      log_int_b: [P,N]
    """
    Q1 = X1.shape[0]
    device = flat_base.device

    integral_a = torch.zeros((P * N,), device=device, dtype=torch.float32)
    integral_b = torch.zeros((P * N,), device=device, dtype=torch.float32)

    for s in range(0, Q1, q_microbatch):
        e = min(s + q_microbatch, Q1)
        xb = X1[s:e, 0]   # [B]
        wb = W1[s:e]      # [B]
        B = xb.numel()

        # We will build two batches (A and B) and concatenate:
        # y_cat has shape [2*B*P*N, D]
        yA = flat_base.unsqueeze(0).expand(B, P * N, D).reshape(B * P * N, D).clone()
        yB = flat_base.unsqueeze(0).expand(B, P * N, D).reshape(B * P * N, D).clone()

        colA = col_a_row.unsqueeze(0).expand(B, P * N).reshape(-1)
        colB = col_b_row.unsqueeze(0).expand(B, P * N).reshape(-1)
        xrep = xb.unsqueeze(1).expand(B, P * N).reshape(-1)

        idx = torch.arange(B * P * N, device=device)
        yA[idx, colA] = xrep
        yB[idx, colB] = xrep

        y = torch.cat([yA, yB], dim=0)

        if use_amp and y.is_cuda:
            with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=True):
                ll = model.log_likelihood(y)
            dens = torch.exp(ll.float())
        else:
            ll = model.log_likelihood(y)
            dens = torch.exp(ll).to(torch.float32)

        densA, densB = dens[: B * P * N], dens[B * P * N :]

        wrep = wb.unsqueeze(1).expand(B, P * N)  # [B,P*N]
        integral_a += (densA.reshape(B, P * N) * wrep).sum(dim=0)
        integral_b += (densB.reshape(B, P * N) * wrep).sum(dim=0)

        del yA, yB, y, ll, dens, densA, densB

    integral_a = torch.clamp(integral_a, min=torch.finfo(torch.float32).tiny)
    integral_b = torch.clamp(integral_b, min=torch.finfo(torch.float32).tiny)

    return torch.log(integral_a).reshape(P, N), torch.log(integral_b).reshape(P, N)


@torch.no_grad()
def _ci_log_dists_many_pairs_glq_stream_fast(
    model,
    data: torch.Tensor,          # [N,D]
    col_i: torch.Tensor,         # [P]
    col_j: torch.Tensor,         # [P]
    X1: torch.Tensor, W1: torch.Tensor,
    X2: torch.Tensor, W2: torch.Tensor,
    qmb_1d: int,
    qmb_2d: int,
    use_amp: bool = False,
    amp_dtype: torch.dtype = torch.bfloat16,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Returns:
      actual: [P,N]
      under : [P,N]
    """
    device = data.device
    N, D = data.shape
    P = col_i.numel()

    # p1 is common for all pairs
    p1 = model.log_likelihood(data)  # [N]

    # Build replicated base ONCE per chunk: [P*N, D]
    base = data.unsqueeze(0).expand(P, N, D).contiguous()
    flat_base = base.reshape(P * N, D)

    # Precompute per-row pair mapping ONCE
    pair_id = torch.arange(P, device=device).repeat_interleave(N)  # [P*N]
    col_i_row = col_i[pair_id]                                     # [P*N]
    col_j_row = col_j[pair_id]                                     # [P*N]

    # p2: integrate out i and j
    p2 = _log_int_out_two_cols_stream_fast(
        model, flat_base, P, N, D,
        col_i_row, col_j_row,
        X2, W2,
        q_microbatch=qmb_2d,
        use_amp=use_amp,
        amp_dtype=amp_dtype,
    )  # [P,N]

    # p3
    p3 = p1[None, :] - p2

    # FUSED p4_num (integrate out j) and p5_num (integrate out i)
    p4_num, p5_num = _log_int_out_two_one_cols_fused_stream_fast(
        model,
        flat_base, P, N, D,
        col_a_row=col_j_row,
        col_b_row=col_i_row,
        X1=X1, W1=W1,
        q_microbatch=qmb_1d,
        use_amp=use_amp,
        amp_dtype=amp_dtype,
    )  # each [P,N]

    p4 = p4_num - p2
    p5 = p5_num - p2

    actual = p3
    under = p4 + p5
    return actual, under


# --------------------------------------
# evaluation data helper (same)
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
# Main: Bayesian CI KLD (faster streaming)
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
    pair_chunk_size: int = 12,
    q_microbatch_1d: int = 256,
    q_microbatch_2d: int = 32,
    # NEW speed toggles:
    use_amp: bool = False,
    amp_dtype: torch.dtype = torch.bfloat16,
):
    """
    Faster OOM-safe Bayesian CI KLD:
      - caches quadrature
      - streams quadrature points
      - precomputes indices once per pair chunk
      - fuses the two 1D integrals (p4_num & p5_num) into one stream
      - optional AMP for A40
    """

    device = next(self.parameters()).device
    dtype = torch.float32

    # Optional: TF32 (Ampere+). Safe and often faster.
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

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
    )

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

                    actual, under = _ci_log_dists_many_pairs_glq_stream_fast(
                        self,
                        evaluation_data,
                        col_i, col_j,
                        X1, W1,
                        X2, W2,
                        qmb_1d=q_microbatch_1d,
                        qmb_2d=q_microbatch_2d,
                        use_amp=use_amp,
                        amp_dtype=amp_dtype,
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
