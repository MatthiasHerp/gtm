import time
import re
from typing import TYPE_CHECKING
from math import log, expm1, exp
import torch
import torch.nn as nn



from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.amp import autocast
from tqdm import tqdm
from gtm.gtm_training.training_bayes.utils import _ELBOConvergence, Trackers
from gtm.gtm_training.training_bayes.variational_model_estimator import VI_Model, VariationalGamma, GammaTauNode, TauPack

if TYPE_CHECKING:
    from ...gtm_model.gtm import GTM # type-only; no runtime import


def _make_key_filter(patterns_include=None, patterns_exclude=None):
    inc = [re.compile(p) for p in (patterns_include or [])]
    exc = [re.compile(p) for p in (patterns_exclude or [])]
    def _keep(k: str) -> bool:
        if inc and not any(r.search(k) for r in inc):
            return False
        if any(r.search(k) for r in exc):
            return False
        return True
    return _keep


def _beta_kl_at(epoch: int, beta_kl_anneal_epochs, beta_kl_start, beta_min=1.0) -> float:
    t = min(1.0, epoch / max(1, beta_kl_anneal_epochs))
    return beta_kl_start * (1.0 - t) + beta_min * t

def _gamma_mean(a, b): return float(a) / max(float(b), 1e-12)

def _seed_q_from_mean(q, tau_current: float):
    # mean(q) = a_hat / b_hat -> fix a_hat, adjust b_hat to match current mean
    a_hat=q.a0 + 0.5 * q.rank_total
    b_hat=max(a_hat / float(tau_current), 1e-12)
    q.a_hat=float(a_hat)
    q.b_hat=float(b_hat)


def _softplus_inv(x: float, eps: float = 1e-8) -> float:
    # Numerically stable inverse softplus for positive x
    x = max(x, eps)
    return float(log(expm1(x)))

def _seed_q_from_eb(q, E_qf_total_mc: float):
    # EB fixed point: a_hat=a0+rank/2, b_hat=b0+0.5*E[qf]
    q.a_hat=float(q.a0 + 0.5 * q.rank_total)
    q.b_hat=float(q.b0 + 0.5 * float(E_qf_total_mc))
    
def _tau_times_qf_target(a, b, r_half, Eqf):
    # r_half = rank/2
    num = (a + r_half) * Eqf
    den = (b + 0.5 * Eqf)
    return float(num / max(den, 1e-12))

def _damped_step(prev, target, eta, band=0.20):
    """
    One-step EMA toward target with per-epoch change capped to [1-band, 1+band]× of the previous value.
    """
    prev = float(prev)
    target = float(target)
    raw = (1.0 - float(eta)) * prev + float(eta) * target
    lo = prev * (1.0 - band)
    hi = prev * (1.0 + band)
    return float(max(lo, min(hi, raw)))

@torch.no_grad()
def _evaluate_epoch(
    VI: 'VI_Model', 
    model: 'GTM',
    val_loader,
    hyper_T,
    hyper_D,
    sample_size_total,
    S_val=8, 
    seed=123,
    tau_nodes=None,
    use_tau_vi_now: bool=False,
    ):
    nn.Module.train(model, False)
    total_loglik_sum, nobs = 0.0, 0
    for y in val_loader:
        y = y.to(model.device, non_blocking = True)
        with autocast(str(model.device), dtype=torch.float16):
            logsum = VI.predictive_loglik_sum_batch(
                y_batch=y,
                model=model,
                hyperparameter_transformation=hyper_T,
                hyperparameter_decorrelation=hyper_D,
                S=S_val,
                sample_size=1,
                seed=seed,
                tau_nodes=tau_nodes,
                use_tau_vi_now=use_tau_vi_now,
            )
        total_loglik_sum += logsum
        nobs += y.shape[0]
    return total_loglik_sum / nobs  # per-observation ELPD (higher is better)

def train_bayes(
    model: "GTM",
    train_dataloader,
    validate_dataloader=None,
    iterations: int = 100,
    lr: float = 1e-3,
    hyperparameters: dict | None = None,
    verbose: bool = True,
    max_batches_per_iter: int | bool = False,

    # --- NEW knobs ---
    mcmc_sample_train: int = 4,       # == S_train
    mcmc_sample_val: int = 8,         # == S_val (keep fixed!)
    mc_ramp_every: int | None = 20,   # every N epochs, double S_train; None to disable
    mc_ramp_max: int = 32,
    global_seed: int = 0,
    patience_val: int = 10,
    min_delta: float = 1e-4,
    rho_lr_multiplier: float = 1.0,
    sched_factor: float = 0.5,
    sched_patience: int = 5,
    sched_threshold: float = 1e-4,
    sched_cooldown: int = 1,
    sched_min_lr: float = 5e-5,

    # --- WARMING PHASES ---
    warm_tau_epochs: int = 5,         # from 3 → 5
    warm_sigma_epochs: int = 18,      # from 8 → 18

    # --- OPTIMIZATION ---
    beta_kl_start: float = 2.5,       # from 2.0 → 2.5
    beta_kl_anneal_epochs: int = 25,  # slower anneal
    use_empirical_bayes: bool = False,
    eb_warm_then_cavi: bool = True,   # EB for first warm_tau_epochs only
    band_tau4 = 0.20,
    band_decor = 0.15,
    
    # --- τ VI toggles (keep EB as default behavior) ---
    tau_vi_mode = "after_warm", #"off" | "after_warm" | "always"
    tau_kl_beta = 1.0,
    tau_vi_sigma_init = 0.25,
    
        # --- VI convergence (no-val) ---
    conv_use_ema: bool = True,
    conv_window_size: int = 5,   # used if conv_use_ema=False
    conv_tol: float = 1e-5,      # absolute ELBO change per-obs
    conv_min_epochs: int = 10,   # don't stop too early
    conv_ema_beta: float = 0.9,  # if conv_use_ema=True

    
):
    TAU4_FLOOR = 1e-3
    TAU1_FLOOR = 1e-2
    TAU2_FLOOR = 1e-2
    S_tau_monitor = 128  
    
    was_training = model.training
    nn.Module.train(model, False)
    N_total = len(train_dataloader.dataset)
    
    decor_present = not (model.number_decorrelation_layers == 0 or model.transform_only)
    
    
    # ------------------- key filter
    if model.number_decorrelation_layers == 0 or model.transform_only:
        parameters_patterns = [r"^transformation\.params\.\d+$"]
        patterns_exclude = [r"decor", r"rho_", r"optimizer", r"running_", r"num_batches_tracked"]
    else:
        parameters_patterns = [r"^transformation\.params\.\d+$",
                               r"^decorrelation_layers\.\d+\.params$"]
        patterns_exclude = [r"rho_", r"optimizer", r"running_", r"num_batches_tracked"]
    key_filter = _make_key_filter(patterns_include=parameters_patterns,
                                  patterns_exclude=patterns_exclude)

    # ------------------- VI, RNG
    K = model.transformation.priors.K_prior_RW2.to(model.device)
    num_margins = int(model.number_variables)
    VI = VI_Model(model=model, device=model.device, key_filter=key_filter).to(model.device)
    gen = torch.Generator(device=model.device); gen.manual_seed(global_seed); VI.set_rng(gen)

    print("Parameters to be affected by the bayesian approach:", [k for k, _ in VI._schema][:50])

    # ------------------- hyperparameters
    if hyperparameters is None:
        hyper_T = model.hyperparameter.get("transformation", {})
        hyper_D = model.hyperparameter.get("decorrelation", {})
    else:
        hyper_T = hyperparameters.get("transformation", {})
        hyper_D = hyperparameters.get("decorrelation", {})

    # ------------------- Transformation (τ4)
    nullspace_dim_T = int(hyper_T.get("nullspace_dim", 2))
    rank_per_margin = K.shape[0] - nullspace_dim_T
    rank_T_total = rank_per_margin * num_margins

    pen_term2_T = hyper_T.get("RW2", {})
    a_lambda = torch.as_tensor(pen_term2_T.get("tau_a", 1.1), device=model.device, dtype=torch.float32)
    b_lambda = torch.as_tensor(pen_term2_T.get("tau_b", 1e-6), device=model.device, dtype=torch.float32)
    print(f"[TRANSFORMATION] λ_a={a_lambda.item()} λ_b={b_lambda.item()}")

    def use_eb_now(epoch): return eb_warm_then_cavi and (epoch < warm_tau_epochs)

    # --- NEW: separate damping etas
    eta_tau4 = float(hyper_T.get("tau_4_update_eta", 0.25))     # default 0.25
    eta_tau1 = float(hyper_D.get("tau_1_update_eta", 0.35))     # default 0.35
    eta_tau2 = float(hyper_D.get("tau_2_update_eta", 0.35))     # default 0.35

    # init τ4
    if use_empirical_bayes or eb_warm_then_cavi:
        tau_4 = torch.as_tensor(pen_term2_T.get("tau_init", _gamma_mean(a_lambda, b_lambda)),
                                device=model.device,  dtype=torch.float32)
    else:
        q_tau4 = VariationalGamma(a_lambda, b_lambda, rank_T_total, init_from_prior=True)
        tau_4 = q_tau4.mean
    hyper_T["tau"] = tau_4

    # ------------------- Decorrelation (τ1, τ2)
    if decor_present:
        K_RW1 = model.decorrelation_layers[0].priors.K_prior_RW1
        K_RW2 = model.decorrelation_layers[0].priors.K_prior_RW2
        rank_T_total_RW1 = (K_RW1.shape[0] - 1) * num_margins
        rank_T_total_RW2 = (K_RW2.shape[0] - 2) * num_margins

        pen_term1_D = hyper_D.get('RW1', {})
        pen_term2_D = hyper_D.get('RW2', {})
        a_lambda_1 = torch.as_tensor(pen_term1_D.get('tau_a', 1.1), device=model.device, dtype=torch.float32)
        b_lambda_1 = torch.as_tensor(pen_term1_D.get('tau_b', 1e-6), device=model.device, dtype=torch.float32)
        a_lambda_2 = torch.as_tensor(pen_term2_D.get('tau_a', 1.1), device=model.device, dtype=torch.float32)
        b_lambda_2 = torch.as_tensor(pen_term2_D.get('tau_b', 1e-6), device=model.device, dtype=torch.float32)
        print(f"[DECOR] τ₁ a={a_lambda_1.item()} b={b_lambda_1.item()} | τ₂ a={a_lambda_2.item()} b={b_lambda_2.item()}")

        if use_empirical_bayes or eb_warm_then_cavi:
            hyper_D["tau_1"] = torch.as_tensor(_gamma_mean(a_lambda_1, b_lambda_1), device=model.device, dtype=torch.float32)
            hyper_D["tau_2"] = torch.as_tensor(_gamma_mean(a_lambda_2, b_lambda_2), device=model.device,  dtype=torch.float32)
        else:
            q_tau1 = VariationalGamma(a_lambda_1, b_lambda_1, rank_T_total_RW1, init_from_prior=True)
            q_tau2 = VariationalGamma(a_lambda_2, b_lambda_2, rank_T_total_RW2, init_from_prior=True)
            hyper_D["tau_1"], hyper_D["tau_2"] = q_tau1.mean, q_tau2.mean
    else:
        print("training only transformation layer!")
        hyper_D["tau_1"], hyper_D["tau_2"] = 0.0, 0.0

    q_tau4 = None; q_tau1 = None; q_tau2 = None
    
    # ------------------- Full Bayesian taus 
    
    use_tau_vi_anytime = (tau_vi_mode in ("after_warm", "always"))
    tau_nodes = None
    if use_tau_vi_anytime:
        
        node4 = GammaTauNode(
            a=float(a_lambda),
            b=float(b_lambda),
            mean_init=float(hyper_T["tau"]),
            cv_init=float(tau_vi_sigma_init),
            device=model.device,
        )
        
        if decor_present:
            node1 = GammaTauNode(
                a=float(a_lambda_1),
                b=float(b_lambda_1),
                mean_init=float(hyper_D["tau_1"]),
                cv_init=float(tau_vi_sigma_init),
                device=model.device,
                )
            node2 = GammaTauNode(
                a=float(a_lambda_2),
                b=float(b_lambda_2),
                mean_init=float(hyper_D["tau_2"]),
                cv_init=float(tau_vi_sigma_init),
                device=model.device,
                )
        else:
            
            node1 = node2 = None
        
        tau_nodes = TauPack(node4=node4, node2=node2, node1=node1)
    

    # ------------------- optimizer & scheduler
    import inspect
    def _make_adam(param_groups, lr: float):
        # Detect supported kwargs on this PyTorch build
        sig = inspect.signature(torch.optim.Adam.__init__).parameters
        supports_fused   = ("fused"   in sig)
        supports_foreach = ("foreach" in sig)

        # Try fused Adam first (fastest on Ampere+), then foreach, then vanilla
        if supports_fused:
            try:
                return torch.optim.Adam(param_groups, lr=lr, fused=True)
            except Exception:
                pass
        if supports_foreach:
            try:
                return torch.optim.Adam(param_groups, lr=lr, foreach=True)
            except Exception:
                pass
        return torch.optim.Adam(param_groups, lr=lr)

    # ------------------- optimizer & scheduler
    if rho_lr_multiplier != 1.0:
        param_groups = [
            {"params": [VI.mu],  "lr": lr},
            {"params": [VI.rho], "lr": lr * rho_lr_multiplier},
        ]
        opt = _make_adam(param_groups, lr=lr)
    else:
        param_groups = [{"params": [p for p in VI.parameters() if p.requires_grad]}]
        opt = _make_adam(param_groups, lr=lr)
        

    # τ nodes (keeps same opt; adds a new param group)
    if use_tau_vi_anytime and tau_nodes is not None:
        pg = tau_nodes.tau_param_group(lr=lr * 1.5, betas=(0.9, 0.999))
        if len(pg["params"]) > 0:
            opt.add_param_group(pg)

        
    sched = ReduceLROnPlateau(opt, mode="min", factor=sched_factor, patience=sched_patience,
                              threshold=sched_threshold, cooldown=sched_cooldown,
                              min_lr=sched_min_lr, verbose=verbose)
    
    if validate_dataloader is None:
        conv_tracker = _ELBOConvergence(
            use_ema=conv_use_ema,
            window_size=conv_window_size,
            tol=conv_tol,
            beta=conv_ema_beta,
            min_epochs=conv_min_epochs,
        )

    best_val = float("inf")
    best_state = {"mu": VI.mu.detach().clone(), "rho": VI.rho.detach().clone()}
    eb_E_qf_num, eb_count = 0.0, 0
    no_improve = 0
    
    monitors = Trackers.new_monitour()
    
    loss_history, val_history = [], []
    start = time.time()

    for epoch in tqdm(range(iterations)):
        beta_kl = _beta_kl_at(epoch, beta_kl_anneal_epochs, beta_kl_start)
        use_tau_vi_now = (
            (tau_vi_mode == "always") or 
            (tau_vi_mode == "after_warm" and epoch >= warm_tau_epochs)
        )
        if mc_ramp_every and epoch > 0 and (epoch % mc_ramp_every == 0):
            mcmc_sample_train = min(mc_ramp_max, max(mcmc_sample_train * 2, 1))

        running, n_batches, obs_seen_epoch = 0.0, 0, 0
        ndp, ntp, ll_batch = 0.0, 0.0, 0.0
        qf_1_dec, qf_2_dec, E_qf_sum_accum = 0.0, 0.0, 0.0

        for b, y in enumerate(train_dataloader):
            y = y.to(model.device)
            B = y.shape[0]
            opt.zero_grad(set_to_none=True)

            if epoch < warm_sigma_epochs:
                VI.rho.requires_grad_(False)
            else:
                VI.rho.requires_grad_(True)

            out = VI.step(samples=y,
                          hyperparameter_transformation=hyper_T,
                          hyperparameter_decorrelation=hyper_D,
                          sample_size_total=N_total,
                          model=model,
                          mcmc_samples=mcmc_sample_train,
                          seed=global_seed + epoch * 10_000 + b,
                          beta_kl=beta_kl,
                          tau_pack=(tau_nodes if use_tau_vi_now else None),
                          beta_tau_kl = tau_kl_beta
                          )

            loss = out["elbo_loss"]
            #if not torch.isfinite(loss):
            #    raise RuntimeError(f"Non-finite loss @ epoch {epoch}, batch {b}: {loss.item()}")
            if not torch.isfinite(loss):
                # emergency backoff
                for pg in opt.param_groups:
                    pg["lr"] = max(pg["lr"] * 0.5, 5e-5)
                print("SKIP BATCH FOR INF... CHECK NUMERIC STABILITY")
                # optional: shrink S_train for a step
                #mcmc_sample_train = max(4, mcmc_sample_train // 2)
                # skip this batch
                continue
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(VI.parameters(), 5.0)
            opt.step()

            with torch.no_grad():
                def sp_inv(s): return log(exp(float(s)) - 1.0)
                rho_min = sp_inv(0.015)  # softplus^-1(0.02)
                rho_max = sp_inv(0.12)
                VI.rho.data.clamp_(min=rho_min, max=rho_max)   # narrow band; widen if needed
            
            running         += float(loss.item()); n_batches += 1; obs_seen_epoch += B
            ndp             += float(out['neg_prior_decorrelation'])
            ntp             += float(out['neg_prior_transformation'])
            ll_batch        += float(out['log_likelihhod_batch'])
            eb_E_qf_num     += float(out['transformation_mean_qf'])
            E_qf_sum_accum  += float(out['transformation_sum_qf'])
            qf_1_dec        += float(out['qf1_decorrelation'])
            qf_2_dec        += float(out['qf2_decorrelation'])
            eb_count        += 1

        if n_batches == 0:
            raise RuntimeError("No batches processed. Check dataloader.")

        train_loss = running / obs_seen_epoch
        loss_history.append(train_loss)

         # --- ELBO per-obs (since train_loss is per-obs loss = -ELBO) ---
        elbo_per_obs = -train_loss

        if validate_dataloader is not None:
            # keep your existing validation-ELPD early stop
            val_dat = validate_dataloader
            val_elpd = _evaluate_epoch(
                VI, model, val_dat,
                hyper_T, hyper_D,
                S_val=mcmc_sample_val,
                sample_size_total=N_total,
                seed=global_seed + 12345,
                tau_nodes=tau_nodes, use_tau_vi_now=use_tau_vi_now,
            )
            val_history.append(val_elpd)
            metric = -val_elpd
            sched.step(metric)

            improved = (metric < best_val - min_delta)
            if improved:
                best_val = metric; no_improve = 0
                best_state = {
                    "mu": VI.mu.detach().clone(),
                    "rho": VI.rho.detach().clone(),
                    "tau_nodes": {
                        "node4_mu": tau_nodes.node4.mu.detach().clone()                 if (tau_nodes and tau_nodes.node4) else None,
                        "node4_log_sigma": tau_nodes.node4.log_sigma.detach().clone()   if (tau_nodes and tau_nodes.node4) else None,
                        "node1_mu": tau_nodes.node1.mu.detach().clone()                 if (tau_nodes and tau_nodes.node1) else None,
                        "node1_log_sigma": tau_nodes.node1.log_sigma.detach().clone()   if (tau_nodes and tau_nodes.node1) else None,
                        "node2_mu": tau_nodes.node2.mu.detach().clone()                 if (tau_nodes and tau_nodes.node2) else None,
                        "node2_log_sigma": tau_nodes.node2.log_sigma.detach().clone()   if (tau_nodes and tau_nodes.node2) else None,
                    }
                    }
            else:
                no_improve += 1
                if verbose: print(f"NOT IMPROVED! Nr. {no_improve}")
        else:
            # --- convergence-based stopping on training ELBO (no validation set) ---
            val_elpd = None
            converged, elbo_smooth = conv_tracker.update(elbo_per_obs)

            # you can still drive the LR scheduler on train loss if you like
            try:
                sched.step(train_loss)
            except Exception:
                pass

            if verbose and ((epoch+1) % 5 == 0):
                print(f"[convergence] ELBO/obs={elbo_per_obs:.6f}  smoothed={elbo_smooth:.6f}  tol={conv_tracker.tol:g}")

            if converged:
                print(f"Converged (ELBO plateau) at epoch {epoch+1} "
                      f"with Δ<tol={conv_tracker.tol:g}.")
                # snapshot current best (since there is no 'val', we keep last)
                best_state = {
                    "mu": VI.mu.detach().clone(), "rho": VI.rho.detach().clone(),
                    "tau_nodes": {
                        "node4_mu": tau_nodes.node4.mu.detach().clone()                 if (tau_nodes and tau_nodes.node4) else None,
                        "node4_log_sigma": tau_nodes.node4.log_sigma.detach().clone()   if (tau_nodes and tau_nodes.node4) else None,
                        "node1_mu": tau_nodes.node1.mu.detach().clone()                 if (tau_nodes and tau_nodes.node1) else None,
                        "node1_log_sigma": tau_nodes.node1.log_sigma.detach().clone()   if (tau_nodes and tau_nodes.node1) else None,
                        "node2_mu": tau_nodes.node2.mu.detach().clone()                 if (tau_nodes and tau_nodes.node2) else None,
                        "node2_log_sigma": tau_nodes.node2.log_sigma.detach().clone()   if (tau_nodes and tau_nodes.node2) else None,
                    }}
                break

        # ------------------- τ updates (EB or CAVI) with per-τ damping
        freeze_decor = (epoch < warm_tau_epochs) 
                # ------------------- τ updates (EB or CAVI) with per-τ damping + warm-up freeze for decor
        # ---- τ updates (EB or CAVI) with per-τ damping + warm-up freeze for decor ----
        if eb_count > 0:
            E_qf_total_mc = E_qf_sum_accum / eb_count
            E_qf1_total_mc = qf_1_dec / eb_count
            E_qf2_total_mc = qf_2_dec / eb_count

            # EB fixed-point targets (for EB/CAVI and monitoring)
            tau4_target = float((a_lambda + 0.5 * rank_T_total) / (b_lambda + 0.5 * E_qf_total_mc))
            if decor_present:
                tau1_target = float((a_lambda_1 + 0.5 * rank_T_total_RW1) / (b_lambda_1 + 0.5 * E_qf1_total_mc))
                tau2_target = float((a_lambda_2 + 0.5 * rank_T_total_RW2) / (b_lambda_2 + 0.5 * E_qf2_total_mc))
            else:
                tau1_target = tau2_target = 0.0

            if use_eb_now(epoch):
                # ----- EB step (damped) -----
                tau_new = _damped_step(hyper_T["tau"], tau4_target, eta_tau4, band_tau4)
                if decor_present and not freeze_decor:
                    tau_1_new = _damped_step(hyper_D["tau_1"], tau1_target, eta_tau1, band_decor)
                    tau_2_new = _damped_step(hyper_D["tau_2"], tau2_target, eta_tau2, band_decor)
                else:
                    tau_1_new = float(hyper_D.get("tau_1", 0.0))
                    tau_2_new = float(hyper_D.get("tau_2", 0.0))
            else:
                # ----- CAVI (variational gamma) -----
                if q_tau4 is None:
                    q_tau4 = VariationalGamma(a_lambda, b_lambda, rank_T_total, init_from_prior=True)
                    _seed_q_from_mean(q_tau4, tau_current=float(hyper_T["tau"]))
                q_tau4.update_from_E_qf_total(E_qf_total=E_qf_total_mc)
                tau_new_mean = float(q_tau4.mean)
                tau_new = _damped_step(hyper_T["tau"], tau_new_mean, eta_tau4)

                if decor_present and not freeze_decor:
                    if q_tau1 is None:
                        q_tau1 = VariationalGamma(a_lambda_1, b_lambda_1, rank_T_total_RW1, init_from_prior=True)
                        _seed_q_from_mean(q_tau1, tau_current=float(hyper_D["tau_1"]))
                    if q_tau2 is None:
                        q_tau2 = VariationalGamma(a_lambda_2, b_lambda_2, rank_T_total_RW2, init_from_prior=True)
                        _seed_q_from_mean(q_tau2, tau_current=float(hyper_D["tau_2"]))
                    q_tau1.update_from_E_qf_total(E_qf_total=E_qf1_total_mc)
                    q_tau2.update_from_E_qf_total(E_qf_total=E_qf2_total_mc)
                    tau1_mean = float(q_tau1.mean)
                    tau2_mean = float(q_tau2.mean)
                    tau_1_new = _damped_step(hyper_D["tau_1"], tau1_mean, eta_tau1)
                    tau_2_new = _damped_step(hyper_D["tau_2"], tau2_mean, eta_tau2)
                else:
                    tau_1_new = float(hyper_D.get("tau_1", 0.0))
                    tau_2_new = float(hyper_D.get("tau_2", 0.0))

            # ---- write-back only if NOT using τ-VI ----
            if not use_tau_vi_now:
                tau_new   = max(float(tau_new),  TAU4_FLOOR)
                tau_1_new = max(float(tau_1_new), TAU1_FLOOR)
                tau_2_new = max(float(tau_2_new), TAU2_FLOOR)
                hyper_T["tau"]   = tau_new
                hyper_D["tau_1"] = tau_1_new
                hyper_D["tau_2"] = tau_2_new
                model.hyperparameter["transformation"]["tau"]   = tau_new
                model.hyperparameter["decorrelation"]["tau_1"]  = tau_1_new
                model.hyperparameter["decorrelation"]["tau_2"]  = tau_2_new
        # τ-VI active → skip write-back (τ sampled from variational nodes)

            lrs = [pg["lr"] for pg in opt.param_groups]
            tgt4 = _tau_times_qf_target(float(a_lambda),  float(b_lambda),  0.5*rank_T_total,      E_qf_total_mc)
            tgt1 = _tau_times_qf_target(float(a_lambda_1),float(b_lambda_1),0.5*rank_T_total_RW1,  E_qf1_total_mc) if decor_present else 0.0
            tgt2 = _tau_times_qf_target(float(a_lambda_2),float(b_lambda_2),0.5*rank_T_total_RW2,  E_qf2_total_mc) if decor_present else 0.0

            # collect τ means (qτ) and targets (EB)
            if use_tau_vi_now and (tau_nodes is not None):
                with torch.no_grad():
                    
                    if tau_nodes.node4 is not None:
                        tau4_mean, tau4_var = tau_nodes.node4.mean_and_var_tau_mc(S_tau_monitor, generator=gen)
                    else:
                        tau4_mean, tau4_var = 0.0, 0.0
                    
                    if decor_present and tau_nodes.node1 and tau_nodes.node2:
                        (tau1_mean, tau1_var) = tau_nodes.node1.mean_and_var_tau_mc(S_tau_monitor, generator=gen)
                        (tau2_mean, tau2_var) = tau_nodes.node2.mean_and_var_tau_mc(S_tau_monitor, generator=gen)
                    else:
                        tau1_mean = tau1_var = tau2_mean= tau2_var = 0.0
            else:
                tau4_mean = tau1_mean = tau2_mean = tau4_var = tau1_var = tau2_var =0.0

            # --- NEW: use qτ means in the monitor if τ-VI is on; otherwise fall back to fixed hyper τs
            tau4_monitor = tau4_mean if use_tau_vi_now else float(hyper_T["tau"])
            tau1_monitor = tau1_mean if use_tau_vi_now else float(hyper_D["tau_1"])
            tau2_monitor = tau2_mean if use_tau_vi_now else float(hyper_D["tau_2"])
            
            with torch.no_grad():
                mu_vec = VI.mu.detach()
                sigma_vec = VI.sigma.detach()
                mu_norm = float(mu_vec.norm())
                mu_mean = float(mu_vec.mean())
                mu_std  = float(mu_vec.std())
                sigma_mean = float(sigma_vec.mean())
                sigma_min  = float(sigma_vec.min())
                sigma_max  = float(sigma_vec.max())
            
             # --- ELBO stats ---
            monitors["epoch"].append(epoch + 1)
            monitors["train_loss"].append(train_loss)
            monitors["elbo_per_obs"].append(elbo_per_obs)
            monitors["val_ELPD"].append(val_elpd if val_elpd is not None else None)

            # taus: hyperparameters
            monitors["tau4"].append(float(hyper_T["tau"]))
            monitors["tau1"].append(float(hyper_D["tau_1"]))
            monitors["tau2"].append(float(hyper_D["tau_2"]))

            # taus: VI means (or equal to hyper τ if VI off)
            monitors["tau4_mean"].append(float(tau4_mean))
            monitors["tau1_mean"].append(float(tau1_mean))
            monitors["tau2_mean"].append(float(tau2_mean))
            
            # taus: VI vars (or equal to hyper τ if VI off)
            monitors["tau4_var"].append(float(tau4_var))
            monitors["tau1_var"].append(float(tau1_var))
            monitors["tau2_var"].append(float(tau2_var))

            # taus: EB targets (only meaningful if eb_count>0; else just reuse previous or 0)
            monitors["tau4_target"].append(float(tau4_target if eb_count > 0 else 0.0))
            monitors["tau1_target"].append(float(tau1_target if eb_count > 0 else 0.0))
            monitors["tau2_target"].append(float(tau2_target if eb_count > 0 else 0.0))

            monitors["tau4_Eqf"].append(float(tau4_monitor * E_qf_total_mc))
            monitors["tau1_Eqf1"].append(float(tau1_monitor * E_qf1_total_mc))
            monitors["tau2_Eqf2"].append(float(tau2_monitor * E_qf2_total_mc))
            monitors["tau4_target_Eqf"].append(float(tgt4))
            monitors["tau1_target_Eqf1"].append(float(tgt1))
            monitors["tau2_target_Eqf2"].append(float(tgt2))

            # θ stats
            monitors["mu_norm"].append(mu_norm)
            monitors["mu_mean"].append(mu_mean)
            monitors["mu_std"].append(mu_std)
            monitors["sigma_mean"].append(sigma_mean)
            monitors["sigma_min"].append(sigma_min)
            monitors["sigma_max"].append(sigma_max)
        
        # ------------------- verbose
        if verbose and ((epoch+1) % 5 == 0 or (validate_dataloader is not None and improved)):
                
            val_str = f"  val_ELPD={val_elpd:.4f}" if val_elpd is not None else ""
            if validate_dataloader is None: 
                track = f"[ELBO] per-obs={elbo_per_obs:.6f}" 
            else: 
                track = f"ELPD trend → {'IMPROVED! Congrats' if improved else 'NOT improved!'} ({no_improve}/{patience_val})"
            print(
                f"\nIteration [{epoch+1}/{iterations}] "
                f"train={train_loss:.4f}  {val_str}  "
                f"S_train={mcmc_sample_train} S_val={mcmc_sample_val}  lr={lrs}  "
                f"σ̄={float(VI.sigma.mean()):.4f} σmin={float(VI.sigma.min()):.4f} σmax={float(VI.sigma.max()):.4f}  "
                f"β_KL={beta_kl:.2f}\n"
                f"τ₄={float(hyper_T['tau']):.5g}  τ₁={float(hyper_D['tau_1']):.5g}  τ₂={float(hyper_D['tau_2']):.5g}\n"
                f"(qτ means) τ₄≈{tau4_mean:.5g}  τ₁≈{tau1_mean:.5g}  τ₂≈{tau2_mean:.5g}  "
                f"|  (EB targets) τ₄*≈{tau4_target:.5g}  τ₁*≈{tau1_target:.5g}  τ₂*≈{tau2_target:.5g}\n"
                f"E_qf_total≈{E_qf_total_mc:.4f}  E_qf1≈{E_qf1_total_mc:.4f}  E_qf2≈{E_qf2_total_mc:.4f}\n"
                f"[monitor] τ₄·E_qf≈{tau4_monitor * E_qf_total_mc:.2f}  target≈{tgt4:.2f} | "
                f"τ₁·E_qf1≈{tau1_monitor * E_qf1_total_mc:.2f}  target≈{tgt1:.2f} | "
                f"τ₂·E_qf2≈{tau2_monitor * E_qf2_total_mc:.2f}  target≈{tgt2:.2f}\n",
                track
            )

        eb_E_qf_num, eb_count = 0.0, 0
        if (validate_dataloader is not None) and (no_improve >= patience_val): 
            print(f"Early stop @ epoch {epoch+1}: no val improvement for {patience_val} epochs.") 
            break

    with torch.no_grad():
        VI.mu.copy_(best_state["mu"]); VI.rho.copy_(best_state["rho"])
        if tau_nodes is not None:
            if tau_nodes.node4 and best_state["tau_nodes"]["node4_mu"] is not None:
                tau_nodes.node4.mu.copy_(best_state["tau_nodes"]["node4_mu"])
                tau_nodes.node4.log_sigma.copy_(best_state["tau_nodes"]["node4_log_sigma"])
            if tau_nodes.node2 and best_state["tau_nodes"]["node2_mu"] is not None:
                tau_nodes.node2.mu.copy_(best_state["tau_nodes"]["node2_mu"])
                tau_nodes.node2.log_sigma.copy_(best_state["tau_nodes"]["node2_log_sigma"])
            if tau_nodes.node1 and best_state["tau_nodes"]["node1_mu"] is not None:
                tau_nodes.node1.mu.copy_(best_state["tau_nodes"]["node1_mu"])
                tau_nodes.node1.log_sigma.copy_(best_state["tau_nodes"]["node1_log_sigma"])
            
            
    VI.set_model_params(VI.mu.detach())
    if was_training: nn.Module.train(model, True)

    return {
        "training_time": time.time() - start,
        "epochs_run": epoch + 1,
        "best_val": best_val,
        "loss_history": loss_history,
        "val_history": val_history if validate_dataloader is not None else None,
        "mu": VI.mu.detach(),
        "rho": VI.rho.detach(),
        "vi_model": VI,
        "monitor": monitors
    }


