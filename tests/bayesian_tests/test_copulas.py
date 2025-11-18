import json
import numpy as np
import torch
from torch.utils.data import DataLoader

import pyvinecopulib as pv

from gtm import GTM   # and your Bayesian GTM class if it's also in gtm.py
from dataset_helpers import Generic_Dataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------
# 1) Helper: build true copula from a config
# ---------------------------------------------------------------------
def build_true_copula(cfg):
    """
    cfg: dict with keys: name, family, params, rotation
    params can be a callable () -> np.ndarray or a np.ndarray itself
    """
    params = cfg["params"]() if callable(cfg["params"]) else cfg["params"]
    return pv.Bicop(
        family=cfg["family"],
        parameters=params,
        rotation=cfg.get("rotation", 0),
    )


# ---------------------------------------------------------------------
# 2) Helper: simulate data from a copula + N(0,1) marginals
# ---------------------------------------------------------------------
def simulate_from_copula(copula_pv, n_train=2000, n_val=2000, n_test=20000):
    # TRAIN
    u_train = copula_pv.simulate(n=n_train)            # np array (n, d)
    x_train = torch.distributions.Normal(0, 1).icdf(
        torch.tensor(u_train, dtype=torch.float32)
    )#.to(device)

    # VALIDATE
    u_val = copula_pv.simulate(n=n_val)
    x_val = torch.distributions.Normal(0, 1).icdf(
        torch.tensor(u_val, dtype=torch.float32)
    )#.to(device)

    # TEST
    u_test = copula_pv.simulate(n=n_test)
    x_test = torch.distributions.Normal(0, 1).icdf(
        torch.tensor(u_test, dtype=torch.float32)
    )#.to(device)

    # TRUE JOINT LOG-LIK (known copula + known N(0,1) marginals)
    def true_loglik(u, x):
        log_c = np.log(copula_pv.pdf(u))
        log_marg = torch.distributions.Normal(0, 1).log_prob(x).sum(1)
        return torch.tensor(log_c) + log_marg#.to(device)

    loglik_true_train = true_loglik(u_train, x_train)
    loglik_true_val = true_loglik(u_val, x_val)
    loglik_true_test = true_loglik(u_test, x_test)

    return {
        "x_train": x_train,
        "x_val": x_val,
        "x_test": x_test,
        "u_train": u_train,
        "u_val": u_val,
        "u_test": u_test,
        "log_true_train": loglik_true_train,
        "log_true_val": loglik_true_val,
        "log_true_test": loglik_true_test,
    }


# ---------------------------------------------------------------------
# 3) Helper: fit a *copula model* to the simulated data (your Copula baseline)
# ---------------------------------------------------------------------
def fit_copula_model(u_train, u_val, u_test, x_train, x_val, x_test, family, rotation):
    """
    This mimics the block in your Cell 4 where you do copula_pv_est.fit(...)
    and combine it with empirical marginals.
    """
    copula_pv_est = pv.Bicop(family=family, rotation=rotation)
    copula_pv_est.fit(u_train)

    means = x_train.mean(0)
    stds = x_train.std(0)  # (you used var; strictly speaking this should be std)

    def est_loglik(u, x):
        log_c = np.log(copula_pv_est.pdf(u))
        log_m = torch.distributions.Normal(means, stds).log_prob(x).sum(1)
        return torch.tensor(log_c) + log_m

    log_train = est_loglik(u_train, x_train)
    log_val = est_loglik(u_val, x_val)
    log_test = est_loglik(u_test, x_test)

    return log_train, log_val, log_test


# ---------------------------------------------------------------------
# 4) Helper: fit Gaussian baseline
# ---------------------------------------------------------------------
def fit_gaussian_baseline(x_train, x_val, x_test):
    mean_mvn = x_train.mean(0)
    cov_mvn = x_train.T.cov()
    mvn = torch.distributions.MultivariateNormal(
        loc=mean_mvn, covariance_matrix=cov_mvn
    )
    log_train = mvn.log_prob(x_train)
    log_val = mvn.log_prob(x_val)
    log_test = mvn.log_prob(x_test)
    return log_train, log_val, log_test


# ---------------------------------------------------------------------
# 5) Helper: fit GTM (frequentist)
#    >>> HERE you paste your GTM block from the notebook <<<
# ---------------------------------------------------------------------
def fit_gtm(x_train, x_val, x_test):
    # --- DataLoaders like in your notebook ---
    ds_train = Generic_Dataset(x_train)
    dl_train = DataLoader(ds_train, batch_size=len(x_train))

    ds_val = Generic_Dataset(x_val)
    dl_val = DataLoader(ds_val, batch_size=len(x_val))

    # --- Instantiate GTM exactly as in your notebook ---
    model_freq = GTM(
        number_variables=2,
        number_transformation_layers=1,
        number_decorrelation_layers=3,
        degree_transformations=10,
        degree_decorrelation=40,
        spline_transformation="bspline",
        spline_decorrelation="bspline",
        transformation_spline_range=(-10, 10),
        device=device,
    )

    # ---- hyperparameter tuning exactly as you already do ----
    # (Copy from your notebook: hyperparameter_tune_penalties call)
    study = model_freq.hyperparameter_tune_penalties(
        train_dataloader=dl_train,
        validate_dataloader=dl_val,
        penalty_decorrelation_ridge_param = None,
        penalty_decorrelation_ridge_first_difference = "sample",
        penalty_decorrelation_ridge_second_difference = "sample",
        penalty_transformation_ridge_second_difference = None,
        penalty_lasso_conditional_independence = None,
        adaptive_lasso_weights_matrix=False,
        optimizer="LBFGS",
        learning_rate=1,
        iterations=2000,
        patience=5,
        min_delta=1e-7,
        seperate_copula_training=False,
        max_batches_per_iter=False,
        pretrained_transformation_layer=True,
        n_trials=30,
        temp_folder=".",
        study_name=None
    )

    penalty_splines_params = torch.FloatTensor(
        [
            0.0,
            study.best_params["penalty_decorrelation_ridge_first_difference"],
            study.best_params["penalty_decorrelation_ridge_second_difference"],
            0.0,
        ]
    )
    
    adaptive_lasso_weights_matrix = False
    penalty_lasso_conditional_independence=False

    # ---- pretrain and train (copy your calls) ----
    _ = model_freq.pretrain_transformation_layer(
        dl_train,
        iterations=1000,
        penalty_splines_params=penalty_splines_params,
        max_batches_per_iter=False,
    )

    _ = model_freq.train(
        train_dataloader=dl_train,
        validate_dataloader=dl_val,
        iterations=1000,
        optimizer="LBFGS",
        penalty_splines_params=penalty_splines_params,
        adaptive_lasso_weights_matrix=False,
        penalty_lasso_conditional_independence=False,
        max_batches_per_iter=False,
    )

    # ---- log-likelihood on all sets ----
    log_train = model_freq.log_likelihood(x_train).detach().cpu()
    log_val = model_freq.log_likelihood(x_val).detach().cpu()
    log_test = model_freq.log_likelihood(x_test).detach().cpu()

    return log_train, log_val, log_test, model_freq


# ---------------------------------------------------------------------
# 6) Helper: fit BGTM (full Bayesian VI)
#    >>> HERE you paste your BGTM + VI config from the notebook <<<
# ---------------------------------------------------------------------
def fit_bgtm(x_train, x_val, x_test):
    # The BGTM uses mini-batches in your notebook:
    ds_train = Generic_Dataset(x_train)
    dl_train = DataLoader(
        ds_train,
        batch_size=35,  # or your batch size
        shuffle=True,
        num_workers=8,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=4,
    )

    ds_val = Generic_Dataset(x_val)
    dl_val = DataLoader(
        ds_val,
        batch_size=30,  # or your batch size
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2,
    )

    # ---- hyperparameters dict (copy from your Cell 10) ----
    hyperparameters = {
        "transformation": {
            "sigma_a": 2.1,
            "sigma_b": 1e6,
            "RW2": {"tau_a": 1e-3, "tau_b": 5e-3},
            "RW1": {"tau_a": 10.0, "tau_b": 15.0},
            "tau_4_update_eta": 0.15,
        },
        "decorrelation": {
            "sigma_a": 2.1,
            "sigma_b": -1e6,
            "RW2": {"tau_a": 1.1, "tau_b": 0.1},
            "RW1": {"tau_a": 1.1, "tau_b": 0.2},
        },
    }

    # ---- instantiate your Bayesian GTM class ----
    # Replace `BayesianGTM` with your actual class name
    model_bayes = GTM(
        number_variables=2,
        number_transformation_layers=1,
        number_decorrelation_layers=3,
        degree_transformations=10,
        degree_decorrelation=40,
        spline_transformation="bspline",
        spline_decorrelation="bspline",
        transformation_spline_range=(-10, 10),
        device=device,
        inference = 'bayesian',
        hyperparameter=hyperparameters,
    )

    # ---- VI / training config (copy from your notebook) ----
    output = model_bayes.train(
        train_dataloader=dl_train,
        validate_dataloader=dl_val,
        iterations=800,
        #verbose=True,
        learning_rate=0.0015,
        mcmc_sample_train=4,            # will ramp
        mcmc_sample_val=32,             # fixed & larger for stable eval
        mc_ramp_every=25,               # 4→8→16→32 at epochs 25/50/75
        mc_ramp_max=32,
                
        patience=10,                # early-stop patience
        min_delta=0.00001, #with val data set 0.00001,                   # ~0.1% absolute of your loss scale
                
        rho_lr_multiplier=0.7,          # slightly faster variance adaption (optional)
        sched_factor=0.5, 
        sched_patience=6,
        sched_threshold=1e-4,
        #WARMING
        warm_tau_epochs = 10,
        warm_sigma_epochs = 30,  # try 5–10
                
        #Optimization method
        beta_kl_start=0.5,    # try 1.5–3.0
        beta_kl_anneal_epochs = 20,  # how fast to decay to 1.0
                
        # --- τ-VI toggles (key difference) ---
        tau_vi_mode = "always", #"off" | "after_warm" | "always"
        tau_kl_beta =2.0,
        tau_vi_sigma_init = 0.15,
                
        # --- VI convergence (no-val) ---
        conv_use_ema= True,
        conv_window_size = 20,   # used if conv_use_ema=False
        conv_tol = 0.01,      # absolute ELBO change per-obs
        conv_min_epochs = 10,   # don't stop too early
        conv_ema_beta = 0.9,  # if conv_use_ema=True
    )

    # (Optional) you can also inspect output["monitor"], output["loss_history"], etc.

    log_train = model_bayes.log_likelihood(x_train).detach().cpu()
    log_val = model_bayes.log_likelihood(x_val).detach().cpu()
    log_test = model_bayes.log_likelihood(x_test).detach().cpu()

    return log_train, log_val, log_test, model_bayes


# ---------------------------------------------------------------------
# 7) Run one full experiment for a given copula
# ---------------------------------------------------------------------
def run_experiment_for_copula(cfg, n_train=2000, n_val=2000, n_test=20000):
    print(f"\n=== Running experiment for copula: {cfg['name']} ===")

    copula_true = build_true_copula(cfg)

    # --- simulate from true model ---
    data = simulate_from_copula(
        copula_true, n_train=n_train, n_val=n_val, n_test=n_test
    )
    x_train, x_val, x_test = data["x_train"], data["x_val"], data["x_test"]
    u_train, u_val, u_test = data["u_train"], data["u_val"], data["u_test"]
    log_true_train = data["log_true_train"]
    log_true_test = data["log_true_test"]

    # --- copula baseline (estimated) ---
    log_cop_train, log_cop_val, log_cop_test = fit_copula_model(
        u_train,
        u_val,
        u_test,
        x_train,
        x_val,
        x_test,
        family=cfg["family"],
        rotation=cfg.get("rotation", 0),
    )

    # --- Gaussian baseline ---
    log_gauss_train, log_gauss_val, log_gauss_test = fit_gaussian_baseline(
        x_train, x_val, x_test
    )

    # --- GTM (frequentist) ---
    log_gtm_train, log_gtm_val, log_gtm_test, model_gtm = fit_gtm(
        x_train, x_val, x_test
    )

    # --- BGTM (full Bayes VI) ---
    log_bgtm_train, log_bgtm_val, log_bgtm_test, model_bgtm = fit_bgtm(
        x_train, x_val, x_test
    )

    # --- Approximate KLs: E_true[ log p_true - log p_model ] ---
    def approx_kl(log_true, log_model):
        return torch.mean(log_true - log_model).item()

    results = {
        "copula_name": cfg["name"],
        "KL_train_BGTM": approx_kl(log_true_train, log_bgtm_train),
        "KL_train_GTM": approx_kl(log_true_train, log_gtm_train),
        "KL_train_Gaussian": approx_kl(log_true_train, log_gauss_train),
        "KL_train_Copula": approx_kl(log_true_train, log_cop_train),
        "KL_test_BGTM": approx_kl(log_true_test, log_bgtm_test),
        "KL_test_GTM": approx_kl(log_true_test, log_gtm_test),
        "KL_test_Gaussian": approx_kl(log_true_test, log_gauss_test),
        "KL_test_Copula": approx_kl(log_true_test, log_cop_test),
    }

    print("Train KLs:")
    print(
        f"  BGTM     : {results['KL_train_BGTM']:.4f}\n"
        f"  GTM      : {results['KL_train_GTM']:.4f}\n"
        f"  Gaussian : {results['KL_train_Gaussian']:.4f}\n"
        f"  Copula   : {results['KL_train_Copula']:.4f}"
    )

    print("Test KLs:")
    print(
        f"  BGTM     : {results['KL_test_BGTM']:.4f}\n"
        f"  GTM      : {results['KL_test_GTM']:.4f}\n"
        f"  Gaussian : {results['KL_test_Gaussian']:.4f}\n"
        f"  Copula   : {results['KL_test_Copula']:.4f}"
    )

    return results


# ---------------------------------------------------------------------
# 8) Main: list of copulas to compare
# ---------------------------------------------------------------------
if __name__ == "__main__":
    # You can add as many as you like here
    copula_configs = [
        # Elliptical
        {
            "name": "Gaussian_rho_0.7",
            "family": pv.BicopFamily.gaussian,
            "params": lambda: np.array([[0.7]]),
            "rotation": 0,
        },
        {
            "name": "t_rho_0.7_nu_4",
            "family": pv.BicopFamily.student,
            "params": lambda: np.array([[0.7],[4.0]]),  # (ρ, ν)
            "rotation": 0,
        },
        # Archimedean
        {
            "name": "Clayton_theta_2",
            "family": pv.BicopFamily.clayton,
            "params": lambda: np.array([[2.0]]),
            "rotation": 0,
        },
        {
            "name": "Gumbel_2.5",
            "family": pv.BicopFamily.gumbel,
            "params": lambda: np.array([[2.5]])
        },
        
        # Rotated
        {
             "name": "Joe_theta_2.5_rot90",
             "family": pv.BicopFamily.joe,
             "params": lambda: np.array([[2.5]]),
             "rotation": 90,
        },
        {
            "name": "Gumbel_rot90_2",
            "family": pv.BicopFamily.gumbel,
            "params": lambda: np.array([[2.0]]),
            "rotation": 90
        },
        # BB families (complex)
        {
            "name": "BB1_2_3", 
            "family": pv.BicopFamily.bb1,
            "params": lambda: np.array([[2.0],[3.0]])
        },
        {
            "name": "BB7_1.2_2.0",
            "family": pv.BicopFamily.bb7,
            "params": lambda: np.array([[1.2],[2.0]])
            },
    ]

    all_results = []
    for cfg in copula_configs:
        res = run_experiment_for_copula(cfg)
        all_results.append(res)

    print("\n=== Summary table ===")
    header = [
        "copula_name",
        "KL_test_BGTM",
        "KL_test_GTM",
        "KL_test_Gaussian",
        "KL_test_Copula",
    ]
    print("\t".join(header)) 
    for r in all_results:
        print(
            f"{r['copula_name']}\t"
            f"{r['KL_test_BGTM']:.4f}\t"
            f"{r['KL_test_GTM']:.4f}\t"
            f"{r['KL_test_Gaussian']:.4f}\t"
            f"{r['KL_test_Copula']:.4f}"
        )
    
    with open("tests/bayesian_tests/results_summary.json", "w") as f:
        json.dump(all_results, f, indent=4)

    print("\nSaved summary to results_summary.json")

