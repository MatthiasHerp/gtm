import pyvinecopulib as pv
import numpy as np, torch
from torch.utils.data import DataLoader
from dataset_helpers import Generic_Dataset
from gtm import GTM
import matplotlib.pyplot as plt
import math

# ----- data: true independency, standard Gaussian marginals -----
N_train, N_val, N_test = 2000, 2000, 20000
cop_indep = pv.Bicop(family=pv.BicopFamily.indep)   # independence copula

def simulate_block(n):
    U = cop_indep.simulate(n=n)                      # U in (0,1)^2
    X = torch.distributions.Normal(0,1).icdf(torch.tensor(U)).float()
    return U, X

U_tr, X_tr = simulate_block(N_train)
U_va, X_va = simulate_block(N_val)
U_te, X_te = simulate_block(N_test)

# true logliks
def loglik_true(U, X, means=None, stds=None):
    # copula density = 1 under independence
    if means is None: means = torch.zeros(X.shape[1])
    if stds  is None: stds  = torch.ones(X.shape[1])
    log_marg = torch.distributions.Normal(means, stds).log_prob(X).sum(1)
    return log_marg

loglik_true_train = loglik_true(U_tr, X_tr)
loglik_true_val   = loglik_true(U_va, X_va)
loglik_true_test  = loglik_true(U_te, X_te)

# dataloaders
dl_tr = DataLoader(Generic_Dataset(X_tr), batch_size=N_train)
dl_va = DataLoader(Generic_Dataset(X_va), batch_size=N_val)

# ----- FREQUENTIST: transform-only, no decorrelation -----
model_freq = GTM(
    number_variables=2,
    number_transformation_layers=1,
    number_decorrelation_layers=0, 
    degree_transformations=10,
    spline_transformation="bspline",
    transformation_spline_range=(-10, 10),
    device="cpu"
)
model_freq.transform_only = True

study = model_freq.hyperparameter_tune_penalties( 
        train_dataloader = dl_tr,
        validate_dataloader = dl_va,
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

# tune penalties only for transformation layer if your API requires a vector
penalty_splines_params = torch.FloatTensor([
    0, #study.best_params["penalty_decorrelation_ridge_param"],
    study.best_params["penalty_decorrelation_ridge_first_difference"],
    study.best_params["penalty_decorrelation_ridge_second_difference"],
    0 #study.best_params["penalty_transformation_ridge_second_difference"]
    ])


out_freq = model_freq.pretrain_transformation_layer(
    dl_tr,
    iterations=1000,
    max_batches_per_iter=False,
    penalty_splines_params=penalty_splines_params
)

#### Training Reference First Transformation ###
adaptive_lasso_weights_matrix = False
penalty_lasso_conditional_independence=False

joint_model = model_freq.train(
    train_dataloader=dl_tr,
    validate_dataloader=dl_va,
    iterations=1000,
    optimizer="LBFGS",
    penalty_splines_params=penalty_splines_params,
    adaptive_lasso_weights_matrix=adaptive_lasso_weights_matrix,
    penalty_lasso_conditional_independence=penalty_lasso_conditional_independence,
    max_batches_per_iter=False
    )

# evaluate: transform X -> Z and check standard normality per margin
with torch.no_grad():
    Z_tr = model_freq.forward(X_tr)['output']  # whatever returns the transformed marginals
    Z_va = model_freq.forward(X_va)['output']

def ks_normality(z):
    # fast, vectorized KS vs N(0,1). Replace by your util if you have one.
    from scipy import stats
    return [stats.kstest(z[:,j].cpu().numpy(), 'norm')[1] for j in range(z.shape[1])]

pvals_tr = ks_normality(Z_tr)

pvals_va = ks_normality(Z_va)
print("Freq KS p-values (train,val):", pvals_tr, pvals_va)


# ----- BAYESIAN: same architecture, VI -----
model_bayes = GTM(
    number_variables=2,
    number_transformation_layers=1,
    number_decorrelation_layers=0,
    degree_transformations=10,
    spline_transformation="bspline",
    transformation_spline_range=(-10, 10),
    device="cpu",
    inference='bayesian',
    hyperparameter={
        "transformation": {
            "sigma_a": 2.1, "sigma_b": 1e6,        #Ignored not used
            # RW2 = tau_4 from Carla & Kneib; keep weak but proper
            "RW2": {"tau_a": 0.8, "tau_b": 0.3},
            "RW1": { "tau_a": 1.5, "tau_b": 0.01 },
        },
        "decorrelation": { # decorrelation priors ignored since layers=0
            "sigma_a": 2.1, "sigma_b": -1e6,              # mean σ² = very small close to 0 (weak)
            "RW2": { "tau_a": 1.5, "tau_b": 0.01 },      # E[κ2] ≈ 0.05   (weak curvature smoothing)
            "RW1": { "tau_a": 1.5, "tau_b": 0.01 },      # E[κ1] ≈ 0.10   (light shrink to linear)
            }
    }
)
model_bayes.transform_only = True

out_bayes = model_bayes.pretrain_transformation_layer(
    train_dataloader=dl_tr,
    validate_dataloader=dl_va,
    iterations=100,
    learning_rate=0.001,
    mcmc_sample_train=4,
    mcmc_sample_val=16,
    mc_ramp_every=10,
    mc_ramp_max=64,
    patience=15,
    min_delta=1e-3,
    rho_lr_multiplier=2,
    sched_factor=0.5, sched_patience=6, sched_threshold=1e-4,
)


output = model_bayes.train(
    train_dataloader=dl_tr,
    validate_dataloader=dl_va,
    hyperparameters=None,
    iterations=100,
    #verbose=True,
    learning_rate=0.01,
    mcmc_sample_train=4,            # will ramp
    mcmc_sample_val=16,             # fixed & larger for stable eval
    mc_ramp_every=10,               # 4→8→16→32 at epochs 25/50/75
    mc_ramp_max=64,
    patience=15,                # early-stop patience
    min_delta=1e-3,                # ~0.1% absolute of your loss scale
    rho_lr_multiplier=2,          # slightly faster variance adaption (optional)
    sched_factor=0.5, sched_patience=6, sched_threshold=1e-4,
    )

with torch.no_grad():
    Z_tr_b = model_bayes.forward(X_tr)['output']
    Z_va_b = model_bayes.forward(X_va)['output']

print("Bayes KS p-values (train,val):", ks_normality(Z_tr_b), ks_normality(Z_va_b))


def qq_plot(z, label):
    from scipy import stats
    sm = stats.norm.ppf(np.linspace(0.01, 0.99, 100))
    emp = np.quantile(z.detach().cpu().numpy(), np.linspace(0.01, 0.99, 100))
    plt.plot(sm, emp, label=label)
    plt.plot(sm, sm, 'k--')

plt.figure(figsize=(5,5))
qq_plot(model_freq.forward(X_tr)['output'][:,1], 'freq dim2')
qq_plot(model_bayes.forward(X_tr)['output'][:,1], 'bayes dim2')
plt.legend()
plt.xlabel('Theoretical')
plt.ylabel('Empirical')
plt.title('Q–Q Plot of Transformed Margin 2')
plt.show()




def empirical_hpd_level(theta_hat, sample_fn, logq_fn, S=5000, device="cpu"):
    with torch.no_grad():
        TH = theta_hat.to(device)
        Lhat = logq_fn(TH)
        cnt = 0
        for _ in range(S):
            th = sample_fn()
            if logq_fn(th) <= Lhat: cnt += 1
        return cnt / S

def meanfield_logq(theta, mu, sigma):
    # theta, mu, sigma: 1D tensors
    const = -0.5*theta.numel()*math.log(2*math.pi) - torch.log(sigma).sum()
    return const - 0.5*(((theta-mu)/sigma)**2).sum()

def meanfield_sample(mu, sigma):
    eps = torch.randn_like(mu)
    return mu + sigma*eps




