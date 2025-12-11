####################################################################################################
####################################### Training GTMs ##############################################
# In this script we train GTMs for both the Hadron and Gamma classes of the MAGIC dataset.
# The GTMs are trained with different numbers of decorrelation layers, and the best penalites
# are selected using a validation set.
# Hyperparameter tuning uses the optuna package to generate proposed penalites.
# The GTM marginal degrees are based on the heuristic on demo_magic_marginal_transformations_degrees.ipynb
# The resulting models are analyzed in demo_magic_classification.ipynb.
# As the dataset is quite large we do not use a lasso penalty towards conditional independence
# Due to flexible marginals we use a marginal curvature penalty
####################################################################################################

import os
import torch
from optuna import Study
from torch import Tensor
from torch.utils.data import DataLoader

from dataset_helpers import Generic_Dataset
from load_analyze_magic_data_helpers import load_magic_data
from gtm import GTM

if __name__ == "__main__":

    for num_decorr_layers in [3, 4, 5, 6, 7, 8, 9]:
        for group in ["h", "g"]:
            print("Running for group:",group," with num_decorr_layers:",num_decorr_layers)

            device = "cuda" if torch.cuda.is_available() else "cpu"

            if group == "h":
                variable_degree_list: list[int]=[20, 35, 10, 10, 5, 30, 30, 40, 40, 10]
            elif group == "g":
                variable_degree_list: list[int]=[145, 150, 15, 5, 10, 30, 55, 60, 150, 10]

            y_train, y_validate, y_test = load_magic_data(
                group=group,
                train_portion=2 / 3,
                data_dims=10,
                poly_span_abs=12,
                cross_validation_folds=5,
                validation_fold_index=4,
                split_random_state=25,
            )
            
            N_train = y_train.size(0)
            N_validate = y_validate.size(0)

            # Create dataset and DataLoader
            dataset_train = Generic_Dataset(y_train)
            dataloader_train = DataLoader(dataset_train, batch_size=N_train)

            dataset_validate = Generic_Dataset(y_validate)
            dataloader_validate = DataLoader(dataset_validate, batch_size=N_validate)

            
            model_freq = GTM(
                number_variables=10,
                number_transformation_layers=1,
                number_decorrelation_layers=num_decorr_layers,
                degree_transformations=variable_degree_list,
                degree_decorrelation=30,
                spline_transformation="bspline",
                spline_decorrelation="bspline",
                transformation_spline_range=(-15, 15),
                device="cpu",
            )

            model_freq.to(device=device)

            study: Study | None = model_freq.hyperparameter_tune_penalties(
                train_dataloader=dataloader_train,
                validate_dataloader=dataloader_validate,
                penalty_decorrelation_ridge_param=None,
                penalty_decorrelation_ridge_first_difference="sample",
                penalty_decorrelation_ridge_second_difference="sample",
                penalty_transformation_ridge_second_difference="sample",
                penalty_lasso_conditional_independence=None,
                adaptive_lasso_weights_matrix=False,
                optimizer="LBFGS",
                learning_rate=1,
                iterations=2000,
                patience=20,
                min_delta=1e-7,
                seperate_copula_training=False,
                max_batches_per_iter=False,
                pretrained_transformation_layer=True,
                n_trials=40,
                temp_folder=".",
                study_name=None,
            )

            penalty_splines_params: Tensor = torch.FloatTensor(
                [
                    0,  # study.best_params["penalty_decorrelation_ridge_param"],
                    study.best_params["penalty_decorrelation_ridge_first_difference"],
                    study.best_params["penalty_decorrelation_ridge_second_difference"],
                    study.best_params["penalty_transformation_ridge_second_difference"],
                ]
            )
            adaptive_lasso_weights_matrix = False
            penalty_lasso_conditional_independence = False

            # pretrain the marginal transformations
            _ = model_freq.pretrain_transformation_layer(
                dataloader_train,
                iterations=1000,
                penalty_splines_params=penalty_splines_params,
            )

            # train the joint model
            _: dict[str, Tensor] = model_freq.train(
                train_dataloader=dataloader_train,
                validate_dataloader=dataloader_validate,
                iterations=1000,
                optimizer="LBFGS",
                penalty_splines_params=penalty_splines_params,
                adaptive_lasso_weights_matrix=adaptive_lasso_weights_matrix,
                penalty_lasso_conditional_independence=penalty_lasso_conditional_independence,
                max_batches_per_iter=False,
            )

            
            def gamma_from_mean_cv(mean, cv=1.0):
                a = 1.0 / (cv ** 2)
                b = a / mean
                return float(a), float(b)

            tau4_mean = 0.1   # oder was immer deinem GTM entspricht
            tau1_mean = 3.0
            tau2_mean = 6.0

            a4, b4 = gamma_from_mean_cv(tau4_mean, cv=1.0)
            a1, b1 = gamma_from_mean_cv(tau1_mean, cv=1.0)
            a2, b2 = gamma_from_mean_cv(tau2_mean, cv=1.0)

            hyperparameters = {
                "transformation": {
                    "sigma_a": 2.1, "sigma_b": 1e6,
                    "RW2": {"tau_a": a4, "tau_b": b4},
                    "RW1": {"tau_a": 1.0, "tau_b": 1.0},
                },
                "decorrelation": {
                    "sigma_a": 2.1, "sigma_b": 1e6,
                    "RW2": {"tau_a": a2, "tau_b": b2},
                    "RW1": {"tau_a": a1, "tau_b": b1},
                },
            }

            
            model = GTM(
                number_variables=10,
                number_transformation_layers=1,
                number_decorrelation_layers=num_decorr_layers,
                degree_transformations=variable_degree_list,
                degree_decorrelation=30,
                spline_transformation="bspline",
                spline_decorrelation="bspline",
                transformation_spline_range=(-15, 15),
                device="cuda",
                ## NEW ARGUMENTS ##
                inference = 'bayesian',
                hyperparameter=hyperparameters
            )           

            
            output = model.train(
                train_dataloader=dataloader_train,
                validate_dataloader=None,
                hyperparameters=None,
                iterations=800,
                mu_init=model_freq.state_dict(),
                optimizer="Adam",
                # More conservative learning rates ##
                #lr_mu = 1e-3,
                #lr_cholesky = 1e-4,
                #lr_rho = 3e-4,
                #lr_tau = 1.5e-3,
                # More conservative learning rates ##
                lr_mu       = 3e-4,
                lr_cholesky = 3e-5,
                lr_rho      = 1e-4,
                lr_tau      = 5e-4,
                
                mcmc_sample_train=8,            # will ramp
                mcmc_sample_val=32,             # fixed & larger for stable eval
                mc_ramp_every=20,               # 4→8→16→32 at epochs 25/50/75
                mc_ramp_max=64,
                
                patience=10,                # early-stop patience
                min_delta=0.00001, #with val data set 0.00001,                   # ~0.1% absolute of your loss scale
                
                sched_factor=0.5, 
                sched_patience=10, 
                sched_threshold=1e-4,
                sched_min_lr=[5e-5, 1e-5, 5e-5],
                #WARMING
                warm_tau_epochs = 15,
                warm_sigma_epochs = 15,  # try 5–10
                
                #Optimization method
                beta_kl_start=0.1,    # try 1.5–3.0
                beta_kl_anneal_epochs = 60,  # how fast to decay to 1.0
                
                # --- τ-VI toggles (key difference) ---
                tau_vi_mode = "always", #"off" | "after_warm" | "always"
                tau_kl_beta =0.2,
                tau_vi_sigma_init = 0.05,
                
                # --- VI convergence (no-val) ---
                conv_use_ema= True,
                conv_window_size = 50,   # used if conv_use_ema=False
                conv_tol = 0.005, #0.001,      # absolute ELBO change per-obs
                conv_min_epochs = 50,   # don't stop too early
                conv_ema_beta = 0.9,  # if conv_use_ema=True

            )

            run_id = f"magic_group_{group}_decorr_{num_decorr_layers}"

            ckpt = {
                "model_state_dict": model.state_dict(),     # GTM parameters (freq+Bayes init)
                "vi_model": output["vi_model"],             # full VI_Model object
                "tau_nodes": output["tau_nodes"],           # full TauPack (GammaTauNode set)
                "hyper_T": output["hyper_T"],               # transformation hyperparameters
                "hyper_D": output["hyper_D"],               # decorrelation hyperparameters
                "decor_present": output["decor_present"],   # bool
                "monitor": output["monitor"],               # training diagnostics
                "loss_history": output["loss_history"],
                "val_history": output["val_history"],
            }

            torch.save(
                ckpt,
                os.path.join("demos", "model_vi", f"{run_id}_bgtm_vi_checkpoint.pth"),
            )
