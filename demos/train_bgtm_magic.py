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
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
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

            
            hyperparameters = {
                "transformation": {
                "sigma_a": 2.1, "sigma_b": 1e6,        #Ignored not used
                "RW2": { "tau_a": 2.0, "tau_b": 5.0 },  #E[λ_T] = 0.40  (smooth but not stiff)
                "RW1": { "tau_a": 10.0,"tau_b": 15.0 }  #Ignored not used
                },
                "decorrelation": {
                "sigma_a": 2.1, "sigma_b": 1e6,              # mean σ² = very small close to 0 (weak)
                "RW2": { "tau_a": 1.5, "tau_b": 30.0 },      # E[κ2] ≈ 0.05   (weak curvature smoothing)
                "RW1": { "tau_a": 1.5, "tau_b": 15.0 },      # E[κ1] ≈ 0.10   (light shrink to linear)
    }
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
                device="cpu",
                ## NEW ARGUMENTS ##
                inference = 'bayesian',
                hyperparameter=hyperparameters
            )

            model.to(device)
            
            adaptive_lasso_weights_matrix = False
            penalty_lasso_conditional_independence = False

            # train the joint model
            """_: dict[str, torch.Tensor] = model.train(
                train_dataloader=dataloader_train,
                validate_dataloader=dataloader_validate,
                iterations=200,
                optimizer="Adam",
                adaptive_lasso_weights_matrix=adaptive_lasso_weights_matrix,
                penalty_lasso_conditional_independence=penalty_lasso_conditional_independence,
                max_batches_per_iter=False,
                mcmc_sample=2,
                learning_rate=0.000001
                
            )"""
            
            
            
            output = model.train(
                train_dataloader=dataloader_train,
                validate_dataloader=dataloader_validate,
                hyperparameters=None,
                iterations=1000,
                verbose=True,
                lr=0.01,
                mcmc_sample_train=4,            # will ramp
                mcmc_sample_val=16,             # fixed & larger for stable eval
                mc_ramp_every=25,               # 4→8→16→32 at epochs 25/50/75
                mc_ramp_max=32,
                patience_val=15,                # early-stop patience
                min_delta=0.001,                # ~0.1% absolute of your loss scale
                rho_lr_multiplier=1.5,          # slightly faster variance adaption (optional)
                sched_factor=0.5, sched_patience=6, sched_threshold=1e-4,
            )

            
            import matplotlib.pyplot as plt

            # Example: replace this with your actual output['loss_history'] values
            loss_history = output['loss_history']
            val_loss = output['val_history']
            
            # Plot
            
            plt.figure(figsize=(8, 5))
            plt.plot(loss_history, label="Loss", linewidth=2)
            plt.plot(val_loss, label="Validation Loss", linewidth=2)
            plt.xlabel("Iteration / Epoch")
            plt.ylabel("Loss")
            plt.title("Loss Evolution")
            plt.legend()
            plt.grid(True)

            # Save the figure before showing it
            plt.savefig("bgtm_model_evaluation.png", dpi=300, bbox_inches="tight")

            plt.show()
            output['mu']
            
            #torch.save(
            #    model,
            #    os.path.join(
            #        "demos",
            #        "models",
            #        f"magic_group_{group}_decorr_{num_decorr_layers}_gtm_state_dict.pth",
            #    ),
            #)
