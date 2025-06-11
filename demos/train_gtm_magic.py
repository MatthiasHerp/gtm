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

from gtm import *
from demos.load_analyze_magic_data_helpers import *
from demos.dataset_helpers import Generic_Dataset
from torch.utils.data import DataLoader
import os

if __name__ == "__main__":

    for num_decorr_layers in [3, 4, 5, 6, 7, 8, 9]:
        for group in ["h", "g"]:
            print(
                "Running for group:",
                group,
                " with num_decorr_layers:",
                num_decorr_layers,
            )

            device = "cuda" if torch.cuda.is_available() else "cpu"

            if group == "h":
                variable_degree_list = [20, 35, 10, 10, 5, 30, 30, 40, 40, 10]
            elif group == "g":
                variable_degree_list = [145, 150, 15, 5, 10, 30, 55, 60, 150, 10]

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
            )

            model.to(device)

            study = model.hyperparameter_tune_penalties(
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

            penalty_splines_params = torch.FloatTensor(
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
            _ = model.pretrain_transformation_layer(
                dataloader_train,
                iterations=1000,
                penalty_splines_params=penalty_splines_params,
            )

            # train the joint model
            _ = model.train(
                train_dataloader=dataloader_train,
                validate_dataloader=dataloader_validate,
                iterations=1000,
                optimizer="LBFGS",
                penalty_splines_params=penalty_splines_params,
                adaptive_lasso_weights_matrix=adaptive_lasso_weights_matrix,
                penalty_lasso_conditional_independence=penalty_lasso_conditional_independence,
                max_batches_per_iter=False,
            )

            torch.save(
                model,
                os.path.join(
                    "demos",
                    "models",
                    f"magic_group_{group}_decorr_{num_decorr_layers}_gtm_state_dict.pth",
                ),
            )
