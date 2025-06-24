# GTM Package including GTM and the plot functions
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch

from gtm import *

if __name__ == "__main__":
    # Get Simulated Data
    from joe_simulated_data import create_simulated_data_joe_copula

    (
        dataloader_train,
        dataloader_validate,
        simulated_data_train,
        simulated_data_validate,
        simulated_data_test,
        loglik_true_train,
        loglik_true_validate,
        loglik_true_test,
        loglik_true_est_train,
        loglik_true_est_validate,
        loglik_true_est_test,
        log_likelihood_train_gaussian,
        log_likelihood_validate_gaussian,
        log_likelihood_test_gaussian,
    ) = create_simulated_data_joe_copula()

    ####### Define GTM Model and Train #######

    device = "cuda"

    model = GTM(
        transformation_spline_range=list([[-10], [10]]),
        decorrelation_spline_range=list([[-10], [10]]),
        degree_decorrelation=10,
        degree_transformations=10,
        num_decorr_layers=4,
        num_trans_layers=1,
        number_variables=2,
        spline_decorrelation="bernstein",
        spline_transformation="bernstein",
        device=device,
    )

    model = model.to(device)

    optimal_degrees_transformation, optimal_degrees_transformation_pvalues = (
        model.find_minimal_transformation_degrees(
            dataloader_train,
            dataloader_validate,
            iterations=100,
            degrees_try_list=list(range(5, 20, 5)),
            max_batches_per_iter=False,
        )
    )

    hyperparametertune = False
    if hyperparametertune == True:
        study = model.hyperparameter_tune_penalties(
            train_dataloader=dataloader_train,
            validate_dataloader=dataloader_validate,
            penvalueridge=[0],  # ["sample"],
            penfirstridge=["sample"],
            pensecondridge=["sample"],
            ctm_pensecondridge=[0],  # ["sample"],
            lambda_penalty_params=[0],  # ["sample"],
            train_covariates=False,
            validate_covariates=False,
            adaptive_lasso_weights_matrix=False,
            learning_rate=1,
            iterations=1000,
            patience=5,
            min_delta=1e-7,
            optimizer="LBFGS",
            lambda_penalty_mode="square",
            objective_type="negloglik",
            seperate_copula_training=False,
            max_batches_per_iter=False,
            tuning_mode="optuna",
            cross_validation_folds=False,
            random_state_KFold=42,
            device=device,
            pretrained_transformation_layer=False,
            n_trials=4,
            temp_folder=".",
            study_name=None,
        )

        penalty_params = torch.FloatTensor(
            [
                0,  # study.best_params["penvalueridge"],
                study.best_params["penfirstridge"],
                study.best_params["pensecondridge"],
                0,  # study.best_params["ctm_pensecondridge"]
            ]
        )
        adaptive_lasso_weights_matrix = False
        lambda_penalty_params = False
    else:
        penalty_params = torch.tensor([0.0000, 0.0000, 0.0000, 0.0000])
        adaptive_lasso_weights_matrix = False
        lambda_penalty_params = False

    # pretrain the marginal transformations
    _ = model.pretrain_tranformation_layer(
        dataloader_train,
        iterations=1000,
        max_batches_per_iter=False,
        penalty_params=penalty_params,
    )

    # train the joint model
    _ = model.__train__(
        train_dataloader=dataloader_train,
        validate_dataloader=dataloader_validate,
        iterations=1000,
        optimizer="LBFGS",
        penalty_params=penalty_params,
        adaptive_lasso_weights_matrix=adaptive_lasso_weights_matrix,
        lambda_penalty_params=lambda_penalty_params,
        max_batches_per_iter=False,
    )

    log_likelihood_train_gtm = (
        model.log_likelihood(simulated_data_train.to(device)).detach().cpu()
    )
    log_likelihood_validate_gtm = (
        model.log_likelihood(simulated_data_validate.to(device)).detach().cpu()
    )
    log_likelihood_test_gtm = (
        model.log_likelihood(simulated_data_test.to(device)).detach().cpu()
    )

    print(
        "KLD GTM      Train Data: ",
        np.round(torch.mean(loglik_true_train - log_likelihood_train_gtm).item(), 4),
    )
    print(
        "KLD Gaussian Train Data: ",
        np.round(
            torch.mean(loglik_true_train - log_likelihood_train_gaussian).item(), 4
        ),
    )
    print(
        "KLD Copula   Train Data: ",
        np.round(torch.mean(loglik_true_train - loglik_true_est_train).item(), 4),
    )

    print(
        "KLD GTM      Test  Data: ",
        np.round(torch.mean(loglik_true_test - log_likelihood_test_gtm).item(), 4),
    )
    print(
        "KLD Gaussian Test  Data: ",
        np.round(torch.mean(loglik_true_test - log_likelihood_test_gaussian).item(), 4),
    )
    print(
        "KLD Copula   Test  Data: ",
        np.round(torch.mean(loglik_true_test - loglik_true_est_test).item(), 4),
    )

    model.approximate_transformation_inverse()
    synthetic_samples = model.sample(2000)

    # _ = plot_splines(model.transformation,simulated_data_train)
    # plt.show()

    # plot_densities(synthetic_samples.numpy())
    # plot_densities(simulated_data_train.numpy())
    # plt.show()

    # _ = plot_splines(model.decorrelation_layers[0],simulated_data_train)
    # plt.show()
    # _ = plot_splines(model.decorrelation_layers[1],simulated_data_train)
    # _ = plot_splines(model.decorrelation_layers[2],simulated_data_train)
