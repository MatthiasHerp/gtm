import torch
from torch.distributions import Normal


def log_likelihood(model, samples, mean_loss=False):
    # train_covariates=False, train=True, evaluate=True,

    return_dict_nf_mctm = model.forward(
        samples
    )  # , covariate=train_covariates, train=train, evaluate=evaluate)

    log_likelihood_latent = Normal(0, 1).log_prob(return_dict_nf_mctm["output"])

    if mean_loss:
        log_likelihood_data = log_likelihood_latent + return_dict_nf_mctm["log_d"]
        log_likelihood_data = torch.mean(log_likelihood_data)
    else:
        if model.number_variables > 1:  # for 1D TM Model
            log_likelihood_data = torch.sum(
                log_likelihood_latent + return_dict_nf_mctm["log_d"], 1
            )
        else:
            log_likelihood_data = log_likelihood_latent + return_dict_nf_mctm["log_d"]

    return_dict_nf_mctm["log_likelihood_data"] = log_likelihood_data

    return return_dict_nf_mctm


def training_objective(
    model,
    samples,
    penalty_params,
    train_covariates=False,
    lambda_penalty_params: torch.Tensor = False,
    adaptive_lasso_weights_matrix=False,
    avg=True,
    lambda_penalty_mode="square",
    objective_type="negloglik",
):
    if objective_type == "negloglik":
        return_dict_model_loss = model.__log_likelihood_loss__(
            samples, mean_loss=True
        )  # covariate=train_covariates, mean_loss=True)
        loss = return_dict_model_loss["negative_log_likelihood_data"]
    # elif objective_type == "vi":
    #    return_dict_model_loss = model.vi_loss(samples, covariate=train_covariates, mean_loss=True)
    #    loss = return_dict_model_loss["vi_loss"]

    if model.__module__ == "gtm.gtm_model.gtm":
        num_splines_per_layer = (
            model.number_variables * (model.number_variables - 1) / 2
        )  # offdiagonal splines (additive effect)
        # if model.affine_layer is not False:
        #    num_splines_per_layer += model.number_variables * (model.number_variables-1) /2 #diagonal splines (multiplicative effect)
        num_params_decorr = (
            num_splines_per_layer
            * model.number_decorrelation_layers
            * model.degree_decorrelation
        )

        # num_params_trans = model.number_variables * model.num_trans_layers * model.degree_transformations
        # sum if we have variable degrees of transformation
        num_params_trans = sum(model.num_trans_layers * model.degree_transformations)
    else:
        # for pretrain marginals this is the case
        num_params_decorr = 1
        num_params_trans = 1
        # lambda_penalty_params is not False for the pretrain to work
    if (
        lambda_penalty_params is not False
        and torch.all(lambda_penalty_params == 0) == False
    ):
        precision_matrix = torch.matmul(
            torch.transpose(return_dict_model_loss["lambda_matrix_global"], 1, 2),
            return_dict_model_loss["lambda_matrix_global"],
        )
        if adaptive_lasso_weights_matrix is False:
            pen_lambda_lasso = (
                (lambda_penalty_params * (torch.square(precision_matrix).sum(0)) ** 0.5)
                .tril(-1)
                .mean()
            )
        else:
            pen_lambda_lasso = (
                (
                    lambda_penalty_params
                    * adaptive_lasso_weights_matrix
                    * (torch.square(precision_matrix).sum(0)) ** 0.5
                )
                .tril(-1)
                .mean()
            )
    else:
        pen_lambda_lasso = 0

    if num_params_decorr == 0:
        num_params_decorr = 1  # for no division errors here
    pen_value_ridge = (
        penalty_params[0]
        * return_dict_model_loss["param_ridge_pen_global"]
        / num_params_decorr
    )
    pen_first_ridge = (
        penalty_params[1]
        * return_dict_model_loss["first_order_ridge_pen_global"]
        / num_params_decorr
    )
    pen_second_ridge = (
        penalty_params[2]
        * return_dict_model_loss["second_order_ridge_pen_global"]
        / num_params_decorr
    )
    if num_params_trans > 0:
        pen_ctm_second_ridge = (
            penalty_params[3]
            * return_dict_model_loss["transformation_second_order_ridge_pen_global"]
            / num_params_trans
        )
    else:
        pen_ctm_second_ridge = 0

    loss_with_penalties = (
        loss
        + pen_value_ridge
        + pen_first_ridge
        + pen_second_ridge
        + pen_lambda_lasso
        + pen_ctm_second_ridge
    )

    return {
        "loss_with_penalties": loss_with_penalties,
        "loss_without_penalties": loss,
        "pen_ctm_second_ridge": pen_ctm_second_ridge,
        "pen_value_ridge": pen_value_ridge,
        "pen_first_ridge": pen_first_ridge,
        "pen_second_ridge": pen_second_ridge,
        "pen_lambda_lasso": pen_lambda_lasso,
    }
