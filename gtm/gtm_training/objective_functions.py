import torch
from torch import Tensor
from torch.distributions import Normal
from typing import Literal, TYPE_CHECKING, Callable
from ..gtm_layers.layer_utils import bayesian_splines

from gtm.gtm_training.training_bayes import VI_Model

if TYPE_CHECKING:
    from ..gtm_model.gtm import GTM 


def log_likelihood(model: "GTM", samples, mean_loss=False):
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
    model: "GTM",
    samples: Tensor,
    penalty_params: Tensor,
    train_covariates: Tensor | bool=False,
    lambda_penalty_params: Tensor | bool = False,
    adaptive_lasso_weights_matrix: Tensor | bool=False,
    avg:bool=True,
    lambda_penalty_mode:Literal['square']="square",
    objective_type:Literal['negloglik']="negloglik",
) -> dict['str', Tensor]:

    if objective_type == "negloglik":
        
        return_dict_model_loss: dict[str, Tensor | float | None] = model.__log_likelihood_loss__(
            y=samples, mean_loss=True # True mean for init
        )  # covariate=train_covariates, mean_loss=True)
        
        loss: Tensor = return_dict_model_loss["negative_log_likelihood_data"]
    
    # elif objective_type == "vi":
    #    return_dict_model_loss = model.vi_loss(samples, covariate=train_covariates, mean_loss=True)
    #    loss = return_dict_model_loss["vi_loss"]

    if model.__module__ == "gtm.gtm_model.gtm":
        
        num_splines_per_layer: float = model.number_variables * (model.number_variables - 1) / 2
        
        # offdiagonal splines (additive effect)
        # if model.affine_layer is not False:
        #    num_splines_per_layer += model.number_variables * (model.number_variables-1) /2 #diagonal splines (multiplicative effect)
        
        num_params_decorr: float = num_splines_per_layer * model.number_decorrelation_layers * model.degree_decorrelation
        
        # num_params_trans = model.number_variables * model.num_trans_layers * model.degree_transformations
        # sum if we have variable degrees of transformation
        num_params_trans: int = sum(model.num_trans_layers * model.degree_transformations)
    else:
        # for pretrain marginals this is the case
        num_params_decorr: int = 1
        num_params_trans: int = 1
        
        # lambda_penalty_params is not False for the pretrain to work
    if (lambda_penalty_params is not False and torch.all(lambda_penalty_params == 0) == False):
        
        A: Tensor = return_dict_model_loss["lambda_matrix_global"]
        A_T:Tensor= A.transpose(1, 2)
        
        precision_matrix: Tensor = torch.matmul(A_T,A)
        
        if adaptive_lasso_weights_matrix is False:
            
            pen_lambda_lasso: Tensor = (
                (
                    lambda_penalty_params 
                    * (torch.square(precision_matrix).sum(0)) ** 0.5
                    )
                .tril(-1)
                .mean()
            )
        else:
            pen_lambda_lasso: Tensor = (
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
    
    pen_value_ridge: Tensor = penalty_params[0] * return_dict_model_loss["param_ridge_pen_global"] / num_params_decorr
    
    pen_first_ridge: Tensor = penalty_params[1] * return_dict_model_loss["first_order_ridge_pen_global"] / num_params_decorr
    
    pen_second_ridge: Tensor = penalty_params[2] * return_dict_model_loss["second_order_ridge_pen_global"] / num_params_decorr
    
    if num_params_trans > 0:
        pen_ctm_second_ridge: Tensor = penalty_params[3] * return_dict_model_loss["transformation_second_order_ridge_pen_global"] / num_params_trans
        
    else:
        pen_ctm_second_ridge = 0

    loss_with_penalties: Tensor = loss + pen_value_ridge + pen_first_ridge + pen_second_ridge + pen_lambda_lasso + pen_ctm_second_ridge
    

    return {
        "loss_with_penalties": loss_with_penalties,
        "loss_without_penalties": loss,
        "pen_ctm_second_ridge": pen_ctm_second_ridge,
        "pen_value_ridge": pen_value_ridge,
        "pen_first_ridge": pen_first_ridge,
        "pen_second_ridge": pen_second_ridge,
        "pen_lambda_lasso": pen_lambda_lasso,
    }



def unnormalized_posterior_computation(
    model: "GTM",
    samples,
    hyperparameter_transformation,
    hyperparameter_decorrelation, 
    sample_size
):
    
    #Likelihood
    return_dict_model_loss: Tensor = model.__log_likelihood_loss__(y=samples)
    nll: Tensor = return_dict_model_loss.get('negative_log_likelihood_data')
    nll=nll.mean()
    
    #Prior Transformation
    ntp = bayesian_splines.defining_prior(
        model=model, 
        hyperparameter= hyperparameter_transformation,
        is_transformation=True
        )
    
    if not model.transform_only:
    #Prior Decorrelation
        ndp = bayesian_splines.defining_prior(
            model=model, 
            hyperparameter=hyperparameter_decorrelation
            )
    else:
        ndp = torch.tensor(0.0, device=model.device, dtype=nll.dtype)
    
    sample_size = torch.as_tensor(sample_size, device=model.device, dtype=nll.dtype)
    
    neg_log_post= nll + (ntp['neg_log_prior_total'] + ndp)/sample_size # log \tilde p(θ, y) = - (NLL + priors)
    
    #print(f"Negative LogLike: {nll.item()}, \nnegative Transformation Prior: {ntp.item()}, \nnegative Decorrelation Prior: {ndp.item()}, \nweigthed negative LogPost: {neg_log_post.item()}")
    
    return {
        'neg_posterior':neg_log_post,
        'negative_decorrelation_prior': ndp,
        'negative_transformation_prior': ntp,
        'negative_log_lik': nll
        }

def bayesian_training_objective(
    model: "GTM",
    samples: Tensor,
    hyperparameter_transformation,
    hyperparameter_decorrelation,
    sample_size,
    objective_type:Literal['negloglik']="negloglik",
    
):
    
    if objective_type == "negloglik":
        
        return unnormalized_posterior_computation(
                model=model,
                samples=samples,
                hyperparameter_transformation=hyperparameter_transformation,
                hyperparameter_decorrelation= hyperparameter_decorrelation,
                sample_size=sample_size
                )