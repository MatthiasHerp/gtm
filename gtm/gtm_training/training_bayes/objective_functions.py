import torch
from torch import Tensor
from torch.distributions import Normal
from typing import Literal, TYPE_CHECKING
from ...gtm_layers.layer_utils import bayesian_splines
if TYPE_CHECKING:
    from gtm_model.gtm import GTM 

def unnormalized_posterior_computation(
    model: "GTM",
    samples,
    hyperparameter_transformation,
    hyperparameter_decorrelation, 
    B,
    N_total
):
    
    #Likelihood
    return_dict_model_loss=model.__log_likelihood_loss__(y=samples)
    nll_batch=return_dict_model_loss.get('negative_log_likelihood_data').sum()
    
    #Prior Transformation
    ntp = bayesian_splines.defining_prior(
        model=model, 
        hyperparameter= hyperparameter_transformation,
        is_transformation=True
        )
    
    #Prior Decorrelation
    if model.transform_only or model.number_decorrelation_layers == 0:
        ndp={
            "neg_log_prior_total": torch.tensor(0.0, device=model.device, dtype=nll_batch.dtype),
            "qf1": torch.tensor(0.0, device=model.device, dtype=nll_batch.dtype),
            "qf2": torch.tensor(0.0, device=model.device, dtype=nll_batch.dtype)
        }
        
    else:
        ndp = bayesian_splines.defining_prior(
            model=model, 
            hyperparameter=hyperparameter_decorrelation
            )
    
    # ---- unbiased minibatch objective ----
    scale = torch.as_tensor(
        (N_total / max(B, 1)),
        device=model.device,
        dtype=nll_batch.dtype
        )
    
    nll_scaled = scale * nll_batch
    neg_log_post = nll_scaled + ntp['neg_log_prior_total'] + ndp['neg_log_prior_total']
    
    return {
        'neg_posterior':neg_log_post,
        'negative_decorrelation_prior': ndp,
        'negative_transformation_prior': ntp,
        'negative_log_lik': nll_scaled,
        'nll_batch': nll_batch
        }

def bayesian_training_objective(
    model: "GTM",
    samples: Tensor,
    hyperparameter_transformation,
    hyperparameter_decorrelation,
    sample_size,
    batch_size,
    objective_type:Literal['negloglik']="negloglik",
    
):
    
    if objective_type == "negloglik":
        
        return unnormalized_posterior_computation(
                model=model,
                samples=samples,
                hyperparameter_transformation=hyperparameter_transformation,
                hyperparameter_decorrelation= hyperparameter_decorrelation,
                N_total=sample_size,
                B = batch_size
                )