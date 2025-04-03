import time
import copy
import torch
import warnings
from torch import nn
import numpy as np
from torch.distributions import Normal, Laplace
import matplotlib.pyplot as plt
from torch import optim
from tqdm import tqdm
import seaborn as sns
#from pytorch_lbfgs.LBFGS import LBFGS, FullBatchLBFGS
#from torch.optim import LBFGS
import copy
import pickle



def log_likelihood(model, samples, train_covariates=False, train=True, evaluate=True, mean_loss=False):
    #z, log_d, transformation_second_order_ridge_pen_sum, second_order_ridge_pen_global, first_order_ridge_pen_global, \
    #param_ridge_pen_global, lambda_matrix_global, scores, hessian \
    #        = model.forward(samples, covariate=train_covariates, train=train, evaluate=evaluate)
    return_dict_nf_mctm = model.forward(samples, covariate=train_covariates, train=train, evaluate=evaluate)
            
    #log_likelihood_latent = Normal(0, 1).log_prob(z)
    #with torch.autograd.enable_grad(): # TODO: why do i need to enable grad here? is it because I defined the function here? is this also required for all other objectives in this file?
    log_likelihood_latent = Normal(0, 1).log_prob(return_dict_nf_mctm["output"])
    
    # sum across dimensions
    #log_likelihood_data = torch.sum(log_likelihood_latent + return_dict_nf_mctm["log_d"],1)

    if mean_loss:
        # !!!! compute the mean across dimensions of data and number of samples, works better for training to also mean across data dimensions somehow
        
        #log_likelihood_data = torch.sum(log_likelihood_latent + log_d )
        log_likelihood_data = log_likelihood_latent + return_dict_nf_mctm["log_d"] #TODO: made the mean also across the dimensionality of the data
        log_likelihood_data = torch.mean(log_likelihood_data)
    else:
        log_likelihood_data = torch.sum(log_likelihood_latent + return_dict_nf_mctm["log_d"],1)
        # log_likelihood_data = log_likelihood_latent + log_d
    #    log_likelihood_data = log_likelihood_latent + return_dict_nf_mctm["log_d"]
        
    return_dict_nf_mctm["log_likelihood_data"] = log_likelihood_data
    
    return return_dict_nf_mctm
    #return log_likelihood_data, transformation_second_order_ridge_pen_sum, second_order_ridge_pen_global, first_order_ridge_pen_global, param_ridge_pen_global, lambda_matrix_global, scores, hessian


def single_sliced_score_matching(model, samples, train_covariates, train, evaluate, score_method = "autograd", mean_loss=True, noise=None, detach=False, noise_type='gaussian'):#'radermacher'):
    samples.requires_grad_(True)
    if noise is None:
        vectors = torch.randn_like(samples)
        if noise_type == 'radermacher':
            vectors = vectors.sign()
        elif noise_type == 'sphere':
            vectors = vectors / torch.norm(vectors, dim=-1, keepdim=True) * np.sqrt(vectors.shape[-1])
        elif noise_type == 'gaussian':
            pass
        else:
            raise ValueError("Noise type not implemented")
    else:
        vectors = noise

    #logp, transformation_second_order_ridge_pen_sum, second_order_ridge_pen_global, first_order_ridge_pen_global, \
    #    param_ridge_pen_global, lambda_matrix_global, scores, hessian = log_likelihood(model, samples, train_covariates=False, train=train, evaluate=evaluate)
    return_dict_nf_mctm = log_likelihood(model, samples, train_covariates=False, train=train, evaluate=evaluate)

    if score_method=="autograd":
        #grad1 = torch.autograd.grad(logp, samples, create_graph=True)[0]
        grad1 = torch.autograd.grad(return_dict_nf_mctm["log_likelihood_data"].sum(), samples, create_graph=True)[0]
    elif score_method=="analytically":
        #grad1 = scores
        grad1 = return_dict_nf_mctm["scores"]

    gradv = torch.sum(grad1 * vectors)
    loss1 = torch.sum(grad1 * vectors, dim=-1) ** 2 * 0.5
    if detach:
        loss1 = loss1.detach()
    grad2 = torch.autograd.grad(gradv, samples, create_graph=True)[0]
    loss2 = torch.sum(vectors * grad2, dim=-1)
    if detach:
        loss2 = loss2.detach()

    loss = (loss1 + loss2)
    
    if mean_loss == True:
        loss.mean()
    else:
        loss.mean(1)
    
    return_dict_nf_mctm["sliced_score_matching_loss"] = loss
    #return loss, transformation_second_order_ridge_pen_sum, second_order_ridge_pen_global, first_order_ridge_pen_global, param_ridge_pen_global, lambda_matrix_global
    return return_dict_nf_mctm


def exact_score_matching(
        model,
        samples,
        train_covariates=False,
        train=True,
        evaluate=True,
        score_method="autograd",
        mean_loss=True):
    
    samples.requires_grad_(True)
    
    #logp, transformation_second_order_ridge_pen_sum, second_order_ridge_pen_global, first_order_ridge_pen_global, \
    #    param_ridge_pen_global, lambda_matrix_global, scores, hessian = log_likelihood(model, samples, train_covariates=False, train=train, evaluate=evaluate)
    #return_dict_nf_mctm = log_likelihood(model, samples, train_covariates=False, train=train, evaluate=evaluate)

    
    if score_method=="autograd":
        return_dict_nf_mctm = model.log_likelihood_loss(y=samples, covariate=False, mean_loss=False, train=train, evaluate=evaluate)
        
        #grad1 = torch.autograd.grad(logp, samples, create_graph=True)[0]
        grad1 = torch.autograd.grad(return_dict_nf_mctm["log_likelihood_data"].sum(), samples, create_graph=True)[0]
        loss1 = torch.norm(grad1, dim=-1) ** 2 / 2.
        
        loss2 = torch.zeros(samples.shape[0], device=samples.device)
        
        # if samples.shape[1] > 100:
        #     iterator = tqdm(range(samples.shape[1]))
        # else:
        iterator = range(samples.shape[1])
        for i in iterator:
            if train:
                grad = torch.autograd.grad(grad1[:, i].sum(), samples, create_graph=True)[0][:, i] #create_graph=True, retain_graph=True
            if not train:
                grad = torch.autograd.grad(grad1[:, i].sum(), samples, create_graph=False)[0][:, i] #create_graph=False, retain_graph=True
                grad = grad.detach()
            loss2 += grad
            
            loss = loss1 + loss2
        
    elif score_method=="analytically":
        return_dict_nf_mctm = model.forward(samples, covariate=train_covariates, train=train, evaluate=evaluate, return_scores_hessian=True)
        #analytical loss
        #loss = (hessian + 0.5 * scores**2).sum(1)
        loss = (return_dict_nf_mctm["hessian"] + 0.5 * return_dict_nf_mctm["scores"]**2).sum(1)

    if not train:
        loss = loss.detach()
        
    if mean_loss == True:
        loss.mean()
        
    return_dict_nf_mctm["exact_score_matching_loss"] = loss

    #return loss, transformation_second_order_ridge_pen_sum, second_order_ridge_pen_global, first_order_ridge_pen_global, param_ridge_pen_global, lambda_matrix_global
    return return_dict_nf_mctm


def sliced_score_matching_vr(model, samples, train_covariates=False, n_particles=1, train=True, evaluate=True, score_method="autograd", mean_loss=True):
    dup_samples = samples.unsqueeze(0).expand(n_particles, *samples.shape).contiguous().view(-1, *samples.shape[1:])
    dup_samples.requires_grad_(True)
    vectors = torch.randn_like(dup_samples)

    #logp, transformation_second_order_ridge_pen_sum, second_order_ridge_pen_global, first_order_ridge_pen_global, \
    #    param_ridge_pen_global, lambda_matrix_global, scores, hessian = log_likelihood(model, samples, train_covariates=False, train=train, evaluate=evaluate)
    return_dict_nf_mctm = log_likelihood(model, samples, train_covariates=False, train=train, evaluate=evaluate)

    if score_method=="autograd":
        #grad1 = torch.autograd.grad(logp, samples, create_graph=True)[0]
        grad1 = torch.autograd.grad(return_dict_nf_mctm["log_likelihood_data"], samples, create_graph=True)[0]
    elif score_method=="analytically":
        #grad1 = scores
        grad1 = return_dict_nf_mctm["scores"]
    loss1 = torch.sum(grad1 * grad1, dim=-1) / 2.
    gradv = torch.sum(grad1 * vectors)
    grad2 = torch.autograd.grad(gradv, dup_samples, create_graph=True)[0]
    loss2 = torch.sum(vectors * grad2, dim=-1)

    loss1 = loss1.view(n_particles, -1).mean(dim=0)
    loss2 = loss2.view(n_particles, -1).mean(dim=0)

    loss = loss1 + loss2
    
    if mean_loss == True:
        loss.mean()
    else:
        loss.mean(1)
    
    return_dict_nf_mctm["sliced_score_matching_vr_loss"] = loss

    #return loss, transformation_second_order_ridge_pen_sum, second_order_ridge_pen_global, first_order_ridge_pen_global, param_ridge_pen_global, lambda_matrix_global
    return return_dict_nf_mctm


def noise_contrasive_estimation(model, samples, train_covariates=False, train=True, evaluate=True, mean_loss=True):
    
    torch.manual_seed(0) # always the same noise
    noise_samples = Normal(0,1).sample(samples.size())
    
    noise_samples = torch.clamp(noise_samples, min=samples.min(), max=samples.max()) #to avoid numerical issues
    covar_samples = torch.cov(samples.T)
    mean_samples = samples.mean(0)
    lambda_matrix_samples = torch.linalg.cholesky(covar_samples)
    noise_samples = mean_samples + noise_samples
    noise_samples = torch.einsum("ij,bj -> bi",lambda_matrix_samples,noise_samples)
    lambda_matrix_samples_inv = torch.inverse(lambda_matrix_samples)
    # for testing compare:
    # torch.cov(noise_samples.T)
    # torch.cov(samples.T)
    
    full_samples = torch.cat([samples, noise_samples], dim=0)
    #class_samples = torch.cat([torch.ones(samples.size(0)), torch.zeros(noise_samples.size(0))], dim=0)
    
    #logp, transformation_second_order_ridge_pen_sum, second_order_ridge_pen_global, first_order_ridge_pen_global, \
    #    param_ridge_pen_global, lambda_matrix_global, scores, hessian = log_likelihood(model, samples, train_covariates=False, train=train, evaluate=evaluate)
    return_dict_nf_mctm = log_likelihood(model, samples, train_covariates=False, train=train, evaluate=evaluate)

    full_samples_demeaned = full_samples - mean_samples
    full_samples_decorrelated = torch.einsum("ij,bj -> bi", lambda_matrix_samples_inv, full_samples_demeaned)
    # for testing:
    # torch.cov(torch.einsum("ij,bj -> bi", lambda_matrix_samples_inv, noise_samples).T) -> approx. Identity matrix
    log_likelihood_noise = Normal(0,1).log_prob(full_samples_decorrelated).sum(1)
    
    # Notation following the NCE original paper section 2
    #G = logp - log_likelihood_noise
    G = return_dict_nf_mctm["log_likelihood_data"] - log_likelihood_noise
    h = 1 / (1 + torch.exp(-G))
    J = 1 / full_samples.size(0) * sum(torch.log(h[:samples.size(0)]) + torch.log(1 - h[samples.size(0):])) #first the sampels and second the noise
    
    # J is an objective which is maximized
    loss = -1 * J#.mean()
    
    if mean_loss == True:
        loss.mean()
    else:
        loss.mean(1)
    
    return_dict_nf_mctm["noise_contrastive_estimation_loss"] = loss
    
    #return loss, transformation_second_order_ridge_pen_sum, second_order_ridge_pen_global, first_order_ridge_pen_global, param_ridge_pen_global, lambda_matrix_global 
    return return_dict_nf_mctm

def training_objective(model, samples, penalty_params, train_covariates=False, lambda_penalty_params: torch.Tensor =False, adaptive_lasso_weights_matrix = False, avg = True,
              lambda_penalty_mode="square", objective_type = "negloglik"):
    
    if objective_type == "negloglik":
        return_dict_model_loss = model.log_likelihood_loss(samples, covariate=train_covariates, mean_loss=True)
        loss = return_dict_model_loss["negative_log_likelihood_data"]
        
    elif objective_type == "score_matching":
        return_dict_model_loss = model.exact_score_matching_loss(samples, covariate=train_covariates, mean_loss=True)
        loss = return_dict_model_loss["exact_score_matching_loss"]
        
    elif objective_type == "single_sliced_score_matching":
        return_dict_model_loss = model.single_sliced_score_matching_loss(samples, covariate=train_covariates, mean_loss=True)
        loss = return_dict_model_loss["sliced_score_matching_loss"]
        
    elif objective_type == "vr_sliced_score_matching":
        return_dict_model_loss = model.vr_sliced_score_matching_loss(samples, covariate=train_covariates, mean_loss=True)
        loss = return_dict_model_loss["sliced_score_matching_vr_loss"]
        
    elif objective_type == "noise_contrastive_estimation":
        return_dict_model_loss = model.noise_contrasive_estimation_loss(samples, covariate=train_covariates, mean_loss=True)
        loss = return_dict_model_loss["noise_contrastive_estimation_loss"]
        
        
    if model.__module__ == "python_nf_mctm.models.nf_mctm":
        num_splines_per_layer = model.number_variables * (model.number_variables-1) /2 #offdiagonal splines (additive effect)
        #if model.affine_layer is not False:
        #    num_splines_per_layer += model.number_variables * (model.number_variables-1) /2 #diagonal splines (multiplicative effect)
        num_params_decorr = num_splines_per_layer * model.number_decorrelation_layers *  model.degree_decorrelation 
        
        #num_params_trans = model.number_variables * model.num_trans_layers * model.degree_transformations 
        # sum if we have variable degrees of transformation
        num_params_trans = model.number_variables * sum(model.num_trans_layers * model.degree_transformations)

    elif model.__module__ == "python_nf_mctm.models.factor_graph":
        num_splines_per_layer = model.data_dims * (model.data_dims-1) /2 #offdiagonal splines (additive effect)
        num_splines_per_layer += model.data_dims * (model.data_dims-1) /2 #diagonal splines (multiplicative effect)
        num_params_decorr = num_splines_per_layer * model.number_decorrelation_layers *  model.degree_decorrelation * model.number_of_pairs
        
        #num_params_trans = model.number_variables * model.num_trans_layers * model.degree_transformations * model.number_of_pairs 
        # sum if we have variable degrees of transformation
        num_params_trans = model.number_variables * sum(model.num_trans_layers * model.degree_transformations)
    else:
        #for pretrain marginals this is the case
        num_params_decorr=1
        num_params_trans=1
        # lambda_penalty_params is not False for the pretrain to work
    if lambda_penalty_params is not False and torch.all(lambda_penalty_params == 0) == False:

        precision_matrix = torch.matmul(torch.transpose(return_dict_model_loss["lambda_matrix_global"], 1, 2),return_dict_model_loss["lambda_matrix_global"])
        if adaptive_lasso_weights_matrix is False:
            pen_lambda_lasso = (lambda_penalty_params                                 * (torch.square(precision_matrix).sum(0)) ** 0.5).tril(-1).mean()
        else:
            pen_lambda_lasso = (lambda_penalty_params * adaptive_lasso_weights_matrix * (torch.square(precision_matrix).sum(0)) ** 0.5).tril(-1).mean()
    else:
        pen_lambda_lasso = 0
        
    if num_params_decorr == 0:
        num_params_decorr = 1 #for no division errors here
    pen_value_ridge = penalty_params[0] * return_dict_model_loss["param_ridge_pen_global"]                         / num_params_decorr
    pen_first_ridge = penalty_params[1] * return_dict_model_loss["first_order_ridge_pen_global"]                   / num_params_decorr
    pen_second_ridge = penalty_params[2] * return_dict_model_loss["second_order_ridge_pen_global"]                 / num_params_decorr
    if num_params_trans > 0:
        pen_ctm_second_ridge = penalty_params[3] * return_dict_model_loss["transformation_second_order_ridge_pen_global"] / num_params_trans
    else:
        pen_ctm_second_ridge = 0
    
    loss_with_penalties = loss + \
        pen_value_ridge + \
        pen_first_ridge + \
        pen_second_ridge + \
        pen_lambda_lasso + \
        pen_ctm_second_ridge
        
    return {"loss_with_penalties" : loss_with_penalties, 
            "loss_without_penalties" : loss,
            "pen_ctm_second_ridge" : pen_ctm_second_ridge, 
            "pen_value_ridge" : pen_value_ridge, 
            "pen_first_ridge" : pen_first_ridge, 
            "pen_second_ridge" : pen_second_ridge, 
            "pen_lambda_lasso" : pen_lambda_lasso}
    