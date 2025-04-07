import torch
from torch import nn
#from gtm.layers.flip import Flip
from gtm.gtm_layers.transformation_layer import *
from gtm.gtm_layers.decorrelation_layer import Decorrelation
from gtm.gtm_layers.layer_utils import generate_diagonal_matrix
from gtm.gtm_training.objective_functions import log_likelihood, exact_score_matching, single_sliced_score_matching, sliced_score_matching_vr, noise_contrasive_estimation, training_objective
from gtm.gtm_training.training_helpers import train, if_float_create_lambda_penalisation_matrix
#from gtm.simulation_study.simulation_study_helpers import plot_marginals, plot_densities

from gtm.gtm_splines.bspline_prediction_old import bspline_prediction

import optuna
from optuna.samplers import TPESampler 
import itertools


class TM(nn.Module):
    def __init__(self, degree, spline_range, monotonically_increasing=True, spline="bspline", span_factor=torch.tensor(0.1),
                 number_covariates=False, initial_log_transform=False, calc_method_bspline="deBoor",span_restriction="reluler", spline_order=3, device="cpu"): #device=None
        super().__init__()
        self.type = "1d_transformation"
        self.degree  = degree
        self.spline_range = torch.FloatTensor(spline_range)
        self.monotonically_increasing = monotonically_increasing
        
        # For padding!
        self.max_degree = self.degree

        self.span_factor = span_factor

        self.multivariate_basis = False
        self.multivariate_basis_derivativ_1 = False

        self.number_covariates = number_covariates
        # TODO: solve how covariate effect is implemented
        self.params_covariate = False
        self.covaraite_effect = False

        self.initial_log_transform = initial_log_transform
        
        self.calc_method_bspline = calc_method_bspline
        
        self.spline_order = spline_order
        
        self.span_restriction=span_restriction

        self.spline = spline
        # param dims: 0: basis, 1: variable
        self.params = nn.Parameter(self.compute_starting_values())
        
        if spline == "bernstein":
            warnings.warn("Warning: TM is not implemented for Bernstein, only for B-Spline.")
        else:
            warnings.warn("Warning: Unknown Spline string passed, use bspline or bernstein instead. bspline is default.")
            
        self.subset_dimension = False
        
        self.num_trans_layers = 0
        
        self.device = device
    
 
    def compute_starting_values(self):
        """
        Computes Starting Values for the Transformation layer with variable knots for different data dimensions.

        :return: starting values tensor
        """
        min_val = self.spline_range[0]
        max_val = self.spline_range[1]
            
        par_unristricted = torch.linspace(min_val, max_val, self.degree + self.spline_order - 1)
        par_restricted_opt = par_unristricted
        par_unristricted[1:] = torch.log(torch.exp(par_restricted_opt[1:] - par_restricted_opt[:-1]) - 1)
            
        if self.number_covariates == 1:
            par_unristricted = par_unristricted.repeat(self.degree + self.spline_order - 1 , 1).T.flatten()
        elif self.number_covariates > 1:
            raise NotImplementedError("Only implemented for 1 or No covariates!")

        return par_unristricted
    
        
    def forward_log_likelihood(self,y):
        z_tilde = bspline_prediction(self.params.unsqueeze(1), y, self.degree, self.spline_range, monotonically_increasing=True, derivativ=0, return_penalties=False, calc_method=self.calc_method_bspline,#'Naive_Basis', #before: deBoor 
                        span_factor=self.span_factor, span_restriction=self.span_restriction,
                        covariate=False, params_covariate=False, covaraite_effect="multiplicativ",
                        penalize_towards=0, order=self.spline_order)
        
        der_1 = bspline_prediction(self.params.unsqueeze(1), y, self.degree, self.spline_range, monotonically_increasing=True, derivativ=1, return_penalties=False, calc_method=self.calc_method_bspline,#'Naive_Basis', #before: deBoor 
                        span_factor=self.span_factor, span_restriction=self.span_restriction,
                        covariate=False, params_covariate=False, covaraite_effect="multiplicativ",
                        penalize_towards=0, order=self.spline_order)
        
        # gaussian log likelihood
        return torch.distributions.Normal(torch.tensor([0.0]), torch.tensor([1.0])).log_prob(z_tilde) + torch.log(der_1)
    
    def latent_space(self,y):
        z_tilde = bspline_prediction(self.params.unsqueeze(1), y, self.degree, self.spline_range, monotonically_increasing=True, derivativ=0, return_penalties=False, calc_method=self.calc_method_bspline,#'Naive_Basis', #before: deBoor 
                        span_factor=self.span_factor, span_restriction=self.span_restriction,
                        covariate=False, params_covariate=False, covaraite_effect="multiplicativ",
                        penalize_towards=0, order=self.spline_order)
        
        return z_tilde
    
    def training_objective(self, y_train, penalty_params, lambda_penalty_params, train_covariates, lambda_penalty_mode, objective_type, adaptive_lasso_weights_matrix):
        if self.subset_dimension is not False:
            y_train_sub = y_train[:,self.subset_dimension]
            
        return {"loss_with_penalties" : -1 * self.forward_log_likelihood(y_train_sub), "loss_without_penalties": -1 * self.forward_log_likelihood(y_train_sub)}
    
    
    def __train__(self, train_dataloader, validate_dataloader=False, train_covariates=False, validate_covariates=False, penalty_params=torch.FloatTensor([0,0,0,0]), adaptive_lasso_weights_matrix = False,
                  lambda_penalty_params=False, learning_rate=1, iterations=2000, verbose=False, patience=5, min_delta=1e-7,
          optimizer='LBFGS', lambda_penalty_mode="square", objective_type="negloglik", ema_decay=False, seperate_copula_training=False,
          max_batches_per_iter=None):

        if lambda_penalty_params is not False:
            lambda_penalty_params = lambda_penalty_params.to(self.device)
            
        if adaptive_lasso_weights_matrix is not False:
            adaptive_lasso_weights_matrix = adaptive_lasso_weights_matrix.to(self.device)
        
        #if seperate_copula_training==True:
        #    self.transformation.params.requires_grad=False
        
        return_dict_model_training = train(self, train_dataloader=train_dataloader, validate_dataloader=validate_dataloader, train_covariates=train_covariates, validate_covariates=validate_covariates, penalty_params=penalty_params, lambda_penalty_params=lambda_penalty_params, learning_rate=learning_rate, 
                     iterations=iterations, verbose=verbose, patience=patience, min_delta=min_delta, optimizer=optimizer, lambda_penalty_mode=lambda_penalty_mode, objective_type=objective_type, adaptive_lasso_weights_matrix = adaptive_lasso_weights_matrix, max_batches_per_iter=max_batches_per_iter)
        
        #if seperate_copula_training==True:
        #    self.transformation.params.requires_grad=True
        
        return return_dict_model_training