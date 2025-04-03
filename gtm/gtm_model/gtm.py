import torch
from torch import nn
#from gtm.layers.flip import Flip
from gtm.gtm_layers.transformation_layer import *
from gtm.gtm_layers.decorrelation_layer import Decorrelation
from gtm.gtm_layers.layer_utils import generate_diagonal_matrix
from gtm.gtm_training.objective_functions import log_likelihood, exact_score_matching, single_sliced_score_matching, sliced_score_matching_vr, noise_contrasive_estimation, training_objective
from gtm.gtm_training.training_helpers import train
#from gtm.simulation_study.simulation_study_helpers import plot_marginals, plot_densities


class GTM(nn.Module):
    def __init__(self, 
                 input_min, 
                 input_max,
                 number_variables,
                 polynomial_range=list([[-15], [15]]), 
                 spline_transformation="bspline", spline_decorrelation="bspline", # bernstein bernstein bspline
                 degree_transformations=15, degree_decorrelation=20, span_factor=torch.tensor(0.), span_restriction="reluler", #span_factor=torch.tensor(0.1)
                 number_covariates=False,
                 num_trans_layers=1,
                 num_decorr_layers=3, list_comprehension=False, initial_log_transform=False,
                 covaraite_effect="multiplicativ",
                 calc_method_bspline="deBoor",
                 affine_decorr_layer=False,
                 degree_multi=False,
                 spline_order=3,
                 device="cpu",
                 transform_only=False):
        super(GTM, self).__init__()
        
        self.transform_only = transform_only

        self.polynomial_range = polynomial_range
      
        self.number_variables = number_variables
        self.input_min = input_min
        self.input_max = input_max

        # required for the varying degree of the transformation layer to work
        # if it is a number then transform into a repeating list of length of number of varaibles
        if isinstance(degree_transformations, int):
            degree_transformations = [degree_transformations] * self.number_variables
        self.degree_transformations = degree_transformations
        self.degree_decorrelation = degree_decorrelation
        
        self.spline_transformation = spline_transformation
        self.spline_decorrelation = spline_decorrelation

        self.span_factor = span_factor
        self.span_restriction = span_restriction

        self.device = device

        # Repeat polynomial ranges for all variables as this is the range for the bsplines essentially
        self.polynomial_range_transformation = list([polynomial_range[0] * self.number_variables,
                                                polynomial_range[1] * self.number_variables])
        self.polynomial_range_decorrelation = list([polynomial_range[0] * self.number_variables,
                                                polynomial_range[1] * self.number_variables])

        self.number_covariates = number_covariates

        self.list_comprehension = list_comprehension

        self.num_trans_layers = num_trans_layers

        self.initial_log_transform = initial_log_transform

        self.covaraite_effect = covaraite_effect
        
        self.calc_method_bspline = calc_method_bspline
        
        self.spline_order = spline_order
        
        self.affine_decorr_layer = affine_decorr_layer
        
        self.degree_multi = degree_multi

        if self.num_trans_layers > 0:
            self.transformation = Transformation(degree=self.degree_transformations, number_variables=self.number_variables,
                                     polynomial_range=self.polynomial_range_transformation, span_factor=self.span_factor,
                                     number_covariates=self.number_covariates, spline=self.spline_transformation,
                                     initial_log_transform=self.initial_log_transform,
                                     calc_method_bspline=self.calc_method_bspline,
                                     span_restriction=self.span_restriction,
                                     spline_order = self.spline_order) 
            
        if self.num_trans_layers == 2:
            warnings.warn("Warning: model is only implemented to have one transformation layer.")

        self.flip_matrix = generate_diagonal_matrix(self.number_variables).to(self.device)

        self.number_decorrelation_layers = num_decorr_layers
        if self.number_decorrelation_layers > 0:
            self.decorrelation_layers = nn.ModuleList([Decorrelation(degree=self.degree_decorrelation, number_variables=self.number_variables,
                                    polynomial_range=self.polynomial_range_decorrelation, span_factor=self.span_factor,
                                    span_restriction=self.span_restriction, spline=self.spline_decorrelation,
                                    number_covariates=self.number_covariates,
                                    list_comprehension = self.list_comprehension,
                                    covaraite_effect = self.covaraite_effect,
                                    calc_method_bspline = self.calc_method_bspline,
                                    spline_order = self.spline_order,
                                    affine_layer = self.affine_decorr_layer,
                                    degree_multi = self.degree_multi
                                                                    ) for i in range(self.number_decorrelation_layers)])
        
    
    def create_return_dict_nf_mctm(self, input):
        return {"output": input.clone(),
                "log_d": torch.zeros(input.size()).to(self.device),
                "transformation_second_order_ridge_pen_global": 0,
                "second_order_ridge_pen_global": 0,
                "first_order_ridge_pen_global": 0,
                "param_ridge_pen_global": 0,
                "lambda_matrix_global": torch.eye(self.number_variables).to(self.device), 
                "der_lambda_matrix_global": None,
                "der2_lambda_matrix_global": None
                }
    
        
    def forward(self, y, covariate=False, train=True, evaluate=True, return_scores_hessian=False, return_lambda_matrix = False):
        
        return_dict_nf_mctm = self.create_return_dict_nf_mctm(y)

        if self.initial_log_transform==True:
            y = y + 0.01 #log(0) does not work
            log_d = - torch.log(y) # = log(1/y)
            y = torch.log(y)
        else:
            log_d = 0

        # Training or evaluation
        if train or evaluate:

            if train:
                if self.num_trans_layers > 0:
                    # new input false to not recompute basis each iteration
                    return_dict_transformation = self.transformation(y, covariate, log_d=log_d, return_log_d=True, new_input=False, store_basis=True, return_scores_hessian=return_scores_hessian)

                    return_dict_nf_mctm["output"] = return_dict_transformation["output"]
                    return_dict_nf_mctm["log_d"] = return_dict_transformation["log_d"]

            elif evaluate:
                if self.num_trans_layers > 0:
                    # new input true as we need to recompute the basis for the validation/test set
                    return_dict_transformation = self.transformation(y, covariate, log_d=log_d, return_log_d=True, new_input=True, store_basis=True, return_scores_hessian=return_scores_hessian)
                    
                    return_dict_nf_mctm["output"] = return_dict_transformation["output"]
                    return_dict_nf_mctm["log_d"] = return_dict_transformation["log_d"]
            
            if self.transform_only == True:
                return return_dict_nf_mctm

            if self.num_trans_layers > 0 and return_scores_hessian == True:
            
                return_dict_nf_mctm["der_lambda_matrix_global"] = return_dict_transformation["scores"]#.unsqueeze(2)

                return_dict_nf_mctm["der2_lambda_matrix_global_list"] = [return_dict_transformation["hessian"]]#.unsqueeze(2)]
            
            if self.number_decorrelation_layers > 0:
                for i in range(self.number_decorrelation_layers):


                    if ((i+1) % 2) == 0:
                        # even: 2,4, 6, ...
                        return_dict_nf_mctm["output"] = (self.flip_matrix @ return_dict_nf_mctm["output"].T).T
                    #else:
                        # odd: 1, 3, 5, ...


                    return_dict_decorrelation = self.decorrelation_layers[i](return_dict_nf_mctm["output"], covariate, 0,#return_dict_nf_mctm["log_d"],
                                                                return_log_d=True, return_penalties=True, return_scores_hessian=return_scores_hessian)
                    
                    return_dict_nf_mctm["output"] = return_dict_decorrelation["output"]
                    #return_dict_nf_mctm["log_d"] = return_dict_decorrelation["log_d"]
                    return_dict_nf_mctm["second_order_ridge_pen_global"] += return_dict_decorrelation["second_order_ridge_pen_sum"]
                    return_dict_nf_mctm["first_order_ridge_pen_global"] += return_dict_decorrelation["first_order_ridge_pen_sum"]
                    return_dict_nf_mctm["param_ridge_pen_global"] += return_dict_decorrelation["param_ridge_pen_sum"]

                    if ((i+1) % 2) == 0:
                        # even
                        if return_lambda_matrix == True:
                            lambda_matrix_upper = self.flip_matrix @ return_dict_decorrelation["lambda_matrix"] @ self.flip_matrix

                        if return_lambda_matrix == True:
                            return_dict_nf_mctm["lambda_matrix_global"] = lambda_matrix_upper @ return_dict_nf_mctm["lambda_matrix_global"]

                        if return_scores_hessian == True:
                            der_lambda_matrix_upper = self.flip_matrix @ return_dict_decorrelation["der_lambda_matrix"] @ self.flip_matrix  
                            return_dict_nf_mctm["der_lambda_matrix_global"] = torch.bmm(der_lambda_matrix_upper, return_dict_nf_mctm["der_lambda_matrix_global"])
                            
                            der2_lambda_matrix_upper = self.flip_matrix @ return_dict_decorrelation["der2_lambda_matrix"] @ self.flip_matrix
                            
                            return_dict_nf_mctm["der2_lambda_matrix_global_list"].append(
                                torch.bmm(der2_lambda_matrix_upper, return_dict_nf_mctm["der_lambda_matrix_global"]) * return_dict_nf_mctm["der_lambda_matrix_global"])
                            
                            for j in range(i+1): #j are all sum elements prior to i
                                return_dict_nf_mctm["der2_lambda_matrix_global_list"][j] = torch.bmm(der_lambda_matrix_upper, return_dict_nf_mctm["der2_lambda_matrix_global_list"][j])
                            
                    else:
                        # odd
                        if return_lambda_matrix == True:
                            return_dict_nf_mctm["lambda_matrix_global"] = return_dict_decorrelation["lambda_matrix"] @ return_dict_nf_mctm["lambda_matrix_global"]
                        
                        if return_scores_hessian == True:
                            return_dict_nf_mctm["der_lambda_matrix_global"] = torch.bmm(return_dict_decorrelation["der_lambda_matrix"], return_dict_nf_mctm["der_lambda_matrix_global"])

                            return_dict_nf_mctm["der2_lambda_matrix_global_list"].append(
                                torch.bmm(return_dict_decorrelation["der2_lambda_matrix"], return_dict_nf_mctm["der_lambda_matrix_global"]) * return_dict_nf_mctm["der_lambda_matrix_global"])
                            
                            for j in range(i+1): #j are all sum elements prior to i
                                return_dict_nf_mctm["der2_lambda_matrix_global_list"][j] = torch.bmm(return_dict_decorrelation["der_lambda_matrix"], return_dict_nf_mctm["der2_lambda_matrix_global_list"][j])

                    if ((i+1) % 2) == 0:
                        # even
                        #output = (self.flip_matrix @ output.T).T
                        return_dict_nf_mctm["output"] = (self.flip_matrix @ return_dict_nf_mctm["output"].T).T
                    #else:
                    #    # odd
                    
                if return_scores_hessian == True:
                    return_dict_nf_mctm["scores"] = -1 * return_dict_nf_mctm["output"] * return_dict_nf_mctm["der_lambda_matrix_global"].squeeze(1)
                
                    for j in range(i+1): #j are all sum elements prior to i
                                return_dict_nf_mctm["der2_lambda_matrix_global_list"][j] = -1 * return_dict_nf_mctm["output"].unsqueeze(2) * return_dict_nf_mctm["der2_lambda_matrix_global_list"][j]
                    return_dict_nf_mctm["der2_lambda_matrix_global_list"][i+1] = -1 * return_dict_nf_mctm["der2_lambda_matrix_global_list"][i+1]
                
                    return_dict_nf_mctm["hessian"] = torch.stack(return_dict_nf_mctm["der2_lambda_matrix_global_list"], dim=0).sum(0).squeeze(2)
            
            return return_dict_nf_mctm


    def latent_space_representation(self, y, covariate=False):
        return_dict = self.forward(y, covariate, train=False, evaluate=True)
        return return_dict["output"]


    def log_likelihood_loss(self, y, covariate=False, train=True, evaluate=True, mean_loss=True):

        return_dict_nf_mctm = log_likelihood(model=self, samples=y, train_covariates=covariate, train=train, evaluate=evaluate, mean_loss=mean_loss)

        return_dict_nf_mctm["negative_log_likelihood_data"] = -1 * return_dict_nf_mctm["log_likelihood_data"]
            
        return return_dict_nf_mctm
    
    
    def log_likelihood(self, samples, covariate=False, mean_loss=False):
        return self.log_likelihood_loss(samples, covariate=False, mean_loss=mean_loss, train=False, evaluate=True)["log_likelihood_data"] #.sum(1)


    def exact_score_matching_loss(self, samples, covariate=False, mean_loss=True):
        samples.requires_grad_(True)

        return_dict_nf_mctm = exact_score_matching(
            model=self,
            samples=samples,
            train_covariates=covariate,
            train=True,
            evaluate=True,
            score_method="autograd",
            mean_loss=mean_loss)

        return return_dict_nf_mctm
    
    
    def exact_score_matching(self, samples, covariate=False):
        return self.exact_score_matching_loss(samples, covariate=False)["exact_score_matching_loss"]
    
    
    def single_sliced_score_matching_loss(self, samples, covariate=False, mean_loss=True):
        samples.requires_grad_(True)

        return_dict_nf_mctm = single_sliced_score_matching(
            model=self,
            samples=samples,
            train_covariates=covariate,
            train=True,
            evaluate=True,
            score_method="autograd",
            mean_loss=mean_loss,
            noise=None, 
            detach=False, 
            noise_type='gaussian')

        return return_dict_nf_mctm
    
    
    def vr_sliced_score_matching_loss(self, samples, covariate=False, mean_loss=True):
        samples.requires_grad_(True)

        return_dict_nf_mctm = sliced_score_matching_vr(
            model=self,
            samples=samples,
            train_covariates=covariate,
            train=True,
            evaluate=True,
            score_method="autograd",
            mean_loss=mean_loss)

        return return_dict_nf_mctm
    
    
    def noise_contrasive_estimation_loss(self, samples, covariate=False, mean_loss=True):
        samples.requires_grad_(True)

        return_dict_nf_mctm = noise_contrasive_estimation(
            model=self,
            samples=samples,
            train_covariates=covariate,
            train=True,
            evaluate=True,
            score_method="autograd",
            mean_loss=mean_loss)

        return return_dict_nf_mctm
    
    
    def training_objective(self, samples, penalty_params, train_covariates=False, lambda_penalty_params: torch.Tensor =False, 
                           adaptive_lasso_weights_matrix: torch.Tensor =False, 
                           avg = True, lambda_penalty_mode="square", objective_type = "negloglik"):
        
        return training_objective(self, samples = samples, penalty_params = penalty_params, train_covariates = train_covariates, lambda_penalty_params = lambda_penalty_params, 
                                  adaptive_lasso_weights_matrix = adaptive_lasso_weights_matrix,
                                  avg = avg, lambda_penalty_mode = lambda_penalty_mode, objective_type = objective_type)
    
    
    def __train__(self, train_dataloader, validate_dataloader=False, train_covariates=False, validate_covariates=False, penalty_params=torch.FloatTensor([0,0,0,0]), adaptive_lasso_weights_matrix = False,
                  lambda_penalty_params=False, learning_rate=1, iterations=2000, verbose=True, patience=5, min_delta=1e-7, return_plot=True,
          optimizer='LBFGS', lambda_penalty_mode="square", objective_type="negloglik", ema_decay=False, seperate_copula_training=False,
          batch_training_size=None):

        if lambda_penalty_params is not False:
            lambda_penalty_params = lambda_penalty_params.to(self.device)
            
        if adaptive_lasso_weights_matrix is not False:
            adaptive_lasso_weights_matrix = adaptive_lasso_weights_matrix.to(self.device)
        
        if seperate_copula_training==True:
            self.transformation.params.requires_grad=False
        
        return_dict_model_training = train(self, train_dataloader=train_dataloader, validate_dataloader=validate_dataloader, train_covariates=train_covariates, validate_covariates=validate_covariates, penalty_params=penalty_params, lambda_penalty_params=lambda_penalty_params, learning_rate=learning_rate, 
                     iterations=iterations, verbose=verbose, patience=patience, min_delta=min_delta, optimizer=optimizer, lambda_penalty_mode=lambda_penalty_mode, objective_type=objective_type, adaptive_lasso_weights_matrix = adaptive_lasso_weights_matrix)
        
        if seperate_copula_training==True:
            self.transformation.params.requires_grad=True
        
        return return_dict_model_training
    
    
    def pretrain_tranformation_layer(self, train_dataloader, validate_dataloader=False, train_covariates=False, validate_covariates=False, penalty_params=torch.FloatTensor([0,0,0,0]), lambda_penalty_params=False, learning_rate=1, iterations=2000, verbose=True, patience=5, min_delta=1e-7, return_plot=True,
          optimizer='LBFGS', lambda_penalty_mode="square", objective_type="negloglik", ema_decay=False, batch_training_size=None):
        
        optimizer='LBFGS'
        warnings.warn("Optimiser for pretrain_tranformation_layer is always LBFGS. If this is an issue change the code.")
        
        self.transform_only = True
        lambda_penalty_params = False # makes objective not check lambda matrix
    
        if lambda_penalty_params is not False:
            # set to false because this is the pretraining of the transformation layer
            lambda_penalty_params = False
            
            
        #if return_plot:
        #    return_dict = self.forward(train_data, covariate=train_covariates, train=True, evaluate=False)
        #    z_tilde_init = return_dict["output"]
        #
        #    fig_init_marginals = plot_marginals(z_tilde_init.detach().cpu().numpy())
            
        return_dict_model_training = train(self, 
                                           train_dataloader=train_dataloader, 
                                           validate_dataloader=validate_dataloader, 
                                           train_covariates=train_covariates, 
                                           validate_covariates=validate_covariates, 
                                           penalty_params=penalty_params, 
                                           lambda_penalty_params=lambda_penalty_params, 
                                           learning_rate=learning_rate, 
                                           iterations=iterations, 
                                           verbose=verbose, 
                                           patience=patience, 
                                           min_delta=min_delta, 
                                           optimizer=optimizer, 
                                           lambda_penalty_mode=lambda_penalty_mode, 
                                           objective_type=objective_type)
        
        
        #if return_plot:
        #    return_dict = self.forward(train_data, covariate=train_covariates, train=True, evaluate=False)
        #    z_tilde_trained = return_dict["output"]
        #
        #    fig_trained_marginals = plot_marginals(z_tilde_trained.detach().cpu().numpy(),covariate=train_covariates)
#
        #    fig_trained_space = plot_densities(z_tilde_trained.detach().cpu().numpy(), covariate=train_covariates)
        
        self.transform_only = False
        
        #if return_plot:   
        #    return return_dict_model_training, \
        #       fig_train, \
        #       fig_init_marginals, \
        #       fig_trained_marginals, \
        #       fig_trained_space
        #else:
        #    return return_dict_model_training
        return return_dict_model_training
        
    

    def compute_precision_matrix(self, y, covariate=False):

        with torch.no_grad():
            return_dict = self.forward(y, covariate=covariate, evaluate=True, train=False, return_lambda_matrix=True)

            precision_matrix = torch.matmul(torch.transpose(return_dict["lambda_matrix_global"], 1, 2),
                                            return_dict["lambda_matrix_global"])

        return precision_matrix


    def sample(self, n_samples, covariate=False):
        
        with torch.no_grad():
            
            z = torch.distributions.Normal(0, 1).sample((n_samples, self.number_variables)).to(device=self.device)
            
            for i in range(self.number_decorrelation_layers -1, -1, -1):

                if ((i+1) % 2) == 0:
                    # even
                    z = (self.flip_matrix @ z.T).T
                    # else:
                    #    # odd

                return_dict = self.decorrelation_layers[i](z, covariate=covariate, return_log_d=False, return_penalties=False, inverse=True)
                z = return_dict["output"]

                if ((i+1) % 2) == 0:
                    # even
                    z = (self.flip_matrix @ z.T).T
                    # else:
                    #    # odd
                    
            if self.num_trans_layers > 0:
                return_dict = self.transformation(z, covariate, new_input=True, inverse=True)
                y = return_dict["output"]
            else:
                y = z

            if self.initial_log_transform==True:
                y = torch.exp(y)
                y = y - 0.01  

            return y

    def after_transformation(self, y, covariate=False):
        if self.initial_log_transform==True:
            y = y + 0.01
            y = torch.log(y)
        else:
            pass
        if self.num_trans_layers > 0:
            # new input true as we need to recompute the basis for the validation/test set
            return_dict = self.transformation(y, covariate, new_input=True)
            output = return_dict["output"]
        else:
            output = y
        return output
        
    def conditional_sample(self, n_samples, sample_indices, sample_range, fixed_values, max_attempts=False, covariate=False):
        """
        Perform importance sampling for conditional generation of samples.
        
        :param n_samples: Number of samples to generate
        :param sample_indices: Indices of data dimensions to sample
        :param sample_range: Tuple (min, max) for sampling range
        :param fixed_values: Dictionary with fixed values for remaining dimensions
        :param covariate: Boolean flag for covariate handling in likelihood
        :return: Generated samples of shape (n_samples, data_dim)
        """
        with torch.no_grad():
            num_dim = self.number_variables
            
            # Initialize samples with fixed values
            if max_attempts == False:
                max_attempts = n_samples * 10  # Sample more than needed for rejection sampling
            samples = torch.full((max_attempts, num_dim), float('nan'))
            for idx, value in fixed_values.items():
                samples[:, idx] = value  # Set fixed values
            
            # Sample the desired dimensions uniformly within sample_range
            for sample_idx in sample_indices:
                min_val, max_val = sample_range[sample_idx]
                samples[:, sample_idx] = torch.rand((max_attempts)) * (max_val - min_val) + min_val
            
            # Compute acceptance probabilities using log likelihood
            log_probs = self.log_likelihood(samples, covariate=covariate)  # Shape: (max_attempts,)
            probs = torch.exp(log_probs - torch.max(log_probs))  # Normalize to avoid overflow
            probs /= torch.max(probs)  # Scale probabilities between 0 and 1
            
            # Accept/reject samples based on computed probabilities
            accept_mask = torch.rand(max_attempts) < probs
            accepted_samples = samples[accept_mask]
            
            # Ensure we return exactly n_samples
            if accepted_samples.shape[0] < n_samples:
                warnings.warn("Not enough accepted samples, increase max_attempts or adjust sampling method.")
            
            return accepted_samples#[:n_samples]
        
        
    def approximate_transformation_inverse(self):
        self.transformation.approximate_inverse(device=self.device)
            
            
            
        