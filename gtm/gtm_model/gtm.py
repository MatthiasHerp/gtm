import torch
from torch import nn
#from gtm.layers.flip import Flip
from gtm.gtm_layers.transformation_layer import *
from gtm.gtm_layers.decorrelation_layer import Decorrelation
from gtm.gtm_layers.layer_utils import generate_diagonal_matrix
from gtm.gtm_training.objective_functions import log_likelihood, training_objective
from gtm.gtm_training.training_helpers import train, if_float_create_lambda_penalisation_matrix
from gtm.gtm_plots_analysis.compute_conditional_independence_kld import *
#from gtm.simulation_study.simulation_study_helpers import plot_marginals, plot_densities

import optuna
from optuna.samplers import TPESampler 
import itertools

import scipy
from gtm.gtm_model.tm import TM


class GTM(nn.Module):
    def __init__(self, 
                 number_variables,
                 transformation_spline_range=list([[-15], [15]]), 
                 decorrelation_spline_range=list([[-15], [15]]), 
                 spline_transformation="bspline", spline_decorrelation="bspline", # bernstein bernstein bspline
                 degree_transformations=15, degree_decorrelation=20, span_factor=torch.tensor(0.1), span_restriction="reluler", #span_factor=torch.tensor(0.1)
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
        
        self.number_variables = number_variables

        # Repeat polynomial ranges for all variables as this is the range for the bsplines essentially
        self.transformation_spline_range = list([transformation_spline_range[0] * self.number_variables,
                                                transformation_spline_range[1] * self.number_variables])
        self.decorrelation_spline_range = list([decorrelation_spline_range[0] * self.number_variables,
                                                decorrelation_spline_range[1] * self.number_variables])
  

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
                                     spline_range=self.transformation_spline_range, span_factor=self.span_factor,
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
                                    spline_range=self.decorrelation_spline_range, span_factor=self.span_factor,
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
    
        
    def forward(self, y, covariate=False, train=True, evaluate=True, return_scores_hessian=False, return_lambda_matrix = True):
        
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
                else:
                    return_dict_nf_mctm["output"] = y.clone()
                    return_dict_nf_mctm["log_d"] = torch.zeros(y.size()).float()
                    

            elif evaluate:
                if self.num_trans_layers > 0:
                    # new input true as we need to recompute the basis for the validation/test set
                    return_dict_transformation = self.transformation(y, covariate, log_d=log_d, return_log_d=True, new_input=True, store_basis=True, return_scores_hessian=return_scores_hessian)
                    
                    return_dict_nf_mctm["output"] = return_dict_transformation["output"]
                    return_dict_nf_mctm["log_d"] = return_dict_transformation["log_d"]
                else:
                    return_dict_nf_mctm["output"] = y.clone()
                    return_dict_nf_mctm["log_d"] = torch.zeros(y.size()).float()
            
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


                    return_dict_decorrelation = self.decorrelation_layers[i](return_dict_nf_mctm["output"], covariate, 0,
                                                                return_log_d=True, return_penalties=True, return_scores_hessian=return_scores_hessian)
                    
                    return_dict_nf_mctm["output"] = return_dict_decorrelation["output"]
                    return_dict_nf_mctm["log_d"] += return_dict_decorrelation["log_d"] #required if the layers are multiplicative
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
    
    def training_objective(self, samples, penalty_params, train_covariates=False, lambda_penalty_params: torch.Tensor =False, 
                           adaptive_lasso_weights_matrix: torch.Tensor =False, 
                           avg = True, lambda_penalty_mode="square", objective_type = "negloglik"):
        
        return training_objective(self, samples = samples, penalty_params = penalty_params, train_covariates = train_covariates, lambda_penalty_params = lambda_penalty_params, 
                                  adaptive_lasso_weights_matrix = adaptive_lasso_weights_matrix,
                                  avg = avg, lambda_penalty_mode = lambda_penalty_mode, objective_type = objective_type)
    
    
    def __train__(self, train_dataloader, validate_dataloader=False, train_covariates=False, validate_covariates=False, penalty_params=torch.FloatTensor([0,0,0,0]), adaptive_lasso_weights_matrix = False,
                  lambda_penalty_params=False, learning_rate=1, iterations=2000, verbose=False, patience=5, min_delta=1e-7,
          optimizer='LBFGS', lambda_penalty_mode="square", objective_type="negloglik", ema_decay=False, seperate_copula_training=False,
          max_batches_per_iter=None):

        if lambda_penalty_params is not False:
            lambda_penalty_params = lambda_penalty_params.to(self.device)
            
        if adaptive_lasso_weights_matrix is not False:
            adaptive_lasso_weights_matrix = adaptive_lasso_weights_matrix.to(self.device)
        
        if seperate_copula_training==True:
            self.transformation.params.requires_grad=False
        
        return_dict_model_training = train(self, train_dataloader=train_dataloader, validate_dataloader=validate_dataloader, train_covariates=train_covariates, validate_covariates=validate_covariates, penalty_params=penalty_params, lambda_penalty_params=lambda_penalty_params, learning_rate=learning_rate, 
                     iterations=iterations, verbose=verbose, patience=patience, min_delta=min_delta, optimizer=optimizer, lambda_penalty_mode=lambda_penalty_mode, objective_type=objective_type, adaptive_lasso_weights_matrix = adaptive_lasso_weights_matrix, max_batches_per_iter=max_batches_per_iter)
        
        if seperate_copula_training==True:
            self.transformation.params.requires_grad=True
        
        return return_dict_model_training
    
    def pretrain_tranformation_layer(self, train_dataloader, validate_dataloader=False, train_covariates=False, validate_covariates=False, penalty_params=torch.FloatTensor([0,0,0,0]), lambda_penalty_params=False, learning_rate=1, iterations=2000, verbose=False, patience=5, min_delta=1e-7, return_plot=True,
          optimizer='LBFGS', lambda_penalty_mode="square", objective_type="negloglik", max_batches_per_iter=None):
        
        optimizer='LBFGS'
        warnings.warn("Optimiser for pretrain_tranformation_layer is always LBFGS. If this is an issue change the code.")
        
        self.transform_only = True
        lambda_penalty_params = False # makes objective not check lambda matrix
    
        if lambda_penalty_params is not False:
            # set to false because this is the pretraining of the transformation layer
            lambda_penalty_params = False
            
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
                                           objective_type=objective_type,
                                           max_batches_per_iter=max_batches_per_iter)
        
        self.transform_only = False
        
        return return_dict_model_training
    
    
    def find_minimal_transformation_degrees(self,train_dataloader, validate_dataloader, iterations=100, degrees_try_list=list(range(5,155,5)), max_batches_per_iter=False):
        
        optimal_degree = []
        optimal_degree_pvalue = []
        for dimension in range(self.number_variables):
            for degree in degrees_try_list:
                print("Starting run for data dim ",dimension," with degrees of ",degree)
                try:
                    tm_model = TM(degree=degree, spline_range=[self.transformation_spline_range[0][dimension],
                                                            self.transformation_spline_range[1][dimension]])
                    tm_model.subset_dimension = dimension
            
                    train_dict = tm_model.__train__(train_dataloader=train_dataloader, validate_dataloader=validate_dataloader, iterations=iterations, optimizer="LBFGS",
                                    penalty_params=[0,0,0,0], adaptive_lasso_weights_matrix=False, lambda_penalty_params=False, 
                                    max_batches_per_iter=max_batches_per_iter)
                    
                    z_tilde = []
                    #y_train_all = []
                    for y_train in validate_dataloader:
                        y_train_sub = y_train[:,tm_model.subset_dimension]
                        z_tilde.append(tm_model.latent_space(y_train_sub))
                        #y_train_all.append(y_train_sub)
                    z_tilde = torch.hstack(z_tilde).detach().numpy()
                    #y_train_all = torch.hstack(y_train_all).detach().numpy()
                    
                    #perform Shapiro-Wilk test for normality
                    pv = scipy.stats.shapiro(z_tilde[:5000]).pvalue # becuase warning that pvalue may not be accurate for larger than 5000 obs in the package
                    #print(pv)
                    #plt.hist(y_train_all,bins=100)
                    #plt.hist(z_tilde,bins=100)
                    #plt.hist(z_tilde,bins=100)
                    if pv >= 0.01:
                    
                        print("pvalue is ",pv," for data dim ",dimension," with degrees of ",degree)
                        
                        optimal_degree.append(degree)
                        optimal_degree_pvalue.append(pv)
                        
                        # for next loop iteration to not always plot
                        pv = 0 
                        
                        break
                    # Handels the case where even the max degree is not enough
                    elif degree == max(degrees_try_list):
                        optimal_degree.append(degree)
                        optimal_degree_pvalue.append(pv)
                except:
                    continue
                
        return optimal_degree, optimal_degree_pvalue
        
    

    def compute_precision_matrix(self, y, covariate=False):

        with torch.no_grad():
            return_dict = self.forward(y, covariate=covariate, evaluate=True, train=False, return_lambda_matrix=True)

            precision_matrix = torch.matmul(torch.transpose(return_dict["lambda_matrix_global"], 1, 2),
                                            return_dict["lambda_matrix_global"])

        return precision_matrix
    
    
    def compute_correlation_matrix(self, y, covariate=False):
        
        def p_to_corr(matrix):
            d = matrix.size(0)
            diag_sqrt = torch.diag(matrix) ** 0.5
            matrix_std_multiplied = np.matmul(torch.reshape(diag_sqrt, (d, 1)), torch.reshape(diag_sqrt, (1, d)))
            return -1 * matrix / matrix_std_multiplied

        with torch.no_grad():
            precision_matrix = self.compute_precision_matrix( y, covariate=False)
            correlation_matrix_train = torch.stack([p_to_corr(precision_matrix[obs_num,:,:]) for obs_num in range(precision_matrix.size(0))])

        return correlation_matrix_train


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
        
        
    def return_objective_for_hyperparameters(self, train_dataloader, validate_dataloader=False, train_covariates=False, validate_covariates=False, penalty_params=torch.FloatTensor([0,0,0,0]), 
                  adaptive_lasso_weights_matrix = False,
                  lambda_penalty_param=False, learning_rate=1, iterations=2000, patience=5, min_delta=1e-7,
          optimizer='LBFGS', lambda_penalty_mode="square", objective_type="negloglik", seperate_copula_training=False,
          max_batches_per_iter=None, pretrained_transformation_layer=False):
    
        import copy
        
        gtm_tuning = copy.deepcopy(self)
        
        gtm_tuning.to(self.device)
        gtm_tuning.device = self.device
        
        # Logic here:
        # we only want to pretrain the transformation layer once if we do not do cross validation folds
        # In that case each pretaining is basically the same, as same init params and same hyperparameters
        # If cross_validation_folds is False then only in the first trial we do a pretrain and then store the transformation layer pretrained model
        # In each subsequent trial we load the pretrained model and directly do the joint training
        # This only works if we pretrain without a penalty on the transformation layer
        if pretrained_transformation_layer == True:
            
            if hasattr(self, 'pretrained_transformation_layer_model_state_dict'): #pretrained_transformation_layer_model
                #gtm_tuning.load_state_dict(self.pretrained_transformation_layer_model.state_dict())
                gtm_tuning.load_state_dict(self.pretrained_transformation_layer_model_state_dict) 
            else:
                gtm_tuning.pretrain_tranformation_layer(train_dataloader=train_dataloader,
                                                    validate_dataloader=validate_dataloader,
                                                    train_covariates=train_covariates,
                                                    validate_covariates=validate_covariates,
                                                    penalty_params=penalty_params,
                                                    lambda_penalty_params=lambda_penalty_param,
                                                    iterations=iterations, 
                                                    learning_rate=learning_rate,
                                                    patience=patience,
                                                    min_delta=min_delta,
                                                    verbose=False,
                                                    optimizer=optimizer,
                                                    lambda_penalty_mode=lambda_penalty_mode,
                                                    objective_type=objective_type,
                                                    max_batches_per_iter=max_batches_per_iter)
                
                if self.cross_validation_folds == False:
                    #self.pretrained_transformation_layer_model = copy.deepcopy(gtm_tuning)
                    self.pretrained_transformation_layer_model_state_dict = gtm_tuning.state_dict()
                
        
        gtm_tuning.__train__(train_dataloader=train_dataloader,
                validate_dataloader=validate_dataloader, 
                train_covariates=train_covariates,
                validate_covariates=validate_covariates,
                penalty_params=penalty_params,
                lambda_penalty_params=lambda_penalty_param,
                adaptive_lasso_weights_matrix = adaptive_lasso_weights_matrix,
                iterations=iterations, 
                learning_rate=learning_rate,
                patience=patience,
                min_delta=min_delta,
                verbose=False,
                optimizer=optimizer,
                lambda_penalty_mode=lambda_penalty_mode,
                objective_type=objective_type,
                seperate_copula_training=seperate_copula_training,
                max_batches_per_iter=max_batches_per_iter)
        
        num_batches=0
        target=0
        for y_validate in validate_dataloader:
            y_validate = y_validate.to(self.device)
            num_batches += 1
            covar_batch = False
            if objective_type == "negloglik":
                target += gtm_tuning.log_likelihood(y_validate, covar_batch).cpu().detach().numpy().mean()  # .mean()
            elif objective_type == "score_matching" or objective_type == "single_sliced_score_matching":
                #TODO: does the -1 * make sense is it not already in the exact_score_loss method
                target += -1 * gtm_tuning.exact_score_matching(y_validate, covar_batch).cpu().detach().numpy().mean()  # .mean() # maximize the reverse score loss e.g.g minimize the loss
            elif objective_type == "noise_contrastive_estimation":
                target += gtm_tuning.log_likelihood(y_validate, covar_batch).cpu().detach().numpy().mean()  #TODO: For Now! Its cheating right?
        
        target = target / num_batches     
        
            
        # Handelling CUDA Out of Memory Error
        if self.device == "cuda":
            # Explicitly delete the model to free up memory
            del gtm_tuning
            
            # Clear the cache
            torch.cuda.empty_cache()
        
        return target
        
    def hyperparameter_tune_penalties(self, 
                                      train_dataloader, 
                                      validate_dataloader, 
                                      penvalueridge: list,
                                      penfirstridge: list,
                                      pensecondridge: list,
                                      ctm_pensecondridge: list,
                                      lambda_penalty_params: list,
                                      train_covariates=False, 
                                      validate_covariates=False, 
                                      adaptive_lasso_weights_matrix = False,
                                      learning_rate=1, 
                                      iterations=2000, 
                                      patience=5, 
                                      min_delta=1e-7, 
                                      optimizer='LBFGS', 
                                      lambda_penalty_mode="square", 
                                      objective_type="negloglik", 
                                      seperate_copula_training=False,
                                      max_batches_per_iter=None,
                                        tuning_mode="optuna",
                              cross_validation_folds=False,
                              random_state_KFold=42,
                              device=None,
                              pretrained_transformation_layer=False,
                              n_trials=15,
                              temp_folder=".", 
                              study_name=None):
        
        
        list_of_lists = [penvalueridge, penfirstridge, pensecondridge, 
                         ctm_pensecondridge,
                         lambda_penalty_params]
        hyperparameter_combinations_list = list(itertools.product(*list_of_lists))
        
        if train_covariates is False:
            number_covariates = 0
        else:
            number_covariates = 1

        if tuning_mode == "optuna":
            penvalueridge, penfirstridge, pensecondridge, ctm_pensecondridge, lambda_penalty_params  = hyperparameter_combinations_list[0]
            
            # so model has no marginal part
            if seperate_copula_training == True:
                num_trans_layers = 0
                
            def optuna_objective(trial, train_dataloader=train_dataloader, 
                                      validate_dataloader=validate_dataloader ): 
                
                if penvalueridge == None:
                    penvalueridge_opt = 0
                elif isinstance(penvalueridge, float) or isinstance(penvalueridge, int):
                    penvalueridge_opt = penvalueridge
                elif penvalueridge == "sample":
                    penvalueridge_opt = trial.suggest_float("penvalueridge", 0.0000001, 30, log=False) #True
                else:
                    warnings.warn("penvalueridge not understood. Please provide a float, int None, or the string \"sample\".")

                if penfirstridge == None:
                    penfirstridge_opt = 0
                elif isinstance(penfirstridge, float) or isinstance(penfirstridge, int):
                    penfirstridge_opt = penfirstridge
                elif penfirstridge == "sample":
                    penfirstridge_opt = trial.suggest_float("penfirstridge", 0.0000001, 30, log=False) # True
                else:
                    warnings.warn("penfirstridge not understood. Please provide a float, int None, or the string \"sample\".")
                    
                if pensecondridge == None:
                    pensecondridge_opt = 0
                elif isinstance(pensecondridge, float) or isinstance(pensecondridge, int):
                    pensecondridge_opt = pensecondridge
                elif pensecondridge == "sample":
                    pensecondridge_opt = trial.suggest_float("pensecondridge", 0.0000001, 30, log=False) # True
                else:
                    warnings.warn("pensecondridge not understood. Please provide a float, int None, or the string \"sample\".")
                    
                if ctm_pensecondridge == None:
                    ctm_pensecondridge_opt = 0
                elif isinstance(ctm_pensecondridge, float) or isinstance(ctm_pensecondridge, int):
                    ctm_pensecondridge_opt = ctm_pensecondridge
                elif ctm_pensecondridge == "sample":
                    ctm_pensecondridge_opt = trial.suggest_float("ctm_pensecondridge", 0.0000001, 30, log=False) # True
                else:
                    warnings.warn("ctm_pensecondridge not understood. Please provide a float, int None, or the string \"sample\".")
                    
                if lambda_penalty_params == None:
                    lambda_penalty_params_opt = 0
                elif isinstance(lambda_penalty_params, float) or isinstance(lambda_penalty_params, int):
                    lambda_penalty_params_opt = lambda_penalty_params
                elif lambda_penalty_params == "sample":
                    lambda_penalty_params_opt = trial.suggest_float("lambda_penalty_params", 0.0000001, 1, log=True)
                else:
                    warnings.warn("lambda_penalty_params not understood. Please provide a float, int None, or the string \"sample\".")
                    
                print("This Trial has the Hyperparameters:", 
                    "penvalueridge_opt:", penvalueridge_opt, " ", 
                    "penfirstridge_opt:", penfirstridge_opt, " ", 
                    "pensecondridge_opt:", pensecondridge_opt, " ", 
                    "ctm_pensecondridge_opt:", ctm_pensecondridge_opt, " ",
                    "lambda_penalty_params_opt:", lambda_penalty_params_opt)
                lambda_penalty_params_opt = if_float_create_lambda_penalisation_matrix(lambda_penalty_params_opt, num_vars=self.number_variables)
                penalty_params_opt = torch.tensor([penvalueridge_opt,
                                            penfirstridge_opt,
                                            pensecondridge_opt,
                                            ctm_pensecondridge_opt])
                
                if cross_validation_folds == False:
                    # define model, train the model with tuning params and return the objective value on the given validation set
                    target = self.return_objective_for_hyperparameters(train_dataloader, validate_dataloader, train_covariates, validate_covariates, penalty_params_opt, 
                                                                        adaptive_lasso_weights_matrix,
                                                                        lambda_penalty_params_opt, learning_rate, iterations, patience, min_delta, 
                                                                        optimizer, lambda_penalty_mode, objective_type, seperate_copula_training, max_batches_per_iter,
                                                                        pretrained_transformation_layer)
                    
                    return target
                else:
                    warnings.warn("cross validation based hyperparameter tuning is not oimplemented yet based on dataloaders")
                    ## Perform cross-validation
                    #for fold, (train_idx, val_idx) in enumerate(kf.split(dataset)):
                    #    print(f"Fold {fold + 1}: Train size = {len(train_idx)}, Val size = {len(val_idx)}")
#
                    #    # Create subset samplers
                    #    train_subset = Subset(dataset, train_idx)
                    #    val_subset = Subset(dataset, val_idx)
#
                    #    # Create DataLoaders
                    #    train_loader = DataLoader(train_subset, batch_size=16, shuffle=True)
                    #    val_loader = DataLoader(val_subset, batch_size=16, shuffle=False)
#
                    #    # Example: Iterate through one batch
                    #    for batch in train_loader:
                    #        x, y = batch
                    #        print(f"Train Batch Shape: {x.shape}, Labels: {y.shape}")
                    #        break  # Just show one batch per fold
                
            study = optuna.create_study(sampler=TPESampler(n_startup_trials=int(np.floor(n_trials/2)), #7
                                                       consider_prior=False, #True # is this useful without a prior weight?
                                                       # Set consider_prior=False othwise with score matching we got the error: raise ValueError("Prior weight must be positive.")
                                                       prior_weight=0,#1.0, #default value 1.0 but then does not explore the space as good I think
                                                       multivariate=True # experimental but very useful here as our parameters are highly correlated
                                                       ),
                                    storage='sqlite:///'+temp_folder+'/hyperparameter_tuning_study.db',
                                    #hyperparameter_tuning_study.db',
                                    direction='maximize',
                                    study_name=study_name,
                                    load_if_exists=True)
                    
            study.optimize(optuna_objective, n_trials=n_trials)
            print("hyperparameter_tuning done")
            return study
        
    def compute_conditional_independence_table(self,
                                        y = None,
                                        x = False,
                                        evaluation_data_type = "data",
                                        num_processes=10,
                                        sample_size = 1000,
                                        num_points_quad=20,
                                        optimized=False,
                                        copula_only=False,
                                        min_val=-5, 
                                        max_val=5):
        
        return compute_conditional_independence_kld(self,
                                        y,
                                        x,
                                        evaluation_data_type,
                                        num_processes,
                                        sample_size,
                                        num_points_quad,
                                        optimized,
                                        copula_only,
                                        min_val,
                                        max_val)
                
