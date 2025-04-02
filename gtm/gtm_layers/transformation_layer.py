import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt
from torch import optim
from tqdm import tqdm
import seaborn as sns
import warnings


from gtm.gtm_splines.bernstein_prediction import bernstein_prediction, compute_multivariate_bernstein_basis, restrict_parameters
from gtm.gtm_splines.bspline_prediction_vectorized import bspline_prediction_vectorized, compute_multivariate_bspline_basis
from gtm.gtm_splines.bspline_prediction_old import bspline_prediction




class Transformation(nn.Module):
    def __init__(self, degree, number_variables, polynomial_range, monotonically_increasing=True, spline="bernstein", span_factor=torch.tensor(0.1),
                 number_covariates=False, initial_log_transform=False, calc_method_bspline="Naive_Basis",span_restriction="None", spline_order=3): #device=None
        super().__init__()
        self.type = "transformation"
        self.degree  = degree
        self.number_variables = number_variables
        self.polynomial_range = torch.FloatTensor(polynomial_range)
        self.monotonically_increasing = monotonically_increasing
        
        # For padding!
        self.max_degree = max(self.degree)

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
        self.params = nn.ParameterList(self.compute_starting_values())
        
        if spline == "bspline" :
            self.spline_prediction = self.bspline_prediction_method
        elif spline == "bernstein":
            self.spline_prediction = self.bernstein_prediction_method
            warnings.warn("Warning: Custom Spline Order is not implemented for Bernstein, only for B-Spline.")
        else:
            warnings.warn("Warning: Unknown Spline string passed, use bspline or bernstein instead. bspline is default.")
        
        # Defining knots for vectorized compute
        max_cols = max(self.degree)+1
        self.knots_list = list()    
        for var in range(self.number_variables):
            
            if self.spline_order == 2:
                n = self.degree[var] + 1
            elif self.spline_order == 3:
                n = self.degree[var] + 2
            
            
            distance_between_knots = (self.polynomial_range[1,var] - self.polynomial_range[0,var]) * (1 + self.span_factor) / (n - 1)
            
            knots = torch.linspace(self.polynomial_range[0,var] * (1 + self.span_factor) - self.spline_order * distance_between_knots,
                            self.polynomial_range[1,var] * (1 + self.span_factor) + self.spline_order * distance_between_knots,
                            n + 4, dtype=torch.float32) 
            
            self.knots_list.append(
                torch.cat([knots, torch.zeros(max_cols + 5 - knots.size(0))+max(self.polynomial_range[1,:])*2 ])
                                   )
        
        self.padded_knots = torch.vstack(
            self.knots_list
            ).T
        
        if len(set(self.degree)) > 1:
            self.varying_degrees = True 
        else :
            self.varying_degrees = False
         
        if self.varying_degrees == False:
            self.padded_knots = self.padded_knots[:,0]
            
        # Create a mask to track valid values
        self.padded_params_mask = torch.vstack([
            torch.cat([torch.ones(p.size(0), dtype=torch.int64), torch.zeros(max_cols - p.size(0), dtype=torch.int64)]) for p in self.params
        ]).T  # Shape (num_params, max_cols)
        
        # Fore more efficient memory allocation e.g. to not create a new tensor every time in the forward pass
        self.padded_params = torch.zeros((max_cols,len(self.params)))  # Pre-allocate
            
        # TODO: Sort this out, I fixed the values here as it is currently the fastest computation by far
        #self.store_basis = True
        self.store_basis_training = False
        #self.calc_method_bspline = "deBoor" #"deBoor" #"deBoor" #"Naive"

    def compute_starting_values(self):
        """
        Computes Starting Values for the Transformation layer with variable knots for different data dimensions.

        :return: starting values tensor
        """
        par_restricted_opts = []
        
        for var_num in range(self.number_variables):
            min_val = self.polynomial_range[0][var_num]
            max_val = self.polynomial_range[1][var_num]
            
            par_unristricted = torch.linspace(min_val, max_val, self.degree[var_num] + 1)
            par_restricted_opt = par_unristricted
            par_unristricted[1:] = torch.log(torch.exp(par_restricted_opt[1:] - par_restricted_opt[:-1]) - 1)
            
            if self.number_covariates == 1:
                par_unristricted = par_unristricted.repeat(self.degree[var_num]  + 1, 1).T.flatten()
            elif self.number_covariates > 1:
                raise NotImplementedError("Only implemented for 1 or No covariates!")

            par_restricted_opts.append(nn.Parameter(par_unristricted))
        return par_restricted_opts
    
    def create_return_dict_transformation(self, input):
        return {"output": input.clone(),
                "log_d": torch.zeros(input.size(), device=input.device),
                "transformation_second_order_ridge_pen_sum": 0,
                "second_order_ridge_pen_sum": 0,
                "first_order_ridge_pen_sum": 0,
                "param_ridge_pen_sum": 0,
                "scores": 0,
                "hessian": 0,
                "output_first_derivativ": torch.zeros(input.size(), device=input.device),
                "output_second_derivativ": torch.zeros(input.size(), device=input.device),
                "output_third_derivativ": torch.zeros(input.size(), device=input.device)
                }
        
    def bspline_prediction_method(self, input, var_num, covariate, derivativ, return_penalties, monotonically_increasing=False, inverse=False):
    
        return bspline_prediction(
                self.params[var_num].unsqueeze(1)  if inverse==False else self.params_inverse[var_num].unsqueeze(1),
                input[:,var_num],
                self.degree[var_num] if not inverse else self.degree_inverse[var_num],
                self.polynomial_range[:, var_num] if inverse==False else self.polynomial_range_inverse[:, var_num],                
                monotonically_increasing=monotonically_increasing,
                derivativ=derivativ,
                return_penalties=return_penalties,
                span_factor=self.span_factor,
                span_restriction=self.span_restriction,
                covariate=covariate,
                params_covariate=self.params_covariate,
                covaraite_effect=self.covaraite_effect,
                calc_method=self.calc_method_bspline,
                order=self.spline_order
            )
        
    def bernstein_prediction_method(self, input, var_num, covariate, derivativ, return_penalties, monotonically_increasing=False, inverse=False):
        
        return bernstein_prediction(
                            self.params[:, var_num] if inverse==False else self.params_inverse[:, var_num],
                            input[:, var_num],
                            self.degree,
                            self.polynomial_range[:, var_num] if inverse==False else self.polynomial_range_inverse[:, var_num],
                            monotonically_increasing=monotonically_increasing if inverse==False else self.monotonically_increasing_inverse,
                            derivativ=derivativ,
                            span_factor=self.span_factor if inverse==False else self.span_factor_inverse,
                            covariate=covariate,
                            params_covariate=self.params_covariate) 
        
        
    def generate_basis(self, input, covariate, inverse=False):

        if not inverse:
            span_factor = self.span_factor
            polynomial_range = torch.FloatTensor(self.polynomial_range).to(input.device)
            #degree = self.degree
            degree = self.degree[0]
            spline = self.spline
        else:
            span_factor = self.span_factor_inverse
            polynomial_range = torch.FloatTensor(self.polynomial_range_inverse).to(input.device)
            #degree = self.degree_inverse
            degree = self.degree_inverse[0]
            spline = self.spline_inverse

        if spline == "bernstein":
            self.multivariate_basis = compute_multivariate_bernstein_basis(input=input,
                                                                                     degree=degree,
                                                                                     polynomial_range=polynomial_range,
                                                                                     span_factor=span_factor,
                                                                                     derivativ=0,
                                                                                     covariate=covariate)

            self.multivariate_basis = self.multivariate_basis.to(input.device)

            self.multivariate_basis_derivativ_1 = compute_multivariate_bernstein_basis(input=input,
                                                                                                 degree=degree,
                                                                                                 polynomial_range=polynomial_range,
                                                                                                 span_factor=span_factor,
                                                                                                 derivativ=1,
                                                                                                 covariate=covariate)

            self.multivariate_basis_derivativ_1 = self.multivariate_basis_derivativ_1.to(input.device)


        elif spline == "bspline":
            
            self.multivariate_basis = compute_multivariate_bspline_basis(input, self.degree, self.polynomial_range, span_factor, covariate=covariate) #device=self.device)

            self.multivariate_basis_derivativ_1 = compute_multivariate_bspline_basis(input, self.degree, self.polynomial_range, span_factor, covariate=covariate, derivativ=1)# device=self.device)

            # second derivative required for score matching
            self.multivariate_basis_derivativ_2 = compute_multivariate_bspline_basis(input, self.degree,
                                                                                               self.polynomial_range,
                                                                                               span_factor,
                                                                                               covariate=covariate,
                                                                                               derivativ=2)  

            # third derivative required for score matching
            self.multivariate_basis_derivativ_3 = compute_multivariate_bspline_basis(input, self.degree,
                                                                                               self.polynomial_range,
                                                                                               span_factor,
                                                                                               covariate=covariate,
                                                                                               derivativ=3) 

    def transformation(self, input, derivativ=0, inverse=False):
        # FunFact:
        # due to the basis being compute in generate_basis, the input is only used to have the correct dimensions for the output

        # input dims: 0: observation number, 1: variable

        if derivativ==0:
            basis = self.multivariate_basis#.to(input.device)
        elif derivativ==1:
            basis = self.multivariate_basis_derivativ_1#.to(input.device)
        elif derivativ == 2: # for score matching
             basis = self.multivariate_basis_derivativ_2  # .to(input.device)
        elif derivativ == 3: # for score matching
             basis = self.multivariate_basis_derivativ_3  # .to(input.device)

        #if self.spline == "bernstein":
        if not inverse:
            params_restricted = restrict_parameters(params_a=self.padded_params, covariate=self.number_covariates, degree=self.max_degree, monotonically_increasing=self.monotonically_increasing, device=input.device)
        else:
            params_restricted = restrict_parameters(params_a=self.params_inverse, covariate=self.number_covariates, degree=self.degree_inverse, monotonically_increasing=self.monotonically_increasing_inverse, device=input.device)
        # Explanation:
        # multivariate_bernstein_basis: 0: observation, 1: basis, 2: variable
        # params: 0: basis, 1: variable
        # output: 0: observation, 1: variable
        # Comment: we do normal multiplication as we want to multiply the parameters of each varaible only with its own basis
        #          we sum over dim 1 which is the basis
        #          note we use the restricted parameters


        output = torch.sum(basis * params_restricted.unsqueeze(0), (1))

        # penalities
        second_order_ridge_pen = torch.diff(params_restricted, n=2, dim=0).square().sum()
        first_order_ridge_pen = 0 
        param_ridge_pen = 0

        return output, second_order_ridge_pen, first_order_ridge_pen, param_ridge_pen
    
    
    def store_basis_forward(self, input, log_d = 0, inverse=False, return_scores_hessian=False):
        
        return_dict = self.create_return_dict_transformation(input)

        # Regenerate the padded parameter tensor every forward pass
        # has to be done in the forward
        self.padded_params = torch.stack([
            torch.nn.functional.pad(p, (0, self.max_degree + 1 - p.size(0))) for p in self.params
        ]).T
        
        if not inverse:
            output, second_order_ridge_pen_sum, _, _ = self.transformation(input, derivativ=0)
            output_first_derivativ, _, _, _ = self.transformation(input, derivativ=1)
            log_d = log_d + torch.log(output_first_derivativ) 
            
            if return_scores_hessian == True:
                output_second_derivativ, _, _, _ = self.transformation(input, derivativ=2)
                output_third_derivativ, _, _, _ = self.transformation(input, derivativ=3)
                scores = -1*output*output_first_derivativ + 1/output_first_derivativ * output_second_derivativ
                hessian = -1 * output * output_second_derivativ -1 * output_first_derivativ * output_first_derivativ + \
                        1/output_first_derivativ * output_third_derivativ - (output_second_derivativ**2) / (output_first_derivativ**2)
            else:
                scores = 0
                hessian = 0
                output_second_derivativ = 0
                output_third_derivativ = 0
            
        else:
            output, second_order_ridge_pen_sum, _, _ = self.transformation(input, derivativ=0, inverse=True)
            log_d = 0
            scores = 0
            hessian = 0
            output_first_derivativ = 0
            output_second_derivativ = 0
            output_third_derivativ = 0
            
        return_dict["output"] = output
        return_dict["log_d"] = log_d
        return_dict["transformation_second_order_ridge_pen_sum"] = second_order_ridge_pen_sum
        return_dict["scores"] = scores
        return_dict["hessian"] = hessian
        return_dict["output_first_derivativ"] = output_first_derivativ
        return_dict["output_second_derivativ"] = output_second_derivativ
        return_dict["output_third_derivativ"] = output_third_derivativ
            
        return return_dict    
    
    def for_loop_forward(self, input, covariate, log_d = 0, inverse=False):
        
        return_dict = self.create_return_dict_transformation(input)
        
        for var_num in range(self.number_variables):
            
            if self.number_covariates > 1:
                warnings.warn("Warning: With for-loop computation only implemented with one or no covariate.")
            
            return_dict["output"][:, var_num], second_order_ridge_pen, _, _ = self.spline_prediction(input, 
                                                      var_num, 
                                                      covariate, 
                                                      derivativ=0, 
                                                      return_penalties=True, 
                                                      monotonically_increasing=True if inverse==False else False,
                                                      inverse=inverse)
            
            return_dict["output_first_derivativ"][:, var_num], _, _, _ = self.spline_prediction(input, 
                                                      var_num, 
                                                      covariate, 
                                                      derivativ=1, 
                                                      return_penalties=True, 
                                                      monotonically_increasing=True if inverse==False else False,
                                                      inverse=inverse)
            
            return_dict["log_d"][:, var_num] = return_dict["output_first_derivativ"][:, var_num]
                
            return_dict["second_order_ridge_pen_sum"] += second_order_ridge_pen
            
        return_dict["log_d"] = torch.log(return_dict["log_d"])
        
        return return_dict    
    
    def vmap_forward(self, input, covariate, log_d = 0, inverse=False):
        
        if self.number_covariates > 1:
                warnings.warn("Warning: With for-loop computation only implemented with one or no covariate.")
            
        return_dict = self.create_return_dict_transformation(input)

        padded_params = torch.vstack([torch.nn.functional.pad(p, (0, self.max_degree + 1 - p.size(0))) for p in self.params]).T
         
        padded_params = restrict_parameters(padded_params, #.contiguous().unsqueeze(1), 
                                            covariate=covariate, degree=self.max_degree, monotonically_increasing=True, device=input.device)
        
        if not hasattr(self, 'padded_knots') or self.padded_knots.device != input.device:
            self.padded_knots = self.padded_knots.to(input.device)

        
        return_dict["output"], return_dict["second_order_ridge_pen_sum"] , _, _ = bspline_prediction_vectorized(padded_params if inverse==False else torch.vstack(self.params_inverse).T,
                           input,
                           self.padded_knots,
                           self.max_degree, 
                           self.polynomial_range[:, 0] if inverse==False else self.polynomial_range_inverse[:, 0], 
                           derivativ=0, 
                           return_penalties=True, 
                           calc_method=self.calc_method_bspline,
                           span_factor=self.span_factor, 
                           span_restriction=self.span_restriction,
                           covariate=covariate, 
                           params_covariate=self.params_covariate, 
                           covaraite_effect=self.covaraite_effect,
                           penalize_towards=0, 
                           order=self.spline_order,
                           varying_degrees=self.varying_degrees,
                           params_a_mask=self.padded_params_mask)
        
        return_dict["log_d"] = bspline_prediction_vectorized(padded_params if inverse==False else torch.vstack(self.params_inverse).T, 
                           input, 
                           self.padded_knots, 
                           self.max_degree, 
                           self.polynomial_range[:, 0] if inverse==False else self.polynomial_range_inverse[:, 0], 
                           derivativ=1, 
                           return_penalties=False, 
                           calc_method=self.calc_method_bspline,
                           span_factor=self.span_factor, 
                           span_restriction=self.span_restriction,
                           covariate=covariate, 
                           params_covariate=self.params_covariate, 
                           covaraite_effect=self.covaraite_effect,
                           penalize_towards=0, 
                           order=self.spline_order,
                           varying_degrees=self.varying_degrees,
                           params_a_mask=self.padded_params_mask)
        
        return_dict["log_d"] = torch.log(return_dict["log_d"])
        
        return return_dict    
        
    def forward(self, input, covariate=False, log_d = 0, inverse = False, return_log_d = False, new_input=True, store_basis=False, return_derivatives=False, return_scores_hessian=False):
        # input dims: 0: observaton number, 1: variable
        # Important: set the default of new input to true, otherwise we might use training set for validation results by accident
        #            Thus only specifiy new_input=False during training
        #             store_basis to redefine the basis (needed for validation step and out of sample prediction as well as sampling)

        if store_basis == True and self.store_basis_training == True:
            if inverse:
                new_input=True
                warnings.warn("Warning: inverse changes stored basis, set new_input True in next pass through model.")
            # We only want to define the basis once for the entire training
            if new_input is True or self.multivariate_basis is False and self.multivariate_basis_derivativ_1 is False:
                self.generate_basis(input=input, covariate=covariate, inverse=inverse)

            return_dict = self.store_basis_forward(input, log_d = 0, inverse=inverse, return_scores_hessian=return_scores_hessian)
            
            
        else:
            if return_scores_hessian == True:
                warnings.warn("Warning: return_scores_hessian is only available if store_basis is True.")
                            
            if inverse:

                return_dict = self.for_loop_forward(input, covariate, log_d = 0, inverse=inverse)
            else:
                return_dict = self.vmap_forward(input, covariate, log_d = 0, inverse=inverse)
        
        return return_dict    

    def approximate_inverse(self, spline_inverse="bspline", degree_inverse=150, span_factor_inverse=0.2,
                            input_covariate=False, num_samples=10000, device="cpu"):

        #if self.initial_log_transform==True:
        #    input = torch.log(input+0.01)

        if input_covariate is not False:
            covariate_space = input_covariate[input_covariate.squeeze().multinomial(num_samples, replacement=True)]
        else:
            covariate_space = False

        if degree_inverse == 0:
            degree_inverse = 2 * self.degree

        self.monotonically_increasing_inverse = False
        self.span_factor_inverse = span_factor_inverse
        self.span_restriction_inverse = False

        input_space = torch.zeros((num_samples, self.number_variables), dtype=torch.float32, device=device)
        for var_number in range(self.number_variables):
            input_space[:, var_number] = torch.linspace(self.polynomial_range[0,var_number].item(),self.polynomial_range[1,var_number].max().item(),num_samples,device=device)

        return_dict = self.forward(input=input_space,covariate=covariate_space)
        output_space = return_dict["output"]

        polynomial_range_inverse = torch.zeros((2, self.number_variables), dtype=torch.float32, device=device)

        for var_number in range(self.number_variables):
            span_var_number = output_space[:, var_number].max() - output_space[:, var_number].min()
            polynomial_range_inverse[:, var_number] = torch.tensor([output_space[:, var_number].min() - span_var_number*span_factor_inverse,
                                                                    output_space[:, var_number].max() + span_var_number*span_factor_inverse],
                                                                   dtype=torch.float32, device=device)

        inv_trans = Transformation(degree=self.number_variables * [degree_inverse],
                                   number_variables=self.number_variables,
                                   polynomial_range=polynomial_range_inverse,
                                   monotonically_increasing=self.monotonically_increasing_inverse,
                                   spline=spline_inverse,
                                   number_covariates=self.number_covariates,
                                   span_restriction=self.span_restriction_inverse)

        inv_trans.generate_basis(input=output_space.detach(),covariate=covariate_space,inverse=False)
        
        params_tensor = inv_trans.params
        
        for num_var in range(len(inv_trans.params)):
            res = np.linalg.lstsq(inv_trans.multivariate_basis[:,:,num_var].detach().numpy(), input_space[:,num_var].detach().numpy(), rcond=None)
            params_tensor[num_var] = nn.Parameter(torch.tensor(res[0], dtype=torch.float32))

        self.polynomial_range_inverse = polynomial_range_inverse
        self.params_inverse = nn.ParameterList(params_tensor)
        self.spline_inverse = spline_inverse
        self.degree_inverse = self.number_variables * [degree_inverse]

    def __repr__(self):
        return f"Transformation(degree={self.degree}, number_variables={self.number_variables}, polynomial_range={self.polynomial_range}, monotonically_increasing={self.monotonically_increasing}, spline={self.spline}, number_covariates={self.number_covariates})"