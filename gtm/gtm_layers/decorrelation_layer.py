import warnings
#from functorch import vmap  # Requires PyTorch 1.10+
import torch
from torch import nn
import numpy as np
#from gtm.gtm_splines.bernstein_prediction import bernstein_prediction
from gtm.gtm_splines.bspline_prediction_vectorized import bspline_prediction_vectorized
from gtm.gtm_splines.bspline_prediction_old import bspline_prediction


class Decorrelation(nn.Module):
    def __init__(self, degree, number_variables, polynomial_range, spline="bspline", span_factor=torch.tensor(0.1), span_restriction="None",
                 number_covariates=False, list_comprehension = False, covaraite_effect="multiplicativ", calc_method_bspline="Naive_Basis",
                 affine_layer=False, degree_multi=False, spline_order=3):
        super().__init__()
        self.type = "decorrelation"
        self.degree  = degree
        self.number_variables = number_variables
        self.polynomial_range = torch.FloatTensor(polynomial_range)

        self.num_lambdas = number_variables * (number_variables-1) / 2
        self.spline = spline
        if spline == "bernstein":
            warnings.warn("Bernstein polynomial penalization is not implemented yet. oonly returns zeros hardcoded in bernstein_prediction.py fct")

        self.span_factor = span_factor

        self.span_restriction = span_restriction

        if spline == "bspline" :
            self.spline_prediction = self.bspline_prediction_method
        elif spline == "bernstein":
            self.spline_prediction = self.bernstein_prediction_method
        else:
            warnings.warn("Warning: Unknown Spline string passed, use bspline or bernstein instead. bspline is default.")
        
        self.spline_order = spline_order
        
        self.params = self.compute_starting_values_bspline(start_value=0.001)

        self.number_covariates = number_covariates

        self.list_comprehension = list_comprehension

        if self.number_covariates is not False:
            if self.number_covariates > 1:
                print("Warning, covariates not implemented for more than 1 covariate")
            self.params_covariate = self.compute_starting_values_bspline(start_value=0.001)
        else:
            self.params_covariate = False
        
        # Defines wether we have an additive or an affine (multiplicative) coupling layer
        self.affine_layer = affine_layer
        self.degree_multi = degree_multi
        if self.affine_layer is not False:
            self.params_multiplier = self.compute_starting_values_bspline(start_value=1.)
        else:
            self.params_multiplier = False

        self.covaraite_effect = covaraite_effect
        
        self.calc_method_bspline = calc_method_bspline
        
        #self.spline_order = spline_order
        
        # TODO: Sort this out, I fixed the values here as it is currently the fastest computation by far
        self.vmap = True #False #True
        self.list_comprehension = False #True #False
        #self.calc_method_bspline = "deBoor" #"deBoor" #"Naive"
        
        ###################################### New ###################################### 
        # The following code ensures that:
        # we have knots equally spanning the range of the number of degrees
        # we have the first an last knot on the bound of the span
        # we get equally space boundary knots outside the spane to ensure that at the boundary of the span we get bspline predicitons without errors
        
        # The distance between knots is the span divided by the number of knots minus 1 
        # because between D points where we have one point at the min and one at the max of a span we have D-1 intervals between knots
        distance_between_knots_in_bounds = (self.polynomial_range[1,0] - self.polynomial_range[0,0]) / (self.degree-1)
        
        number_of_bound_knots_per_side = self.spline_order #- 1 # for cubcic splines (order 3) we need 2 knots on each side 
        
        # We get the eqully spaced knots by equal spacing on the extended span
        self.knots = torch.linspace(
                            self.polynomial_range[0,0] - number_of_bound_knots_per_side * distance_between_knots_in_bounds,
                            self.polynomial_range[1,0] + number_of_bound_knots_per_side * distance_between_knots_in_bounds,
                            self.degree + 2 * number_of_bound_knots_per_side, # 2* because of two sides
                            dtype=torch.float32)
        ###################################### New ###################################### 
        
        #if self.spline_order == 2:
        #    n = self.degree + 1
        #elif self.spline_order == 3:
        #    n = self.degree + 2
        #    
        #    
        #distance_between_knots = (self.polynomial_range[1,0] - self.polynomial_range[0,0]) * (1 + self.span_factor) / (n - 1)
        #    
        #    
        #self.knots = torch.linspace(self.polynomial_range[0,0] * (1 + self.span_factor) - self.spline_order * distance_between_knots,
        #                    self.polynomial_range[1,0] * (1 + self.span_factor) + self.spline_order * distance_between_knots,
        #                    n + 4, dtype=torch.float32)
        
        self.var_num_list, self.covar_num_list = torch.tril_indices(self.number_variables, self.number_variables, offset=-1)
        
    
    def create_return_dict_decorrelation(self, input):
        lambda_matrix_general = torch.eye(self.number_variables, device=input.device).expand(input.size()[0],self.number_variables,self.number_variables).requires_grad_(True)
        return {"output": input.clone(), #.requires_grad_(True),
                "log_d": torch.zeros(input.size(),device=input.device), #.requires_grad_(True),
                "transformation_second_order_ridge_pen_sum": 0,
                "second_order_ridge_pen_sum": 0,
                "first_order_ridge_pen_sum": 0,
                "param_ridge_pen_sum": 0,
                "lambda_matrix": lambda_matrix_general.clone()#, 
                #"der_lambda_matrix": lambda_matrix_general.clone(),
                #"der2_lambda_matrix": lambda_matrix_general.clone()
                }
        
    
    def compute_starting_values_bspline(self,start_value=0.001):
        #p = torch.FloatTensor(np.repeat(np.repeat(start_value, self.degree + 1),
        #                                self.num_lambdas))
        p = torch.FloatTensor(np.repeat(np.repeat(start_value, self.degree + self.spline_order - 1),
                                        self.num_lambdas)) 

        if self.num_lambdas == 1:
            params = nn.Parameter(p.unsqueeze(1))
        else:
            #params = nn.Parameter(torch.reshape(p, (self.degree + 1, int(self.num_lambdas))))
            params = nn.Parameter(torch.reshape(p, (self.degree + self.spline_order - 1, int(self.num_lambdas))))

        return params
        
        
    def bspline_prediction_method(self, input, params_index, covariate, covar_num, derivativ, return_penalties, monotonically_increasing=False, multi=False):
        
        return bspline_prediction(
                            self.params[:, params_index] if multi==False else self.params_multiplier[:, params_index],
                            input[:, covar_num],
                            self.degree if multi==False else self.degree_multi,
                            self.polynomial_range[:, 0],
                            monotonically_increasing=monotonically_increasing,
                            derivativ=derivativ,
                            return_penalties=return_penalties,
                            span_factor=self.span_factor,
                            span_restriction=self.span_restriction,
                            covariate=covariate,
                            params_covariate=False, #self.params_covariate[:, covar_num],
                            covaraite_effect=self.covaraite_effect,
                            calc_method=self.calc_method_bspline,
                            order = self.spline_order)  
        
        
        
    def bernstein_prediction_method(self, input, params_index, covariate, covar_num, derivativ, return_penalties, monotonically_increasing=False):
        
        return bernstein_prediction(
                            self.params[:, params_index],
                            input[:, covar_num],
                            self.degree,
                            self.polynomial_range[:, covar_num],
                            monotonically_increasing=monotonically_increasing,
                            derivativ=derivativ,
                            span_factor=self.span_factor,
                            covariate=covariate,
                            params_covariate=self.params_covariate[:, covar_num])  
        
        
    def list_comprehension_forward(self, input, log_d, covariate, return_penalties=True): 
        

        return_dict = self.create_return_dict_decorrelation(input)

        # This is a list comprehension implementation of the foward pass that should be faster than the nested for loop
        def forward_pass_row(var_num, covar_num):

            num_gtm_splines = max(var_num * (var_num - 1) / 2,0)
            params_index = int(num_gtm_splines + covar_num)
            
            lambda_value, second_order_ridge_pen_current, \
                first_order_ridge_pen_current, param_ridge_pen_current = self.spline_prediction(input, 
                                                                                                params_index, 
                                                                                                covariate, 
                                                                                                covar_num, 
                                                                                                derivativ=0, 
                                                                                                return_penalties=True, 
                                                                                                monotonically_increasing=False)

            return input[:, covar_num] * lambda_value, \
                second_order_ridge_pen_current, first_order_ridge_pen_current, param_ridge_pen_current,\
                lambda_value

        def forward_pass_col(var_num):

            if var_num == 0:
                return torch.zeros(input.size()[0]), 0, 0, 0
            else:
                res = [forward_pass_row(var_num, covar_num) for covar_num in range(var_num)]

                add_to_output = sum(res[covar_num][0] for covar_num in range(var_num))

                second_order_ridge_pen_row_sum = sum(res[covar_num][1] for covar_num in range(var_num))
                first_order_ridge_pen_row_sum = sum(res[covar_num][2] for covar_num in range(var_num))
                param_ridge_pen_row_sum = sum(res[covar_num][3] for covar_num in range(var_num))

                lambda_matrix_entries = torch.cat([res[covar_num][4].unsqueeze(0) for covar_num in range(var_num)])

                return add_to_output, \
                        second_order_ridge_pen_row_sum, first_order_ridge_pen_row_sum, param_ridge_pen_row_sum, \
                    lambda_matrix_entries

        res = [forward_pass_col(var_num) for var_num in range(self.number_variables)]
        
        return_dict["output"] += torch.vstack([res[var_num][0].to(input.device) for var_num in range(self.number_variables)]).T

        return_dict["second_order_ridge_pen_sum"] = sum(res[var_num][1] for var_num in range(self.number_variables))
        return_dict["first_order_ridge_pen_sum"] = sum(res[var_num][2] for var_num in range(self.number_variables))
        return_dict["param_ridge_pen_sum"] = sum(res[var_num][3] for var_num in range(self.number_variables))

        for var_num in range(1,self.number_variables): #1 because the first row has no precision matrix entries
            return_dict["lambda_matrix"][:,var_num,0:var_num] = res[var_num][4].T

        return return_dict

        
    
    def vmap_forward(self, input, log_d, covariate, return_penalties=True): 
        
        return_dict = self.create_return_dict_decorrelation(input)
        
        self.var_num_list = self.var_num_list.to(input.device)
        self.covar_num_list = self.covar_num_list.to(input.device)
        self.knots = self.knots.to(input.device)    
    
        return_dict["lambda_matrix"][:, self.var_num_list, self.covar_num_list], return_dict["second_order_ridge_pen_sum"], return_dict["first_order_ridge_pen_sum"], return_dict["param_ridge_pen_sum"] = bspline_prediction_vectorized(self.params,
                           input.index_select(1,self.covar_num_list), 
                           self.knots,
                           self.degree, 
                           self.polynomial_range[:, 0], 
                           derivativ=0, 
                           return_penalties=return_penalties, 
                           calc_method=self.calc_method_bspline, 
                           span_factor=self.span_factor, 
                           span_restriction=self.span_restriction,
                           covariate=covariate, 
                           params_covariate=self.params_covariate, 
                           covaraite_effect=self.covaraite_effect,
                           penalize_towards=0, 
                           order=self.spline_order,
                           varying_degrees=False,
                           params_a_mask=None)

        return_dict["output"] = return_dict["output"].unsqueeze_(2)
        return_dict["output"] = torch.bmm(return_dict["lambda_matrix"], return_dict["output"]).squeeze(2)
         
        return return_dict
            
        
    def for_loop_forward(self, input, log_d, covariate, inverse=False, return_penalties=True, return_scores_hessian=False):
        
        return_dict = self.create_return_dict_decorrelation(input)
        
        params_index = 0
        
        if inverse:

            for var_num in range(self.number_variables):
    
                # loop over all before variables
                lambda_value_multiplier_total = torch.ones(input.size(0), device=input.device)
                lambda_value_total = torch.zeros(input.size(0), device=input.device)
                for covar_num in range(var_num):
                    
                    lambda_value = self.spline_prediction(input, 
                                                          params_index, 
                                                          covariate, 
                                                          covar_num, 
                                                          derivativ=0, 
                                                          return_penalties=False, 
                                                          monotonically_increasing=False)
                    
                    if self.params_multiplier is not False:
                        lambda_value_multiplier = self.spline_prediction(input, 
                                                          params_index, 
                                                          covariate, 
                                                          covar_num, 
                                                          derivativ=0, 
                                                          return_penalties=False, 
                                                          monotonically_increasing=False,
                                                          multi=True)

                    if self.params_multiplier is not False:
                        lambda_value_multiplier_total = lambda_value_multiplier_total * lambda_value_multiplier
                    lambda_value_total += lambda_value * input[:, covar_num]

                    params_index += 1

                    # filling the lambda matrix with the computed entries
                    return_dict["lambda_matrix"][:, var_num, covar_num] = lambda_value

                # update
                input[:, var_num] = (input[:, var_num] - lambda_value_total) / lambda_value_multiplier_total  
            
            # needs to be done at the end as we need to update input vector as well as it is the input to the subsequent computations (inverse iteratively)    
            return_dict["output"] = input 
        else:
            for var_num in range(self.number_variables):

                # loop over all before variables
                lambda_value_multiplier_total = torch.ones(input.size(0), device=input.device)
                lambda_value_total = torch.zeros(input.size(0), device=input.device)
                for covar_num in range(var_num):
                    
                    lambda_value, second_order_ridge_pen_current, \
                    first_order_ridge_pen_current, param_ridge_pen_current = self.spline_prediction(input, 
                                                          params_index, 
                                                          covariate, 
                                                          covar_num, 
                                                          derivativ=0, 
                                                          return_penalties=True, 
                                                          monotonically_increasing=False)
                    
                    return_dict["second_order_ridge_pen_sum"] += second_order_ridge_pen_current
                    return_dict["first_order_ridge_pen_sum"] += first_order_ridge_pen_current
                    return_dict["param_ridge_pen_sum"] += param_ridge_pen_current
                    
                    if self.params_multiplier is not False:
                            lambda_value_multiplier, second_order_ridge_pen_current, \
                            first_order_ridge_pen_current, param_ridge_pen_current = self.spline_prediction(input, 
                                                          params_index, 
                                                          covariate, 
                                                          covar_num, 
                                                          derivativ=0, 
                                                          return_penalties=True, 
                                                          monotonically_increasing=False,
                                                          multi=True)
                            
                            return_dict["second_order_ridge_pen_sum"] += second_order_ridge_pen_current
                            return_dict["first_order_ridge_pen_sum"] += first_order_ridge_pen_current
                            return_dict["param_ridge_pen_sum"] += param_ridge_pen_current
                            
                    if return_scores_hessian == True:
                        if self.params_multiplier is not False:
                            warnings.warn("Warning: return_scores_hessian not implemented for multiplicative effect. The der_lambda_matrix and der2_lambda_matrix will be wrong.")
            
                        der_lambda_value = self.spline_prediction(input, 
                                                          params_index, 
                                                          covariate, 
                                                          covar_num, 
                                                          derivativ=1, ######### <---------------------------- #########
                                                          return_penalties=True, 
                                                          monotonically_increasing=False,
                                                          multi=False)
                        
                        der2_lambda_value = self.spline_prediction(input, 
                                                          params_index, 
                                                          covariate, 
                                                          covar_num, 
                                                          derivativ=2, ######### <---------------------------- #########
                                                          return_penalties=True, 
                                                          monotonically_increasing=False,
                                                          multi=False)
                    else:
                        der_lambda_value = 0
                        der2_lambda_value = 0


                    if self.params_multiplier is not False:
                        lambda_value_multiplier_total = lambda_value_multiplier_total * lambda_value_multiplier
                    lambda_value_total += lambda_value * input[:, covar_num]

                    params_index += 1

                    # filling the lambda matrix with the computed entries
                    return_dict["lambda_matrix"][:, var_num, covar_num] = lambda_value

                    # filling the derivative lambda matrix with the derivative values ,see obsidean
                    return_dict["der_lambda_matrix"][:, var_num, covar_num] = der_lambda_value * input[:, covar_num] + lambda_value
                    return_dict["der2_lambda_matrix"][:, var_num, covar_num] = der_lambda_value + der2_lambda_value * input[:, covar_num] + der_lambda_value
                    
                # filling in the multiplicative effect into the lambda matrix
                return_dict["lambda_matrix"][:, var_num, var_num] = lambda_value_multiplier_total
                
                return_dict["output"][:, var_num] = lambda_value_multiplier_total * input[:, var_num] + lambda_value_total
                
                return_dict["log_d"][:, var_num] += torch.log(torch.abs(lambda_value_multiplier_total))

        return return_dict
  

    def forward(self, input, covariate=False, log_d = 0, inverse = False, return_log_d = False, return_penalties=False, return_scores_hessian=False):  
        
        if not inverse:
            if self.list_comprehension == True:
                if return_scores_hessian == True:
                        warnings.warn("Warning: return_scores_hessian not implemented for list comprehension. The der_lambda_matrix and der2_lambda_matrix will be zeros.")
                return_dict = self.list_comprehension_forward(input, log_d, covariate, return_penalties=return_penalties)
            elif self.vmap == True:
                return_dict = self.vmap_forward(input, log_d, covariate, return_penalties=return_penalties)
            else:
                return_dict = self.for_loop_forward(input, log_d, covariate, inverse=False, return_penalties=return_penalties, return_scores_hessian=return_scores_hessian)
        else:
            if self.list_comprehension == True:
                        warnings.warn("Warning: Inverse can only be computed iteratively using a for-loop. List comprehension is not implemented for inverse. Using for-loop instead. ")
            return_dict = self.for_loop_forward(input, log_d, covariate, inverse=inverse, return_penalties=return_penalties, return_scores_hessian=return_scores_hessian)
            
        return return_dict

    def __repr__(self):
        return f"Decorrelation(degree={self.degree}, number_variables={self.number_variables}, polynomial_range={self.polynomial_range}, spline={self.spline}, span_factor={self.span_factor}, span_restriction={self.span_restriction}, number_covariates={self.number_covariates}, list_comprehension={self.list_comprehension})"
    
    
    
    