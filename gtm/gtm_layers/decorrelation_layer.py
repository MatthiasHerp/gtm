import warnings

import numpy as np
# from functorch import vmap  # Requires PyTorch 1.10+
import torch
from torch import nn, Tensor
from torch.distributions import InverseGamma
from typing import List,Literal

from gtm.gtm_splines.bernstein_prediction_vectorized import (
    bernstein_prediction_vectorized, binomial_coeffs)
from gtm.gtm_splines.bspline_prediction_vectorized import \
    bspline_prediction_vectorized


class Decorrelation(nn.Module):
    def __init__(
        self,
        degree: int,
        number_variables: int,
        spline_range: List[List[float]],
        #hyperparameters:dict,
        #diff_order:int= 2,
        spline: Literal['bspline'] | Literal['bernstein'] ="bspline",
        span_factor: Tensor =torch.tensor(0.1),
        span_restriction: Literal['reluler'] ="reluler",
        number_covariates: bool =False,
        covariate_effect: str ="multiplicativ",
        calc_method_bspline="deBoor",
        affine_layer=False,
        degree_multi=False,
        spline_order=3,
        inference="frequentist",
        device="cpu",
    ) -> None:
        super().__init__()
        self.type = "decorrelation"
        self.degree = degree
        self.number_variables = number_variables
        self.spline_range = torch.FloatTensor(spline_range)
        self.inference = inference
        self.hyperparameters = {
                        'alpha_sigma': 1.0,
                        'beta_sigma': 0.005,
                        'alpha_tau': 1.0,
                        'beta_tau': 0.005
                        } #hyperparameters
        self.device = device

        self.num_lambdas = number_variables * (number_variables - 1) / 2
        self.spline: Literal['bspline'] | Literal['bernstein'] = spline
        if spline == "bernstein":
            warnings.warn(
                "Bernstein polynomial penalization is not implemented yet. only returns zeros hardcoded in bernstein_prediction.py fct"
            )
            n = self.degree + 1
            self.binom_n = binomial_coeffs(n, device=self.device)
            self.binom_n1 = binomial_coeffs(n - 1, device=self.device)
            self.binom_n2 = binomial_coeffs(n - 2, device=self.device)

        self.span_factor: Tensor = span_factor

        self.span_restriction = span_restriction

        if self.spline == "bspline":
            self.spline_prediction = self.bspline_prediction_method
        elif self.spline == "bernstein":
            self.spline_prediction = self.bernstein_prediction_method
        else:
            warnings.warn(
                message="Warning: Unknown Spline string passed, use bspline or bernstein instead. bspline is default."
            )


        self.spline_order: int = spline_order

        self.params: torch.Tensor = self.compute_starting_values_bspline(start_value=0.001)

        self.number_covariates: bool = number_covariates


        if self.number_covariates is not False:
            if self.number_covariates > 1:
                print("Warning, covariates not implemented for more than 1 covariate")
            self.params_covariate: torch.Tensor = self.compute_starting_values_bspline(
                start_value=0.001
            )
        else:
            self.params_covariate = False

        # Defines wether we have an additive or an affine (multiplicative) coupling layer
        self.affine_layer: bool = affine_layer
        self.degree_multi: bool = degree_multi
        if self.affine_layer is not False:
            self.params_multiplier: torch.Tensor = self.compute_starting_values_bspline(
                start_value=1.0
            )
        else:
            self.params_multiplier = False

        self.covariate_effect: str = covariate_effect

        if self.inference == 'bayesian':
            ## Hierarchical Bayes
            # Error of Regression
            self.alpha_sigma = torch.tensor([self.hyperparameters['alpha_sigma']])
            self.beta_sigma = torch.tensor([self.hyperparameters['beta_sigma']])
            # Random Walk Error
            self.alpha_tau = torch.tensor([self.hyperparameters['alpha_tau']])
            self.beta_tau= torch.tensor([self.hyperparameters['beta_tau']])
            
            
            ## Defining HyperParameters
            self.tau2_prior = InverseGamma(self.alpha_tau,self.beta_tau)
            self.sigma2_prior = InverseGamma(self.alpha_sigma,self.beta_sigma)

            self.order_prior_diff: int = 2#diff_order # Standard 2 order random walk.... #TODO Generalize

            self.K_prior = torch.tensor(
                data=self._difference_penalty_matrix(
                    order=self.order_prior_diff,
                    n_params=self.params.shape[0]
                    ),
                dtype=torch.float32,
                device=self.device
                )
            
            self.prior_epsilon = torch.exp(-self.beta_sigma/self.sigma2_prior)/torch.pow(self.sigma2_prior, self.alpha_sigma + 1)
            self.prior_rw = torch.exp(-self.beta_tau/self.tau2_prior)/torch.pow(self.tau2_prior, self.alpha_tau + 1)
            self.tau_given_gamma = torch.log(torch.exp(-self.K_prior/(2*self.tau2_prior, 2))/self.tau2_prior) #TODO Generalize with RANK

        self.calc_method_bspline: str = calc_method_bspline

        # TODO: Sort this out, I fixed the values here as it is currently the fastest computation by far
        self.vmap = True  # False #True
        # self.calc_method_bspline = "deBoor" #"deBoor" #"Naive"

        ### Old
        # The following code ensures that:
        # we have knots equally spanning the range of the number of degrees
        # we have the first an last knot on the bound of the span
        # we get equally space boundary knots outside the spane to ensure that at the boundary of the span we get bspline predicitons without errors

        # The distance between knots is the span divided by the number of knots minus 1
        # because between D points where we have one point at the min and one at the max of a span we have D-1 intervals between knots
        distance_between_knots_in_bounds = (
            self.spline_range[1, 0] - self.spline_range[0, 0]
        ) / (self.degree - 1)

        number_of_bound_knots_per_side = (
            self.spline_order
        )  # for cubcic splines (order 3) we need 2 knots on each side

        # We get the eqully spaced knots by equal spacing on the extended span
        self.knots = torch.linspace(
            self.spline_range[0, 0]
            - number_of_bound_knots_per_side * distance_between_knots_in_bounds,
            self.spline_range[1, 0]
            + number_of_bound_knots_per_side * distance_between_knots_in_bounds,
            self.degree + 2 * number_of_bound_knots_per_side,  # 2* because of two sides
            dtype=torch.float32,
        )

        self.knots = self.knots.to(self.device)

        ##### Update
        # if self.spline_order == 2:
        #    n = self.degree + 1
        # elif self.spline_order == 3:
        #    n = self.degree + 2
        #
        # distance_between_knots = (self.spline_range[1,0] - self.spline_range[0,0]) * (1 + self.span_factor) / (n - 1)
        #
        #
        # self.knots = torch.linspace(self.spline_range[0,0] * (1 + self.span_factor) - self.spline_order * distance_between_knots,
        #                    self.spline_range[1,0] * (1 + self.span_factor) + self.spline_order * distance_between_knots,
        #                    n + 4, dtype=torch.float32)

        self.var_num_list, self.covar_num_list = torch.tril_indices(
            self.number_variables, self.number_variables, offset=-1
        )
        
    ####### BAYESIAN DECOLAYER ###########
        
    def _difference_penalty_matrix(self, order, n_params):
        D = np.eye(n_params)
        for _ in range(order):
            D = np.diff(D, n=1, axis=0)
        return D.T @ D # K = DᵀD

    def log_prior(self):

        K = self.K_prior
        sigma2 = torch.exp(self.log_tau2_prior)
        log_prior = 0.0
        for j in range(self.params.shape[1]):
            theta_j = self.params[:, j]
            quad_form = theta_j @ K @ theta_j
            log_prior += -0.5 * quad_form / sigma2

        log_det_term = -0.5 * self.params.shape[1] * K.shape[0] * torch.log(sigma2)
        return log_prior + log_det_term
    
    ####### BAYESIAN DECORELATION LAYER ###########

    def create_return_dict_decorrelation(self, input):
        lambda_matrix_general = (
            torch.eye(self.number_variables, device=input.device)
            .expand(input.size()[0], self.number_variables, self.number_variables)
            .requires_grad_(True)
        )
        return {
            "output": input.clone(),  # .requires_grad_(True),
            "log_d": torch.zeros(
                input.size(), device=self.device
            ),  # .requires_grad_(True),
            "transformation_second_order_ridge_pen_sum": 0,
            "second_order_ridge_pen_sum": 0,
            "first_order_ridge_pen_sum": 0,
            "param_ridge_pen_sum": 0,
            "lambda_matrix": lambda_matrix_general.clone(),  # ,
            # "der_lambda_matrix": lambda_matrix_general.clone(),
            # "der2_lambda_matrix": lambda_matrix_general.clone()
        }

    def compute_starting_values_bspline(self, start_value=0.001):
        p = torch.FloatTensor(
            np.repeat(
                np.repeat(start_value, self.degree + self.spline_order - 1),
                self.num_lambdas,
            )
        )

        if self.num_lambdas == 1:
            params = nn.Parameter(p.unsqueeze(1))
        else:
            params = nn.Parameter(
                torch.reshape(
                    p, (self.degree + self.spline_order - 1, int(self.num_lambdas))
                )
            )

        return params

    def bspline_prediction_method(
        self,
        input,
        params_index,
        covariate,
        covar_num,
        derivativ,
        return_penalties,
        monotonically_increasing=False,
        multi=False,
    ):

        # return bspline_prediction(
        #                    self.params[:, params_index] if multi==False else self.params_multiplier[:, params_index],
        #                    input[:, covar_num],
        #                    self.degree if multi==False else self.degree_multi,
        #                    self.spline_range[:, 0],
        #                    monotonically_increasing=monotonically_increasing,
        #                    derivativ=derivativ,
        #                    return_penalties=return_penalties,
        #                    span_factor=self.span_factor,
        #                    span_restriction=self.span_restriction,
        #                    covariate=covariate,
        #                    params_covariate=False, #self.params_covariate[:, covar_num],
        #                    covariate_effect=self.covariate_effect,
        #                    calc_method=self.calc_method_bspline,
        #                    order = self.spline_order)

        return bspline_prediction_vectorized(
            (
                self.params[:, params_index].unsqueeze(1)
                if multi == False
                else self.params_multiplier[:, params_index].unsqueeze(1)
            ),
            input[:, covar_num].unsqueeze(1),
            self.knots,
            self.degree,
            self.spline_range[:, 0],
            derivativ=derivativ,
            return_penalties=return_penalties,
            calc_method=self.calc_method_bspline,
            span_factor=self.span_factor,
            span_restriction=self.span_restriction,
            covariate=covariate,
            params_covariate=self.params_covariate,
            covariate_effect=self.covariate_effect,
            penalize_towards=0,
            order=self.spline_order,
            varying_degrees=False,
            params_a_mask=None,
        )

    def bernstein_prediction_method(
        self,
        input,
        params_index,
        covariate,
        covar_num,
        derivativ,
        return_penalties,
        monotonically_increasing=False,
        multi=False,
    ):

        return bernstein_prediction_vectorized(
            params_a=(
                self.params[:, params_index].unsqueeze(1)
                if multi == False
                else self.params_multiplier[:, params_index].unsqueeze(1)
            ),
            input_a=input[:, covar_num].unsqueeze(1),
            # knots=self.knots,
            degree=self.degree,
            spline_range=self.spline_range[:, covar_num],
            monotonically_increasing=monotonically_increasing,
            derivativ=derivativ,
            span_factor=self.span_factor,
            covariate=covariate,
            params_covariate=False,
            binom_n=self.binom_n,
            binom_n1=self.binom_n1,
            binom_n2=self.binom_n2,
            return_penalties=return_penalties,
        )  # self.params_covariate[:, covar_num])

    def vmap_forward(self, input, log_d, covariate, return_penalties=True):

        return_dict = self.create_return_dict_decorrelation(input)

        self.var_num_list = self.var_num_list.to(input.device)
        self.covar_num_list = self.covar_num_list.to(input.device)
        self.knots = self.knots.to(input.device)

        if self.spline == "bspline":
            (
                return_dict["lambda_matrix"][:, self.var_num_list, self.covar_num_list],
                return_dict["second_order_ridge_pen_sum"],
                return_dict["first_order_ridge_pen_sum"],
                return_dict["param_ridge_pen_sum"],
            ) = bspline_prediction_vectorized(
                self.params,
                input.index_select(1, self.covar_num_list),
                self.knots,
                self.degree,
                self.spline_range[:, 0],
                derivativ=0,
                return_penalties=return_penalties,
                calc_method=self.calc_method_bspline,
                span_factor=self.span_factor,
                span_restriction=self.span_restriction,
                covariate=covariate,
                params_covariate=self.params_covariate,
                covariate_effect=self.covariate_effect,
                penalize_towards=0,
                order=self.spline_order,
                varying_degrees=False,
                params_a_mask=None,
            )
        elif self.spline == "bernstein":
            # note that it also returns the log determinant
            (
                return_dict["lambda_matrix"][:, self.var_num_list, self.covar_num_list],
                return_dict["second_order_ridge_pen_sum"],
                return_dict["first_order_ridge_pen_sum"],
                return_dict["param_ridge_pen_sum"],
            ) = bernstein_prediction_vectorized(
                self.params,
                input.index_select(1, self.covar_num_list),
                # self.knots,
                self.degree,
                self.spline_range[:, 0],
                derivativ=0,
                return_penalties=return_penalties,
                calc_method=self.calc_method_bspline,
                span_factor=self.span_factor,
                span_restriction=self.span_restriction,
                covariate=covariate,
                params_covariate=self.params_covariate,
                covariate_effect=self.covariate_effect,
                penalize_towards=0,
                order=self.spline_order,
                varying_degrees=False,
                params_a_mask=None,
                binom_n=self.binom_n,
                binom_n1=self.binom_n1,
                binom_n2=self.binom_n2,
            )

        return_dict["output"] = return_dict["output"].unsqueeze_(2)
        return_dict["output"] = torch.bmm(
            return_dict["lambda_matrix"], return_dict["output"]
        ).squeeze(2)

        return return_dict

    def for_loop_forward(
        self,
        input,
        log_d,
        covariate,
        inverse=False,
        return_penalties=True,
        return_scores_hessian=False,
    ):

        return_dict = self.create_return_dict_decorrelation(input)

        params_index = 0

        if inverse:

            for var_num in range(self.number_variables):

                # loop over all before variables
                lambda_value_multiplier_total = torch.ones(
                    input.size(0), device=input.device
                )
                lambda_value_total = torch.zeros(input.size(0), device=input.device)
                for covar_num in range(var_num):

                    lambda_value = self.spline_prediction(
                        input,
                        params_index,
                        covariate,
                        covar_num,
                        derivativ=0,
                        return_penalties=False,
                        monotonically_increasing=False,
                    )

                    if self.params_multiplier is not False:
                        lambda_value_multiplier = self.spline_prediction(
                            input,
                            params_index,
                            covariate,
                            covar_num,
                            derivativ=0,
                            return_penalties=False,
                            monotonically_increasing=False,
                            multi=True,
                        )

                    if self.params_multiplier is not False:
                        lambda_value_multiplier_total = (
                            lambda_value_multiplier_total * lambda_value_multiplier
                        )
                    lambda_value_total += lambda_value * input[:, covar_num]

                    params_index += 1

                    # filling the lambda matrix with the computed entries
                    return_dict["lambda_matrix"][:, var_num, covar_num] = lambda_value

                # update
                input[:, var_num] = (
                    input[:, var_num] - lambda_value_total
                ) / lambda_value_multiplier_total

            # needs to be done at the end as we need to update input vector as well as it is the input to the subsequent computations (inverse iteratively)
            return_dict["output"] = input
        else:
            for var_num in range(self.number_variables):

                # loop over all before variables
                lambda_value_multiplier_total = torch.ones(
                    input.size(0), device=input.device
                )
                lambda_value_total = torch.zeros(input.size(0), device=input.device)
                for covar_num in range(var_num):

                    (
                        lambda_value,
                        second_order_ridge_pen_current,
                        first_order_ridge_pen_current,
                        param_ridge_pen_current,
                    ) = self.spline_prediction(
                        input,
                        params_index,
                        covariate,
                        covar_num,
                        derivativ=0,
                        return_penalties=True,
                        monotonically_increasing=False,
                    )

                    return_dict[
                        "second_order_ridge_pen_sum"
                    ] += second_order_ridge_pen_current
                    return_dict[
                        "first_order_ridge_pen_sum"
                    ] += first_order_ridge_pen_current
                    return_dict["param_ridge_pen_sum"] += param_ridge_pen_current

                    if self.params_multiplier is not False:
                        (
                            lambda_value_multiplier,
                            second_order_ridge_pen_current,
                            first_order_ridge_pen_current,
                            param_ridge_pen_current,
                        ) = self.spline_prediction(
                            input,
                            params_index,
                            covariate,
                            covar_num,
                            derivativ=0,
                            return_penalties=True,
                            monotonically_increasing=False,
                            multi=True,
                        )

                        return_dict[
                            "second_order_ridge_pen_sum"
                        ] += second_order_ridge_pen_current
                        return_dict[
                            "first_order_ridge_pen_sum"
                        ] += first_order_ridge_pen_current
                        return_dict["param_ridge_pen_sum"] += param_ridge_pen_current

                    if return_scores_hessian == True:
                        if self.params_multiplier is not False:
                            warnings.warn(
                                "Warning: return_scores_hessian not implemented for multiplicative effect. The der_lambda_matrix and der2_lambda_matrix will be wrong."
                            )

                        der_lambda_value = self.spline_prediction(
                            input,
                            params_index,
                            covariate,
                            covar_num,
                            derivativ=1,  ######### <---------------------------- #########
                            return_penalties=True,
                            monotonically_increasing=False,
                            multi=False,
                        )

                        der2_lambda_value = self.spline_prediction(
                            input,
                            params_index,
                            covariate,
                            covar_num,
                            derivativ=2,  ######### <---------------------------- #########
                            return_penalties=True,
                            monotonically_increasing=False,
                            multi=False,
                        )
                    else:
                        der_lambda_value = 0
                        der2_lambda_value = 0

                    if self.params_multiplier is not False:
                        lambda_value_multiplier_total = (
                            lambda_value_multiplier_total * lambda_value_multiplier
                        )
                    lambda_value_total += lambda_value * input[:, covar_num]

                    params_index += 1

                    # filling the lambda matrix with the computed entries
                    return_dict["lambda_matrix"][:, var_num, covar_num] = lambda_value

                    # filling the derivative lambda matrix with the derivative values ,see obsidean
                    # return_dict["der_lambda_matrix"][:, var_num, covar_num] = der_lambda_value * input[:, covar_num] + lambda_value
                    # return_dict["der2_lambda_matrix"][:, var_num, covar_num] = der_lambda_value + der2_lambda_value * input[:, covar_num] + der_lambda_value

                # filling in the multiplicative effect into the lambda matrix
                return_dict["lambda_matrix"][
                    :, var_num, var_num
                ] = lambda_value_multiplier_total

                return_dict["output"][:, var_num] = (
                    lambda_value_multiplier_total * input[:, var_num]
                    + lambda_value_total
                )

                return_dict["log_d"][:, var_num] += torch.log(
                    torch.abs(lambda_value_multiplier_total)
                )

        return return_dict

    def forward(
        self,
        input,
        covariate=False,
        log_d=0,
        inverse=False,
        return_log_d=False,
        return_penalties=False,
        return_scores_hessian=False,
    ):

        if inverse == True or self.number_variables == 2 or self.vmap == False:
            return_dict = self.for_loop_forward(
                input,
                log_d,
                covariate,
                inverse=inverse,
                return_penalties=return_penalties,
                return_scores_hessian=return_scores_hessian,
            )
        # elif self.vmap == True and self.number_variables > 2:
        else:
            return_dict = self.vmap_forward(
                input, log_d, covariate, return_penalties=return_penalties
            )

        return return_dict

    def __repr__(self):
        return f"Decorrelation(degree={self.degree}, number_variables={self.number_variables}, spline_range={self.spline_range}, spline={self.spline}, span_factor={self.span_factor}, span_restriction={self.span_restriction}, number_covariates={self.number_covariates}, vectorised_mapping={self.vmap})"
