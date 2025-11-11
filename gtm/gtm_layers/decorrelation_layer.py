import warnings

import numpy as np
# from functorch import vmap  # Requires PyTorch 1.10+
import torch
from torch import nn
import torch.nn.functional as F

from gtm.gtm_layers.layer_utils import BayesianInitializer, BayesianPriors
from gtm.gtm_splines.bernstein_prediction_vectorized import (
    bernstein_prediction_vectorized, binomial_coeffs)
from gtm.gtm_splines.bspline_prediction_vectorized import \
    bspline_prediction_vectorized


class Decorrelation(nn.Module):
    def __init__(
        self,
        degree,
        number_variables,
        spline_range,
        hyperparameter = None,
        spline="bspline",
        span_factor=torch.tensor(0.1),
        span_restriction="reluler",
        number_covariates=False,
        covariate_effect="multiplicativ",
        calc_method_bspline="deBoor",
        affine_layer=False,
        degree_multi=False,
        spline_order=3,
        inference = 'likelihood',
        device="cpu",
    ):
        super().__init__()
        self.type = "decorrelation"
        self.degree = degree
        self.number_variables = number_variables
        self.spline_range = torch.FloatTensor(spline_range)

        self.inference = inference
        self.device = device

        self.num_lambdas = number_variables * (number_variables - 1) / 2
        self.spline = spline
        if spline == "bernstein":
            warnings.warn(
                "Bernstein polynomial penalization is not implemented yet. only returns zeros hardcoded in bernstein_prediction.py fct"
            )
            n = self.degree + 1
            self.binom_n = binomial_coeffs(n, device=self.device)
            self.binom_n1 = binomial_coeffs(n - 1, device=self.device)
            self.binom_n2 = binomial_coeffs(n - 2, device=self.device)

        self.span_factor = span_factor

        self.span_restriction = span_restriction

        if self.spline == "bspline":
            self.spline_prediction = self.bspline_prediction_method
        elif self.spline == "bernstein":
            self.spline_prediction = self.bernstein_prediction_method
        else:
            warnings.warn(
                "Warning: Unknown Spline string passed, use bspline or bernstein instead. bspline is default."
            )

        self.spline_order = spline_order

        self.params = self.compute_starting_values_bspline(start_value=0.001)

        self.number_covariates = number_covariates

        if self.number_covariates is not False:
            if self.number_covariates > 1:
                print("Warning, covariates not implemented for more than 1 covariate")
            self.params_covariate = self.compute_starting_values_bspline(
                start_value=0.001
            )
        else:
            self.params_covariate = False

        # Defines wether we have an additive or an affine (multiplicative) coupling layer
        self.affine_layer = affine_layer
        self.degree_multi = degree_multi
        if self.affine_layer is not False:
            self.params_multiplier = self.compute_starting_values_bspline(
                start_value=1.0
            )
        else:
            self.params_multiplier = False

        self.covariate_effect = covariate_effect

        self.calc_method_bspline = calc_method_bspline

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
        
        
        if self.inference == 'bayesian':
            max_params = self.degree + self.spline_order - 1
            self.hyperparameter_decorrelation: dict[str, float] = hyperparameter
            priors: BayesianPriors = BayesianInitializer.build(
                model=self,
                hyperparameter=self.hyperparameter_decorrelation or {},
                n_params=max_params, 
                is_transformation=False
                )
            # Either store the whole dataclass…
            self.priors: BayesianPriors = priors


    def create_return_dict_decorrelation(self, input):
        lambda_matrix_general = (
            torch.eye(self.number_variables, device=input.device)
            .expand(input.size()[0], self.number_variables, self.number_variables)
            #.requires_grad_(True)
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
            ).to(self.device),
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
        B, V = input.size(0), self.number_variables
        device = input.device

        def add_lambda_ij(lm, i, j, val_b):  # lm: [B,V,V], val_b: [B]
            e_i = F.one_hot(torch.tensor(i, device=device), V).float().view(1, V, 1)
            e_j = F.one_hot(torch.tensor(j, device=device), V).float().view(1, 1, V)
            update = val_b.view(B, 1, 1) * (e_i * e_j)          # [B,V,V]
            return lm + update                                   # out-of-place

        def replace_col(mat, j, newcol_b):  # mat: [B,V], newcol_b: [B]
            e_j = F.one_hot(torch.tensor(j, device=device), V).float().view(1, V)
            delta = (newcol_b - mat[:, j]).view(B, 1)            # [B,1]
            return mat + delta * e_j                              # out-of-place

        if inverse:
            for var_num in range(V):
                lambda_mult_tot = torch.ones(B, device=device)
                lambda_add_tot  = torch.zeros(B, device=device)

                for covar_num in range(var_num):
                    lambda_val = self.spline_prediction(
                        input, params_index, covariate, covar_num,
                        derivativ=0, return_penalties=False, monotonically_increasing=False,
                    )
                    if lambda_val.ndim == 2 and lambda_val.size(1) == V:
                        lambda_val = lambda_val[:, covar_num]

                    if self.params_multiplier is not False:
                        lambda_mult = self.spline_prediction(
                            input, params_index, covariate, covar_num,
                            derivativ=0, return_penalties=False, monotonically_increasing=False, multi=True,
                        )
                        if lambda_mult.ndim == 2 and lambda_mult.size(1) == V:
                            lambda_mult = lambda_mult[:, covar_num]
                        lambda_mult_tot = lambda_mult_tot * lambda_mult  # out-of-place ok

                    lambda_add_tot = lambda_add_tot + lambda_val * input[:, covar_num]
                    params_index += 1

                    # return_dict["lambda_matrix"][:, var_num, covar_num] = lambda_val
                    return_dict["lambda_matrix"] = add_lambda_ij(return_dict["lambda_matrix"], var_num, covar_num, lambda_val)

                # input[:, var_num] = (input[:, var_num] - lambda_add_tot) / lambda_mult_tot
                newcol = (input[:, var_num] - lambda_add_tot) / lambda_mult_tot
                input = replace_col(input, var_num, newcol)

            return_dict["output"] = input
            return return_dict

        # ----- forward (not inverse) -----
        for var_num in range(V):
            lambda_mult_tot = torch.ones(B, device=device)
            lambda_add_tot  = torch.zeros(B, device=device)

            for covar_num in range(var_num):
                (lambda_val,
                pen2, pen1, pen0) = self.spline_prediction(
                    input, params_index, covariate, covar_num,
                    derivativ=0, return_penalties=True, monotonically_increasing=False,
                )
                if lambda_val.ndim == 2 and lambda_val.size(1) == V:
                    lambda_val = lambda_val[:, covar_num]

                return_dict["second_order_ridge_pen_sum"] += pen2
                return_dict["first_order_ridge_pen_sum"]  += pen1
                return_dict["param_ridge_pen_sum"]        += pen0

                if self.params_multiplier is not False:
                    (lambda_mult,
                    pen2m, pen1m, pen0m) = self.spline_prediction(
                        input, params_index, covariate, covar_num,
                        derivativ=0, return_penalties=True, monotonically_increasing=False, multi=True,
                    )
                    if lambda_mult.ndim == 2 and lambda_mult.size(1) == V:
                        lambda_mult = lambda_mult[:, covar_num]
                    return_dict["second_order_ridge_pen_sum"] += pen2m
                    return_dict["first_order_ridge_pen_sum"]  += pen1m
                    return_dict["param_ridge_pen_sum"]        += pen0m
                    lambda_mult_tot = lambda_mult_tot * lambda_mult  # out-of-place ok

                if return_scores_hessian is True and self.params_multiplier is not False:
                    warnings.warn(
                        "Warning: return_scores_hessian not implemented for multiplicative effect. "
                        "The der_lambda_matrix and der2_lambda_matrix will be wrong."
                    )
                if return_scores_hessian is True:
                    _ = self.spline_prediction(input, params_index, covariate, covar_num,
                                            derivativ=1, return_penalties=True, monotonically_increasing=False, multi=False)
                    _ = self.spline_prediction(input, params_index, covariate, covar_num,
                                            derivativ=2, return_penalties=True, monotonically_increasing=False, multi=False)

                lambda_add_tot = lambda_add_tot + lambda_val * input[:, covar_num]
                params_index += 1

                # return_dict["lambda_matrix"][:, var_num, covar_num] = lambda_val
                return_dict["lambda_matrix"] = add_lambda_ij(return_dict["lambda_matrix"], var_num, covar_num, lambda_val)

            # diag multiplier
            return_dict["lambda_matrix"] = add_lambda_ij(return_dict["lambda_matrix"], var_num, var_num, lambda_mult_tot)

            # output col update
            newcol = lambda_mult_tot * input[:, var_num] + lambda_add_tot
            return_dict["output"] = replace_col(return_dict["output"], var_num, newcol)

            # log|multiplier| col update
            incr = torch.log(torch.abs(lambda_mult_tot))
            return_dict["log_d"] = replace_col(return_dict["log_d"], var_num, return_dict["log_d"][:, var_num] + incr)

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
                input.to(self.device),
                log_d,
                covariate,
                inverse=inverse,
                return_penalties=return_penalties,
                return_scores_hessian=return_scores_hessian,
            )
        # elif self.vmap == True and self.number_variables > 2:
        else:
            return_dict = self.vmap_forward(
                input.to(self.device), log_d, covariate, return_penalties=return_penalties
            )

        return return_dict

    def __repr__(self):
        return f"Decorrelation(degree={self.degree}, number_variables={self.number_variables}, spline_range={self.spline_range}, spline={self.spline}, span_factor={self.span_factor}, span_restriction={self.span_restriction}, number_covariates={self.number_covariates}, vectorised_mapping={self.vmap})"