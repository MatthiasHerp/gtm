import os

import numpy as np
# from run_simulation_study import run_simulation_study, log_mlflow_plot
import pandas as pd
import torch

from gtm.gtm_plots_analysis.nd_quad import (  # https://github.com/harisankar95/nd-quadrature
    getQuad, integrateQuad, transformQuad)


##################################################################################################################################################################
def compute_ci_probability_deviance_31_glq(
    model, y_subset, min_val=-5, max_val=5, n=20
):

    # compute p1 = log(f(Y_1,Y_3,Y_2)) from the model
    p1_glq = model.log_likelihood(y_subset,return_lambda_matrix=False)

    # compute p2 = log(f(Y_2))
    # First define montecarlo integration over Y_1, Y_3
    def list_comprehension_fct_Y_1_Y_3(
        model, Y_1_min, Y_1_max, Y_2, Y_3_min, Y_3_max, n
    ):
        def f(Y_3_samples, Y_1_samples):
            Y_3_samples = torch.FloatTensor(Y_3_samples)
            Y_1_samples = torch.FloatTensor(Y_1_samples)
            Y_synth = torch.vstack(
                [Y_1_samples, Y_2.repeat(Y_3_samples.size(0)), Y_3_samples]
            ).T
            return torch.exp(model.log_likelihood(Y_synth,return_lambda_matrix=False)).detach().numpy()

        # Define limits
        limits = [[Y_1_min, Y_1_max], [Y_3_min, Y_3_max]]

        # Integrate function
        approx_integral = integrateQuad(f=f, limits=limits, npoints=n)

        return torch.FloatTensor([approx_integral])

    p2_glq = torch.hstack(
        [
            list_comprehension_fct_Y_1_Y_3(
                model=model,
                Y_2=y_subset[i, 1],
                Y_1_min=min_val,
                Y_1_max=max_val,
                Y_3_min=min_val,
                Y_3_max=max_val,
                n=n**2,
            )
            for i in range(y_subset.size(0))
        ]
    )

    p2_glq = torch.log(p2_glq)

    # compute p3 = log(f(Y_1,Y_3|Y_2)) = log(f(Y_1,Y_3,Y_2)) - log(f(Y_2))
    p3_glq = p1_glq - p2_glq

    # compute p4 = log(f(Y_1|Y_2)) = log(f(Y_1,Y_2)) - log(f(Y_2))
    # First define montecarlo integration over Y_3
    def list_comprehension_fct(model, Y_1, Y_2, Y_3_min, Y_3_max, n):
        # Example usage:
        # Define the function to integrate
        def f(Y_3_samples):
            Y_3_samples = torch.FloatTensor(Y_3_samples)
            Y_synth = torch.vstack(
                [
                    Y_1.repeat(Y_3_samples.size(0)),
                    Y_2.repeat(Y_3_samples.size(0)),
                    Y_3_samples,
                ]
            ).T
            return torch.exp(model.log_likelihood(Y_synth,return_lambda_matrix=False)).detach().numpy()

        # Define limits
        limits = [[Y_3_min, Y_3_max]]

        # Integrate function
        approx_integral = integrateQuad(f=f, limits=limits, npoints=n)

        return torch.FloatTensor([approx_integral])

    p4_glq = torch.hstack(
        [
            list_comprehension_fct(
                model=model,
                Y_1=y_subset[i, 0],
                Y_2=y_subset[i, 1],
                Y_3_min=min_val,
                Y_3_max=max_val,
                n=20,
            )
            for i in range(y_subset.size(0))
        ]
    )

    p4_glq = torch.log(p4_glq)

    p4_glq = p4_glq - p2_glq

    # compute p5 = log(f(Y_3|Y_2)) = log(f(Y_3,Y_2)) - log(f(Y_2))
    # First define montecarlo integration over Y_1
    def list_comprehension_fct(model, Y_1_min, Y_1_max, Y_2, Y_3, n):
        # Example usage:
        # Define the function to integrate
        def f(Y_1_samples):
            Y_1_samples = torch.FloatTensor(Y_1_samples)
            Y_synth = torch.vstack(
                [
                    Y_1_samples,
                    Y_2.repeat(Y_1_samples.size(0)),
                    Y_3.repeat(Y_1_samples.size(0)),
                ]
            ).T
            return torch.exp(model.log_likelihood(Y_synth,return_lambda_matrix=False)).detach().numpy()

        # Define limits
        limits = [[Y_1_min, Y_1_max]]

        # Integrate function
        approx_integral = integrateQuad(f=f, limits=limits, npoints=n)

        return torch.FloatTensor([approx_integral])

    p5_glq = torch.hstack(
        [
            list_comprehension_fct(
                model=model,
                Y_3=y_subset[i, 2],
                Y_2=y_subset[i, 1],
                Y_1_min=min_val,
                Y_1_max=max_val,
                n=20,
            )
            for i in range(y_subset.size(0))
        ]
    )

    p5_glq = torch.log(p5_glq) - p2_glq

    actual_log_distribution = p3_glq
    under_ci_assumption_log_distribution = p4_glq + p5_glq

    return actual_log_distribution, under_ci_assumption_log_distribution


def compute_ci_probability_deviance_21(model, y_subset, min_val=-5, max_val=5):
    # compute p1 = log(f(Y_1,Y_3,Y_2)) from the model
    p1_mc = model.log_likelihood(y_subset,return_lambda_matrix=False)

    # First define montecarlo integration over Y_1, Y_2
    def monte_carlo_integration_over_Y_1_Y_2(
        model, Y_1_min, Y_1_max, Y_2_min, Y_2_max, Y_3, samples=10000
    ):
        Y_2_samples = torch.FloatTensor(1, samples).uniform_(Y_2_min, Y_2_max)
        Y_1_samples = torch.FloatTensor(1, samples).uniform_(Y_1_min, Y_1_max)

        Y_synth = torch.vstack([Y_1_samples, Y_2_samples, Y_3.repeat(samples)]).T

        likelihoods = torch.exp(model.log_likelihood(Y_synth,return_lambda_matrix=False,))
        mean_likelihood = likelihoods.mean()

        volume = (Y_2_max - Y_2_min) * (Y_1_max - Y_1_min)
        approx_integral = mean_likelihood * volume

        return approx_integral

    p2_mc_test = torch.hstack(
        [
            monte_carlo_integration_over_Y_1_Y_2(
                model=model,
                Y_1_min=min_val,
                Y_1_max=max_val,
                Y_2_min=-4,
                Y_2_max=4,
                Y_3=y_subset[i, 2],
                samples=100000,
            )
            for i in range(y_subset.size(0))
        ]
    )

    p2_mc_test = torch.log(p2_mc_test)

    # compute p3
    p3_mc_test = p1_mc - p2_mc_test

    # compute p4 = log(f(Y_1|Y_3)) = log(f(Y_1,Y_3)) - log(f(Y_3))
    # First define montecarlo integration over Y_3
    def monte_carlo_integration_over_Y_2(
        model, Y_1, Y_2_min, Y_2_max, Y_3, samples=10000
    ):
        Y_2_samples = torch.FloatTensor(1, samples).uniform_(Y_2_min, Y_2_max)

        Y_synth = torch.vstack(
            [Y_1.repeat(samples), Y_2_samples, Y_3.repeat(samples)]
        ).T

        # we are computing f(Y_1,Y_2) so we need to divide by f(Y_2) to get the conditional

        likelihoods = torch.exp(model.log_likelihood(Y_synth,return_lambda_matrix=False))
        mean_likelihood = likelihoods.mean()

        approx_integral = mean_likelihood * (Y_2_max - Y_2_min)

        return approx_integral

    p4_mc_test = torch.hstack(
        [
            monte_carlo_integration_over_Y_2(
                model=model,
                Y_1=y_subset[i, 0],
                Y_2_min=min_val,
                Y_2_max=max_val,
                Y_3=y_subset[i, 2],
                samples=50000,
            )
            for i in range(y_subset.size(0))
        ]
    )

    p4_mc_test = torch.log(p4_mc_test) - p2_mc_test

    # compute p4 = log(f(Y_2|Y_3)) = log(f(Y_2,Y_3)) - log(f(Y_3))
    # First define montecarlo integration over Y_3
    def monte_carlo_integration_over_Y_1(
        model, Y_1_min, Y_1_max, Y_2, Y_3, samples=10000
    ):
        Y_1_samples = torch.FloatTensor(1, samples).uniform_(Y_1_min, Y_1_max)

        Y_synth = torch.vstack(
            [Y_1_samples, Y_2.repeat(samples), Y_3.repeat(samples)]
        ).T

        # we are computing f(Y_1,Y_2) so we need to divide by f(Y_2) to get the conditional

        likelihoods = torch.exp(model.log_likelihood(Y_synth,return_lambda_matrix=False))
        mean_likelihood = likelihoods.mean()

        approx_integral = mean_likelihood * (Y_1_max - Y_1_min)

        return approx_integral

    p5_mc_test = torch.hstack(
        [
            monte_carlo_integration_over_Y_1(
                model=model,
                Y_1_min=min_val,
                Y_1_max=max_val,
                Y_2=y_subset[i, 1],
                Y_3=y_subset[i, 2],
                samples=50000,
            )
            for i in range(y_subset.size(0))
        ]
    )

    p5_mc_test = torch.log(p5_mc_test) - p2_mc_test

    actual_log_distribution = p3_mc_test
    under_ci_assumption_log_distribution = p4_mc_test + p5_mc_test

    return actual_log_distribution, under_ci_assumption_log_distribution


##################################################################################################################################################################
# Generalizing the Functions


def list_comprehension_integrate_out_one_variable(
    model,
    data,
    num_col_to_integrate_out,
    min_val,
    max_val,
    num_points_quad,
    copula=False,
):
    # num_points_quad is the number of points per variable

    # Define the function to integrate
    def f(sample):
        sample = torch.FloatTensor(sample)
        data_synth = data.clone().repeat(sample.size(0), 1)
        data_synth[:, num_col_to_integrate_out] = sample

        ll = model.log_likelihood(data_synth,return_lambda_matrix=False).detach()
        if copula == True:
            marginal_ll = torch.distributions.Normal(0, 1).log_prob(data_synth).sum(1)
        else:
            marginal_ll = 0
        ll = torch.exp(ll - marginal_ll)

        if ll.isnan().any():
            print("ll:", ll)
            print("data_synth:", data_synth)
        if ll.isinf().any():
            print("ll:", ll)
            print("data_synth:", data_synth)

        return ll.numpy()
        # return model.log_likelihood(data_synth,return_lambda_matrix=False).detach().numpy()

    # Define limits
    limits = [[min_val, max_val]]

    # Integrate function
    approx_integral = integrateQuad(f=f, limits=limits, npoints=num_points_quad)

    return torch.log(torch.FloatTensor([approx_integral]))


def list_comprehension_integrate_out_two_variables(
    model,
    data,
    num_col_to_integrate_out_1,
    num_col_to_integrate_out_2,
    min_val,
    max_val,
    num_points_quad,
    copula=False,
):
    # Example usage:
    # Define the function to integrate
    def f(sample_1, sample_2):
        sample_1 = torch.FloatTensor(sample_1)
        sample_2 = torch.FloatTensor(sample_2)
        data_synth = data.clone().repeat(sample_1.size(0), 1)
        data_synth[:, num_col_to_integrate_out_1] = sample_1
        data_synth[:, num_col_to_integrate_out_2] = sample_2

        ll = model.log_likelihood(data_synth,return_lambda_matrix=False).detach()
        if copula == True:
            marginal_ll = torch.distributions.Normal(0, 1).log_prob(data_synth).sum(1)
        else:
            marginal_ll = 0
        ll = torch.exp(ll - marginal_ll)

        if ll.isnan().any():
            print("ll:", ll)
            print("data_synth:", data_synth)
        if ll.isinf().any():
            print("ll:", ll)
            print("data_synth:", data_synth)

        return ll.numpy()
        # return torch.exp(model.log_likelihood(data_synth,return_lambda_matrix=False)).detach().numpy()
        # return model.log_likelihood(data_synth,return_lambda_matrix=False).detach().numpy()

    # Define limits
    limits = [[min_val, max_val], [min_val, max_val]]

    # Integrate function
    approx_integral = integrateQuad(f=f, limits=limits, npoints=num_points_quad)

    return torch.log(torch.FloatTensor([approx_integral]))


def list_comprehension_integrate_out_one_variable_optimized(
    model,
    data,
    num_col_to_integrate_out,
    min_val,
    max_val,
    num_points_quad,
    copula=False,
):
    limits = [[min_val, max_val]]

    # Generate quadrature points and weights
    x, w = getQuad(num_points_quad, ndim=len(limits))
    # Transform quadrature points and weights
    x, w = transformQuad(x, w, limits)
    # Evaluate function at quadrature points
    # y = f(*x.T)
    # Calculate result
    # res = np.sum(w * y)

    X = torch.FloatTensor(x).to(model.device)
    W = torch.FloatTensor(w).to(model.device)
    N = data.size(0)
    D = data.size(1)

    repeated_X = X.repeat(N, 1)
    repeated_W = W.repeat(N)

    repeated_data_rows = data.unsqueeze(1).repeat(1, len(X), 1)
    expanded_data = repeated_data_rows.view(-1, D)

    expanded_data[:, num_col_to_integrate_out] = repeated_X[:, 0]

    l_expanded_data = torch.exp(model.log_likelihood(expanded_data,return_lambda_matrix=False))

    # if l_expanded_data.isnan().any():
    #        print("ll:",l_expanded_data.isnan().sum() / l_expanded_data.size(0) )
    # if l_expanded_data.isinf().any():
    #        print("ll:",l_expanded_data.isinf().sum() / l_expanded_data.size(0) )

    l_expanded_data_weighted = repeated_W * l_expanded_data

    l_expanded_data_weighted_reshaped = l_expanded_data_weighted.view(-1, len(X))
    approx_integral = l_expanded_data_weighted_reshaped.sum(dim=1)

    return torch.log(approx_integral)


def list_comprehension_integrate_out_two_variables_optimized(
    model,
    data,
    num_col_to_integrate_out_1,
    num_col_to_integrate_out_2,
    min_val,
    max_val,
    num_points_quad,
    copula=False,
):
    limits = [[min_val, max_val], [min_val, max_val]]

    # Generate quadrature points and weights
    x, w = getQuad(num_points_quad, ndim=len(limits))
    # Transform quadrature points and weights
    x, w = transformQuad(x, w, limits)
    # Evaluate function at quadrature points
    # y = f(*x.T)
    # Calculate result
    # res = np.sum(w * y)

    X = torch.FloatTensor(x).to(model.device)
    W = torch.FloatTensor(w).to(model.device)
    N = data.size(0)
    D = data.size(1)

    repeated_X = X.repeat(N, 1)
    repeated_W = W.repeat(N)

    repeated_data_rows = data.unsqueeze(1).repeat(1, len(X), 1)
    expanded_data = repeated_data_rows.view(-1, D)

    expanded_data[:, num_col_to_integrate_out_1] = repeated_X[:, 0]
    expanded_data[:, num_col_to_integrate_out_2] = repeated_X[:, 1]
    
    print()

    l_expanded_data = torch.exp(model.log_likelihood(expanded_data,return_lambda_matrix=False))

    l_expanded_data_weighted = repeated_W * l_expanded_data

    l_expanded_data_weighted_reshaped = l_expanded_data_weighted.view(-1, len(X))
    approx_integral = l_expanded_data_weighted_reshaped.sum(dim=1)

    return torch.log(approx_integral)


def compute_ci_probability_deviance_two_dim_glq(
    model,
    data,
    num_col_1,
    num_col_2,
    min_val,
    max_val,
    num_points_quad,
    copula=False,
    optimized=False,
):
    with torch.no_grad():
        # compute p1 = log(f(Y)) from the model
        p1_glq = model.log_likelihood(data, return_lambda_matrix=False)

        # compute p2 = log(f(Y_{/ij}))
        # z_tilde = model.after_transformation(data)
        # p2_glq = torch.distributions.Normal(0, 1).log_prob(z_tilde[:, num_col_1])
        if optimized == False:
            p2_glq = torch.hstack(
                [
                    list_comprehension_integrate_out_two_variables(
                        model=model,
                        data=data[i, :],
                        num_col_to_integrate_out_1=num_col_1,
                        num_col_to_integrate_out_2=num_col_2,
                        min_val=min_val,
                        max_val=max_val,
                        num_points_quad=num_points_quad,
                        copula=copula,
                    )
                    for i in range(  # no need to **2 the number of points as the package we use already does that
                        data.size(0)
                    )
                ]
            )
        else:
            p2_glq = list_comprehension_integrate_out_two_variables_optimized(
                model=model,
                data=data,
                num_col_to_integrate_out_1=num_col_1,
                num_col_to_integrate_out_2=num_col_2,
                min_val=min_val,
                max_val=max_val,
                num_points_quad=num_points_quad,
                copula=copula,
            )
        p2_glq = p2_glq

        # compute p3 = log(f(Y_i,Y_j|Y_{/i,j})) = log(f(Y)) - log(f(Y_{/i,j}))
        p3_glq = p1_glq - p2_glq

        # compute p4 = log(f(Y_i|Y_{/ij})) = log(f(Y_i,Y_{/ij})) - log(f(Y_{/ij}))
        if optimized == False:
            p4_glq = torch.hstack(
                [
                    list_comprehension_integrate_out_one_variable(
                        model=model,
                        data=data[i, :],
                        num_col_to_integrate_out=num_col_2,
                        min_val=min_val,
                        max_val=max_val,
                        num_points_quad=num_points_quad,
                        copula=copula,
                    )
                    for i in range(data.size(0))
                ]
            )
        else:
            p4_glq = list_comprehension_integrate_out_one_variable_optimized(
                model=model,
                data=data,
                num_col_to_integrate_out=num_col_2,
                min_val=min_val,
                max_val=max_val,
                num_points_quad=num_points_quad,
                copula=copula,
            )
        p4_glq = p4_glq - p2_glq

        # compute p5 = log(f(Y_j|Y_{/ij})) = log(f(Y_j,Y_{/ij})) - log(f(Y_{/ij}))
        if optimized == False:
            p5_glq = torch.hstack(
                [
                    list_comprehension_integrate_out_one_variable(
                        model=model,
                        data=data[i, :],
                        num_col_to_integrate_out=num_col_1,
                        min_val=min_val,
                        max_val=max_val,
                        num_points_quad=num_points_quad,
                        copula=copula,
                    )
                    for i in range(data.size(0))
                ]
            )
        else:
            p5_glq = list_comprehension_integrate_out_one_variable_optimized(
                model=model,
                data=data,
                num_col_to_integrate_out=num_col_1,
                min_val=min_val,
                max_val=max_val,
                num_points_quad=num_points_quad,
                copula=copula,
            )
        p5_glq = p5_glq - p2_glq

        actual_log_distribution_glq = p3_glq  # .detach().cpu().numpy()
        under_ci_assumption_log_distribution_glq = (
            p4_glq + p5_glq
        )  # .detach().cpu().numpy()

        # Compute Probability of full Data under the CI Assumption
        # = f(Y)
        # = f(Y_i, Y_j|Y_{/ij}) * f(Y_{/ij})
        # = f(Y_i|Y_{/ij}) * f(Y_j|Y_{/ij}) * f(Y_{/ij})
        #
        # log of the data:
        #
        # = log(f(Y_i|Y_{/ij})) + log(f(Y_j|Y_{/ij})) + log(f(Y_{/ij}))
        # = log( f(Y_i,Y_{/ij}) / f(Y_{/ij}) ) + log( f(Y_j,Y_{/ij}) / f(Y_{/ij}) ) + log(f(Y_{/ij}))
        #
        # p4 = log(f(Y_i|Y_{/ij})) = log(f(Y_i,Y_{/ij})) - log(f(Y_{/ij}))
        # p5 = log(f(Y_j|Y_{/ij})) = log(f(Y_j,Y_{/ij})) - log(f(Y_{/ij}))
        # p2 = log(f(Y_{/ij}))
        # = p4 + p5 + p2
        under_ci_assumption_log_distribution_glq_full_data = p4_glq + p5_glq + p2_glq

    return (
        actual_log_distribution_glq,
        under_ci_assumption_log_distribution_glq,
        under_ci_assumption_log_distribution_glq_full_data,
    )


def independence_kld_process_row(
    row_num,
    precision_matrix_summary_statistics,
    evaluation_data,
    model_alternative,
    num_points_quad=20,
    optimized=False,
    min_val=-5,
    max_val=5,
):
    var_row_num = int(precision_matrix_summary_statistics.iloc[row_num]["var_row"])
    var_col_num = int(precision_matrix_summary_statistics.iloc[row_num]["var_col"])

    print(
        f"Processing row {row_num} with var_row_num {var_row_num} and var_col_num {var_col_num}."
    )

    (
        actual_log_distribution_glq,
        under_ci_assumption_log_distribution_glq,
        under_ci_assumption_log_distribution_glq_full_data,
    ) = compute_ci_probability_deviance_two_dim_glq(
        model_alternative,
        data=evaluation_data,
        num_col_1=var_row_num,
        num_col_2=var_col_num,
        min_val=min_val,  # Adjust as needed
        max_val=max_val,  # Adjust as needed
        num_points_quad=num_points_quad,
        copula=False,  ####### False
        optimized=optimized,
    )

    return (
        actual_log_distribution_glq.cpu(),
        under_ci_assumption_log_distribution_glq.cpu(),
        under_ci_assumption_log_distribution_glq_full_data.cpu(),
    )
