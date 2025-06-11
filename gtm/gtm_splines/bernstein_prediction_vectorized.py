import numpy as np
import torch

from gtm.gtm_splines.splines_utils import ReLULeR, custom_sigmoid


def binomial_coeffs(n, device=None):
    i = torch.arange(n + 1, device=device)
    return torch.exp(
        torch.lgamma(torch.tensor(n + 1.0, device=device))
        - torch.lgamma(i + 1.0)
        - torch.lgamma(torch.tensor(n - i + 1.0, device=device))
    )  # (D,)


def bernstein_basis_batched(x, n, binom):
    """
    Compute Bernstein basis for batched x.

    Args:
        x: Tensor (B, N) with values in [0, 1]
        n: degree
        binom: (D,) optional precomputed binomial coefficients

    Returns:
        Tensor of shape (B, N, D)
    """
    # B, N = x.shape
    D = n + 1
    device = x.device

    i = torch.arange(D, device=device).view(1, 1, D)  # (1, 1, D)
    x = x.unsqueeze(-1)  # (B, N, 1)

    basis = binom.view(1, 1, D) * x**i * (1 - x) ** (n - i)  # (B, N, D)
    return basis


def bernstein_prediction_batched(x, theta, n, binom):
    """
    Compute Bernstein predictions for batched inputs.

    Args:
        x: Tensor (B, N)
        theta: Tensor (B, D)

    Returns:
        f(x): Tensor (B, N)
    """
    # B, D = theta.shape
    # n = D - 1

    # if binom is None:
    #    binom = binomial_coeffs(n, device=device)  # (D,)

    basis = bernstein_basis_batched(x, n, binom=binom)  # (B, N, D)
    return torch.matmul(basis, theta.unsqueeze(-1)).squeeze(-1)  # (B, N)


def bernstein_first_derivative_batched(x, theta, n, binom):
    # B, D = theta.shape
    # n = D - 1

    # if binom is None:
    #    binom = binomial_coeffs(n - 1, x.device)  # (D,)

    dtheta = n * (theta[:, 1:] - theta[:, :-1])  # (B, D-1)
    basis = bernstein_basis_batched(
        x, n - 1, binom
    )  # binomial_coeffs(n - 1, x.device))  # (B, N, D-1)
    return torch.matmul(basis, dtheta.unsqueeze(-1)).squeeze(-1)  # (B, N)


def bernstein_second_derivative_batched(x, theta, n, binom):
    # B, D = theta.shape
    # n = D - 1

    # if binom is None:
    #    binomial_coeffs(n - 2, x.device)  # (D,)

    ddtheta = (
        n * (n - 1) * (theta[:, 2:] - 2 * theta[:, 1:-1] + theta[:, :-2])
    )  # (B, D-2)
    basis = bernstein_basis_batched(
        x, n - 2, binom
    )  # binomial_coeffs(n - 2, x.device))  # (B, N, D-2)
    return torch.matmul(basis, ddtheta.unsqueeze(-1)).squeeze(-1)  # (B, N)


##################################################################################################################################################################################################################
########################## Bernstein Prediction Method ##########################
##################################################################################################################################################################################################################


# Bspline Prediction using the deBoor algorithm
def bernstein_prediction_vectorized(
    params_a,
    input_a,
    degree,
    spline_range,
    knots=None,
    monotonically_increasing=False,
    derivativ=0,
    return_penalties=False,
    calc_method="does_not_matter",
    span_factor=0.1,
    span_restriction="reluler",
    covariate=False,
    params_covariate=False,
    covariate_effect="multiplicativ",
    penalize_towards=0,
    order=3,
    varying_degrees=True,
    params_a_mask=None,
    binom_n=None,
    binom_n1=None,
    binom_n2=None,
):  # device=None

    input_a_clone = input_a

    if span_restriction == "sigmoid":
        input_a_clone = custom_sigmoid(input_a_clone, spline_range)
    elif span_restriction == "reluler":
        reluler = ReLULeR(spline_range)
        input_a_clone = reluler.forward(input_a_clone)
    else:
        pass

    # B, N, D = 10, 1000, 16  # 4 polynomials, 100 points each, degree 15
    # input_a  # (B, N)
    # params_a  # (B, D)

    n = params_a.shape[0] - 1

    # get into the 0,1 range
    span_data = spline_range[1] - spline_range[0]
    input_a_clone = input_a_clone / span_data + 0.5

    if derivativ == 0:
        prediction = bernstein_prediction_batched(
            input_a_clone.T, params_a.T, n=n, binom=binom_n
        )  # (B, N)
    elif derivativ == 1:
        prediction = bernstein_first_derivative_batched(
            input_a_clone.T, params_a.T, n=n, binom=binom_n1
        )  # (B, N)
        prediction = prediction / span_data
    # second_derivativ = bernstein_second_derivative_batched(input_a_clone.T, params_a.T, n=n, binom=binom_n2)  # (B, N)

    # Adding Covariate in a GAM manner
    # if covariate is not False:
    #    params_covariate_restricted = params_covariate.clone().contiguous()
    #
    #    prediction_covariate = ...
    #
    #    prediction = prediction * prediction_covariate

    if return_penalties:
        # if varying_degrees == False:
        second_order_ridge_pen = torch.sum(
            torch.diff(params_a, n=2, dim=0) ** 2
        )  # torch.mean(second_derivativ**2)
        first_order_ridge_pen = torch.sum(
            torch.diff(params_a, n=1, dim=0) ** 2
        )  # torch.mean(first_derivativ**2)
        param_ridge_pen = torch.sum(
            (params_a - penalize_towards) ** 2
        )  # penalize_towards
        # else:
        #    second_order_ridge_pen = torch.sum(torch.diff(params_a,n=2,dim=0)[params_a_mask[2:,:].bool()]**2)
        #    first_order_ridge_pen = torch.sum(torch.diff(params_a,n=1,dim=0)[params_a_mask[1:,:].bool()]**2)
        #    param_ridge_pen = torch.sum((params_a - penalize_towards)[params_a_mask.bool()]**2)
        # Adding Covariate parameter penalisation values
        # if covariate is not False:
        #    second_order_ridge_pen += torch.sum(torch.diff(params_covariate_restricted, n=2, dim=0) ** 2)
        #    first_order_ridge_pen += torch.sum(torch.diff(params_covariate_restricted, n=1, dim=0) ** 2)
        #    param_ridge_pen += torch.sum((params_covariate_restricted - penalize_towards) ** 2) #penalize_towards

        return (
            prediction.T.squeeze(),
            second_order_ridge_pen,
            first_order_ridge_pen,
            param_ridge_pen,
        )
    else:
        return prediction.T.squeeze()
