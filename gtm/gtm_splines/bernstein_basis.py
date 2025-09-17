import torch
from torch import Tensor
from torch.nn.functional import softplus

from gtm.gtm_splines.splines_utils import adjust_ploynomial_range


# https://github.com/pytorch/pytorch/issues/47841
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.binom.html
def torch_binom(n, v):
    # mask = n.detach() >= v.detach()
    # n = mask * n
    # v = mask * v
    a = torch.lgamma(n + 1) - torch.lgamma((n - v) + 1) - torch.lgamma(v + 1)
    return torch.exp(a)  # * mask


# https://en.wikipedia.org/wiki/Bernstein_polynomial
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.BPoly.html
def b(v, n, x):
    # print("torch_binom(n, v)", torch_binom(n, v).to(x.device))
    return torch_binom(n, v) * x**v * (1 - x) ** (n - v)


def compute_bernstein_basis(
    x, degree, spline_range, span_factor, derivativ=0
):  # device=None
    """

    :param x:
    :param degree:
    :return:
    - axis 0: Observation Number
    - axis 1: Basis Function Number
    """

    # Adjust polynomial range to be a bit wider
    # Empirically found that this helps with the fit
    # print("compute_bernstein_basis spline_range", spline_range)
    spline_range = adjust_ploynomial_range(spline_range, span_factor)
    # print("adjust_ploynomial_range spline_range", spline_range)

    # Standardising the Data
    normalizing_range = spline_range[1] - spline_range[0]
    x = (x - spline_range[0]) / (normalizing_range)

    # print("x", x.device)
    # print("normalizing_range", normalizing_range.device)

    n = degree
    if derivativ == 0:
        return torch.vstack(
            [
                b(
                    torch.FloatTensor([v]).to(x.device),
                    torch.FloatTensor([degree]).to(x.device),
                    x,
                )
                for v in range(degree + 1)
            ]
        ).T
    # TODO: write theory on why this is correct way to do the derivativ even when we have a covariate
    elif derivativ == 1:
        # The Bernstein polynomial basis: A centennial retrospective p.391 (17)
        # aded the normalizing range due to the standartisation transformation
        return torch.vstack(
            [
                1
                / normalizing_range.to(x.device)
                * torch.FloatTensor([n]).to(x.device)
                * (
                    b(
                        torch.FloatTensor([v - 1]).to(x.device),
                        torch.FloatTensor([n - 1]).to(x.device),
                        x,
                    )
                    - b(
                        torch.FloatTensor([v]).to(x.device),
                        torch.FloatTensor([n - 1]).to(x.device),
                        x,
                    )
                )
                for v in range(n + 1)
            ]
        ).T


def kron(input_basis, covariate_basis):
    """
    My custom kronecker product implementation
    from (N,D) and (N,D) to (N,D^2)

    torch.kron(torch.tensor([1,2,3]),torch.tensor([1,1,1])) = tensor([1, 1, 1, 2, 2, 2, 3, 3, 3])
    # so for out case we have the intercepts first, thenn all the beta1, then all beta2 etc...
    """
    return torch.vstack(
        [
            torch.kron(input_basis[i, :], covariate_basis.T[:, i])
            for i in range(input_basis.size(0))
        ]
    )


def compute_multivariate_bernstein_basis(
    input, degree, spline_range, span_factor, derivativ=0, covariate=False
):  # device=None
    # We essentially do a tensor prodcut of two splines! : https://en.wikipedia.org/wiki/Bernstein_polynomial#Generalizations_to_higher_dimension

    if covariate is not False:
        multivariate_bernstein_basis = torch.empty(
            size=(input.size(0), (degree + 1) * (degree + 1), input.size(1))
        )
    else:
        multivariate_bernstein_basis = torch.empty(
            size=(input.size(0), (degree + 1), input.size(1))
        )

    # print("compute_multivariate_bernstein_basis spline_range", spline_range)
    for var_num in range(input.size(1)):
        input_basis = compute_bernstein_basis(
            x=input[:, var_num],
            degree=degree,
            spline_range=spline_range[:, var_num],
            span_factor=span_factor,
            derivativ=derivativ,
        )  # device=device
        if covariate is not False:
            # covariate are transformed between 0 and 1 before inputting into the model
            # dont take the derivativ w.r.t to the covariate when computing jacobian of the transformation
            covariate_basis = compute_bernstein_basis(
                x=covariate,
                degree=degree,
                spline_range=torch.tensor([0, 1]),
                span_factor=span_factor,
                derivativ=derivativ,
            )  # device=device
            basis = kron(input_basis, covariate_basis)
        else:
            basis = input_basis

        multivariate_bernstein_basis[:, :, var_num] = basis

    return multivariate_bernstein_basis


# TODO: do this with softplus, somehow gives issue, only works with exp
# def restrict_parameters(params_a, covariate, degree, monotonically_increasing,device=None): #####Requires device as input not passed
#    if monotonically_increasing:
#    # check out Bayesian CTM book 2.1 theorem!!!
#
#        #params_restricted = torch.randn((16*16))
#        params_restricted = params_a.clone()
#        for num_var in range(params_a.size(1)):
#            if covariate == 1:
#                # exp() for all parameters except the intercepts for each different covariate value
#                #softplus = torch.nn.Softplus()
#                #params_restricted[degree:,num_var] = torch.exp(params_restricted[degree:,num_var])
#                params_restricted[degree:,num_var] = torch.log(1 + torch.exp(params_restricted[degree:,num_var]))
#                #params_restricted[degree:,num_var] = torch.nn.functional.softplus(params_a[degree:,num_var])
#                ############## before used torch.exp, johannes suggested to use softplus as then it does not explode that much
#                # Summing up of each value with all its prior values for each different covariate value
#                params_restricted[:,num_var] = torch.matmul(params_restricted[:,num_var],
#                                                 torch.kron(torch.triu(torch.ones(degree+1, degree+1)),
#                                                            torch.eye(degree+1))
#                                                            )
#            else:
#                # simple case without covariate
#                # exp() for all parameters except the intercept
#                #params_restricted[1:,num_var] = torch.exp(params_restricted[1:,num_var])
#                params_restricted[1:,num_var] = torch.log(1 + torch.exp(params_restricted[1:,num_var]))
#                #softplus = torch.nn.Softplus()
#                #params_restricted[1:,num_var] = torch.nn.functional.softplus(params_a[degree:,num_var])
#                # Summing up of each value with all its prior values
#                #summing_matrix = torch.ones(degree+1, degree+1, device=device) #the input.device is npassed here
#
#                summing_matrix = torch.ones(params_restricted.size(0), params_restricted.size(0), device=device) #the input.device is npassed here
#
#
#                #if dev is not False:
#                #    summing_matrix.to(dev)
#
#                summing_matrix = torch.triu(summing_matrix)
#                summing_matrix = summing_matrix.to(params_restricted.device)
#
#                params_restricted[:,num_var] = torch.matmul(params_restricted[:,num_var],summing_matrix)
#    else:
#        params_restricted = params_a
#
#    return params_restricted


# Updated version that is vectorised (no for loop over dimensions)
def restrict_parameters(
    params_a: Tensor,
    covariate: bool,
    degree: int,
    monotonically_increasing: bool,
    device: torch.device=None
    ) -> Tensor:
    
    if not monotonically_increasing:
        return params_a.clone()

    params_restricted: Tensor = params_a.clone().T  # [B, K]
    B, K = params_restricted.shape

    if covariate:
        # Infer number of variables
        num_vars = K // (degree + 1)
        if K % (degree + 1) != 0:
            raise ValueError("K must be divisible by (degree + 1) when covariate == 1")

        # Apply softplus to all but the intercepts (i.e., first entry in each degree+1 block)
        params_restricted = params_restricted.view(B, degree + 1, num_vars)  # [B, D+1, V]
        
        params_restricted[:, 1:, :] = torch.nn.functional.softplus(params_restricted[:, 1:, :])
        
        params_restricted = params_restricted.view(B, K)  # back to [B, K]

        # Build Kronecker sum matrix: [K, K]
        tri = torch.triu(
            torch.ones(degree + 1, degree + 1, device=device)
        )  # [D+1, D+1]
        eye = torch.eye(num_vars, device=device)  # [V, V]
        sum_matrix = torch.kron(tri, eye)  # [K, K]

        # Apply cumulative summation via matrix multiplication
        params_restricted = torch.matmul(params_restricted, sum_matrix.T)  # [B, K]

    else:
        # Apply softplus to non-intercept parameters (index 1 onward)
        #params_restricted[:, 1:] = torch.log(
        #    1 + torch.exp(params_restricted[:, 1:])
        #)  # torch.nn.functional.
        
        params_restricted[:, 1:] = softplus(params_restricted[:, 1:])
        
        # Create upper triangular summing matrix: [K, K]
        sum_matrix: Tensor = torch.triu(input=torch.ones(K, K, device=device))  # [K, K]
        
        # Apply cumulative sum: [B, K] x [K, K]ᵗ = [B, K]
        params_restricted: Tensor = torch.matmul(input=params_restricted, other=sum_matrix)

    return params_restricted.T

    # restrict the last knot value
    # params_restricted2 = params_restricted.clone()
    # for num_var in range(params_a.size(1)):
    #    if params_restricted[params_restricted.size(0)-1,num_var] > 15.1:
    #        print("Warning, restricting the max upper value of the spline!")
    #        params_restricted2[:,num_var] = params_restricted[:,num_var] / params_restricted[params_restricted.size(0)-1,num_var] * 15


#
# return params_restricted2
