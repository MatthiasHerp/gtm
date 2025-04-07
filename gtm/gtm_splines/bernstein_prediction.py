import torch
from gtm.gtm_splines.splines_utils import adjust_ploynomial_range

# https://github.com/pytorch/pytorch/issues/47841
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.binom.html
def torch_binom(n, v):
    #mask = n.detach() >= v.detach()
    #n = mask * n
    #v = mask * v
    a = torch.lgamma(n + 1) - torch.lgamma((n - v) + 1) - torch.lgamma(v + 1)
    return torch.exp(a)# * mask

# https://en.wikipedia.org/wiki/Bernstein_polynomial
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.BPoly.html
def b(v, n, x):
    #print("torch_binom(n, v)", torch_binom(n, v).to(x.device))
    return torch_binom(n, v) * x**v * (1 - x)**(n - v)


def compute_bernstein_basis(x, degree, spline_range, span_factor, derivativ=0): #device=None
    """

    :param x:
    :param degree:
    :return:
    - axis 0: Observation Number
    - axis 1: Basis Function Number
    """

    # Adjust polynomial range to be a bit wider
    # Empirically found that this helps with the fit
    #print("compute_bernstein_basis spline_range", spline_range)
    spline_range = adjust_ploynomial_range(spline_range, span_factor)
    #print("adjust_ploynomial_range spline_range", spline_range)

    # Standardising the Data
    normalizing_range = spline_range[1] - spline_range[0]
    x = (x - spline_range[0]) / (normalizing_range)

    #print("x", x.device)
    #print("normalizing_range", normalizing_range.device)

    n = degree
    if derivativ==0:
        return torch.vstack([b(torch.FloatTensor([v]).to(x.device), torch.FloatTensor([degree]).to(x.device), x) for v in range(degree + 1)]).T
    #TODO: write theory on why this is correct way to do the derivativ even when we have a covariate
    elif derivativ==1:
        # The Bernstein polynomial basis: A centennial retrospective p.391 (17)
        # aded the normalizing range due to the standartisation transformation
        return torch.vstack([1/normalizing_range.to(x.device) * torch.FloatTensor([n]).to(x.device) *
                             (b(torch.FloatTensor([v-1]).to(x.device), torch.FloatTensor([n-1]).to(x.device), x) -
                              b(torch.FloatTensor([v]).to(x.device), torch.FloatTensor([n-1]).to(x.device), x)) for v in range(n+1)]).T


def kron(input_basis, covariate_basis):
    """
    My custom kronecker product implementation
    from (N,D) and (N,D) to (N,D^2)

    torch.kron(torch.tensor([1,2,3]),torch.tensor([1,1,1])) = tensor([1, 1, 1, 2, 2, 2, 3, 3, 3])
    # so for out case we have the intercepts first, thenn all the beta1, then all beta2 etc...
    """
    return torch.vstack([torch.kron(input_basis[i,:],covariate_basis.T[:,i]) for i in range(input_basis.size(0))])


def compute_multivariate_bernstein_basis(input, degree, spline_range, span_factor, derivativ=0, covariate=False): #device=None
    # We essentially do a tensor prodcut of two splines! : https://en.wikipedia.org/wiki/Bernstein_polynomial#Generalizations_to_higher_dimension

    if covariate is not False:
        multivariate_bernstein_basis = torch.empty(size=(input.size(0), (degree+1)*(degree+1), input.size(1)))
    else:
        multivariate_bernstein_basis = torch.empty(size=(input.size(0), (degree+1), input.size(1)))

    #print("compute_multivariate_bernstein_basis spline_range", spline_range)
    for var_num in range(input.size(1)):
        input_basis = compute_bernstein_basis(x=input[:, var_num], degree=degree, spline_range=spline_range[:, var_num], span_factor=span_factor, derivativ=derivativ) #device=device
        if covariate is not False:
            #covariate are transformed between 0 and 1 before inputting into the model
            # dont take the derivativ w.r.t to the covariate when computing jacobian of the transformation
            covariate_basis = compute_bernstein_basis(x=covariate, degree=degree, spline_range=torch.tensor([0,1]), span_factor=span_factor, derivativ=derivativ) #device=device
            basis = kron(input_basis, covariate_basis)
        else:
            basis = input_basis

        multivariate_bernstein_basis[:,:,var_num] = basis

    return multivariate_bernstein_basis


#TODO: do this with softplus, somehow gives issue, only works with exp
def restrict_parameters(params_a, covariate, degree, monotonically_increasing,device=None): #####Requires device as input not passed
    if monotonically_increasing:
    # check out Bayesian CTM book 2.1 theorem!!!

        #params_restricted = torch.randn((16*16))
        params_restricted = params_a.clone()
        for num_var in range(params_a.size(1)):
            if covariate == 1:
                # exp() for all parameters except the intercepts for each different covariate value
                #softplus = torch.nn.Softplus()
                #params_restricted[degree:,num_var] = torch.exp(params_restricted[degree:,num_var])
                params_restricted[degree:,num_var] = torch.log(1 + torch.exp(params_restricted[degree:,num_var]))
                #params_restricted[degree:,num_var] = torch.nn.functional.softplus(params_a[degree:,num_var])
                ############## before used torch.exp, johannes suggested to use softplus as then it does not explode that much
                # Summing up of each value with all its prior values for each different covariate value
                params_restricted[:,num_var] = torch.matmul(params_restricted[:,num_var],
                                                 torch.kron(torch.triu(torch.ones(degree+1, degree+1)),
                                                            torch.eye(degree+1))
                                                            )
            else:
                # simple case without covariate
                # exp() for all parameters except the intercept
                #params_restricted[1:,num_var] = torch.exp(params_restricted[1:,num_var])
                params_restricted[1:,num_var] = torch.log(1 + torch.exp(params_restricted[1:,num_var]))
                #softplus = torch.nn.Softplus()
                #params_restricted[1:,num_var] = torch.nn.functional.softplus(params_a[degree:,num_var])
                # Summing up of each value with all its prior values
                #summing_matrix = torch.ones(degree+1, degree+1, device=device) #the input.device is npassed here
                
                summing_matrix = torch.ones(params_restricted.size(0), params_restricted.size(0), device=device) #the input.device is npassed here
                
                
                #if dev is not False:
                #    summing_matrix.to(dev)

                summing_matrix = torch.triu(summing_matrix)
                summing_matrix = summing_matrix.to(params_restricted.device)

                params_restricted[:,num_var] = torch.matmul(params_restricted[:,num_var],summing_matrix)
    else:
        params_restricted = params_a

    return params_restricted
    
    #restrict the last knot value
    #params_restricted2 = params_restricted.clone()
    #for num_var in range(params_a.size(1)):
    #    if params_restricted[params_restricted.size(0)-1,num_var] > 15.1:
    #        print("Warning, restricting the max upper value of the spline!")
    #        params_restricted2[:,num_var] = params_restricted[:,num_var] / params_restricted[params_restricted.size(0)-1,num_var] * 15
#
    #return params_restricted2





def bernstein_prediction(multivariate_bernstein_basis, multivariate_bernstein_basis_derivativ_1,
                         params_a,
                         #input_a,
                         degree,
                         #spline_range,
                         monotonically_increasing=False,
                         derivativ=0,
                         #span_factor=0.1,
                         covariate=False): #device=None

    #if covariate is not False:
    #    input_basis = compute_bernstein_basis(input_a, degree)
    #    covariate_basis = compute_bernstein_basis(covariate, degree)
    #    basis = kron(input_basis, covariate_basis)
    #else:
    #    basis = compute_bernstein_basis(input_a, degree)
#

    params_restricted = restrict_parameters(params_a, covariate, degree, monotonically_increasing, device=multivariate_bernstein_basis.device)

    #if monotonically_increasing:
    ## check out Bayesian CTM book 2.1 theorem!!!
#
    #    #params_restricted = torch.randn((16*16))
    #    params_restricted = params_a.clone()
    #    if covariate is not False:
    #        # exp() for all parameters except the intercepts for each different covariate value
    #        params_restricted[degree:] = torch.exp(params_restricted[degree:])
    #        # Summing up of each value with all its prior values for each different covariate value
    #        params_restricted = torch.matmul(params_restricted,
    #                                         torch.kron(torch.triu(torch.ones(degree+1, degree+1)),
    #                                                    torch.eye(degree+1))
    #                                                    )
    #    else:
    #        # simple case without covariate
    #        # exp() for all parameters except the intercept
    #        params_restricted[1:] = torch.exp(params_restricted[1:])
    #        # Summing up of each value with all its prior values
    #        params_restricted[1:] = torch.matmul(params_restricted,torch.triu(torch.ones(degree+1,degree+1)))

    # Explanation:
    # multivariate_bernstein_basis: 0: observation, 1: basis, 2: variable
    # params: 0: basis, 1: variable
    # output: 0: observation, 1: variable
    # Comment: we do normal multiplication as we want to multiply the parameters of each varaible only with its own basis
    #          we sum over dim 1 which is the basis
    #          note we use the restricted parameters
    if derivativ==0:
        output = torch.sum(multivariate_bernstein_basis * params_restricted.unsqueeze(0), (1))
    elif derivativ==1:
        output = torch.sum(multivariate_bernstein_basis_derivativ_1 * params_restricted.unsqueeze(0), (1))



    #Derivativ to y: check mathmatically if its not just derivativ to the basis of y and then multiplied with thte basis of x
    # if so put derivativ arguement in the basis creating fct

    # Adjust polynomial range to be a bit wider
    # Empirically found that this helps with the fit
    #spline_range = adjust_ploynomial_range(spline_range, span_factor)

    # Restricts Monotonically increasing by insuring that params increase
    #if monotonically_increasing:
#
    #    params_restricted = params_a.clone()
    #    #relu_obj = nn.ReLU()
    #    params_restricted[1:] = torch.matmul(torch.exp(params_a[1:]),torch.triu(torch.ones(degree,degree))) + params_a[0] #abs or exp
    #    #.double() otherwise I started getting the folowing error:
    #    #{RuntimeError}Expected object of scalar type Double but got scalar type Float for argument #2 'mat2' in call to _th_mm
    #    #alternativ set dtype=torch.float64 in torch.ones(degree,degree)
#
    #    # cannot use exp as it destabilizes the optimisation using LBFGS, get inf for params fast?
    #    # however with exp works really well and fast with adam
    #    # Relu works for both, results are good except at lower end there is a cutoff somehow
#
    #    # We also need to restrict covariates to ensure the sum of the gam is monotonically increasing
    #    #if covariate is not False:
    #    #    params_covariate_restricted = params_covariate.clone()
##
    #    #    params_covariate_restricted[1:] = torch.matmul(torch.exp(params_covariate[1:]), torch.triu(torch.ones(degree, degree))) + \
    #    #                            params_covariate[0]
#
    #else:
    #    params_restricted = params_a.clone()
    #    #params_covariate_restricted = params_covariate.clone()
    #n = degree
    ##return sum((params[1] + sum(torch.abs(params[1:(v-1)]))) * b(torch.FloatTensor([v]), torch.FloatTensor([n]), input) for v in range(n+1))

    # penalities
    second_order_ridge_pen = 0
    first_order_ridge_pen = 0
    param_ridge_pen = 0


    #normalizing_range = spline_range[1] - spline_range[0]
    #input_a = (input_a - spline_range[0]) / (normalizing_range)

    #if derivativ == 0:
        # Explanation:
        # multivariate_bernstein_basis: 0: observation, 1: basis, 2: variable
        # params: 0: basis, 1: variable
        # output: 0: observation, 1: variable
        # Comment: we do normal multiplication as we want to multiply the parameters of each varaible only with its own basis
        #          we sum over dim 1 which is the basis
    #    output = torch.sum(self.multivariate_bernstein_basis * self.params.unsqueeze(0), (1))



    #    output = sum(params_restricted[v] * b(torch.FloatTensor([v]), torch.FloatTensor([n]), input_a) for v in range(n+1)) #before we had: params_restricted[v-1]

    #elif derivativ == 1:
        #output = sum(params_restricted[v] * b(torch.FloatTensor([v - 1]), torch.FloatTensor([n]), input_a) * (
        #            torch.FloatTensor([v]) - torch.FloatTensor([n]) * input_a) for v in range(n + 1))
        # My derivativ see goodnotes
    #    output = 1/normalizing_range * sum(params_restricted[v] * torch.FloatTensor([n]) * (b(torch.FloatTensor([v-1]), torch.FloatTensor([n-1]), input_a) -
    #                                                                                        b(torch.FloatTensor([v]), torch.FloatTensor([n-1]), input_a)) for v in range(n+1))
        # The Bernstein polynomial basis: A centennial retrospective p.391 (17)

    # In a GAM we add the covariate effect as a spline model
    #if covariate is not False:
    #    if derivativ == 0:
    #        covariate_effect = sum(params_covariate_restricted[v] * b(torch.FloatTensor([v]), torch.FloatTensor([n]), covariate) for v in
    #                     range(n + 1))
    #        output = output + covariate_effect#.squeeze()
    #        #ODO: why do we need to squeeze here? dont we do the same thing for the covariates as for the input?
    #        # also same question for derivativ below
    #    if derivativ == 1:
    #        # For additiv terms one simple computes derivativ of both individually
    #        covariate_effect = 1 / normalizing_range * sum(params_covariate_restricted[v] * torch.FloatTensor([n]) * (b(torch.FloatTensor([v - 1]), torch.FloatTensor([n - 1]), covariate) -
    #                                                                                        b(torch.FloatTensor([v]), torch.FloatTensor([n - 1]), covariate)) for v in range(n + 1))
    #        output = output + covariate_effect#.squeeze()


    return output, second_order_ridge_pen, first_order_ridge_pen, param_ridge_pen
    #return (output + spline_range[0]) / (spline_range[1] - spline_range[0])