import torch
import numpy as np
from gtm.gtm_splines.splines_utils import adjust_ploynomial_range, ReLULeR, custom_sigmoid
from gtm.gtm_splines.bernstein_basis import kron


################################################################################################################################################################################################################## 
########################## Naive ##########################
##################################################################################################################################################################################################################
################################################################################################################################################################################################################## 
########################## Naive Basis Fixed Degree ##########################
##################################################################################################################################################################################################################


def x_in_intervall_fixed_degree(x, i, t):
    # if t[i] <= x < t[i+1] then this is one, otherwise zero
    return torch.where(t[i] <= x,
                       torch.FloatTensor([1.0]).to(x.device), #need to.device() to not have the legacy error
                       torch.FloatTensor([0.0]).to(x.device)) * \
           torch.where(x < t[i+1],
                       torch.FloatTensor([1.0]).to(x.device),
                       torch.FloatTensor([0.0]).to(x.device))

def B_fixed_degree(x, k, i, t):
    """

    :param x: observatioon vector
    :param k: degree of the basis function
    :param i:
    :param t: knots vector
    :return:
    """

    # added due to derivativ computation of Bspline
    if k < 0:
        return torch.FloatTensor([0.0]).to(x.device)
    if k == 0:
       return x_in_intervall_fixed_degree(x, i, t) 
    if t[i+k] == t[i]:
       c1 = torch.FloatTensor([0.0]).to(x.device)
    else:
       c1 = (x - t[i])/(t[i+k] - t[i]) * B_fixed_degree(x, k-1, i, t)
    if t[i+k+1] == t[i+1]:
       c2 = torch.FloatTensor([0.0]).to(x.device)
    else:
       c2 = (t[i+k+1] - x)/(t[i+k+1] - t[i+1]) * B_fixed_degree(x, k-1, i+1, t)
    return c1 + c2


def B_derivativ_fixed_degree(x, p, i, t, derivativ):
    if derivativ == 0:
        return B_fixed_degree(x, p, i, t)
    elif derivativ > 0:
        return p*(B_derivativ_fixed_degree(x, p-1, i, t, derivativ=derivativ-1)/(t[i+p]-t[i] + 1e-9) - B_derivativ_fixed_degree(x, p-1, i+1, t, derivativ = derivativ-1)/(t[i+p+1]-t[i+1] + 1e-9))
    
    
def Naive_fixed_degree(x, t, c, p, d
                       ):
     
    n = c.size(0)
    
    pred = torch.stack([c[i].unsqueeze(1) * B_derivativ_fixed_degree(x, p, i, t, d) for i in range(n)]).sum(0)

    return pred

################################################################################################################################################################################################################## 
########################## Naive Basis Varying Degree ##########################
##################################################################################################################################################################################################################

import torch

def B_varying_degree(x, k, i, t):
    """
    Compute B-spline basis functions in a vectorized manner.

    :param x: Observation vector of shape [batch_size, num_x]
    :param k: Degree of the B-spline
    :param i: Knot index tensor of shape [batch_size, num_x]
    :param t: Knot vector of shape [batch_size, num_knots]
    :return: Basis function values of shape [batch_size, num_x]
    """
    device = x.device

    # Base case: k == 0 -> Indicator function for the knot interval
    if k == 0:
        return ((torch.gather(t, 1, i) <= x) & (x < torch.gather(t, 1, i + 1))).float()
    

    # Compute the two terms of the recursive definition
    t_i = torch.gather(t, 1, i)            # t[i]
    t_ik = torch.gather(t, 1, i + k)       # t[i+k]
    t_ik1 = torch.gather(t, 1, i + k + 1)  # t[i+k+1]
    t_ip1 = torch.gather(t, 1, i + 1)      # t[i+1]

    # Compute coefficients, avoiding division by zero
    denom1 = t_ik - t_i
    denom2 = t_ik1 - t_ip1
    mask1 = denom1 != 0
    mask2 = denom2 != 0

    c1 = torch.where(mask1, ((x - t_i) / (denom1 + 1e-9)) * B_varying_degree(x, k - 1, i, t), torch.zeros_like(x))
    c2 = torch.where(mask2, ((t_ik1 - x) / (denom2 + 1e-9)) * B_varying_degree(x, k - 1, i + 1, t), torch.zeros_like(x))

    return c1 + c2

def B_derivativ_varying_degree(x, p, i, t, derivativ):
    i = torch.tensor([i], device=x.device).expand(t.size(0),1)
    if derivativ == 0:
        return B_varying_degree(x, p, i, t)
    elif derivativ > 0:
        #return p*(B_derivativ(x, p-1, i, t, derivativ=derivativ-1)/(t[i+p]-t[i]) - B_derivativ(x, p-1, i+1, t, derivativ = derivativ-1)/(t[i+p+1]-t[i+1]))
        
        # Context: the i[0,0] is to handle making "i" a 2D element above in the recusiv fct definition. That way gather works here and inside the recursion i gets redefined
        # Context: the + 1e-9 handle the case where we have to do padding and hence the knots we addd out of the borders are on top of each other and thus there distance is zero resulting in a nan because of division through zero
        #          recall that these dont matter anyway as there parameters are set to zero and the basis will always be zero as the knots are far out of where we restrict the data due to the reluler
        return p*(B_derivativ_varying_degree(x, p-1, i[0,0], t, derivativ=derivativ-1)/(torch.gather(t, 1, i+p)-torch.gather(t, 1, i) + 1e-9) - B_derivativ_varying_degree(x, p-1, i[0,0]+1, t, derivativ = derivativ-1)/(torch.gather(t, 1, i+p+1)-torch.gather(t, 1, i+1) + 1e-9))

def Naive_varying_degree(x, t, c, p, d
                         ): 
    n = c.size(0)
    
    pred = torch.stack([c[i].unsqueeze(1) * B_derivativ_varying_degree(x, p, i, t, d) for i in range(n)]).sum(0)

    return pred

################################################################################################################################################################################################################## 
########################## Naive Basis (store_basis==True) ##########################
##################################################################################################################################################################################################################
 

def Naive_Basis(x, spline_range, degree, span_factor, knots, derivativ=0,order=3):
    #print("Naive_Basis spline_range", spline_range)
    #basic order is 3 (cubic spline) so that the third derivative is nonzero for score matching
    p = order
    if order == 2:
        n = degree + 1
    elif order == 3:
        n = degree + 2

    distance_between_knots = (spline_range[1] - spline_range[0]) * (1+span_factor) / (n - 1)

    #knots = torch.linspace(spline_range[0] * (1+span_factor) - order * distance_between_knots,
    #                       spline_range[1] * (1+span_factor) + order * distance_between_knots,
    #                                 n + 4, dtype=torch.float32, device=x.device)

    t = knots

    n = len(t) - p - 1 - 1
    return torch.vstack([B_derivativ_fixed_degree(x, p, i, t, derivativ=derivativ) for i in range(n)]).T


def compute_multivariate_bspline_basis(input, degree, spline_range, span_factor, knots, covariate=False, derivativ=0): 
    # We essentially do a tensor prodcut of two splines! : https://en.wikipedia.org/wiki/Bernstein_polynomial#Generalizations_to_higher_dimension
       
    bspline_basis_list = []
    for var_num in range(input.size(1)):
        input_basis = Naive_Basis(x=input[:, var_num], degree=degree[var_num], spline_range=spline_range[:, var_num], knots=knots,#[:, var_num], 
                                  span_factor=span_factor, derivativ=derivativ)
        if covariate is not False:
            #covariate are transformed between 0 and 1 before inputting into the model
            # dont take the derivativ w.r.t to the covariate when computing jacobian of the transformation
            covariate_basis = Naive_Basis(x=covariate, degree=degree, spline_range=torch.tensor([0,1],device=input.device), span_factor=span_factor, derivativ=derivativ)
            basis = kron(input_basis, covariate_basis)
        else:
            basis = input_basis
        
        bspline_basis_list.append(basis)
       
     
    padded_multivariate_bspline_basis = torch.stack([
            torch.nn.functional.pad(b, (0, max(degree) + 1 - b.size(1))) for b in bspline_basis_list
            ],dim=2)
    
    return padded_multivariate_bspline_basis

################################################################################################################################################################################################################## 
################################################################################################################################################################################################################## 
########################## DeBoor ##########################
##################################################################################################################################################################################################################
########################## DeBoor Varying Degree ##########################
################################################################################################################################################################################################################## 

def compute_k_varying_degrees(x, t
                              ):
    """
    Finds the knot interval index for each x in a batch-friendly way.

    Args:
        x: (batch, num_x) - Input values for each spline
        t: List of tensors [(num_knots_i,)] of different knot positions per spline
        n: (batch,) - Number of control points for each spline

    Returns:
        k: (batch, num_x) - Indices of knot intervals
    """

    # Vectorized searchsorted
    k = torch.searchsorted(t, x.contiguous(), right=False) - 1
    
    return k

def deboor_algorithm_varying_degrees(x, k, t, c, p):
    """
    Vectorized De Boor algorithm for B-spline evaluation.

    Args:
        x: (batch, num_x) - Input values
        k: (batch, num_x) - Knot interval indices
        t: List of tensors [(num_knots_i,)] - Knot positions per spline
        c: List of tensors [(num_control_i, output_dim)] - Control points per spline
        p: (scalar) - Degree of the B-spline

    Returns:
        y: (batch, num_x, output_dim) - Evaluated spline values
    """
    batch_size, num_x = x.shape

    d = torch.stack([ torch.stack( [c[i][j + k[i] - p] for j in range(p + 1)] , dim=0) for i in range(0,batch_size) ])

    # Recursive De Boor iterations
    for r in range(1, p + 1):
        for j in range(p, r - 1, -1):
            
            alpha = (x - torch.gather(t, 1, j + k - p) ) / ( torch.gather(t, 1, j + 1 + k - r) - torch.gather(t, 1, j + k - p) + 1e-9)  # Avoid div by zero
            
            d[:,j] = (1 - alpha) * d[:,j - 1] + alpha * d[:,j]
    
    return d[:,p]

def deboor_algorithm_varying_degrees_first_derivativ(x, k, t, c, p):
        batch_size, num_x = x.shape
        
        # https://stackoverflow.com/questions/57507696/b-spline-derivative-using-de-boors-algorithm
        # range(p) not range(p+1) as in the non derivativ form
        #q = torch.stack([ torch.stack( [
        #                                  p * (c[i][j+k[i]-p+1] - c[i][j+k[i]-p]) / (t[i][j+k[i]+1] - t[i][j+k[i]-p+1])
        #                                     for j in range(p + 1)] , dim=0) for i in range(0,batch_size) ])
        
        q = torch.stack([ torch.stack( [
                                          p * (c[i][j+k[i]-p+1] - c[i][j+k[i]-p]) / (t[i][j+k[i]+1] - t[i][j+k[i]-p+1])
                                             for j in range(p)] , dim=0) for i in range(0,batch_size) ])

        for r in range(1, p):
            for j in range(p-1, r-1, -1):
                right = j+1+k-r
                left = j+k-(p-1)

                alpha = (x - torch.gather(t, 1, left)) / (torch.gather(t, 1, right) - torch.gather(t, 1, left))
                q[:,j] = (1.0 - alpha) * q[:,j-1] + alpha * q[:,j]
                
        return q[:,p-1]
    
    
def run_deBoor_varying_degrees(x, t, c, p, d):
    # Compute knot indices
    k = compute_k_varying_degrees(x, t) #, n)

    if d == 0:
        # Compute B-spline outputs
        prediction = deboor_algorithm_varying_degrees(x, 
                                                k, 
                                                t, 
                                                c, 
                                                p).squeeze()
    elif d == 1:
        # Compute B-spline outputs
        prediction = deboor_algorithm_varying_degrees_first_derivativ(x, 
                                                k, 
                                                t, 
                                                c, 
                                                p).squeeze()
            
    return prediction

################################################################################################################################################################################################################## 
########################## DeBoor Fixed Degree  ##########################
##################################################################################################################################################################################################################

def compute_k_fixed_degrees(x, t#, n
                            ):
    """
    Finds the knot interval index for each x in a batch-friendly way.
    
    Args:
        x: (batch, num_x) - Input values for each spline
        t: (batch, num_knots) - Knot positions for each spline
        n: (scalar) - Number of control points (same for all splines)

    Returns:
        k: (batch, num_x) - Indices of knot intervals
    """
    k = torch.searchsorted(t, x.contiguous()) - 1
    return k

def deboor_algorithm_fixed_degrees(x, k, t, c, p):
    """
    Vectorized De Boor algorithm for B-spline evaluation.
    
    Args:
        x: (batch, num_x) - Input values
        k: (batch, num_x) - Knot interval indices
        t: (num_knots) - Knot positions
        c: (batch, num_control, output_dim) - Control points for each spline
        p: (scalar) - Degree of the B-spline
    
    Returns:
        y: (batch, num_x, output_dim) - Evaluated spline values
    """
    batch_size, num_x = x.shape

    d = torch.stack([ torch.stack( [c[i][j + k[i] - p] for j in range(p + 1)] , dim=0) for i in range(0,batch_size) ])

    # Recursive De Boor iterations
    for r in range(1, p + 1):
        for j in range(p, r - 1, -1):
            alpha = (x - t[j + k - p]) / (t[j + 1 + k - r] - t[j + k - p] + 1e-9)  # Avoid div by zero
            
            new_d = d.clone()
            new_d[:, j] = (1 - alpha) * d[:,j - 1] + alpha * d[:,j]
            d = new_d 


    return d[:,p]  # Final evaluated values


def deboor_algorithm_fixed_degrees_first_derivativ(x, k, t, c, p):
        batch_size, num_x = x.shape
        
        #q = torch.stack([ torch.stack( [
        #                                  p * (c[i][j+k[i]-p+1] - c[i][j+k[i]-p]) / (t[j+k[i]+1] - t[j+k[i]-p+1])
        #                                     for j in range(p + 1)] , dim=0) for i in range(0,batch_size) ])
        
        # https://stackoverflow.com/questions/57507696/b-spline-derivative-using-de-boors-algorithm
        # range(p) not range(p+1) as in the non derivativ form
        q = torch.stack([ torch.stack( [
                                          p * (c[i][j+k[i]-p+1] - c[i][j+k[i]-p]) / (t[j+k[i]+1] - t[j+k[i]-p+1])
                                             for j in range(p)] , dim=0) for i in range(0,batch_size) ])

        for r in range(1, p):
            for j in range(p-1, r-1, -1):
                right = j+1+k-r
                left = j+k-(p-1)
                alpha = (x - t[left]) / (t[right] - t[left]) 
                q[:,j] = (1.0 - alpha) * q[:,j-1] + alpha * q[:,j]
                
        return q[:,p-1]


def run_deBoor_fixed_degrees(x, t, c, p, d):
    # Compute knot indices
    k = compute_k_fixed_degrees(x, t#, n
                                )

    if d == 0:
        # Compute B-spline outputs
        prediction = deboor_algorithm_fixed_degrees(x, 
                                                k, 
                                                t, 
                                                c, 
                                                p).squeeze()
    elif d == 1:
        # Compute B-spline outputs
        prediction = deboor_algorithm_fixed_degrees_first_derivativ(x, 
                                                k, 
                                                t, 
                                                c, 
                                                p).squeeze()
            
    return prediction

################################################################################################################################################################################################################## 
########################## B-Spline Prediction Method ##########################
##################################################################################################################################################################################################################


# Bspline Prediction using the deBoor algorithm
def bspline_prediction_vectorized(params_a, input_a, knots,
                       degree, spline_range, monotonically_increasing=False, 
                       derivativ=0, return_penalties=False, calc_method="deBoor",#'Naive_Basis', #before: deBoor 
                       span_factor=0.1, span_restriction="reluler",
                       covariate=False, params_covariate=False, covaraite_effect="multiplicativ",
                       penalize_towards=0, order=3, varying_degrees=True, params_a_mask=None): #device=None

    input_a_clone = input_a
    
    if span_restriction == "sigmoid":
        input_a_clone = custom_sigmoid(input_a_clone, spline_range)
    elif span_restriction == "reluler":
        reluler = ReLULeR(spline_range)
        input_a_clone = reluler.forward(input_a_clone)
    else:
        pass

    if calc_method == "deBoor":

        
        if varying_degrees == True:
            prediction = run_deBoor_varying_degrees(x=input_a_clone.T,
                                                    t=knots.T,
                                                    c=params_a.T,
                                                    p=order,
                                                    d=derivativ
                                                    )
        else:
            prediction = run_deBoor_fixed_degrees(x=input_a_clone.T,
                                                  t=knots.T,
                                                  c=params_a.T,
                                                  p=order,
                                                  d=derivativ
                                                  )
    elif calc_method == "Naive":
        if varying_degrees == True:
            prediction = Naive_varying_degree(  x=input_a_clone.T,
                                                t=knots.T,
                                                c=params_a,
                                                p=order,
                                                d=derivativ
                                                )
        else:
            prediction = Naive_fixed_degree(  x=input_a_clone.T,
                                              t=knots.T,
                                              c=params_a,
                                              p=order,
                                              d=derivativ
                                              )
        
    # Adding Covariate in a GAM manner
    if covariate is not False:
        params_covariate_restricted = params_covariate.clone().contiguous()
        
        if order == 2:
            n = degree + 1
        elif order == 3:
            n = degree + 2

        knots_covariate = torch.linspace(0 - order * 1,
                                         1 + order * 1,
                                         n + 4, dtype=torch.float32, device=input_a.device)

        prediction_covariate = run_deBoor_fixed_degrees(x=covariate,
                                t=knots_covariate,
                                c=params_covariate_restricted,
                                p=order,
                                d=0)

        prediction = prediction * prediction_covariate

    if return_penalties:
        if varying_degrees == False:
            second_order_ridge_pen = torch.sum(torch.diff(params_a,n=2,dim=0)**2)
            first_order_ridge_pen = torch.sum(torch.diff(params_a,n=1,dim=0)**2)
            param_ridge_pen = torch.sum((params_a - penalize_towards)**2) #penalize_towards
        else:
            second_order_ridge_pen = torch.sum(torch.diff(params_a,n=2,dim=0)[params_a_mask[2:,:].bool()]**2)
            first_order_ridge_pen = torch.sum(torch.diff(params_a,n=1,dim=0)[params_a_mask[1:,:].bool()]**2)
            param_ridge_pen = torch.sum((params_a - penalize_towards)[params_a_mask.bool()]**2)
        # Adding Covariate parameter penalisation values
        if covariate is not False:
            second_order_ridge_pen += torch.sum(torch.diff(params_covariate_restricted, n=2, dim=0) ** 2)
            first_order_ridge_pen += torch.sum(torch.diff(params_covariate_restricted, n=1, dim=0) ** 2)
            param_ridge_pen += torch.sum((params_covariate_restricted - penalize_towards) ** 2) #penalize_towards


        return prediction.T, second_order_ridge_pen, first_order_ridge_pen, param_ridge_pen
    else:
        return prediction.T