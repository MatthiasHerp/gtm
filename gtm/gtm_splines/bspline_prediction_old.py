import torch
import numpy as np
#import functorch
from gtm.gtm_splines.splines_utils import adjust_ploynomial_range, ReLULeR, custom_sigmoid
from gtm.gtm_splines.bernstein_prediction import restrict_parameters

def x_in_intervall(x, i, t):
    # if t[i] <= x < t[i+1] then this is one, otherwise zero
    return torch.where(t[i] <= x,
                       torch.FloatTensor([1.0]).to(x.device), #need to.device() to not have the legacy error
                       torch.FloatTensor([0.0]).to(x.device)) * \
           torch.where(x < t[i+1],
                       torch.FloatTensor([1.0]).to(x.device),
                       torch.FloatTensor([0.0]).to(x.device))

def B(x, k, i, t):
    """

    :param x: observatioon vector
    :param k: degree of the basis function
    :param i:
    :param t: knots vector
    :return:
    """
    #print("x",x.device)
    #print("t",t.device)

    # added due to derivativ computation of Bspline
    if k < 0:
        return torch.FloatTensor([0.0]).to(x.device)
    if k == 0:
       return x_in_intervall(x, i, t) #torch.FloatTensor([1.0]) if t[i] <= x < t[i+1] else torch.FloatTensor([0.0])
    if t[i+k] == t[i]:
       c1 = torch.FloatTensor([0.0]).to(x.device)
    else:
       c1 = (x - t[i])/(t[i+k] - t[i]) * B(x, k-1, i, t)
    if t[i+k+1] == t[i+1]:
       c2 = torch.FloatTensor([0.0]).to(x.device)
    else:
       c2 = (t[i+k+1] - x)/(t[i+k+1] - t[i+1]) * B(x, k-1, i+1, t)
    return c1 + c2


def B_derivativ(x, p, i, t, derivativ):
    if derivativ == 0:
        return B(x, p, i, t)
    elif derivativ > 0:
        return p*(B_derivativ(x, p-1, i, t, derivativ=derivativ-1)/(t[i+p]-t[i]) - B_derivativ(x, p-1, i+1, t, derivativ = derivativ-1)/(t[i+p+1]-t[i+1]))
 

def B_vmap(x, k, i, t, knot):

    if knot < t[i-k] or knot > t[i+k]:
       return torch.FloatTensor([0.0]).to(x.device)
    if k == 0:
       return x_in_intervall(x, i, t) #torch.FloatTensor([1.0]) if t[i] <= x < t[i+1] else torch.FloatTensor([0.0])
    if t[i+k] == t[i]:
       c1 = torch.FloatTensor([0.0]).to(x.device)
    else:
       c1 = (x - t[i])/(t[i+k] - t[i]) * B(x, k-1, i, t)
    if t[i+k+1] == t[i+1]:
       c2 = torch.FloatTensor([0.0]).to(x.device)
    else:
       c2 = (t[i+k+1] - x)/(t[i+k+1] - t[i+1]) * B(x, k-1, i+1, t)
    return c1 + c2

#def k_is_zero(k):
#    # if k == 0 then this is zero, otherwise one
#    return torch.where(k==torch.tensor(0), torch.FloatTensor([0.0]), torch.FloatTensor([1.0]))
#
#def x_in_intervall(x, i, t):
#    # if t[i] <= x < t[i+1] then this is one, otherwise zero
#    return torch.where(t[i] <= x,
#                       torch.FloatTensor([1.0]),
#                       torch.FloatTensor([0.0])) * \
#           torch.where(x < t[i+1],
#                       torch.FloatTensor([1.0]),
#                       torch.FloatTensor([0.0]))
#
#def c1(x, k, i, t):
#    # computes first part of the bspline
#    return torch.where(k>torch.tensor(0),
#                       torch.where(t[i + k] == t[i],
#                                   torch.FloatTensor([0.0]),
#                                    (x - t[i]) / (t[i + k] - t[i]) * B_vmap(x, k - 1, i, t)),
#                          torch.FloatTensor([0.0]))
#
#def c2(x, k, i, t):
#    # computes second part of the bspline
#    return torch.where(k>torch.tensor(0),
#                       torch.where(t[i + k + 1] == t[i + 1],
#                                   torch.FloatTensor([0.0]),
#                                   (t[i + k + 1] - x) / (t[i + k + 1] - t[i + 1]) * B_vmap(x, k - 1, i + 1, t)),
#                          torch.FloatTensor([0.0]))
#def B_vmap(x, k, i, t):
#    print(k)
#    # computes the bspline with 2 options:
#    # 1. if k == 0 then return 1 if x is in the intervall [t[i], t[i+1]) and 0 otherwise
#    # 2. if k != 0 then return the bspline
#    return torch.where(k>torch.tensor(0),
#                c1(x, k, i, t) + c2(x, k, i, t),
#                x_in_intervall(x, i, t))
#    #return (c1(x, k, i, t) + c2(x, k, i, t)) * k_is_zero(k) + (1-k_is_zero(k)) * x_in_intervall(x, i, t)

def Naive(x, t, c, p, d):
    n = len(t) - p - 1 -1
    #assert (n >= p+1) and (len(c) >= n)
    #pred = x.clone()
    #for obs_num in range(x.size(0)):
    #    pred[obs_num] = sum(c[i] * B(x[obs_num], p, i, t) for i in range(n))

    #pred = sum(c[i] * B(x, p, i, t) for i in range(n))
    
    pred = sum(c[i] * B_derivativ(x, p, i, t, d) for i in range(n))

    return pred

#from python_nf_mctm.splines.bernstein_prediction import kron

def Naive_Basis(x, spline_range, degree, span_factor,derivativ=0,order=3):
    #print("Naive_Basis spline_range", spline_range)
    #order = 3 #2 TODO: changed it to 3 so that the third derivative is nonzero for score matching
    p = order
    #n = degree  + 2 #TODO: for the order 3 made +2 instead of +1 before for order 2
    if order == 2:
        n = degree + 1
    elif order == 3:
        n = degree + 2

    distance_between_knots = (spline_range[1] - spline_range[0]) * (1+span_factor) / (n - 1)

    #print("Naive_Basis distance_between_knots", distance_between_knots)

    knots = torch.linspace(spline_range[0] * (1+span_factor) - order * distance_between_knots,
                           spline_range[1] * (1+span_factor) + order * distance_between_knots,
                                     n + 4, dtype=torch.float32, device=x.device)

    t = knots
    #print(device)
    #print("t", t.device)
    #print("knots", knots.device)


    n = len(t) - p - 1 - 1
    return torch.vstack([B_derivativ(x, p, i, t, derivativ=derivativ) for i in range(n)]).T
    # In the Code below there is some issue in computing the third derivativ see naive_basis_derivative_error.ipynb 
    #if derivativ==0:
    #    return torch.vstack([B(x, p, i, t) for i in range(n)]).T
    #elif derivativ==1:
    #    return torch.vstack([p*(B(x, p-1, i, t)/(t[i+p]-t[i]) - B(x, p-1, i+1, t)/(t[i+p+1]-t[i+1])) for i in range(n)]).T
    #elif derivativ==2:
    #    return torch.vstack([p*(
    #            (p-1)*(B(x, p-2, i, t)/(t[i+p-1]-t[i]) - B(x, p-2, i+1, t)/(t[i+p]-t[i+1]))
    #            /(t[i+p]-t[i]) -
    #            (p-1)*(B(x, p-2, i+1, t)/(t[i+p]-t[i+1]) - B(x, p-2, i+2, t)/(t[i+p+1]-t[i+2]))
    #            /(t[i+p+1]-t[i+1]))
    #                         for i in range(n)]).T
    #elif derivativ==3: #TODO: for order two we have a third derivative of zero
    #    return torch.vstack([p*(
    #            (p-1)*(
    #            (p-2)*(B(x, p-3, i, t)/(t[i+p-1]-t[i]) - B(x, p-3, i+1, t)/(t[i+p]-t[i+1])) #B(x, p-2, i, t)
    #            /(t[i+p-1]-t[i]) -
    #            (p-2)*(B(x, p-3, i+1, t)/(t[i+p]-t[i+1]) - B(x, p-3, i+2, t)/(t[i+p+1]-t[i+2])) #B(x, p-2, i+1, t)
    #            /(t[i+p]-t[i+1]))
    #            /(t[i+p]-t[i]) -
    #            (p-1)*(
    #                    (p-2)*(B(x, p-3, i+1, t)/(t[i+p]-t[i+1]) - B(x, p-3, i+2, t)/(t[i+p+1]-t[i+2])) #B(x, p-2, i+1, t)
    #                    /(t[i+p]-t[i+1]) -
    #                    (p-2)*(B(x, p-3, i+2, t)/(t[i+p+1]-t[i+2]) - B(x, p-3, i+3, t)/(t[i+p+2]-t[i+3])) #B(x, p-2, i+2, t)
    #                    /(t[i+p+1]-t[i+2]))
    #            /(t[i+p+1]-t[i+1]))
    #                         for i in range(n)]).T



def compute_multivariate_bspline_basis(input, degree, spline_range, span_factor, covariate=False, derivativ=0): #device=None
    # We essentially do a tensor prodcut of two splines! : https://en.wikipedia.org/wiki/Bernstein_polynomial#Generalizations_to_higher_dimension

    #print("compute_multivariate_bspline_basis",device)

    #if covariate is not False:
    #    multivariate_bspline_basis = torch.empty(size=(input.size(0), (max(degree)+1)*(max(degree)+1), input.size(1)), device=input.device)
    #else:
    #    multivariate_bspline_basis = torch.empty(size=(input.size(0), (max(degree)+1), input.size(1)), device=input.device)
#
    #    #print("compute_multivariate_bspline_basis spline_range",spline_range)
#
    #for var_num in range(input.size(1)):
    #    input_basis = Naive_Basis(x=input[:, var_num], degree=degree[var_num], spline_range=spline_range[:, var_num], span_factor=span_factor, derivativ=derivativ)
    #    if covariate is not False:
    #        #covariate are transformed between 0 and 1 before inputting into the model
    #        # dont take the derivativ w.r.t to the covariate when computing jacobian of the transformation
    #        covariate_basis = Naive_Basis(x=covariate, degree=degree, spline_range=torch.tensor([0,1],device=input.device), span_factor=span_factor, derivativ=derivativ)
    #        basis = kron(input_basis, covariate_basis)
    #    else:
    #        basis = input_basis
#
    #    multivariate_bspline_basis[:,:,var_num] = basis
    #    
    #return multivariate_bspline_basis
       
    bspline_basis_list = []
    for var_num in range(input.size(1)):
        input_basis = Naive_Basis(x=input[:, var_num], degree=degree[var_num], spline_range=spline_range[:, var_num], span_factor=span_factor, derivativ=derivativ)
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


# cannot use vmap here as B(x, k, i, t) contains if statments which vmap does not support yet:
# https://github.com/pytorch/functorch/issues/257
class Naive_vmap():
    def __init__(self,t,c,p):
        self.t=t
        self.c=c
        self.p=p
        self.n = t.size(0) - 2 * self.p

    def compute_knot(self, x):

        k = torch.searchsorted(self.t, x) - 1
        k[k > (self.n - 1)] = 2 + 1
        k[k > (self.n - 1)] = 2 + (self.n - 1) - 1

        self.knots = k

    def compute_prediction(self, x):
        n = len(self.t) - self.p - 1 -1
        #assert (n >= self.p+1) and (len(self.c) >= n)
        #pred = x.clone()
        #for obs_num in range(x.size(0)):
            #knot_num = knot[torch.tensor(obs_num)[None]][0]
            #pred[obs_num] = sum(self.c[i] * B(x[obs_num], self.p, i, self.t) for i in range(n)) #knot_num-self.p,knot_num+self.p
        #Works
        #pred = sum(self.c[i] * torch.vstack([B_vmap(x[obs_num], self.p, 0, self.t, self.knots[obs_num]) for obs_num in range(x.size(0))]).squeeze() for i in range(n))
        # Does not work
        #pred = sum(self.c[i] * functorch.vmap(B_vmap)(x, self.p, i, self.t, self.knots) for i in range(n))

        def B_vmap(i,obs_num):
            if self.knots[obs_num] < self.t[i - self.p] or self.knots[obs_num] > self.t[i + self.p]:
                return torch.FloatTensor([0.0])
            if k == 0: #TODO: error here
                return x_in_intervall(x[obs_num], i,
                                      self.t)  # torch.FloatTensor([1.0]) if t[i] <= x < t[i+1] else torch.FloatTensor([0.0])
            if self.t[i + self.p] == self.t[i]:
                c1 = torch.FloatTensor([0.0])
            else:
                c1 = (x[obs_num] - self.t[i]) / (self.t[i + self.p] - self.t[i]) * B(x[obs_num], self.k - 1, i, self.t)
            if self.t[i + self.p + 1] == self.t[i + 1]:
                c2 = torch.FloatTensor([0.0])
            else:
                c2 = (self.t[i + self.p + 1] - x[obs_num]) / (self.t[i + self.p + 1] - self.t[i + 1]) * B(x[obs_num], self.p - 1, i + 1, self.t)
            return c1 + c2

        pred = sum(self.c[i] * torch.vstack([B_vmap(i,obs_num) for obs_num in range(x.size(0))]).squeeze() for i in range(n))

        return pred

def run_Naive_vmap(x, t, c, p):
    Naive_vmap_obj = Naive_vmap(t=t, c=c, p=p)

    Naive_vmap_obj.compute_knot(x)

    Naive_vmap_func_vectorized = torch.vmap(Naive_vmap_obj.compute_prediction) #functorch.vmap(Naive_vmap_obj.compute_prediction)
    
    return Naive_vmap_func_vectorized(torch.unsqueeze(x, 0)).squeeze()

class deBoor():
    def __init__(self,t,c,p): #,d):
        self.t=t
        self.c=c
        #print("init self.c.size()",self.c.size())
        self.p=p
        self.n = t.size(0) - 2 * self.p
        #self.d = d

    def compute_k(self, x):

        k = torch.searchsorted(self.t, x) - 1
        #k.index_select(0,torch.nonzero(k > (self.n - 1)).squeeze()) analog to k[k > (self.n - 1)]
        k[k > (self.n - 1)] = 2 + 1
        k[k > (self.n - 1)] = 2 + (self.n - 1) - 1
        #k = torch.where(k > (self.n - 1), torch.tensor(2 + 1), k)
        #k = torch.where(k > (self.n - 1), torch.tensor(2 + (self.n - 1) - 1), k)
        #k.index_select(0,torch.nonzero(k > (self.n - 1)).squeeze()) - k.index_select(0,torch.nonzero(k > (self.n - 1)).squeeze()) +  2 + 1
        #k.index_select(0,torch.nonzero(k > (self.n - 1)).squeeze()) - k.index_select(0,torch.nonzero(k > (self.n - 1)).squeeze()) +  2 + (self.n - 1) - 1

        return k
    
    def compute_prediction_derivative(self, x, k):
        if self.d == 0:
            return self.compute_prediction(x, k)
        elif self.d == 1:
            return self.compute_prediction_first_derivativ(x, k)
            


    def compute_prediction(self, x, k):
        """Evaluates S(x).

        Arguments
        ---------
        k: Index of knot interval that contains x.
        x: Position.
        t: Array of knot positions, needs to be padded as described above.
        c: Array of control points.
        p: Degree of B-spline.
        d: derivative order
        """

        # Here is the issue we have k which is normally a vector and now a matrix with second dim the splines
        # for c we have the same its normally the parameter vector but now it is a matrix with second dim the splines
        # That is why we would want to vmap here as we want to do this for the k and the c that belong together
        d = [self.c[j + k - self.p] for j in range(0, self.p + 1)]
        
        #print("compute prediction")
        
        #print("len(d)",len(d))
        
        #print("self.p.size()",self.p)
        #print("self.c.size()",self.c.size())

        for r in range(1, self.p + 1):
            for j in range(self.p, r - 1, -1):
                alpha = (x - self.t[j + k - self.p]) / (self.t[j + 1 + k - r] - self.t[j + k - self.p])
                #print("d[j].size()", d[j].size())
                #print("alpha.size()", alpha.size())
                #d[j] = (1.0 - alpha.unsqueeze(2)) * d[j - 1] + alpha.unsqueeze(2) * d[j] #(1.0 - alpha) * d[j - 1] + alpha * d[j]
                d[j] = (1.0 - alpha) * d[j - 1] + alpha * d[j] #(1.0 - alpha) * d[j - 1] + alpha * d[j]


        return d[self.p]
    
    def compute_prediction_first_derivativ(self, x, k):
        """
        Evaluates S(x).

        Args
        ----
        k: index of knot interval that contains x
        x: position
        t: array of knot positions, needs to be padded as described above
        c: array of control points
        p: degree of B-spline
        """
        q = [self.p * (self.c[j+k-self.p+1] - self.c[j+k-self.p]) / (self.t[j+k+1] - self.t[j+k-self.p+1]) for j in range(0, self.p)]

        for r in range(1, self.p):
            for j in range(self.p-1, r-1, -1):
                right = j+1+k-r
                left = j+k-(self.p-1)
                alpha = (x - self.t[left]) / (self.t[right] - self.t[left])
                q[j] = (1.0 - alpha) * q[j-1] + alpha * q[j]
                
        return q[self.p-1]

# Runs DeBoor Algorithm vectorized using functorch
# (much faster (30 sec vs 8:30min) alternative to using a for loop or list comprehension)
def run_deBoor(x, t, c, p, d):
    deBoor_obj = deBoor(t=t, c=c, p=p)#, d=d)
    if d==0:
        deBorr_func_vectorized = torch.vmap(deBoor_obj.compute_prediction) #functorch.vmap(deBoor_obj.compute_prediction)
    elif d==1:
        deBorr_func_vectorized = torch.vmap(deBoor_obj.compute_prediction_first_derivativ) #functorch.vmap(deBoor_obj.compute_prediction_first_derivativ)
    k = deBoor_obj.compute_k(x)

    return deBorr_func_vectorized(torch.unsqueeze(x,0), torch.unsqueeze(k,0)).squeeze()


# Bspline Prediction using the deBoor algorithm
def bspline_prediction(params_a, input_a, degree, spline_range, monotonically_increasing=False, derivativ=0, return_penalties=False, calc_method="Naive_Basis",#'Naive_Basis', #before: deBoor 
                       span_factor=0.1, span_restriction="reluler",
                       covariate=False, params_covariate=False, covaraite_effect="multiplicativ",
                       penalize_towards=0, order=3): #device=None

    # Adjust polynomial range to be a bit wider
    # Empirically found that this helps with the fit
    #spline_range = adjust_ploynomial_range(spline_range, span_factor)

    #print("spline_range after:", spline_range)  
    
    #print("params_a.size()", params_a.size())
    if monotonically_increasing:
        params_restricted = restrict_parameters(params_a, #.contiguous().unsqueeze(1), 
                                                covariate=covariate, degree=degree, monotonically_increasing=monotonically_increasing,device=None)#.squeeze(1)
    else:
        params_restricted = params_a
        
    #params_restricted = params_a.clone().contiguous()
    input_a_clone = input_a.clone().contiguous()
    if order == 2:
        n = degree + 1
    elif order == 3:
        n = degree + 2

    distance_between_knots = (spline_range[1] - spline_range[0]) * (1 + span_factor) / (n - 1)

    knots = torch.linspace(spline_range[0] * (1 + span_factor) - order * distance_between_knots,
                           spline_range[1] * (1 + span_factor) + order * distance_between_knots,
                           n + 4, dtype=torch.float32, device=input_a.device)

    #input_a_clone = (torch.sigmoid(input_a_clone/((spline_range[1] - spline_range[0])) * 10) - 0.5) * (spline_range[1] - spline_range[0])/2
    #input_a_clone = custom_sigmoid(input=input_a_clone, min=spline_range[0], max=spline_range[1])

    if span_restriction == "sigmoid":
        input_a_clone = custom_sigmoid(input_a_clone, spline_range)
    elif span_restriction == "reluler":
        reluler = ReLULeR(spline_range)
        input_a_clone = reluler.forward(input_a_clone)
    else:
        #print("span_restriction is not used!!!!!!!!!")
        pass

    #calc_method = "Naive"

    if calc_method == "deBoor":
        
        if params_restricted.dim() == 2:
            params_restricted = params_restricted.squeeze(1)
        
        prediction = run_deBoor(x=input_a_clone,
                                t=knots,
                                c=params_restricted,
                                p=order,
                                d=derivativ)
    elif calc_method == "Naive":
        prediction = Naive(x=input_a_clone,
                           t=knots,
                           c=params_restricted,
                           p=order,
                           d=derivativ
                           )

    elif calc_method == "Naive_vmap":
        prediction = run_Naive_vmap(x=input_a_clone,
                                    t=knots,
                                    c=params_restricted,
                                    p=order)
    elif calc_method == "Naive_Basis":
        basis = Naive_Basis(x=input_a_clone, 
                                 spline_range=spline_range, 
                                 degree=degree, 
                                 span_factor=span_factor,
                                 derivativ=derivativ,
                                 order=order)
        
        prediction = torch.sum(basis * params_restricted.unsqueeze(0), (1)) #torch.matmul(basis, params_restricted) # Its the same as used in the transformation with stored_basis
        #torch.matmul(basis, params_restricted) == torch.sum(basis * params_restricted.unsqueeze(0), (1))

    # Adding Covariate in a GAM manner
    if covariate is not False:
        params_covariate_restricted = params_covariate.clone().contiguous()

        #if dev is not False:
        #    params_covariate_restricted.to(dev)

        knots_covariate = torch.linspace(0 - order * 1,
                                         1 + order * 1,
                                         n + 4, dtype=torch.float32, device=input_a.device)

        prediction_covariate = run_deBoor(x=covariate,
                                t=knots_covariate,
                                c=params_covariate_restricted,
                                p=order)

        #if covaraite_effect == "additive":
        #    prediction = prediction + prediction_covariate
        ##elif covaraite_effect == "multiplicative":
        #    prediction = prediction * prediction_covariate

        prediction = prediction * prediction_covariate


    #if prediction.isnan().sum() > 0:
    #   print("prediction contains NaNs")
    #   print("prediction is nan:",prediction[prediction.isnan()])
    #   print("knots:",knots)
    
    # Check with this
    #a=0
    #for i in range(params_restricted.size(1)):
    #    a = a + sum(torch.diff(params_restricted[:,i],n=2)**2)
    if return_penalties:
        second_order_ridge_pen = torch.sum(torch.diff(params_restricted,n=2,dim=0)**2)
        first_order_ridge_pen = torch.sum(torch.diff(params_restricted,n=1,dim=0)**2)
        param_ridge_pen = torch.sum((params_restricted - penalize_towards)**2) #penalize_towards

        # Adding Covariate parameter penalisation values
        if covariate is not False:
            second_order_ridge_pen += torch.sum(torch.diff(params_covariate_restricted, n=2, dim=0) ** 2)
            first_order_ridge_pen += torch.sum(torch.diff(params_covariate_restricted, n=1, dim=0) ** 2)
            param_ridge_pen += torch.sum((params_covariate_restricted - penalize_towards) ** 2) #penalize_towards


        return prediction, second_order_ridge_pen, first_order_ridge_pen, param_ridge_pen
    else:
        return prediction