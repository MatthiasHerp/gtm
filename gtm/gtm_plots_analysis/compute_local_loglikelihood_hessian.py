import torch

def compute_local_loglikelihood_hessian(self,
                              y,
                              copula_only=True):
        if copula_only == True:
            y = self.after_transformation(y)
            store = self.num_trans_layers
            self.num_trans_layers = 0
        
        samples = y.detach().clone().requires_grad_(True)  # (N, D)
        N, D = samples.shape

        score = self.log_likelihood(samples).sum()
        grad = torch.autograd.grad(score, samples, create_graph=True)[0]  # Shape: (1, D)

        hessian_vmap = [torch.autograd.grad(grad.sum(0)[d], samples, create_graph=True)[0] for d in range(D)]  # Shape: (1, D)

        hessian_vmap = torch.stack(hessian_vmap).permute(1, 0, 2)
        
        if copula_only == True: 
            self.num_trans_layers = store
        
        return hessian_vmap
    
    
######## Possible test to compare to: two implementations

# # Assume samples is (N, D)
# samples = z_tilde_train.detach().clone().requires_grad_(True)
# N, D = samples.shape# 
# loglik = model_decorr.log_likelihood(samples).sum()# 
# def log_like_fn(x):
#     # x is 1D tensor of shape (D,), requires_grad=True
#     x = x.unsqueeze(0)  # convert to shape (1, D) for your model, if needed
#     return model_decorr.log_likelihood(x).sum()
#     
# hessians_lc = [torch.autograd.functional.hessian(log_like_fn, samples[n]) for n in range(z_tilde_train.size(0))]
# # hessians is now a list of [D x D] tensors, one per sample
# hessians_lc = torch.stack(hessians_lc)  # shape: (N, D, D)

# # Assume samples is (N, D)
# samples = z_tilde_train.detach().clone().requires_grad_(True)
# N, D = samples.shape
# 
# loglik = model_decorr.log_likelihood(samples).sum()
# 
# hessians = []
# 
# for n in range(z_tilde_train.size(0)):
#     def log_like_fn(x):
#         # x is 1D tensor of shape (D,), requires_grad=True
#         x = x.unsqueeze(0)  # convert to shape (1, D) for your model, if needed
#         return model_decorr.log_likelihood(x).sum()
#     
#     hess = torch.autograd.functional.hessian(log_like_fn, samples[n])
#     hessians.append(hess.detach())
# # hessians is now a list of [D x D] tensors, one per sample
# hessians = torch.stack(hessians)  # shape: (N, D, D)

# (hessian_vmap - hessians_lc).abs().max()
# (hessian_vmap - hessians).abs().max()
        