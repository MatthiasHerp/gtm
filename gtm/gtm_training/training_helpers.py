import time
import copy
import torch
#import warnings
#from torch import nn
import numpy as np
#from torch.distributions import Normal, Laplace
#import matplotlib.pyplot as plt
from torch import optim
from tqdm import tqdm
#import seaborn as sns
#from pytorch_lbfgs.LBFGS import LBFGS, FullBatchLBFGS
from torch.optim import LBFGS
import copy
import pickle
import os

#from torch_ema import ExponentialMovingAverage

class EarlyStopper:
    def __init__(self, patience=1, min_delta=0., global_min_loss=-np.inf):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_loss = torch.inf
        self.global_min_loss = global_min_loss

    def early_stop(self, current_loss, model, verbose=False):
        if current_loss < self.min_loss - self.min_delta:
            self.best_model_state = copy.deepcopy(model.state_dict())
            self.min_loss = current_loss
            self.counter = 0

        else:
            self.counter += 1

            if self.counter >= self.patience:
                if verbose:
                    print("Early stopping due to no improvement in loss for",self.patience,"iterations")
                return True

        if current_loss < self.global_min_loss:
            print("Early stopping due to global minimum loss reached")
            return True

        return False
    
import math
def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, num_cycles=0.5, last_epoch=-1):
    """ Create a schedule with a learning rate that decreases following the
    values of the cosine function between 0 and ⁠ pi * cycles ⁠ after a warmup
    period during which it increases linearly between 0 and 1.
    """

    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch)

    
def train(model, train_dataloader, validate_dataloader=False, train_covariates=False, validate_covariates=False, penalty_params=torch.FloatTensor([0,0,0,0]), lambda_penalty_params=False, learning_rate=1, iterations=2000, verbose=True, patience=5, min_delta=1e-7, global_min_loss=-np.inf,
          optimizer='LBFGS', lambda_penalty_mode="square", objective_type="negloglik", #ema_decay=False, 
          adaptive_lasso_weights_matrix=False, 
          max_batches_per_iter = False):
    # max_batches_per_iter infos
    # then use random sampling data_loader  
    # always 1 for validation data
    
    start = time.time()
  
    opt = LBFGS(model.parameters(), lr=learning_rate, history_size=1, line_search_fn="strong_wolfe", max_iter=1, max_eval=40)

    #if ema_decay is not False:
    #    ema = ExponentialMovingAverage(model.parameters(), decay=ema_decay)

    def closure():
        opt.zero_grad()
        return_dict_model_objective  = model.training_objective(y_train, penalty_params, lambda_penalty_params=lambda_penalty_params, train_covariates=train_covariates, lambda_penalty_mode=lambda_penalty_mode, objective_type=objective_type, adaptive_lasso_weights_matrix=adaptive_lasso_weights_matrix) # use the `objective` function
        
        loss = return_dict_model_objective["loss_with_penalties"].mean()
        
        if verbose == True:
            print("current_loss:",loss)

        # Note to myself:
        # retain Graph makes it much slower in negloglik training and retaining the graph increases memory usage more and more as iterations increase. 
        # Further it also seems to make the cuda usage more instable in the sense of having spikes, now it runs super smooth and with literaly no memory usage
        if objective_type == "score_matching":

            loss.backward(retain_graph=True)
        else:
            loss.backward()
        return loss

    early_stopper = EarlyStopper(patience=patience, min_delta=min_delta, global_min_loss=global_min_loss)

    if optimizer == "Adam":
        opt = optim.Adam(model.parameters(), lr = learning_rate, weight_decay=0)
        scheduler = get_cosine_schedule_with_warmup(opt, num_warmup_steps=5, num_training_steps=iterations, num_cycles=0.5, last_epoch=-1) ##3,4,5 warmup steps machen 

    loss_list = []

    if validate_dataloader is not False:
        with open("model.pkl", "wb") as f:
            pickle.dump(model, f)
        with open("model.pkl", "rb") as f:
            model_val = pickle.load(f)
        os.remove("model.pkl")
        if model_val.num_trans_layers > 0:
            model_val.transformation.multivariate_basis = False
            model_val.transformation.multivariate_basis_derivativ_1 = False
            
    for i in tqdm(range(iterations)):
        number_iterations = i
        num_processed_batches = 0
        for y_train in train_dataloader:
            num_processed_batches += 1
            
            y_train = y_train.to(model.device)
            
            if optimizer == "Adam":
                opt.zero_grad()
                return_dict_model_objective  = model.training_objective(y_train, penalty_params, lambda_penalty_params=lambda_penalty_params, train_covariates=train_covariates, lambda_penalty_mode=lambda_penalty_mode, objective_type=objective_type, adaptive_lasso_weights_matrix=adaptive_lasso_weights_matrix) # use the `objective` function
                
                loss = return_dict_model_objective["loss_with_penalties"].mean()
                loss.backward()
                opt.step()
                scheduler.step()
                current_loss = loss
                if verbose == True:
                    print("current_loss:",loss)
            elif optimizer == "LBFGS":
                current_loss = opt.step(closure)
        
            #if ema_decay is not False:
            #    ema.update()
            
            if max_batches_per_iter is not False and num_processed_batches >= max_batches_per_iter:
                break
            
        loss_list.append(current_loss.item())

        if validate_dataloader is not False:
            y_validate = next(iter(validate_dataloader))
            y_validate = y_validate.to(model.device)
            model_val.load_state_dict(model.state_dict())
            
            if objective_type is "negloglik":
                with torch.no_grad():
                    return_dict_model_objective_val  = model_val.training_objective(y_validate, penalty_params, lambda_penalty_params=lambda_penalty_params, train_covariates=validate_covariates, lambda_penalty_mode=lambda_penalty_mode, objective_type=objective_type, adaptive_lasso_weights_matrix=adaptive_lasso_weights_matrix)
                    current_loss_val = return_dict_model_objective_val["loss_without_penalties"].mean() # No penalties as on validation set
            else:
                return_dict_model_objective_val  = model.training_objective(y_validate, penalty_params, lambda_penalty_params=lambda_penalty_params, train_covariates=validate_covariates, lambda_penalty_mode=lambda_penalty_mode, objective_type=objective_type, adaptive_lasso_weights_matrix=adaptive_lasso_weights_matrix)
                current_loss_val = return_dict_model_objective_val["loss_without_penalties"].mean() # No penalties as on validation set

            if verbose:
                print("current_loss_val: ",current_loss_val.item())

            if early_stopper.early_stop(current_loss_val, model):
                if verbose:
                    print("Early Stop at iteration", i, "with minimal loss", early_stopper.min_loss, "and patience", patience,
                        "and min_delta", min_delta)
                # early stop means best model was at current iteration - patience
                number_iterations = number_iterations-patience
                break
        else:
            if early_stopper.early_stop(current_loss, model):
                if verbose:
                    print("Early Stop at iteration", i, "with minimal loss", early_stopper.min_loss, "and patience", patience,
                        "and min_delta", min_delta)
                # early stop means best model was at current iteration - patience
                number_iterations = number_iterations - patience
                break
        
    # Return the best model which is not necessarily the last model
    model.load_state_dict(early_stopper.best_model_state)
    
    # Rerun model at the end to get final penalties
    return_dict_model_training  = model.training_objective(y_train, penalty_params, lambda_penalty_params=lambda_penalty_params, train_covariates=train_covariates, lambda_penalty_mode=lambda_penalty_mode, objective_type=objective_type, adaptive_lasso_weights_matrix=adaptive_lasso_weights_matrix)
    return_dict_model_training["loss_list_training"] = loss_list
    return_dict_model_training["number_iterations"] = number_iterations
    
    end = time.time()

    training_time = end - start
    return_dict_model_training["training_time"] = training_time
    
    return return_dict_model_training
    
    
        
def if_float_create_lambda_penalisation_matrix(lambda_penalty_params, num_vars):

    lambda_penalty_params = torch.tensor(lambda_penalty_params, dtype=torch.float32)
    if lambda_penalty_params.size() == torch.Size([]):
        lambda_penalty_params = torch.tril(lambda_penalty_params.repeat(num_vars, num_vars)) - torch.eye( #before got warning with torch.tensor(lambda_penalty_params) in the tril
            num_vars, num_vars) * lambda_penalty_params

    return lambda_penalty_params