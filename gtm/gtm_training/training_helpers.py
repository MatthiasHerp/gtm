# import seaborn as sns
# from pytorch_lbfgs.LBFGS import LBFGS, FullBatchLBFGS
# from torch.optim import LBFGS
import copy
import os
import pickle
import time
import re
from functools import reduce
from typing import Literal, TYPE_CHECKING
from math import pi, cos, isnan, log, expm1, exp
# import warnings
# from torch import nn
import numpy as np
import torch
import torch.nn as nn

# from torch.distributions import Normal, Laplace
# import matplotlib.pyplot as plt
from torch import optim, Tensor
from torch.optim.lr_scheduler import LambdaLR
from torch.optim import Optimizer, Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from torch.amp import autocast
from tqdm import tqdm
from gtm.gtm_training.training_bayes import VI_Model, VariationalGamma, TauNode, TauPack

if TYPE_CHECKING:
    from ..gtm_model.gtm import GTM # type-only; no runtime import

# from torch_ema import ExponentialMovingAverage


########################################################################################################################
# Solves this error: File "/Users/matthiasherp/anaconda3/envs/mctm_pytorch/lib/python3.11/site-packages/torch/optim/lbfgs.py", line 262, in _gather_flat_grad
#    view = p.grad.view(-1)
#           ^^^^^^^^^^^^^^^
# RuntimeError: view size is not compatible with input tensor's size and stride (at least one dimension spans across two contiguous subspaces). Use .reshape(...) instead.
# (mctm_pytorch) (base)


__all__ = ["LBFGS"]


def _cubic_interpolate(x1, f1, g1, x2, f2, g2, bounds=None):
    # ported from https://github.com/torch/optim/blob/master/polyinterp.lua
    # Compute bounds of interpolation area
    if bounds is not None:
        xmin_bound, xmax_bound = bounds
    else:
        xmin_bound, xmax_bound = (x1, x2) if x1 <= x2 else (x2, x1)

    # Code for most common case: cubic interpolation of 2 points
    #   w/ function and derivative values for both
    # Solution in this case (where x2 is the farthest point):
    #   d1 = g1 + g2 - 3*(f1-f2)/(x1-x2);
    #   d2 = sqrt(d1^2 - g1*g2);
    #   min_pos = x2 - (x2 - x1)*((g2 + d2 - d1)/(g2 - g1 + 2*d2));
    #   t_new = min(max(min_pos,xmin_bound),xmax_bound);
    d1 = g1 + g2 - 3 * (f1 - f2) / (x1 - x2)
    d2_square = d1**2 - g1 * g2
    if d2_square >= 0:
        d2 = d2_square.sqrt()
        if x1 <= x2:
            min_pos = x2 - (x2 - x1) * ((g2 + d2 - d1) / (g2 - g1 + 2 * d2))
        else:
            min_pos = x1 - (x1 - x2) * ((g1 + d2 - d1) / (g1 - g2 + 2 * d2))
        return min(max(min_pos, xmin_bound), xmax_bound)
    else:
        return (xmin_bound + xmax_bound) / 2.0


def _strong_wolfe(
    obj_func, x, t, d, f, g, gtd, c1=1e-4, c2=0.9, tolerance_change=1e-9, max_ls=25
):
    # ported from https://github.com/torch/optim/blob/master/lswolfe.lua
    d_norm = d.abs().max()
    g = g.clone(memory_format=torch.contiguous_format)
    # evaluate objective and gradient using initial step
    f_new, g_new = obj_func(x, t, d)
    ls_func_evals = 1
    gtd_new = g_new.dot(d)

    # bracket an interval containing a point satisfying the Wolfe criteria
    t_prev, f_prev, g_prev, gtd_prev = 0, f, g, gtd
    done = False
    ls_iter = 0
    while ls_iter < max_ls:
        # check conditions
        if f_new > (f + c1 * t * gtd) or (ls_iter > 1 and f_new >= f_prev):
            bracket = [t_prev, t]
            bracket_f = [f_prev, f_new]
            bracket_g = [g_prev, g_new.clone(memory_format=torch.contiguous_format)]
            bracket_gtd = [gtd_prev, gtd_new]
            break

        if abs(gtd_new) <= -c2 * gtd:
            bracket = [t]
            bracket_f = [f_new]
            bracket_g = [g_new]
            done = True
            break

        if gtd_new >= 0:
            bracket = [t_prev, t]
            bracket_f = [f_prev, f_new]
            bracket_g = [g_prev, g_new.clone(memory_format=torch.contiguous_format)]
            bracket_gtd = [gtd_prev, gtd_new]
            break

        # interpolate
        min_step = t + 0.01 * (t - t_prev)
        max_step = t * 10
        tmp = t
        t = _cubic_interpolate(
            t_prev, f_prev, gtd_prev, t, f_new, gtd_new, bounds=(min_step, max_step)
        )

        # next step
        t_prev = tmp
        f_prev = f_new
        g_prev = g_new.clone(memory_format=torch.contiguous_format)
        gtd_prev = gtd_new
        f_new, g_new = obj_func(x, t, d)
        ls_func_evals += 1
        gtd_new = g_new.dot(d)
        ls_iter += 1

    # reached max number of iterations?
    if ls_iter == max_ls:
        bracket = [0, t]
        bracket_f = [f, f_new]
        bracket_g = [g, g_new]

    # zoom phase: we now have a point satisfying the criteria, or
    # a bracket around it. We refine the bracket until we find the
    # exact point satisfying the criteria
    insuf_progress = False
    # find high and low points in bracket
    low_pos, high_pos = (0, 1) if bracket_f[0] <= bracket_f[-1] else (1, 0)
    while not done and ls_iter < max_ls:
        # line-search bracket is so small
        if abs(bracket[1] - bracket[0]) * d_norm < tolerance_change:
            break

        # compute new trial value
        t = _cubic_interpolate(
            bracket[0],
            bracket_f[0],
            bracket_gtd[0],
            bracket[1],
            bracket_f[1],
            bracket_gtd[1],
        )

        # test that we are making sufficient progress:
        # in case `t` is so close to boundary, we mark that we are making
        # insufficient progress, and if
        #   + we have made insufficient progress in the last step, or
        #   + `t` is at one of the boundary,
        # we will move `t` to a position which is `0.1 * len(bracket)`
        # away from the nearest boundary point.
        eps = 0.1 * (max(bracket) - min(bracket))
        if min(max(bracket) - t, t - min(bracket)) < eps:
            # interpolation close to boundary
            if insuf_progress or t >= max(bracket) or t <= min(bracket):
                # evaluate at 0.1 away from boundary
                if abs(t - max(bracket)) < abs(t - min(bracket)):
                    t = max(bracket) - eps
                else:
                    t = min(bracket) + eps
                insuf_progress = False
            else:
                insuf_progress = True
        else:
            insuf_progress = False

        # Evaluate new point
        f_new, g_new = obj_func(x, t, d)
        ls_func_evals += 1
        gtd_new = g_new.dot(d)
        ls_iter += 1

        if f_new > (f + c1 * t * gtd) or f_new >= bracket_f[low_pos]:
            # Armijo condition not satisfied or not lower than lowest point
            bracket[high_pos] = t
            bracket_f[high_pos] = f_new
            bracket_g[high_pos] = g_new.clone(memory_format=torch.contiguous_format)
            bracket_gtd[high_pos] = gtd_new
            low_pos, high_pos = (0, 1) if bracket_f[0] <= bracket_f[1] else (1, 0)
        else:
            if abs(gtd_new) <= -c2 * gtd:
                # Wolfe conditions satisfied
                done = True
            elif gtd_new * (bracket[high_pos] - bracket[low_pos]) >= 0:
                # old high becomes new low
                bracket[high_pos] = bracket[low_pos]
                bracket_f[high_pos] = bracket_f[low_pos]
                bracket_g[high_pos] = bracket_g[low_pos]
                bracket_gtd[high_pos] = bracket_gtd[low_pos]

            # new point becomes new low
            bracket[low_pos] = t
            bracket_f[low_pos] = f_new
            bracket_g[low_pos] = g_new.clone(memory_format=torch.contiguous_format)
            bracket_gtd[low_pos] = gtd_new

    # return stuff
    t = bracket[low_pos]
    f_new = bracket_f[low_pos]
    g_new = bracket_g[low_pos]
    return f_new, g_new, t, ls_func_evals


class LBFGS(Optimizer):
    """Implements L-BFGS algorithm, heavily inspired by `minFunc
    <https://www.cs.ubc.ca/~schmidtm/Software/minFunc.html>`_.

    .. warning::
        This optimizer doesn't support per-parameter options and parameter
        groups (there can be only one).

    .. warning::
        Right now all parameters have to be on a single device. This will be
        improved in the future.

    .. note::
        This is a very memory intensive optimizer (it requires additional
        ``param_bytes * (history_size + 1)`` bytes). If it doesn't fit in memory
        try reducing the history size, or use a different algorithm.

    Args:
        lr (float): learning rate (default: 1)
        max_iter (int): maximal number of iterations per optimization step
            (default: 20)
        max_eval (int): maximal number of function evaluations per optimization
            step (default: max_iter * 1.25).
        tolerance_grad (float): termination tolerance on first order optimality
            (default: 1e-5).
        tolerance_change (float): termination tolerance on function
            value/parameter changes (default: 1e-9).
        history_size (int): update history size (default: 100).
        line_search_fn (str): either 'strong_wolfe' or None (default: None).
    """

    def __init__(
        self,
        params,
        lr=1,
        max_iter=20,
        max_eval=None,
        tolerance_grad=1e-7,
        tolerance_change=1e-9,
        history_size=100,
        line_search_fn=None,
    ):
        if max_eval is None:
            max_eval = max_iter * 5 // 4
        defaults = dict(
            lr=lr,
            max_iter=max_iter,
            max_eval=max_eval,
            tolerance_grad=tolerance_grad,
            tolerance_change=tolerance_change,
            history_size=history_size,
            line_search_fn=line_search_fn,
        )
        super(LBFGS, self).__init__(params, defaults)

        if len(self.param_groups) != 1:
            raise ValueError(
                "LBFGS doesn't support per-parameter options " "(parameter groups)"
            )

        self._params = self.param_groups[0]["params"]
        self._numel_cache = None

    def _numel(self):
        if self._numel_cache is None:
            self._numel_cache = reduce(
                lambda total, p: total + p.numel(), self._params, 0
            )
        return self._numel_cache

    def _gather_flat_grad(self):
        views = []
        for p in self._params:
            if p.grad is None:
                view = p.new(p.numel()).zero_()
            elif p.grad.is_sparse:
                view = p.grad.to_dense().contiguous().view(-1)
            else:
                view = p.grad.contiguous().view(-1)
            views.append(view)
        return torch.cat(views, 0)

    def _add_grad(self, step_size, update):
        offset = 0
        for p in self._params:
            numel = p.numel()
            # view as to avoid deprecated pointwise semantics
            p.add_(update[offset : offset + numel].view_as(p), alpha=step_size)
            offset += numel
        assert offset == self._numel()

    def _clone_param(self):
        return [p.clone(memory_format=torch.contiguous_format) for p in self._params]

    def _set_param(self, params_data):
        for p, pdata in zip(self._params, params_data):
            p.copy_(pdata)

    def _directional_evaluate(self, closure, x, t, d):
        self._add_grad(t, d)
        loss = float(closure())
        flat_grad = self._gather_flat_grad()
        self._set_param(x)
        return loss, flat_grad

    @torch.no_grad()
    def step(self, closure):
        """Performs a single optimization step.

        Args:
            closure (Callable): A closure that reevaluates the model
                and returns the loss.
        """
        assert len(self.param_groups) == 1

        # Make sure the closure is always called with grad enabled
        closure = torch.enable_grad()(closure)

        group = self.param_groups[0]
        lr = group["lr"]
        max_iter = group["max_iter"]
        max_eval = group["max_eval"]
        tolerance_grad = group["tolerance_grad"]
        tolerance_change = group["tolerance_change"]
        line_search_fn = group["line_search_fn"]
        history_size = group["history_size"]

        # NOTE: LBFGS has only global state, but we register it as state for
        # the first param, because this helps with casting in load_state_dict
        state = self.state[self._params[0]]
        state.setdefault("func_evals", 0)
        state.setdefault("n_iter", 0)

        # evaluate initial f(x) and df/dx
        orig_loss = closure()
        loss = float(orig_loss)
        current_evals = 1
        state["func_evals"] += 1

        flat_grad = self._gather_flat_grad()
        opt_cond = flat_grad.abs().max() <= tolerance_grad

        # optimal condition
        if opt_cond:
            return orig_loss

        # tensors cached in state (for tracing)
        d = state.get("d")
        t = state.get("t")
        old_dirs = state.get("old_dirs")
        old_stps = state.get("old_stps")
        ro = state.get("ro")
        H_diag = state.get("H_diag")
        prev_flat_grad = state.get("prev_flat_grad")
        prev_loss = state.get("prev_loss")

        n_iter = 0
        # optimize for a max of max_iter iterations
        while n_iter < max_iter:
            # keep track of nb of iterations
            n_iter += 1
            state["n_iter"] += 1

            ############################################################
            # compute gradient descent direction
            ############################################################
            if state["n_iter"] == 1:
                d = flat_grad.neg()
                old_dirs = []
                old_stps = []
                ro = []
                H_diag = 1
            else:
                # do lbfgs update (update memory)
                y = flat_grad.sub(prev_flat_grad)
                s = d.mul(t)
                ys = y.dot(s)  # y*s
                if ys > 1e-10:
                    # updating memory
                    if len(old_dirs) == history_size:
                        # shift history by one (limited-memory)
                        old_dirs.pop(0)
                        old_stps.pop(0)
                        ro.pop(0)

                    # store new direction/step
                    old_dirs.append(y)
                    old_stps.append(s)
                    ro.append(1.0 / ys)

                    # update scale of initial Hessian approximation
                    H_diag = ys / y.dot(y)  # (y*y)

                # compute the approximate (L-BFGS) inverse Hessian
                # multiplied by the gradient
                num_old = len(old_dirs)

                if "al" not in state:
                    state["al"] = [None] * history_size
                al = state["al"]

                # iteration in L-BFGS loop collapsed to use just one buffer
                q = flat_grad.neg()
                for i in range(num_old - 1, -1, -1):
                    al[i] = old_stps[i].dot(q) * ro[i]
                    q.add_(old_dirs[i], alpha=-al[i])

                # multiply by initial Hessian
                # r/d is the final direction
                d = r = torch.mul(q, H_diag)
                for i in range(num_old):
                    be_i = old_dirs[i].dot(r) * ro[i]
                    r.add_(old_stps[i], alpha=al[i] - be_i)

            if prev_flat_grad is None:
                prev_flat_grad = flat_grad.clone(memory_format=torch.contiguous_format)
            else:
                prev_flat_grad.copy_(flat_grad)
            prev_loss = loss

            ############################################################
            # compute step length
            ############################################################
            # reset initial guess for step size
            if state["n_iter"] == 1:
                t = min(1.0, 1.0 / flat_grad.abs().sum()) * lr
            else:
                t = lr

            # directional derivative
            gtd = flat_grad.dot(d)  # g * d

            # directional derivative is below tolerance
            if gtd > -tolerance_change:
                break

            # optional line search: user function
            ls_func_evals = 0
            if line_search_fn is not None:
                # perform line search, using user function
                if line_search_fn != "strong_wolfe":
                    raise RuntimeError("only 'strong_wolfe' is supported")
                else:
                    x_init = self._clone_param()

                    def obj_func(x, t, d):
                        return self._directional_evaluate(closure, x, t, d)

                    loss, flat_grad, t, ls_func_evals = _strong_wolfe(
                        obj_func, x_init, t, d, loss, flat_grad, gtd
                    )
                self._add_grad(t, d)
                opt_cond = flat_grad.abs().max() <= tolerance_grad
            else:
                # no line search, simply move with fixed-step
                self._add_grad(t, d)
                if n_iter != max_iter:
                    # re-evaluate function only if not in last iteration
                    # the reason we do this: in a stochastic setting,
                    # no use to re-evaluate that function here
                    with torch.enable_grad():
                        loss = float(closure())
                    flat_grad = self._gather_flat_grad()
                    opt_cond = flat_grad.abs().max() <= tolerance_grad
                    ls_func_evals = 1

            # update func eval
            current_evals += ls_func_evals
            state["func_evals"] += ls_func_evals

            ############################################################
            # check conditions
            ############################################################
            if n_iter == max_iter:
                break

            if current_evals >= max_eval:
                break

            # optimal condition
            if opt_cond:
                break

            # lack of progress
            if d.mul(t).abs().max() <= tolerance_change:
                break

            if abs(loss - prev_loss) < tolerance_change:
                break

        state["d"] = d
        state["t"] = t
        state["old_dirs"] = old_dirs
        state["old_stps"] = old_stps
        state["ro"] = ro
        state["H_diag"] = H_diag
        state["prev_flat_grad"] = prev_flat_grad
        state["prev_loss"] = prev_loss

        return orig_loss


########################################################################################################################


class EarlyStopper:
    def __init__(self, patience=1, min_delta=0.0, global_min_loss=-np.inf):
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
                    print(
                        "Early stopping due to no improvement in loss for",
                        self.patience,
                        "iterations",
                    )
                return True

        if current_loss < self.global_min_loss:
            print("Early stopping due to global minimum loss reached")
            return True

        return False


def get_cosine_schedule_with_warmup(optimizer: Adam,
                                    num_warmup_steps: int,
                                    num_training_steps: int,
                                    num_cycles: float=0.5,
                                    last_epoch: int=-1
                                    ) -> LambdaLR:
    """Create a schedule with a learning rate that decreases following the
    values of the cosine function between 0 and ⁠ pi * cycles ⁠ after a warmup
    period during which it increases linearly between 0 and 1.
    """
    
    def lr_lambda(current_step) -> float:
        
        current_step:float = float(current_step)
        
        if current_step < num_warmup_steps:
            
            return current_step/ max(1, num_warmup_steps)
            
        progress: float = (current_step - num_warmup_steps) / max(
            1, num_training_steps - num_warmup_steps
        )
        
        return max(0.0, 0.5 * (1.0 + cos(pi * num_cycles * 2.0 * progress)))

    return LambdaLR(optimizer=optimizer, lr_lambda=lr_lambda, last_epoch=last_epoch)


def train_freq(
    model: "GTM",
    train_dataloader: DataLoader,
    validate_dataloader: DataLoader|bool=False,
    train_covariates:DataLoader|bool=False,
    validate_covariates: DataLoader|bool=False,
    penalty_params=None,
    #param_ridge_pen_global, first_order_ridge_pen_global, second_order_ridge_pen_global, transformation_second_order_ridge_pen_global
    lambda_penalty_params: Tensor | bool=False,
    learning_rate: float=1.0,
    iterations: int=2000,
    verbose: bool=True,
    patience: int=5,
    min_delta: float=1e-7,
    global_min_loss: float=-np.inf,
    optimizer: Literal['LBFGS', 'Adam']="LBFGS",
    lambda_penalty_mode: Literal['square']="square",
    objective_type: Literal['negloglik']="negloglik",  # ema_decay=False,
    adaptive_lasso_weights_matrix: Tensor| bool=False,
    max_batches_per_iter: bool|int|None=False,
) -> dict[str, Tensor]:    # max_batches_per_iter infos
    # then use random sampling data_loader
    # always 1 for validation data

    penalty_params: Tensor = torch.FloatTensor([0, 0, 0, 0]) if penalty_params is None else penalty_params
    start: float = time.time()

    opt: LBFGS = LBFGS(
        params=model.parameters(),
        lr=learning_rate,
        history_size=1,
        line_search_fn="strong_wolfe",
        max_iter=1,
        max_eval=40,
    )

    # if ema_decay is not False:
    #    ema = ExponentialMovingAverage(model.parameters(), decay=ema_decay)

    def closure() -> Tensor:
        opt.zero_grad()
        return_dict_model_objective: dict[str, Tensor] = model.__training_objective__(
            samples=y_train,
            penalty_params=penalty_params,
            lambda_penalty_params=lambda_penalty_params,
            train_covariates=train_covariates,
            lambda_penalty_mode=lambda_penalty_mode,
            objective_type=objective_type,
            adaptive_lasso_weights_matrix=adaptive_lasso_weights_matrix,
        )  # use the `objective` function

        loss: Tensor = return_dict_model_objective["loss_with_penalties"].mean()

        if verbose:
            print("current_loss:", loss)

        # Note to myself:
        # retain Graph makes it much slower in negloglik training and retaining the graph increases memory usage more and more as iterations increase.
        # Further it also seems to make the cuda usage more instable in the sense of having spikes, now it runs super smooth and with literaly no memory usage
        if objective_type == "score_matching":

            loss.backward(retain_graph=True)
            
        else:
            loss.backward()
            
        return loss

    early_stopper: EarlyStopper = EarlyStopper(patience=patience, min_delta=min_delta, global_min_loss=global_min_loss)

    if optimizer == "Adam":
        opt: Adam = Adam(params=model.parameters(), lr=learning_rate, weight_decay=0)
        scheduler: LambdaLR = get_cosine_schedule_with_warmup(
            optimizer=opt,
            num_warmup_steps=5,
            num_training_steps=iterations,
            num_cycles=0.5,
            last_epoch=-1,
        )  ##3,4,5 warmup steps machen

    loss_list: list = []
    loss_list_val: list = []

    if validate_dataloader is not False:
        with open("model.pkl", "wb") as f:
            pickle.dump(model, f)
        with open("model.pkl", "rb") as f:
            model_val: "GTM" = pickle.load(f)
        os.remove("model.pkl")
        if model_val.num_trans_layers > 0:
            model_val.transformation.multivariate_basis = False
            model_val.transformation.multivariate_basis_derivativ_1 = False

    for i in tqdm(iterable=range(iterations)):
        number_iterations: int = i
        num_processed_batches:int = 0
        
        for y_train in train_dataloader:
            
            num_processed_batches += 1
            y_train: Tensor = y_train.to(device=model.device)

            if optimizer == "Adam":
                opt.zero_grad()
                return_dict_model_objective: dict[str, Tensor] = model.__training_objective__(
                    samples=y_train,
                    penalty_params=penalty_params,
                    lambda_penalty_params=lambda_penalty_params,
                    train_covariates=train_covariates,
                    lambda_penalty_mode=lambda_penalty_mode,
                    objective_type=objective_type,
                    adaptive_lasso_weights_matrix=adaptive_lasso_weights_matrix,
                )  # use the `objective` function

                loss: Tensor = return_dict_model_objective["loss_with_penalties"].mean()
                loss.backward()
                opt.step()
                scheduler.step()
                current_loss: Tensor = loss
                if verbose:
                    print("current_loss:", loss)
            elif optimizer == "LBFGS":
                current_loss = opt.step(closure)

            # if ema_decay is not False:
            #    ema.update()

            if max_batches_per_iter and num_processed_batches >= max_batches_per_iter:
                break

        loss_list.append(current_loss.item())

        if validate_dataloader is not False:
            y_validate: Tensor = next(iter(validate_dataloader))
            y_validate = y_validate.to(model.device, non_blocking=True)
            model_val.load_state_dict(model.state_dict())

            if objective_type == "negloglik":
                
                with torch.no_grad():
                    return_dict_model_objective_val: dict[str, Tensor] = model_val.__training_objective__(
                        samples=y_validate,
                        penalty_params=penalty_params,
                        lambda_penalty_params=lambda_penalty_params,
                        train_covariates=validate_covariates,
                        lambda_penalty_mode=lambda_penalty_mode,
                        objective_type=objective_type,
                        adaptive_lasso_weights_matrix=adaptive_lasso_weights_matrix,
                    )
                    current_loss_val: Tensor = return_dict_model_objective_val[
                        "loss_without_penalties"
                    ].mean()  # No penalties as on validation set
            else:
                return_dict_model_objective_val: dict[str, Tensor] = model.__training_objective__(
                    samples=y_validate,
                    penalty_params=penalty_params,
                    lambda_penalty_params=lambda_penalty_params,
                    train_covariates=validate_covariates,
                    lambda_penalty_mode=lambda_penalty_mode,
                    objective_type=objective_type,
                    adaptive_lasso_weights_matrix=adaptive_lasso_weights_matrix,
                )
                current_loss_val = return_dict_model_objective_val[
                    "loss_without_penalties"
                ].mean()  # No penalties as on validation set

            loss_list_val.append(current_loss_val.item())

            if verbose:
                print("current_loss_val: ", current_loss_val.item())

            if early_stopper.early_stop(current_loss=current_loss_val, model=model):
                if verbose:
                    print(
                        "Early Stop at iteration",
                        i,
                        "with minimal loss",
                        early_stopper.min_loss,
                        "and patience",
                        patience,
                        "and min_delta",
                        min_delta,
                    )
                # early stop means best model was at current iteration - patience
                number_iterations -= patience
                break
        else:
            if early_stopper.early_stop(current_loss, model):
                if verbose:
                    print(
                        "Early Stop at iteration",
                        i,
                        "with minimal loss",
                        early_stopper.min_loss,
                        "and patience",
                        patience,
                        "and min_delta",
                        min_delta,
                    )
                # early stop means best model was at current iteration - patience
                number_iterations -= patience
                break

    # Return the best model which is not necessarily the last model
    model.load_state_dict(early_stopper.best_model_state)

    # Rerun model at the end to get final penalties
    return_dict_model_training: dict[str, Tensor] = model.__training_objective__(
        samples=y_train,
        penalty_params=penalty_params,
        lambda_penalty_params=lambda_penalty_params,
        train_covariates=train_covariates,
        lambda_penalty_mode=lambda_penalty_mode,
        objective_type=objective_type,
        adaptive_lasso_weights_matrix=adaptive_lasso_weights_matrix,
    )
    return_dict_model_training["loss_list_training"] = loss_list
    return_dict_model_training["loss_list_validation"] = loss_list_val
    return_dict_model_training["number_iterations"] = number_iterations

    end: float = time.time()

    training_time: float = end - start
    return_dict_model_training["training_time"] = training_time

    return return_dict_model_training


@torch.no_grad()
def _evaluate_epoch(
    VI: 'VI_Model', 
    model: 'GTM',
    val_loader,
    hyper_T,
    hyper_D,
    sample_size_total,
    S_val=8, 
    seed=123,
    tau_nodes=None,
    use_tau_vi_now: bool=False,
    ):
    nn.Module.train(model, False)
    total_loglik_sum, nobs = 0.0, 0
    for y in val_loader:
        y = y.to(model.device, non_blocking = True)
        with autocast(str(model.device), dtype=torch.float16):
            logsum = VI.predictive_loglik_sum_batch(
                y_batch=y,
                model=model,
                hyperparameter_transformation=hyper_T,
                hyperparameter_decorrelation=hyper_D,
                S=S_val,
                sample_size=1,
                seed=seed,
                tau_nodes=tau_nodes,
                use_tau_vi_now=use_tau_vi_now,
            )
        total_loglik_sum += logsum
        nobs += y.shape[0]
    return total_loglik_sum / nobs  # per-observation ELPD (higher is better)


def _make_key_filter(patterns_include=None, patterns_exclude=None):
    inc = [re.compile(p) for p in (patterns_include or [])]
    exc = [re.compile(p) for p in (patterns_exclude or [])]
    def _keep(k: str) -> bool:
        if inc and not any(r.search(k) for r in inc):
            return False
        if any(r.search(k) for r in exc):
            return False
        return True
    return _keep


def _beta_kl_at(epoch: int, beta_kl_anneal_epochs, beta_kl_start, beta_min=1.1) -> float:
    t = min(1.0, epoch / max(1, beta_kl_anneal_epochs))
    return beta_kl_start * (1.0 - t) + beta_min * t

def _gamma_mean(a, b): return float(a) / max(float(b), 1e-12)

def _seed_q_from_mean(q, tau_current: float):
    # mean(q) = a_hat / b_hat -> fix a_hat, adjust b_hat to match current mean
    a_hat=q.a0 + 0.5 * q.rank_total
    b_hat=max(a_hat / float(tau_current), 1e-12)
    q.a_hat=float(a_hat)
    q.b_hat=float(b_hat)


def _softplus_inv(x: float, eps: float = 1e-8) -> float:
    # Numerically stable inverse softplus for positive x
    x = max(x, eps)
    return float(log(expm1(x)))

def _seed_q_from_eb(q, E_qf_total_mc: float):
    # EB fixed point: a_hat=a0+rank/2, b_hat=b0+0.5*E[qf]
    q.a_hat=float(q.a0 + 0.5 * q.rank_total)
    q.b_hat=float(q.b0 + 0.5 * float(E_qf_total_mc))
    
def _tau_times_qf_target(a, b, r_half, Eqf):
    # r_half = rank/2
    num = (a + r_half) * Eqf
    den = (b + 0.5 * Eqf)
    return float(num / max(den, 1e-12))

def _damped_step(prev, target, eta, band=0.20):
    """
    One-step EMA toward target with per-epoch change capped to [1-band, 1+band]× of the previous value.
    """
    prev = float(prev)
    target = float(target)
    raw = (1.0 - float(eta)) * prev + float(eta) * target
    lo = prev * (1.0 - band)
    hi = prev * (1.0 + band)
    return float(max(lo, min(hi, raw)))

def train_bayes(
    model: "GTM",
    train_dataloader,
    validate_dataloader=None,
    iterations: int = 100,
    lr: float = 1e-3,
    hyperparameters: dict | None = None,
    verbose: bool = True,
    max_batches_per_iter: int | bool = False,

    # --- NEW knobs ---
    mcmc_sample_train: int = 4,       # == S_train
    mcmc_sample_val: int = 8,         # == S_val (keep fixed!)
    mc_ramp_every: int | None = 20,   # every N epochs, double S_train; None to disable
    mc_ramp_max: int = 32,
    global_seed: int = 0,
    patience_val: int = 10,
    min_delta: float = 1e-4,
    rho_lr_multiplier: float = 1.0,
    sched_factor: float = 0.5,
    sched_patience: int = 5,
    sched_threshold: float = 1e-4,
    sched_cooldown: int = 1,
    sched_min_lr: float = 5e-5,

    # --- WARMING PHASES ---
    warm_tau_epochs: int = 5,         # from 3 → 5
    warm_sigma_epochs: int = 18,      # from 8 → 18

    # --- OPTIMIZATION ---
    beta_kl_start: float = 2.5,       # from 2.0 → 2.5
    beta_kl_anneal_epochs: int = 25,  # slower anneal
    use_empirical_bayes: bool = False,
    eb_warm_then_cavi: bool = True,   # EB for first warm_tau_epochs only
    band_tau4 = 0.20,
    band_decor = 0.15,
    
    # --- τ VI toggles (keep EB as default behavior) ---
    tau_vi_mode = "after_warm", #"off" | "after_warm" | "always"
    tau_kl_beta = 1.0,
    tau_vi_sigma_init = 0.25
    
):
    TAU4_FLOOR = 1e-3
    TAU1_FLOOR = 1e-2
    TAU2_FLOOR = 1e-2
    S_tau_monitor = 128  
    
    was_training = model.training
    nn.Module.train(model, False)
    N_total = len(train_dataloader.dataset)
    
    decor_present = not (model.number_decorrelation_layers == 0 or model.transform_only)
    
    
    # ------------------- key filter
    if model.number_decorrelation_layers == 0 or model.transform_only:
        parameters_patterns = [r"^transformation\.params\.\d+$"]
        patterns_exclude = [r"decor", r"rho_", r"optimizer", r"running_", r"num_batches_tracked"]
    else:
        parameters_patterns = [r"^transformation\.params\.\d+$",
                               r"^decorrelation_layers\.\d+\.params$"]
        patterns_exclude = [r"rho_", r"optimizer", r"running_", r"num_batches_tracked"]
    key_filter = _make_key_filter(patterns_include=parameters_patterns,
                                  patterns_exclude=patterns_exclude)

    # ------------------- VI, RNG
    K = model.transformation.priors.K_prior_RW2.to(model.device)
    num_margins = int(model.number_variables)
    VI = VI_Model(model=model, device=model.device, key_filter=key_filter).to(model.device)
    gen = torch.Generator(device=model.device); gen.manual_seed(global_seed); VI.set_rng(gen)

    print("Parameters to be affected by the bayesian approach:", [k for k, _ in VI._schema][:50])

    # ------------------- hyperparameters
    if hyperparameters is None:
        hyper_T = model.hyperparameter.get("transformation", {})
        hyper_D = model.hyperparameter.get("decorrelation", {})
    else:
        hyper_T = hyperparameters.get("transformation", {})
        hyper_D = hyperparameters.get("decorrelation", {})

    # ------------------- Transformation (τ4)
    nullspace_dim_T = int(hyper_T.get("nullspace_dim", 2))
    rank_per_margin = K.shape[0] - nullspace_dim_T
    rank_T_total = rank_per_margin * num_margins

    pen_term2_T = hyper_T.get("RW2", {})
    a_lambda = torch.as_tensor(pen_term2_T.get("tau_a", 1.1), device=model.device, dtype=torch.float32)
    b_lambda = torch.as_tensor(pen_term2_T.get("tau_b", 1e-6), device=model.device, dtype=torch.float32)
    print(f"[TRANSFORMATION] λ_a={a_lambda.item()} λ_b={b_lambda.item()}")

    def use_eb_now(epoch): return eb_warm_then_cavi and (epoch < warm_tau_epochs)

    # --- NEW: separate damping etas
    eta_tau4 = float(hyper_T.get("tau_4_update_eta", 0.25))     # default 0.25
    eta_tau1 = float(hyper_D.get("tau_1_update_eta", 0.35))     # default 0.35
    eta_tau2 = float(hyper_D.get("tau_2_update_eta", 0.35))     # default 0.35

    # init τ4
    if use_empirical_bayes or eb_warm_then_cavi:
        tau_4 = torch.as_tensor(pen_term2_T.get("tau_init", _gamma_mean(a_lambda, b_lambda)),
                                device=model.device,  dtype=torch.float32)
    else:
        q_tau4 = VariationalGamma(a_lambda, b_lambda, rank_T_total, init_from_prior=True)
        tau_4 = q_tau4.mean
    hyper_T["tau"] = tau_4

    # ------------------- Decorrelation (τ1, τ2)
    if decor_present:
        K_RW1 = model.decorrelation_layers[0].priors.K_prior_RW1
        K_RW2 = model.decorrelation_layers[0].priors.K_prior_RW2
        rank_T_total_RW1 = (K_RW1.shape[0] - 1) * num_margins
        rank_T_total_RW2 = (K_RW2.shape[0] - 2) * num_margins

        pen_term1_D = hyper_D.get('RW1', {})
        pen_term2_D = hyper_D.get('RW2', {})
        a_lambda_1 = torch.as_tensor(pen_term1_D.get('tau_a', 1.1), device=model.device, dtype=torch.float32)
        b_lambda_1 = torch.as_tensor(pen_term1_D.get('tau_b', 1e-6), device=model.device, dtype=torch.float32)
        a_lambda_2 = torch.as_tensor(pen_term2_D.get('tau_a', 1.1), device=model.device, dtype=torch.float32)
        b_lambda_2 = torch.as_tensor(pen_term2_D.get('tau_b', 1e-6), device=model.device, dtype=torch.float32)
        print(f"[DECOR] τ₁ a={a_lambda_1.item()} b={b_lambda_1.item()} | τ₂ a={a_lambda_2.item()} b={b_lambda_2.item()}")

        if use_empirical_bayes or eb_warm_then_cavi:
            hyper_D["tau_1"] = torch.as_tensor(_gamma_mean(a_lambda_1, b_lambda_1), device=model.device, dtype=torch.float32)
            hyper_D["tau_2"] = torch.as_tensor(_gamma_mean(a_lambda_2, b_lambda_2), device=model.device,  dtype=torch.float32)
        else:
            q_tau1 = VariationalGamma(a_lambda_1, b_lambda_1, rank_T_total_RW1, init_from_prior=True)
            q_tau2 = VariationalGamma(a_lambda_2, b_lambda_2, rank_T_total_RW2, init_from_prior=True)
            hyper_D["tau_1"], hyper_D["tau_2"] = q_tau1.mean, q_tau2.mean
    else:
        print("training only transformation layer!")
        hyper_D["tau_1"], hyper_D["tau_2"] = 0.0, 0.0

    q_tau4 = None; q_tau1 = None; q_tau2 = None
    
    # ------------------- Full Bayesian taus 
    
    use_tau_vi_anytime = (tau_vi_mode in ("after_warm", "always"))
    tau_nodes = None
    if use_tau_vi_anytime:
        
        mu4 = _softplus_inv(float(hyper_T["tau"]))
        node4 = TauNode(a=float(a_lambda), b=float(b_lambda),
                        mu_init=mu4, log_sigma_init=log(tau_vi_sigma_init), device=model.device)
        
        if decor_present:
            mu1 = _softplus_inv(float(hyper_D["tau_1"]))
            mu2 = _softplus_inv(float(hyper_D["tau_2"]))
            
            node1 = TauNode(a=float(a_lambda_1), b=float(b_lambda_1),
                        mu_init=mu1, log_sigma_init=log(tau_vi_sigma_init), device=model.device)
            
            node2 = TauNode(a=float(a_lambda_2), b=float(b_lambda_2),
                        mu_init=mu2, log_sigma_init=log(tau_vi_sigma_init), device=model.device)
        else:
            
            node1 = node2 = None
        
        tau_nodes = TauPack(node4=node4, node2=node2, node1=node1)
    

    # ------------------- optimizer & scheduler
    import inspect
    def _make_adam(param_groups, lr: float):
        # Detect supported kwargs on this PyTorch build
        sig = inspect.signature(torch.optim.Adam.__init__).parameters
        supports_fused   = ("fused"   in sig)
        supports_foreach = ("foreach" in sig)

        # Try fused Adam first (fastest on Ampere+), then foreach, then vanilla
        if supports_fused:
            try:
                return torch.optim.Adam(param_groups, lr=lr, fused=True)
            except Exception:
                pass
        if supports_foreach:
            try:
                return torch.optim.Adam(param_groups, lr=lr, foreach=True)
            except Exception:
                pass
        return torch.optim.Adam(param_groups, lr=lr)

    # ------------------- optimizer & scheduler
    if rho_lr_multiplier != 1.0:
        param_groups = [
            {"params": [VI.mu],  "lr": lr},
            {"params": [VI.rho], "lr": lr * rho_lr_multiplier},
        ]
        opt = _make_adam(param_groups, lr=lr)
    else:
        param_groups = [{"params": [p for p in VI.parameters() if p.requires_grad]}]
        opt = _make_adam(param_groups, lr=lr)
        

    # τ nodes (keeps same opt; adds a new param group)
    if use_tau_vi_anytime and tau_nodes is not None:
        pg = tau_nodes.tau_param_group(lr=lr * 1.5, betas=(0.9, 0.999))
        if len(pg["params"]) > 0:
            opt.add_param_group(pg)

        
    sched = ReduceLROnPlateau(opt, mode="min", factor=sched_factor, patience=sched_patience,
                              threshold=sched_threshold, cooldown=sched_cooldown,
                              min_lr=sched_min_lr, verbose=verbose)

    best_val = float("inf")
    best_state = {"mu": VI.mu.detach().clone(), "rho": VI.rho.detach().clone()}
    eb_E_qf_num, eb_count = 0.0, 0
    no_improve = 0
    loss_history, val_history = [], []
    start = time.time()

    for epoch in tqdm(range(iterations)):
        beta_kl = _beta_kl_at(epoch, beta_kl_anneal_epochs, beta_kl_start)
        use_tau_vi_now = (
            (tau_vi_mode == "always") or 
            (tau_vi_mode == "after_warm" and epoch >= warm_tau_epochs)
        )
        if mc_ramp_every and epoch > 0 and (epoch % mc_ramp_every == 0):
            mcmc_sample_train = min(mc_ramp_max, max(mcmc_sample_train * 2, 1))

        running, n_batches, obs_seen_epoch = 0.0, 0, 0
        ndp, ntp, ll_batch = 0.0, 0.0, 0.0
        qf_1_dec, qf_2_dec, E_qf_sum_accum = 0.0, 0.0, 0.0

        for b, y in enumerate(train_dataloader):
            B = y.shape[0]
            y = y.to(model.device)
            opt.zero_grad(set_to_none=True)

            if epoch < warm_sigma_epochs:
                VI.rho.requires_grad_(False)
            else:
                VI.rho.requires_grad_(True)

            out = VI.step(samples=y,
                          hyperparameter_transformation=hyper_T,
                          hyperparameter_decorrelation=hyper_D,
                          sample_size_total=N_total,
                          model=model,
                          mcmc_samples=mcmc_sample_train,
                          seed=global_seed + epoch * 10_000 + b,
                          beta_kl=beta_kl,
                          tau_pack=(tau_nodes if use_tau_vi_now else None),
                          beta_tau_kl = tau_kl_beta
                          )

            loss = out["elbo_loss"]
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite loss @ epoch {epoch}, batch {b}: {loss.item()}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(VI.parameters(), 5.0)
            opt.step()

            with torch.no_grad():
                def sp_inv(s): return log(exp(float(s)) - 1.0)
                rho_min = sp_inv(0.02)  # softplus^-1(0.02)
                rho_max = sp_inv(0.06)
                VI.rho.data.clamp_(min=rho_min, max=rho_max)   # narrow band; widen if needed
            
            running         += float(loss.item()); n_batches += 1; obs_seen_epoch += B
            ndp             += float(out['neg_prior_decorrelation'])
            ntp             += float(out['neg_prior_transformation'])
            ll_batch        += float(out['log_likelihhod_batch'])
            eb_E_qf_num     += float(out['transformation_mean_qf'])
            E_qf_sum_accum  += float(out['transformation_sum_qf'])
            qf_1_dec        += float(out['qf1_decorrelation'])
            qf_2_dec        += float(out['qf2_decorrelation'])
            eb_count        += 1

        if n_batches == 0:
            raise RuntimeError("No batches processed. Check dataloader.")

        train_loss = running / obs_seen_epoch
        loss_history.append(train_loss)

        #if validate_dataloader is not None:
        #    val_elpd = _evaluate_epoch(VI, model, validate_dataloader,
        #                               hyper_T, hyper_D,
        #                               S_val=mcmc_sample_val,
        #                               sample_size_total=N_total,
        #                               seed=global_seed + 12345,
        #                               tau_nodes=tau_nodes, use_tau_vi_now=use_tau_vi_now,
        #                               )
        #    val_history.append(val_elpd); metric = -val_elpd
        #else:
        #    metric = train_loss
        #    val_elpd = None
            
        val_dat = train_dataloader if not validate_dataloader else validate_dataloader
        
        val_elpd = _evaluate_epoch(VI, model, val_dat,
                                       hyper_T, hyper_D,
                                       S_val=mcmc_sample_val,
                                       sample_size_total=N_total,
                                       seed=global_seed + 12345,
                                       tau_nodes=tau_nodes, use_tau_vi_now=use_tau_vi_now,
                                       )
        val_history.append(val_elpd); metric = -val_elpd
        
        sched.step(metric)
        improved = (metric < best_val - min_delta)
        if improved:
            best_val = metric; no_improve = 0
            best_state = {"mu": VI.mu.detach().clone(), "rho": VI.rho.detach().clone()}
        else:
            no_improve += 1
            if verbose: print(f"NOT IMPROVED! Nr. {no_improve}")

        # ------------------- τ updates (EB or CAVI) with per-τ damping
        freeze_decor = (epoch < warm_tau_epochs) 
                # ------------------- τ updates (EB or CAVI) with per-τ damping + warm-up freeze for decor
        # ---- τ updates (EB or CAVI) with per-τ damping + warm-up freeze for decor ----
        if eb_count > 0:
            E_qf_total_mc = E_qf_sum_accum / eb_count
            E_qf1_total_mc = qf_1_dec / eb_count
            E_qf2_total_mc = qf_2_dec / eb_count

            # EB fixed-point targets (for EB/CAVI and monitoring)
            tau4_target = float((a_lambda + 0.5 * rank_T_total) / (b_lambda + 0.5 * E_qf_total_mc))
            if decor_present:
                tau1_target = float((a_lambda_1 + 0.5 * rank_T_total_RW1) / (b_lambda_1 + 0.5 * E_qf1_total_mc))
                tau2_target = float((a_lambda_2 + 0.5 * rank_T_total_RW2) / (b_lambda_2 + 0.5 * E_qf2_total_mc))
            else:
                tau1_target = tau2_target = 0.0

            if use_eb_now(epoch):
                # ----- EB step (damped) -----
                tau_new = _damped_step(hyper_T["tau"], tau4_target, eta_tau4, band_tau4)
                if decor_present and not freeze_decor:
                    tau_1_new = _damped_step(hyper_D["tau_1"], tau1_target, eta_tau1, band_decor)
                    tau_2_new = _damped_step(hyper_D["tau_2"], tau2_target, eta_tau2, band_decor)
                else:
                    tau_1_new = float(hyper_D.get("tau_1", 0.0))
                    tau_2_new = float(hyper_D.get("tau_2", 0.0))
            else:
                # ----- CAVI (variational gamma) -----
                if q_tau4 is None:
                    q_tau4 = VariationalGamma(a_lambda, b_lambda, rank_T_total, init_from_prior=True)
                    _seed_q_from_mean(q_tau4, tau_current=float(hyper_T["tau"]))
                q_tau4.update_from_E_qf_total(E_qf_total=E_qf_total_mc)
                tau_new_mean = float(q_tau4.mean)
                tau_new = _damped_step(hyper_T["tau"], tau_new_mean, eta_tau4)

                if decor_present and not freeze_decor:
                    if q_tau1 is None:
                        q_tau1 = VariationalGamma(a_lambda_1, b_lambda_1, rank_T_total_RW1, init_from_prior=True)
                        _seed_q_from_mean(q_tau1, tau_current=float(hyper_D["tau_1"]))
                    if q_tau2 is None:
                        q_tau2 = VariationalGamma(a_lambda_2, b_lambda_2, rank_T_total_RW2, init_from_prior=True)
                        _seed_q_from_mean(q_tau2, tau_current=float(hyper_D["tau_2"]))
                    q_tau1.update_from_E_qf_total(E_qf_total=E_qf1_total_mc)
                    q_tau2.update_from_E_qf_total(E_qf_total=E_qf2_total_mc)
                    tau1_mean = float(q_tau1.mean)
                    tau2_mean = float(q_tau2.mean)
                    tau_1_new = _damped_step(hyper_D["tau_1"], tau1_mean, eta_tau1)
                    tau_2_new = _damped_step(hyper_D["tau_2"], tau2_mean, eta_tau2)
                else:
                    tau_1_new = float(hyper_D.get("tau_1", 0.0))
                    tau_2_new = float(hyper_D.get("tau_2", 0.0))

            # ---- write-back only if NOT using τ-VI ----
            if not use_tau_vi_now:
                tau_new   = max(float(tau_new),  TAU4_FLOOR)
                tau_1_new = max(float(tau_1_new), TAU1_FLOOR)
                tau_2_new = max(float(tau_2_new), TAU2_FLOOR)
                hyper_T["tau"]   = tau_new
                hyper_D["tau_1"] = tau_1_new
                hyper_D["tau_2"] = tau_2_new
                model.hyperparameter["transformation"]["tau"]   = tau_new
                model.hyperparameter["decorrelation"]["tau_1"]  = tau_1_new
                model.hyperparameter["decorrelation"]["tau_2"]  = tau_2_new
        # τ-VI active → skip write-back (τ sampled from variational nodes)

        # ------------------- verbose
        if verbose and ((epoch+1) % 5 == 0 or improved):
            lrs = [pg["lr"] for pg in opt.param_groups]
            tgt4 = _tau_times_qf_target(float(a_lambda),  float(b_lambda),  0.5*rank_T_total,      E_qf_total_mc)
            tgt1 = _tau_times_qf_target(float(a_lambda_1),float(b_lambda_1),0.5*rank_T_total_RW1,  E_qf1_total_mc) if decor_present else 0.0
            tgt2 = _tau_times_qf_target(float(a_lambda_2),float(b_lambda_2),0.5*rank_T_total_RW2,  E_qf2_total_mc) if decor_present else 0.0

            # collect τ means (qτ) and targets (EB)
            if use_tau_vi_now and (tau_nodes is not None):
                with torch.no_grad():
                    
                    tau4_mean = float(tau_nodes.node4.mean_tau_mc(S_tau_monitor, generator=gen)) if tau_nodes.node4 else 0.0
                    if decor_present and tau_nodes.node1 and tau_nodes.node2:
                        tau1_mean = float(tau_nodes.node1.mean_tau_mc(S_tau_monitor, generator=gen))
                        tau2_mean = float(tau_nodes.node2.mean_tau_mc(S_tau_monitor, generator=gen))
                    else:
                        tau1_mean = tau2_mean = 0.0
            else:
                tau4_mean = tau1_mean = tau2_mean = 0.0

            # --- NEW: use qτ means in the monitor if τ-VI is on; otherwise fall back to fixed hyper τs
            tau4_monitor = tau4_mean if use_tau_vi_now else float(hyper_T["tau"])
            tau1_monitor = tau1_mean if use_tau_vi_now else float(hyper_D["tau_1"])
            tau2_monitor = tau2_mean if use_tau_vi_now else float(hyper_D["tau_2"])
            
            val_str = f"  val_ELPD={val_elpd:.4f}" if val_elpd is not None else ""
            
            print(
                f"\nIteration [{epoch+1}/{iterations}] "
                f"train={train_loss:.4f}  {val_str}  "
                f"S_train={mcmc_sample_train} S_val={mcmc_sample_val}  lr={lrs}  "
                f"σ̄={float(VI.sigma.mean()):.4f} σmin={float(VI.sigma.min()):.4f} σmax={float(VI.sigma.max()):.4f}  "
                f"β_KL={beta_kl:.2f}\n"
                f"τ₄={float(hyper_T['tau']):.5g}  τ₁={float(hyper_D['tau_1']):.5g}  τ₂={float(hyper_D['tau_2']):.5g}\n"
                f"(qτ means) τ₄≈{tau4_mean:.5g}  τ₁≈{tau1_mean:.5g}  τ₂≈{tau2_mean:.5g}  "
                f"|  (EB targets) τ₄*≈{tau4_target:.5g}  τ₁*≈{tau1_target:.5g}  τ₂*≈{tau2_target:.5g}\n"
                f"E_qf_total≈{E_qf_total_mc:.4f}  E_qf1≈{E_qf1_total_mc:.4f}  E_qf2≈{E_qf2_total_mc:.4f}\n"
                f"[monitor] τ₄·E_qf≈{tau4_monitor * E_qf_total_mc:.2f}  target≈{tgt4:.2f} | "
                f"τ₁·E_qf1≈{tau1_monitor * E_qf1_total_mc:.2f}  target≈{tgt1:.2f} | "
                f"τ₂·E_qf2≈{tau2_monitor * E_qf2_total_mc:.2f}  target≈{tgt2:.2f}\n"
                f"ELPD trend → {'IMPROVED! Congrats' if improved else 'NOT improved!'} ({no_improve}/{patience_val})"
            )

        eb_E_qf_num, eb_count = 0.0, 0
        if no_improve >= patience_val:
            
            if validate_dataloader is not None:
                print(f"Early stop @ epoch {epoch+1}: no val improvement for {patience_val} epochs.")
            else:
                print(f"Early stop @ epoch {epoch+1}: no train-loglikelihood improvement for {patience_val} epochs.")
            break

    with torch.no_grad():
        VI.mu.copy_(best_state["mu"]); VI.rho.copy_(best_state["rho"])
    VI.set_model_params(VI.mu.detach())
    if was_training: nn.Module.train(model, True)

    return {
        "training_time": time.time() - start,
        "epochs_run": epoch + 1,
        "best_val": best_val,
        "loss_history": loss_history,
        "val_history": val_history if validate_dataloader is not None else None,
        "mu": VI.mu.detach(),
        "rho": VI.rho.detach(),
        "vi_model": VI,
    }


def if_float_create_lambda_penalisation_matrix(lambda_penalty_params, num_vars) -> Tensor:

    lambda_penalty_params = torch.tensor(lambda_penalty_params, dtype=torch.float32)
    if lambda_penalty_params.size() == torch.Size([]):
        lambda_penalty_params = (
            torch.tril(lambda_penalty_params.repeat(num_vars, num_vars))
            - torch.eye(  # before got warning with torch.tensor(lambda_penalty_params) in the tril
                num_vars, num_vars
            )
            * lambda_penalty_params
        )

    return lambda_penalty_params
