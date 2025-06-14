# import seaborn as sns
# from pytorch_lbfgs.LBFGS import LBFGS, FullBatchLBFGS
# from torch.optim import LBFGS
import copy
import os
import pickle
import time
from functools import reduce

# import warnings
# from torch import nn
import numpy as np
import torch
# from torch.distributions import Normal, Laplace
# import matplotlib.pyplot as plt
from torch import optim
from torch.optim.optimizer import Optimizer
from tqdm import tqdm

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


import math


def get_cosine_schedule_with_warmup(
    optimizer, num_warmup_steps, num_training_steps, num_cycles=0.5, last_epoch=-1
):
    """Create a schedule with a learning rate that decreases following the
    values of the cosine function between 0 and ⁠ pi * cycles ⁠ after a warmup
    period during which it increases linearly between 0 and 1.
    """

    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps)
        )
        return max(
            0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress))
        )

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch)


def train(
    model,
    train_dataloader,
    validate_dataloader=False,
    train_covariates=False,
    validate_covariates=False,
    penalty_params=torch.FloatTensor([0, 0, 0, 0]),
    lambda_penalty_params=False,
    learning_rate=1,
    iterations=2000,
    verbose=True,
    patience=5,
    min_delta=1e-7,
    global_min_loss=-np.inf,
    optimizer="LBFGS",
    lambda_penalty_mode="square",
    objective_type="negloglik",  # ema_decay=False,
    adaptive_lasso_weights_matrix=False,
    max_batches_per_iter=False,
):
    # max_batches_per_iter infos
    # then use random sampling data_loader
    # always 1 for validation data

    start = time.time()

    opt = LBFGS(
        model.parameters(),
        lr=learning_rate,
        history_size=1,
        line_search_fn="strong_wolfe",
        max_iter=1,
        max_eval=40,
    )

    # if ema_decay is not False:
    #    ema = ExponentialMovingAverage(model.parameters(), decay=ema_decay)

    def closure():
        opt.zero_grad()
        return_dict_model_objective = model.__training_objective__(
            y_train,
            penalty_params,
            lambda_penalty_params=lambda_penalty_params,
            train_covariates=train_covariates,
            lambda_penalty_mode=lambda_penalty_mode,
            objective_type=objective_type,
            adaptive_lasso_weights_matrix=adaptive_lasso_weights_matrix,
        )  # use the `objective` function

        loss = return_dict_model_objective["loss_with_penalties"].mean()

        if verbose == True:
            print("current_loss:", loss)

        # Note to myself:
        # retain Graph makes it much slower in negloglik training and retaining the graph increases memory usage more and more as iterations increase.
        # Further it also seems to make the cuda usage more instable in the sense of having spikes, now it runs super smooth and with literaly no memory usage
        if objective_type == "score_matching":

            loss.backward(retain_graph=True)
        else:
            loss.backward()
        return loss

    early_stopper = EarlyStopper(
        patience=patience, min_delta=min_delta, global_min_loss=global_min_loss
    )

    if optimizer == "Adam":
        opt = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0)
        scheduler = get_cosine_schedule_with_warmup(
            opt,
            num_warmup_steps=5,
            num_training_steps=iterations,
            num_cycles=0.5,
            last_epoch=-1,
        )  ##3,4,5 warmup steps machen

    loss_list = []
    loss_list_val = []

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
                return_dict_model_objective = model.__training_objective__(
                    y_train,
                    penalty_params,
                    lambda_penalty_params=lambda_penalty_params,
                    train_covariates=train_covariates,
                    lambda_penalty_mode=lambda_penalty_mode,
                    objective_type=objective_type,
                    adaptive_lasso_weights_matrix=adaptive_lasso_weights_matrix,
                )  # use the `objective` function

                loss = return_dict_model_objective["loss_with_penalties"].mean()
                loss.backward()
                opt.step()
                scheduler.step()
                current_loss = loss
                if verbose == True:
                    print("current_loss:", loss)
            elif optimizer == "LBFGS":
                current_loss = opt.step(closure)

            # if ema_decay is not False:
            #    ema.update()

            if (
                max_batches_per_iter is not False
                and num_processed_batches >= max_batches_per_iter
            ):
                break

        loss_list.append(current_loss.item())

        if validate_dataloader is not False:
            y_validate = next(iter(validate_dataloader))
            y_validate = y_validate.to(model.device)
            model_val.load_state_dict(model.state_dict())

            if objective_type is "negloglik":
                with torch.no_grad():
                    return_dict_model_objective_val = model_val.__training_objective__(
                        y_validate,
                        penalty_params,
                        lambda_penalty_params=lambda_penalty_params,
                        train_covariates=validate_covariates,
                        lambda_penalty_mode=lambda_penalty_mode,
                        objective_type=objective_type,
                        adaptive_lasso_weights_matrix=adaptive_lasso_weights_matrix,
                    )
                    current_loss_val = return_dict_model_objective_val[
                        "loss_without_penalties"
                    ].mean()  # No penalties as on validation set
            else:
                return_dict_model_objective_val = model.__training_objective__(
                    y_validate,
                    penalty_params,
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

            if early_stopper.early_stop(current_loss_val, model):
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
                number_iterations = number_iterations - patience
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
                number_iterations = number_iterations - patience
                break

    # Return the best model which is not necessarily the last model
    model.load_state_dict(early_stopper.best_model_state)

    # Rerun model at the end to get final penalties
    return_dict_model_training = model.__training_objective__(
        y_train,
        penalty_params,
        lambda_penalty_params=lambda_penalty_params,
        train_covariates=train_covariates,
        lambda_penalty_mode=lambda_penalty_mode,
        objective_type=objective_type,
        adaptive_lasso_weights_matrix=adaptive_lasso_weights_matrix,
    )
    return_dict_model_training["loss_list_training"] = loss_list
    return_dict_model_training["loss_list_validation"] = loss_list_val
    return_dict_model_training["number_iterations"] = number_iterations

    end = time.time()

    training_time = end - start
    return_dict_model_training["training_time"] = training_time

    return return_dict_model_training


def if_float_create_lambda_penalisation_matrix(lambda_penalty_params, num_vars):

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
