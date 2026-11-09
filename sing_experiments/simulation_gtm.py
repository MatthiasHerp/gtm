"""GTM fitting with training-only tuning, adaptive lasso, and batched evaluation."""
from __future__ import annotations

import gc
from pathlib import Path
import tempfile
import time

import numpy as np
import optuna
from sklearn.metrics import average_precision_score, roc_auc_score
import torch
from torch.utils.data import DataLoader

from gtm import GTM
from gtm.gtm_plots_analysis.compute_normalised_hessian_metric import global_nuclear_normalize_vectorised
from sing_experiments.simulation_common import (
    atomic_json, likelihood_metrics, standardize, to_original_order, versions,
)

PENALTY_KEYS = (
    "penalty_decorrelation_ridge_param", "penalty_decorrelation_ridge_first_difference",
    "penalty_decorrelation_ridge_second_difference", "penalty_transformation_ridge_second_difference",
    "penalty_lasso_conditional_independence",
)


def resolve_device(requested):
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(requested)
    if device.type not in ("cpu", "cuda"):
        raise ValueError("GTM study supports cpu, cuda, cuda:N, or auto")
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested, but this Torch installation has no available CUDA device")
        if device.index is not None and device.index >= torch.cuda.device_count():
            raise ValueError("Requested CUDA device index is unavailable")
    return str(device)


def validate_gtm_config(config):
    if config["number_transformation_layers"] != 1 or config["number_decorrelation_layers"] != 3:
        raise ValueError("Paper setup requires 1 marginal layer and 3 decorrelation layers")
    if config["degree_transformations"] != 10 or config["degree_decorrelation"] != 10:
        raise ValueError("Paper setup uses 10 basis functions in both kinds of splines")
    if config["transformation_spline_range"] != [-5, 5] or config["decorrelation_spline_range"] != [-10, 10]:
        raise ValueError("Paper knot ranges are [-5,5] for margins and [-10,10] for decorrelation")
    if config["penalty_transformation_ridge_second_difference"] != 0:
        raise ValueError("The marginal splines must be unpenalized")
    if not 0 < config["validation_fraction"] < 1:
        raise ValueError("validation_fraction must lie in (0,1)")
    for key in ("n_trials", "iterations", "patience", "score_batch_size", "loglik_batch_size"):
        if not isinstance(config[key], int) or config[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    if config["adaptive_weight_epsilon"] <= 0:
        raise ValueError("adaptive_weight_epsilon must be positive")
    if config["edge_threshold"] is not None:
        raise ValueError("GTM threshold selection is deferred; leave edge_threshold null")
    if config["score_metric"] not in ("global_normalized_hessian", "squared_hessian", "pseudo_precision"):
        raise ValueError("Unknown GTM score metric")


def _make_model(config, device, dimension, seed):
    torch.manual_seed(seed)
    if device.startswith("cuda"):
        torch.cuda.manual_seed_all(seed)
    model = GTM(
        number_variables=dimension, number_transformation_layers=1,
        number_decorrelation_layers=config["number_decorrelation_layers"],
        degree_transformations=config["degree_transformations"],
        degree_decorrelation=config["degree_decorrelation"],
        spline_transformation="bspline", spline_decorrelation="bspline",
        transformation_spline_range=config["transformation_spline_range"],
        decorrelation_spline_range=config["decorrelation_spline_range"], device=device,
    )
    model.to(device)
    # Cached spline bases must never reuse training values on validation/test
    # samples or detach the input needed for Hessians.
    model.transformation.store_basis_training = False
    return model


def _loader(values):
    return DataLoader(torch.as_tensor(values, dtype=torch.float32), batch_size=len(values), shuffle=False)


def _best_epochs(history):
    losses = history["loss_list_validation"]
    if losses:
        return int(np.argmin(losses)) + 1
    return len(history["loss_list_training"])


def _fit(config, device, train, validation, seed, directory, penalties,
         adaptive_weights=False, epochs=None, pretrain_epochs=None, pretrained_state=None):
    model = _make_model(config, device, train.shape[1], seed)
    train_loader = _loader(train)
    validation_loader = False if validation is None else _loader(validation)
    iterations = config["iterations"] if epochs is None else max(1, epochs)
    patience = config["patience"] if validation is not None else iterations + 1
    with tempfile.TemporaryDirectory(prefix="fit-", dir=directory) as temporary:
        if config["pretrain_marginals"] and pretrained_state is not None:
            model.load_state_dict(pretrained_state["state_dict"])
            selected_pretrain_epochs = pretrained_state["epochs"]
        elif config["pretrain_marginals"]:
            pre_iterations = config["iterations"] if pretrain_epochs is None else max(1, pretrain_epochs)
            pre = model.pretrain_transformation_layer(
                train_loader, validate_dataloader=validation_loader,
                penalty_splines_params=torch.zeros(4, device=device),
                iterations=pre_iterations, patience=config["patience"] if validation is not None else pre_iterations + 1,
                min_delta=config["min_delta"], optimizer=config["optimizer"],
                learning_rate=config["learning_rate"], temp_folder=temporary,
            )
            if not np.isfinite(pre["loss_list_training"]).all():
                raise FloatingPointError("Nonfinite marginal-pretraining objective")
            selected_pretrain_epochs = _best_epochs(pre)
            del pre
        else:
            selected_pretrain_epochs = 0
        # All trials have identical unpenalized marginal pretraining. Retain a
        # detached copy before joint training so the 30 trials can reuse it.
        model._simulation_pretrained_state = dict(
            state_dict={key: value.detach().cpu().clone() for key, value in model.state_dict().items()},
            epochs=selected_pretrain_epochs,
        )
        scalar_lasso = float(penalties[PENALTY_KEYS[4]])
        lasso = torch.tril(torch.full((train.shape[1], train.shape[1]), scalar_lasso,
                                    dtype=torch.float32, device=device), diagonal=-1)
        weights = (torch.as_tensor(adaptive_weights, dtype=torch.float32, device=device)
                   if adaptive_weights is not False else False)
        history = model.train_method(
            train_loader, validate_dataloader=validation_loader,
            penalty_splines_params=torch.tensor([penalties[key] for key in PENALTY_KEYS[:4]],
                                               dtype=torch.float32, device=device),
            penalty_lasso_conditional_independence=lasso, adaptive_lasso_weights_matrix=weights,
            optimizer=config["optimizer"], learning_rate=config["learning_rate"],
            iterations=iterations, patience=patience, min_delta=config["min_delta"], temp_folder=temporary,
        )
    if not np.isfinite(history["loss_list_training"]).all():
        raise FloatingPointError("Nonfinite GTM training objective")
    if validation is not None and not np.isfinite(history["loss_list_validation"]).all():
        raise FloatingPointError("Nonfinite GTM validation objective")
    info = dict(epochs=_best_epochs(history), pretrain_epochs=selected_pretrain_epochs,
                train_loss=history["loss_list_training"], validation_loss=history["loss_list_validation"])
    del history
    return model, info


def logpdf(model, values, batch_size):
    outputs = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(values), batch_size):
            batch = torch.as_tensor(values[start:start + batch_size], dtype=torch.float32, device=model.device)
            outputs.append(model.log_likelihood(batch, return_lambda_matrix=False).detach().cpu().numpy())
    return np.concatenate(outputs).astype(float)


def pseudo_precision_mean(model, values, batch_size):
    total = np.zeros((values.shape[1], values.shape[1]), dtype=float)
    for start in range(0, len(values), batch_size):
        batch = torch.as_tensor(values[start:start + batch_size], dtype=torch.float32, device=model.device)
        precision = model.compute_pseudo_precision_matrix(batch)
        total += precision.detach().abs().sum(0).cpu().numpy()
    return total / len(values)


def adaptive_weights(model, values, config):
    # Match the existing adaptive-lasso experiment: inverse of training
    # mean absolute pseudo-precision, with a documented numerical floor.
    mean = pseudo_precision_mean(model, values, config["score_batch_size"])
    if not np.isfinite(mean).all():
        raise FloatingPointError("Nonfinite adaptive-lasso pilot score")
    return np.tril(1 / np.maximum(mean, config["adaptive_weight_epsilon"]), k=-1)


def dependence_scores(model, values, config):
    """Train-data scores in input/model column order; no test observations used."""
    totals = {name: np.zeros((values.shape[1], values.shape[1]), dtype=float)
              for name in ("global_normalized_hessian", "squared_hessian", "abs_hessian")}
    batch_size = config["score_batch_size"]
    model.eval()
    for start in range(0, len(values), batch_size):
        batch = torch.as_tensor(values[start:start + batch_size], dtype=torch.float32, device=model.device)
        original_layers = model.num_trans_layers
        try:
            hessian = model.compute_local_loglikelihood_hessian(batch, copula_only=config["score_copula_only"]).detach()
        finally:
            model.num_trans_layers = original_layers
        if not torch.isfinite(hessian).all():
            raise FloatingPointError("Nonfinite GTM log-density Hessians")
        normalized = global_nuclear_normalize_vectorised(hessian)
        totals["global_normalized_hessian"] += normalized.abs().sum(0).cpu().numpy()
        totals["squared_hessian"] += hessian.square().sum(0).cpu().numpy()
        totals["abs_hessian"] += hessian.abs().sum(0).cpu().numpy()
    scores = {key: value / len(values) for key, value in totals.items()}
    scores["pseudo_precision"] = pseudo_precision_mean(model, values, batch_size)
    for key, score in scores.items():
        # Numerical autograd asymmetry must not change which matrix half is scored.
        score = (score + score.T) / 2
        if not np.isfinite(score).all():
            raise FloatingPointError(f"Nonfinite {key} scores")
        scores[key] = score
    return scores


def fit_gtm(data, metadata, directory, config, device):
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    started = time.perf_counter()
    streams = metadata["seed_streams"]
    split = np.random.default_rng(streams["validation"]).permutation(len(data["train"]))
    n_validation = max(1, int(round(len(split) * config["validation_fraction"])))
    if n_validation >= len(split) - 1:
        raise ValueError("Too few observations in the inner training split")
    validation_rows, fitting_rows = split[:n_validation], split[n_validation:]
    # The internal validation fold does not contribute even its mean/std.
    fit_values, validation_values, inner_center, inner_scale = standardize(
        data["train"][fitting_rows], data["train"][validation_rows])
    order = data["variable_order"]
    fit_values, validation_values = fit_values[:, order], validation_values[:, order]
    pilot_penalties = {key: 0.0 for key in PENALTY_KEYS}
    pilot, pilot_info = _fit(config, device, fit_values, validation_values, streams["fit"],
                             directory, pilot_penalties)
    weights_inner = adaptive_weights(pilot, fit_values, config)
    pretrained_inner = pilot._simulation_pretrained_state
    del pilot
    trial_infos = {}

    def objective(trial):
        penalties = {}
        for key in PENALTY_KEYS:
            setting = config[key]
            if setting == "sample":
                lasso = key == PENALTY_KEYS[-1]
                upper = config["penalty_lasso_max"] if lasso else config["penalty_ridge_max"]
                penalties[key] = trial.suggest_float(key, 1e-7, upper, log=lasso)
            else:
                penalties[key] = float(setting)
        try:
            model, info = _fit(config, device, fit_values, validation_values, streams["fit"],
                               directory, penalties, adaptive_weights=weights_inner,
                               pretrained_state=pretrained_inner)
            score = logpdf(model, validation_values, config["loglik_batch_size"]).mean()
            if not np.isfinite(score):
                raise FloatingPointError("Nonfinite held-out tuning log likelihood")
            trial.set_user_attr("epochs", info["epochs"])
            trial.set_user_attr("pretrain_epochs", info["pretrain_epochs"])
            trial_infos[trial.number] = dict(penalties=penalties, info=info)
            del model
            return float(score)
        finally:
            gc.collect()
            if device.startswith("cuda"):
                torch.cuda.empty_cache()

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=streams["fit"]))
    try:
        study.optimize(objective, n_trials=config["n_trials"], catch=(FloatingPointError, RuntimeError))
    finally:
        # Keep tuning failures even if every candidate fails.
        study.trials_dataframe().to_csv(directory / "tuning_trials.csv", index=False)
    selected = trial_infos[study.best_trial.number]
    atomic_json(directory / "tuning.json", dict(best_trial=study.best_trial.number,
                validation_mean_loglik_standardized=study.best_value, selected=selected,
                pilot=pilot_info, validation_rows=validation_rows, fitting_rows=fitting_rows,
                inner_center=inner_center, inner_scale=inner_scale))
    # Refit the pilot and the adaptive-lasso model on ALL N observations. The
    # refit epochs come from internal validation; the external test set is unused.
    full_pilot, _ = _fit(config, device, data["x_train"], None, streams["fit"], directory,
                         pilot_penalties, epochs=pilot_info["epochs"],
                         pretrain_epochs=pilot_info["pretrain_epochs"])
    weights_full = adaptive_weights(full_pilot, data["x_train"], config)
    pretrained_full = full_pilot._simulation_pretrained_state
    del full_pilot
    model, refit_info = _fit(config, device, data["x_train"], None, streams["fit"], directory,
                            selected["penalties"], adaptive_weights=weights_full,
                            epochs=selected["info"]["epochs"],
                            pretrain_epochs=selected["info"]["pretrain_epochs"],
                            pretrained_state=pretrained_full)
    if device.startswith("cuda"):
        torch.cuda.synchronize(device)
    fit_seconds = time.perf_counter() - started
    likelihood, log_arrays = likelihood_metrics(
        logpdf(model, data["x_train"], config["loglik_batch_size"]),
        logpdf(model, data["x_test"], config["loglik_batch_size"]), data["scale"])
    scores = {key: to_original_order(value, order)
              for key, value in dependence_scores(model, data["x_train"], config).items()}
    indices = np.triu_indices_from(data["truth"], k=1)
    target = data["truth"][indices].astype(int)
    metrics = dict(likelihood, device=device, degree=None, fit_seconds=fit_seconds,
                   score_metric=config["score_metric"], score_copula_only=config["score_copula_only"],
                   threshold_method="deferred", precision=None, recall=None, f1=None, mcc=None,
                   tp=None, fp=None, fn=None, tn=None, n_trials=config["n_trials"],
                   n_successful_trials=len(trial_infos), refit_n_train=len(data["train"]),
                   dtype="float32", selected_penalties=selected["penalties"],
                   versions=versions(["torch", "optuna", "numpy", "scipy", "pyvinecopulib"]))
    for key, score in scores.items():
        metrics[f"roc_auc_{key}"] = float(roc_auc_score(target, score[indices]))
    metrics["roc_auc"] = metrics[f"roc_auc_{config['score_metric']}"]
    metrics["average_precision"] = float(average_precision_score(target, scores[config["score_metric"]][indices]))
    atomic_json(directory / "refit_history.json", refit_info)
    # CPU state dictionaries remain portable between CPU/GPU machines.
    torch.save({key: value.detach().cpu() for key, value in model.state_dict().items()}, directory / "model_state.pt")
    artifacts = dict(truth=data["truth"], scores=scores[config["score_metric"]],
                     variable_order=order, adaptive_weights=to_original_order(weights_full, order),
                     adaptive_weights_model_order=weights_full,
                     adaptive_weights_inner_model_order=weights_inner,
                     validation_rows=validation_rows, fitting_rows=fitting_rows, **log_arrays)
    artifacts.update({f"scores_{key}": value for key, value in scores.items()})
    del model
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    return metrics, artifacts
