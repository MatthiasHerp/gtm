#!/usr/bin/env python3
"""Fit SING MAGIC classifiers, order 2 then 3, saving analysis without live models."""
import argparse
import gc
import hashlib
import importlib.metadata as metadata
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
# Set before importing NumPy; callers can explicitly override these settings.
for variable in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(variable, '1')
os.environ.setdefault('MPLCONFIGDIR', '/tmp/matplotlib')

import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.metrics import (accuracy_score, average_precision_score, balanced_accuracy_score,
                             brier_score_loss, f1_score, matthews_corrcoef, precision_score,
                             recall_score, roc_auc_score, roc_curve)
from sing_experiments.sing_helpers import FixedOrdering, SING_COMMIT, load_sing
from sing_experiments.magic_nonparanormal import apply_nonparanormal, fit_nonparanormal

GROUPS = ('g', 'h')
SPLITS = ('train', 'validate', 'test')
FEATURE_NAMES = [f'y_{i}' for i in range(10)]
FPR_LEVELS = np.array([0.01, 0.02, 0.05, 0.10, 0.20])


def save_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def load_data(data_path, split_path, *, nonparanormal=False, return_preprocessing=False):
    if not split_path.exists():
        raise FileNotFoundError(f'{split_path} is missing. Run prepare_magic_splits.py once in a Torch environment.')
    frame = pd.read_csv(data_path, names=FEATURE_NAMES + ['group'])
    assert frame.group.isin(GROUPS).all()
    values = frame[FEATURE_NAMES].to_numpy(dtype=np.float64)
    assert np.isfinite(values).all()
    data_hash = hashlib.sha256(data_path.read_bytes()).hexdigest()
    with np.load(split_path, allow_pickle=False) as archive:
        split_metadata = json.loads(archive['metadata_json'].item())
        rows = {g: {s: archive[f'{g}_{s}'].copy() for s in SPLITS} for g in GROUPS}
    assert split_metadata['sha256'] == data_hash, 'CSV differs from the split-index source'
    for key, expected in dict(train_portion=2 / 3, cross_validation_folds=5,
                              validation_fold_index=4, split_random_state=25).items():
        assert split_metadata[key] == expected, f'Unexpected split setting: {key}'
    for group in GROUPS:
        used = np.concatenate(list(rows[group].values()))
        assert len(used) == len(np.unique(used))
        np.testing.assert_array_equal(np.sort(used), np.flatnonzero(frame.group.to_numpy() == group))
    pooled = np.vstack([values[rows[g]['train']] for g in GROUPS])
    preprocessing = {}
    preprocessing_info = dict(method='training_mean_std', fitted_on='pooled fitting-fold rows',
                               loglik_scale='original observations')
    if nonparanormal:
        print('Fitting shared nonparanormal CDFs on pooled training rows only.', flush=True)
        preprocessing = fit_nonparanormal(pooled)
        # All classes and splits use this same training-fitted map.
        values = apply_nonparanormal(values, preprocessing)
        pooled = np.vstack([values[rows[g]['train']] for g in GROUPS])
        preprocessing_info = dict(
            method='pooled_training_empirical_gaussianization', fitted_on='pooled fitting-fold rows',
            training_probabilities='(average_rank - 0.5) / n_pooled_train',
            heldout='linear interpolation of training CDF knots; clip at training endpoint probabilities',
            loglik_scale='pooled-training Gaussianized coordinates',
            original_scale_jacobian=False,
        )
    center, scale = pooled.mean(0), pooled.std(0, ddof=0)
    assert np.isfinite(scale).all() and np.all(scale > 0)
    standardized = {g: {s: (values[rows[g][s]] - center) / scale for s in SPLITS} for g in GROUPS}
    print('GTM-compatible split counts:', flush=True)
    print(pd.DataFrame({g: {s: len(rows[g][s]) for s in SPLITS} for g in GROUPS}).to_string(), flush=True)
    result = (standardized, rows, center, scale, split_metadata, data_hash)
    if return_preprocessing:
        return (*result, preprocessing, preprocessing_info)
    return result


def diagnostics_from_history(history):
    return [[dict(success=bool(log['success']),
                  accepted_by_gradient=bool(log['accepted_by_gradient']), message=str(log['message']),
                  iterations=int(log['nit']),
                  gradient_inf_norm=float(np.linalg.norm(log['jac'], ord=np.inf)))
             for log in iteration['solver_logs']] for iteration in history]


def fit_class(group, standardized, rows, scale, args, directory):
    checkpoint = directory / f'class_{group}.npz'
    manifest = directory / f'class_{group}.json'
    if checkpoint.exists() and manifest.exists():
        print(f'Order {args.worker_order}, class {group}: loading completed checkpoint.', flush=True)
        with np.load(checkpoint, allow_pickle=False) as archive:
            arrays = {key: archive[key].copy() for key in archive.files}
        return arrays, json.loads(manifest.read_text())
    seed = args.fit_seed + GROUPS.index(group)
    np.random.seed(seed)
    random.seed(seed)
    print(f'Order {args.worker_order}: fitting class {group}, '
          f'{len(standardized[group]["train"])} training observations, derivative batch size {args.batch_size}.', flush=True)
    sing, history = load_sing(batch_size=args.batch_size, optimizer_derivatives=args.optimizer_derivatives,
                              optimizer_maxiter=args.optimizer_maxiter, retain_density=True,
                              progress=True, sing_directory=args.sing_dir)
    started = time.perf_counter()
    omega = sing(standardized[group]['train'].copy(), args.worker_order, FixedOrdering(), args.delta)
    elapsed = time.perf_counter() - started
    assert np.isfinite(omega).all() and np.allclose(omega, omega.T)
    density = history[-1]['density']
    adjacency = omega != 0
    np.fill_diagonal(adjacency, False)
    arrays = dict(graph=adjacency, thresholded_scores=omega.copy(),
                  initial_scores=history[0]['omega_raw'].copy(), final_scores=history[-1]['omega_raw'].copy(),
                  map_coefficients=np.asarray(density.transport_map.coeffs).copy())
    info = dict(group=group, n_train=len(standardized[group]['train']), iterations=len(history),
                edges=int(adjacency.sum() // 2), elapsed_seconds=elapsed, fit_seed=seed,
                solver_diagnostics=diagnostics_from_history(history))
    jacobian = float(np.log(scale).sum())
    # Evaluate immediately so the fitted density can be released before the next class.
    for target_group in GROUPS:
        for split in SPLITS:
            points = standardized[target_group][split]
            parts = []
            for start in range(0, len(points), args.logpdf_batch_size):
                end = min(start + args.logpdf_batch_size, len(points))
                print(f'Order {args.worker_order}, density {group}, {target_group}/{split}: '
                      f'logpdf observations {start + 1}-{end} of {len(points)}', flush=True)
                parts.append(np.asarray(density.log_pdf(points[start:end])).reshape(-1))
            logpdf = np.concatenate(parts) - jacobian
            if len(logpdf) != len(points) or not np.isfinite(logpdf).all():
                raise FloatingPointError(f'Nonfinite or incorrectly shaped class-{group} log densities')
            arrays[f'logpdf_{target_group}_{split}'] = logpdf
    temporary = checkpoint.with_suffix('.tmp.npz')
    np.savez_compressed(temporary, **arrays)
    temporary.replace(checkpoint)
    save_json(manifest, info)
    print(f'Order {args.worker_order}, class {group}: checkpoint saved; fit took {elapsed:.1f}s.', flush=True)
    # The upstream module keeps closures over this list; release its density too.
    history.clear()
    del density, history, sing
    gc.collect()
    return arrays, info


def classification_analysis(class_arrays, rows):
    y_true = np.concatenate([np.ones(len(rows['g']['test']), dtype=int),
                             np.zeros(len(rows['h']['test']), dtype=int)])
    logpdf_gamma = np.concatenate([class_arrays['g'][f'logpdf_{g}_test'] for g in GROUPS])
    logpdf_hadron = np.concatenate([class_arrays['h'][f'logpdf_{g}_test'] for g in GROUPS])
    # Equal class priors, preserving the original notebook's Bayes classifier.
    log_odds = logpdf_gamma - logpdf_hadron
    probability = expit(log_odds)
    predicted = (log_odds >= 0).astype(int)
    fpr, tpr, thresholds = roc_curve(y_true, log_odds, pos_label=1)
    metrics = dict(roc_auc=float(roc_auc_score(y_true, log_odds)),
                   average_precision=float(average_precision_score(y_true, log_odds)),
                   accuracy=float(accuracy_score(y_true, predicted)),
                   balanced_accuracy=float(balanced_accuracy_score(y_true, predicted)),
                   precision=float(precision_score(y_true, predicted, zero_division=0)),
                   recall=float(recall_score(y_true, predicted, zero_division=0)),
                   f1=float(f1_score(y_true, predicted, zero_division=0)),
                   mcc=float(matthews_corrcoef(y_true, predicted)),
                   binary_log_loss=float(np.logaddexp(0, np.where(y_true == 1, -log_odds, log_odds)).mean()),
                   brier_score=float(brier_score_loss(y_true, probability)))
    predictions = pd.DataFrame(dict(source_row_0based=np.concatenate([rows[g]['test'] for g in GROUPS]),
                                    true_group=np.where(y_true == 1, 'g', 'h'), gamma_label=y_true,
                                    predicted_gamma_label=predicted, posterior_gamma=probability,
                                    log_odds_gamma=log_odds, logpdf_gamma=logpdf_gamma, logpdf_hadron=logpdf_hadron))
    roc = pd.DataFrame(dict(fpr=fpr, tpr=tpr, log_odds_threshold=thresholds))
    tpr_table = pd.DataFrame({'FPR': FPR_LEVELS, 'SING': [float(tpr[fpr <= level].max()) for level in FPR_LEVELS]})
    return metrics, predictions, roc, tpr_table


def run_order(args):
    nonparanormal = getattr(args, 'nonparanormal', False)
    (standardized, rows, center, scale, split_metadata, data_hash,
     preprocessing, preprocessing_info) = load_data(
        args.data_path, args.split_indices, nonparanormal=nonparanormal, return_preprocessing=True)
    directory = args.output_dir / f'p{args.worker_order}_delta{args.delta:g}_seed{args.fit_seed}_{data_hash[:12]}'
    directory.mkdir(parents=True, exist_ok=True)
    config = dict(order=args.worker_order, delta=args.delta, fit_seed=args.fit_seed, batch_size=args.batch_size,
                  optimizer_derivatives=args.optimizer_derivatives, optimizer_maxiter=args.optimizer_maxiter,
                  logpdf_batch_size=args.logpdf_batch_size, class_priors={'g': 0.5, 'h': 0.5},
                  standardization='pooled fitting-fold mean/std, ddof=0', nonparanormal_transform=nonparanormal,
                  shuffle_variables=False, variable_order=FEATURE_NAMES, device='cpu')
    if nonparanormal:
        config['nonparanormal_method'] = preprocessing_info
    versions = {p: metadata.version(p) for p in ['TransportMaps', 'numpy', 'scipy', 'scikit-learn']}
    identity = dict(config=config, data_sha256=data_hash,
                    split_sha256=hashlib.sha256(args.split_indices.read_bytes()).hexdigest(),
                    sing_commit=SING_COMMIT, package_versions=versions)
    config_path = directory / 'run_config.json'
    if config_path.exists() and json.loads(config_path.read_text()) != identity:
        raise ValueError(f'Configuration changed for {directory}; choose a new --output-dir.')
    if not config_path.exists() and any(directory.glob('class_*')):
        raise ValueError(f'Checkpoint configuration is missing in {directory}; choose a new --output-dir.')
    save_json(config_path, identity)
    if (directory / 'summary.json').exists() and (directory / 'status.json').exists():
        if json.loads((directory / 'status.json').read_text())['status'] == 'ok':
            print(f'Order {args.worker_order}: already completed: {directory}', flush=True)
            return
    save_json(directory / 'status.json', dict(status='running', order=args.worker_order))
    try:
        save_json(directory / 'preprocessing.json', preprocessing_info)
        if nonparanormal:
            np.savez_compressed(directory / 'preprocessing.npz', **preprocessing)
        class_arrays, class_info = {}, {}
        for group in GROUPS:
            class_arrays[group], class_info[group] = fit_class(group, standardized, rows, scale, args, directory)
        metrics, predictions, roc, tpr_table = classification_analysis(class_arrays, rows)
        predictions.to_csv(directory / 'test_predictions.csv', index=False)
        roc.to_csv(directory / 'roc_curve.csv', index=False)
        tpr_table.to_csv(directory / 'tpr_at_fpr.csv', index=False)
        pd.DataFrame([metrics], index=['SING']).to_csv(directory / 'classification_metrics.csv')
        density_rows = []
        for group in GROUPS:
            for split in SPLITS:
                values = class_arrays[group][f'logpdf_{group}_{split}']
                density_rows.append(dict(group=group, split=split, n=len(values),
                                         loglik_mean=float(values.mean()), loglik_sum=float(values.sum())))
        pd.DataFrame(density_rows).to_csv(directory / 'density_loglik.csv', index=False)
        pd.DataFrame({'training_mean': center, 'training_std': scale}, index=FEATURE_NAMES).to_csv(
            directory / 'feature_scaling.csv', index_label='feature')
        fit_summary = [{key: value for key, value in class_info[g].items() if key != 'solver_diagnostics'} for g in GROUPS]
        pd.DataFrame(fit_summary).set_index('group').to_csv(directory / 'fit_summary.csv')
        arrays = dict(training_mean=center, training_std=scale)
        for name in ['source_row_0based', 'gamma_label', 'predicted_gamma_label', 'posterior_gamma',
                     'log_odds_gamma', 'logpdf_gamma', 'logpdf_hadron']:
            arrays[name] = predictions[name].to_numpy()
        # Preserve the original notebook's artifact names as well.
        arrays.update(test_row_indices=arrays['source_row_0based'], y_true=arrays['gamma_label'],
                      y_pred=arrays['predicted_gamma_label'])
        for group in GROUPS:
            arrays.update({f'{group}_{key}': value for key, value in class_arrays[group].items()})
            arrays.update({f'{group}_{split}_row_indices': rows[group][split] for split in SPLITS})
        np.savez_compressed(directory / 'arrays.npz', **arrays)
        save_json(directory / 'summary.json', dict(config=config, split_metadata=split_metadata,
                  sing_commit=SING_COMMIT, package_versions=versions, classification_metrics=metrics,
                  loglik_scale=preprocessing_info['loglik_scale'], n_test=len(predictions), fit_summary=fit_summary,
                  solver_diagnostics={g: class_info[g]['solver_diagnostics'] for g in GROUPS}))
        save_json(directory / 'status.json', dict(status='ok', order=args.worker_order))
        print(pd.DataFrame([metrics], index=[f'SING order {args.worker_order}']).to_string(), flush=True)
        print(tpr_table.to_string(index=False), flush=True)
        print(f'Saved order {args.worker_order} MAGIC results: {directory}', flush=True)
    except BaseException as error:
        save_json(directory / 'status.json', dict(status='interrupted' if isinstance(error, KeyboardInterrupt) else 'failed',
                                                  order=args.worker_order, error=f'{type(error).__name__}: {error}'))
        (directory / 'failure.txt').write_text(traceback.format_exc())
        raise


def main(argv=None, *, nonparanormal_default=False):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--orders', nargs='+', type=int, choices=(2, 3), default=[2, 3])
    parser.add_argument('--data-path', type=Path, default=ROOT / 'demos/data/magic04.csv')
    parser.add_argument('--split-indices', type=Path, default=ROOT / 'sing_experiments/magic_gtm_split_indices.npz')
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--nonparanormal', action='store_true', default=nonparanormal_default,
                        help='Fit one shared empirical Gaussianization on pooled training rows')
    default_sing = ROOT / '.external/SING'
    if not default_sing.exists() and Path('/opt/gtm/.external/SING').exists():
        default_sing = Path('/opt/gtm/.external/SING')
    parser.add_argument('--sing-dir', type=Path, default=default_sing)
    parser.add_argument('--delta', type=float, default=1.0)
    parser.add_argument('--fit-seed', type=int, default=25)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--logpdf-batch-size', type=int, default=512)
    parser.add_argument('--optimizer-derivatives', type=int, choices=(1, 2), default=2)
    parser.add_argument('--optimizer-maxiter', type=int, default=150)
    parser.add_argument('--worker-order', type=int, choices=(2, 3), help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.output_dir is None:
        folder = 'magic_classification_nonparanormal' if args.nonparanormal else 'magic_classification'
        args.output_dir = ROOT / 'sing_experiments/results' / folder
    if not np.isfinite(args.delta) or args.delta <= 0 or min(args.batch_size, args.logpdf_batch_size, args.optimizer_maxiter) < 1:
        parser.error('delta must be positive and finite; batch sizes and optimizer maxiter must be positive')
    if args.fit_seed < 0 or args.fit_seed >= 2**32 - 1:
        parser.error('fit-seed must be between 0 and 2**32 - 2')
    if args.worker_order is not None:
        run_order(args)
        return 0
    if not args.split_indices.exists():
        parser.error(f'Missing {args.split_indices}; run prepare_magic_splits.py once in a Torch environment.')
    failures = 0
    # Fresh process per order releases the allocator and all upstream state.
    for order in sorted(set(args.orders)):
        variant = 'nonparanormal' if args.nonparanormal else 'standardization only'
        print(f'\n=== Starting MAGIC SING order {order} ({variant}) ===', flush=True)
        command = [sys.executable, '-u', str(Path(__file__).resolve()), '--worker-order', str(order)]
        if args.nonparanormal:
            command.append('--nonparanormal')
        for name in ['data_path', 'split_indices', 'output_dir', 'sing_dir', 'delta', 'fit_seed',
                     'batch_size', 'logpdf_batch_size', 'optimizer_derivatives', 'optimizer_maxiter']:
            command += ['--' + name.replace('_', '-'), str(getattr(args, name))]
        result = subprocess.run(command, check=False)
        if result.returncode:
            failures += 1
            print(f'Order {order} failed (exit {result.returncode}); completed class checkpoints are retained.', flush=True)
        else:
            print(f'=== Finished MAGIC SING order {order} ===', flush=True)
    return int(failures > 0)


if __name__ == '__main__':
    raise SystemExit(main())
