#!/usr/bin/env python3
"""Evaluate the saved MAGIC GTMs on the shared test rows, on the CSV data scale."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def prepare_test_data(data_path, split_path):
    frame = pd.read_csv(data_path, header=None)
    values = frame.iloc[:, :10].to_numpy(dtype=float)
    data_hash = hashlib.sha256(data_path.read_bytes()).hexdigest()
    with np.load(split_path, allow_pickle=False) as archive:
        metadata = json.loads(archive['metadata_json'].item())
        if metadata['sha256'] != data_hash:
            raise ValueError('Split indices do not match the MAGIC CSV.')
        rows = {g: {s: archive[f'{g}_{s}'].copy() for s in ['train', 'validate', 'test']}
                for g in ['g', 'h']}
    for group in rows:
        group_rows = np.flatnonzero(frame.iloc[:, 10].to_numpy() == group)
        pool_size = int(np.ceil(len(group_rows) * (2 / 3)))
        np.testing.assert_array_equal(rows[group]['test'], group_rows[pool_size:])
        np.testing.assert_array_equal(np.sort(np.concatenate([rows[group]['train'], rows[group]['validate']])),
                                      group_rows[:pool_size])
    # Reproduce load_magic_data(poly_span_abs=12): min/max from the pooled
    # train+validation pool, as used when the saved GTMs were fitted.
    pool = np.concatenate([rows[g][s] for g in ['g', 'h'] for s in ['train', 'validate']])
    lower, upper = values[pool].min(0), values[pool].max(0)
    span = upper - lower
    if not np.isfinite(values).all() or not (span > 0).all():
        raise ValueError('Nonfinite MAGIC values or constant feature.')
    transformed = {g: ((values[rows[g]['test']] - lower) / span - 0.5) * 24
                   for g in ['g', 'h']}
    log_jacobian = float(np.log(24 / span).sum())
    return transformed, rows, lower, upper, log_jacobian, data_hash


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-path', type=Path, default=ROOT / 'demos/data/magic04.csv')
    parser.add_argument('--split-indices', type=Path, default=ROOT / 'sing_experiments/magic_gtm_split_indices.npz')
    parser.add_argument('--models-dir', type=Path, default=ROOT / 'demos/models')
    parser.add_argument('--output', type=Path,
                        default=ROOT / 'sing_experiments/results/magic_classification/gtm_test_loglik.npz')
    parser.add_argument('--batch-size', type=int, default=256)
    args = parser.parse_args(argv)
    if args.batch_size < 1:
        parser.error('batch-size must be positive')
    import torch

    torch.set_num_threads(1)
    data, rows, lower, upper, jacobian, data_hash = prepare_test_data(args.data_path, args.split_indices)
    arrays = dict(normalization_min=lower, normalization_max=upper)
    manifests, summary = {}, []
    for group, layers in [('h', 8), ('g', 6)]:
        path = args.models_dir / f'magic_group_{group}_decorr_{layers}_gtm_state_dict.pth'
        print(f'Evaluating class {group}: {path}', flush=True)
        model = torch.load(path, map_location='cpu', weights_only=False).to('cpu')
        model.device = 'cpu'
        parts = []
        with torch.no_grad():
            for start in range(0, len(data[group]), args.batch_size):
                points = torch.tensor(data[group][start:start + args.batch_size], dtype=torch.float32)
                parts.append(model.log_likelihood(points).detach().cpu().numpy().reshape(-1))
        normalized_logpdf = np.concatenate(parts).astype(float)
        if len(normalized_logpdf) != len(data[group]) or not np.isfinite(normalized_logpdf).all():
            raise ValueError(f'Invalid GTM test log densities for class {group}.')
        original_logpdf = normalized_logpdf + jacobian
        arrays[f'{group}_test_row_indices'] = rows[group]['test']
        arrays[f'{group}_test_logpdf'] = original_logpdf
        manifests[group] = dict(layers=layers, file=path.name,
                                sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        summary.append(dict(group=group, n=len(original_logpdf),
                            normalized_loglik_mean=float(normalized_logpdf.mean()),
                            loglik_mean=float(original_logpdf.mean())))
        del model
    metadata = dict(data_sha256=data_hash, split_sha256=hashlib.sha256(args.split_indices.read_bytes()).hexdigest(),
                    loglik_scale='original observations', normalization='pooled train+validation min/max to [-12,12]',
                    log_jacobian=jacobian, models=manifests, torch_version=str(torch.__version__))
    arrays['metadata_json'] = json.dumps(metadata)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix('.tmp.npz')
    np.savez_compressed(temporary, **arrays)
    temporary.replace(args.output)
    print(pd.DataFrame(summary).to_string(index=False), flush=True)
    print(f'Saved: {args.output}', flush=True)


if __name__ == '__main__':
    main()
