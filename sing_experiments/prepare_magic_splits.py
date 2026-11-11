#!/usr/bin/env python3
"""Export the exact GTM MAGIC split indices; requires Torch only for this step."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def prepare_splits(data_path, output):
    import torch
    from torch.utils.data import random_split

    frame = pd.read_csv(data_path, header=None)
    archive = {}
    for group in ('g', 'h'):
        rows = np.flatnonzero(frame.iloc[:, -1].to_numpy() == group)
        pool_size = int(np.ceil(len(rows) * (2 / 3)))
        pool = rows[:pool_size]
        fold_size = pool_size // 5
        folds = random_split(pool, [fold_size] * 4 + [pool_size - 4 * fold_size],
                             generator=torch.Generator().manual_seed(25))
        archive[f'{group}_train'] = pool[np.concatenate([fold.indices for fold in folds[:4]])]
        archive[f'{group}_validate'] = pool[folds[4].indices]
        archive[f'{group}_test'] = rows[pool_size:]
    metadata = dict(sha256=hashlib.sha256(data_path.read_bytes()).hexdigest(),
                    train_portion=2 / 3, cross_validation_folds=5,
                    validation_fold_index=4, split_random_state=25,
                    torch_version=str(torch.__version__))
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, metadata_json=json.dumps(metadata), **archive)
    print(f'Saved GTM-compatible split indices: {output}', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-path', type=Path, default=ROOT / 'demos/data/magic04.csv')
    parser.add_argument('--output', type=Path, default=ROOT / 'sing_experiments/magic_gtm_split_indices.npz')
    args = parser.parse_args()
    prepare_splits(args.data_path, args.output)
