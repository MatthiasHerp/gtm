"""Training-only Gaussianization and saved MAGIC analysis regression checks."""
from contextlib import redirect_stdout
import hashlib
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import norm, rankdata

from sing_experiments.magic_nonparanormal import apply_nonparanormal, fit_nonparanormal
from sing_experiments import run_magic_classification as runner
from sing_experiments import run_magic_classification_nonparanormal as npn_runner


class NonparanormalTests(unittest.TestCase):
    def test_training_scores_match_average_ranks_and_heldout_is_clipped(self):
        training = np.array([[1., 10.], [1., 20.], [3., 30.], [4., 40.]])
        state = fit_nonparanormal(training)
        np.testing.assert_allclose(apply_nonparanormal(training, state),
                                   norm.ppf((rankdata(training, axis=0) - 0.5) / len(training)))
        extreme = apply_nonparanormal(np.array([[-1e100, -1e100], [1e100, 1e100]]), state)
        self.assertTrue(np.isfinite(extreme).all())
        np.testing.assert_allclose(extreme[0], apply_nonparanormal(training, state).min(0))
        np.testing.assert_allclose(extreme[1], apply_nonparanormal(training, state).max(0))
        self.assertEqual(apply_nonparanormal(np.array([[2., 25.]]), state).shape, (1, 2))
        with self.assertRaises(ValueError):
            fit_nonparanormal(np.ones((3, 2)))

    def make_fixture(self, directory, *, heldout_shift=0):
        data_path, split_path = directory / 'magic.csv', directory / 'split.npz'
        values = np.random.default_rng(123).lognormal(size=(24, 10))
        heldout = np.r_[4:12, 16:24]
        values[heldout] += heldout_shift
        frame = pd.DataFrame(values)
        frame['group'] = ['g'] * 12 + ['h'] * 12
        frame.to_csv(data_path, header=False, index=False)
        split_metadata = dict(sha256=hashlib.sha256(data_path.read_bytes()).hexdigest(),
                              train_portion=2 / 3, cross_validation_folds=5,
                              validation_fold_index=4, split_random_state=25)
        arrays = {}
        for group, offset in [('g', 0), ('h', 12)]:
            for split, indices in [('train', np.arange(4)), ('validate', np.arange(4, 8)),
                                   ('test', np.arange(8, 12))]:
                arrays[f'{group}_{split}'] = indices + offset
        np.savez_compressed(split_path, metadata_json=json.dumps(split_metadata), **arrays)
        return data_path, split_path

    def test_heldout_values_do_not_change_transform_or_scaling(self):
        with tempfile.TemporaryDirectory() as temporary, redirect_stdout(io.StringIO()):
            directory = Path(temporary)
            a = runner.load_data(*self.make_fixture(directory), nonparanormal=True, return_preprocessing=True)
            b = runner.load_data(*self.make_fixture(directory, heldout_shift=1e6),
                                 nonparanormal=True, return_preprocessing=True)
            for key in a[6]:
                np.testing.assert_array_equal(a[6][key], b[6][key])
            for index in [2, 3]:
                np.testing.assert_array_equal(a[index], b[index])
            for group in runner.GROUPS:
                np.testing.assert_array_equal(a[0][group]['train'], b[0][group]['train'])
            pooled = np.vstack([a[0][g]['train'] for g in runner.GROUPS])
            np.testing.assert_allclose(pooled.mean(0), 0, atol=1e-12)
            np.testing.assert_allclose(pooled.std(0), 1, atol=1e-12)

    def test_worker_commands_keep_nonparanormal_flag_and_order(self):
        with tempfile.TemporaryDirectory() as temporary:
            _, split_path = self.make_fixture(Path(temporary))
            with patch.object(runner.subprocess, 'run', return_value=SimpleNamespace(returncode=0)) as run, \
                 redirect_stdout(io.StringIO()):
                self.assertEqual(npn_runner.main(['--split-indices', str(split_path)]), 0)
            self.assertEqual(run.call_count, 2)
            for order, call in zip([2, 3], run.call_args_list):
                command = call.args[0]
                self.assertIn('--nonparanormal', command)
                self.assertEqual(command[command.index('--worker-order') + 1], str(order))
                self.assertTrue(command[command.index('--output-dir') + 1].endswith('magic_classification_nonparanormal'))

    def test_artifacts_resume_original_isolation_and_notebook(self):
        with tempfile.TemporaryDirectory() as temporary, redirect_stdout(io.StringIO()):
            directory = Path(temporary)
            data_path, split_path = self.make_fixture(directory)
            args = SimpleNamespace(worker_order=2, data_path=data_path, split_indices=split_path,
                                   output_dir=directory / 'raw', delta=1., fit_seed=25, batch_size=2,
                                   optimizer_derivatives=2, optimizer_maxiter=150,
                                   logpdf_batch_size=2, sing_dir=None, nonparanormal=False)
            calls = []
            def fake_load(**settings):
                history = []
                def fit(points, order, ordering, delta):
                    calls.append((args.nonparanormal, order))
                    center = points.mean(0)
                    density = SimpleNamespace(transport_map=SimpleNamespace(coeffs=center.copy()),
                                              log_pdf=lambda x: -np.square(x - center).sum(1))
                    history.append(dict(omega_raw=np.ones((10, 10)), density=density, solver_logs=[
                        dict(success=True, accepted_by_gradient=False, message='test', nit=1, jac=np.zeros(2))]))
                    return np.ones((10, 10))
                return fit, history
            with patch.object(runner, 'load_sing', side_effect=fake_load):
                for order in [2, 3]:
                    args.worker_order = order
                    runner.run_order(args)
                raw_before = {str(p.relative_to(directory)): p.read_bytes() for p in (directory / 'raw').rglob('*') if p.is_file()}
                args.nonparanormal = True
                args.output_dir = directory / 'npn'
                for order in [2, 3]:
                    args.worker_order = order
                    runner.run_order(args)
                runner.run_order(args)
            self.assertEqual(calls, [(False, 2), (False, 2), (False, 3), (False, 3),
                                     (True, 2), (True, 2), (True, 3), (True, 3)])
            for name, content in raw_before.items():
                self.assertEqual((directory / name).read_bytes(), content)
            folder = next((directory / 'npn').glob('p2_*'))
            summary = json.loads((folder / 'summary.json').read_text())
            self.assertTrue(summary['config']['nonparanormal_transform'])
            self.assertEqual(summary['loglik_scale'], 'pooled-training Gaussianized coordinates')
            with np.load(folder / 'preprocessing.npz', allow_pickle=False) as state:
                self.assertEqual(len(state.files), 20)
            # Recompute classifier scores solely from saved predictions.
            predictions = pd.read_csv(folder / 'test_predictions.csv')
            self.assertAlmostEqual(runner.roc_auc_score(predictions.gamma_label, predictions.log_odds_gamma),
                                   summary['classification_metrics']['roc_auc'])
            # Run every analysis cell on all four saved fixtures, then on raw-only results.
            notebook = json.loads((runner.ROOT / 'sing_experiments/sing_magic_classification.ipynb').read_text())
            for npn_dir in [directory / 'npn', directory / 'missing']:
                namespace = {}
                for cell in notebook['cells']:
                    if cell['cell_type'] != 'code':
                        continue
                    source = ''.join(line for line in cell['source'] if not line.startswith('%'))
                    source = source.replace('from IPython.display import display', 'def display(*items): pass')
                    exec(compile(source, '<MAGIC analysis notebook>', 'exec'), namespace)
                    if 'from pathlib import Path' in source:
                        namespace.update(RESULTS_DIR=directory / 'raw', NONPARANORMAL_RESULTS_DIR=npn_dir,
                                         DATA_SHA256=hashlib.sha256(data_path.read_bytes()).hexdigest())
                self.assertEqual(len(namespace['available']), 4 if npn_dir.exists() else 2)
                for number in plt.get_fignums():
                    plt.figure(number).canvas.draw()
                plt.close('all')


if __name__ == '__main__':
    unittest.main()
