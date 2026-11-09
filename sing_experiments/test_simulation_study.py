"""Study design, likelihood-scale, pairing, and cache regression checks."""
from contextlib import redirect_stdout
import importlib.util
import io
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from scipy.stats import norm

from sing_experiments.simulation_common import (
    DEFAULT_STUDY, Case, ensure_config, get_dataset, likelihood_metrics,
    paired_comparisons, read_json, select_cases, standardize, to_original_order,
    validate_study, run_case,
)


class StudyTests(unittest.TestCase):
    def setUp(self):
        self.study = read_json(DEFAULT_STUDY)

    def test_grid_and_fixed_settings(self):
        validate_study(self.study)
        self.assertEqual(len(select_cases(self.study)), 90)
        self.assertEqual(len(self.study['sing_degrees']) * len(select_cases(self.study)), 180)
        for field, value in (('nonparanormal_transform', True), ('shuffle_variables', False)):
            with self.subTest(field=field), self.assertRaises(ValueError):
                validate_study(dict(self.study, **{field: value}))

    def test_train_only_standardization(self):
        train = np.array([[1, 2], [3, 8], [5, 14]], dtype=float)
        test = np.array([[1000, -1000], [2000, 2000]], dtype=float)
        a, b, center, scale = standardize(train, test)
        np.testing.assert_allclose(center, [3, 8])
        np.testing.assert_allclose(a.mean(axis=0), 0, atol=1e-15)
        np.testing.assert_allclose(a.std(axis=0), 1)
        np.testing.assert_allclose(b, (test - center) / scale)

    def test_original_scale_likelihood_jacobian(self):
        scale = np.array([2.0, 3.0])
        center = np.array([4.0, -2.0])
        raw = np.array([[4, -2], [6, 1], [0, 4]], dtype=float)
        z = (raw - center) / scale
        standardized = norm.logpdf(z).sum(axis=1)
        metrics, arrays = likelihood_metrics(standardized, standardized, scale)
        expected = norm.logpdf(raw, loc=center, scale=scale).sum(axis=1)
        np.testing.assert_allclose(arrays['test_logpdf'], expected)
        self.assertAlmostEqual(metrics['test_loglik_mean'], expected.mean())
        self.assertAlmostEqual(metrics['train_loglik_sum'], expected.sum())

    def test_two_axis_label_restoration(self):
        scores = np.array([[0, 7, 1], [7, 0, 3], [1, 3, 0]])
        order = np.array([2, 0, 1])
        np.testing.assert_array_equal(to_original_order(scores[np.ix_(order, order)], order), scores)

    def test_cached_datasets_and_distinct_vine_seeds(self):
        study = dict(self.study, n_test=32)
        with tempfile.TemporaryDirectory() as temporary:
            a, meta_a = get_dataset(temporary, study, Case(125, 'gauss', 1))
            cached, meta_cached = get_dataset(temporary, study, Case(125, 'gauss', 1))
            self.assertEqual(meta_a, meta_cached)
            for key in a:
                np.testing.assert_array_equal(a[key], cached[key])
            b, meta_b = get_dataset(temporary, study, Case(500, 'gauss', 1))
            np.testing.assert_array_equal(a['test'], b['test'])
            np.testing.assert_array_equal(a['variable_order'], b['variable_order'])
            c, meta_c = get_dataset(temporary, study, Case(125, 'gauss', 2))
            self.assertNotEqual(meta_a['dataset_hash'], meta_c['dataset_hash'])
            self.assertFalse(np.array_equal(a['variable_order'], c['variable_order']))
            np.testing.assert_allclose(a['x_train'].mean(axis=0), 0, atol=1e-15)
            self.assertFalse(np.array_equal(a['train'][:32], a['test']))
            with self.assertRaises(ValueError):
                ensure_config(Path(temporary) / 'study.json', dict(study, n_test=100))

    def test_paired_summary_uses_complete_matching_seeds(self):
        rows = []
        for seed in (1, 2):
            for model, auc in (('GTM', 0.9), ('SING degree 2', 0.8)):
                rows.append(dict(n_train=125, marginals='gauss', seed=seed, model=model,
                                 roc_auc=auc, status='ok', dataset_hash=f'data-{seed}'))
        rows.append(dict(n_train=125, marginals='gauss', seed=3, model='GTM',
                         roc_auc=0.95, status='ok', dataset_hash='data-3'))
        summary = paired_comparisons(pd.DataFrame(rows), 'roc_auc')
        self.assertEqual(summary.iloc[0].n_pairs, 2)
        self.assertAlmostEqual(summary.iloc[0].mean_difference, 0.1)
        rows[1]['dataset_hash'] = 'wrong'
        with self.assertRaisesRegex(ValueError, 'different datasets'):
            paired_comparisons(pd.DataFrame(rows), 'roc_auc')

    def test_failures_are_retried_and_successes_are_resumed(self):
        calls = []

        def fit(data, metadata, directory):
            calls.append(1)
            if len(calls) == 1:
                raise RuntimeError('deliberate numerical failure')
            return dict(roc_auc=0.8, train_loglik_mean=-10, test_loglik_mean=-11,
                        precision=None, recall=None, f1=None, mcc=None), dict(scores=np.eye(10))

        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary) / 'run'
            args = SimpleNamespace(output_dir=Path(temporary), run_label='test')
            metadata = dict(dataset_hash='identical-data', variable_order_1based=list(range(1, 11)))
            with patch('sing_experiments.simulation_common.get_dataset', return_value=({}, metadata)), \
                    redirect_stdout(io.StringIO()):
                self.assertFalse(run_case(directory, Case(125, 'gauss', 1), self.study, args, 'GTM', {}, fit))
                self.assertEqual(read_json(directory / 'result.json')['status'], 'failed')
                self.assertTrue((directory / 'failure.txt').exists())
                self.assertTrue(run_case(directory, Case(125, 'gauss', 1), self.study, args, 'GTM', {}, fit))
                self.assertEqual(read_json(directory / 'result.json')['status'], 'ok')
                self.assertIsNone(read_json(directory / 'result.json')['precision'])
                self.assertFalse((directory / 'failure.txt').exists())
                self.assertTrue(run_case(directory, Case(125, 'gauss', 1), self.study, args, 'GTM', {}, fit))
                self.assertEqual(len(calls), 2)

    def test_interruption_is_recorded_without_marking_a_success(self):
        def fit(*args):
            raise KeyboardInterrupt()

        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary) / 'run'
            args = SimpleNamespace(output_dir=Path(temporary), run_label='test')
            with patch('sing_experiments.simulation_common.get_dataset', return_value=({}, {})), \
                    redirect_stdout(io.StringIO()), self.assertRaises(KeyboardInterrupt):
                run_case(directory, Case(125, 'gauss', 1), self.study, args, 'GTM', {}, fit)
            result = read_json(directory / 'result.json')
            self.assertEqual(result['status'], 'interrupted')
            self.assertNotIn('roc_auc', result)


@unittest.skipUnless(importlib.util.find_spec('TransportMaps'), 'SING environment required')
class SingLikelihoodTests(unittest.TestCase):
    def test_final_density_is_retained_and_can_score_holdout(self):
        from sing_experiments.sing_helpers import FixedOrdering, load_sing
        rng = np.random.default_rng(14)
        samples = rng.multivariate_normal([0, 0], [[1, 0.6], [0.6, 1]], size=200)
        sing, history = load_sing(batch_size=16, retain_density=True, progress=True)
        output = io.StringIO()
        with redirect_stdout(output):
            graph = sing(samples, 2, FixedOrdering(), 1.0)
        self.assertIn('Input Hessians: observations', output.getvalue())
        self.assertIn('Delta-method derivatives: observations', output.getvalue())
        self.assertEqual(graph.shape, (2, 2))
        self.assertIn('density', history[-1])
        self.assertTrue(all('density' not in entry for entry in history[:-1]))
        holdout = history[-1]['density'].log_pdf(samples[:5])
        self.assertEqual(holdout.shape, (5,))
        self.assertTrue(np.isfinite(holdout).all())


if __name__ == '__main__':
    unittest.main()
