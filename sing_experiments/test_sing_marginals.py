"""Numerical checks for marginal transforms and the known-density audit.

Run with: .venv-sing/bin/python -m unittest sing_experiments.test_sing_marginals
"""
import unittest

import numpy as np
import pandas as pd
import pyvinecopulib as pv
from scipy.stats import expon, norm, t as student_t

from sing_experiments.sing_helpers import (
    _marginal_ppf, make_rvine, marginal_cdf, oracle_mixed_hessians,
    nonparanormal_transform, reference_graphs,
)


class MarginalTests(unittest.TestCase):
    def test_default_is_original_normal_sampler(self):
        vine, data, table = make_rvine(n=100, dimensionality=4, sample_seed=7)
        uniforms = vine.simulate(100, seeds=[7])
        np.testing.assert_array_equal(data, norm.ppf(np.clip(uniforms, 1e-12, 1 - 1e-12)))
        self.assertEqual(table.attrs['marginal_spec']['kind'], 'gauss')

    def test_margins_preserve_the_vine_and_copula_uniforms(self):
        baseline_vine, _, baseline_pairs = make_rvine(n=100, dimensionality=4, sample_seed=7)
        expected = baseline_vine.simulate(100, seeds=[7])
        for kind in ('gauss', 'exp', 't', 'gaussmix'):
            with self.subTest(marginals=kind):
                vine, data, pairs = make_rvine(n=100, dimensionality=4, sample_seed=7, marginals=kind)
                self.assertEqual(vine.to_json(), baseline_vine.to_json())
                pd.testing.assert_frame_equal(pairs, baseline_pairs)
                self.assertTrue(np.isfinite(data).all())
                np.testing.assert_allclose(marginal_cdf(data, pairs.attrs['marginal_spec']),
                                           expected, atol=2e-11, rtol=2e-11)
                if kind == 'exp':
                    self.assertTrue((data > 0).all())
                    np.testing.assert_allclose(data, expon.ppf(expected))
                elif kind == 't':
                    np.testing.assert_allclose(data, student_t.ppf(expected, df=2.5))

    def test_mixture_parameters_and_tail_quantiles(self):
        _, _, table = make_rvine(n=2, dimensionality=4, marginals='gaussmix')
        spec = table.attrs['marginal_spec']
        means, variances = np.asarray(spec['means']), np.asarray(spec['variances'])
        self.assertEqual(means.shape, (4, 3))
        self.assertEqual(variances.shape, (4, 3))
        self.assertTrue(((means >= -2) & (means <= 2)).all())
        self.assertTrue(((variances >= 0.5) & (variances <= 2)).all())
        np.testing.assert_allclose(spec['weights'], [1 / 3] * 3)
        u = np.repeat(np.array([1e-12, 1e-8, 0.01, 0.5, 0.99, 1 - 1e-8, 1 - 1e-12])[:, None], 4, axis=1)
        x = _marginal_ppf(u, spec)
        self.assertTrue((np.diff(x, axis=0) > 0).all())
        np.testing.assert_allclose(marginal_cdf(x, spec), u, atol=3e-16, rtol=2e-13)
        _, _, other = make_rvine(n=20, dimensionality=4, sample_seed=99, marginals='gaussmix')
        self.assertEqual(spec, other.attrs['marginal_spec'])
        _, _, changed = make_rvine(n=2, dimensionality=4, copula_seed=2, marginals='gaussmix')
        self.assertNotEqual(spec, changed.attrs['marginal_spec'])

    def test_unknown_marginals_fail_clearly(self):
        with self.assertRaisesRegex(ValueError, 'marginals must be one of'):
            make_rvine(marginals='unknown')

    def test_oracle_matches_analytic_gaussian_copula_derivatives(self):
        rho = 0.4
        copula = pv.Bicop.from_family(pv.BicopFamily.gaussian, parameters=np.array([[rho]]))
        vine = pv.Vinecop.from_structure(structure=pv.DVineStructure([1, 2]), pair_copulas=[[copula]])
        u = np.array([[0.15, 0.65], [0.5, 0.4], [0.8, 0.2]])
        for kind in ('gauss', 'exp', 't', 'gaussmix'):
            with self.subTest(marginals=kind):
                _, _, table = make_rvine(n=2, dimensionality=2, active_trees=1, marginals=kind)
                spec = table.attrs['marginal_spec']
                x = _marginal_ppf(u, spec)
                if kind == 'gauss':
                    density = norm.pdf(x)
                elif kind == 'exp':
                    density = expon.pdf(x)
                elif kind == 't':
                    density = student_t.pdf(x, df=2.5)
                else:
                    density = (norm.pdf(x[..., None], loc=spec['means'], scale=np.sqrt(spec['variances']))
                               * spec['weights']).sum(axis=-1)
                jacobian = density / norm.pdf(norm.ppf(u))
                expected = rho / (1 - rho**2) * jacobian.prod(axis=1)
                actual = oracle_mixed_hessians(vine, x, step=1e-4, marginal_spec=spec)[:, 0, 1]
                np.testing.assert_allclose(actual, expected, atol=2e-7, rtol=2e-5)

    def test_reference_audit_for_all_four_margins(self):
        for kind in ('gauss', 'exp', 't', 'gaussmix'):
            with self.subTest(marginals=kind):
                vine, data, pairs = make_rvine(n=125, sample_seed=1, marginals=kind)
                spec = pairs.attrs['marginal_spec']
                u = marginal_cdf(data, spec)
                central = data[np.all((u > 0.025) & (u < 0.975), axis=1)][:64]
                self.assertGreaterEqual(len(central), 16)
                scores = np.stack([np.abs(oracle_mixed_hessians(vine, central, step=h,
                                   marginal_spec=spec)).max(axis=0) for h in (1e-3, 5e-4)])
                detected = np.all(scores > 1e-4, axis=0)
                np.fill_diagonal(detected, False)
                _, reference = reference_graphs(pairs, dim=10)
                np.testing.assert_array_equal(detected, reference)


class NonparanormalTests(unittest.TestCase):
    def test_columnwise_scores_and_average_ties(self):
        data = np.array([[4, 0], [1, 10], [1, 5], [3, 5]], dtype=float)
        original = data.copy()
        scores = nonparanormal_transform(data)
        expected_probabilities = np.array([[0.875, 0.125], [0.25, 0.875],
                                           [0.25, 0.5], [0.625, 0.5]])
        np.testing.assert_allclose(norm.cdf(scores), expected_probabilities)
        np.testing.assert_array_equal(data, original)
        self.assertTrue(np.isfinite(scores).all())

    def test_identical_scores_across_generating_margins(self):
        reference = None
        for kind in ('gauss', 'exp', 't', 'gaussmix'):
            with self.subTest(marginals=kind):
                _, data, _ = make_rvine(n=64, dimensionality=4, marginals=kind)
                scores = nonparanormal_transform(data)
                if reference is None:
                    reference = scores
                np.testing.assert_array_equal(scores, reference)

    def test_commutes_with_row_and_column_permutations(self):
        data = np.random.default_rng(7).normal(size=(20, 4))
        row_order = np.arange(19, -1, -1)
        column_order = np.array([2, 0, 3, 1])
        expected = nonparanormal_transform(data)[np.ix_(row_order, column_order)]
        actual = nonparanormal_transform(data[np.ix_(row_order, column_order)])
        np.testing.assert_array_equal(actual, expected)

    def test_invalid_data_and_constant_column(self):
        for data in (np.ones(3), np.empty((0, 2)), np.ones((1, 2)),
                     np.empty((2, 0)), [[1], [np.nan]], [[np.inf], [2]]):
            with self.subTest(data=data):
                with self.assertRaises(ValueError):
                    nonparanormal_transform(data)
        np.testing.assert_array_equal(nonparanormal_transform(np.ones((5, 1))),
                                      np.zeros((5, 1)))


if __name__ == '__main__':
    unittest.main()
