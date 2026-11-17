"""Check the notebook's graph comparison using cached GTM tables and small graphs."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


class MagicGraphComparisonTests(unittest.TestCase):
    def setUp(self):
        notebook = json.loads((ROOT / 'sing_experiments/sing_magic_classification.ipynb').read_text())
        source = next(''.join(c['source']) for c in notebook['cells']
                      if c.get('id') == 'magic-graph-functions')
        self.env = dict(np=np, pd=pd, plt=plt, json=json, display=Mock(),
                        GTM_CI_DIR=ROOT / 'demos/ci_tables', GTM_IAE_THRESHOLD=0.1)
        exec(compile(source, '<notebook graph functions>', 'exec'), self.env)
        self.names = [f'y_{i}' for i in range(10)]

    def tearDown(self):
        plt.close('all')

    def test_gtm_threshold_and_variable_order(self):
        for group in ['g', 'h']:
            graph, iae = self.env['load_gtm_graph'](group, self.names)
            table = pd.read_csv(self.env['GTM_CI_DIR'] / f'conditional_independence_table_group_{group}.csv')
            expected = {tuple(sorted((int(r.var_row), int(r.var_col))))
                        for r in table.itertuples() if r.iae >= 0.1 and r.iae > 0}
            self.assertEqual(set(zip(*np.where(np.triu(graph, 1)))), expected)
            reversed_graph, reversed_iae = self.env['load_gtm_graph'](group, self.names[::-1])
            np.testing.assert_array_equal(reversed_graph, graph[::-1, ::-1])
            np.testing.assert_array_equal(reversed_iae, iae[::-1, ::-1])

    def test_agreement_counts_and_empty_graphs(self):
        sing, gtm = [np.zeros((10, 10), dtype=bool) for _ in range(2)]
        for graph, edges in [(sing, [(0, 1), (0, 2)]), (gtm, [(0, 1), (2, 3)])]:
            for i, j in edges:
                graph[i, j] = graph[j, i] = True
        metrics, edges = self.env['compare_graph_edges'](sing, gtm, self.names, np.zeros((10, 10)))
        self.assertEqual(metrics, dict(sing_edges=2, gtm_edges=2, shared_edges=1,
                                      sing_only=1, gtm_only=1, neither=42, shd=2, jaccard=1 / 3))
        self.assertEqual(len(edges), 45)
        self.assertEqual(len(edges.loc[edges.SING != edges.GTM]), 2)
        empty = np.zeros((10, 10), dtype=bool)
        metrics, _ = self.env['compare_graph_edges'](empty, empty, self.names, empty)
        self.assertEqual((metrics['jaccard'], metrics['shd']), (1, 0))
        sing[1, 0] = False
        with self.assertRaises(ValueError):
            self.env['compare_graph_edges'](sing, gtm, self.names, empty)

    def test_threshold_boundary_and_reject_incomplete_gtm_table(self):
        with tempfile.TemporaryDirectory() as directory:
            self.env['GTM_CI_DIR'] = Path(directory)
            table = pd.DataFrame([(j, i, 0.1) for i in range(10) for j in range(i + 1, 10)],
                                 columns=['var_row', 'var_col', 'iae'])
            path = Path(directory) / 'conditional_independence_table_group_g.csv'
            table.to_csv(path)
            graph, _ = self.env['load_gtm_graph']('g', self.names)
            self.assertEqual(graph.sum() // 2, 45)
            table.iloc[:-1].to_csv(path)
            with self.assertRaises(ValueError):
                self.env['load_gtm_graph']('g', self.names)

    def test_render_both_classes_and_missing_gtm(self):
        arrays = {f'{g}_graph': self.env['load_gtm_graph'](g, self.names)[0] for g in ['h', 'g']}
        arrays.update({f'{g}_final_scores': self.env['load_gtm_graph'](g, self.names)[1]
                       for g in ['h', 'g']})
        result = dict(summary=dict(config=dict(order=2, variable_order=self.names)), arrays=arrays)
        comparison = self.env['analyze_graphs'](result)
        self.assertEqual(len(comparison['summary']), 2)
        self.assertTrue((comparison['summary'].shd == 0).all())
        np.testing.assert_allclose(comparison['summary'].spearman_rho, 1)
        self.assertTrue(comparison['summary'].same_edge_order.all())
        self.assertTrue(comparison['summary'].top_x_same_edges.all())
        np.testing.assert_allclose(comparison['summary'].top_x_overlap_fraction, 1)
        figure = plt.gcf()
        self.assertEqual(len(figure.axes), 4)
        with tempfile.TemporaryDirectory() as directory:
            figure.savefig(Path(directory) / 'graphs.png')
            self.assertGreater((Path(directory) / 'graphs.png').stat().st_size, 0)
            self.env['GTM_CI_DIR'] = Path(directory)
            comparison = self.env['analyze_graphs'](result)
            self.assertTrue(comparison['summary'].empty)
            self.assertEqual(comparison['edge_tables'], {})

    def test_spearman_ranking_and_ties(self):
        def matrix(values):
            result = np.zeros((4, 4))
            result[np.triu_indices(4, 1)] = values
            return result + result.T

        names = ['a', 'b', 'c', 'd']
        scores = matrix([1, 2, 3, 4, 5, 6])
        compare = self.env['compare_graph_rankings']
        metrics, ranks = compare(scores ** 2, scores, names)
        self.assertAlmostEqual(metrics['spearman_rho'], 1)
        self.assertTrue(metrics['same_edge_order'])
        self.assertEqual(ranks.SING_rank.tolist(), [6, 5, 4, 3, 2, 1])
        metrics, _ = compare(matrix([6, 5, 4, 3, 2, 1]), scores, names)
        self.assertAlmostEqual(metrics['spearman_rho'], -1)
        self.assertFalse(metrics['same_edge_order'])
        # One adjacent swap: rho = 1 - 6 * 2 / (6 * (6**2 - 1)).
        metrics, _ = compare(matrix([2, 1, 3, 4, 5, 6]), scores, names)
        self.assertAlmostEqual(metrics['spearman_rho'], 1 - 12 / 210)
        tied = matrix([1, 1, 3, 4, 5, 6])
        metrics, ranks = compare(tied, tied ** 2, names)
        self.assertAlmostEqual(metrics['spearman_rho'], 1)
        self.assertTrue(metrics['same_edge_order'])
        self.assertEqual(ranks.SING_rank.tolist()[:2], [5.5, 5.5])
        metrics, _ = compare(matrix([0] * 6), scores, names)
        self.assertTrue(np.isnan(metrics['spearman_rho']))
        self.assertFalse(metrics['same_edge_order'])
        scores[0, 1] = np.nan
        with self.assertRaises(ValueError):
            compare(scores, tied, names)

    def test_gtm_top_x_matches_significant_sing_set(self):
        edges = pd.DataFrame(dict(variable_1=['a', 'a', 'b'], variable_2=['b', 'c', 'c'],
                                  SING=[True, False, True], GTM_IAE=[0.2, 0.1, 0.3]))
        compare = self.env['compare_top_gtm_edges']
        metrics, top = compare(edges)
        self.assertEqual(metrics, dict(top_x=2, top_x_overlap=2, top_x_overlap_fraction=1,
                                       top_x_same_edges=True, top_x_cutoff_tied=False))
        self.assertEqual(top.tolist(), [True, False, True])
        edges['SING'] = [True, True, False]
        metrics, top = compare(edges)
        self.assertEqual((metrics['top_x'], metrics['top_x_overlap']), (2, 1))
        self.assertEqual(metrics['top_x_overlap_fraction'], 0.5)
        self.assertFalse(metrics['top_x_same_edges'])
        edges['SING'] = False
        metrics, top = compare(edges)
        self.assertEqual((metrics['top_x'], top.sum()), (0, 0))
        self.assertTrue(metrics['top_x_same_edges'])
        self.assertEqual(metrics['top_x_overlap_fraction'], 1)
        edges['SING'] = True
        metrics, top = compare(edges)
        self.assertEqual((metrics['top_x'], top.sum()), (3, 3))
        self.assertTrue(metrics['top_x_same_edges'])
        self.assertFalse(metrics['top_x_cutoff_tied'])
        edges['SING'] = [True, False, True]
        edges['GTM_IAE'] = [0.1, 0.1, 0.3]
        metrics, top = compare(edges)
        self.assertTrue(metrics['top_x_cutoff_tied'])
        self.assertTrue(metrics['top_x_same_edges'])
        shuffled_metrics, shuffled_top = compare(edges.iloc[::-1])
        self.assertEqual(shuffled_metrics, metrics)
        pd.testing.assert_series_equal(shuffled_top.sort_index(), top.sort_index())

    def test_kendall_subsets_and_ties(self):
        compare = self.env['compare_kendall_rankings']
        edges = pd.DataFrame(dict(SING_score=[1, 2, 3, 4], GTM_IAE=[4, 3, 2, 1],
                                  SING=[True, True, False, False],
                                  GTM_top_x=[False, False, True, True]))
        metrics = compare(edges)
        for value in metrics.values():
            self.assertAlmostEqual(value, -1)
        edges['GTM_IAE'] = [1, 2, 4, 3]
        metrics = compare(edges)
        self.assertAlmostEqual(metrics['kendall_tau_all'], 2 / 3)
        self.assertAlmostEqual(metrics['kendall_tau_sing_significant'], 1)
        self.assertAlmostEqual(metrics['kendall_tau_gtm_top_x'], -1)
        edges['SING_score'] = [1, 1, 3, 4]
        self.assertTrue(np.isnan(compare(edges)['kendall_tau_sing_significant']))
        edges['GTM_top_x'] = [False, False, False, True]
        self.assertTrue(np.isnan(compare(edges)['kendall_tau_gtm_top_x']))
        edges['SING_score'] = edges.GTM_IAE = [1, 1, 3, 4]
        self.assertAlmostEqual(compare(edges)['kendall_tau_all'], 1)

    def test_test_loglik_mean_and_cache_alignment(self):
        result = dict(summary=dict(config=dict(order=2), split_metadata=dict(sha256='dataset'),
                                   loglik_scale='original observations'), arrays={})
        cache = dict(metadata_json=json.dumps(dict(data_sha256='dataset', loglik_scale='original observations')))
        for group in ['g', 'h']:
            result['arrays'][f'{group}_test_row_indices'] = np.array([11, 12])
            result['arrays'][f'{group}_logpdf_{group}_test'] = np.array([-5., -7.])
            cache[f'{group}_test_row_indices'] = np.array([11, 12])
            cache[f'{group}_test_logpdf'] = np.array([-3., -5.])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'gtm.npz'
            self.env['GTM_TEST_LOGLIK_PATH'] = path
            compare = self.env['compare_test_loglik']
            table = compare(result)
            self.assertTrue(table.GTM_mean_test_loglik.isna().all())
            np.savez(path, **cache)
            table = compare(result)
            np.testing.assert_allclose(table.SING_mean_test_loglik, -6)
            np.testing.assert_allclose(table.GTM_mean_test_loglik, -4)
            np.testing.assert_allclose(table.SING_minus_GTM, -2)
            cache['g_test_row_indices'] = np.array([12, 11])
            np.savez(path, **cache)
            with self.assertRaises(AssertionError):
                compare(result)
            cache['g_test_row_indices'] = np.array([11, 12])
            cache['metadata_json'] = json.dumps(dict(data_sha256='other', loglik_scale='original observations'))
            np.savez(path, **cache)
            with self.assertRaises(ValueError):
                compare(result)
            cache['metadata_json'] = json.dumps(dict(data_sha256='dataset', loglik_scale='normalized'))
            np.savez(path, **cache)
            with self.assertRaises(ValueError):
                compare(result)


if __name__ == '__main__':
    unittest.main()
