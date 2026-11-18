"""Check memory arithmetic and full-sample likelihood calls without long fits."""
from contextlib import redirect_stdout
import io
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from sing_experiments.estimate_sing_memory import estimate_memory
from sing_experiments import run_magic_classification as magic
from sing_experiments import run_sing_simulation as simulation


class MemoryTests(unittest.TestCase):
    def test_dense_degree3_storage_and_batch_scaling(self):
        estimate = estimate_memory(6576)
        self.assertEqual(estimate['n_coeffs'], 1000)
        self.assertEqual(estimate['arrays_bytes']['coefficient_hessians'], 8 * 6576 * 1000 ** 2)
        self.assertEqual(estimate['arrays_bytes']['coefficient_input_hessian_gradients'],
                         8 * 6576 * 1000 * 10 ** 2)
        small = estimate_memory(6576, batch_size=16)
        self.assertEqual(small['arrays_bytes']['coefficient_hessians'] * 411,
                         estimate['arrays_bytes']['coefficient_hessians'])
        self.assertEqual(estimate_memory(5, batch_size=16)['batch_observations'], 5)
        self.assertEqual(estimate_memory(10, n_coeffs=3)['n_coeffs'], 3)
        for invalid in [0, -1, True]:
            with self.assertRaises(ValueError):
                estimate_memory(10, batch_size=invalid)

    def test_magic_evaluates_each_whole_split_once(self):
        calls = []
        data = {g: {s: np.ones((n, 10)) for s, n in [('train', 1025), ('validate', 513), ('test', 1027)]}
                for g in magic.GROUPS}
        density = SimpleNamespace(transport_map=SimpleNamespace(coeffs=np.zeros(2)))
        density.log_pdf = lambda points: calls.append(len(points)) or -np.ones(len(points))
        history = [dict(density=density, omega_raw=np.ones((10, 10)), solver_logs=[
            dict(success=True, accepted_by_gradient=False, message='ok', nit=1, jac=np.zeros(2))])]
        args = SimpleNamespace(worker_order=3, fit_seed=25, batch_size=None, logpdf_batch_size=None,
                               optimizer_derivatives=2, optimizer_maxiter=150, sing_dir=None, delta=1.)
        with tempfile.TemporaryDirectory() as directory, \
                patch.object(magic, 'load_sing', return_value=(lambda *a: np.ones((10, 10)), history)) as load, \
                redirect_stdout(io.StringIO()):
            magic.fit_class('g', data, {}, np.ones(10), args, Path(directory))
        self.assertIsNone(load.call_args.kwargs['batch_size'])
        self.assertEqual(calls, [1025, 513, 1027, 1025, 513, 1027])

    def test_simulation_evaluates_whole_train_and_test_once(self):
        calls = []
        density = SimpleNamespace(log_pdf=lambda x: calls.append(len(x)) or -np.ones(len(x)))
        history = [dict(density=density, omega_raw=np.ones((2, 2)), solver_logs=[
            dict(success=True, accepted_by_gradient=False, message='ok', nit=1, jac=np.zeros(2))])]
        data = dict(x_train=np.ones((1025, 2)), x_test=np.ones((2051, 2)), scale=np.ones(2),
                    variable_order=np.arange(2), truth=np.array([[0, 1], [1, 0]]))
        config = dict(batch_size=None, loglik_batch_size=None, optimizer_derivatives=2,
                      optimizer_maxiter=300, delta=1., progress=False)
        with tempfile.TemporaryDirectory() as directory, \
                patch.object(simulation, 'load_sing', return_value=(lambda *a: np.ones((2, 2)), history)), \
                patch.object(simulation, 'versions', return_value={}), redirect_stdout(io.StringIO()):
            simulation.fit_sing(data, dict(seed_streams=dict(fit=25)), Path(directory), 3, config)
        self.assertEqual(calls, [1025, 2051])


if __name__ == '__main__':
    unittest.main()
