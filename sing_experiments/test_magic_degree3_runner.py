"""Verify degree-3 selection and argument forwarding without model fitting."""
from contextlib import redirect_stderr, redirect_stdout
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from sing_experiments import run_magic_classification as runner
from sing_experiments import run_magic_classification_degree3 as degree3


class MagicDegree3RunnerTests(unittest.TestCase):
    def test_only_degree3_is_launched_and_options_are_forwarded(self):
        with patch.object(runner.subprocess, 'run') as launch, redirect_stdout(io.StringIO()):
            launch.return_value.returncode = 0
            self.assertEqual(degree3.main(['--batch-size', '4']), 0)
        launch.assert_called_once()
        command = launch.call_args.args[0]
        self.assertEqual(command[command.index('--worker-order') + 1], '3')
        self.assertEqual(command[command.index('--batch-size') + 1], '4')
        self.assertEqual(command[command.index('--optimizer-maxiter') + 1], '150')
        self.assertTrue(command[command.index('--output-dir') + 1].endswith('results/magic_classification_degree3_fullsample'))

    def test_defaults_and_worker_roundtrip_use_all_observations(self):
        with patch.object(runner.subprocess, 'run') as launch, redirect_stdout(io.StringIO()):
            launch.return_value.returncode = 0
            self.assertEqual(degree3.main([]), 0)
        command = launch.call_args.args[0]
        for flag in ['--batch-size', '--logpdf-batch-size']:
            self.assertEqual(command[command.index(flag) + 1], 'all')
        with patch.object(runner, 'run_order') as worker:
            self.assertEqual(runner.main(command[3:]), 0)
        self.assertIsNone(worker.call_args.args[0].batch_size)
        self.assertIsNone(worker.call_args.args[0].logpdf_batch_size)
        with patch.object(runner.subprocess, 'run') as launch, redirect_stdout(io.StringIO()):
            launch.return_value.returncode = 0
            runner.main([])
        command = launch.call_args.args[0]
        self.assertEqual(command[command.index('--batch-size') + 1], '16')
        self.assertEqual(command[command.index('--logpdf-batch-size') + 1], '512')

    def test_rejects_degree2_including_worker_override(self):
        for flags in [['--orders', '2'], ['--orders', '2', '3'], ['--worker-order', '2']]:
            with self.subTest(flags=flags), redirect_stderr(io.StringIO()), \
                    patch.object(runner.subprocess, 'run') as launch, self.assertRaises(SystemExit) as caught:
                degree3.main(flags)
            self.assertEqual(caught.exception.code, 2)
            launch.assert_not_called()

    def test_existing_runner_still_defaults_to_both_orders(self):
        with patch.object(runner.subprocess, 'run') as launch, redirect_stdout(io.StringIO()):
            launch.return_value.returncode = 0
            self.assertEqual(runner.main([]), 0)
        self.assertEqual([call.args[0][call.args[0].index('--worker-order') + 1]
                          for call in launch.call_args_list], ['2', '3'])

    def test_hadron_only_command_preserves_class_selection_and_iteration_limit(self):
        with patch.object(runner.subprocess, 'run') as launch, redirect_stdout(io.StringIO()):
            launch.return_value.returncode = 0
            degree3.main(['--groups', 'h', '--optimizer-maxiter', '300',
                          '--optimizer-derivatives', '1',
                          '--reuse-results-from', '/previous', '--output-dir', '/new'])
        command = launch.call_args.args[0]
        with patch.object(runner, 'run_order') as worker:
            runner.main(command[3:])
        args = worker.call_args.args[0]
        self.assertEqual(args.groups, ['h'])
        self.assertEqual(args.optimizer_maxiter, 300)
        self.assertEqual(args.optimizer_derivatives, 1)
        self.assertEqual(args.reuse_results_from, Path('/previous'))
        self.assertIsNone(args.batch_size)

    def test_checkpoint_reuse_keeps_source_and_records_its_original_cap(self):
        old = dict(config=dict(optimizer_maxiter=150, optimizer_derivatives=2, delta=1), data_sha256='data')
        new = dict(config=dict(optimizer_maxiter=300, optimizer_derivatives=1, delta=1), data_sha256='data')
        arrays = {key: np.ones((2, 2)) for key in
                  ['graph', 'thresholded_scores', 'initial_scores', 'final_scores', 'map_coefficients']}
        arrays.update({f'logpdf_{g}_{s}': np.ones(2) for g in runner.GROUPS for s in runner.SPLITS})
        with tempfile.TemporaryDirectory() as temporary, redirect_stdout(io.StringIO()):
            source, target = Path(temporary) / 'source', Path(temporary) / 'target'
            source.mkdir()
            target.mkdir()
            runner.save_json(source / 'run_config.json', old)
            runner.save_json(source / 'class_g.json', dict(group='g'))
            np.savez(source / 'class_g.npz', **arrays)
            before = {p.name: p.read_bytes() for p in source.iterdir()}
            runner.reuse_class_checkpoint('g', source, target, new)
            manifest = json.loads((target / 'class_g.json').read_text())
            self.assertEqual(manifest['optimizer_maxiter'], 150)
            self.assertEqual(manifest['optimizer_derivatives'], 2)
            self.assertEqual((target / 'class_g.npz').read_bytes(), before['class_g.npz'])
            self.assertEqual({p.name: p.read_bytes() for p in source.iterdir()}, before)
            with self.assertRaises(ValueError):
                runner.reuse_class_checkpoint('g', source, target, dict(new, data_sha256='different'))
            with self.assertRaises(ValueError):
                runner.reuse_class_checkpoint('g', source, target,
                    dict(new, config=dict(new['config'], delta=2)))

    def test_hadron_only_fits_h_and_combines_saved_gamma(self):
        with tempfile.TemporaryDirectory() as temporary, redirect_stdout(io.StringIO()):
            root = Path(temporary)
            args = SimpleNamespace(data_path=root / 'data.csv', split_indices=root / 'split.npz',
                output_dir=root / 'new', worker_order=3, delta=1., fit_seed=25,
                batch_size=None, logpdf_batch_size=None, optimizer_derivatives=1,
                optimizer_maxiter=300, groups=['h'], reuse_results_from=root / 'old')
            args.split_indices.write_bytes(b'split')
            rows = {g: {s: np.arange(2) for s in runner.SPLITS} for g in runner.GROUPS}
            arrays = {f'logpdf_{g}_{s}': np.array([-2., -3.]) for g in runner.GROUPS for s in runner.SPLITS}
            info = dict(group='h', optimizer_maxiter=300, optimizer_derivatives=1, solver_diagnostics=[])
            def reuse(group, source, target, identity):
                np.savez(target / 'class_g.npz', **arrays)
                runner.save_json(target / 'class_g.json', dict(info, group='g', optimizer_maxiter=150,
                                                              optimizer_derivatives=2))
            with patch.object(runner, 'load_data', return_value=({}, rows, np.zeros(10), np.ones(10), {}, 'abc')), \
                    patch.object(runner, 'fit_class', return_value=(arrays, info)) as fit, \
                    patch.object(runner, 'reuse_class_checkpoint', side_effect=reuse):
                runner.run_order(args)
            fit.assert_called_once()
            self.assertEqual(fit.call_args.args[0], 'h')
            directory = args.output_dir / 'p3_delta1_seed25_abc'
            summary = json.loads((directory / 'summary.json').read_text())
            self.assertEqual(summary['config']['optimizer_maxiter_by_class'], {'g': 150, 'h': 300})
            self.assertEqual(summary['config']['optimizer_derivatives_by_class'], {'g': 2, 'h': 1})
            self.assertEqual(json.loads((directory / 'status.json').read_text())['status'], 'ok')


if __name__ == '__main__':
    unittest.main()
