"""Check degree-specific job selection and isolation without fitting SING."""
from contextlib import redirect_stdout
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from sing_experiments import run_sing_simulation_degree2 as degree2
from sing_experiments import run_sing_simulation_degree3 as degree3
from sing_experiments.run_sing_simulation import HERE
from sing_experiments.simulation_common import DEFAULT_STUDY, atomic_json, read_json


class DegreeRunnerTests(unittest.TestCase):
    def list_jobs(self, runner, extra=()):
        output = io.StringIO()
        with redirect_stdout(output):
            self.assertEqual(runner.main(['--list-only', *extra]), 0)
        return output.getvalue().splitlines()

    def test_degree2_keeps_original_grid_and_settings(self):
        lines = self.list_jobs(degree2)
        jobs = [line for line in lines if line.startswith('SING degree')]
        self.assertEqual(len(jobs), 90)
        self.assertTrue(all(line.startswith('SING degree 2:') for line in jobs))
        self.assertIn('maxiter=150; run label=default', lines[0])

    def test_degree3_reduced_grid_and_order(self):
        lines = self.list_jobs(degree3)
        jobs = [line for line in lines if line.startswith('SING degree')]
        self.assertEqual(len(jobs), 12)
        self.assertTrue(all('n2000_' in line for line in jobs[:6]))
        self.assertTrue(all('n500_' in line for line in jobs[6:]))
        self.assertTrue(all(line.endswith(('seed01', 'seed02')) for line in jobs))
        self.assertEqual({line.split('_')[1] for line in jobs}, {'gauss', 'exp', 't'})
        self.assertIn('maxiter=300; run label=degree3_maxiter300', lines[0])
        self.assertEqual(len(self.list_jobs(degree3, ['--n-train', '2000'])) - 1, 6)

    def test_degree_entry_points_reject_other_degree(self):
        for runner, other in [(degree2, '3'), (degree3, '2')]:
            with self.subTest(other=other), self.assertRaises(SystemExit) as caught:
                runner.main(['--list-only', '--degrees', other])
            self.assertEqual(caught.exception.code, 2)

    def test_new_degree3_profile_preserves_existing_results_and_study(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            original = output / 'runs/sing/default/degree2/n2000_gauss_seed01/result.json'
            atomic_json(original, {'status': 'ok', 'sentinel': 'keep degree 2'})
            old3 = output / 'runs/sing/default/degree3/config.json'
            atomic_json(old3, dict(read_json(HERE / 'sing_simulation_config.json'), degree=3,
                                   study=read_json(DEFAULT_STUDY)))
            before = (original.read_bytes(), old3.read_bytes())
            # Keep import/checkout checks independent of the local environment.
            with patch('sing_experiments.run_sing_simulation.importlib.util.find_spec', return_value=True), \
                 patch('sing_experiments.run_sing_simulation.Path.exists', return_value=True), \
                 patch('sing_experiments.run_sing_simulation.run_case', return_value=True) as run_case, \
                 patch('sing_experiments.run_sing_simulation.ensure_config') as ensure_config, \
                 redirect_stdout(io.StringIO()):
                self.assertEqual(degree3.main(['--output-dir', str(output)]), 0)
            self.assertEqual(run_case.call_count, 12)
            profile, config = ensure_config.call_args.args
            self.assertEqual(profile, output / 'runs/sing/degree3_maxiter300/degree3/config.json')
            self.assertEqual(config['optimizer_maxiter'], 300)
            self.assertEqual(config['study'], read_json(DEFAULT_STUDY))
            self.assertTrue(all(call.args[4] == 'SING degree 3' for call in run_case.call_args_list))
            self.assertEqual(before, (original.read_bytes(), old3.read_bytes()))


if __name__ == '__main__':
    unittest.main()
