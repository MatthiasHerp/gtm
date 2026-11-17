"""Verify degree-3 selection and argument forwarding without model fitting."""
from contextlib import redirect_stderr, redirect_stdout
import io
import unittest
from unittest.mock import patch

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
        self.assertTrue(command[command.index('--output-dir') + 1].endswith('results/magic_classification'))

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


if __name__ == '__main__':
    unittest.main()
