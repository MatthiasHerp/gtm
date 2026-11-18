"""Check full-sample derivative equivalence and live output without a large fit."""
from contextlib import redirect_stdout
import importlib.util
import io
from pathlib import Path
from types import SimpleNamespace
import threading
import unittest

import numpy as np

from sing_experiments.simulation_common import progress_output


class ProgressTests(unittest.TestCase):
    def test_live_output_and_heartbeat_reach_both_terminal_and_log(self):
        heartbeat = threading.Event()

        class Terminal(io.StringIO):
            def write(self, value):
                if "RUNNING" in value:
                    heartbeat.set()
                return super().write(value)

        terminal, log = Terminal(), io.StringIO()
        with redirect_stdout(terminal), progress_output(log, "seed 1", True, heartbeat_seconds=0.01) as output:
            output.write("START full-sample derivatives\n")
            self.assertTrue(heartbeat.wait(timeout=2))
        self.assertEqual(terminal.getvalue(), log.getvalue())
        self.assertIn("RUNNING seed 1", terminal.getvalue())
        self.assertIn("START full-sample derivatives", terminal.getvalue())


@unittest.skipUnless(importlib.util.find_spec("TransportMaps"), "SING environment required")
class FullSampleTests(unittest.TestCase):
    def test_all_2000_rows_in_one_call_matches_batched_score_and_variance(self):
        from sing_experiments.sing_helpers import load_sing

        class Density:
            def __init__(self):
                self.calls = {"hessian": [], "gradient": [], "fisher": []}
                self.transport_map = SimpleNamespace(n_coeffs=3)

            def hess_x_log_pdf(self, x):
                self.calls["hessian"].append(len(x))
                return (1 + x[:, 0, None, None]) * np.array([[1., 0.2], [0.2, 2.]])

            def grad_a_hess_x_log_pdf(self, x):
                self.calls["gradient"].append(len(x))
                return np.ones((len(x), 3, 2, 2))

            def hess_a_log_pdf(self, x):
                self.calls["fisher"].append(len(x))
                return np.broadcast_to(-np.eye(3), (len(x), 3, 3))

        values = np.random.default_rng(4).normal(size=(2000, 2))
        results = []
        for batch_size in (16, None):
            density = Density()
            output = io.StringIO()
            with redirect_stdout(output):
                sing, _ = load_sing(batch_size=batch_size, progress=True,
                                     sing_directory=(Path("/opt/gtm/.external/SING")
                                                     if Path("/opt/gtm/.external/SING").exists() else None))
                sing.__globals__["_record_solve"](dict(success=True, jac=np.zeros(3)))
                gp = sing.__globals__["GP"]
                scores = gp.gen_precision(density, values)
                variance = gp.var_omega(density, values)
            results.append((scores, variance))
            if batch_size is None:
                self.assertEqual(density.calls, {"hessian": [2000, 2000],
                                                 "gradient": [2000], "fisher": [2000]})
                self.assertIn("observations 1-2000 of 2000", output.getvalue())
                self.assertIn("DONE delta-method Fisher matrix solve", output.getvalue())
        np.testing.assert_allclose(results[0][0], results[1][0], rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(results[0][1], results[1][1], rtol=1e-12, atol=1e-12)


if __name__ == "__main__":
    unittest.main()
