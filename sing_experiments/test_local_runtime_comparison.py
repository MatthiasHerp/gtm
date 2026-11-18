"""Check benchmark timing boundaries and preservation of the GTM fit protocol."""
from contextlib import redirect_stdout
import csv
import importlib.util
import io
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from sing_experiments import run_local_runtime_comparison as runner


class RuntimeTests(unittest.TestCase):
    def test_default_and_three_seed_job_counts(self):
        for arguments, expected in (([], 6), (["--seeds", "1", "2", "3"], 18)):
            output = io.StringIO()
            with redirect_stdout(output):
                self.assertEqual(runner.main([*arguments, "--list-only"]), 0)
            self.assertEqual(output.getvalue().count("marginals=gauss device=cpu"), expected)

    def test_timer_excludes_preparation_and_exports_only_runtime(self):
        events = []
        train = np.zeros((125, 10))

        def dataset(*args):
            events.append("data")
            return {"x_train": train}, {"dataset_hash": "shared-data", "seed_streams": {"fit": 4}}

        def load(**kwargs):
            events.append("import")
            self.assertFalse(kwargs["retain_density"])

            def algorithm(values, degree, ordering, delta):
                events.append("fit")
                np.testing.assert_array_equal(values, train)
                self.assertEqual(degree, 2)
                return np.eye(10)

            return algorithm, []

        def clock():
            events.append("clock")
            return 10.0 if events.count("clock") == 1 else 13.25

        with tempfile.TemporaryDirectory() as temporary:
            args = runner.parser().parse_args(["--worker", "sing", "--n-train", "125",
                                               "--output-dir", temporary])
            with patch("sing_experiments.simulation_common.get_dataset", side_effect=dataset), \
                    patch("sing_experiments.sing_helpers.load_sing", side_effect=load), \
                    patch.object(runner.time, "perf_counter", side_effect=clock), redirect_stdout(io.StringIO()):
                self.assertEqual(runner.run_worker(args), 0)
            self.assertEqual(events, ["data", "import", "clock", "fit", "clock"])
            output = Path(temporary)
            self.assertEqual([path.name for path in output.iterdir()], ["runtimes.csv"])
            with (output / "runtimes.csv").open() as handle:
                row = next(csv.DictReader(handle))
            self.assertEqual(row["status"], "ok")
            self.assertEqual(float(row["total_runtime_seconds"]), 3.25)
            self.assertEqual(row["dataset_hash"], "shared-data")


@unittest.skipUnless(importlib.util.find_spec("torch") and importlib.util.find_spec("optuna"),
                     "GTM Python environment required")
class GtmRuntimeTests(unittest.TestCase):
    def test_runtime_mode_preserves_tuning_pilots_refits_and_graph_estimation(self):
        from sing_experiments import simulation_gtm as gtm
        from sing_experiments.simulation_common import read_json
        config = dict(read_json(runner.HERE / "gtm_simulation_config.json"), n_trials=2)
        train = np.random.default_rng(1).normal(size=(125, 10))
        # No test observations or truth graph: runtime mode must not need them.
        data = dict(train=train, x_train=train, variable_order=np.arange(10))
        metadata = dict(seed_streams={"fit": 1, "validation": 2})

        def fit(*args, **kwargs):
            model = SimpleNamespace(_simulation_pretrained_state={"state_dict": {}, "epochs": 1})
            return model, dict(epochs=1, pretrain_epochs=1)

        with tempfile.TemporaryDirectory() as temporary, \
                patch.object(gtm, "_fit", side_effect=fit) as fits, \
                patch.object(gtm, "adaptive_weights", return_value=np.ones((10, 10))) as weights, \
                patch.object(gtm, "logpdf", return_value=np.ones(25)) as likelihood, \
                patch.object(gtm, "dependence_scores") as scores:
            self.assertEqual(gtm.fit_gtm(data, metadata, Path(temporary), config, "cpu", runtime_only=True),
                             ({}, {}))
            self.assertEqual(fits.call_count, config["n_trials"] + 3)
            self.assertEqual([call.args[2].shape[0] for call in fits.call_args_list],
                             [100, 100, 100, 125, 125])
            self.assertEqual(weights.call_count, 2)
            self.assertEqual(likelihood.call_count, config["n_trials"])
            scores.assert_called_once()
            np.testing.assert_array_equal(scores.call_args.args[1], train)
            self.assertFalse((Path(temporary) / "model_state.pt").exists())
            self.assertFalse((Path(temporary) / "refit_history.json").exists())


if __name__ == "__main__":
    unittest.main()
