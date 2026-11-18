# SING and GTM comparison

The study is defined in `simulation_study.json`. It has 90 paired datasets:

| Setting | Values |
| --- | --- |
| Training observations N | 125, 500, 2,000 |
| Marginals | `gauss`, `exp` (rate 1), `t` (mean 0, df 2.5, scale 1) |
| Replicate seeds | 1–10; each seed gives a different R-vine and column shuffle |
| Dimension | 10 |
| Dependent vine trees | First 3; later trees are independence copulas |
| Independent test observations | 20,000 per dataset, shared by both models |
| PIT / rank Gaussianization | Disabled |
| Standardization | Training column mean and std, `ddof=0` |
| SING degrees | 2, 3: 180 fits |
| GTM | Paper configuration: 90 fits |

Across sample sizes and marginal choices, a replicate shares its generating
vine, shuffle, and test copula uniforms. Separate deterministic seed streams
control training observations, test observations, shuffling, fitting, and the
validation split. Both models read exactly the same cached datasets. A file
lock prevents simultaneous CPU/GPU jobs from generating different copies.
Each dataset stores the original vine, marginal parameters, permutation,
training statistics, raw observations, standardized/shuffled inputs, truth
matrices, and an oracle reference audit. The oracle is used only to check truth.

## GTM protocol

`gtm_simulation_config.json` fixes the paper architecture:

- One marginal transformation layer with 10 B-spline basis functions per
  variable, knots spanning `[-5, 5]`.
- Three decorrelation layers with 10 basis functions per spline, knots spanning
  `[-10, 10]`.
- No penalty on marginal splines or decorrelation coefficient values.
- 30 seeded Optuna trials jointly tune adaptive lasso and decorrelation
  first- and second-difference penalties. The smoothness penalties use
  `[1e-7, 30]` on a linear scale; lasso uses `[1e-7, 1]` on a log scale,
  matching the existing experiment code's search ranges.

N denotes the total available training sample. Tuning uses an internal 80/20
split. The inner fitting fold alone determines standardization and adaptive
weights. An unpenalized pilot supplies weights
`1 / max(mean(abs(pseudo_precision)), 1e-6)` on the lower triangle, matching
the existing adaptive-lasso experiment's inverse-score convention with an
explicit numerical floor. The validation fold selects penalties and the
numbers of pretraining/joint-training iterations.

The pilot and the selected adaptive-lasso model are then refit on **all N**
observations, using full-training statistics and newly estimated full-training
adaptive weights. The final fit uses the selected iteration budgets and no
test-based early stopping. Default maximum training iterations are 2,000,
with patience 5 and improvement tolerance `1e-7` during tuning. These numerical
settings can be changed in the model configuration.

The primary GTM ROC-AUC score is the training mean absolute **globally
nuclear-normalized Hessian**, using `copula_only=True` as in the existing
paper-study evaluation. The denominator is the nuclear norm of the complete
local Hessian. Squared/absolute Hessian and mean absolute pseudo-precision
matrices and their ROC-AUC values are also saved. The Hessian alternatives use
the same `score_copula_only` setting; they are not raw-marginal Hessians.

GTM thresholds are deliberately deferred. Its precision, recall, F1 and MCC
are stored as JSON `null` / missing CSV values, with `threshold_method=deferred`.
The score matrices retain everything needed to evaluate a later threshold.
No threshold is chosen against the known truth.

## SING protocol

`sing_simulation_config.json` keeps the pinned upstream algorithm, with
`delta=1`, fixed ordering after the dataset's shuffle, Newton-CG, and up to
150 iterations per map component. The base runner accumulates scores/variances
in batches of 16 to limit memory. The dedicated degree-3 runner described below
uses the full training sample at once. No rank/PIT transformation
or GPU port is introduced. The pinned SING/TransportMaps implementation is
NumPy-based and has no GPU backend; SING always runs on CPU.

ROC-AUC uses the initial dense map's unthresholded mean squared Hessian,
matching `sing_rvine_10D.ipynb`. Precision/recall/F1/MCC use the final
delta-method graph. The final raw score matrix and its ROC-AUC are saved too.
Train/test likelihoods use the **final fitted density** retained from SING.
Degree-3 maps can be substantially slower and use more memory. Numerical
failures are recorded as failed fits, not replaced by scores from another
degree or an unconverged model.
Derivative batch ranges are written to `run.log`, so a slow calculation can
be distinguished from a completed iteration. A 10D degree-3 smoke fit
converged in optimization but was stopped after more than ten minutes in its
first Hessian-score calculation; degree 3 has not been validated end to end
in this workspace. Allocate longer CPU jobs for that part of the grid.

## Run the study

### SING with Docker (Linux, CPU)

From the repository root, build the Python 3.11 image:

```sh
docker build -t mhe_sing -f sing_experiments/Dockerfile .
```

The image installs the Linux runtime dependencies and includes the pinned SING
checkout and runner code. Rebuild after changing the code or JSON settings.
Mount only the results directory so the image's upstream checkout stays visible;
the same host directory is used by the GTM runner.

```sh
mkdir -p sing_experiments/results
docker run --rm --init --user "$(id -u):$(id -g)" \
  -v "$PWD/sing_experiments/results:/opt/gtm/sing_experiments/results" \
  mhe_sing python sing_experiments/run_sing_simulation.py
```

SING needs no GPU flag. The image defaults BLAS/OpenMP to one thread per job;
override `OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS` and `MKL_NUM_THREADS` with
`docker run -e` if needed. To inspect the grid without fitting, append
`--list-only`. For a single degree-2 fit, append
`--n-train 125 --marginals gauss --seeds 1 --degrees 2`.

`run_sing_simulation_degree3.py` selects only degree 3, maxiter 300, seeds 1/2,
and N=2000 followed by N=500, under the `degree3_maxiter300_fullsample` run label.
Its `batch_size: null` setting evaluates derivatives on all training observations
in one call (2000 or 500), including the coefficient derivatives and Fisher
information. This can use substantially more memory than batches of 16;
runtime improvement is not guaranteed. Held-out likelihood evaluation retains
its separate batch size of 512.

Live terminal output is enabled by default for this runner, with iteration and
derivative-stage start/completion messages, timings, likelihood progress, and
30-second elapsed-time heartbeats during long calls. These are also retained in
`run.log`. A heartbeat reports elapsed time, rather than a percentage of an
individual derivative call. Use `--no-live-progress` to log only, or
`--live-progress` to enable the same live output on the other study runners.

An already running Python process keeps its loaded code and configuration.
To apply these settings, stop it with Ctrl-C, wait for it to exit, then run:

```sh
.venv-sing/bin/python -u sing_experiments/run_sing_simulation_degree3.py
```

The unfinished fit restarts from the beginning. The new run label preserves the
previous `degree3_maxiter300` profile and prevents configurations from mixing.
Use `--n-train 2000` for the six N=2000 fits, or `--list-only` to inspect
the selection. The runner also supports mounting the whole repository at
`/workspace`: it uses the pinned checkout at `/opt/gtm/.external/SING` when
there is no local checkout. `--sing-dir` explicitly selects a checkout.

### Using existing Python environments

Run these commands from the repository root. The two runners can use separate
Python environments and run concurrently on the same results directory.
The default output is `sing_experiments/results/sing_gtm_comparison/`, which
is already ignored by Git. The scripts also work when invoked by absolute path.

Inspect the grid without training:

```sh
.venv-sing/bin/python sing_experiments/run_sing_simulation.py --list-only
python sing_experiments/run_gtm_simulation.py --list-only
```

Optionally prepare all datasets first (the runners do this automatically too):

```sh
.venv-sing/bin/python sing_experiments/prepare_simulation_data.py
```

Run both SING degrees:

```sh
.venv-sing/bin/python sing_experiments/run_sing_simulation.py
```

Run GTM in a Python environment containing this repository's GTM dependencies
(Torch, Optuna, pyvinecopulib, NumPy, SciPy, pandas, scikit-learn, matplotlib,
seaborn, tqdm). The existing SING environment does not contain Torch/Optuna.
No MLflow server or database is required by these runners.

```sh
python sing_experiments/run_gtm_simulation.py --device cuda
```

`--device auto` uses CUDA when available and otherwise CPU; `--device cpu`
forces CPU. `--device cuda:1` selects another GPU. Explicit CUDA requests fail
clearly if unavailable. CUDA execution needs a CUDA-enabled Torch installation
and an NVIDIA GPU. GTM uses its existing float32 implementation; SING uses
float64. GPU results can differ slightly due to floating-point arithmetic.

Start with one dataset, or distribute disjoint seeds across jobs:

```sh
.venv-sing/bin/python sing_experiments/run_sing_simulation.py --n-train 125 --marginals gauss --seeds 1 --degrees 2
python sing_experiments/run_gtm_simulation.py --n-train 125 --marginals gauss --seeds 1 --device cpu
```

Both runners accept `--study-config`, `--model-config`, `--output-dir`,
`--run-label`, `--n-train`, `--marginals`, and `--seeds`. SING also accepts
`--degrees`; GTM accepts `--device` and `--threads` (Torch CPU threads; default 1).
Use a new `--run-label` for different model settings and a new `--output-dir`
for different dataset settings. Configurations are checked before resuming.
Dataset and run locks prevent concurrent writers from corrupting a fit.

Successful fits are skipped on rerun. Failed/interrupted fits are retried;
failures do not stop the remaining grid, but the runner exits nonzero if its
own selected jobs fail. Inspect each failed run's `failure.txt` and `run.log`.
Ctrl-C records `status=interrupted` and stops the runner; restarting retries
that fit. A hard process kill can leave `status=running`, which is retried once
its process lock is released.
Do not regard an incomplete grid as 10 successful replicates.

## Local runtime comparison

`run_local_runtime_comparison.py` measures only total wall-clock runtime for
degree-2 SING and GTM on Gaussian-marginal R-vines at N=125, 500, and 2000.
It uses the same 10D data, column shuffles, standardization, seeds, and model
JSON settings as the study runners. GTM includes all 30 tuning trials, both
pilots, final refitting, and training-data dependence scores. SING includes its
complete iterative map fitting, Hessian scores, and delta-method thresholding.
Timing excludes imports, dataset preparation, external test likelihoods, and
accuracy metrics. Both run sequentially on CPU with one thread by default.

From the repository root, in your GTM environment:

```sh
conda activate mctm_pytorch
python sing_experiments/run_local_runtime_comparison.py
```

If `.venv-sing` is currently activated, run `deactivate` first. The script
launches `.venv-sing/bin/python` for SING and the invoking Python for GTM,
checking both environments before fitting. Alternatively, from either environment:

```sh
python sing_experiments/run_local_runtime_comparison.py \
  --gtm-python "$HOME/anaconda3/envs/mctm_pytorch/bin/python"
```

Append `--seeds 1 2 3` for three seeds (18 fits instead of 6), `--list-only`
to preview, or `--threads 4` to use four threads for both methods. Interpreter
paths, SING checkout, dataset cache, and output directory are configurable; see
`--help`. Run without competing compute jobs for a local comparison.

Results are appended after each fit to
`sing_experiments/results/local_runtime_comparison/runtimes.csv`, with one
`total_runtime_seconds` value per successful fit plus model, N, seed, dataset
and configuration hashes, interpreter, and thread count. Failed/interrupted
fits have a status and error with an empty runtime. Each invocation performs
fresh fits and appends measurements, including seeds measured previously.
Models, likelihoods, graph matrices, and optimizer diagnostics are discarded;
the existing shared dataset cache is retained. Data generation may still create
the study's test set and truth audit, entirely outside the measured interval.

## Outputs and comparison

Each study run writes:

- `result.json`: metrics, status, data checksum, configuration, seeds, versions,
  timing, device, likelihood scale, and scoring/threshold conventions.
- `evaluation.npz`: original-label truth and ranking matrices, permutation,
  per-observation train/test log densities; GTM also saves adaptive weights and
  inner-split indices, while SING also saves its final thresholded graph.
- `run.log`, plus `failure.txt` if unsuccessful.
- GTM: tuning trials, selected settings, refit histories, and a CPU-portable
  model state dictionary. SING: per-iteration optimizer diagnostics.

All graph comparisons use the 45 unordered off-diagonal variable pairs, after
undoing **both axes** of the shuffle. The known reference is the factor-scope
graph, checked against mixed derivatives of the generating density at two
finite-difference steps. Binary/ranking metrics are never scored in mismatched
variable order.

Likelihoods are joint per-observation densities on the original data scale:
`log p_original(x) = log p_standardized(z) - sum(log(training_std))`.
Sums and means are stored for both train and test. The comparison uses mean
test log likelihood; standardization is never fitted on test observations.

Open `sing_gtm_comparison.ipynb` in the existing SING kernel. It shows coverage
and failures, ROC-AUC/test-likelihood panels by marginal family, mean/SD
summaries, paired GTM-minus-SING comparisons, and the stored binary metrics.
Error bars are standard errors across seeds, and paired counts are explicit.
The notebook refuses to pair results with different dataset checksums.
It can run before any training, and shows "not run" rather than invented data.

To collect JSON results into CSV without opening the notebook:

```sh
.venv-sing/bin/python sing_experiments/collect_simulation_results.py
```

Run the numerical and workflow checks:

```sh
.venv-sing/bin/python -m unittest sing_experiments.test_sing_marginals sing_experiments.test_simulation_study
```

The full production grid is intentionally not launched by creating these
scripts. GTM's grid includes 2,700 tuning fits, plus pilots and full-N refits;
SING degree 3 also has substantial derivative costs. Reduced-iteration smoke
runs are for checking the workflow, not for scientific comparison.
