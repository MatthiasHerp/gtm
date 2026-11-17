# SING R-vine example

For the paired 10D SING/GTM simulation study, see
[SIMULATION_STUDY.md](SIMULATION_STUDY.md) and
`sing_gtm_comparison.ipynb`. The study runners use the requested fixed
standardization, no PIT/rank transformation, and shared train/test datasets;
their settings are independent of the interactive single-run notebook below.

`run_magic_classification.py` fits one SING density per MAGIC class, first for
order 2 and then for order 3, and saves Bayes classification with equal class
priors, ROC-AUC, and TPR at the same fixed FPR levels as
`demos/demo_magic_classification.ipynb`. It uses only pooled fitting-fold
mean/std standardization. `sing_magic_classification.ipynb` loads the saved
results and shows the full order-2 analysis before the order-3 analysis,
without refitting or importing Torch/TransportMaps.

### Run MAGIC on the server

Generate the exact GTM split indices once in a Torch environment if
`sing_experiments/magic_gtm_split_indices.npz` is missing:

```sh
python sing_experiments/prepare_magic_splits.py
```

With the existing server Docker images, the same preparation is:

```sh
docker run --rm -v "$PWD:/workspace" -w /workspace \
  --entrypoint python3 mhe_torch_gtm \
  sing_experiments/prepare_magic_splits.py
```

Then run from the repository root (no Jupyter installation is needed):

```sh
docker run --rm --init --name sing_magic \
  -v "$PWD:/workspace" -w /workspace \
  mhe_sing python -u sing_experiments/run_magic_classification.py
```

The mounted code uses the image's pinned SING checkout at
`/opt/gtm/.external/SING` if the repository has no local checkout. For an
existing Python environment, use
`python -u sing_experiments/run_magic_classification.py` instead.

Keep the command in a `tmux` session for long server runs. Optimizer/Hessian,
delta-method derivative, and log-density batch progress stays visible in the
terminal. The default derivative batch size is 16; append `--batch-size 4`
to reduce derivative working memory, at the cost of more batches. This does
not bound optimizer or dense Fisher-matrix memory. SING runs on CPU; no GPU
flag is needed, and the command sets no Docker memory cap. Order 3 can be
substantially more demanding than order 2.

To run only degree 3 after degree 2 has completed, use the dedicated entry point:

```sh
docker run --rm --init --name sing_magic_degree3 \
  -v "$PWD:/workspace" -w /workspace \
  mhe_sing python -u sing_experiments/run_magic_classification_degree3.py
```

It uses the same settings, output directory, and checkpoint handling as the
original runner, and rejects attempts to select degree 2. Run inside a Screen
session (for example, `screen -S sing_magic_degree3`) to keep it attached to a
persistent terminal. After successful completion, rerun the analysis notebook
to show the degree-3 graphs, rank comparisons, and test likelihoods.

Each class is fitted and evaluated before its density is released. Each
order runs in a fresh process. Results go to
`sing_experiments/results/magic_classification/p{order}_delta1_seed25_{data_hash}/`:

- `class_g.npz` / `class_h.npz` and their JSON manifests checkpoint each
  completed class, including its graph, scores, coefficients, all class/split
  log densities, timings, and solver diagnostics.
- `test_predictions.csv`, `roc_curve.csv`, `tpr_at_fpr.csv`, and
  `classification_metrics.csv` support the classification analysis.
- `density_loglik.csv`, `feature_scaling.csv`, `fit_summary.csv`, and
  `arrays.npz` retain density diagnostics, scaling, exact row indices, and
  numerical artifacts. `summary.json` records the configuration and metrics;
  `status.json` records completion or failure, with traceback in `failure.txt`.

Rerun the same command to reuse completed classes/orders. Settings, package
versions, dataset, and split checksums must match; use a new `--output-dir`
when changing settings (including batch size). `--orders 2` or `--orders 3`
restricts execution. Open and run `sing_magic_classification.ipynb` afterward
to display saved results. The notebook needs NumPy, pandas, Matplotlib,
scikit-learn, and a Jupyter Python kernel, but performs no model fitting.

For each order, the notebook also plots the identified hadron and gamma SING
graphs beside GTM graphs from `demos/ci_tables`. SING uses its saved final
thresholded graph; GTM uses positive IAE values at least 0.1, matching the
existing MAGIC GTM graph plots. `GTM_IAE_THRESHOLD` is configurable. Fixed node
positions make edges easy to compare. Tables report shared edges, edges unique
to each method, SHD, and Jaccard agreement, plus the disagreeing pairs. Full
45-pair tables remain in `order2['graphs']['edge_tables']` and
`order3['graphs']['edge_tables']`. These compare estimates, not a known true
graph. Missing GTM tables are reported without hiding the SING graphs; existing
SING results can be used without refitting.
The comparison also reports Spearman correlation of all 45 pairwise dependence
rankings, using GTM IAE and SING's final unthresholded Hessian scores. Rank 1 is
strongest; ties use average ranks. `spearman_rho=1` indicates identical ordering,
while `same_edge_order` checks exact rank equality. Constant score vectors give
undefined correlation (`NaN`). Full pair tables include scores and ranks from
both methods, independently of the graph threshold.
It also compares SING's significant edge set with the top x GTM edges by IAE,
where x is the number of significant SING edges for that class and order.
`top_x_same_edges` reports set equality; overlap counts/fractions and mismatching
pairs show partial agreement. This selection ignores the GTM graph cutoff.
Ties across the top-x cutoff are flagged and broken by variable labels.
Kendall's tau-b is reported for all pairs, GTM's top x pairs, and SING's
significant pairs. Both models are ranked on the same pairs within each subset.

To compare own-class mean test log likelihoods with the existing GTMs, export
their test densities once (hadron 8 layers, gamma 6 layers; no refitting):

```sh
docker run --rm --init -v "$PWD:/workspace" -w /workspace \
  --entrypoint python3 mhe_torch_gtm \
  sing_experiments/export_magic_gtm_test_loglik.py
```

The notebook reads `results/magic_classification/gtm_test_loglik.npz` and checks
the dataset and exact test rows before comparing to each SING order. Both
likelihoods use natural logarithms per observation on the original CSV scale,
including each preprocessing Jacobian. GTM reproduces its original pooled
train+validation min/max scaling to [-12,12]; SING uses fitting-fold mean/std.
The cache includes GTM checkpoint checksums and scaling metadata. Missing GTM
cache values are displayed as NaN with the command needed to generate them.

`sing_coefficient_count_tests.ipynb` checks the dense SING coefficient-count
formula over 22 combinations of dimensionality and degree. It verifies the
separate `c` and `h` counts against the map components and infers the total from
an actual parameter-Hessian tensor, using the same method as the R-vine notebook.
Run it with **Python (GTM SING)**; no model fitting is required.

`gtm_sing_scaling.ipynb` compares GTM and SING parameter counts and derivative-work
proxies as dimensionality increases. GTM uses `R N I P_GTM + R N D²`, with
`R=100` bootstrap repetitions and `I=2000` training iterations per refit. SING
uses `2 (N D² + N D² P_SING + N P_SING²)`, including its variance-estimator
derivatives and assuming two iterations with the same dense work count.
Both plots show GTM with `B_GTM=10` and SING with `B=2` and `B=3`.
The `counts` DataFrame retains per-iteration contributions and the combined SING
totals; `SING_ITERATIONS=2` is configurable. Other defaults are
`L=3`, `N=2000`, and `D=1..100`.
It uses the same **Python (GTM SING)** kernel.

The scaling notebook includes report text, assumptions, references, and figure
captions. After running its cells, export the report with code hidden:

```sh
./sing_experiments/export_scaling_report.sh
```

The PDF is written to `output/pdf/gtm_sing_scaling_report.pdf`. Export requires
Pandoc and XeLaTeX; the script also detects Pandoc bundled with RStudio.

### Manuscript math commands

`math.tex` is copied from the GTM manuscript repository
`../64da42743dede701976c8d45/math.tex` (the manuscript titled *Graphical
Transformation Models*, including the current Hessian-score notation). The PDF template loads this file in its
preamble. The first, visually empty Markdown cell defines its macros for
MathJax; render that cell before using the commands in subsequent Markdown.
For example:

```tex
$\bsx \in \bbR^D$, $\Var(X)$, and $X_i \independence X_j \mid X_{-ij}$
```

After changing the local `math.tex`, refresh the notebook definitions with:

```sh
.venv-sing/bin/python sing_experiments/sync_math_macros.py
```

The generated cell is omitted from the PDF because its definitions are already
loaded in the preamble. Browser definitions adapt text-size, math-mode, and
page-alignment commands for MathJax; the LaTeX file itself is preserved verbatim.

Open `sing_rvine_10D.ipynb` with the `.venv-sing/bin/python` interpreter, or run
from the repository root:

```sh
./sing_experiments/start_jupyter.sh
```

Select **Python (GTM SING)**. The kernel is installed inside the environment;
the launcher keeps Jupyter state inside this repository. The existing GTM
environment and requirements are untouched.

The default experiment uses 2,000 observations, 10 normal margins, a random
R-vine with the first three trees active, mixed pair-copula families, copula
seed 1, sample seed 11, map order 2 and SING delta 1. The notebook configuration
cell exposes these choices. Samples are double precision and standardized before
SING. Its pair-copula sampler is extracted from the existing GTM study helper
using Python's AST, avoiding that module's Torch/GTM imports and a duplicated
sampler implementation. The structure and sample seeds are separate. Unlike the
study generator's train/validation calls, this notebook only creates one training
set and does not reuse a seeded sample as independent validation data.

Set `DIMENSIONALITY` and `ACTIVE_TREES` in the notebook's configuration cell, then
restart the kernel and run all cells. Dimensionality must be an integer of at
least 2; active trees must be between 0 and `DIMENSIONALITY - 1`. For example,
`DIMENSIONALITY = 5` and `ACTIVE_TREES = 2` creates a 5D vine with two active
trees. Zero active trees produces independent variables; all trees active
produces a full vine. ROC AUC and average precision are NaN when the reference
has only one class. Plots, pair counts, and saved result folders follow both
settings automatically.

Set `MARGINALS` to `"gauss"`, `"exp"`, `"t"`, or `"gaussmix"`. The same option
is available as `make_rvine(..., marginals="gaussmix")`, preserving the existing
`(vine, data, pair_table)` return signature:

- `gauss` (default): standard normal.
- `exp`: exponential with rate 1.
- `t`: Student's t with mean 0, 2.5 degrees of freedom, and scale 1.
- `gaussmix`: three equally weighted Gaussian components per variable, with
  independent means uniform on [-2, 2] and component variances uniform on
  [0.5, 2]. The standard deviations are the square roots of these variances.

Mixture parameters are drawn using a local generator seeded by `copula_seed`,
so changing the sample seed or sample size keeps the generating density fixed.
Changing margins applies inverse CDFs to the same vine uniforms and preserves
the conditional-independence structure. Exact parameters are available through
`pair_table.attrs['marginal_spec']` and saved to `marginals.json`. When auditing
non-normal data, pass that specification to
`oracle_mixed_hessians(..., marginal_spec=pair_table.attrs['marginal_spec'])`.
The notebook handles this automatically and keeps saved runs for different
marginal choices in separate folders.

Set `NONPARANORMAL_TRANSFORM = True` (default) to Gaussianize every training
column before SING, for any choice of `MARGINALS`. The helper
`nonparanormal_transform(data)` uses average ranks for ties and returns
`norm.ppf((rank - 0.5) / n_train)`, giving finite scores even at the sample
extremes. Only observed training ranks are used. The scores are then
standardized and passed to SING in the selected variable order. Set the toggle
to `False` for the original standardization of the generated observations.
The oracle audit still uses the original observations and generating margins.

Saved arrays include original `data`, `training_data` before standardization,
and the exact SING input `processed_model_order`. `center` and `scale` describe
`training_data` in original variable order. `preprocessing.json` records the
method; result-folder suffixes `_nonparanormal` and `_raw` separate the two
settings. This is a training-sample transform; it does not fit an out-of-sample
transformation.

## Source and compatibility

- SING: <https://github.com/baptistar/SING>, commit
  `73e1c2ad9a8886854c2ad22c75508ad3cc36df44`, cloned to `.external/SING`.
- TransportMaps: <https://pypi.org/project/transportmaps/3.2.0/>.
- Python 3.11; exact installed package versions are in `requirements.lock.txt`.

`sing_helpers.py` loads the pinned upstream SING source and applies these explicit
adaptations in memory:

1. Update distribution imports and use TransportMaps' current parametric
   pullback fitting API. The fitted KL objective is mathematically equivalent
   under change of variables to upstream's pushforward objective.
2. Replace upstream's removed `np.int` alias with `int`.
3. Keep upstream Newton-CG fitting (`ders=2`, maximum 100 iterations), expose
   BFGS (`ders=1`) as an optional alternative, and check convergence of every
   fitted component. Both settings are configurable in the notebook.
   A precision-loss flag is accepted only if the finite gradient infinity norm
   is at most `1e-5`. The summary JSON records these cases and optimizer messages.
4. Batch generalized-precision, score-gradient and Fisher-information sums.
   Use a linear solve for the same delta-method variance rather than an explicit
   matrix inverse. No Fisher ridge or pseudoinverse is silently introduced.
5. Use a fixed variable order, avoiding scikit-sparse/CHOLMOD. This is an explicit
   ordering choice, not the reverse-Cholesky ordering used in some upstream demos.

SING's iterative sparsification and statistical threshold are preserved. Raw
dense-map scores are retained for ranking metrics. The notebook scores the final
thresholded graph over all unordered off-diagonal pairs (45 by default) and reports exact recovery,
SHD, TP/FP/FN/TN, precision, recall, F1, MCC, ROC AUC and average precision.

The reference graph comes from the known density's factor scopes and is audited
using mixed log-density finite differences at two step sizes. The notebook also
reports the GTM study's conditioned-pair labels. Numerical finite differences
are diagnostics, not a proof of independence. A single seed is not a simulation
study, and no threshold is selected using the known graph.

## Recreate the environment

From the repository root, with Python 3.11 available:

```sh
python3.11 -m venv .venv-sing
.venv-sing/bin/python -m pip install -r sing_experiments/requirements.lock.txt
git clone https://github.com/baptistar/SING.git .external/SING
git -C .external/SING checkout 73e1c2ad9a8886854c2ad22c75508ad3cc36df44
.venv-sing/bin/python -m ipykernel install --prefix .venv-sing --name gtm-sing --display-name 'Python (GTM SING)'
./sing_experiments/start_jupyter.sh
```

SpectralToolbox builds a small native extension, requiring a C++ compiler. The
lock file records the verified macOS ARM64/Python 3.11 installation; other
platforms may need compatible package versions. Higher map orders substantially
increase derivative costs. Results are saved to ignored `sing_experiments/results/`
folders, including the vine JSON, data arrays, per-pair CSV and summary JSON.
