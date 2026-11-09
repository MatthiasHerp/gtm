# SING R-vine example

For the paired 10D SING/GTM simulation study, see
[SIMULATION_STUDY.md](SIMULATION_STUDY.md) and
`sing_gtm_comparison.ipynb`. The study runners use the requested fixed
standardization, no PIT/rank transformation, and shared train/test datasets;
their settings are independent of the interactive single-run notebook below.

`sing_magic_classification.ipynb` fits one SING density per MAGIC class and
computes Bayes classification with equal class priors, ROC-AUC, and TPR at the
same fixed FPR levels as `demos/demo_magic_classification.ipynb`. It uses only
pooled training-mean/std standardization. `magic_gtm_split_indices.npz` stores
the exact GTM training/validation/test row indices, so the SING kernel does not
need Torch to reproduce the split. The notebook is delivered unexecuted.

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
