"""Run the pinned upstream SING implementation on the GTM study's R-vines."""
from __future__ import annotations

import ast
import itertools
from pathlib import Path
import random
import subprocess
import sys
import types

import numpy as np
import pandas as pd
import pyvinecopulib as pv
from scipy.stats import expon, norm, rankdata, t as student_t
from sklearn.metrics import (average_precision_score, confusion_matrix,
                             f1_score, matthews_corrcoef, precision_score,
                             recall_score, roc_auc_score)

ROOT = Path(__file__).resolve().parents[1]
SING_COMMIT = "73e1c2ad9a8886854c2ad22c75508ad3cc36df44"


def nonparanormal_transform(data):
    """Gaussianize each training column using only its empirical ranks.

    Return Phi^-1((rank - 0.5) / n), with average ranks for ties. Midpoint
    probabilities stay strictly inside (0, 1), so extreme observations have
    finite scores. Row and column order are preserved; the input is unchanged.
    This transforms the supplied training sample, without fitting an
    out-of-sample CDF or using the generating marginal distributions.
    """
    x = np.asarray(data, dtype=float)
    if x.ndim != 2 or x.shape[0] < 2 or x.shape[1] < 1:
        raise ValueError("data must be a 2D training array with at least two rows and one column")
    if not np.isfinite(x).all():
        raise ValueError("data must contain only finite observations")
    ranks = rankdata(x, axis=0, method="average")
    return norm.ppf((ranks - 0.5) / len(x))


def _marginal_specification(marginals, dimensionality, seed):
    """JSON-serializable marginal parameters in original variable order."""
    if marginals == "gauss":
        return dict(kind=marginals, loc=0.0, scale=1.0)
    if marginals == "exp":
        return dict(kind=marginals, loc=0.0, scale=1.0)
    if marginals == "t":
        return dict(kind=marginals, df=2.5, loc=0.0, scale=1.0)
    if marginals == "gaussmix":
        # Each variable has its own three components. A local RNG keeps the
        # marginal parameters independent of the number/seed of observations.
        rng = np.random.default_rng(seed)
        return dict(kind=marginals, seed=int(seed), weights=[1 / 3] * 3,
                    means=rng.uniform(-2, 2, size=(dimensionality, 3)).tolist(),
                    variances=rng.uniform(0.5, 2, size=(dimensionality, 3)).tolist())
    raise ValueError("marginals must be one of 'gauss', 'exp', 't', 'gaussmix'")


def marginal_cdf(data, marginal_spec=None):
    """Map observations to copula coordinates using the generating margins.

    Pass ``pair_table.attrs['marginal_spec']`` returned by ``make_rvine``.
    Omitting the specification preserves the original standard-normal behavior.
    """
    x = np.asarray(data, dtype=float)
    spec = marginal_spec if marginal_spec is not None else dict(kind="gauss", loc=0.0, scale=1.0)
    kind = spec["kind"]
    if kind == "gauss":
        return norm.cdf(x, loc=spec["loc"], scale=spec["scale"])
    if kind == "exp":
        return expon.cdf(x, loc=spec["loc"], scale=spec["scale"])
    if kind == "t":
        return student_t.cdf(x, df=spec["df"], loc=spec["loc"], scale=spec["scale"])
    if kind == "gaussmix":
        means, scales = np.asarray(spec["means"]), np.sqrt(spec["variances"])
        return np.sum(norm.cdf(x[..., None], loc=means, scale=scales) * spec["weights"], axis=-1)
    raise ValueError(f"Unknown marginal kind: {kind!r}")


def _marginal_ppf(uniforms, spec):
    """Apply inverse marginal CDFs, preserving the simulated copula exactly."""
    # Float64 throughout SING; clipping only guards against infinite quantiles.
    u = np.clip(np.asarray(uniforms, dtype=float), 1e-12, 1 - 1e-12)
    kind = spec["kind"]
    if kind == "gauss":
        return norm.ppf(u, loc=spec["loc"], scale=spec["scale"])
    if kind == "exp":
        return expon.ppf(u, loc=spec["loc"], scale=spec["scale"])
    if kind == "t":
        return student_t.ppf(u, df=spec["df"], loc=spec["loc"], scale=spec["scale"])
    if kind == "gaussmix":
        # Component quantiles bracket the mixture quantile at the same u.
        # Invert the mixture CDF rather than randomly selecting a component,
        # which would alter the dependence encoded by the vine uniforms.
        components = norm.ppf(u[..., None], loc=np.asarray(spec["means"]),
                              scale=np.sqrt(spec["variances"]))
        lower, upper = components.min(axis=-1), components.max(axis=-1)
        for _ in range(60):
            midpoint = (lower + upper) / 2
            below = marginal_cdf(midpoint, spec) < u
            lower = np.where(below, midpoint, lower)
            upper = np.where(below, upper, midpoint)
        return (lower + upper) / 2
    raise ValueError(f"Unknown marginal kind: {kind!r}")


def make_rvine(n=2000, copula_seed=1, sample_seed=11, active_trees=3,
               tau_mean=0.3, tau_range=0.2, dimensionality=10, marginals="gauss"):
    """Reuse the actual study sampler, without importing GTM/Torch/MLflow.

    The function definition is extracted with ast, so changes to the study's
    sampler are picked up rather than maintained as a second implementation.

    ``marginals`` selects standard normal ("gauss", default), rate-1 exponential
    ("exp"), Student's t with mean 0, df=2.5 and scale 1 ("t"), or an equally
    weighted three-Gaussian mixture per variable ("gaussmix"). Mixture means
    are uniform on [-2, 2] and component variances on [0.5, 2], seeded by
    ``copula_seed``. Changing margins preserves the vine and copula uniforms.
    The return signature stays ``(vine, data, pair_table)``; exact marginal
    parameters are available in ``pair_table.attrs['marginal_spec']``.
    """
    for name, value in (("dimensionality", dimensionality),
                        ("active_trees", active_trees), ("n", n)):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
            raise ValueError(f"{name} must be an integer")
    if dimensionality < 2:
        raise ValueError("dimensionality must be at least 2")
    if not 0 <= active_trees < dimensionality:
        raise ValueError("active_trees must be between 0 and dimensionality - 1")
    if n < 2:
        raise ValueError("n must be at least 2")
    marginal_spec = _marginal_specification(marginals, dimensionality, copula_seed)
    source = ROOT / "demos/pyvinecopulib_simulation_helpers.py"
    tree = ast.parse(source.read_text())
    definition = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                      and node.name == "sample_random_pair_copulas")
    namespace = {"np": np, "pv": pv}
    exec(compile(ast.Module(body=[definition], type_ignores=[]), str(source), "exec"), namespace)
    np.random.seed(copula_seed)
    random.seed(copula_seed)
    structure = pv.RVineStructure.simulate(dimensionality, seeds=[copula_seed])
    copulas = namespace["sample_random_pair_copulas"](
        dimensionality, Independence_tree=active_trees, tau_mean=tau_mean,
        tau_range=tau_range, negative_tau=True)
    vine = pv.Vinecop.from_structure(structure=structure, pair_copulas=copulas)
    uniforms = vine.simulate(n, seeds=[sample_seed])
    data = _marginal_ppf(uniforms, marginal_spec)
    matrix = structure.matrix
    rows = []
    for t in range(dimensionality - 1):
        for e in range(dimensionality - 1 - t):
            a, b = int(matrix[dimensionality - 1 - e, e]), int(matrix[t, e])
            conditioning = tuple(int(v) for v in matrix[:t, e])
            copula = copulas[t][e]
            rows.append(dict(tree=t + 1, edge=e + 1, a=a, b=b,
                             conditioning=conditioning, family=copula.family.name,
                             rotation=copula.rotation, tau=copula.tau,
                             study_dependence=int(t < active_trees)))
    table = pd.DataFrame(rows)
    table.attrs["marginal_spec"] = marginal_spec
    assert len({tuple(sorted((r.a, r.b))) for r in table.itertuples()}) == dimensionality * (dimensionality - 1) // 2
    return vine, data, table


def reference_graphs(table, dim=10):
    """Study labels plus a conservative graph from joint-density factor scopes.

    Missing scope edges imply exact full conditional independence. Present scope
    edges are generic candidates: special parameter cancellations can remove them.
    The notebook audits these candidates against oracle mixed log derivatives.
    """
    study = np.zeros((dim, dim), dtype=bool)
    scopes = study.copy()
    for r in table.itertuples():
        if r.study_dependence:
            study[r.a - 1, r.b - 1] = study[r.b - 1, r.a - 1] = True
        if r.family != "indep":
            for a, b in itertools.combinations((r.a, r.b, *r.conditioning), 2):
                scopes[a - 1, b - 1] = scopes[b - 1, a - 1] = True
    return study, scopes


def oracle_mixed_hessians(vine, data, step=1e-3, marginal_spec=None):
    """Central finite differences of the known copula log density.

    Pass ``pair_table.attrs['marginal_spec']`` for non-Gaussian data. All
    univariate marginal log densities have zero off-diagonal derivatives, so
    only log c(F_1(x_1), ..., F_d(x_d)) is needed. These are diagnostics at
    finitely many points, not an analytic proof of independence.
    """
    data = np.asarray(data, dtype=float)
    n, dim = data.shape
    result = np.zeros((n, dim, dim))
    def logpdf(x):
        if marginal_spec is not None and marginal_spec["kind"] == "exp" and np.any(x <= marginal_spec["loc"]):
            raise FloatingPointError("Oracle step crosses exponential support; choose interior audit points")
        density = vine.pdf(marginal_cdf(x, marginal_spec))
        if np.any(density <= 0) or not np.isfinite(density).all():
            raise FloatingPointError("Oracle density underflow; choose less extreme audit points")
        return np.log(density)
    for a, b in itertools.combinations(range(dim), 2):
        da, db = np.zeros(dim), np.zeros(dim)
        da[a] = db[b] = step
        values = (logpdf(data + da + db) - logpdf(data + da - db)
                  - logpdf(data - da + db) + logpdf(data - da - db)) / (4 * step**2)
        result[:, a, b] = result[:, b, a] = values
    return result


class FixedOrdering:
    """Keep the declared variable order, avoiding an optional CHOLMOD dependency."""
    def perm(self, gp):
        return np.arange(len(gp))


def load_sing(batch_size=256, optimizer_derivatives=2, optimizer_maxiter=100,
              retain_density=False, progress=False):
    """Load upstream code with explicit compatibility/memory adaptations.

    No global NumPy monkey patches. The graph algorithm and threshold are upstream.
    The variance uses the same delta-method formula, accumulated in small batches.
    With ``retain_density=True``, the final history entry also retains its fitted
    ``density`` for held-out log-density evaluation; earlier maps are released.
    ``progress=True`` logs the observation range before each derivative batch.
    """
    if optimizer_derivatives not in (1, 2) or optimizer_maxiter < 1:
        raise ValueError("Use derivative order 1 (BFGS) or 2 (Newton-CG), with a positive iteration limit")
    directory = ROOT / ".external/SING"
    commit = subprocess.check_output(["git", "-C", str(directory), "rev-parse", "HEAD"], text=True).strip()
    if commit != SING_COMMIT:
        raise RuntimeError(f"Expected SING {SING_COMMIT}, found {commit}")
    path = directory / "SING"
    gp = types.ModuleType("GeneralizedPrecision")
    exec(compile((path / "GeneralizedPrecision.py").read_text(), str(path / "GeneralizedPrecision.py"), "exec"), gp.__dict__)
    history = []
    solver_logs = []

    def record_solve(log):
        logs = log if isinstance(log, list) else [log]
        # Newton-CG can report line-search precision loss at a stationary point.
        # Accept that specific flag only if the gradient meets the requested tol.
        for entry in logs:
            jac = np.asarray(entry.get("jac", [np.inf]))
            stationary = (np.isfinite(jac).all() and np.linalg.norm(jac, ord=np.inf) <= 1e-5)
            entry["accepted_by_gradient"] = bool(
                not entry.get("success", False) and stationary
                and "precision loss" in str(entry.get("message", "")).lower())
        failed = [entry for entry in logs
                  if not (entry.get("success", False) or entry["accepted_by_gradient"])]
        if failed:
            details = [{k: entry.get(k) for k in ("message", "nit", "fval")} for entry in failed]
            raise RuntimeError(f"Transport-map optimization failed: {details}")
        solver_logs.append(logs)
        if any(entry["accepted_by_gradient"] for entry in logs):
            print("Precision-loss flag accepted: gradient infinity norm meets 1e-5 tolerance.", flush=True)
        print(f"Map fit converged ({len(logs)} components); computing Hessian scores.", flush=True)

    def batches(data, label=None):
        for start in range(0, len(data), batch_size):
            if progress and label:
                print(f"{label}: observations {start + 1}-{min(start + batch_size, len(data))} of {len(data)}",
                      flush=True)
            yield data[start:start + batch_size]

    def precision(pb, data):
        value = sum(np.square(pb.hess_x_log_pdf(x)).sum(axis=0)
                    for x in batches(data, "Input Hessians")) / len(data)
        if not np.isfinite(value).all():
            raise FloatingPointError("Nonfinite generalized precision")
        if retain_density and history:
            history[-1].pop("density", None)
        history.append({"omega_raw": value.copy(), "solver_logs": solver_logs[-1]})
        if retain_density:
            history[-1]["density"] = pb
        print("Computing batched delta-method variance.", flush=True)
        return value

    def variance(pb, data):
        n = len(data)
        fisher, jac = None, None
        for x in batches(data, "Delta-method derivatives"):
            hess = pb.hess_x_log_pdf(x)
            grad = pb.grad_a_hess_x_log_pdf(x)
            partial_jac = np.einsum("nij,naij->aij", 2 * hess, grad)
            partial_fisher = -pb.hess_a_log_pdf(x).sum(axis=0)
            jac = partial_jac if jac is None else jac + partial_jac
            fisher = partial_fisher if fisher is None else fisher + partial_fisher
        jac, fisher = jac / n, fisher / n
        flat = jac.reshape(jac.shape[0], -1)
        covariance_times_jac = np.linalg.solve(fisher, flat) / n
        value = np.einsum("ai,ai->i", flat, covariance_times_jac).reshape(jac.shape[1:])
        if not np.isfinite(value).all() or value.min() < -1e-10:
            raise FloatingPointError("Invalid delta-method variance; inspect map fit/Fisher information")
        value = np.maximum(value, 0)
        history[-1]["variance"] = value.copy()
        return value

    gp.gen_precision, gp.var_omega = precision, variance
    sys.modules["GeneralizedPrecision"] = gp
    module = types.ModuleType("upstream_sing")
    source = (path / "SparsityIdentificationNonGaussian.py").read_text()
    source = source.replace(
        "    StandardNormalDistribution, DistributionFromSamples, \\\n    PushForwardTransportMapDistribution, PullBackTransportMapDistribution",
        "    StandardNormalDistribution, DistributionFromSamples\n"
        "from TransportMaps.TransportMapDistributions import "
        "PullBackParametricTransportMapDistribution as PullBackTransportMapDistribution")
    # KL(T_push pi || eta) = KL(pi || T_pull eta), by change of variables.
    # Current TransportMaps fits the latter through its parametric pullback API.
    source = source.replace(
        "PushForwardTransportMapDistribution(tm_approx, pi)",
        "PullBackTransportMapDistribution(tm_approx, eta)")
    source = source.replace("minimize_kl_divergence(eta, qtype=qtype,",
                            "minimize_kl_divergence(pi, qtype=qtype,")
    source = source.replace("    ders = 2", f"    ders = {optimizer_derivatives}")
    source = source.replace("tol=tol, ders=ders)",
                            f"tol=tol, ders=ders, maxit={optimizer_maxiter})")
    source = source.replace("       pb_density =", "       _record_solve(solve)\n       pb_density =")
    source = source.replace("np.int((dim**2 - dim)/2)", "int((dim**2 - dim)/2)")
    module.__dict__["_record_solve"] = record_solve
    exec(compile(source, str(path / "SparsityIdentificationNonGaussian.py"), "exec"), module.__dict__)
    return module.SING, history


def graph_metrics(truth, adjacency, scores):
    indices = np.triu_indices_from(truth, k=1)
    y, pred, score = truth[indices].astype(int), adjacency[indices].astype(int), scores[indices]
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return dict(tp=int(tp), fp=int(fp), fn=int(fn), tn=int(tn),
                precision=precision_score(y, pred, zero_division=0),
                recall=recall_score(y, pred, zero_division=0),
                f1=f1_score(y, pred, zero_division=0), mcc=matthews_corrcoef(y, pred),
                shd=int(fp + fn), exact_recovery=bool(np.array_equal(y, pred)),
                roc_auc=roc_auc_score(y, score) if len(np.unique(y)) == 2 else np.nan,
                average_precision=average_precision_score(y, score) if len(np.unique(y)) == 2 else np.nan)
