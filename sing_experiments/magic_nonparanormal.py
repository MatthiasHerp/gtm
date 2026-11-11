"""Training-fitted empirical Gaussianization with a reusable held-out transform."""
import numpy as np
from scipy.stats import norm


def fit_nonparanormal(training):
    """Store distinct training values and average-rank midpoint probabilities."""
    training = np.asarray(training, dtype=np.float64)
    if training.ndim != 2 or len(training) < 2 or not np.isfinite(training).all():
        raise ValueError('Expected a finite 2D training array with at least two rows')
    state = {}
    for column in range(training.shape[1]):
        values, counts = np.unique(training[:, column], return_counts=True)
        if len(values) < 2:
            raise ValueError(f'Constant pooled training feature {column}')
        # (average rank - 0.5) / N, including ties.
        probabilities = (np.cumsum(counts) - counts / 2) / len(training)
        state[f'feature_{column}_values'] = values
        state[f'feature_{column}_probabilities'] = probabilities
    return state


def apply_nonparanormal(values, state):
    """Interpolate training CDF knots and clip tails at training endpoint scores."""
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2 or not np.isfinite(values).all():
        raise ValueError('Expected a finite 2D array')
    if len(state) != 2 * values.shape[1]:
        raise ValueError('Feature count differs from the fitted transformation')
    transformed = np.empty_like(values)
    for column in range(values.shape[1]):
        probabilities = np.interp(values[:, column], state[f'feature_{column}_values'],
                                  state[f'feature_{column}_probabilities'])
        transformed[:, column] = norm.ppf(probabilities)
    if not np.isfinite(transformed).all():
        raise FloatingPointError('Nonfinite Gaussianized observations')
    return transformed
