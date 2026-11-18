#!/usr/bin/env python3
"""Estimate dense SING derivative storage without allocating tensors or fitting."""
import argparse
from math import comb

GIB = 2 ** 30


def estimate_memory(n, dimension=10, degree=3, batch_size=None, n_coeffs=None):
    """Float64 array sizes; planning headroom is a heuristic, not a peak bound.

    TransportMaps returns coefficient Hessians (B,P,P), coefficient gradients
    of input Hessians (B,P,D,D), and input Hessians (B,D,D). In the current
    variance implementation these arrays can be live at the same time.
    Internal component tensors, basis caches, optimizer state, BLAS workspaces,
    and allocator overhead are additional. The dense first iteration gives the
    largest coefficient count; sparse later iterations can have fewer.
    """
    for label, value in [('n', n), ('dimension', dimension), ('degree', degree)]:
        if not isinstance(value, int) or isinstance(value, bool) or value < 1:
            raise ValueError(f'{label} must be a positive integer')
    if batch_size is not None and (not isinstance(batch_size, int)
                                   or isinstance(batch_size, bool) or batch_size < 1):
        raise ValueError('batch_size must be positive or None')
    p = comb(dimension + degree + 1, degree + 1) - 1 if n_coeffs is None else n_coeffs
    if not isinstance(p, int) or isinstance(p, bool) or p < 1:
        raise ValueError('n_coeffs must be a positive integer')
    b = n if batch_size is None else min(batch_size, n)
    arrays = dict(coefficient_hessians=8 * b * p * p,
                  coefficient_input_hessian_gradients=8 * b * p * dimension ** 2,
                  input_hessians=8 * b * dimension ** 2)
    # Fisher, Jacobian, partial sums, normalization/solve results: allow several
    # reduced copies, separate from the much larger per-observation tensors.
    reduced = 8 * (4 * p * p + 4 * p * dimension ** 2)
    baseline = sum(arrays.values()) + reduced
    return dict(n=n, batch_observations=b, dimension=dimension, degree=degree,
                n_coeffs=p, arrays_bytes=arrays, reduced_arrays_bytes=reduced,
                baseline_gib=baseline / GIB,
                planning_low_gib=2 * baseline / GIB, planning_high_gib=3 * baseline / GIB)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--observations', nargs='+', type=int, default=[500, 2000, 3564, 6576])
    parser.add_argument('--dimension', type=int, default=10)
    parser.add_argument('--degree', type=int, default=3)
    parser.add_argument('--batch-size', type=int, help='Omit for all observations at once')
    args = parser.parse_args(argv)
    print('Float64 derivative storage; GiB = 2^30 bytes. No model fitting or large allocation.')
    print(f'{"N":>7} {"B":>7} {"P":>7} {"Hess(a) GiB":>13} {"Grad Hess(x)":>13} '
          f'{"Baseline GiB":>13} {"2-3x planning GiB":>20}')
    try:
        for n in args.observations:
            row = estimate_memory(n, args.dimension, args.degree, args.batch_size)
            arrays = row['arrays_bytes']
            print(f'{n:7d} {row["batch_observations"]:7d} {row["n_coeffs"]:7d} '
                  f'{arrays["coefficient_hessians"] / GIB:13.2f} '
                  f'{arrays["coefficient_input_hessian_gradients"] / GIB:13.2f} '
                  f'{row["baseline_gib"]:13.2f} '
                  f'{row["planning_low_gib"]:8.1f}-{row["planning_high_gib"]:<8.1f}')
    except ValueError as error:
        parser.error(str(error))
    print('Planning range is unmeasured headroom, not a guaranteed upper bound.')
    print('Concurrent independent fits require the sum of their memory, not the maximum.')
    print('Calibrate with measured peak RSS on a smaller run using the same dimension and degree.')


if __name__ == '__main__':
    main()
