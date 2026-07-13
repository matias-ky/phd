# -*- coding: utf-8 -*-
"""Aggregation helpers for per-avalanche computations.

This module provides `aggregate_by_lim_a` which computes sums and means
over intervals defined by `lim_a` (avalanche boundaries).
"""

import numpy as np


def aggregate_by_lim_a(values, lim_a, ops=("sum", "mean")):
    """Aggregate 1D array `values` over intervals defined by `lim_a`.

    Parameters
    ----------
    values : array-like
        1D numeric array aligned with simulation iterations.
    lim_a : array-like
        Array of indices marking avalanche boundaries (zeros in e_lib).
    ops : tuple
        Operations to compute. Supported: 'sum', 'mean'.

    Returns
    -------
    result : dict
        Keys for each requested op mapping to numpy arrays with one entry
        per avalanche (in the same order used by timing functions). Also
        returns 'n_empty' with the count of empty slices encountered.
    """
    values = np.asarray(values)
    lim_a = np.asarray(lim_a, dtype=int)

    sums = []
    means = []
    n_empty = 0

    # Use cumsum for efficient interval sums
    c = np.concatenate(([0.0], np.cumsum(values, dtype=np.float64)))

    for i in range(len(lim_a) - 1):
        start = int(lim_a[i]) + 1
        end = int(lim_a[i + 1])
        length = end - start
        if length <= 0:
            sums.append(np.nan)
            means.append(np.nan)
            n_empty += 1
            continue

        total = float(c[end] - c[start])
        sums.append(total)
        means.append(total / length if length > 0 else np.nan)

    result = {}
    if "sum" in ops:
        result["sum"] = np.array(sums, dtype=np.float64)
    if "mean" in ops:
        result["mean"] = np.array(means, dtype=np.float64)
    result["n_empty"] = n_empty
    return result
