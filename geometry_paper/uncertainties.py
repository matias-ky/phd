# -*- coding: utf-8 -*-
"""uncertainties.py -- realistic uncertainties of the power-law exponents.

Same truncated power-law MLE as analysis.mle_exponent (continuous approximation,
p(x) ~ x^alpha on [x_min, x_max]), but evaluated in O(log n) per window on a sorted
sample, so that many windows can be explored:

  * statistical error  : batch means over N_BLOCKS contiguous blocks in time (avalanches
                         that are close in time are correlated, the i.i.d. bootstrap is not
                         appropriate for the deterministic model).
  * systematic error   : half the range of alpha over a 5 x 5 grid of windows, x_min and
                         x_max varied INDEPENDENTLY by factors 1/2, 1/sqrt2, 1, sqrt2, 2
                         (= removing/adding up to ~3 log-bins of width 0.1 dex at each end).
  * local exponent     : MLE in sliding windows of half a decade (diagnostic of curvature).
"""
import numpy as np
from scipy.optimize import minimize_scalar

N_BLOCKS = 20
GRID_FACTORS = (0.5, 2 ** -0.5, 1.0, 2 ** 0.5, 2.0)
MIN_RATIO = 4.0          # windows of the grid must span at least x_max/x_min = 4
MIN_POINTS = 50


class SortedSample:
    def __init__(self, x):
        x = np.asarray(x, float)
        self.x = np.sort(x[x > 0])
        self.cl = np.concatenate([[0.0], np.cumsum(np.log(self.x))])

    def n_sumlog(self, lo, hi):
        i = np.searchsorted(self.x, lo, "left")
        j = np.searchsorted(self.x, hi, "right")
        return j - i, self.cl[j] - self.cl[i]


def mle_window(S, lo, hi=np.inf):
    """alpha (negative) of p ~ x^alpha on [lo, hi]; hi = inf -> Clauset Eq. (3.1)."""
    n, sl = S.n_sumlog(lo, hi)
    if n < MIN_POINTS:
        return np.nan
    if not np.isfinite(hi):
        return -(1 + n / (sl - n * np.log(lo)))

    def nll(a):
        if abs(a - 1) < 1e-9:
            logC = -np.log(np.log(hi / lo))
        else:
            logC = np.log(abs(1 - a)) - np.log(abs(hi ** (1 - a) - lo ** (1 - a)))
        return -(n * logC - a * sl)
    return -minimize_scalar(nll, bounds=(0.05, 6.0), method="bounded",
                            options={"xatol": 1e-7}).x


def window_grid(S, lo, hi, factors_hi=None):
    """alphas over the grid of windows (x_min*f1, x_max*f2); hi may be inf."""
    vals = []
    for f1 in GRID_FACTORS:
        for f2 in (factors_hi if factors_hi is not None else GRID_FACTORS):
            l, h = lo * f1, hi * f2
            if h / l < MIN_RATIO:
                continue
            vals.append(mle_window(S, l, h))
    return np.array(vals, float)


def batch_means_error(x_time_ordered, lo, hi=np.inf, n_blocks=N_BLOCKS):
    blocks = np.array_split(np.asarray(x_time_ordered, float), n_blocks)
    v = np.array([mle_window(SortedSample(b), lo, hi) for b in blocks])
    v = v[np.isfinite(v)]
    return float(np.std(v, ddof=1) / np.sqrt(len(v))) if len(v) > 2 else np.nan


def exponent_with_errors(x_time_ordered, lo, hi=np.inf, sys_hi=None):
    """dict(alpha, err (statistical), err_sys, alpha_min, alpha_max, n_tail, xmin, xmax).
    sys_hi: list of alternative upper bounds for the grid when hi = inf (energy fits)."""
    S = SortedSample(x_time_ordered)
    a = mle_window(S, lo, hi)
    if sys_hi is None:
        g = window_grid(S, lo, hi)
    else:
        g = np.array([mle_window(S, lo * f, h) for f in GRID_FACTORS for h in sys_hi
                      if h / (lo * f) >= MIN_RATIO])
    g = np.append(g[np.isfinite(g)], a)
    n, _ = S.n_sumlog(lo, hi)
    return {"alpha": float(a), "err": batch_means_error(x_time_ordered, lo, hi),
            "err_sys": float((g.max() - g.min()) / 2), "alpha_min": float(g.min()),
            "alpha_max": float(g.max()), "n_tail": int(n), "xmin": float(lo),
            "xmax": float(hi) if np.isfinite(hi) else None,
            "decades": float(np.log10(hi / lo)) if np.isfinite(hi) else None}


def local_exponent(x, centers, half_width_dex=0.25):
    S = SortedSample(x)
    f = 10 ** half_width_dex
    return np.array([mle_window(S, c / f, c * f) for c in centers])
