# -*- coding: utf-8 -*-
"""
analysis.py -- statistics used in the paper.

1. Box counting (fractal dimension of avalanche masks), validated on sets of known
   dimension (filled square -> 2, line -> 1, Sierpinski gasket -> log3/log2).
2. Power-law fits by maximum likelihood (MLE) with the lower cut-off x_min chosen
   by minimising the Kolmogorov-Smirnov distance (Clauset, Shalizi & Newman 2009,
   SIAM Review 51, 661) and an optional upper cut-off x_max (finite-size cut-off),
   i.e. a truncated power law  p(x) ~ x^alpha  on [x_min, x_max].
   Uncertainty = bootstrap standard deviation (re-sampling avalanches and re-doing
   the full fit including the x_min selection).
3. Logarithmically binned PDFs that account for the integer nature of areas.
"""
import numpy as np
import scipy.ndimage as ndi
from scipy.optimize import minimize_scalar

# ----------------------------------------------------------------------------- #
# 1. Box counting
# ----------------------------------------------------------------------------- #
MAX_L_POW = 8
BOX_SIZES_ALL = 2 ** np.arange(MAX_L_POW + 1)


def bounding_box(mask):
    rows = np.where(mask.any(axis=1))[0]
    cols = np.where(mask.any(axis=0))[0]
    return rows[0], rows[-1] + 1, cols[0], cols[-1] + 1


def extent(mask):
    r0, r1, c0, c1 = bounding_box(mask)
    return max(r1 - r0, c1 - c0)


def crop_and_pad(mask, bbox):
    """Crop to bbox and zero-pad (bottom/right) to the next power of two."""
    r0, r1, c0, c1 = bbox
    sub = mask[r0:r1, c0:c1]
    L = max(sub.shape)
    P = 1 << int(np.ceil(np.log2(max(L, 1))))
    out = np.zeros((P, P), dtype=bool)
    out[: sub.shape[0], : sub.shape[1]] = sub
    return out


def box_counts(padded, sizes=BOX_SIZES_ALL):
    P = padded.shape[0]
    counts = np.full(len(sizes), np.nan)
    for i, l in enumerate(sizes):
        if l > P:
            break
        n = P // l
        counts[i] = padded.reshape(n, l, n, l).any(axis=(1, 3)).sum()
    return counts


def fit_dimension(sizes, counts):
    ok = np.isfinite(counts) & (counts > 0)
    if ok.sum() < 2:
        return np.nan
    return -np.polyfit(np.log(sizes[ok]), np.log(counts[ok]), 1)[0]


def l_max_for_extent(L):
    return int(2 ** np.floor(np.log2(max(L / 4.0, 1))))


def fractal_pair(total_mask, peak_mask):
    """D of the total mask and of the peak mask, both counted on the SAME grid
    (bounding box of the total mask, boxes l = 1, 2, ..., l_max <= L/4)."""
    bbox = bounding_box(total_mask)
    L = max(bbox[1] - bbox[0], bbox[3] - bbox[2])
    ct = box_counts(crop_and_pad(total_mask, bbox))
    cp = box_counts(crop_and_pad(peak_mask, bbox))
    sel = BOX_SIZES_ALL <= l_max_for_extent(L)
    return (fit_dimension(BOX_SIZES_ALL[sel], ct[sel]),
            fit_dimension(BOX_SIZES_ALL[sel], cp[sel]) if peak_mask.any() else np.nan,
            L, ct, cp)


EIGHT = np.ones((3, 3), bool)


def largest_cluster(mask):
    lab, n = ndi.label(mask, structure=EIGHT)
    if n <= 1:
        return mask.astype(bool), max(n, 1)
    sizes = np.bincount(lab.ravel())[1:]
    return lab == (np.argmax(sizes) + 1), n


def validate_box_counting(verbose=True):
    n = 64
    sq = np.ones((n, n), bool)
    line = np.zeros((n, n), bool); line[n // 2, :] = True
    i, j = np.indices((n, n)); sier = (i & j) == 0
    res = {}
    for name, m, true in [("filled square", sq, 2.0), ("line", line, 1.0),
                          ("Sierpinski gasket", sier, np.log(3) / np.log(2))]:
        D = fractal_pair(m, m)[0]
        res[name] = (D, true)
        if verbose:
            print(f"  [box-counting validation] {name:18s} D = {D:.3f} (exact {true:.3f})")
        assert abs(D - true) < 1e-6, name
    return res


# ----------------------------------------------------------------------------- #
# 2. Power-law MLE (truncated, continuous approximation)
# ----------------------------------------------------------------------------- #
def _negloglik(a, x, xmin, xmax):
    """-log L for p(x) = C x^{-a} on [xmin, xmax] (a > 0 here; alpha = -a)."""
    n = len(x)
    if xmax is None or not np.isfinite(xmax):
        if a <= 1:
            return np.inf
        logC = np.log(a - 1) + (a - 1) * np.log(xmin)
    else:
        if abs(a - 1) < 1e-9:
            logC = -np.log(np.log(xmax / xmin))
        else:
            logC = np.log(abs(1 - a)) - np.log(abs(xmax ** (1 - a) - xmin ** (1 - a)))
    return -(n * logC - a * np.sum(np.log(x)))


def mle_exponent(x, xmin, xmax=None):
    x = x[(x >= xmin) & ((x <= xmax) if xmax else True)]
    if len(x) < 10:
        return np.nan, len(x)
    if xmax is None:
        return 1 + len(x) / np.sum(np.log(x / xmin)), len(x)     # Clauset Eq. (3.1)
    r = minimize_scalar(_negloglik, bounds=(0.05, 6.0), args=(x, xmin, xmax),
                        method="bounded", options={"xatol": 1e-6})
    return r.x, len(x)


def _cdf_model(xs, a, xmin, xmax):
    if xmax is None:
        return 1 - (xs / xmin) ** (1 - a)
    return (xs ** (1 - a) - xmin ** (1 - a)) / (xmax ** (1 - a) - xmin ** (1 - a))


def ks_distance(x, a, xmin, xmax=None):
    x = np.sort(x[(x >= xmin) & ((x <= xmax) if xmax else True)])
    if len(x) == 0:
        return np.inf
    emp = np.arange(1, len(x) + 1) / len(x)
    return np.max(np.abs(emp - _cdf_model(x, a, xmin, xmax)))


def fit_power_law(x, xmax=None, xmin_candidates=None, min_tail=200, n_boot=100, seed=0,
                  xmin_fixed=None):
    """Return dict(alpha (negative), err, xmin, xmax, n_tail, ks).
    x_min minimises the KS distance (Clauset et al. 2009) among candidates, keeping at
    least `min_tail` points in [x_min, x_max] and at least one decade of range."""
    x = np.asarray(x, float)
    x = x[x > 0]

    def one_fit(xx):
        if xmin_fixed is not None:
            a, n = mle_exponent(xx, xmin_fixed, xmax)
            return a, xmin_fixed, ks_distance(xx, a, xmin_fixed, xmax), n
        cands = xmin_candidates
        if cands is None:
            u = np.unique(xx)
            hi = (xmax / 10.0) if xmax else np.quantile(xx, 0.999) / 10.0
            u = u[u <= hi]
            if len(u) > 60:
                u = np.unique(np.quantile(u, np.linspace(0, 1, 60)))
            cands = u
        best = (np.nan, np.nan, np.inf, 0)
        for xm in cands:
            a, n = mle_exponent(xx, xm, xmax)
            if n < min_tail or not np.isfinite(a):
                continue
            d = ks_distance(xx, a, xm, xmax)
            if d < best[2]:
                best = (a, xm, d, n)
        return best

    a, xm, d, n = one_fit(x)
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        xb = rng.choice(x, size=len(x), replace=True)
        boots.append(one_fit(xb)[0])
    boots = np.array(boots)
    err = np.nanstd(boots) if n_boot else (a - 1) / np.sqrt(max(n, 1))
    return {"alpha": -a, "err": err, "err_analytic": (a - 1) / np.sqrt(max(n, 1)),
            "xmin": xm, "xmax": xmax, "n_tail": n, "n_total": len(x), "ks": d}


def validate_mle(verbose=True):
    """Recover known exponents from synthetic samples."""
    rng = np.random.default_rng(1)
    out = []
    for a_true in (1.5, 2.0):
        u = rng.random(50000)
        xmin, xmax = 10.0, 1e4
        # inverse CDF of truncated power law
        x = (xmin ** (1 - a_true) + u * (xmax ** (1 - a_true) - xmin ** (1 - a_true))) ** (1 / (1 - a_true))
        x = np.concatenate([x, rng.uniform(1, 10, 20000)])   # contaminating head
        f = fit_power_law(x, xmax=xmax, n_boot=20)
        out.append((a_true, -f["alpha"], f["err"], f["xmin"]))
        if verbose:
            print(f"  [MLE validation] true alpha=-{a_true}: fitted {f['alpha']:.3f} +/- {f['err']:.3f}"
                  f" (x_min={f['xmin']:.1f})")
        assert abs(-f["alpha"] - a_true) < 0.03
    return out


# ----------------------------------------------------------------------------- #
# 3. PDFs
# ----------------------------------------------------------------------------- #
def log_binned_pdf(x, n_bins=40, integer=False, xmin=None):
    x = np.asarray(x, float)
    x = x[x > 0]
    lo = xmin if xmin else x.min()
    edges = np.logspace(np.log10(lo), np.log10(x.max() * 1.0001), n_bins + 1)
    if integer:
        edges = np.unique(np.ceil(edges))
        if edges[-1] <= x.max():
            edges = np.append(edges, x.max() + 1)
        counts, _ = np.histogram(x, bins=edges)
        widths = np.diff(edges)                       # number of integers in [a, b)
        centers = np.sqrt(edges[:-1] * (edges[1:] - 1).clip(min=edges[:-1]))
    else:
        counts, _ = np.histogram(x, bins=edges)
        widths = np.diff(edges)
        centers = np.sqrt(edges[:-1] * edges[1:])
    pdf = counts / (len(x) * widths)
    ok = counts > 0
    return centers[ok], pdf[ok], counts[ok]


def power_law_curve(fit, x_all, xs):
    """Model PDF (normalised to the full sample) to overplot on log_binned_pdf."""
    a = -fit["alpha"]; xmin = fit["xmin"]; xmax = fit["xmax"]
    frac = fit["n_tail"] / len(np.asarray(x_all)[np.asarray(x_all) > 0])
    if xmax is None:
        C = (a - 1) * xmin ** (a - 1)
    else:
        C = (1 - a) / (xmax ** (1 - a) - xmin ** (1 - a))
    return frac * C * xs ** (-a)
