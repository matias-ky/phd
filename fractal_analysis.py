# -*- coding: utf-8 -*-
"""
fractal_analysis.py
===================

Stand-alone script that produces the new Section 3.2 material requested for the
paper (box-counting fractal dimensions of avalanches):

  Fig. A  fig_boxcounting_example.pdf : box-counting method illustrated on one
                                        LH91 avalanche and one deterministic
                                        avalanche (style of Morales &
                                        Charbonneau 2009, Fig. 6), plus the
                                        log N(l) vs log l fit.
  Fig. B  fig_D_distributions.pdf     : PDFs of D^t and D^P for both models.
  Fig. C  fig_D_vs_delta_eps.pdf      : (a) mean D^t vs box-size window delta
                                        (style of M&C 2009, Fig. 7);
                                        (b) mean D^t and D^P vs driving eps.
  table_fractal.tex / summary_fractal.csv : numbers to quote in the text.

Place this file in the SAME folder as main_simulation.py (it imports the
simulation kernels from libs/functions.py, so the avalanche physics is exactly
the one used in the rest of the paper) and run:

    python fractal_analysis.py              # simulate (cached) + plots
    python fractal_analysis.py --plots-only # only re-make plots from cache

Box-counting algorithm (this is what the paper text describes)
---------------------------------------------------------------
For every avalanche:
 1. Binary masks. Total mask  = nodes flagged at least once during the
    avalanche (union of the per-iteration masks, i.e. the set whose size is
    A^t).  Peak mask = nodes flagged at the iteration of MAXIMUM ENERGY
    RELEASE (argmax of e_lib inside the avalanche), i.e. the set whose size
    is A^P.
 2. Selection. Linear extent L = max(height, width) of the bounding box of
    the total mask. Only avalanches with L >= N/4 are kept.
 3. Covering. The bounding box of the total mask is zero-padded to the next
    power of two and covered with a regular grid of square boxes of side
    l = 1, 2, 4, ..., l_max (lattice units), grid origin at the bounding-box
    corner, l_max = largest power of 2 <= L/4 (so >= 3 box sizes for N=64,
    >= 4 for N=128).  N(l) = number of boxes containing >= 1 flagged node.
    The SAME grid is used for the total and the peak mask.
 4. Fit. D = - slope of the ordinary least-squares fit of log N(l) vs log l
    over l in [1, l_max].
 5. Statistics. Reported values are mean +/- standard deviation over the
    selected avalanches (standard error also written to the CSV).
The method is validated at start-up on sets of known dimension
(filled square -> 2, line -> 1, Sierpinski gasket -> 1.585).
"""

import os
import sys
import csv
import time
import argparse
import multiprocessing as mp

import numpy as np
import scipy.ndimage as ndi
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# -----------------------------------------------------------------------------
# CONFIGURATION (edit here)
# -----------------------------------------------------------------------------
SEED = 7
ZC = 1
ITER_PER_CHUNK = 1000          # same meaning as ITERATIONS in main_simulation.py
N_CHUNKS = 2000                # 2000 x 1000 = 2e6 iterations (as main_simulation.py)
BURN_IN_CHUNKS = 500           # chunks discarded before measuring. The deterministic
                               # model needs ~1/eps iterations to reach the stationary
                               # state from the LH91 grid (~300 chunks for eps=1e-6, N=128);
                               # check fig_stationarity_check_all.pdf
MIN_EXTENT_FRACTION = 0.25     # keep avalanches with bounding-box extent >= N/4
D_NC = 0.1                     # non-conservation parameter of deterministic model
                               # (same value hard-coded in simulacion_completa)
# Lattices: key = label used in the paper, value = file in initial_grids/
GRIDS = {64: "B_N62_Zc1.npz", 128: "B_N128_Zc1.npz"}
EPSILONS = [1e-6, 1e-5, 1e-4]
N_WORKERS = max(1, min(8, (os.cpu_count() or 2) - 1))
OUT_DIR = "fractal_results"
MODES = ["cluster", "full"]    # both are computed; plots/tables made for each
MODE = "cluster"
N_EXAMPLES = 40                # masks stored for Fig. A (extent between 0.4N and 0.6N)
MAX_L_POW = 7                  # store N(l) for l = 2^0 ... 2^7

# -----------------------------------------------------------------------------
# BOX COUNTING
# -----------------------------------------------------------------------------
BOX_SIZES_ALL = 2 ** np.arange(MAX_L_POW + 1)


def bounding_box(mask):
    rows = np.where(mask.any(axis=1))[0]
    cols = np.where(mask.any(axis=0))[0]
    return rows[0], rows[-1] + 1, cols[0], cols[-1] + 1


def crop_and_pad(mask, bbox):
    """Crop mask to bbox and zero-pad (bottom/right) to the next power of two."""
    r0, r1, c0, c1 = bbox
    sub = mask[r0:r1, c0:c1]
    L = max(sub.shape)
    P = 1 << int(np.ceil(np.log2(max(L, 1))))
    out = np.zeros((P, P), dtype=bool)
    out[: sub.shape[0], : sub.shape[1]] = sub
    return out


def box_counts(padded, sizes):
    """N(l) for each box side l (powers of two dividing padded.shape[0])."""
    P = padded.shape[0]
    counts = np.full(len(sizes), np.nan)
    for i, l in enumerate(sizes):
        if l > P:
            break
        n = P // l
        blocks = padded.reshape(n, l, n, l).any(axis=(1, 3))
        counts[i] = blocks.sum()
    return counts


def fit_dimension(sizes, counts):
    ok = np.isfinite(counts) & (counts > 0)
    if ok.sum() < 2:
        return np.nan
    slope = np.polyfit(np.log(sizes[ok]), np.log(counts[ok]), 1)[0]
    return -slope


def l_max_for_extent(L):
    return int(2 ** np.floor(np.log2(max(L / 4.0, 1))))


def fractal_dimensions(total_mask, peak_mask):
    """Return (D_t, D_P, L, counts_t, counts_P) following the documented recipe."""
    bbox = bounding_box(total_mask)
    L = max(bbox[1] - bbox[0], bbox[3] - bbox[2])
    pt = crop_and_pad(total_mask, bbox)
    pp = crop_and_pad(peak_mask, bbox)          # same grid for the peak mask
    ct = box_counts(pt, BOX_SIZES_ALL)
    cp = box_counts(pp, BOX_SIZES_ALL)
    lmax = l_max_for_extent(L)
    sel = BOX_SIZES_ALL <= lmax
    return (fit_dimension(BOX_SIZES_ALL[sel], ct[sel]),
            fit_dimension(BOX_SIZES_ALL[sel], cp[sel]), L, ct, cp)


def validate_box_counting():
    """Check the algorithm on sets of known dimension (printed at start-up)."""
    n = 64
    sq = np.ones((n, n), bool)
    line = np.zeros((n, n), bool); line[n // 2, :] = True
    i, j = np.indices((n, n)); sier = (i & j) == 0
    out = {}
    for name, m, true in [("filled square", sq, 2.0), ("line", line, 1.0),
                          ("Sierpinski", sier, np.log(3) / np.log(2))]:
        D, _, _, _, _ = fractal_dimensions(m, m)
        out[name] = (D, true)
        print(f"  [validation] {name:14s} D = {D:.3f}  (exact {true:.3f})")
    return out


# -----------------------------------------------------------------------------
# SIMULATION + MEASUREMENT
# -----------------------------------------------------------------------------
def segment_avalanches(e_lib, areas):
    """Yield (list_of_masks, list_of_energies) for each avalanche in a chunk.
    An avalanche = maximal run of consecutive iterations with flagged nodes."""
    cur_m, cur_e = [], []
    for e, m in zip(e_lib, areas):
        if m.any():
            cur_m.append(m > 0); cur_e.append(e)
        elif cur_m:
            yield cur_m, cur_e
            cur_m, cur_e = [], []
    if cur_m:
        yield cur_m, cur_e


def run_config(args):
    """Simulate one (model, N, eps) configuration and measure avalanches."""
    model, N_label, eps, seed = args
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from libs.functions import lu_ham_standard, lu_ham_deterministic, set_seed

    tag = config_tag(model, N_label, eps)
    path = os.path.join(OUT_DIR, f"{tag}.npz")
    if os.path.exists(path):
        print(f"[{tag}] cached -> {path}")
        return path

    set_seed(seed)
    B = np.load(os.path.join("initial_grids", GRIDS[N_label]))["B"].astype(np.float32)
    N_int = B.shape[0] - 2
    min_extent = MIN_EXTENT_FRACTION * N_label

    rec = {k: [] for k in ("A_t", "A_P", "L_full", "L_cl", "n_clusters",
                           "D_t_full", "D_P_full", "D_t_cluster", "D_P_cluster")}
    cnt = {k: [] for k in ("counts_t_full", "counts_P_full", "counts_t_cluster", "counts_P_cluster")}
    ex = {"full": [], "cluster": []}       # (size, total_mask, peak_mask)
    etot_trace = []
    nanc = np.full(len(BOX_SIZES_ALL), np.nan)
    n_aval = 0
    t0 = time.time()
    for chunk in range(BURN_IN_CHUNKS + N_CHUNKS):
        if model == "standard":
            e_lib, e_tot, B, _, areas = lu_ham_standard(B, N_int, ZC, ITER_PER_CHUNK)
        else:
            e_lib, e_tot, B, _, areas = lu_ham_deterministic(B, ZC, ITER_PER_CHUNK, eps, D_NC)
        etot_trace.append(float(np.mean(e_tot)))
        if chunk < BURN_IN_CHUNKS:
            continue
        for masks, energies in segment_avalanches(e_lib, areas):
            n_aval += 1
            total = np.logical_or.reduce(masks)
            ipk = int(np.argmax(energies))          # iteration of max energy release
            peak = masks[ipk]
            bb = bounding_box(total)
            L_full = max(bb[1] - bb[0], bb[3] - bb[2])
            if L_full < min_extent:                 # cluster extent <= full extent
                continue
            lab, ncl = ndi.label(total, structure=np.ones((3, 3)))
            sizes = np.bincount(lab.ravel())[1:]
            cl = lab == (np.argmax(sizes) + 1)
            bbc = bounding_box(cl)
            L_cl = max(bbc[1] - bbc[0], bbc[3] - bbc[2])
            dtf, dpf, _, ctf, cpf = fractal_dimensions(total, peak)
            rec["A_t"].append(total.sum()); rec["A_P"].append(peak.sum())
            rec["L_full"].append(L_full); rec["L_cl"].append(L_cl); rec["n_clusters"].append(ncl)
            rec["D_t_full"].append(dtf); rec["D_P_full"].append(dpf)
            cnt["counts_t_full"].append(ctf); cnt["counts_P_full"].append(cpf)
            if 0.4 * N_label <= L_full <= 0.6 * N_label and len(ex["full"]) < N_EXAMPLES:
                ex["full"].append((total.sum(), total, peak))
            if L_cl >= min_extent:
                dtc, dpc, _, ctc, cpc = fractal_dimensions(cl, peak & cl)
                if 0.4 * N_label <= L_cl <= 0.6 * N_label and len(ex["cluster"]) < N_EXAMPLES:
                    ex["cluster"].append((cl.sum(), cl, peak & cl))
            else:
                dtc = dpc = np.nan; ctc = cpc = nanc
            rec["D_t_cluster"].append(dtc); rec["D_P_cluster"].append(dpc)
            cnt["counts_t_cluster"].append(ctc); cnt["counts_P_cluster"].append(cpc)
        if (chunk + 1) % max(1, (BURN_IN_CHUNKS + N_CHUNKS) // 10) == 0:
            print(f"[{tag}] chunk {chunk+1}/{BURN_IN_CHUNKS+N_CHUNKS} "
                  f"selected={len(rec['A_t'])} total_aval={n_aval} ({time.time()-t0:.0f}s)")

    out = {k: np.array(v, dtype=float) for k, v in rec.items()}
    out.update({k: np.array(v, dtype=float).reshape(-1, len(BOX_SIZES_ALL)) for k, v in cnt.items()})
    for k in ex:
        shape = (0,) + total_shape(B)
        out[f"ex_total_{k}"] = np.array([e[1] for e in ex[k]]) if ex[k] else np.zeros(shape, bool)
        out[f"ex_peak_{k}"] = np.array([e[2] for e in ex[k]]) if ex[k] else np.zeros(shape, bool)
    os.makedirs(OUT_DIR, exist_ok=True)
    np.savez_compressed(path, etot_trace=np.array(etot_trace), n_avalanches=n_aval,
                        burn_in=BURN_IN_CHUNKS, N=N_label, eps=eps if eps else 0.0, **out)
    print(f"[{tag}] done: {len(rec['A_t'])} (full) / "
          f"{int(np.isfinite(out['D_t_cluster']).sum())} (cluster) selected of {n_aval} -> {path}")
    return path


def total_shape(B):
    return (B.shape[0] + 2, B.shape[1] + 2)


def config_tag(model, N_label, eps):
    return f"st_N{N_label}" if model == "standard" else f"det_N{N_label}_eps{eps:.0e}"


def all_configs():
    cfgs = []
    for k, N in enumerate(GRIDS):
        cfgs.append(("standard", N, None, SEED + 100 * k))
        for j, eps in enumerate(EPSILONS):
            cfgs.append(("deterministic", N, eps, SEED + 100 * k + j + 1))
    return cfgs


def load(model, N, eps=None, mode=None):
    """Load cached results; expose generic keys D_t, D_P, counts_t, ex_total ...
    for the chosen mode ('cluster' = largest connected cluster, 'full' = whole
    avalanche, possibly made of disconnected islands)."""
    mode = mode or MODE
    r = dict(np.load(os.path.join(OUT_DIR, config_tag(model, N, eps) + ".npz")))
    ok = np.isfinite(r[f"D_t_{mode}"])
    for k in ("D_t", "D_P", "counts_t", "counts_P"):
        src = f"{k}_{mode}"
        r[k] = r[src][ok]
    r["ex_total"], r["ex_peak"] = r[f"ex_total_{mode}"], r[f"ex_peak_{mode}"]
    r["n_selected"] = int(ok.sum())
    return r


# -----------------------------------------------------------------------------
# PLOTS
# -----------------------------------------------------------------------------
plt.rcParams.update({"font.size": 10, "axes.labelsize": 11, "legend.fontsize": 9,
                     "xtick.direction": "in", "ytick.direction": "in"})
C_ST, C_DET = "#1f4e79", "#c0392b"


def pick_example(r):
    """Stored avalanche (extent ~N/2) whose D_t is closest to the population median."""
    if len(r["ex_total"]) == 0:
        raise RuntimeError("no selected avalanches stored for this configuration")
    med = np.nanmedian(r["D_t"])
    best, bestscore = 0, np.inf
    for i, (tot, pk) in enumerate(zip(r["ex_total"], r["ex_peak"])):
        d, _, L, _, _ = fractal_dimensions(tot, pk)
        score = abs(d - med)
        if score < bestscore:
            best, bestscore = i, score
    return r["ex_total"][best], r["ex_peak"][best]


def draw_coarse(ax, padded, l, L_show, title):
    n = padded.shape[0] // l
    blocks = padded.reshape(n, l, n, l).any(axis=(1, 3))
    for (i, j) in zip(*np.where(blocks)):
        ax.add_patch(Rectangle((j * l, i * l), l, l, color="k", lw=0))
    for x in range(0, L_show + 1, l):
        ax.axvline(x, color="0.7", lw=0.3 if l == 1 else 0.6)
        ax.axhline(x, color="0.7", lw=0.3 if l == 1 else 0.6)
    ax.set_xlim(0, L_show); ax.set_ylim(L_show, 0)
    ax.set_aspect("equal"); ax.set_title(title, fontsize=9)
    ax.tick_params(labelsize=7)
    return int(blocks.sum())


def fig_boxcounting_example(N=128, eps=1e-6, fname="fig_boxcounting_example"):
    rows = [("LH91", load("standard", N)),
            (rf"Deterministic, $\epsilon=10^{{{int(np.log10(eps))}}}$",
             load("deterministic", N, eps))]
    fig, axes = plt.subplots(2, 4, figsize=(12, 6.2),
                             gridspec_kw={"width_ratios": [1, 1, 1, 1.15]})
    for r, (name, res) in enumerate(rows):
        tot, pk = pick_example(res)
        bbox = bounding_box(tot)
        L = max(bbox[1] - bbox[0], bbox[3] - bbox[2])
        padded = crop_and_pad(tot, bbox)
        lmax = l_max_for_extent(L)
        show = [1, max(2, lmax // 4), lmax]
        for c, l in enumerate(show):
            nb = draw_coarse(axes[r, c], padded, l, L,
                             rf"$\ell={l}$, $N(\ell)={{}}$".format(0))
            axes[r, c].set_title(rf"$\ell={l}$,  $N(\ell)={nb}$", fontsize=9)
        axes[r, 0].set_ylabel(name)
        sizes = BOX_SIZES_ALL[BOX_SIZES_ALL <= lmax]
        ct = box_counts(padded, sizes)
        D = fit_dimension(sizes, ct)
        a = axes[r, 3]
        a.loglog(sizes, ct, "o", color=C_ST if r == 0 else C_DET, mfc="none")
        p = np.polyfit(np.log(sizes), np.log(ct), 1)
        xx = np.array([sizes[0] * 0.8, sizes[-1] * 1.25])
        a.loglog(xx, np.exp(np.polyval(p, np.log(xx))), "k--", lw=1,
                 label=rf"$D^t={D:.2f}$")
        a.set_xlabel(r"$\ell$ (lattice units)"); a.set_ylabel(r"$N(\ell)$")
        a.legend(frameon=False); a.set_title(rf"$A^t={int(tot.sum())}$, $L={L}$", fontsize=9)
    fig.tight_layout()
    save(fig, fname)


def fig_D_distributions(N=128, eps=1e-6, fname="fig_D_distributions"):
    st, det = load("standard", N), load("deterministic", N, eps)
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
    for ax, key, lab in [(axes[0], "D_t", r"$D^t$"), (axes[1], "D_P", r"$D^P$")]:
        lo = np.nanmin([np.nanmin(st[key]), np.nanmin(det[key])])
        hi = np.nanmax([np.nanmax(st[key]), np.nanmax(det[key])])
        bins = np.linspace(lo, hi, 30)
        for res, c, name in [(st, C_ST, "LH91"),
                             (det, C_DET, rf"Det. $\epsilon=10^{{{int(np.log10(eps))}}}$")]:
            d = res[key][np.isfinite(res[key])]
            ax.hist(d, bins=bins, density=True, histtype="step", lw=1.6, color=c,
                    label=rf"{name}: ${np.mean(d):.2f}\pm{np.std(d):.2f}$ (n={len(d)})")
            ax.axvline(np.mean(d), color=c, ls=":", lw=1)
        ax.set_xlabel(lab); ax.set_ylabel(f"PDF({lab})")
        ax.legend(frameon=False, fontsize=8, loc="upper left")
        ax.set_ylim(top=ax.get_ylim()[1] * 1.3)
    axes[0].set_title(f"(a) total area, N={N}", fontsize=10)
    axes[1].set_title(f"(b) peak of energy release, N={N}", fontsize=10)
    fig.tight_layout()
    save(fig, fname)


def fig_D_vs_delta_eps(fname="fig_D_vs_delta_eps"):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.6))
    # (a) D^t vs delta = l_max / N using the same selected sample at every delta
    ax = axes[0]
    markers = {64: "^", 128: "s"}
    for N in GRIDS:
        lmaxs = [l for l in (2, 4, 8, 16) if l <= N // 16]
        for model, eps, c, off in [("standard", None, C_ST, -0.0015),
                                   ("deterministic", 1e-6, C_DET, 0.0015)]:
            r = load(model, N, eps)
            means, stds, deltas = [], [], []
            for lm in lmaxs:
                sel = BOX_SIZES_ALL <= lm
                d = np.array([fit_dimension(BOX_SIZES_ALL[sel], ct[sel]) for ct in r["counts_t"]])
                means.append(np.nanmean(d)); stds.append(np.nanstd(d)); deltas.append(lm / N)
            lab = f"{'LH91' if model == 'standard' else 'Det.'}, N={N}"
            ax.errorbar(np.array(deltas) + off, means, yerr=stds, fmt=markers[N],
                        color=c, mfc="none" if model == "standard" else c, ms=6,
                        capsize=2, label=lab)
    ax.set_xlabel(r"$\delta=\ell_{\max}/N$"); ax.set_ylabel(r"$\langle D^t\rangle$")
    ax.set_title(r"(a) dependence on box-size window ($\epsilon=10^{-6}$)", fontsize=10)
    ax.legend(frameon=False, ncol=2, fontsize=8)
    # (b) D^t and D^P vs eps
    ax = axes[1]
    for N in GRIDS:
        st = load("standard", N)
        for key, ls in [("D_t", "-"), ("D_P", "--")]:
            m = [np.nanmean(load("deterministic", N, e)[key]) for e in EPSILONS]
            s = [np.nanstd(load("deterministic", N, e)[key]) for e in EPSILONS]
            lab = rf"${'D^t' if key == 'D_t' else 'D^P'}$ det., N={N}"
            ax.errorbar(EPSILONS, m, yerr=s, fmt=markers[N] + ls, color=C_DET,
                        mfc=C_DET if N == 128 else "none", capsize=2, label=lab)
            ms, ss = np.nanmean(st[key]), np.nanstd(st[key])
            ax.axhline(ms, color=C_ST, ls=ls, lw=1 if N == 64 else 1.6, alpha=0.8)
    ax.plot([], [], color=C_ST, ls="-", label=r"LH91 $D^t$ (thin N=64, thick N=128)")
    ax.plot([], [], color=C_ST, ls="--", label=r"LH91 $D^P$")
    ax.set_xscale("log"); ax.set_xlabel(r"$\epsilon$"); ax.set_ylabel(r"$\langle D\rangle$")
    ax.set_title(r"(b) dependence on driving rate $\epsilon$", fontsize=10)
    ax.legend(frameon=False, fontsize=7, ncol=2, loc="upper center",
              bbox_to_anchor=(0.5, -0.18))
    fig.tight_layout()
    save(fig, fname)


def fig_stationarity(fname="fig_stationarity_check"):
    """Mean total energy sum(B^2) per chunk: verify the burn-in is long enough."""
    fig, axes = plt.subplots(1, len(GRIDS), figsize=(10, 3.4))
    for ax, N in zip(np.atleast_1d(axes), GRIDS):
        for model, eps in [("standard", None)] + [("deterministic", e) for e in EPSILONS]:
            r = dict(np.load(os.path.join(OUT_DIR, config_tag(model, N, eps) + ".npz")))
            ax.plot(r["etot_trace"], lw=0.8, label=config_tag(model, N, eps))
        ax.axvline(int(r["burn_in"]), color="k", ls=":", lw=1)
        ax.set_xlabel("chunk (x %d iterations)" % ITER_PER_CHUNK)
        ax.set_ylabel(r"$\langle \sum B^2 \rangle$"); ax.set_title(f"N={N}")
        ax.legend(frameon=False, fontsize=7)
    fig.tight_layout()
    global MODE
    m, MODE = MODE, "all"
    save(fig, fname)
    MODE = m


def save(fig, fname):
    os.makedirs(os.path.join(OUT_DIR, "plots"), exist_ok=True)
    fname = f"{fname}_{MODE}"
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUT_DIR, "plots", f"{fname}.{ext}"), dpi=300,
                    bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {OUT_DIR}/plots/{fname}.pdf/.png")



# -----------------------------------------------------------------------------
# PAPER FIGURES (combine 'full' and 'cluster' measurements)
# -----------------------------------------------------------------------------
def _ms(arr):
    a = arr[np.isfinite(arr)]
    return (np.mean(a), np.std(a), len(a)) if len(a) else (np.nan, np.nan, 0)


def fig_paper_distributions(N=128, eps=1e-6, fname="fig_paper_D_distributions"):
    """PDF(D^t) and PDF(D^P): LH91, deterministic whole avalanche, deterministic
    largest connected cluster."""
    st = load("standard", N, mode="full")
    df = load("deterministic", N, eps, mode="full")
    dc = load("deterministic", N, eps, mode="cluster")
    e = int(np.log10(eps))
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    for ax, key, lab in [(axes[0], "D_t", r"$D^t$"), (axes[1], "D_P", r"$D^P$")]:
        bins = np.linspace(0.3, 2.0, 52)
        for res, c, ls, name in [(st, C_ST, "-", "LH91"),
                                 (df, C_DET, "-", rf"Det. $\epsilon=10^{{{e}}}$, whole avalanche"),
                                 (dc, C_DET, "--", rf"Det. $\epsilon=10^{{{e}}}$, largest cluster")]:
            m, sd, n = _ms(res[key])
            ax.hist(res[key][np.isfinite(res[key])], bins=bins, density=True,
                    histtype="step", lw=1.5, color=c, ls=ls,
                    label=rf"{name}: ${m:.2f}\pm{sd:.2f}$")
        ax.set_xlabel(lab); ax.set_ylabel(f"PDF({lab})")
        ax.set_yscale("log"); ax.set_ylim(1e-3, 30)
        ax.legend(frameon=False, fontsize=7.5, loc="upper left")
    axes[0].set_title(f"(a) total area $A^t$, N={N}", fontsize=10)
    axes[1].set_title(f"(b) peak area $A^P$, N={N}", fontsize=10)
    fig.tight_layout()
    _save_plain(fig, fname)


def fig_paper_summary(fname="fig_paper_D_summary"):
    """(a) <D^t> vs delta = l_max/N, (b) <D^t> vs eps, (c) <D^P> vs eps.
    Filled symbols: whole avalanche; open symbols: largest connected cluster."""
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.0))
    mk = {64: "^", 128: "s"}
    # (a)
    ax = axes[0]
    for N in GRIDS:
        lmaxs = [l for l in (2, 4, 8, 16) if l <= N // 16]
        for model, eps, c, mode, off in [("standard", None, C_ST, "full", -0.0012),
                                         ("deterministic", 1e-6, C_DET, "full", 0.0),
                                         ("deterministic", 1e-6, C_DET, "cluster", 0.0012)]:
            r = load(model, N, eps, mode=mode)
            m, sd = [], []
            for lm in lmaxs:
                sel = BOX_SIZES_ALL <= lm
                d = np.array([fit_dimension(BOX_SIZES_ALL[sel], ct[sel]) for ct in r["counts_t"]])
                m.append(np.nanmean(d)); sd.append(np.nanstd(d))
            name = "LH91" if model == "standard" else f"Det. ({'whole' if mode == 'full' else 'cluster'})"
            ax.errorbar(np.array(lmaxs) / N + off, m, yerr=sd, fmt=mk[N], color=c, ms=6,
                        mfc="none" if mode == "cluster" else c, capsize=2,
                        label=f"{name}, N={N}")
    ax.set_xlabel(r"$\delta=\ell_{\max}/N$"); ax.set_ylabel(r"$\langle D^t\rangle$")
    ax.set_title(r"(a) box-size window ($\epsilon=10^{-6}$)", fontsize=10)
    ax.legend(frameon=False, fontsize=6.5, ncol=2, loc="lower left")
    # (b), (c)
    for ax, key, lab in [(axes[1], "D_t", r"$\langle D^t\rangle$"), (axes[2], "D_P", r"$\langle D^P\rangle$")]:
        for N in GRIDS:
            for mode, mfc, ls, off in [("full", None, "-", 0.94), ("cluster", "none", "--", 1.06)]:
                m = [_ms(load("deterministic", N, e, mode=mode)[key])[0] for e in EPSILONS]
                sd = [_ms(load("deterministic", N, e, mode=mode)[key])[1] for e in EPSILONS]
                ax.errorbar(np.array(EPSILONS) * off, m, yerr=sd, fmt=mk[N] + ls, color=C_DET,
                            mfc=mfc if mfc else C_DET, ms=6, capsize=2, lw=1,
                            label=f"Det. {'whole' if mode == 'full' else 'cluster'}, N={N}")
            ms_, sd_, _ = _ms(load("standard", N, mode="full")[key])
            ax.axhspan(ms_ - sd_, ms_ + sd_, color=C_ST, alpha=0.08 if N == 64 else 0.15, lw=0)
            ax.axhline(ms_, color=C_ST, lw=1 if N == 64 else 2, ls=":" if N == 64 else "-",
                       label=rf"LH91 N={N} (mean $\pm$ std)")
        ax.set_xscale("log"); ax.set_xlabel(r"$\epsilon$"); ax.set_ylabel(lab)
        ax.legend(frameon=False, fontsize=6.5, ncol=2, loc="lower left")
    axes[1].set_title(r"(b) total area vs driving rate", fontsize=10)
    axes[2].set_title(r"(c) peak area vs driving rate", fontsize=10)
    axes[1].set_ylim(0.9, 2.0); axes[2].set_ylim(0.7, 1.7)
    fig.tight_layout()
    _save_plain(fig, fname)


def _save_plain(fig, fname):
    global MODE
    m, MODE = MODE, "paper"
    save(fig, fname)
    MODE = m

# -----------------------------------------------------------------------------
# TABLES
# -----------------------------------------------------------------------------
def write_tables():
    rows = []
    for mode, (model, N, eps, _) in [(m, c) for m in MODES for c in all_configs()]:
        r = load(model, N, eps, mode)
        row = {"mode": mode, "model": "LH91" if model == "standard" else "Deterministic",
               "N": N, "eps": "-" if eps is None else f"{eps:.0e}",
               "n_avalanches": int(r["n_avalanches"]), "n_selected": r["n_selected"],
               "mean_n_clusters_selected": float(np.mean(r["n_clusters"])) if len(r["n_clusters"]) else np.nan}
        for key in ("D_t", "D_P"):
            d = r[key][np.isfinite(r[key])]
            if len(d) == 0:
                d = np.array([np.nan])
            row[f"{key}_mean"] = np.mean(d); row[f"{key}_std"] = np.std(d)
            row[f"{key}_sem"] = np.std(d) / np.sqrt(len(d)); row[f"{key}_median"] = np.median(d)
        rows.append(row)
    with open(os.path.join(OUT_DIR, "summary_fractal.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    for mode in MODES:
      with open(os.path.join(OUT_DIR, f"table_fractal_{mode}.tex"), "w") as f:
        f.write("\\begin{tabular}{|l|c|c|c|c|c|}\n\\hline\n")
        f.write("Model & $N$ & $\\epsilon$ & $n$ & $D^{t}$ & $D^{P}$ \\\\ \\hline\n")
        for r in [x for x in rows if x["mode"] == mode]:
            eps = "--" if r["eps"] == "-" else f"$10^{{{int(np.log10(float(r['eps'])))}}}$"
            f.write(f"{r['model']} & {r['N']} & {eps} & {r['n_selected']} & "
                    f"${r['D_t_mean']:.2f} \\pm {r['D_t_std']:.2f}$ & "
                    f"${r['D_P_mean']:.2f} \\pm {r['D_P_std']:.2f}$ \\\\\n")
        f.write("\\hline\n\\end{tabular}\n")
    # combined table for the paper
    with open(os.path.join(OUT_DIR, "table_fractal_paper.tex"), "w") as f:
        f.write("\\begin{tabular}{|c|l|c|c|c|c|c|}\n\\hline\n")
        f.write("$N$ & Model & $\\epsilon$ & $D^{t}$ (whole) & $D^{t}$ (cluster) & "
                "$D^{P}$ (whole) & $D^{P}$ (cluster) \\\\ \\hline\n")
        for N in GRIDS:
            for r in [x for x in rows if x["mode"] == "full" and x["N"] == N]:
                c = [x for x in rows if x["mode"] == "cluster" and x["N"] == N
                     and x["model"] == r["model"] and x["eps"] == r["eps"]][0]
                eps = "--" if r["eps"] == "-" else f"$10^{{{int(np.log10(float(r['eps'])))}}}$"
                f.write(f"{N} & {r['model']} & {eps} & "
                        f"${r['D_t_mean']:.2f}\\pm{r['D_t_std']:.2f}$ & ${c['D_t_mean']:.2f}\\pm{c['D_t_std']:.2f}$ & "
                        f"${r['D_P_mean']:.2f}\\pm{r['D_P_std']:.2f}$ & ${c['D_P_mean']:.2f}\\pm{c['D_P_std']:.2f}$ \\\\\n")
            f.write("\\hline\n")
        f.write("\\end{tabular}\n")
    print("\nSUMMARY (mean +/- std over selected avalanches)")
    for r in rows:
        print(f"  [{r['mode']:7s}] {r['model']:13s} N={r['N']:3d} eps={r['eps']:>6s} n={r['n_selected']:6d}  "
              f"D_t={r['D_t_mean']:.3f}+/-{r['D_t_std']:.3f}  D_P={r['D_P_mean']:.3f}+/-{r['D_P_std']:.3f}")


# -----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plots-only", action="store_true")
    ap.add_argument("--chunks", type=int, default=None, help="override N_CHUNKS")
    ap.add_argument("--burn-in", type=int, default=None, help="override BURN_IN_CHUNKS")
    ap.add_argument("--out", default=None, help="override OUT_DIR")
    a = ap.parse_args()
    global N_CHUNKS, BURN_IN_CHUNKS, OUT_DIR
    if a.chunks is not None: N_CHUNKS = a.chunks
    if a.burn_in is not None: BURN_IN_CHUNKS = a.burn_in
    if a.out is not None: OUT_DIR = a.out
    os.makedirs(OUT_DIR, exist_ok=True)

    print("Box-counting validation:")
    validate_box_counting()
    if not a.plots_only:
        cfgs = all_configs()
        print(f"\nRunning {len(cfgs)} configurations on {N_WORKERS} workers "
              f"({BURN_IN_CHUNKS}+{N_CHUNKS} chunks x {ITER_PER_CHUNK} it)")
        ctx = mp.get_context("spawn")
        with ctx.Pool(N_WORKERS, initializer=_init_worker,
                      initargs=(N_CHUNKS, BURN_IN_CHUNKS, OUT_DIR)) as pool:
            pool.map(run_config, cfgs, chunksize=1)
    write_tables()
    fig_stationarity()
    fig_paper_distributions()
    fig_paper_summary()
    global MODE
    for MODE in MODES:
        print(f"\nPlots for mode = {MODE}")
        for f in (fig_boxcounting_example, fig_D_distributions, fig_D_vs_delta_eps):
            try:
                f()
            except Exception as err:          # e.g. too few avalanches in a short test run
                print(f"  could not make {f.__name__}: {err}")


def _init_worker(n_chunks, burn_in, out_dir):
    global N_CHUNKS, BURN_IN_CHUNKS, OUT_DIR
    N_CHUNKS, BURN_IN_CHUNKS, OUT_DIR = n_chunks, burn_in, out_dir


if __name__ == "__main__":
    main()
