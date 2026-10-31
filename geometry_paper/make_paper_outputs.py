# -*- coding: utf-8 -*-
"""
make_paper_outputs.py -- all figures, tables and numbers of the paper from results/.

    python make_paper_outputs.py                     # default: results/, area = M
    python make_paper_outputs.py --area U            # only unstable nodes (robustness)

Area definitions
  M : unstable nodes plus their four nearest neighbours, i.e. every node whose value
      changes during the iteration (default; definition of libs/areas.py)
  U : only the unstable nodes (robustness check)
Peak area A^P = maximum number of nodes simultaneously involved in one iteration.

Outputs in <results>/paper_<area>/ : PDF+PNG figures, LaTeX tables, numbers.json
"""
import os
import sys
import json
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.gridspec import GridSpec

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import analysis as an                                   # noqa: E402
import uncertainties as un                              # noqa: E402
from run_simulations import config_tag, SIZES, EPSILONS  # noqa: E402

# ----------------------------------------------------------------------------- #
# style (colours of the original figures: matplotlib C0 blue / C1 orange)
# ----------------------------------------------------------------------------- #
C_LH = "#1f77b4"            # LH91
C_DET = "#ff7f0e"           # deterministic (eps = 1e-6, reference)
C_EPS = {1e-6: "#ff7f0e", 1e-5: "#2ca02c", 1e-4: "#d62728"}
MK_N = {64: "^", 128: "s", 256: "o"}
plt.rcParams.update({
    "font.size": 11, "axes.labelsize": 13, "legend.fontsize": 9,
    "xtick.direction": "in", "ytick.direction": "in", "xtick.top": True, "ytick.right": True,
    "axes.grid": True, "grid.linestyle": ":", "grid.alpha": 0.35, "grid.color": "0.3",
    "legend.frameon": False, "savefig.bbox": "tight", "savefig.dpi": 300,
})
REF_N = 128
REF_EPS = 1e-6


def eps_tex(e):
    return rf"10^{{{int(round(np.log10(e)))}}}"


class Results:
    def __init__(self, folder, area, lh_folder=None):
        self.folder = folder; self.area = area
        self.lh_folder = lh_folder or folder
        self._cache = {}

    def _dir(self, model, N, eps):
        base = self.lh_folder if model == "LH91" else self.folder
        return os.path.join(base, config_tag(model, N, eps))

    def av(self, model, N, eps=0.0):
        key = ("av", model, N, eps)
        if key not in self._cache:
            self._cache[key] = dict(np.load(os.path.join(self._dir(model, N, eps), "avalanches.npz")))
        return self._cache[key]

    def fr(self, model, N, eps=0.0):
        key = ("fr", model, N, eps)
        if key not in self._cache:
            self._cache[key] = dict(np.load(os.path.join(self._dir(model, N, eps), "fractal.npz")))
        return self._cache[key]

    def ex(self, model, N, eps=0.0):
        return np.load(os.path.join(self._dir(model, N, eps), "examples.npz"))

    def meta(self, model, N, eps=0.0):
        return json.load(open(os.path.join(self._dir(model, N, eps), "DONE.json")))

    def has(self, model, N, eps=0.0):
        return os.path.exists(os.path.join(self._dir(model, N, eps), "DONE.json"))

    # area columns
    def At(self, model, N, eps=0.0):
        return self.av(model, N, eps)["A_t" if self.area == "U" else "A_t_M"]

    def AP(self, model, N, eps=0.0):
        return self.av(model, N, eps)["A_P" if self.area == "U" else "A_P_M"]

    def D(self, model, N, eps, which, mode):
        """which in {'t','P'}, mode in {'whole','cluster'}; finite values only."""
        f = self.fr(model, N, eps)
        d = f[f"D{which}_{self.area}_{mode}"]
        return d[np.isfinite(d)]

    def counts(self, model, N, eps, mode):
        f = self.fr(model, N, eps)
        c = f[f"cnt_{self.area}_{mode}"]
        return c[np.isfinite(c[:, 0])]


def mstd(x):
    x = np.asarray(x); x = x[np.isfinite(x)]
    return (np.mean(x), np.std(x), len(x)) if len(x) else (np.nan, np.nan, 0)


def save(fig, out, name):
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(out, f"{name}.{ext}"))
    plt.close(fig)
    print("  saved", name)


# ----------------------------------------------------------------------------- #
# power-law fits
# ----------------------------------------------------------------------------- #
# Fixed fitting windows in units of N^2 (same rule for every model, N and eps):
#   A^t : [max(15, 1e-3 N^2), 1e-1 N^2]     A^P : [max(15, 10^-3.5 N^2), 10^-1.5 N^2]
# The upper bound excludes the finite-size range (fall-off of LH91, system-size bump of
# the deterministic model, identified from the local logarithmic slope of the PDFs);
# the floor of 15 nodes avoids the lattice-discreteness range of small areas.
WINDOW = {"A_t": (1e-3, 1e-1), "A_P": (10 ** -3.5, 10 ** -1.5)}
AREA_FLOOR = 15.0


def _window(N, key, shift=1.0):
    lo, hi = WINDOW[key]
    return max(AREA_FLOOR, lo * N * N * shift), hi * N * N * shift


def area_fit(x, N, n_boot=None, key="A_t"):
    """Truncated power-law MLE on the fixed window (uncertainties.py):
    err = batch-means error over 20 blocks in time (statistical),
    err_sys = half range of alpha when x_min and x_max are varied independently by
    factors 1/2 ... 2 (removing/adding up to ~3 log-bins at each end of the window)."""
    lo, hi = _window(N, key)
    return un.exponent_with_errors(x, lo, hi)


def energy_fit(E):
    """x_min from the KS criterion (Clauset et al. 2009), no upper bound; err_sys from
    x_min * (1/2 ... 2) and x_max in {inf, 99.99%, 99.9% quantile}."""
    E = np.asarray(E, float)
    xmin = an.fit_power_law(E, n_boot=0)["xmin"]
    q = [np.inf, float(np.quantile(E, 0.9999)), float(np.quantile(E, 0.999))]
    return un.exponent_with_errors(E, xmin, np.inf, sys_hi=q)


# Fixed windows in absolute area (same for all N): diagnostic of the N dependence.
ABS_WINDOWS = {"A_t": [(30, 300), (100, 1000), (300, 3000)], "A_P": [(30, 300), (60, 600)]}


def absolute_window_fits(R, sizes):
    res = {}
    for model, e in [("LH91", 0.0), ("DET", REF_EPS)]:
        for key in ("A_t", "A_P"):
            for lo, hi in ABS_WINDOWS[key]:
                for N in sizes:
                    # only windows well below the cutoff of that lattice
                    if hi > WINDOW[key][1] * N * N * 1.01 or not R.has(model, N, e):
                        continue
                    x = R.At(model, N, e) if key == "A_t" else R.AP(model, N, e)
                    f = un.exponent_with_errors(x, lo, hi)
                    res[f"{config_tag(model, N, e)}|{key}|{lo}-{hi}"] = f
    return res


def all_fits(R, sizes, n_boot):
    fits = {}
    for N in sizes:
        for model, eps in [("LH91", 0.0)] + [("DET", e) for e in EPSILONS]:
            if not R.has(model, N, eps):
                continue
            a = R.av(model, N, eps)
            fits[(model, N, eps)] = {
                "A_t": area_fit(R.At(model, N, eps), N, key="A_t"),
                "A_P": area_fit(R.AP(model, N, eps), N, key="A_P"),
                "E": energy_fit(a["E"]),
                "n_avalanches": int(len(a["E"])),
            }
            print(f"  fit {config_tag(model, N, eps)}: a_At={fits[(model, N, eps)]['A_t']['alpha']:.3f}"
                  f" a_AP={fits[(model, N, eps)]['A_P']['alpha']:.3f} a_E={fits[(model, N, eps)]['E']['alpha']:.3f}")
    return fits


# ----------------------------------------------------------------------------- #
# figures
# ----------------------------------------------------------------------------- #
def plot_pdf_with_fit(ax, x, fit, color, label, integer):
    xc, pdf, _ = an.log_binned_pdf(x, n_bins=40, integer=integer)
    ax.loglog(xc, pdf, "o", mfc="none", mec=color, ms=6, label=label)
    xs = np.logspace(np.log10(fit["xmin"]), np.log10(fit["xmax"] or xc.max()), 50)
    ax.loglog(xs, an.power_law_curve(fit, x, xs), "--", color="k", lw=1.6,
              label=rf"MLE fit: $\alpha={fit['alpha']:.2f}\pm{up2(total_err(fit)):.2f}$")


def fig_E(R, fits, out, N=None, eps=REF_EPS):
    N = N or REF_N
    fig, ax = plt.subplots(figsize=(5.2, 4.0))
    x = R.av("DET", N, eps)["E"]
    plot_pdf_with_fit(ax, x, fits[("DET", N, eps)]["E"], C_DET, rf"Deterministic, $\epsilon={eps_tex(eps)}$", False)
    ax.set_xlabel(r"$E$ [$e_0$]"); ax.set_ylabel(r"PDF($E$)"); ax.legend(loc="lower left")
    save(fig, out, "fig_E_det")


def fig_local_slope(R, out, sizes, eps=REF_EPS):
    """Local MLE exponent in sliding windows of half a decade, vs absolute area, for all N.
    Grey band: fitting window of the reference lattice. Curves stop at the upper bound of
    the fitting window of each N (beyond it the finite-size cutoff dominates)."""
    cent = 10 ** np.arange(1.25, 4.6, 0.125)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.0), sharey=True)
    for ax, key, lab in zip(axes, ("A_t", "A_P"), (r"A^{t}", r"A^{P}")):
        for model, e, c in [("LH91", 0.0, C_LH), ("DET", eps, C_DET)]:
            for N in sizes:
                if not R.has(model, N, e):
                    continue
                x = R.At(model, N, e) if key == "A_t" else R.AP(model, N, e)
                m = cent * 10 ** 0.25 <= _window(N, key)[1]
                ax.semilogx(cent[m], un.local_exponent(x, cent[m]), MK_N[N] + "-", color=c,
                            mfc="none", ms=5, lw=1,
                            label=f"{'LH91' if model == 'LH91' else 'Det.'}, N = {N}")
        lo, hi = _window(REF_N, key)
        ax.axvspan(lo, hi, color="0.85", zorder=0)
        ax.set_xlabel(rf"${lab}$"); ax.set_ylim(-2.2, -1.2)
        ax.set_title(f"({'a' if key == 'A_t' else 'b'})", loc="left", fontsize=11)
    axes[0].set_ylabel(r"local exponent $\alpha_{\rm loc}$")
    axes[0].legend(ncol=2, fontsize=8, loc="lower right")
    fig.tight_layout()
    save(fig, out, "fig_local_slope")


def fig_area_pdfs(R, fits, out, key, N=None, eps=REF_EPS):
    N = N or REF_N
    lab = r"A^{t}" if key == "A_t" else r"A^{P}"
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.0), sharey=True)
    for ax, (model, e, c, name) in zip(axes, [("LH91", 0.0, C_LH, "LH91"),
                                             ("DET", eps, C_DET, rf"Deterministic, $\epsilon={eps_tex(eps)}$")]):
        x = R.At(model, N, e) if key == "A_t" else R.AP(model, N, e)
        plot_pdf_with_fit(ax, x, fits[(model, N, e)][key], c, name, True)
        ax.set_xlabel(rf"${lab}$"); ax.legend(loc="lower left")
        ax.set_title(f"N = {N}", fontsize=10)
    axes[0].set_ylabel(rf"PDF(${lab}$)")
    fig.tight_layout()
    save(fig, out, f"fig_{key}_pdf")


def _draw_lattice_mask(ax, mask, title):
    h, w = mask.shape
    ax.imshow(np.where(mask, 0.35, 1.0), cmap="gray", vmin=0, vmax=1, origin="upper",
              extent=(0, w, h, 0), interpolation="nearest")
    ax.set_title(title, fontsize=11)
    ax.grid(False)
    ax.tick_params(labelsize=7)


def fig_avalanche_examples(R, out, N=None, eps=REF_EPS):
    """(A)-(C) LH91, (D)-(F) deterministic: first unstable nodes, peak, total area.
    Each row is drawn in a square window (side = extent of the avalanche + margin);
    axes in lattice units relative to that window."""
    N = N or REF_N
    fig, axes = plt.subplots(2, 3, figsize=(10.5, 7.4))
    for r, (model, e, kind) in enumerate([("LH91", 0.0, "mid"), ("DET", eps, "multi")]):
        ex = R.ex(model, N, e)
        keys = sorted({k.rsplit("_", 1)[0] for k in ex.files if k.startswith(kind)},
                      key=lambda k: int(k.split("_")[1]))
        if not keys:
            keys = sorted({k.rsplit("_", 1)[0] for k in ex.files})
        iu, ip = (0, 1) if R.area == "U" else (2, 3)
        # LH91: example with median-like size; DET: example with >=2 islands, largest area
        cand = keys[:20]
        best = max(cand, key=lambda k: ex[k + "_m4"][iu].sum())
        m4 = ex[best + "_m4"]; first = ex[best + "_first"].astype(bool)
        if R.area == "M":                      # show the first iteration with its neighbours
            f2 = first.copy()
            f2[1:, :] |= first[:-1, :]; f2[:-1, :] |= first[1:, :]
            f2[:, 1:] |= first[:, :-1]; f2[:, :-1] |= first[:, 1:]
            first = f2 & m4[2].astype(bool)
        tot, pk = m4[iu].astype(bool), m4[ip].astype(bool)
        h, w = tot.shape; S = max(h, w) + 8
        def sq(m):
            o = np.zeros((S, S), bool); r0 = (S - h) // 2; c0 = (S - w) // 2
            o[r0:r0 + h, c0:c0 + w] = m; return o
        for c, m in enumerate([first, pk, tot]):
            ax = axes[r, c]
            ax.imshow(np.where(sq(m), 0.3, 1.0), cmap="gray", vmin=0, vmax=1,
                      extent=(0, S, S, 0), interpolation="nearest")
            for x in range(0, S + 1, max(1, S // 32)):
                ax.axvline(x, color="0.85", lw=0.3); ax.axhline(x, color="0.85", lw=0.3)
            ax.grid(False); ax.tick_params(labelsize=7)
            ax.set_title(f"({'ABCDEF'[3 * r + c]})", fontsize=12)
        axes[r, 0].set_ylabel("LH91" if model == "LH91" else rf"Deterministic, $\epsilon={eps_tex(eps)}$")
    fig.tight_layout()
    save(fig, out, "fig_avalanche_examples")


def _pick_typical(R, model, N, eps):
    ex = R.ex(model, N, eps)
    keys = sorted({k.rsplit("_", 1)[0] for k in ex.files if k.startswith("mid")})
    med = np.median(R.D(model, N, eps, "t", "whole"))
    iu = 0 if R.area == "U" else 2
    best, sc = None, np.inf
    for k in keys:
        m = ex[k + "_m4"][iu].astype(bool)
        d = an.fractal_pair(m, m)[0]
        if abs(d - med) < sc:
            best, sc = m, abs(d - med)
    return best


def fig_boxcounting(R, out, N=None, eps=REF_EPS):
    N = N or REF_N
    fig, axes = plt.subplots(2, 4, figsize=(12.5, 6.3), gridspec_kw={"width_ratios": [1, 1, 1, 1.15]})
    for r, (model, e, c, name) in enumerate([("LH91", 0.0, C_LH, "LH91"),
                                            ("DET", eps, C_DET, rf"Det., $\epsilon={eps_tex(eps)}$")]):
        tot = _pick_typical(R, model, N, e)
        bbox = an.bounding_box(tot); L = max(bbox[1] - bbox[0], bbox[3] - bbox[2])
        padded = an.crop_and_pad(tot, bbox); lmax = an.l_max_for_extent(L)
        for col, l in enumerate([1, max(2, lmax // 4), lmax]):
            ax = axes[r, col]; n = padded.shape[0] // l
            blocks = padded.reshape(n, l, n, l).any(axis=(1, 3))
            for (i, j) in zip(*np.where(blocks)):
                ax.add_patch(Rectangle((j * l, i * l), l, l, color="k", lw=0))
            for x in range(0, L + 1, l):
                ax.axvline(x, color="0.75", lw=0.3 if l == 1 else 0.6)
                ax.axhline(x, color="0.75", lw=0.3 if l == 1 else 0.6)
            ax.set_xlim(0, L); ax.set_ylim(L, 0); ax.set_aspect("equal"); ax.grid(False)
            ax.set_title(rf"$\ell={l}$,  $N(\ell)={int(blocks.sum())}$", fontsize=10)
            ax.tick_params(labelsize=7)
        axes[r, 0].set_ylabel(name)
        sizes = an.BOX_SIZES_ALL[an.BOX_SIZES_ALL <= lmax]
        ct = an.box_counts(padded, sizes); D = an.fit_dimension(sizes, ct)
        a = axes[r, 3]
        a.loglog(sizes, ct, "o", mfc="none", mec=c, ms=8)
        p = np.polyfit(np.log(sizes), np.log(ct), 1)
        xx = np.array([sizes[0] * 0.8, sizes[-1] * 1.25])
        a.loglog(xx, np.exp(np.polyval(p, np.log(xx))), "k--", lw=1.2, label=rf"$D^t={D:.2f}$")
        a.set_xlabel(r"$\ell$ (lattice units)"); a.set_ylabel(r"$N(\ell)$"); a.legend()
        a.set_title(rf"$A^t={int(tot.sum())}$, $L={L}$", fontsize=10)
    fig.tight_layout()
    save(fig, out, "fig_boxcounting_example")


def fig_D_distributions(R, out, N=None, eps=REF_EPS):
    N = N or REF_N
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.0))
    for ax, which, lab in [(axes[0], "t", r"$D^t$"), (axes[1], "P", r"$D^P$")]:
        bins = np.linspace(0.0, 2.05, 63)
        for model, e, mode, c, ls, name in [
                ("LH91", 0.0, "whole", C_LH, "-", "LH91"),
                ("DET", eps, "whole", C_DET, "-", rf"Det. $\epsilon={eps_tex(eps)}$, whole avalanche"),
                ("DET", eps, "cluster", C_DET, "--", rf"Det. $\epsilon={eps_tex(eps)}$, largest cluster")]:
            d = R.D(model, N, e, which, mode); m, s, n = mstd(d)
            ax.hist(d, bins=bins, density=True, histtype="step", lw=1.6, color=c, ls=ls,
                    label=rf"{name}: ${m:.2f}\pm{s:.2f}$")
        ax.set_yscale("log"); ax.set_ylim(2e-3, 40)
        ax.set_xlabel(lab); ax.set_ylabel(f"PDF({lab})"); ax.legend(loc="upper left", fontsize=8)
    axes[0].set_title(rf"(a) total area, N={N}", fontsize=10)
    axes[1].set_title(rf"(b) peak area, N={N}", fontsize=10)
    fig.tight_layout()
    save(fig, out, "fig_D_distributions")


def fig_D_summary(R, out, sizes):
    """Top (large): <D^t> vs eps for all N (whole: filled, cluster: open), LH91 bands.
    Bottom-left: <D^t> vs box-size window delta.  Bottom-right: <D^P> vs eps."""
    fig = plt.figure(figsize=(10, 9.2))
    gs = GridSpec(2, 2, height_ratios=[1.25, 1], hspace=0.32, wspace=0.28)
    axT = fig.add_subplot(gs[0, :]); axL = fig.add_subplot(gs[1, 0]); axR = fig.add_subplot(gs[1, 1])

    def vs_eps(ax, which):
        for k, N in enumerate(sizes):
            off = 1 + 0.07 * (k - (len(sizes) - 1) / 2)
            for mode, filled, ls in [("whole", True, "-"), ("cluster", False, "--")]:
                m = [mstd(R.D("DET", N, e, which, mode))[0] for e in EPSILONS]
                s = [mstd(R.D("DET", N, e, which, mode))[1] for e in EPSILONS]
                ax.errorbar(np.array(EPSILONS) * off * (0.97 if filled else 1.03), m, yerr=s,
                            fmt=MK_N[N] + ls, color=C_DET, mfc=C_DET if filled else "white",
                            ms=7, capsize=2, lw=1,
                            label=f"Det. {'whole' if filled else 'cluster'}, N={N}")
            ml, sl, _ = mstd(R.D("LH91", N, 0.0, which, "whole"))
            ax.axhline(ml, color=C_LH, lw=1 + 0.6 * k, ls=[":", "-.", "-"][k % 3],
                       label=rf"LH91 N={N}: ${ml:.2f}\pm{sl:.2f}$")
        ax.set_xscale("log"); ax.set_xlabel(r"$\epsilon$")

    vs_eps(axT, "t"); axT.set_ylabel(r"$\langle D^t\rangle$")
    axT.set_title(r"(a) fractal dimension of the total area vs driving rate", fontsize=11)
    axT.legend(ncol=3, fontsize=7.5, loc="lower left")
    vs_eps(axR, "P"); axR.set_ylabel(r"$\langle D^P\rangle$")
    axR.set_title(r"(c) peak area vs driving rate", fontsize=11)
    h, l = axR.get_legend_handles_labels()
    # bottom-left: dependence on the box-size window
    for k, N in enumerate(sizes):
        lmaxs = [l_ for l_ in (2, 4, 8, 16) if l_ <= N // 16]
        for model, e, mode, c, filled in [("LH91", 0.0, "whole", C_LH, True),
                                          ("DET", REF_EPS, "whole", C_DET, True),
                                          ("DET", REF_EPS, "cluster", C_DET, False)]:
            cts = R.counts(model, N, e, mode)
            m, s = [], []
            for lm in lmaxs:
                sel = an.BOX_SIZES_ALL <= lm
                d = np.array([an.fit_dimension(an.BOX_SIZES_ALL[sel], c_[sel]) for c_ in cts])
                m.append(np.nanmean(d)); s.append(np.nanstd(d))
            off = {"LH91": 0.97, "whole": 1.0, "cluster": 1.03}["LH91" if model == "LH91" else mode]
            axL.errorbar(np.array(lmaxs) / N * off, m, yerr=s, fmt=MK_N[N] + "-", color=c,
                         mfc=c if filled else "white", ms=6, capsize=2, lw=0.8)
    axL.set_xscale("log", base=2); axL.set_xlabel(r"$\delta=\ell_{\max}/N$")
    axL.set_ylabel(r"$\langle D^t\rangle$")
    axL.set_title(rf"(b) box-size window ($\epsilon={eps_tex(REF_EPS)}$)", fontsize=11)
    from matplotlib.lines import Line2D
    hl = [Line2D([], [], color=C_LH, marker="s", ls="", label="LH91"),
          Line2D([], [], color=C_DET, marker="s", ls="", label="Det. whole"),
          Line2D([], [], color=C_DET, marker="s", mfc="white", ls="", label="Det. cluster")] + \
         [Line2D([], [], color="0.3", marker=MK_N[N], ls="", label=f"N={N}") for N in sizes]
    axL.legend(handles=hl, fontsize=7.5, ncol=2, loc="lower left")
    save(fig, out, "fig_D_summary")


def fig_stationarity(R, out, sizes):
    fig, axes = plt.subplots(1, len(sizes), figsize=(4.2 * len(sizes), 3.4))
    for ax, N in zip(np.atleast_1d(axes), sizes):
        for model, e, c in [("LH91", 0.0, C_LH)] + [("DET", e_, C_EPS[e_]) for e_ in EPSILONS]:
            if not R.has(model, N, e):
                continue
            en = np.load(os.path.join(R._dir(model, N, e), "energy.npy"))
            meta = R.meta(model, N, e)
            ax.plot(np.arange(len(en)) * meta["chunk"], en / en[-len(en) // 2:].mean(), color=c, lw=0.7,
                    label="LH91" if model == "LH91" else rf"Det. $\epsilon={eps_tex(e)}$")
            ax.axvline(meta["burn_in_chunks"] * meta["chunk"], color="k", ls=":", lw=1)
        ax.set_xlabel("iteration"); ax.set_ylabel(r"$\sum B^2$ / stationary mean"); ax.set_title(f"N={N}")
        ax.legend(fontsize=7)
    fig.tight_layout()
    save(fig, out, "fig_stationarity_check")


# ----------------------------------------------------------------------------- #
# tables and numbers
# ----------------------------------------------------------------------------- #
def fmt(v, e):
    return rf"${v:.2f}\pm{max(e, 0.005):.2f}$"


def up2(e):
    """uncertainty rounded UP to two decimals (never quoted as 0.00)."""
    return max(np.ceil(e * 100 - 1e-9) / 100, 0.01)


def total_err(f):
    """single uncertainty quoted in the paper: statistical (batch means) and fitting-range
    (systematic) uncertainties added in quadrature; in practice it equals the latter."""
    return float(np.hypot(f["err"], f["err_sys"]))


def fmt_sys(f):
    return rf"${f['alpha']:.2f}\pm{up2(total_err(f)):.2f}$"


def write_tables(R, fits, out, sizes):
    numbers = {"area_definition": R.area, "fits": {}, "fractal": {}}
    # Table 1: exponents
    L = [r"\begin{tabular}{|c|l|c|c|c|c|}", r"\hline",
         r"\rowcolor[HTML]{C0C0C0}",
         r"$N$ & Model & $\epsilon$ & $\alpha_{A^{t}}$ & $\alpha_{A^{P}}$ & $\alpha_{E}$ \\ \hline"]
    for N in sizes:
        for model, e in [("LH91", 0.0)] + [("DET", e_) for e_ in EPSILONS]:
            if (model, N, e) not in fits:
                continue
            f = fits[(model, N, e)]
            L.append(f"{N} & {'LH91' if model == 'LH91' else 'Det.'} & "
                     f"{'--' if model == 'LH91' else '$' + eps_tex(e) + '$'} & "
                     f"{fmt_sys(f['A_t'])} & {fmt_sys(f['A_P'])} & "
                     f"{fmt_sys(f['E'])} \\\\")
            numbers["fits"][config_tag(model, N, e)] = {
                k: {kk: (float(vv) if vv is not None else None) for kk, vv in f[k].items()}
                for k in ("A_t", "A_P", "E")}
            numbers["fits"][config_tag(model, N, e)]["n_avalanches"] = f["n_avalanches"]
        L.append(r"\hline")
    L.append(r"\end{tabular}")
    open(os.path.join(out, "table_exponents.tex"), "w").write("\n".join(L) + "\n")
    # Table 2: fractal dimensions
    L = [r"\begin{tabular}{|c|l|c|c|c|c|c|}", r"\hline", r"\rowcolor[HTML]{C0C0C0}",
         r"$N$ & Model & $\epsilon$ & $D^{t}$ (whole) & $D^{t}$ (cluster) & $D^{P}$ (whole) & $D^{P}$ (cluster) \\ \hline"]
    for N in sizes:
        for model, e in [("LH91", 0.0)] + [("DET", e_) for e_ in EPSILONS]:
            if not R.has(model, N, e):
                continue
            vals = {}
            for which in ("t", "P"):
                for mode in ("whole", "cluster"):
                    vals[(which, mode)] = mstd(R.D(model, N, e, which, mode))
            fr = R.fr(model, N, e)
            ncl = fr[f"ncl_{R.area}"]; selw = np.isfinite(fr[f"Dt_{R.area}_whole"])
            L.append(f"{N} & {'LH91' if model == 'LH91' else 'Det.'} & "
                     f"{'--' if model == 'LH91' else '$' + eps_tex(e) + '$'} & " +
                     " & ".join(fmt(vals[(w, m)][0], vals[(w, m)][1])
                                for w, m in [("t", "whole"), ("t", "cluster"), ("P", "whole"), ("P", "cluster")])
                     + r" \\")
            numbers["fractal"][config_tag(model, N, e)] = {
                f"D{w}_{m}": {"mean": float(vals[(w, m)][0]), "std": float(vals[(w, m)][1]),
                              "n": int(vals[(w, m)][2]),
                              "sem": float(vals[(w, m)][1] / np.sqrt(max(vals[(w, m)][2], 1)))}
                for w in ("t", "P") for m in ("whole", "cluster")}
            numbers["fractal"][config_tag(model, N, e)]["mean_n_clusters_selected"] = \
                float(np.nanmean(ncl[selw])) if selw.any() else None
            numbers["fractal"][config_tag(model, N, e)]["frac_multi_island_selected"] = \
                float(np.nanmean(ncl[selw] > 1)) if selw.any() else None
            # delta dependence
            dd = {}
            for mode in ("whole", "cluster"):
                cts = R.counts(model, N, e, mode)
                for lm in (2, 4, 8, 16):
                    if lm > N // 16:
                        continue
                    sel = an.BOX_SIZES_ALL <= lm
                    d = np.array([an.fit_dimension(an.BOX_SIZES_ALL[sel], c_[sel]) for c_ in cts])
                    dd[f"{mode}_lmax{lm}"] = float(np.nanmean(d))
            numbers["fractal"][config_tag(model, N, e)]["Dt_vs_lmax"] = dd
            numbers["fractal"][config_tag(model, N, e)]["meta"] = R.meta(model, N, e)
        L.append(r"\hline")
    L.append(r"\end{tabular}")
    open(os.path.join(out, "table_fractal.tex"), "w").write("\n".join(L) + "\n")
    json.dump(numbers, open(os.path.join(out, "numbers.json"), "w"), indent=1)
    print("  saved tables + numbers.json")


def main():
    global REF_N
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=os.path.join(HERE, "results"))
    ap.add_argument("--area", choices=["U", "M"], default="M")
    ap.add_argument("--lh-results", default=None, help="folder with the LH91 runs (default: --results)")
    ap.add_argument("--out", default=None, help="output folder (default <results>/paper_<area>)")
    ap.add_argument("--sizes", type=int, nargs="+", default=SIZES)
    ap.add_argument("--boot", type=int, default=100, help="(unused since v4: errors from uncertainties.py; kept for compatibility)")
    ap.add_argument("--ref-n", type=int, default=REF_N, help="lattice used in the single-N figures")
    a = ap.parse_args()
    REF_N = a.ref_n
    for label, folder in (("results", a.results), ("LH91 results", a.lh_results)):
        if folder is not None and not os.path.isdir(folder):
            print(f"[make_paper_outputs] The {label} folder does not exist:\n    {os.path.abspath(folder)}\n"
                  f"  Run 'python run_simulations.py' first (or pass --results / --lh-results). "
                  f"Nothing was generated.", file=sys.stderr)
            sys.exit(1)
    R = Results(a.results, a.area, a.lh_results)
    sizes = [N for N in a.sizes if R.has("LH91", N)]
    if not sizes:
        print(f"[make_paper_outputs] No finished LH91 runs (LH91_N*/DONE.json) for sizes {a.sizes} in\n"
              f"    {os.path.abspath(R.lh_folder)}\n"
              f"  Run 'python run_simulations.py' first. Nothing was generated.", file=sys.stderr)
        sys.exit(1)
    out = a.out or os.path.join(a.results, f"paper_{a.area}"); os.makedirs(out, exist_ok=True)
    print("sizes available:", sizes)
    an.validate_box_counting(); an.validate_mle()
    fits = all_fits(R, sizes, a.boot)
    write_tables(R, fits, out, sizes)
    fig_stationarity(R, out, sizes)
    fig_E(R, fits, out)
    fig_area_pdfs(R, fits, out, "A_t")
    fig_area_pdfs(R, fits, out, "A_P")
    fig_local_slope(R, out, sizes)
    nj = os.path.join(out, "numbers.json"); numbers = json.load(open(nj))
    numbers["absolute_windows"] = absolute_window_fits(R, sizes)
    json.dump(numbers, open(nj, "w"), indent=1)
    fig_avalanche_examples(R, out)
    fig_boxcounting(R, out)
    fig_D_distributions(R, out)
    fig_D_summary(R, out, sizes)


if __name__ == "__main__":
    main()
