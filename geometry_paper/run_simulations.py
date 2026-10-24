# -*- coding: utf-8 -*-
"""
run_simulations.py -- produce all the data used in the paper (LH91 vs deterministic).

    python run_simulations.py                       # all configurations (see CONFIGS)
    python run_simulations.py --sizes 64 128        # subset of lattice sizes
    python run_simulations.py --iters 5e6           # shorter measurement
    python run_simulations.py --workers 4           # limit parallel processes

Runs are chunked (CHUNK iterations each) and checkpointed every CHECKPOINT_EVERY
chunks: if the run is interrupted, simply launch the same command again and it
resumes from the last checkpoint.  The random seed of chunk k is SEED*1000003 + k,
so the result is independent of interruptions.

Memory is O(N^2): no lattice history is stored (see models.py).

Output (one folder per configuration in OUT_DIR/<tag>/):
  avalanches.npz : per-avalanche E, P, T, A_t, A_P, A_t_M, A_P_M, L, L_M, ...
  fractal.npz    : per selected avalanche (L >= N/4) the box-counting dimensions
  examples.npz   : a few masks used for the illustration figures
  energy.npy     : lattice energy sum(B^2) at the end of every chunk (stationarity)
"""
import os
import sys
import time
import json
import argparse
import multiprocessing as mp
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

# ----------------------------------------------------------------------------- #
# CONFIGURATION
# ----------------------------------------------------------------------------- #
SEED = 7
ZC = 1.0
D_NC = 0.1                    # non-conservation parameter (Strugarek et al. 2014, NC0)
R0_PER_NODE = True            # True: one r0 per unstable node (Strugarek et al. 2014);
                              # False: one r0 per iteration (legacy libs/functions.py)
SIZES = [64, 128, 256]
EPSILONS = [1e-6, 1e-5, 1e-4]
CHUNK = 10_000                # iterations per chunk
BURN_IN = 1_000_000           # iterations discarded (transient), all models
MEASURE = 20_000_000          # iterations measured after the burn-in
CHECKPOINT_EVERY = 50         # chunks
MIN_EXTENT_FRACTION = 0.25    # fractal analysis only for L >= N/4
N_EXAMPLES = 30
OUT_DIR = os.path.join(HERE, "results")
# geometry_paper/ lives inside the phd repository: phd/geometry_paper/ -> phd/initial_grids/
PHD_DIR = os.path.dirname(HERE)
GRID_DIR = os.path.join(PHD_DIR, "initial_grids")
REQUIRED_GRIDS = ("B_N62_Zc1.npz",)   # always needed (N = 64 is interpolated from it)


def check_grid_dir(grid_dir=None):
    """True if the initial_grids folder exists with the required files; prints a notice otherwise."""
    grid_dir = grid_dir or GRID_DIR
    if not os.path.isdir(grid_dir):
        print(f"[run_simulations] The folder with the initial grids does not exist:\n    {grid_dir}\n"
              f"  It must be phd/initial_grids/ (geometry_paper/ must be inside phd/). "
              f"Nothing was run.", file=sys.stderr)
        return False
    missing = [f for f in REQUIRED_GRIDS if not os.path.exists(os.path.join(grid_dir, f))]
    if missing:
        print(f"[run_simulations] Missing initial grid(s) in {grid_dir}: {', '.join(missing)}. "
              f"Nothing was run.", file=sys.stderr)
        return False
    return True


def initial_grid(N):
    """LH91 critical state with N x N interior nodes (zero boundaries), float64."""
    f = os.path.join(GRID_DIR, f"B_N{N}_Zc1.npz")
    if os.path.exists(f):
        B = np.load(f)["B"].astype(np.float64)
        assert B.shape == (N + 2, N + 2)
        return B
    # N = 64: interpolate the 62x62 critical state and rescale (B_max ~ (N+1)^2),
    # the LH91 burn-in then relaxes it to the stationary state.
    from scipy.ndimage import zoom
    src = np.load(os.path.join(GRID_DIR, "B_N62_Zc1.npz"))["B"].astype(np.float64)
    n0 = src.shape[0] - 2
    inner = zoom(src, (N + 2) / (n0 + 2), order=1)
    B = np.zeros((N + 2, N + 2))
    B[1:-1, 1:-1] = inner[1:-1, 1:-1] * ((N + 1) / (n0 + 1)) ** 2
    return B


def config_tag(model, N, eps):
    return f"LH91_N{N}" if model == "LH91" else f"DET_N{N}_eps{eps:.0e}"


def all_configs(sizes):
    cfg = []
    for N in sizes:
        cfg.append(("LH91", N, 0.0))
        for e in EPSILONS:
            cfg.append(("DET", N, e))
    # biggest lattices first (better load balance)
    return sorted(cfg, key=lambda c: -c[1])


# ----------------------------------------------------------------------------- #
# per-avalanche fractal analysis
# ----------------------------------------------------------------------------- #
FR_FIELDS = []
for _d in ("U", "M"):                      # U: unstable nodes, M: + nearest neighbours
    for _w in ("whole", "cluster"):
        FR_FIELDS += [f"Dt_{_d}_{_w}", f"DP_{_d}_{_w}", f"L_{_d}_{_w}", f"A_{_d}_{_w}"]
    FR_FIELDS += [f"ncl_{_d}"]


def analyse_masks(m4, N, an):
    """m4: uint8 (4,h,w) = Utot, Upeak, Mtot, Mpeak. Returns (row, counts dict)."""
    min_ext = MIN_EXTENT_FRACTION * N
    row = {k: np.nan for k in FR_FIELDS}
    counts = {}
    for d, (it, ip) in (("U", (0, 1)), ("M", (2, 3))):
        tot = m4[it].astype(bool); pk = m4[ip].astype(bool)
        if not tot.any():
            continue
        L = an.extent(tot)
        if L >= min_ext:
            dt, dp, L, ct, cp = an.fractal_pair(tot, pk)
            row.update({f"Dt_{d}_whole": dt, f"DP_{d}_whole": dp,
                        f"L_{d}_whole": L, f"A_{d}_whole": tot.sum()})
            counts[f"{d}_whole"] = ct
        cl, ncl = an.largest_cluster(tot)
        row[f"ncl_{d}"] = ncl
        Lc = an.extent(cl)
        if Lc >= min_ext:
            dt, dp, Lc, ct, cp = an.fractal_pair(cl, pk & cl)
            row.update({f"Dt_{d}_cluster": dt, f"DP_{d}_cluster": dp,
                        f"L_{d}_cluster": Lc, f"A_{d}_cluster": cl.sum()})
            counts[f"{d}_cluster"] = ct
    return row, counts


# ----------------------------------------------------------------------------- #
# worker
# ----------------------------------------------------------------------------- #
def run_config(args):
    model_name, N, eps, burn_in, measure, seed = args
    import models as md
    import analysis as an
    tag = config_tag(model_name, N, eps)
    folder = os.path.join(OUT_DIR, tag)
    os.makedirs(folder, exist_ok=True)
    done_flag = os.path.join(folder, "DONE.json")
    if os.path.exists(done_flag):
        print(f"[{tag}] already done"); return tag
    model = md.LH91 if model_name == "LH91" else md.DET
    n_burn = int(np.ceil(burn_in / CHUNK)); n_meas = int(np.ceil(measure / CHUNK))
    min_ext = int(np.ceil(MIN_EXTENT_FRACTION * N))
    ck = os.path.join(folder, "checkpoint.npz")

    # state
    if os.path.exists(ck):
        z = np.load(ck, allow_pickle=True)
        B = z["B"]; k0 = int(z["chunk"]); energy = list(z["energy"])
        av = [z["av"]] if len(z["av"]) else []
        fr = [z["fr"]] if len(z["fr"]) else []
        cnt = {k: [z[f"cnt_{k}"]] for k in ("U_whole", "U_cluster", "M_whole", "M_cluster")
               if f"cnt_{k}" in z.files and len(z[f"cnt_{k}"])}
        ex = dict(z["ex"].item()) if "ex" in z.files else {}
        n_it = int(z["n_it"])
        print(f"[{tag}] resuming at chunk {k0}")
    else:
        B = initial_grid(N); k0 = 0; energy = []; av = []; fr = []; cnt = {}; ex = {}; n_it = 0
    ex.setdefault("mid", []); ex.setdefault("multi", [])
    cnt_lists = {k: list(v) for k, v in cnt.items()}
    t0 = time.time()

    def save(k, final=False):
        avs = np.concatenate(av) if av else np.zeros((0, len(md.AV_FIELDS)))
        frs = np.concatenate(fr) if fr else np.zeros((0, len(FR_FIELDS)))
        cdict = {f"cnt_{k_}": (np.concatenate(v) if v else np.zeros((0, an.MAX_L_POW + 1)))
                 for k_, v in cnt_lists.items()}
        np.savez(ck + ".tmp.npz", B=B, chunk=k, energy=np.array(energy), av=avs, fr=frs,
                 n_it=n_it, ex=np.array(ex, dtype=object), **cdict)
        os.replace(ck + ".tmp.npz", ck)
        if final:
            np.savez_compressed(os.path.join(folder, "avalanches.npz"),
                                **{f: avs[:, i] for i, f in enumerate(md.AV_FIELDS)})
            np.savez_compressed(os.path.join(folder, "fractal.npz"),
                                **{f: frs[:, i] for i, f in enumerate(FR_FIELDS)}, **cdict)
            exs = {}
            for kind in ("mid", "multi"):
                for i, (m4, first) in enumerate(ex[kind]):
                    exs[f"{kind}_{i}_m4"] = m4; exs[f"{kind}_{i}_first"] = first
            np.savez_compressed(os.path.join(folder, "examples.npz"), **exs)
            np.save(os.path.join(folder, "energy.npy"), np.array(energy))
            meta = dict(model=model_name, N=N, eps=eps, D_nc=D_NC, r0_per_node=R0_PER_NODE,
                        Zc=ZC, burn_in_chunks=n_burn, measure_chunks=n_meas, chunk=CHUNK,
                        iterations_measured=n_it, n_avalanches=int(len(avs)),
                        n_fractal=int(len(frs)), seed=seed,
                        wall_time_s=time.time() - t0)
            json.dump(meta, open(done_flag, "w"), indent=1)

    for k in range(k0, n_burn + n_meas):
        md.seed_numba(seed * 1000003 + k)
        measuring = k >= n_burn
        res, masks, firsts, idx, it = md.simulate_chunk(
            B, model, CHUNK, eps, D_NC, ZC, R0_PER_NODE, min_ext if measuring else 10 ** 9)
        energy.append(md.lattice_energy(B))
        if measuring:
            n_it += it
            av.append(res.astype(np.float64))
            rows = []
            for m4, first in zip(masks, firsts):
                row, c = analyse_masks(m4, N, an)
                rows.append([row[f] for f in FR_FIELDS])
                for key in ("U_whole", "U_cluster", "M_whole", "M_cluster"):
                    cnt_lists.setdefault(key, []).append(
                        c.get(key, np.full(an.MAX_L_POW + 1, np.nan))[None, :])
                # examples: extent ~N/2 (box-counting figure) and multi-island DET
                L = row["L_U_whole"]
                if np.isfinite(L) and 0.4 * N <= L <= 0.6 * N and len(ex["mid"]) < N_EXAMPLES:
                    ex["mid"].append((np.array(m4), np.array(first)))
                if (model == md.DET and row["ncl_U"] >= 2 and np.isfinite(L) and L >= 0.5 * N
                        and len(ex["multi"]) < N_EXAMPLES):
                    ex["multi"].append((np.array(m4), np.array(first)))
            if rows:
                fr.append(np.array(rows, float))
        if (k + 1) % CHECKPOINT_EVERY == 0:
            save(k + 1)
            nf = sum(len(x) for x in fr)
            print(f"[{tag}] chunk {k+1}/{n_burn+n_meas} "
                  f"({'measure' if measuring else 'burn-in'}) fractal-sample={nf} "
                  f"elapsed {time.time()-t0:.0f}s", flush=True)
    save(n_burn + n_meas, final=True)
    print(f"[{tag}] DONE in {time.time()-t0:.0f}s", flush=True)
    return tag


def main():
    global OUT_DIR, R0_PER_NODE
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", type=int, nargs="+", default=SIZES)
    ap.add_argument("--iters", type=float, default=MEASURE)
    ap.add_argument("--burn-in", type=float, default=BURN_IN)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--out", default=OUT_DIR)
    ap.add_argument("--r0-per-iteration", action="store_true",
                    help="legacy variant: one r0 shared by all unstable nodes of an iteration")
    ap.add_argument("--only", nargs="*", default=None, help="tags to run, e.g. DET_N64_eps1e-06")
    a = ap.parse_args()
    if not check_grid_dir():
        sys.exit(1)
    OUT_DIR = a.out
    R0_PER_NODE = not a.r0_per_iteration
    import analysis as an
    an.validate_box_counting()
    cfgs = [(m, N, e, a.burn_in, a.iters, SEED + 100 * N + i)
            for i, (m, N, e) in enumerate(all_configs(a.sizes))]
    if a.only:
        cfgs = [c for c in cfgs if config_tag(c[0], c[1], c[2]) in a.only]
    print(f"{len(cfgs)} configurations, {a.workers} workers, burn-in {a.burn_in:.0e}, "
          f"measure {a.iters:.0e} iterations, r0_per_node={R0_PER_NODE}")
    ctx = mp.get_context("spawn")
    with ctx.Pool(a.workers, initializer=_init, initargs=(OUT_DIR, R0_PER_NODE)) as pool:
        for tag in pool.imap_unordered(run_config, cfgs):
            print("finished", tag, flush=True)


def _init(out_dir, r0pn):
    global OUT_DIR, R0_PER_NODE
    OUT_DIR = out_dir; R0_PER_NODE = r0pn


if __name__ == "__main__":
    main()
