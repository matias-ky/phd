# -*- coding: utf-8 -*-
"""
models.py -- cellular-automaton kernels for the paper
"Geometrical Properties of a Deterministically Driven Avalanche Model of Solar Flares".

Two models (Strugarek et al. 2014, SoPh 289, 2993; Strugarek & Charbonneau 2014):
  * LH91  : Lu & Hamilton (1991) model, stochastic localized driving.
  * DET   : deterministic global driving  A -> A (1+eps)  with the non-conservative
            redistribution rule of model NC0 (Strugarek et al. 2014, Eqs. 12-13):
                A_ij  -> A_ij - 4/5 Z ,  A_nn -> A_nn + r0/5 Z ,  r0 ~ U[D_nc, 1].

Differences with libs/functions.py (kept untouched):
  * float64 everywhere (float32 cannot resolve the eps=1e-6 multiplicative driving
    accurately: eps*B is only 8-17 float32 ulps).
  * Energy release normalised by e0 = 4 Zc^2 / 5 in BOTH models; the DET energy is
    Eq. (17) of Strugarek et al. (2014).  (libs/functions.py adds the un-normalised
    LH term to the normalised extra terms and divides everything by e0 again, which
    over-estimates the DET energy by a factor ~1/e0 = 1.25.)
  * LH91 random node drawn in 1..N (random.randint is inclusive in numba; the legacy
    randint(1, N+1) can hit the boundary).
  * Avalanches are processed on-line: no list of lattice states is kept, so memory
    does not grow with the chunk length (N=256 is affordable on a laptop).
  * Z = Zc * sign(Delta B) as in Strugarek et al. (2014), Eq. (2)-(3).
  * r0 can be drawn once per iteration (legacy behaviour) or once per unstable node.

Per avalanche the kernel returns:
  E (total energy / e0), P (peak energy per iteration / e0), T (iterations),
  A_t (# distinct nodes that were unstable at least once),
  A_P (max # of nodes simultaneously unstable in one iteration),
  A_t_M, A_P_M : same quantities counting also the 4 nearest neighbours of every
                 unstable node (definition used by libs/areas.py),
  L  (linear extent of the bounding box of the unstable nodes),
  L_M (same for the neighbour-dilated set),
  t0 (starting iteration inside the chunk),
  A_P_at_Epeak (# unstable nodes at the iteration of maximum energy release).
For avalanches with max(L, L_M) >= min_extent the binary masks (cropped to the
bounding box of the dilated set) are exported for fractal analysis.
"""
import numpy as np
import random
from numba import njit
from numba.typed import List

LH91 = 0
DET = 1
AV_FIELDS = ["E", "P", "T", "A_t", "A_P", "A_t_M", "A_P_M", "L", "L_M", "t0", "A_P_at_Epeak"]


@njit(cache=True)
def seed_numba(seed):
    np.random.seed(seed)
    random.seed(seed)


@njit(cache=True)
def lattice_energy(B):
    return np.sum(B * B)


@njit(cache=True)
def simulate_chunk(B, model, n_iter, eps, D_nc, Zc, r0_per_node, min_extent):
    """Advance the lattice B ((N+2)x(N+2), float64, zero boundaries) in place.

    Runs at least n_iter iterations and stops at the first quiet iteration after that
    (so no avalanche is split between chunks).
    """
    Np2 = B.shape[0]
    N = Np2 - 2
    C = np.zeros((Np2, Np2))
    st_U = np.zeros((Np2, Np2), np.int64)     # == av_id : unstable at least once
    st_M = np.zeros((Np2, Np2), np.int64)     # == av_id : unstable or neighbour
    st_Mit = np.zeros((Np2, Np2), np.int64)   # == it    : dilated set, this iteration
    cu_i = np.empty(N * N, np.int64); cu_j = np.empty(N * N, np.int64)
    cm_i = np.empty(N * N, np.int64); cm_j = np.empty(N * N, np.int64)
    pu_i = np.empty(N * N, np.int64); pu_j = np.empty(N * N, np.int64)
    pm_i = np.empty(N * N, np.int64); pm_j = np.empty(N * N, np.int64)
    fu_i = np.empty(N * N, np.int64); fu_j = np.empty(N * N, np.int64)
    au_i = np.empty(N * N, np.int64); au_j = np.empty(N * N, np.int64)
    am_i = np.empty(N * N, np.int64); am_j = np.empty(N * N, np.int64)

    out = List()            # per-avalanche rows (length 11)
    masks = List()          # uint8 (4, h, w): Utot, Upeak, Mtot, Mpeak
    mask_first = List()     # uint8 (h, w): unstable nodes of the first iteration
    mask_idx = List()       # avalanche index of each exported mask

    av_id = 0
    in_av = False
    it = 0
    n_av = 0
    E = 0.0; P = 0.0; T = 0; At = 0; AtM = 0; AP = 0; APM = 0; AP_Ep = 0; t0 = 0
    imin = 0; imax = 0; jmin = 0; jmax = 0
    mimin = 0; mimax = 0; mjmin = 0; mjmax = 0
    nfu = 0; npu = 0; npm = 0
    keep_going = True
    while keep_going:
        it += 1
        e_it = 0.0
        nu = 0
        r0_it = 1.0
        if model == DET:
            r0_it = random.uniform(D_nc, 1.0)
        for i in range(1, N + 1):
            for j in range(1, N + 1):
                sumnn = B[i + 1, j] + B[i - 1, j] + B[i, j + 1] + B[i, j - 1]
                dA = B[i, j] - 0.25 * sumnn
                if abs(dA) > Zc:
                    s = 1.0 if dA > 0 else -1.0
                    Z = Zc * s
                    r0 = 1.0
                    if model == DET:
                        r0 = random.uniform(D_nc, 1.0) if r0_per_node else r0_it
                    C[i, j] -= 0.8 * Z
                    C[i + 1, j] += 0.2 * r0 * Z
                    C[i - 1, j] += 0.2 * r0 * Z
                    C[i, j + 1] += 0.2 * r0 * Z
                    C[i, j - 1] += 0.2 * r0 * Z
                    # energy released / e0  (Strugarek et al. 2014, Eqs. 4 and 17)
                    e_it += 2.0 * s * dA / Zc - (4.0 + r0 * r0) / 5.0 \
                        + (1.0 - r0) * s * sumnn / (2.0 * Zc)
                    cu_i[nu] = i; cu_j[nu] = j; nu += 1

        if nu > 0:
            if not in_av:
                in_av = True
                av_id += 1
                E = 0.0; P = 0.0; T = 0; At = 0; AtM = 0; AP = 0; APM = 0; AP_Ep = 0
                t0 = it
                imin = N + 2; imax = -1; jmin = N + 2; jmax = -1
                mimin = N + 2; mimax = -1; mjmin = N + 2; mjmax = -1
                nfu = nu
                for k in range(nu):
                    fu_i[k] = cu_i[k]; fu_j[k] = cu_j[k]
                npu = 0; npm = 0
            T += 1
            E += e_it
            if e_it > P:
                P = e_it
                AP_Ep = nu
            for k in range(nu):
                i = cu_i[k]; j = cu_j[k]
                if st_U[i, j] != av_id:
                    st_U[i, j] = av_id
                    au_i[At] = i; au_j[At] = j
                    At += 1
                    if i < imin: imin = i
                    if i > imax: imax = i
                    if j < jmin: jmin = j
                    if j > jmax: jmax = j
            nm = 0
            for k in range(nu):
                ci = cu_i[k]; cj = cu_j[k]
                for d in range(5):
                    i = ci; j = cj
                    if d == 1: i = ci + 1
                    elif d == 2: i = ci - 1
                    elif d == 3: j = cj + 1
                    elif d == 4: j = cj - 1
                    if i < 1 or i > N or j < 1 or j > N:
                        continue
                    if st_Mit[i, j] != it:
                        st_Mit[i, j] = it
                        cm_i[nm] = i; cm_j[nm] = j; nm += 1
                        if st_M[i, j] != av_id:
                            st_M[i, j] = av_id
                            am_i[AtM] = i; am_j[AtM] = j
                            AtM += 1
                            if i < mimin: mimin = i
                            if i > mimax: mimax = i
                            if j < mjmin: mjmin = j
                            if j > mjmax: mjmax = j
            if nu > AP:
                AP = nu; npu = nu
                for k in range(nu):
                    pu_i[k] = cu_i[k]; pu_j[k] = cu_j[k]
            if nm > APM:
                APM = nm; npm = nm
                for k in range(nm):
                    pm_i[k] = cm_i[k]; pm_j[k] = cm_j[k]
            for i in range(1, N + 1):
                for j in range(1, N + 1):
                    B[i, j] += C[i, j]
                    C[i, j] = 0.0
            # neighbours on the boundary receive nothing (A = 0 there)
            for k in range(Np2):
                C[0, k] = 0.0; C[N + 1, k] = 0.0; C[k, 0] = 0.0; C[k, N + 1] = 0.0
        else:
            if in_av:
                in_av = False
                L = max(imax - imin + 1, jmax - jmin + 1)
                LM = max(mimax - mimin + 1, mjmax - mjmin + 1)
                row = np.empty(11)
                row[0] = E; row[1] = P; row[2] = T; row[3] = At; row[4] = AP
                row[5] = AtM; row[6] = APM; row[7] = L; row[8] = LM; row[9] = t0
                row[10] = AP_Ep
                out.append(row)
                if L >= min_extent or LM >= min_extent:
                    h = mimax - mimin + 1; w = mjmax - mjmin + 1
                    m = np.zeros((4, h, w), np.uint8)
                    for k in range(At):
                        m[0, au_i[k] - mimin, au_j[k] - mjmin] = 1
                    for k in range(npu):
                        m[1, pu_i[k] - mimin, pu_j[k] - mjmin] = 1
                    for k in range(AtM):
                        m[2, am_i[k] - mimin, am_j[k] - mjmin] = 1
                    for k in range(npm):
                        m[3, pm_i[k] - mimin, pm_j[k] - mjmin] = 1
                    f = np.zeros((h, w), np.uint8)
                    for k in range(nfu):
                        f[fu_i[k] - mimin, fu_j[k] - mjmin] = 1
                    masks.append(m); mask_first.append(f); mask_idx.append(n_av)
                n_av += 1
            if it > n_iter:
                keep_going = False
            if model == LH91:
                i = random.randint(1, N)
                j = random.randint(1, N)
                B[i, j] += random.random() - 0.2
            else:
                for i in range(1, N + 1):
                    for j in range(1, N + 1):
                        B[i, j] *= (1.0 + eps)

    res = np.empty((n_av, 11))
    for k in range(n_av):
        res[k, :] = out[k]
    return res, masks, mask_first, mask_idx, it
