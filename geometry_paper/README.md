# geometry_paper — code for "Geometrical Properties of a Deterministically Driven Avalanche Model of Solar Flares"

Self-contained pipeline that produces every number, table and figure of the paper.
It does **not** modify `libs/` or `main_simulation.py` (used by previous papers), and it
only includes the two models compared in the paper: LH91 and the deterministically
driven model of Strugarek et al. (2014, model NC0).

It lives inside the `phd` repository and reads the LH91 initial critical states from
`phd/initial_grids/` (the scripts print a notice and stop if that folder is missing).
Results are written to `phd/geometry_paper/results/` (git-ignored, ~600 MB); the final
figures and tables of the paper are kept in `results_final/`.

```
phd/
  initial_grids/           LH91 critical states B_N{32,62,128,256}_Zc1.npz (shared with libs/)
  geometry_paper/
    models.py              numba kernels (LH91, deterministic NC0), float64, on-line avalanche processing
    analysis.py            box counting, MLE power-law fits (Clauset et al. 2009), log-binned PDFs
    run_simulations.py     runs all (model, N, eps) configurations in parallel, chunked + resumable
    make_paper_outputs.py  figures (PDF/PNG), LaTeX tables and numbers.json from the results
    tests.py               validation tests (box counting, MLE, energy conservation, resume)
    CODE_REVIEW.md         issues found in the legacy code and their impact
```

## Quick start

```bash
pip install numpy scipy numba matplotlib
cd phd/geometry_paper
python tests.py                                  # ~1 min, all checks must pass
python run_simulations.py                        # all runs (see timing below)
python make_paper_outputs.py                     # figures + tables -> results/paper_M/
```

Useful options:

```bash
python run_simulations.py --sizes 64 128 --workers 4      # subset / fewer processes
python run_simulations.py --iters 5e6                     # shorter measurement
python run_simulations.py --r0-per-iteration --out results_r0iter   # legacy r0 variant
python make_paper_outputs.py --area U                     # areas = unstable nodes only
```

**Resuming.** Every configuration writes a checkpoint every 50 chunks (5e5 iterations).
If the run is interrupted (Ctrl-C, reboot), launch the same command again: finished
configurations are skipped and unfinished ones resume from their checkpoint. Chunk *k*
uses the random seed `SEED*1000003 + k`, so an interrupted and an uninterrupted run give
*identical* results (checked in `tests.py`). Do not run two instances on the same output
folder at the same time.

**Memory.** O(N^2) for the lattice (the lattice history is not stored); the per-avalanche
tables grow with the number of avalanches (largest run: ~1e6 avalanches, ~100 MB).

**Timing** (measured for the paper runs: 2e6 burn-in + 2e7 measured iterations per
configuration, 9 processes in parallel on a 10-core laptop): N=64 ≈ 3–5 min, N=128 ≈ 10–13 min,
N=256 ≈ 50–55 min per configuration. All 12 configurations finish in about one hour.

## Models (as in the paper, Section 2)

* Lattice of N×N interior nodes, B = 0 on the boundary, Zc = 1.
* Instability: |ΔB_k| > Zc with ΔB_k = B_k − ¼ Σ_nn B_nn; synchronous update; driving is
  suspended during avalanches.
* **LH91**: increment δB ~ U[−0.2, 0.8] at one random interior node per stable iteration;
  B_k → B_k − 4/5 Z, B_nn → B_nn + 1/5 Z, Z = Zc sgn(ΔB_k).
* **Deterministic (NC0)**: B → B(1+ε) on all interior nodes per stable iteration;
  B_nn → B_nn + r0/5 Z with r0 ~ U[D_nc, 1] drawn independently for every unstable node
  (D_nc = 0.1). Energy release per unstable node: Eq. (17) of Strugarek et al. (2014).
* Energies are in units of e0 = 4 Zc²/5 for both models.

## Avalanche observables

* `A_t_M` (A^t in the paper): number of distinct nodes involved at least once; a node is
  involved in an iteration if it is unstable or a nearest neighbour of an unstable node
  (all nodes whose value changes). `A_t` counts unstable nodes only (robustness check).
* `A_P_M` (A^P): maximum number of nodes involved in a single iteration.
* `E`, `P`, `T`: total energy, peak energy per iteration (units of e0), duration.
* Fractal dimensions (`fractal.npz`) for avalanches with linear extent L ≥ N/4: box counting
  with boxes ℓ = 1, 2, 4, …, ℓ_max ≤ L/4 on the bounding box of the avalanche, for the whole
  avalanche and for its largest 8-connected cluster; D^P uses the iteration with the largest
  number of involved nodes, counted on the same grid.

Reference for the deterministic model: Strugarek, Charbonneau, Joseph & Pirot (2014), Sol. Phys. 289, 2993, arXiv:1402.4730 — Eqs. 12–13 (non-conservative redistribution, p. 7–8), Eq. 17 (energy release of NC0, p. 19), and Sect. 3.2 (p. 16: 'a random fraction ... being lost from each unstable node').
