# -*- coding: utf-8 -*-
"""tests.py -- validation of the pipeline (run before production: python tests.py)."""
import os
import sys
import shutil
import tempfile
import subprocess
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import models as md          # noqa: E402
import analysis as an        # noqa: E402


def test_box_counting():
    an.validate_box_counting()


def test_mle():
    an.validate_mle()


def _det_state(N=32, eps=1e-5, r0pn=True):
    B = np.zeros((N + 2, N + 2))
    md.seed_numba(3)
    # build a stationary LH91 state quickly, then switch to deterministic driving
    for _ in range(300):
        md.simulate_chunk(B, md.LH91, 10_000, 0.0, 0.1, 1.0, r0pn, 10 ** 9)
    for _ in range(60):
        md.simulate_chunk(B, md.DET, 10_000, eps, 0.1, 1.0, r0pn, 10 ** 9)
    return B


def test_energy_deterministic():
    """E reported by the kernel == decrease of sum(B^2)/e0 during each avalanche."""
    for r0pn in (True, False):
        eps = 1e-5
        B = _det_state(eps=eps, r0pn=r0pn)
        rel = []
        for _ in range(4000):
            S0 = md.lattice_energy(B)
            res, *_ = md.simulate_chunk(B, md.DET, 0, eps, 0.1, 1.0, r0pn, 10 ** 9)
            if len(res):
                S1 = md.lattice_energy(B) / (1 + eps) ** 2     # undo the final driving step
                rel.append(res[0, 0] / ((S0 - S1) / 0.8))
        rel = np.array(rel)
        print(f"  [energy] DET r0_per_node={r0pn}: {len(rel)} avalanches, "
              f"E_kernel/E_true median {np.median(rel):.5f}")
        assert len(rel) > 5 and abs(np.median(rel) - 1) < 1e-3


def test_resume_identical():
    """An interrupted + resumed run gives exactly the same output as a straight run."""
    tmp = tempfile.mkdtemp()
    try:
        base = [sys.executable, os.path.join(HERE, "run_simulations.py"), "--sizes", "64",
                "--iters", "6e5", "--burn-in", "1e5", "--workers", "1", "--only", "LH91_N64"]
        subprocess.run(base + ["--out", os.path.join(tmp, "a")], check=True, capture_output=True)
        # interrupted run: stop after 51 chunks (shorter run), then resume to the full length
        out_b = os.path.join(tmp, "b")
        d = os.path.join(out_b, "LH91_N64")
        subprocess.run([sys.executable, os.path.join(HERE, "run_simulations.py"), "--sizes", "64",
                        "--iters", str(5e5 - 1e5 + 1), "--burn-in", "1e5", "--workers", "1",
                        "--only", "LH91_N64", "--out", out_b], check=True, capture_output=True)
        os.remove(os.path.join(d, "DONE.json"))
        subprocess.run(base + ["--out", out_b], check=True, capture_output=True)
        a = np.load(os.path.join(tmp, "a", "LH91_N64", "avalanches.npz"))
        b = np.load(os.path.join(d, "avalanches.npz"))
        same = all(np.array_equal(a[k], b[k]) for k in a.files)
        print(f"  [resume] identical avalanche tables after resume: {same} ({len(a['E'])} avalanches)")
        assert same
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    from run_simulations import check_grid_dir
    if not check_grid_dir():      # test_resume_identical runs run_simulations.py on N = 64
        print("[tests] Tests not run: phd/initial_grids/ is required.", file=sys.stderr)
        sys.exit(1)
    for t in (test_box_counting, test_mle, test_energy_deterministic, test_resume_identical):
        print(t.__name__); t()
    print("ALL TESTS PASSED")
