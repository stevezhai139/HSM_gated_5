#!/usr/bin/env python3
"""
bench_complexity.py
-------------------
Empirical timing benchmark for the HSM kernel as a function of window size.

Produces two CSVs used by the paper:

    results/complexity.csv           (n_pts, time_s, method)       -> fig06
    results/scale_analysis.csv       (N, advisor_time_s, model)    -> supp01

The benchmark feeds synthetic template-frequency vectors and q(t) time
series of increasing length into ``hsm_v2_kernel.hsm_v2``.  For each N we
repeat R trials and store the median wall-clock.  Baseline comparison
uses a naive full O(N^2) DTW on the raw q(t) series, capped at N=3000
to keep runtime sane.

Supporting Lemma 4 (tight linear complexity) and Thm 6 / Corollary
(optimal window / cross-scale consistency) in the paper.
"""

from __future__ import annotations

import csv
import os
import sys
import time
from typing import List, Tuple

import numpy as np

HERE      = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(REPO_ROOT, "code", "experiments"))

import hsm_v2_kernel as K  # noqa: E402

# ---- configuration ---------------------------------------------------
N_GRID  = [100, 300, 1000, 3000, 10000, 30000]
N_DTW_CAP = 3000      # naive DTW is O(N^2) memory/time; cap here
V_VOCAB = 50           # template vocabulary size (kept constant)
N_REPS  = 5
SEED    = 42


# ---------------------------------------------------------------------
def synth_workload(n: int, rng: np.random.Generator
                   ) -> Tuple[np.ndarray, np.ndarray, list, list, set, set, set, set, int, int]:
    """Generate a synthetic (A, B) workload pair with qps series length n."""
    # template-frequency vectors (shared vocabulary V)
    freq_a = rng.integers(0, 20, size=V_VOCAB).astype(float)
    freq_b = (freq_a + rng.integers(-3, 4, size=V_VOCAB)).clip(min=0).astype(float)

    # arrival-time series of length ~n events, yielding q(t) of length n
    # hsm_v2 -> sp_v2 will bucket to 1 s windows; feed `times` as floats.
    times_a = np.sort(rng.uniform(0.0, float(n), size=n)).tolist()
    times_b = np.sort(rng.uniform(0.0, float(n), size=n)).tolist()

    # table/column sets
    tables = {f"t{i}" for i in range(8)}
    cols   = {f"c{i}" for i in range(16)}

    n_a = int(freq_a.sum())
    n_b = int(freq_b.sum())
    return freq_a, freq_b, times_a, times_b, tables, tables, cols, cols, n_a, n_b


def time_hsm(n: int, rng: np.random.Generator) -> float:
    fa, fb, ta, tb, Ta, Tb, Ca, Cb, na, nb = synth_workload(n, rng)
    t0 = time.perf_counter()
    K.hsm_v2(freq_a=fa, freq_b=fb, n_a=na, n_b=nb,
             tables_a=Ta, tables_b=Tb, cols_a=Ca, cols_b=Cb,
             times_a=ta, times_b=tb)
    return time.perf_counter() - t0


def time_naive_dtw(n: int, rng: np.random.Generator) -> float:
    """Naive O(N^2) DTW on two q(t) series of length n (baseline)."""
    # reuse hsm synth but compute DTW manually
    qa = rng.uniform(0.0, 10.0, size=n)
    qb = rng.uniform(0.0, 10.0, size=n)
    t0 = time.perf_counter()
    # full DP: D[i,j] = |qa[i]-qb[j]| + min(D[i-1,j], D[i,j-1], D[i-1,j-1])
    D = np.empty((n, n), dtype=np.float32)
    D[0, 0] = abs(qa[0] - qb[0])
    for j in range(1, n):
        D[0, j] = D[0, j-1] + abs(qa[0] - qb[j])
    for i in range(1, n):
        D[i, 0] = D[i-1, 0] + abs(qa[i] - qb[0])
        prev = D[i-1]
        curr = D[i]
        for j in range(1, n):
            curr[j] = abs(qa[i] - qb[j]) + min(prev[j], curr[j-1], prev[j-1])
    _ = float(D[-1, -1])
    return time.perf_counter() - t0


# ---------------------------------------------------------------------
def main() -> int:
    rng = np.random.default_rng(SEED)
    results_dir = os.path.join(REPO_ROOT, "results")
    os.makedirs(results_dir, exist_ok=True)

    # ---- HSM timing ---------------------------------------------------
    hsm_rows: List[Tuple[int, float]] = []
    print("== HSM kernel timing ==")
    for n in N_GRID:
        samples = []
        # warm-up (JIT-less, but numpy import + DWT plans)
        _ = time_hsm(min(n, 1000), rng)
        for _ in range(N_REPS):
            samples.append(time_hsm(n, rng))
        med = float(np.median(samples))
        hsm_rows.append((n, med))
        print(f"  n={n:6d}  median={med*1000:8.2f} ms  (reps={N_REPS})")

    # ---- Naive DTW baseline ------------------------------------------
    dtw_rows: List[Tuple[int, float]] = []
    print("== Naive O(N^2) DTW baseline ==")
    for n in N_GRID:
        if n > N_DTW_CAP:
            print(f"  n={n:6d}  skipped (exceeds N_DTW_CAP={N_DTW_CAP})")
            continue
        samples = []
        for _ in range(max(1, N_REPS // 2)):
            samples.append(time_naive_dtw(n, rng))
        med = float(np.median(samples))
        dtw_rows.append((n, med))
        print(f"  n={n:6d}  median={med*1000:8.2f} ms")

    # ---- write complexity.csv (fig06) --------------------------------
    comp_path = os.path.join(results_dir, "complexity.csv")
    with open(comp_path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["n_pts", "time_s", "method"])
        for n, t in hsm_rows:
            w.writerow([n, f"{t:.6f}", "HSM"])
        for n, t in dtw_rows:
            w.writerow([n, f"{t:.6f}", "DTW baseline"])
    print(f"[write] {comp_path}  ({len(hsm_rows)+len(dtw_rows)} rows)")

    # ---- write scale_analysis.csv (supp01) ---------------------------
    # empirical = HSM timings; fit = a * N * log2(N) + b via least-squares
    Ns = np.array([n for n, _ in hsm_rows], dtype=float)
    Ts = np.array([t for _, t in hsm_rows], dtype=float)
    X  = np.column_stack([Ns * np.log2(np.maximum(Ns, 2.0)), np.ones_like(Ns)])
    coef, *_ = np.linalg.lstsq(X, Ts, rcond=None)
    a, b = float(coef[0]), float(coef[1])
    fit = (X @ coef).clip(min=1e-9)
    print(f"  O(N log N) fit: T(N) = {a:.3e} * N log2(N) + {b:.3e}")

    scale_path = os.path.join(results_dir, "scale_analysis.csv")
    with open(scale_path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["N", "advisor_time_s", "model"])
        for n, t in zip(Ns.astype(int), Ts):
            w.writerow([int(n), f"{float(t):.6f}", "empirical"])
        for n, t in zip(Ns.astype(int), fit):
            w.writerow([int(n), f"{float(t):.6f}", "a*N*log(N)+b fit"])
    print(f"[write] {scale_path}  ({2*len(hsm_rows)} rows)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
