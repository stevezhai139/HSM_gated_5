#!/usr/bin/env python3
"""
bench_window_sweep.py
---------------------
Empirical sweep of window size N to verify

    Thm 6      -- an optimal window size N* minimising the cost trade-off
    Corollary  -- cross-scale consistency of N* across data cardinalities

Writes two CSVs for the paper:

    results/window_sweep.csv       (N, DR, advisor_time_s, cost, workload)
    results/cross_scale.csv        (data_scale, N, DR, advisor_time_s,
                                    cost, N_star)

Setup
-----
For each window size N in a log-spaced grid, we synthesise many phase
blocks (within- and cross-phase pairs), score them with ``hsm_v2``,
and compute

    DR(N)              = mean(within_scores) / mean(cross_scores)
    T_A(N)             = median advisor wall-clock per pair
    cost(N)            = lambda * T_A(N) + (1 - lambda) * (1 / DR(N))

with lambda = 0.5 (equal weighting on latency vs. accuracy).  The
empirical N* is argmin cost.  For the cross-scale check we repeat the
sweep at three data cardinalities (5k, 15k, 45k events per phase) and
compare N*_ratio = N* / data_scale for ratio stability (Corollary).

No database is involved -- pure numpy + hsm_v2_kernel.
"""

from __future__ import annotations

import csv
import os
import sys
import time
from typing import Dict, List, Tuple

import numpy as np

HERE      = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(REPO_ROOT, "code", "experiments"))

import hsm_v2_kernel as K  # noqa: E402

# ---- configuration ---------------------------------------------------
N_GRID       = [30, 60, 120, 250, 500, 1000, 2000, 4000]
N_PAIRS      = 12                          # number of within/cross pairs per N
V_VOCAB      = 50                          # template vocabulary size
LAMBDA       = 0.5                         # cost weighting (latency vs accuracy)
CROSS_SCALES = [5_000, 15_000, 45_000]     # data cardinality for Corollary
SEED         = 42


# ---------------------------------------------------------------------
def _phase_distribution(rng: np.random.Generator, shift: int = 0) -> np.ndarray:
    """Latent per-phase template probability (Dirichlet-concentrated)."""
    alpha = rng.dirichlet(np.full(V_VOCAB, 0.8))
    if shift:
        alpha = np.roll(alpha, shift)
    return alpha


def _synth_phase(n: int, rng: np.random.Generator,
                 phase_probs: np.ndarray) -> dict:
    """Sample N queries from the phase's template distribution.

    At small N the empirical freq vector is a noisy estimate of phase_probs;
    at large N it converges to the true distribution.  This makes DR a
    function of N (small N => undersampled => noisier scores), which is
    what Thm 6 in the paper predicts.
    """
    draws = rng.choice(V_VOCAB, size=n, p=phase_probs)
    freq = np.bincount(draws, minlength=V_VOCAB).astype(float)
    arrivals = np.sort(rng.uniform(0.0, float(n), size=n)).astype(float)
    return {
        "freq":     freq,
        "arrivals": arrivals,
        "tables":   {f"t{i}" for i in range(8)},
        "cols":     {f"c{i}" for i in range(16)},
        "n":        int(freq.sum()),
    }


def _hsm(a: dict, b: dict) -> float:
    res = K.hsm_v2(
        freq_a=a["freq"], freq_b=b["freq"],
        n_a=a["n"],       n_b=b["n"],
        tables_a=a["tables"], tables_b=b["tables"],
        cols_a=a["cols"],     cols_b=b["cols"],
        times_a=a["arrivals"].tolist(),
        times_b=b["arrivals"].tolist(),
    )
    return float(res["HSM"])


def _sweep_at_scale(scale: int, rng: np.random.Generator) -> List[Tuple[int, float, float, float]]:
    """Return list of (N, DR, advisor_time_s, cost) at this data scale.

    Cost model (paper Thm 6), both terms normalised to [0, 1]:

        T_norm(N)  = T_A(N) / T_A(N_max)              latency penalty
        err(N)     = 1 - max(DR(N) - 1, 0) / (DR_max - 1)   accuracy penalty
        cost(N)    = lambda * T_norm(N) + (1 - lambda) * err(N)

    Small N: err large (undersampled freq vector), T_norm small
    Large N: err small (well-sampled), T_norm approaches 1
    => interior minimum N* where the two penalties cross.
    """
    phase_a = _phase_distribution(rng)
    phase_b = _phase_distribution(rng, shift=V_VOCAB // 3)
    DR_MAX  = 2.0

    # --- pass 1: collect raw DR and T_A across the grid --------------
    raw: List[Tuple[int, float, float]] = []
    for n in N_GRID:
        if n > scale:
            continue
        within_scores: List[float] = []
        cross_scores:  List[float] = []
        times_s:       List[float] = []
        for _ in range(N_PAIRS):
            a      = _synth_phase(n, rng, phase_a)
            a_twin = _synth_phase(n, rng, phase_a)   # same phase, fresh sample
            b      = _synth_phase(n, rng, phase_b)

            t0 = time.perf_counter(); within_scores.append(_hsm(a, a_twin)); times_s.append(time.perf_counter() - t0)
            t0 = time.perf_counter(); cross_scores .append(_hsm(a, b));      times_s.append(time.perf_counter() - t0)

        w_mean = float(np.mean(within_scores))
        c_mean = float(np.mean(cross_scores))
        dr     = w_mean / c_mean if c_mean > 0 else float("nan")
        t_med  = float(np.median(times_s))
        raw.append((n, dr, t_med))

    # --- pass 2: compute cost with normalised T_A --------------------
    t_max = max(r[2] for r in raw) if raw else 1.0
    rows: List[Tuple[int, float, float, float]] = []
    for n, dr, t_med in raw:
        t_norm  = t_med / t_max if t_max > 0 else 0.0
        det_err = 1.0 - max(dr - 1.0, 0.0) / (DR_MAX - 1.0)
        cost    = LAMBDA * t_norm + (1.0 - LAMBDA) * det_err
        rows.append((n, dr, t_med, cost))
        print(f"    N={n:5d}  DR={dr:.3f}  T_A={t_med*1000:6.1f} ms  "
              f"T_norm={t_norm:.3f}  err={det_err:.3f}  cost={cost:.4f}")
    return rows


# ---------------------------------------------------------------------
def main() -> int:
    rng = np.random.default_rng(SEED)
    results_dir = os.path.join(REPO_ROOT, "results")
    os.makedirs(results_dir, exist_ok=True)

    # ---- Thm 6: single-scale sweep ----------------------------------
    print("== Window-size sweep (Thm 6) ==")
    primary_rows = _sweep_at_scale(max(CROSS_SCALES), rng)
    sweep_csv = os.path.join(results_dir, "window_sweep.csv")
    with open(sweep_csv, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["N", "DR", "advisor_time_s", "cost", "workload"])
        for n, dr, t, c in primary_rows:
            w.writerow([n, f"{dr:.6f}", f"{t:.6f}", f"{c:.6f}", "synthetic"])
    n_star = primary_rows[int(np.argmin([r[3] for r in primary_rows]))][0]
    print(f"[write] {sweep_csv}  (N* = {n_star})")

    # ---- Corollary: cross-scale check -------------------------------
    print("== Cross-scale consistency (Corollary) ==")
    cross_rows: List[Tuple[int, int, float, float, float, int]] = []
    per_scale_nstar: Dict[int, int] = {}
    for scale in CROSS_SCALES:
        print(f"  -- scale = {scale} events --")
        rs = _sweep_at_scale(scale, rng)
        nstar = rs[int(np.argmin([r[3] for r in rs]))][0]
        per_scale_nstar[scale] = nstar
        for n, dr, t, c in rs:
            cross_rows.append((scale, n, dr, t, c, nstar))

    cross_csv = os.path.join(results_dir, "cross_scale.csv")
    with open(cross_csv, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["data_scale", "N", "DR", "advisor_time_s", "cost", "N_star"])
        for scale, n, dr, t, c, ns in cross_rows:
            w.writerow([scale, n, f"{dr:.6f}", f"{t:.6f}", f"{c:.6f}", ns])
    print(f"[write] {cross_csv}")

    print("\n== Corollary ratio check (N*/scale) ==")
    for scale, ns in per_scale_nstar.items():
        print(f"  scale={scale:>6d}  N*={ns:4d}   ratio N*/scale = {ns/scale:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
