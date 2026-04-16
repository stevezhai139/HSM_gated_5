#!/usr/bin/env python3
"""
bench_overhead.py
-----------------
Three-panel overhead breakdown of the HSM kernel for paper fig04.

Panel A -- per-component time (ms)   for S_R, S_V, S_T, S_A, S_P
Panel B -- memory footprint (KB)     for the core feature objects
Panel C -- end-to-end advisor (s)    over a sweep of window sizes N

Writes results/overhead.csv with schema  (panel, label, value, err).
Reuses bench_complexity's timings for panel C.
"""

from __future__ import annotations

import csv
import os
import statistics
import sys
import time
import tracemalloc
from typing import List, Tuple

import numpy as np

HERE      = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(REPO_ROOT, "code", "experiments"))

import hsm_v2_kernel as K  # noqa: E402

# ---- configuration ---------------------------------------------------
N_BENCH = 3000           # nominal window size for Panel A
N_REPS  = 50              # per-component repetitions
V_VOCAB = 50              # template-frequency vocabulary size
N_ADVISOR_GRID = [300, 1000, 3000, 10000]  # Panel C x-axis
SEED    = 42


# ---------------------------------------------------------------------
def _median_us(samples: List[float]) -> Tuple[float, float]:
    """Return (median, stdev) in microseconds."""
    arr = np.asarray(samples, dtype=float) * 1e6
    return float(np.median(arr)), float(arr.std(ddof=1)) if arr.size > 1 else 0.0


def bench_components(n: int, rng: np.random.Generator) -> dict:
    """Microbenchmark each S_* component individually at window size n."""
    # ---- prepare shared fixtures ---------------------------------------
    freq_a = rng.integers(0, 20, size=V_VOCAB).astype(float)
    freq_b = (freq_a + rng.integers(-3, 4, size=V_VOCAB)).clip(min=0).astype(float)
    n_a = int(freq_a.sum())
    n_b = int(freq_b.sum())
    tables_a = tables_b = {f"t{i}" for i in range(8)}
    cols_a   = cols_b   = {f"c{i}" for i in range(16)}
    times_a  = np.sort(rng.uniform(0.0, float(n), size=n)).tolist()
    times_b  = np.sort(rng.uniform(0.0, float(n), size=n)).tolist()

    out: dict = {}

    # S_R ---------------------------------------------------------------
    reps = []
    for _ in range(N_REPS):
        t0 = time.perf_counter(); K.sr_v2(freq_a, freq_b); reps.append(time.perf_counter() - t0)
    out["S_R"] = _median_us(reps)

    # S_V ---------------------------------------------------------------
    reps = []
    for _ in range(N_REPS):
        t0 = time.perf_counter(); K.sv_v2(n_a, n_b);       reps.append(time.perf_counter() - t0)
    out["S_V"] = _median_us(reps)

    # S_T ---------------------------------------------------------------
    reps = []
    for _ in range(N_REPS):
        t0 = time.perf_counter(); K.st_v2(freq_a, freq_b); reps.append(time.perf_counter() - t0)
    out["S_T"] = _median_us(reps)

    # S_A ---------------------------------------------------------------
    reps = []
    for _ in range(N_REPS):
        t0 = time.perf_counter(); K.sa_v2(tables_a, tables_b, cols_a, cols_b); reps.append(time.perf_counter() - t0)
    out["S_A"] = _median_us(reps)

    # S_P ---------------------------------------------------------------
    reps = []
    for _ in range(min(N_REPS, 10)):   # S_P dominates; fewer reps
        t0 = time.perf_counter(); K.sp_v2(times_a, times_b); reps.append(time.perf_counter() - t0)
    out["S_P"] = _median_us(reps)

    return out


# ---------------------------------------------------------------------
def bench_memory(n: int, rng: np.random.Generator) -> dict:
    """Peak memory footprint (KB) of the core feature objects."""
    out: dict = {}

    # freq vector
    tracemalloc.start()
    freq = rng.integers(0, 20, size=V_VOCAB).astype(float)
    _, peak = tracemalloc.get_traced_memory(); tracemalloc.stop()
    out["freq_vec"] = peak / 1024.0

    # qps series (input + bucketed)
    tracemalloc.start()
    times = np.sort(rng.uniform(0.0, float(n), size=n))
    qps = K.build_qps_series(rng.exponential(scale=10.0, size=n),
                             bin_seconds=1.0)
    _, peak = tracemalloc.get_traced_memory(); tracemalloc.stop()
    out["qps_series"] = peak / 1024.0

    # DWT coefficients
    tracemalloc.start()
    # emulate: running sp_v2 allocates DWT coefficients
    K.sp_v2(times.tolist(), times.tolist())
    _, peak = tracemalloc.get_traced_memory(); tracemalloc.stop()
    out["sp_v2_full"] = peak / 1024.0

    # full feature dict (freq + tables + cols + qps)
    tracemalloc.start()
    feat = {
        "freq_map": {f"q{i}": int(freq[i]) for i in range(V_VOCAB)},
        "tables":   {f"t{i}" for i in range(8)},
        "cols":     {f"c{i}" for i in range(16)},
        "arrivals": times.tolist(),
    }
    _, peak = tracemalloc.get_traced_memory(); tracemalloc.stop()
    out["feature_dict"] = peak / 1024.0

    return out


# ---------------------------------------------------------------------
def bench_advisor_sweep(grid: List[int], rng: np.random.Generator) -> List[Tuple[int, float, float]]:
    """End-to-end hsm_v2 wall-time at several window sizes (sec)."""
    rows: List[Tuple[int, float, float]] = []
    tables_a = tables_b = {f"t{i}" for i in range(8)}
    cols_a   = cols_b   = {f"c{i}" for i in range(16)}
    for n in grid:
        freq_a = rng.integers(0, 20, size=V_VOCAB).astype(float)
        freq_b = (freq_a + rng.integers(-3, 4, size=V_VOCAB)).clip(min=0).astype(float)
        n_a = int(freq_a.sum()); n_b = int(freq_b.sum())
        times_a = np.sort(rng.uniform(0.0, float(n), size=n)).tolist()
        times_b = np.sort(rng.uniform(0.0, float(n), size=n)).tolist()

        reps = []
        # warm-up
        K.hsm_v2(freq_a=freq_a, freq_b=freq_b, n_a=n_a, n_b=n_b,
                 tables_a=tables_a, tables_b=tables_b, cols_a=cols_a, cols_b=cols_b,
                 times_a=times_a, times_b=times_b)
        for _ in range(5):
            t0 = time.perf_counter()
            K.hsm_v2(freq_a=freq_a, freq_b=freq_b, n_a=n_a, n_b=n_b,
                     tables_a=tables_a, tables_b=tables_b, cols_a=cols_a, cols_b=cols_b,
                     times_a=times_a, times_b=times_b)
            reps.append(time.perf_counter() - t0)
        med = float(np.median(reps))
        sd  = float(np.std(reps, ddof=1)) if len(reps) > 1 else 0.0
        rows.append((n, med, sd))
        print(f"  advisor  n={n:6d}  median={med*1000:8.2f} ms  sd={sd*1000:6.2f} ms")
    return rows


# ---------------------------------------------------------------------
def main() -> int:
    rng = np.random.default_rng(SEED)
    results_dir = os.path.join(REPO_ROOT, "results")
    os.makedirs(results_dir, exist_ok=True)
    out_csv = os.path.join(results_dir, "overhead.csv")

    print(f"== Panel A: per-component time at n={N_BENCH} ==")
    comp = bench_components(N_BENCH, rng)
    for k, (m, s) in comp.items():
        print(f"  {k:4s}  median={m/1000:8.3f} ms   sd={s/1000:6.3f} ms")

    print("== Panel B: memory footprint (KB) ==")
    mem = bench_memory(N_BENCH, rng)
    for k, v in mem.items():
        print(f"  {k:14s}  peak={v:8.2f} KB")

    print("== Panel C: advisor sweep ==")
    adv = bench_advisor_sweep(N_ADVISOR_GRID, rng)

    # ---- write CSV ----------------------------------------------------
    with open(out_csv, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["panel", "label", "value", "err"])
        # Panel A: time in ms (convert from microseconds)
        for k in ("S_R", "S_V", "S_T", "S_A", "S_P"):
            m_us, s_us = comp[k]
            w.writerow(["time", k, f"{m_us/1000:.4f}", f"{s_us/1000:.4f}"])
        # Panel B: memory in KB
        mem_labels = [("freq_vec", "freq"), ("qps_series", "qps"),
                      ("sp_v2_full", "DWT+SAX+DTW"), ("feature_dict", "feat_dict")]
        for key, lab in mem_labels:
            w.writerow(["mem", lab, f"{mem[key]:.2f}", "0"])
        # Panel C: advisor time in seconds
        for n, med, sd in adv:
            w.writerow(["advisor", f"N={n}", f"{med:.6f}", f"{sd:.6f}"])

    print(f"[write] {out_csv}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
