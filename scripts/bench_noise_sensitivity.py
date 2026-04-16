#!/usr/bin/env python3
"""
bench_noise_sensitivity.py
--------------------------
Empirical robustness of the HSM discrimination ratio (DR) under five
controlled noise injection modes N1..N5 specified in paper §IV.

Schema written to results/noise_sensitivity.csv:
    noise_level, condition, dr
where condition in {"N1","N2","N3","N4","N5"} and noise_level is the
relative perturbation magnitude (0.0 = no noise, 1.0 = maximum).

Noise catalogue
---------------
    N1  Gaussian noise on the template-frequency vector
    N2  Template dropout  (Bernoulli removal with probability = level)
    N3  Phase jitter on the q(t) arrival times, +/- level * mean inter-arrival
    N4  Amplitude scaling of q(t):   q <- q * (1 +/- level)
    N5  Spurious template insertion: add floor(level * V) new templates

DR is defined as mean(within) / mean(cross) over a bank of synthetic
phase pairs; noise is injected only on the "B" side of each pair.

No DB run is required -- everything runs off synthetic phase templates.
"""

from __future__ import annotations

import csv
import os
import sys
from typing import Dict, List, Tuple

import numpy as np

HERE      = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(REPO_ROOT, "code", "experiments"))

import hsm_v2_kernel as K  # noqa: E402

# ---- configuration ---------------------------------------------------
NOISE_LEVELS = [0.0, 0.1, 0.2, 0.4, 0.6]
CONDITIONS   = ["N1", "N2", "N3", "N4", "N5"]
N_PAIRS      = 20        # number of phase pairs per level (for variance)
V_VOCAB      = 50        # template vocabulary
N_QPS        = 3000      # qps time-series length (seconds)
SEED         = 42


# ---------------------------------------------------------------------
def _make_phase(rng: np.random.Generator, shift: int = 0) -> dict:
    """Build a synthetic phase feature dict."""
    freq = rng.integers(0, 20, size=V_VOCAB).astype(float)
    # Circular shift mimics "different phase" via rotated template usage.
    if shift:
        freq = np.roll(freq, shift)
    arrivals = np.sort(rng.uniform(0.0, float(N_QPS), size=N_QPS))
    return {
        "freq":     freq.astype(float),
        "arrivals": arrivals.astype(float),
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


# ---------------------------------------------------------------------
def apply_noise(feat: dict, condition: str, level: float,
                rng: np.random.Generator) -> dict:
    """Return a noisy copy of `feat` for the given Nk condition."""
    out = {k: (v.copy() if hasattr(v, "copy") else set(v) if isinstance(v, set) else v)
           for k, v in feat.items()}
    if level <= 0.0 or condition == "base":
        return out

    f = out["freq"].astype(float).copy()
    q_t = out["arrivals"].astype(float).copy()

    if condition == "N1":
        # additive Gaussian noise on freq vector (sigma = level * max freq)
        sigma = level * (float(f.max()) if f.max() > 0 else 1.0)
        f = np.clip(f + rng.normal(0.0, sigma, size=f.size), 0.0, None)
    elif condition == "N2":
        # Bernoulli dropout on templates
        mask = rng.random(size=f.size) > level
        f = f * mask
    elif condition == "N3":
        # phase jitter on arrivals
        mean_interval = float(np.mean(np.diff(q_t))) if q_t.size > 1 else 1.0
        jitter = rng.normal(0.0, level * mean_interval, size=q_t.size)
        q_t = np.sort(np.clip(q_t + jitter, 0.0, None))
    elif condition == "N4":
        # amplitude scaling -- perturb arrival DENSITY by subsampling/duplicating
        factor = 1.0 + rng.choice([-1.0, 1.0]) * level
        if factor > 1.0:
            extra = int((factor - 1.0) * q_t.size)
            inject = rng.uniform(float(q_t.min()), float(q_t.max()),
                                 size=extra).astype(float)
            q_t = np.sort(np.concatenate([q_t, inject]))
        elif factor < 1.0:
            keep = max(1, int(factor * q_t.size))
            idx = rng.choice(q_t.size, size=keep, replace=False)
            q_t = np.sort(q_t[idx])
    elif condition == "N5":
        # spurious template insertion: append new templates with small counts
        n_new = max(1, int(level * V_VOCAB))
        f = np.concatenate([f, rng.integers(1, 5, size=n_new).astype(float)])
        # re-project B-side freq onto original vocab length via truncation
        # (hsm_v2 aligns to max-length before scoring)
    out["freq"] = f
    out["arrivals"] = q_t
    out["n"] = int(f.sum())
    return out


def align_for_hsm(a: dict, b: dict) -> Tuple[dict, dict]:
    """Zero-pad freq vectors to equal length (needed for N5)."""
    fa = a["freq"]; fb = b["freq"]
    V = max(fa.size, fb.size)
    if fa.size < V:
        fa = np.concatenate([fa, np.zeros(V - fa.size)])
    if fb.size < V:
        fb = np.concatenate([fb, np.zeros(V - fb.size)])
    a2 = dict(a); a2["freq"] = fa; a2["n"] = int(fa.sum())
    b2 = dict(b); b2["freq"] = fb; b2["n"] = int(fb.sum())
    return a2, b2


# ---------------------------------------------------------------------
def main() -> int:
    rng = np.random.default_rng(SEED)
    results_dir = os.path.join(REPO_ROOT, "results")
    os.makedirs(results_dir, exist_ok=True)

    # Pre-compute N_PAIRS of (phaseA, phaseA'=within-twin, phaseB=cross)
    base: List[Tuple[dict, dict, dict]] = []
    for _ in range(N_PAIRS):
        a = _make_phase(rng)
        # within-phase twin: small identity perturbation
        a_twin = {
            "freq":     np.clip(a["freq"] + rng.normal(0, 0.5, size=V_VOCAB), 0, None),
            "arrivals": np.sort(a["arrivals"] + rng.normal(0, 0.5, size=N_QPS)),
            "tables":   set(a["tables"]),
            "cols":     set(a["cols"]),
            "n":        None,
        }
        a_twin["n"] = int(a_twin["freq"].sum())
        # cross-phase: rotated template usage + shuffled arrivals
        b = _make_phase(rng, shift=V_VOCAB // 3)
        base.append((a, a_twin, b))

    rows: List[Tuple[float, str, float]] = []
    print(f"== Noise sensitivity sweep (N_PAIRS={N_PAIRS}, levels={NOISE_LEVELS}) ==")
    for level in NOISE_LEVELS:
        for cond in CONDITIONS:
            within_scores: List[float] = []
            cross_scores:  List[float] = []
            for (a, a_twin, b) in base:
                # inject noise on the B side of each pair
                a_twin_n = apply_noise(a_twin, cond, level, rng)
                b_n      = apply_noise(b,      cond, level, rng)
                aa, bb = align_for_hsm(a, a_twin_n)
                within_scores.append(_hsm(aa, bb))
                aa, bb = align_for_hsm(a, b_n)
                cross_scores.append(_hsm(aa, bb))
            w_mean = float(np.mean(within_scores))
            c_mean = float(np.mean(cross_scores))
            dr = w_mean / c_mean if c_mean > 0 else float("nan")
            rows.append((level, cond, dr))
            print(f"  level={level:.2f}  {cond}  "
                  f"within={w_mean:.3f}  cross={c_mean:.3f}  DR={dr:.3f}")

    out_csv = os.path.join(results_dir, "noise_sensitivity.csv")
    with open(out_csv, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["noise_level", "condition", "dr"])
        for lvl, cond, dr in rows:
            w.writerow([f"{lvl:.2f}", cond, f"{dr:.6f}"])
    print(f"[write] {out_csv}  ({len(rows)} rows)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
