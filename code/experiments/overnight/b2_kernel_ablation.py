#!/usr/bin/env python3
"""
b2_kernel_ablation.py — F3 receipt: angular vs raw-cosine vs Euclidean.

Generates the burst-mid-Q workload trace deterministically (same seeds as
the existing Mid-Q rerun), builds per-window feature tuples WITHOUT touching
PostgreSQL, then computes the composite HSM under three kernel families:

  * angular  : current kernel — S_R = 1 - arccos(ρ_s)/π,
                                S_T = 1 - (2/π)·arccos(cos θ)
  * cosine   : raw-cosine     — S_R = (ρ_s + 1)/2,
                                S_T = (cos θ + 1)/2
  * euclid   : Euclidean      — S_R, S_T = 1 - ||a_norm - b_norm||/√2

S_V, S_A, S_P remain untouched (they do not use angular distance).

For each kernel, we compute Youden θ* and AUC on the within-phase /
cross-phase pair split over 10 blocks × 3 phases × W=5 (≡ 420 windows).

OUTPUT:  results/overnight_2026-04-16/b2_kernel_ablation/
    kernel_ablation_summary.csv  (kernel, workload, theta*, J*, TPR, TNR, AUC, n_within, n_cross)
    kernel_ablation_scores.csv   (block, window, phase, is_cross, kernel, hsm, S_R, S_V, S_T, S_A, S_P)

Runtime: ~5 min (pure CPU — no DB, no advisor).
"""
from __future__ import annotations

import csv
import math
import sys
import os
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve()
REPO = HERE.parents[3]
sys.path.insert(0, str(REPO / "code" / "experiments"))
sys.path.insert(0, str(REPO / "code" / "experiments" / "tier2"))

from hsm_similarity import build_window, DEFAULT_WEIGHTS  # noqa: E402
from hsm_burst_end_to_end import build_query_pool, build_trace, PHASE_ORDER  # noqa: E402
import hsm_burst_end_to_end as _burst  # noqa: E402

# ── Re-use the existing S_V, S_A, S_P kernels; override S_R/S_T per variant.
from hsm_v2_kernel import sv_v2, sa_v2, sp_v2  # noqa: E402
from hsm_similarity import s_v as _sv, s_a as _sa, s_p as _sp  # noqa: E402


QUERIES_PER_PHASE = 700   # Mid-Q parity with existing Mid-Q rerun
WINDOW_SIZE = 5
N_BLOCKS = 10
BASE_SEED = 0             # build_query_pool(seed=block) — same as main runner
OUT_DIR = REPO / "results" / "overnight_2026-04-16" / "b2_kernel_ablation"

# Patch the imported module's global so build_trace() uses our Q=700 (not its default 35).
_burst.QUERIES_PER_PHASE = QUERIES_PER_PHASE


def _aligned(fa, fb):
    vocab = sorted(set(fa) | set(fb))
    if not vocab:
        return np.zeros(1), np.zeros(1)
    va = np.array([fa.get(t, 0) for t in vocab], dtype=float)
    vb = np.array([fb.get(t, 0) for t in vocab], dtype=float)
    return va, vb


# ── Three S_R / S_T variants ────────────────────────────────────────────────

def _spearman_rho(a, b):
    from scipy.stats import spearmanr
    if a.size < 2 or b.size < 2 or a.sum() == 0 or b.sum() == 0:
        return 0.0
    rho, _ = spearmanr(a, b)
    if np.isnan(rho):
        rho = 1.0
    return float(np.clip(rho, -1.0, 1.0))


def _cos(a, b):
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom < 1e-12:
        return 1.0
    return float(np.clip(np.dot(a, b) / denom, -1.0, 1.0))


def s_r_angular(a, b):
    rho = _spearman_rho(a, b)
    return float(1.0 - math.acos(rho) / math.pi)


def s_r_cosine(a, b):
    rho = _spearman_rho(a, b)
    return float((rho + 1.0) / 2.0)


def s_r_euclid(a, b):
    # Normalise each vector to unit L2, then Euclidean distance on the sphere.
    # Two unit vectors have ||a-b|| ∈ [0, 2]; divide by √2 to match [0, 1].
    if a.size == 0 or b.size == 0:
        return 1.0
    na = np.linalg.norm(a); nb = np.linalg.norm(b)
    if na < 1e-12 or nb < 1e-12:
        return 1.0
    au = a / na; bu = b / nb
    d = float(np.linalg.norm(au - bu))
    return float(max(0.0, 1.0 - d / math.sqrt(2.0)))


def s_t_angular(a, b):
    c = _cos(a, b)
    return float(1.0 - (2.0 / math.pi) * math.acos(c))


def s_t_cosine(a, b):
    c = _cos(a, b)
    return float((c + 1.0) / 2.0)


def s_t_euclid(a, b):
    return s_r_euclid(a, b)  # same geometry — unit-norm Euclidean


KERNELS = {
    "angular": (s_r_angular, s_t_angular),
    "cosine":  (s_r_cosine,  s_t_cosine),
    "euclid":  (s_r_euclid,  s_t_euclid),
}


def hsm_variant(w_a, w_b, kernel, weights=DEFAULT_WEIGHTS):
    """Composite HSM using the chosen S_R/S_T family; S_V, S_A, S_P unchanged."""
    sr_fn, st_fn = KERNELS[kernel]
    va, vb = _aligned(w_a.template_freq, w_b.template_freq)
    sr = sr_fn(va, vb)
    st = st_fn(va, vb)
    sv = _sv(w_a, w_b)
    sa = _sa(w_a, w_b)
    sp = _sp(w_a, w_b)
    hsm = (weights["w_R"] * sr + weights["w_V"] * sv + weights["w_T"] * st
           + weights["w_A"] * sa + weights["w_P"] * sp)
    return float(hsm), {"S_R": sr, "S_V": sv, "S_T": st, "S_A": sa, "S_P": sp}


def youden(within, cross, step=0.0005):
    best = (0.0, -2.0, 0.0, 0.0)
    n_w = len(within); n_c = len(cross)
    if n_w == 0 or n_c == 0:
        return best + (0.0,)
    for i in range(int(1 / step) + 1):
        th = i * step
        tp = sum(1 for s in cross if s < th)
        tn = sum(1 for s in within if s >= th)
        J = tp / n_c + tn / n_w - 1
        if J > best[1]:
            best = (th, J, tp / n_c, tn / n_w)
    auc = sum(1 for sc in cross for sw in within if sc < sw) / (n_w * n_c)
    return (*best, auc)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summary_rows = []
    scores_rows = []

    for kernel in KERNELS:
        within_scores, cross_scores = [], []
        for block in range(1, N_BLOCKS + 1):
            pool  = build_query_pool(seed=block)
            trace = build_trace(pool, seed=block)
            n_per_phase = QUERIES_PER_PHASE // WINDOW_SIZE
            TRUE_TRANSITIONS = {n_per_phase, 2 * n_per_phase}
            # Split into windows; each window is a WorkloadWindow
            windows = []
            for w_idx in range(0, len(trace), WINDOW_SIZE):
                seg = trace[w_idx:w_idx + WINDOW_SIZE]
                if len(seg) < WINDOW_SIZE:
                    break
                sqls = [r[3] for r in seg]
                windows.append(build_window(sqls, window_id=w_idx // WINDOW_SIZE))
            # Pairwise adjacent windows
            for i in range(1, len(windows)):
                prev_w = windows[i - 1]
                curr_w = windows[i]
                is_cross = 1 if i in TRUE_TRANSITIONS else 0
                hsm_val, dims = hsm_variant(prev_w, curr_w, kernel)
                if is_cross:
                    cross_scores.append(hsm_val)
                else:
                    within_scores.append(hsm_val)
                scores_rows.append({
                    "block": block, "window": i, "phase_trans": is_cross,
                    "kernel": kernel, "hsm": round(hsm_val, 6),
                    "S_R": round(dims["S_R"], 6), "S_V": round(dims["S_V"], 6),
                    "S_T": round(dims["S_T"], 6), "S_A": round(dims["S_A"], 6),
                    "S_P": round(dims["S_P"], 6),
                })
        th, J, tpr, tnr, auc = youden(within_scores, cross_scores)
        summary_rows.append({
            "kernel": kernel, "workload": "burst-mid-Q",
            "n_within": len(within_scores), "n_cross": len(cross_scores),
            "theta_star": round(th, 4), "J_star": round(J, 4),
            "TPR": round(tpr, 4), "TNR": round(tnr, 4), "AUC": round(auc, 4),
        })
        print(f"  {kernel:7s}  n_w={len(within_scores):>4d}  n_c={len(cross_scores):>3d}  "
              f"θ*={th:.3f}  J*={J:.4f}  AUC={auc:.4f}")

    # Write summary + full scores
    summary_fp = OUT_DIR / "kernel_ablation_summary.csv"
    with summary_fp.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        w.writeheader(); w.writerows(summary_rows)
    scores_fp = OUT_DIR / "kernel_ablation_scores.csv"
    with scores_fp.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(scores_rows[0].keys()))
        w.writeheader(); w.writerows(scores_rows)
    print(f"\nwrote {summary_fp}")
    print(f"wrote {scores_fp}")


if __name__ == "__main__":
    main()
