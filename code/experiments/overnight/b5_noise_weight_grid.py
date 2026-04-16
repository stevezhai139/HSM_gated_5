#!/usr/bin/env python3
"""
b5_noise_weight_grid.py — Robustness receipt: noise × weight joint sensitivity.

Existing noise/weight sweeps in Version 5 are 1-D (noise alone, weight alone).
Reviewers commonly ask: what if BOTH perturb simultaneously? This script runs
a full 2-D grid on burst-mid-Q and reports Youden θ*, J*, AUC, and deployed-θ
(=0.826) precision/recall per cell.

Grid:
  noise_level  z ∈ {0.0, 0.2, 0.4, 0.6, 0.8}      (Gaussian noise stdev
                                                     as fraction of qps-mean)
  weight_pert  δ ∈ {0.0, 0.025, 0.05, 0.075, 0.1} (symmetric Uniform[-δ, δ]
                                                     added to each weight,
                                                     then re-normalised)

For each (z, δ) cell we draw 10 block workloads × 3 weight draws = 30 runs,
compute per-window HSM scores, and report:
  * theta_star_Youden, J_star_Youden, AUC
  * precision_at_0826, recall_at_0826  (deployed θ from Theorem 3 at Q=700)

OUTPUT:  results/overnight_2026-04-16/b5_noise_weight/
    noise_weight_grid.csv  (z, delta, n_within, n_cross, theta*, J*, AUC,
                             precision_0826, recall_0826)

Runtime: ~1.5 h (5×5 grid × 3 draws × 10 blocks × S_P DTW cost dominates).
"""
from __future__ import annotations

import csv
import random
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve()
REPO = HERE.parents[3]
sys.path.insert(0, str(REPO / "code" / "experiments"))
sys.path.insert(0, str(REPO / "code" / "experiments" / "tier2"))

from hsm_similarity import build_window, DEFAULT_WEIGHTS, hsm_score  # noqa: E402
from hsm_burst_end_to_end import build_query_pool, build_trace  # noqa: E402
import hsm_burst_end_to_end as _burst  # noqa: E402

QUERIES_PER_PHASE = 700
WINDOW_SIZE = 5
N_BLOCKS = 10
N_WEIGHT_DRAWS = 3
THETA_DEPLOYED = 0.826   # = 1 - Q_min/Q for Q=700

NOISE_LEVELS  = [0.0, 0.2, 0.4, 0.6, 0.8]
WEIGHT_DELTAS = [0.0, 0.025, 0.05, 0.075, 0.1]

OUT_DIR = REPO / "results" / "overnight_2026-04-16" / "b5_noise_weight"

# Patch the imported module's global so build_trace() uses our Q=700 (not its default 35).
_burst.QUERIES_PER_PHASE = QUERIES_PER_PHASE


def perturb_weights(weights, delta, rng):
    """Symmetric Uniform[-delta, delta] perturbation, re-normalised to sum=1."""
    if delta == 0:
        return dict(weights)
    keys = list(weights.keys())
    new = {k: max(0.0, weights[k] + rng.uniform(-delta, delta)) for k in keys}
    s = sum(new.values())
    return {k: v / s for k, v in new.items()} if s > 0 else dict(weights)


def add_arrival_noise(window, z, rng):
    """Return a new WorkloadWindow with arrival timestamps jittered by
    N(0, z·t_mean). z=0 is a no-op; z>=0.8 is extreme.
    """
    if z == 0:
        return window
    from copy import deepcopy
    w = deepcopy(window)
    if not w.queries:
        return w
    ts = [q.timestamp for q in w.queries]
    t_mean = float(np.mean(ts)) if ts else 1.0
    sigma = max(1e-6, z * t_mean)
    for q in w.queries:
        q.timestamp = max(0.0, q.timestamp + rng.gauss(0.0, sigma))
    return w


def youden_and_deployed(within, cross, theta_deployed, step=0.0005):
    n_w, n_c = len(within), len(cross)
    if n_w == 0 or n_c == 0:
        return (0.0, 0.0, 0.0, 0.0, 0.0)
    best = (0.0, -2.0, 0.0, 0.0)
    for i in range(int(1 / step) + 1):
        th = i * step
        tp = sum(1 for s in cross if s < th)
        tn = sum(1 for s in within if s >= th)
        J = tp / n_c + tn / n_w - 1
        if J > best[1]:
            best = (th, J, tp / n_c, tn / n_w)
    auc = sum(1 for sc in cross for sw in within if sc < sw) / (n_w * n_c)
    # Deployed θ operating point
    tp_d = sum(1 for s in cross if s < theta_deployed)
    fp_d = sum(1 for s in within if s < theta_deployed)
    prec_d = tp_d / (tp_d + fp_d) if (tp_d + fp_d) > 0 else 0.0
    rec_d  = tp_d / n_c
    return (best[0], best[1], auc, prec_d, rec_d)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    weight_rng = random.Random(4242)

    for z in NOISE_LEVELS:
        for delta in WEIGHT_DELTAS:
            within_all, cross_all = [], []
            for draw in range(N_WEIGHT_DRAWS):
                weights = perturb_weights(DEFAULT_WEIGHTS, delta, weight_rng)
                for block in range(1, N_BLOCKS + 1):
                    pool  = build_query_pool(seed=block)
                    trace = build_trace(pool, seed=block)
                    n_per_phase = QUERIES_PER_PHASE // WINDOW_SIZE
                    TRUE_TRANSITIONS = {n_per_phase, 2 * n_per_phase}
                    noise_rng = random.Random(int(block * 100 + draw))
                    windows = []
                    for w_idx in range(0, len(trace), WINDOW_SIZE):
                        seg = trace[w_idx:w_idx + WINDOW_SIZE]
                        if len(seg) < WINDOW_SIZE:
                            break
                        sqls = [r[3] for r in seg]
                        w = build_window(sqls, window_id=w_idx // WINDOW_SIZE)
                        w = add_arrival_noise(w, z, noise_rng)
                        windows.append(w)
                    for i in range(1, len(windows)):
                        hsm_val, _ = hsm_score(windows[i - 1], windows[i], weights=weights)
                        if i in TRUE_TRANSITIONS:
                            cross_all.append(hsm_val)
                        else:
                            within_all.append(hsm_val)
            theta_s, J_s, auc, prec_d, rec_d = youden_and_deployed(
                within_all, cross_all, THETA_DEPLOYED)
            rows.append({
                "z": z, "delta": delta,
                "n_within": len(within_all), "n_cross": len(cross_all),
                "theta_star": round(theta_s, 4), "J_star": round(J_s, 4),
                "AUC": round(auc, 4),
                "precision_0826": round(prec_d, 4),
                "recall_0826": round(rec_d, 4),
            })
            print(f"  z={z:.1f}  δ={delta:.3f}  "
                  f"θ*={theta_s:.3f} J*={J_s:.4f} AUC={auc:.4f}  "
                  f"P@0.826={prec_d:.3f} R@0.826={rec_d:.3f}")

    fp = OUT_DIR / "noise_weight_grid.csv"
    with fp.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"\nwrote {fp}")


if __name__ == "__main__":
    main()
