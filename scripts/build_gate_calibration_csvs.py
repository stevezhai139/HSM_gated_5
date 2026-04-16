#!/usr/bin/env python3
"""
build_gate_calibration_csvs.py
------------------------------
Produce the two calibration CSVs consumed by the supplementary figures:

    results/gate_sensitivity.csv     (z, trigger_fraction, workload)
    results/theta_calibration.csv    (theta, cost_mean, ci_low, ci_high, workload)

Both CSVs are derived entirely from the per-workload pair-score CSVs written
by the validation scripts, i.e. no database run is required.

supp02 -- Gate sensitivity:
    For each workload we treat the within-phase scores as the null
    distribution.  The gate threshold expressed in sigma units is
        theta(z) = mu_within - z * sigma_within
    trigger_fraction is the empirical false-trigger rate on the WITHIN
    pairs, i.e. P(score < theta(z) | within-phase).  As z grows the
    curve is driven towards zero at a rate governed by the Hoeffding
    concentration bound (Thm 4 in the paper).

supp03 -- theta calibration:
    For each candidate theta in [0.30, 0.95] step 0.05 we compute the
    balanced detection error
        cost(theta) = 0.5 * FPR(theta) + 0.5 * FNR(theta)
    where
        FPR(theta) = P(score <  theta | within)   (false trigger)
        FNR(theta) = P(score >= theta | cross)    (missed change)
    The 95 % confidence interval is obtained by B = 1000 bootstrap
    resamples of within and cross scores independently (fixed seed).

Both files pool across all workloads whose pair-score CSV is present
under code/results/.
"""

from __future__ import annotations

import csv
import os
import sys
from typing import Dict, List, Tuple

import numpy as np

# ---- workload catalog ------------------------------------------------
# (figure label,  pair-score CSV path relative to HSM_gated/)
WORKLOAD_SPECS: List[Tuple[str, str]] = [
    ("oltp",            "code/results/oltp_validation/oltp_hsm_static_pair_scores.csv"),
    ("burst",           "code/results/burst_validation/burst_hsm_pair_scores.csv"),
    ("burst_v2",        "code/results/burst_v2_validation/burst_v2_hsm_pair_scores.csv"),
    ("burst_v3",        "code/results/burst_v3_validation/burst_v3_hsm_pair_scores.csv"),
    ("job",             "code/results/job_validation/job_hsm_execute_pair_scores.csv"),
    ("job_complexity",  "code/results/job_validation/job_hsm_complexity_execute_pair_scores.csv"),
    ("sdss",            "code/results/sdss_validation/sdss_hsm_pair_scores.csv"),
]


# ---------------------------------------------------------------------
def _read_pair_csv(path: str) -> Tuple[np.ndarray, np.ndarray]:
    """Return (within_scores, cross_scores) as float arrays."""
    within: List[float] = []
    cross:  List[float] = []
    with open(path, newline="") as fh:
        reader = csv.reader(fh)
        header = next(reader, None)
        # Handle both schemas:
        #   (workload, group, score)   -- from dump_pair_scores_csv()
        #   (score, group)             -- legacy 2-col form
        if header and len(header) == 3:
            g_idx, s_idx = 1, 2
        else:
            g_idx, s_idx = 1, 0
        for row in reader:
            if len(row) <= max(g_idx, s_idx):
                continue
            try:
                s = float(row[s_idx])
            except ValueError:
                continue
            g = row[g_idx].strip().lower()
            if g.startswith("within"):
                within.append(s)
            elif g.startswith("cross"):
                cross.append(s)
    return np.asarray(within, dtype=float), np.asarray(cross, dtype=float)


# ---- supp02 : gate sensitivity --------------------------------------
def build_gate_sensitivity(workloads: Dict[str, Tuple[np.ndarray, np.ndarray]],
                           out_csv: str) -> int:
    z_grid = np.arange(0.0, 4.01, 0.25)
    n = 0
    with open(out_csv, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["z", "trigger_fraction", "workload"])
        for wl, (within, _cross) in sorted(workloads.items()):
            if within.size == 0:
                continue
            mu = float(within.mean())
            sd = float(within.std(ddof=1)) if within.size > 1 else 0.0
            for z in z_grid:
                theta = mu - z * sd
                frac = float((within < theta).mean())
                w.writerow([f"{z:.2f}", f"{frac:.6f}", wl])
                n += 1
    return n


# ---- supp03 : theta calibration --------------------------------------
def _cost_curve(within: np.ndarray,
                cross:  np.ndarray,
                theta_grid: np.ndarray) -> np.ndarray:
    """Return balanced cost at each theta: 0.5 FPR + 0.5 FNR."""
    if within.size == 0 or cross.size == 0:
        return np.full_like(theta_grid, np.nan, dtype=float)
    fpr = (within[:, None] <  theta_grid[None, :]).mean(axis=0)
    fnr = (cross [:, None] >= theta_grid[None, :]).mean(axis=0)
    return 0.5 * fpr + 0.5 * fnr


def build_theta_calibration(workloads: Dict[str, Tuple[np.ndarray, np.ndarray]],
                            out_csv: str,
                            n_boot: int = 1000,
                            seed: int = 42) -> int:
    theta_grid = np.round(np.arange(0.30, 0.951, 0.05), 4)
    rng = np.random.default_rng(seed)
    n = 0
    with open(out_csv, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["theta", "cost_mean", "ci_low", "ci_high", "workload"])
        for wl, (within, cross) in sorted(workloads.items()):
            if within.size == 0 or cross.size == 0:
                continue
            point = _cost_curve(within, cross, theta_grid)         # (G,)
            # bootstrap
            boot = np.empty((n_boot, theta_grid.size), dtype=float)
            for b in range(n_boot):
                wb = rng.choice(within, size=within.size, replace=True)
                cb = rng.choice(cross,  size=cross.size,  replace=True)
                boot[b] = _cost_curve(wb, cb, theta_grid)
            lo = np.nanpercentile(boot,  2.5, axis=0)
            hi = np.nanpercentile(boot, 97.5, axis=0)
            for t, m, l, h in zip(theta_grid, point, lo, hi):
                w.writerow([f"{t:.4f}", f"{m:.6f}", f"{l:.6f}", f"{h:.6f}", wl])
                n += 1
    return n


# ---------------------------------------------------------------------
def main() -> int:
    here = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.dirname(here)
    results_dir = os.path.join(repo_root, "results")
    os.makedirs(results_dir, exist_ok=True)

    workloads: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    for label, rel in WORKLOAD_SPECS:
        path = os.path.join(repo_root, rel)
        if not os.path.exists(path):
            print(f"[skip]   {label:16s}  (no pair-scores at {rel})")
            continue
        within, cross = _read_pair_csv(path)
        if within.size == 0 and cross.size == 0:
            print(f"[skip]   {label:16s}  (empty)")
            continue
        workloads[label] = (within, cross)
        print(f"[ok]     {label:16s}  within={within.size:5d}  cross={cross.size:5d}")

    if not workloads:
        print("[error] no pair-score CSVs found -- run the validation scripts first")
        return 1

    supp02 = os.path.join(results_dir, "gate_sensitivity.csv")
    supp03 = os.path.join(results_dir, "theta_calibration.csv")

    n2 = build_gate_sensitivity(workloads, supp02)
    print(f"[write]  {supp02}  ({n2} rows)")

    n3 = build_theta_calibration(workloads, supp03, n_boot=1000, seed=42)
    print(f"[write]  {supp03}  ({n3} rows)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
