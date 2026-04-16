#!/usr/bin/env python3
"""
analyze_overnight.py — Morning analyzer for the B1–B5 overnight batch.

Reads all CSV outputs, prints a compact per-experiment summary, and
tabulates bootstrap 95% CIs where applicable. Designed to be glance-able
over morning coffee before diving into the paper rewrite.

USAGE:  python3 code/experiments/overnight/analyze_overnight.py
"""
from __future__ import annotations

import csv
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve()
REPO = HERE.parents[3]
OUT = REPO / "results" / "overnight_2026-04-16"


def _read_csv(fp):
    if not fp.exists():
        return None
    with fp.open() as f:
        return list(csv.DictReader(f))


def _bootstrap_ci(xs, stat=np.median, B=2000, alpha=0.05, seed=7):
    if not xs:
        return None, None, None
    rng = np.random.default_rng(seed)
    arr = np.array(xs, dtype=float)
    boots = np.array([stat(rng.choice(arr, size=len(arr), replace=True)) for _ in range(B)])
    lo, hi = np.quantile(boots, [alpha/2, 1-alpha/2])
    return float(stat(arr)), float(lo), float(hi)


def print_header(t):
    print("\n" + "═" * 72)
    print(f"  {t}")
    print("═" * 72)


def _by_condition(rows, key="condition"):
    g = defaultdict(list)
    for r in rows:
        g[r[key]].append(r)
    return g


def analyse_burst_csv(rows, label):
    """Input: rows from burst_*_results.csv — one row per (block, condition)."""
    g = _by_condition(rows)
    print(f"  {label}:  conditions={sorted(g.keys())}  n_rows={len(rows)}")
    for cond, rs in sorted(g.items()):
        wall = [float(r["wall_time_s"]) for r in rs]
        calls = [float(r["advisor_calls"]) for r in rs]
        prec = [float(r.get("trigger_precision_strict", 0)) for r in rs if r.get("trigger_precision_strict") not in (None, "", "nan")]
        p50 = [float(r["latency_p50_ms"]) for r in rs]
        p95 = [float(r["latency_p95_ms"]) for r in rs]
        w_m, w_lo, w_hi = _bootstrap_ci(wall)
        c_m, c_lo, c_hi = _bootstrap_ci(calls, stat=np.mean)
        print(f"    {cond:<28s}  wall={w_m:>6.2f}s [{w_lo:>6.2f},{w_hi:>6.2f}]  "
              f"calls={c_m:>4.1f} [{c_lo:>4.1f},{c_hi:>4.1f}]  "
              f"p95={statistics.median(p95):>6.2f}ms  "
              f"prec={statistics.mean(prec):.3f}" if prec else
              f"    {cond:<28s}  wall={w_m:>6.2f}s  calls={c_m:>4.1f}  p95={statistics.median(p95):.2f}ms")


def b1_burst_large_q():
    print_header("B1  Burst large-Q (Q=3000 per phase)")
    main_fp = REPO / "results" / "burst_end_to_end" / "burst_large_q_results.csv"
    naive_fp = REPO / "results" / "burst_end_to_end" / "burst_large_q_naive_results.csv"
    main_rows = _read_csv(main_fp); naive_rows = _read_csv(naive_fp)
    if not main_rows:
        print("  (missing burst_large_q_results.csv)"); return
    analyse_burst_csv(main_rows, f"θ=0.875 clamp ({len(main_rows)} rows)")
    if naive_rows:
        print(f"\n  Theorem-3-naive control (θ=0.959):")
        analyse_burst_csv(naive_rows, f"θ=0.959 naive ({len(naive_rows)} rows)")


def b2_kernel():
    print_header("B2  Angular vs raw-cosine vs Euclidean kernel ablation")
    fp = OUT / "b2_kernel_ablation" / "kernel_ablation_summary.csv"
    rows = _read_csv(fp)
    if not rows:
        print("  (missing kernel_ablation_summary.csv)"); return
    print(f"  {'kernel':<10s} {'n_w':>5s} {'n_c':>4s} {'θ*':>7s} {'J*':>7s} "
          f"{'TPR':>6s} {'TNR':>6s} {'AUC':>6s}")
    for r in rows:
        print(f"  {r['kernel']:<10s} {r['n_within']:>5s} {r['n_cross']:>4s} "
              f"{float(r['theta_star']):>7.3f} {float(r['J_star']):>7.4f} "
              f"{float(r['TPR']):>6.3f} {float(r['TNR']):>6.3f} {float(r['AUC']):>6.3f}")


def b3_theta_transfer():
    print_header("B3  Bidirectional θ-transfer")
    # Pass A: PG at θ=0.55
    fp_a = REPO / "results" / "burst_end_to_end" / "burst_theta055_results.csv"
    rows_a = _read_csv(fp_a)
    if rows_a:
        print("  Pass A  PostgreSQL @ θ=0.55 (Mongo-optimal on PG):")
        analyse_burst_csv(rows_a, f"PG θ=0.55 ({len(rows_a)} rows)")
    else:
        print("  (Pass A missing)")
    # Pass B: Mongo at θ=0.775 — latest run dir
    mongo_root = REPO.parent / "Version 3" / "code" / "results" / "cross_engine" / "mongo" / "adaptation_theta0775"
    if not mongo_root.exists():
        print("\n  (Pass B: adaptation_theta0775 directory missing)"); return
    runs = sorted([d for d in mongo_root.iterdir() if d.is_dir()])
    if not runs:
        print("\n  (Pass B: no run dirs)"); return
    latest = runs[-1]
    print(f"\n  Pass B  MongoDB @ θ=0.775 (PG-optimal on Mongo):  {latest.name}")
    bm = _read_csv(latest / "block_metrics.csv")
    if not bm:
        print("  (block_metrics.csv not found)"); return
    g = _by_condition(bm, key="strategy")
    for strat, rs in sorted(g.items()):
        calls = [float(r["advisor_calls"]) for r in rs]
        precs = [float(r["precision"]) for r in rs if r.get("precision") not in ("nan", "", None)]
        recs  = [float(r["recall"]) for r in rs if r.get("recall") not in ("nan", "", None)]
        print(f"    {strat:<14s}  calls={statistics.mean(calls):.1f}  "
              f"prec={statistics.mean(precs):.3f}" if precs else
              f"    {strat:<14s}  calls={statistics.mean(calls):.1f}")


def b4_inphase():
    print_header("B4  In-phase amplitude perturbation (F3 receipt)")
    fp = OUT / "b4_inphase" / "inphase_summary.csv"
    rows = _read_csv(fp)
    if not rows:
        print("  (missing inphase_summary.csv)"); return
    print(f"  {'k':>5s} {'S_R':>7s} {'S_V':>7s} {'S_T':>7s} {'S_A':>7s} "
          f"{'S_P':>7s} {'HSM':>7s}")
    for r in rows:
        print(f"  {float(r['k']):>5.1f} {float(r['S_R_mean']):>7.4f} "
              f"{float(r['S_V_mean']):>7.4f} {float(r['S_T_mean']):>7.4f} "
              f"{float(r['S_A_mean']):>7.4f} {float(r['S_P_mean']):>7.4f} "
              f"{float(r['HSM_mean']):>7.4f}")
    print("  expected: S_R & S_T ≡ 1.0 at all k (angular = amplitude-invariant);")
    print("            S_V ↓ as k ≠ 1 (volume-sensitive).")


def b5_noise_weight():
    print_header("B5  Noise × Weight 2-D robustness grid")
    fp = OUT / "b5_noise_weight" / "noise_weight_grid.csv"
    rows = _read_csv(fp)
    if not rows:
        print("  (missing noise_weight_grid.csv)"); return
    print(f"  {'z':>5s} {'δ':>7s} {'θ*':>7s} {'J*':>7s} {'AUC':>6s} "
          f"{'P@0.826':>8s} {'R@0.826':>8s}")
    for r in rows:
        print(f"  {float(r['z']):>5.2f} {float(r['delta']):>7.3f} "
              f"{float(r['theta_star']):>7.3f} {float(r['J_star']):>7.4f} "
              f"{float(r['AUC']):>6.3f} {float(r['precision_0826']):>8.3f} "
              f"{float(r['recall_0826']):>8.3f}")


def manifest():
    fp = OUT / "overnight_manifest.json"
    if not fp.exists():
        print("\n(manifest missing — batch may not have finished)"); return
    m = json.loads(fp.read_text())
    print_header("Manifest")
    print(f"  git_sha: {m['git_sha']}")
    print(f"  start:   {m['start_utc']}")
    print(f"  end:     {m['end_utc']}")
    print(f"  host:    {m.get('host','?')}")
    print(f"  statuses:")
    for k, v in m["statuses"].items():
        print(f"    {k}: {v}")


def main():
    if not OUT.exists():
        print(f"ERROR: {OUT} does not exist. Has the batch run?", file=sys.stderr)
        sys.exit(1)
    b1_burst_large_q()
    b2_kernel()
    b3_theta_transfer()
    b4_inphase()
    b5_noise_weight()
    manifest()
    print("\n")


if __name__ == "__main__":
    main()
