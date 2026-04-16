#!/usr/bin/env python3
"""
b4_inphase_perturbation.py — F3 receipt: angular kernel preserves in-phase structure.

Synthetic test. For each of 10 seeded blocks:
  1. Generate a burst-mid-Q window trace (same generator as Mid-Q rerun).
  2. Pick the first within-phase adjacent pair (W_i, W_{i+1}).
  3. Construct W'_{i+1} by scaling every template frequency by a factor k,
     WITHOUT changing rank order or template identity ("pure amplitude
     perturbation, in-phase structure preserved").
  4. Compute S_R, S_V, S_T, S_A, S_P, HSM for (W_i, W'_{i+1}) at
     k ∈ {0.5, 1.0, 2.0, 5.0, 10.0}.

Expected outcome (the receipt):
  * S_R (angular, rank-based) ≡ 1.000 at every k (Spearman ρ invariant to
    monotonic scaling).
  * S_T (angular, unit-vector cos) ≡ 1.000 at every k (cos θ = 1 for
    parallel vectors regardless of magnitude).
  * S_V changes (amplitude-sensitive) — drops from 1.0 as k ≠ 1.
  * S_A ≡ 1.000 (Jaccard on identical sets).
  * S_P varies because bucketed qps series changes shape.

This separates "in-phase amplitude" (should NOT fire) from "phase drift"
(should fire) — the kernel's built-in amplitude-invariance on S_R and S_T.

OUTPUT:  results/overnight_2026-04-16/b4_inphase/
    inphase_amplitude.csv  (block, k, S_R, S_V, S_T, S_A, S_P, HSM)
    inphase_summary.csv    (k, S_R_mean, S_R_sd, ..., HSM_mean, HSM_sd)

Runtime: ~1 min.
"""
from __future__ import annotations

import csv
import sys
import statistics
from collections import Counter
from copy import deepcopy
from pathlib import Path

HERE = Path(__file__).resolve()
REPO = HERE.parents[3]
sys.path.insert(0, str(REPO / "code" / "experiments"))
sys.path.insert(0, str(REPO / "code" / "experiments" / "tier2"))

from hsm_similarity import build_window, hsm_score  # noqa: E402
from hsm_burst_end_to_end import build_query_pool, build_trace  # noqa: E402

QUERIES_PER_PHASE = 700
WINDOW_SIZE = 5
N_BLOCKS = 10
K_FACTORS = [0.5, 1.0, 2.0, 5.0, 10.0]

OUT_DIR = REPO / "results" / "overnight_2026-04-16" / "b4_inphase"


def scale_window_frequencies(window, k):
    """Return a new WorkloadWindow whose query list is scaled by factor k,
    preserving the template proportions exactly.

    For integer k: queries are replicated k times.
    For fractional k (e.g. 0.5): queries are truncated/extended proportionally.
    Rank order and direction of the frequency vector are preserved, so
    S_R (Spearman) and S_T (cos θ) should stay at 1.0. S_V drops because
    the query count changes. S_A stays at 1.0 (same tables/columns). S_P
    varies because the bucketed arrival series depends on timing.
    """
    src = list(window.queries)
    n_src = len(src)
    if n_src == 0:
        return deepcopy(window)
    n_new = max(1, int(round(k * n_src)))
    if n_new >= n_src:
        # Replicate: queries × (n_new // n_src) + prefix
        reps = n_new // n_src
        new_queries = []
        for _ in range(reps):
            new_queries.extend(deepcopy(q) for q in src)
        new_queries.extend(deepcopy(q) for q in src[: n_new - reps * n_src])
    else:
        # Subsample uniformly in index-space: preserves template proportions
        stride = n_src / n_new
        new_queries = [deepcopy(src[int(i * stride)]) for i in range(n_new)]
    from hsm_similarity import WorkloadWindow
    return WorkloadWindow(queries=new_queries, window_id=window.window_id,
                          duration_s=window.duration_s)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []

    for block in range(1, N_BLOCKS + 1):
        pool = build_query_pool(seed=block)
        trace = build_trace(pool, seed=block)
        # First two windows — always within Phase 1.
        w0_sqls = [r[3] for r in trace[0:WINDOW_SIZE]]
        w1_sqls = [r[3] for r in trace[WINDOW_SIZE:2 * WINDOW_SIZE]]
        w0 = build_window(w0_sqls, window_id=0)
        w1 = build_window(w1_sqls, window_id=1)
        for k in K_FACTORS:
            w1_scaled = scale_window_frequencies(w1, k)
            hsm, dims = hsm_score(w0, w1_scaled)
            rows.append({
                "block": block, "k": k,
                "S_R": round(dims["S_R"], 6), "S_V": round(dims["S_V"], 6),
                "S_T": round(dims["S_T"], 6), "S_A": round(dims["S_A"], 6),
                "S_P": round(dims["S_P"], 6), "HSM": round(hsm, 6),
            })
            print(f"  block={block:02d}  k={k:>4.1f}  "
                  f"S_R={dims['S_R']:.4f}  S_V={dims['S_V']:.4f}  "
                  f"S_T={dims['S_T']:.4f}  S_A={dims['S_A']:.4f}  "
                  f"S_P={dims['S_P']:.4f}  HSM={hsm:.4f}")

    detail_fp = OUT_DIR / "inphase_amplitude.csv"
    with detail_fp.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"\nwrote {detail_fp}")

    # Summary stats per k
    summary = []
    for k in K_FACTORS:
        subset = [r for r in rows if r["k"] == k]
        def ms(field):
            vals = [r[field] for r in subset]
            return statistics.mean(vals), (statistics.stdev(vals) if len(vals) > 1 else 0.0)
        sr_m, sr_s = ms("S_R"); sv_m, sv_s = ms("S_V")
        st_m, st_s = ms("S_T"); sa_m, sa_s = ms("S_A")
        sp_m, sp_s = ms("S_P"); hsm_m, hsm_s = ms("HSM")
        summary.append({
            "k": k, "n": len(subset),
            "S_R_mean": round(sr_m, 4), "S_R_sd": round(sr_s, 4),
            "S_V_mean": round(sv_m, 4), "S_V_sd": round(sv_s, 4),
            "S_T_mean": round(st_m, 4), "S_T_sd": round(st_s, 4),
            "S_A_mean": round(sa_m, 4), "S_A_sd": round(sa_s, 4),
            "S_P_mean": round(sp_m, 4), "S_P_sd": round(sp_s, 4),
            "HSM_mean": round(hsm_m, 4), "HSM_sd": round(hsm_s, 4),
        })
    summary_fp = OUT_DIR / "inphase_summary.csv"
    with summary_fp.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
        w.writeheader(); w.writerows(summary)
    print(f"wrote {summary_fp}")


if __name__ == "__main__":
    main()
