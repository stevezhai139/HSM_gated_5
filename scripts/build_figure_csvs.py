#!/usr/bin/env python3
"""
build_figure_csvs.py
--------------------
Assemble the per-workload raw pair-score CSVs (emitted by the seven
validation scripts) into the figure-schema files expected by

    code/figures/plot_fig01_score_distribution.py      (score, group)
    code/figures/plot_fig03_within_cross_phase.py      (workload, group, score)

Inputs scanned under code/results/ :
    oltp_validation/oltp_hsm_execute_pair_scores.csv
    burst_validation/burst_hsm_pair_scores.csv
    burst_v2_validation/burst_v2_hsm_pair_scores.csv
    burst_v3_validation/burst_v3_hsm_pair_scores.csv
    job_validation/job_hsm_execute_pair_scores.csv
    job_validation/job_hsm_complexity_execute_pair_scores.csv
    sdss_validation/sdss_hsm_pair_scores.csv

Outputs written to HSM_gated/results/ :
    score_distribution.csv     pooled within/cross across all workloads
    within_cross_phase.csv     two-workload view (workload ∈ {tpch, sdss})
    trigger_timeseries.csv     consecutive-window HSM score + gate flag
                               (from the OLTP execute reference seed)

Workload mapping for fig03 (the figure uses TPC-H-style read-heavy vs
SDSS analytical):
    tpch  ← job_execute          (IMDB / JOB is our TPC-H analogue)
    sdss  ← sdss

Any missing inputs are silently skipped — the figure scripts fall back
to the watermarked placeholder when their CSV is absent.
"""

from __future__ import annotations

import csv
import os
import sys
from typing import List, Optional, Tuple

PAIR_CSV_PATHS: List[Tuple[str, str, str]] = [
    # (repo-relative path,                      workload-label, group-prefix)
    ("results/oltp_validation/oltp_hsm_execute_pair_scores.csv",                 "oltp_execute",           ""),
    ("results/burst_validation/burst_hsm_pair_scores.csv",                       "burst",                  ""),
    ("results/burst_v2_validation/burst_v2_hsm_pair_scores.csv",                 "burst_v2",               ""),
    ("results/burst_v3_validation/burst_v3_hsm_pair_scores.csv",                 "burst_v3",               ""),
    ("results/job_validation/job_hsm_execute_pair_scores.csv",                   "job_execute",            ""),
    ("results/job_validation/job_hsm_complexity_execute_pair_scores.csv",        "job_complexity_execute", ""),
    ("results/sdss_validation/sdss_hsm_pair_scores.csv",                         "sdss",                   ""),
]

# Mapping used for within_cross_phase.csv
# (source workload label as emitted in pair-score CSV) -> (figure label)
FIG03_MAP = {
    "job_execute": "tpch",
    "sdss":        "sdss",
}


def read_pair_csv(path: str) -> Optional[List[Tuple[str, str, str]]]:
    """Return list of (workload, group, score) rows, or None if missing."""
    if not os.path.exists(path):
        return None
    rows: List[Tuple[str, str, str]] = []
    with open(path, newline="") as fh:
        reader = csv.reader(fh)
        header = next(reader, None)
        for row in reader:
            if len(row) < 3:
                continue
            rows.append((row[0], row[1], row[2]))
    return rows


def main() -> int:
    here = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.dirname(here)
    code_dir = os.path.join(repo_root, "code")
    # Figure scripts resolve CSV_REL = "results/..." against repo_root,
    # so the aggregated figure CSVs must live at HSM_gated/results/*.csv
    # (not code/results/).  See regenerate_all_figures.py for the contract.
    results_dir = os.path.join(repo_root, "results")
    os.makedirs(results_dir, exist_ok=True)

    pooled: List[Tuple[str, str, str]] = []
    fig03: List[Tuple[str, str, str]] = []
    missing: List[str] = []
    present: List[str] = []

    for rel, label, _ in PAIR_CSV_PATHS:
        abs_path = os.path.join(code_dir, rel)
        rows = read_pair_csv(abs_path)
        if rows is None:
            missing.append(rel)
            continue
        present.append(rel)
        for w, g, s in rows:
            pooled.append((w, g, s))
            if w in FIG03_MAP:
                fig03.append((FIG03_MAP[w], g, s))

    # score_distribution.csv — (score, group) pooled across workloads.
    out1 = os.path.join(results_dir, "score_distribution.csv")
    with open(out1, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["score", "group"])
        for _, g, s in pooled:
            writer.writerow([s, g])
    print(f"[build_figure_csvs] wrote {out1}  ({len(pooled)} rows)")

    # within_cross_phase.csv — (workload, group, score) for fig03.
    out3 = os.path.join(results_dir, "within_cross_phase.csv")
    with open(out3, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["workload", "group", "score"])
        for w, g, s in fig03:
            writer.writerow([w, g, s])
    print(f"[build_figure_csvs] wrote {out3}  ({len(fig03)} rows)")

    if present:
        print(f"[build_figure_csvs]   present ({len(present)}):")
        for p in present:
            print(f"    + {p}")
    if missing:
        print(f"[build_figure_csvs]   missing ({len(missing)}):")
        for m in missing:
            print(f"    - {m}")

    # trigger_timeseries.csv — copied from a workload whose consecutive
    # windows span both sides of θ=0.75 (paper §IV default).  Priority:
    #   1. burst_v2  — phase-marked synthetic trace designed to alternate
    #      Steady / Burst_Alt / Burst_Grp, giving clear triggered vs.
    #      not-triggered transitions.
    #   2. oltp static — fallback: θ cuts between within_mean=0.897 and
    #      cross_mean=0.780 so some windows still trigger.
    #   3. oltp execute — last resort: in EXECUTE mode S_P depresses every
    #      consecutive score below 0.75 (all triggered, illustration flat).
    trig_candidates = [
        os.path.join(code_dir, "results", "burst_v2_validation",
                     "burst_v2_hsm_trigger_timeseries.csv"),
        os.path.join(code_dir, "results", "oltp_validation",
                     "oltp_hsm_static_trigger_timeseries.csv"),
        os.path.join(code_dir, "results", "oltp_validation",
                     "oltp_hsm_execute_trigger_timeseries.csv"),
    ]
    trig_in = next((p for p in trig_candidates if os.path.exists(p)),
                   trig_candidates[-1])
    trig_out = os.path.join(results_dir, "trigger_timeseries.csv")
    if os.path.exists(trig_in):
        # Re-emit with only the columns plot_fig02 requires to keep schema
        # tight: window_idx, score, gate_triggered.
        with open(trig_in, newline="") as src, \
             open(trig_out, "w", newline="") as dst:
            reader = csv.DictReader(src)
            writer = csv.writer(dst)
            writer.writerow(["window_idx", "score", "gate_triggered"])
            n = 0
            for row in reader:
                writer.writerow([row["window_idx"], row["score"],
                                 row["gate_triggered"]])
                n += 1
        print(f"[build_figure_csvs] wrote {trig_out}  ({n} rows)")
    else:
        print(f"[build_figure_csvs] skip trigger_timeseries.csv"
              f"  (missing: {trig_in})")

    return 0


if __name__ == "__main__":
    sys.exit(main())
