#!/usr/bin/env python3
"""
build_summary_all.py
--------------------
Aggregate the seven per-workload validation summary CSVs (key/value
format) into a single paper-citable table at
    code/results/summary_all.csv
with one row per (workload, mode) and a canonical column order.

Columns
    workload, mode,
    n_queries, n_windows_ref, n_within_ref, n_cross_ref,
    within_mean, cross_mean,
    DR_median, DR_mean, DR_SD,
    DR_CI_lo, DR_CI_hi,
    MWU_p_median, r_biserial, ICC_2_1,
    dominant_dim, delta_S_R, delta_S_V, delta_S_T, delta_S_A, delta_S_P

Any field a particular workload did not record is left blank — this is
deliberate so the table matches what each script actually measured.

The per-workload CSV schema is heterogeneous (some scripts record
DR_CI_lo_emp/_boot, some just DR_CI_lo).  We pick the empirical CI when
both variants exist because that is what §VI quotes.

Usage
    python scripts/build_summary_all.py
or, from inside code/ :
    python ../scripts/build_summary_all.py
"""

from __future__ import annotations

import csv
import os
import sys
from typing import Dict, List, Tuple

# ── Workload catalogue ──────────────────────────────────────────────────────
# (workload label, mode label, relative-path list to per-workload CSV)
WORKLOADS: List[Tuple[str, str, str]] = [
    ("oltp",            "static",  "results/oltp_validation/oltp_hsm_static.csv"),
    ("oltp",            "execute", "results/oltp_validation/oltp_hsm_execute.csv"),
    ("burst",           "execute", "results/burst_validation/burst_hsm_execute.csv"),
    ("burst_v2",        "execute", "results/burst_v2_validation/burst_v2_hsm_execute.csv"),
    ("burst_v3",        "execute", "results/burst_v3_validation/burst_v3_hsm_execute.csv"),
    ("job",             "execute", "results/job_validation/job_hsm_execute.csv"),
    ("job_complexity",  "execute", "results/job_validation/job_hsm_complexity_execute.csv"),
]

# Canonical output column order.
OUT_COLUMNS = [
    "workload", "mode",
    "n_queries", "n_windows_ref", "n_within_ref", "n_cross_ref",
    "within_mean", "cross_mean",
    "DR_median", "DR_mean", "DR_SD",
    "DR_CI_lo", "DR_CI_hi",
    "MWU_p_median", "r_biserial", "ICC_2_1",
    "dominant_dim",
    "delta_S_R", "delta_S_V", "delta_S_T", "delta_S_A", "delta_S_P",
]

# Aliases from per-workload schemas to the canonical names.
# First match wins, so order matters when a script records both.
ALIASES: Dict[str, List[str]] = {
    "n_queries":      ["n_queries"],
    "n_windows_ref":  ["n_windows_ref", "n_windows"],
    "n_within_ref":   ["n_within_ref", "n_within_pairs"],
    "n_cross_ref":    ["n_cross_ref", "n_cross_pairs"],
    "within_mean":    ["within_mean"],
    "cross_mean":     ["cross_mean"],
    "DR_median":      ["DR_median", "DR"],
    "DR_mean":        ["DR_mean"],
    "DR_SD":          ["DR_sd", "DR_SD"],
    # Prefer empirical CI when available (matches §VI narrative).
    "DR_CI_lo":       ["DR_CI_lo_emp", "DR_CI_lo"],
    "DR_CI_hi":       ["DR_CI_hi_emp", "DR_CI_hi"],
    "MWU_p_median":   ["MWU_p_median", "MWU_p"],
    "r_biserial":     ["r_biserial"],
    "ICC_2_1":        ["ICC_2_1"],
    "dominant_dim":   ["dominant_dim"],
    "delta_S_R":      ["delta_S_R"],
    "delta_S_V":      ["delta_S_V"],
    "delta_S_T":      ["delta_S_T"],
    "delta_S_A":      ["delta_S_A"],
    "delta_S_P":      ["delta_S_P"],
}


def parse_kv_csv(path: str) -> Dict[str, str]:
    """Read a two-column metric,value CSV into a dict (str keys, str values)."""
    out: Dict[str, str] = {}
    with open(path, newline="") as fh:
        reader = csv.reader(fh)
        for row in reader:
            if not row or len(row) < 2:
                continue
            key = row[0].strip()
            if key == "metric":
                continue  # header row
            out[key] = row[1].strip()
    return out


def first_match(d: Dict[str, str], keys: List[str]) -> str:
    for k in keys:
        if k in d and d[k] != "":
            return d[k]
    return ""


def main() -> int:
    # Locate repo root = HSM_gated/
    here = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.dirname(here)
    # Per-workload summary CSVs still live under code/results/*_validation/
    # but the aggregate table belongs next to the figure CSVs at
    # HSM_gated/results/ so a paper build that uses only the repo-top
    # results/ directory has everything it needs.
    results_root = os.path.join(repo_root, "results")
    os.makedirs(results_root, exist_ok=True)

    rows: List[Dict[str, str]] = []
    missing: List[str] = []
    for workload, mode, rel in WORKLOADS:
        abs_path = os.path.join(repo_root, "code", rel)
        if not os.path.exists(abs_path):
            missing.append(rel)
            continue
        kv = parse_kv_csv(abs_path)
        row: Dict[str, str] = {"workload": workload, "mode": mode}
        for canonical, aliases in ALIASES.items():
            row[canonical] = first_match(kv, aliases)
        rows.append(row)

    out_path = os.path.join(results_root, "summary_all.csv")
    with open(out_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=OUT_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in OUT_COLUMNS})

    print(f"[build_summary_all] wrote {out_path}")
    print(f"[build_summary_all]   rows   : {len(rows)}")
    if missing:
        print(f"[build_summary_all]   missing: {len(missing)}")
        for m in missing:
            print(f"    - {m}")

    # Also emit a terse stdout table so it can be eyeballed without
    # opening the CSV.
    print()
    print(f"{'workload':<18}{'mode':<8}{'DR':>9}{'p':>12}{'dom':>6}{'ΔS_T':>9}{'ΔS_A':>9}")
    for row in rows:
        dr   = row.get("DR_median", "") or row.get("DR_mean", "")
        p    = row.get("MWU_p_median", "")
        dom  = row.get("dominant_dim", "")
        dt   = row.get("delta_S_T", "")
        da   = row.get("delta_S_A", "")
        label = f"{row['workload']}/{row['mode']}"
        print(f"{row['workload']:<18}{row['mode']:<8}{dr:>9}{p:>12}{dom:>6}{dt:>9}{da:>9}")

    return 0 if not missing else 2


if __name__ == "__main__":
    sys.exit(main())
