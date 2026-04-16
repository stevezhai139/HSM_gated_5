#!/usr/bin/env python3
"""
aggregate_throughput.py
-----------------------
Pool per-SF raw results from experiment_runner.py into a per-SF
normalized table for plot_fig07_throughput_comparison.py.

Input  : results/sf<X>/raw_results.csv   (one per SF)
Output : results/throughput.csv          with columns
           sf, config, qps_raw, qps_norm, ci_low_norm, ci_high_norm, n

Normalization policy:
  * `wall_qps` is used when present (includes advisor overhead),
    falling back to `qps`;
  * rows where errors > 0 are dropped;
  * for each (sf, block), each condition's qps is divided by the
    same-block baseline (no_advise) qps  →  paired within-block ratio
    that cancels out workload-trace variance;
  * a percentile bootstrap on those per-block ratios (B=2000, seed=42)
    yields a 95% CI on the mean fraction of baseline throughput.

The baseline (no_advise) row is included for completeness with
ratio = 1.0 by construction; downstream plot omits it from the
grouped bars and shows it as a reference line.
"""

from __future__ import annotations

import os
import sys
from glob import glob

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.normpath(os.path.join(HERE, os.pardir))

RESULTS_DIR = os.path.join(REPO, "results")
OUT_CSV     = os.path.join(RESULTS_DIR, "throughput.csv")

# Paper-consistent display order.
CONFIG_ORDER = ["baseline", "periodic", "always_on", "hsm_gated"]
CONFIG_LABEL = {
    "baseline":  "no_advise",
    "periodic":  "periodic",
    "always_on": "always_on",
    "hsm_gated": "HSM_gated",
}


def _bootstrap_ci(x: np.ndarray, B: int = 2000, alpha: float = 0.05,
                  seed: int = 42) -> tuple[float, float]:
    if len(x) == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), size=(B, len(x)))
    means = x[idx].mean(axis=1)
    lo = float(np.percentile(means, 100.0 * alpha / 2))
    hi = float(np.percentile(means, 100.0 * (1.0 - alpha / 2)))
    return lo, hi


def _qps_column(df: pd.DataFrame) -> np.ndarray:
    """Prefer wall_qps (includes advisor overhead); fall back to qps."""
    wall = pd.to_numeric(df.get("wall_qps"), errors="coerce")
    base = pd.to_numeric(df.get("qps"),      errors="coerce")
    return wall.where(wall.notna(), base).to_numpy(dtype=float)


def _load_raw() -> pd.DataFrame:
    pattern = os.path.join(RESULTS_DIR, "sf*", "raw_results.csv")
    files   = sorted(glob(pattern))
    if not files:
        print(f"[aggregate_throughput] no input found at {pattern}",
              file=sys.stderr)
        sys.exit(2)
    frames: list[pd.DataFrame] = []
    for f in files:
        try:
            df = pd.read_csv(f)
        except Exception as exc:
            print(f"[aggregate_throughput] skip {f}: {exc}", file=sys.stderr)
            continue
        df["__file__"] = f
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def main() -> None:
    df = _load_raw()
    errs = pd.to_numeric(df.get("errors", 0), errors="coerce").fillna(0)
    df = df[errs == 0].copy()
    df["__qps__"] = _qps_column(df)
    df = df.dropna(subset=["__qps__"])
    df["sf"]    = pd.to_numeric(df["sf"],    errors="coerce")
    df["block"] = pd.to_numeric(df["block"], errors="coerce")
    df = df.dropna(subset=["sf", "block"])

    rows = []
    for sf in sorted(df["sf"].unique()):
        sf_df = df[df["sf"] == sf]
        # Pivot to (block × condition) for paired ratios.
        wide = sf_df.pivot_table(
            index="block", columns="condition", values="__qps__",
            aggfunc="mean",
        )
        if "baseline" not in wide.columns:
            print(f"[aggregate_throughput] SF={sf}: no baseline rows; skip",
                  file=sys.stderr)
            continue
        baseline_per_block = wide["baseline"]

        for cond in CONFIG_ORDER:
            if cond not in wide.columns:
                continue
            cond_per_block = wide[cond]
            paired = pd.concat(
                [cond_per_block, baseline_per_block], axis=1
            ).dropna()
            paired.columns = ["cond", "base"]
            if paired.empty:
                continue
            ratios = (paired["cond"] / paired["base"]).to_numpy(dtype=float)
            mean_raw   = float(cond_per_block.dropna().mean())
            mean_ratio = float(np.mean(ratios))
            lo, hi     = _bootstrap_ci(ratios)
            rows.append({
                "sf":            float(sf),
                "config":        CONFIG_LABEL[cond],
                "qps_raw":       round(mean_raw,   4),
                "qps_norm":      round(mean_ratio, 4),
                "ci_low_norm":   round(lo, 4),
                "ci_high_norm":  round(hi, 4),
                "n":             int(len(ratios)),
            })

    if not rows:
        print("[aggregate_throughput] no valid rows after filtering",
              file=sys.stderr)
        sys.exit(3)

    out = pd.DataFrame(rows)
    out.to_csv(OUT_CSV, index=False)
    print(f"[aggregate_throughput] wrote {OUT_CSV} ({len(out)} rows)\n")
    for sf in sorted(out["sf"].unique()):
        print(f"  SF={sf}")
        for _, r in out[out["sf"] == sf].iterrows():
            print(f"    {r['config']:10s}  "
                  f"qps_raw={r['qps_raw']:8.3f}  "
                  f"norm={r['qps_norm']:.3f}  "
                  f"95%CI=[{r['ci_low_norm']:.3f}, {r['ci_high_norm']:.3f}]  "
                  f"n={r['n']}")


if __name__ == "__main__":
    main()
