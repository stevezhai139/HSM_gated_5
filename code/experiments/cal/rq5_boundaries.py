"""RQ5 — Boundary conditions and failure analysis (foundational RQ).

This RQ runs FIRST because it establishes W_min which every other RQ uses.

Usage (future):
    # Window-size sweep to locate W_min
    python -m code.experiments.cal.rq5_boundaries \\
        --mode window_sweep \\
        --workload sdss_dr18 \\
        --window-grid 5,10,20,50,100 \\
        --out results/cal/rq5_window_sweep_sdss.json

    # Failure scenarios
    python -m code.experiments.cal.rq5_boundaries \\
        --mode scenarios \\
        --workload sdss_dr18 \\
        --scenarios micro_shift,syntactic_semantic_mismatch \\
        --out results/cal/rq5_scenarios_sdss.json

See `Paper3B_Cal_RQs_v0.md` RQ5 for broken-stick regression details and the
two failure scenarios F1, F2.
"""

from __future__ import annotations

import argparse
import sys


SCENARIOS = ["micro_shift", "syntactic_semantic_mismatch"]


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Paper 3B-Cal RQ5 driver")
    p.add_argument("--mode", required=True, choices=["window_sweep", "scenarios"])
    p.add_argument("--workload", required=True)
    p.add_argument(
        "--window-grid",
        default="5,10,20,50,100",
        help="Comma-separated window sizes (queries) for W_min sweep",
    )
    p.add_argument(
        "--scenarios",
        default=",".join(SCENARIOS),
        help="Comma-separated failure scenarios to run",
    )
    p.add_argument("--out", required=True)
    p.add_argument("--seed", type=int, default=42)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    # TODO(Paper 3B-Cal):
    #   mode=window_sweep:
    #     1. For each window size in the grid, rebin the workload.
    #     2. Compute gate F1/FPR/FNR at theta=0.75, W0 weights, vs. ground
    #        truth labels.
    #     3. Fit broken-stick regression F1 ~ log(|W|); bootstrap CI on the
    #        break-point location.
    #     4. Dump W_min (point estimate + CI) to --out.
    #   mode=scenarios:
    #     1. For each requested scenario, generate baseline/perturbed
    #        ScenarioWindows.
    #     2. Compute gate predictions; aggregate to per-scenario gate F1.
    #     3. Paired t-test (or Wilcoxon) vs baseline; Cohen's d.
    #     4. Dump scenario-level table to --out.
    raise NotImplementedError(
        f"rq5_boundaries.main: TODO — scaffolding only. Args parsed: {args}"
    )


if __name__ == "__main__":
    sys.exit(main())
