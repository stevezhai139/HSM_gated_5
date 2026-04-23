"""RQ3 — Pareto frontier of (CostSavings, QualityLoss) under θ sweep.

Usage (future):
    python -m code.experiments.cal.rq3_pareto \\
        --advisor dexter \\
        --workload sdss_dr18 \\
        --theta-grid 0.50,0.55,0.60,0.65,0.70,0.75,0.80,0.85,0.90,0.95 \\
        --out results/cal/rq3_dexter_sdss.json

See `Paper3B_Cal_RQs_v0.md` RQ3 for near-knee criterion and binomial test.
"""

from __future__ import annotations

import argparse
import sys


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Paper 3B-Cal RQ3 driver")
    p.add_argument("--advisor", required=True)
    p.add_argument("--workload", required=True)
    p.add_argument(
        "--theta-grid",
        default="0.50,0.55,0.60,0.65,0.70,0.75,0.80,0.85,0.90,0.95",
        help="Comma-separated theta values to sweep",
    )
    p.add_argument("--w-min", type=int, required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--seed", type=int, default=42)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    # TODO(Paper 3B-Cal): implement pipeline
    #   1. For each theta in theta-grid: run RQ2 pipeline, collect
    #      (CostSavings, QualityLoss) at that theta.
    #   2. Call pareto.pareto_front over the 10 points → efficient subset.
    #   3. Call pareto.knee_point → knee coordinates.
    #   4. Compute normalised_distance_to_knee for theta=0.75.
    #   5. Dump frontier + knee + 0.75-distance to --out.
    raise NotImplementedError(
        f"rq3_pareto.main: TODO — scaffolding only. Args parsed: {args}"
    )


if __name__ == "__main__":
    sys.exit(main())
