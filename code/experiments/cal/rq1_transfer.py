"""RQ1 — Threshold transferability (TPC-H → SDSS DR18).

Usage (future):
    python -m code.experiments.cal.rq1_transfer \\
        --workload sdss_dr18 \\
        --w-min 20 \\
        --out results/cal/rq1.json

See `Paper3B_Cal_RQs_v0.md` RQ1 for the formal hypothesis and success
criterion (FPR ≤ 0.10 ∧ FNR ≤ 0.05; McNemar vs Indexer++ baseline).
"""

from __future__ import annotations

import argparse
import sys


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Paper 3B-Cal RQ1 driver")
    p.add_argument("--workload", required=True, choices=["sdss_dr18", "stackoverflow"])
    p.add_argument("--w-min", type=int, required=True, help="Window size (queries)")
    p.add_argument("--theta", type=float, default=0.75)
    p.add_argument("--indexerpp-predictions", help="Path to Indexer++ predictions for McNemar")
    p.add_argument("--out", required=True, help="Output JSON path")
    p.add_argument("--seed", type=int, default=42)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    # TODO(Paper 3B-Cal): implement pipeline
    #   1. Load workload into windows of size --w-min.
    #   2. Compute K_w(W_i, W_{i+1}) for all adjacent pairs using W0 weights.
    #   3. Derive gate predictions at theta.
    #   4. Compute performance-clustering ground truth (k-means on per-query
    #      index-quality vectors, silhouette ≥ 0.4 criterion).
    #   5. Call metrics.fpr_fnr, bootstrap_ci.
    #   6. If --indexerpp-predictions given: metrics.mcnemar_test.
    #   7. Dump results as JSON to --out.
    raise NotImplementedError(
        f"rq1_transfer.main: TODO — scaffolding only. Args parsed: {args}"
    )


if __name__ == "__main__":
    sys.exit(main())
