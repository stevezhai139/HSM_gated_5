"""RQ2 — Cost-benefit of HSM upstream gating across advisors × workloads.

Usage (future):
    python -m code.experiments.cal.rq2_cost_benefit \\
        --advisor dexter \\
        --workload sdss_dr18 \\
        --theta 0.75 \\
        --out results/cal/rq2_dexter_sdss.json

See `Paper3B_Cal_RQs_v0.md` RQ2 for the formal hypothesis and success
criterion (≥ 1 of 4 cells: 40 % cost saving ∧ ≤ 10 % quality loss;
Wilcoxon p < 0.05).
"""

from __future__ import annotations

import argparse
import sys


ADVISORS = ["dexter", "supabase_index_advisor"]
WORKLOADS = ["sdss_dr18", "stackoverflow"]


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Paper 3B-Cal RQ2 driver")
    p.add_argument("--advisor", required=True, choices=ADVISORS)
    p.add_argument("--workload", required=True, choices=WORKLOADS)
    p.add_argument("--theta", type=float, default=0.75)
    p.add_argument("--w-min", type=int, required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--seed", type=int, default=42)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    # TODO(Paper 3B-Cal): implement pipeline
    #   1. Instantiate advisor wrapper (Dexter or Supabase) from advisor_wrappers.
    #   2. Load workload windows.
    #   3. Oracle baseline: invoke advisor on every window, record
    #      c_adv(X, W) and Q(X, W) per window.
    #   4. Gated pipeline: only invoke advisor when gate.decide()==1;
    #      inherit previous recommendation otherwise.
    #   5. Compute CostSavings, QualityLoss via metrics.
    #   6. Per-window bootstrap CI + Wilcoxon signed-rank.
    #   7. Dump cell result to --out.
    raise NotImplementedError(
        f"rq2_cost_benefit.main: TODO — scaffolding only. Args parsed: {args}"
    )


if __name__ == "__main__":
    sys.exit(main())
