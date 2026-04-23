"""RQ4 — BO offline calibration (a) + EMA online drift-tracker (b).

Usage (future):
    # (a) Offline BO
    python -m code.experiments.cal.rq4_calibration \\
        --mode offline \\
        --advisor dexter --workload sdss_dr18 \\
        --bo-evals 50 --out results/cal/rq4a_dexter_sdss.json

    # (b) Online EMA with injected drift
    python -m code.experiments.cal.rq4_calibration \\
        --mode online \\
        --advisor dexter --workload sdss_dr18 \\
        --drift-start 0.5 --mix-fraction 0.3 \\
        --tracker ema --out results/cal/rq4b_dexter_sdss.json

See `Paper3B_Cal_RQs_v0.md` RQ4 for hypotheses H1a / H1b and success
criteria (≥ 10 % cost-benefit gain in ≥ 1 cell for (a); post-drift F1 ≥ 0.8
on ≥ 90 % windows for (b)).
"""

from __future__ import annotations

import argparse
import sys


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Paper 3B-Cal RQ4 driver")
    p.add_argument("--mode", required=True, choices=["offline", "online"])
    p.add_argument("--advisor", required=True)
    p.add_argument("--workload", required=True)
    p.add_argument("--w-min", type=int, required=True)
    # Offline-only
    p.add_argument("--bo-evals", type=int, default=50)
    p.add_argument(
        "--compare-cma",
        action="store_true",
        help="Also run CMA-ES secondary baseline (RQ4a secondary)",
    )
    # Online-only
    p.add_argument(
        "--drift-start",
        type=float,
        default=0.5,
        help="Fraction of trace at which drift is injected",
    )
    p.add_argument("--mix-fraction", type=float, default=0.3)
    p.add_argument("--tracker", choices=["ema", "none"], default="ema")
    # Common
    p.add_argument("--out", required=True)
    p.add_argument("--seed", type=int, default=42)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    # TODO(Paper 3B-Cal):
    #   mode=offline:
    #     1. Split trace 75/25 train/test.
    #     2. BOCalibrator.run on train → (w*, θ*).
    #     3. (optional) CMAESCalibrator.run on train → (w*_cma, θ*_cma).
    #     4. Evaluate (W0, 0.75), (w*, θ*), optionally (w*_cma, θ*_cma) on test.
    #     5. Paired Wilcoxon on per-window cost-benefit scalar.
    #     6. Report hypervolume convergence curves and L2 distances.
    #   mode=online:
    #     1. Compute BO-calibrated (w*, θ*) on train segment.
    #     2. Inject drift via drift_injection.inject_mix_shift.
    #     3. Run gated pipeline with EMATracker (or NoUpdateBaseline).
    #     4. Per-window F1 vs. ground truth.
    #     5. KS test on post-drift F1 distributions (EMA vs no-update).
    raise NotImplementedError(
        f"rq4_calibration.main: TODO — scaffolding only. Args parsed: {args}"
    )


if __name__ == "__main__":
    sys.exit(main())
