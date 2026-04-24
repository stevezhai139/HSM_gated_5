"""RQ1 — Threshold transferability (TPC-H → SDSS DR18).

Research question (verbatim from Paper3B_Cal_RQs_v0.md §RQ1):

    Does the HSM similarity threshold θ* = 0.75, calibrated on TPC-H,
    transfer to the live SDSS DR18 query log without per-workload
    recalibration, while preserving classification quality?

Hypotheses:

- **H0 (null):**   FPR > 0.10 OR FNR > 0.05 on held-out SDSS pairs.
- **H1 (alt.):**   FPR ≤ 0.10 AND FNR ≤ 0.05 on held-out SDSS pairs.

Implementation notes:

- Ground truth ``y_true`` is derived from Paper 3A's stored phase labels:
  ``y_true = 1`` iff ``phase_a != phase_b`` (shift as positive class).
- Prediction ``y_pred`` is Paper 3A's stored gate decision at θ=0.75 with
  fixed W₀ = (0.25, 0.20, 0.20, 0.20, 0.15). This IS the "transferred"
  threshold — no recalibration.
- Bootstrap 95% CI on FPR and FNR (10,000 resamples).
- McNemar vs Indexer++ baseline is SKIPPED when predictions are not
  available (Indexer++ not vendored yet). The primary FPR/FNR comparison
  stands on its own per RQ1 success criterion.

Usage:

    # Primary: SDSS DR18 (using stored Paper 3A CSV)
    python code/experiments/cal/rq1_transfer.py \\
        --workload sdss --w-min 20 \\
        --out results/cal/rq1_sdss.json

See ``Paper3B_Cal_RQs_v0.md`` §RQ1 for full spec.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

# sys.path fixup so `cal.*` imports work when run as a script
_THIS = Path(__file__).resolve()
_EXPERIMENTS_DIR = _THIS.parents[1]
if str(_EXPERIMENTS_DIR) not in sys.path:
    sys.path.insert(0, str(_EXPERIMENTS_DIR))

from cal.validation.paper3a_loader import load_experiment  # noqa: E402
from cal.metrics import fpr_fnr, bootstrap_ci  # noqa: E402


# ---------------------------------------------------------------------------
# RQ1 success criterion
# ---------------------------------------------------------------------------

#: FPR upper bound for H1 (from Paper3B_Cal_RQs_v0.md §RQ1)
FPR_THRESHOLD = 0.10

#: FNR upper bound for H1 (from Paper3B_Cal_RQs_v0.md §RQ1)
FNR_THRESHOLD = 0.05


def _evaluate_transfer(
    y_true: np.ndarray, y_pred: np.ndarray, *, n_bootstrap: int = 10_000,
    confidence: float = 0.95, seed: int = 42,
) -> Dict[str, Any]:
    """Compute FPR, FNR, bootstrap 95% CI, and RQ1 success boolean."""
    rates = fpr_fnr(y_true.tolist(), y_pred.tolist())

    # Bootstrap CI: resample pair indices with replacement; recompute FPR/FNR.
    rng = np.random.default_rng(seed)
    n = len(y_true)
    fpr_samples = np.empty(n_bootstrap)
    fnr_samples = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        r = fpr_fnr(y_true[idx].tolist(), y_pred[idx].tolist())
        fpr_samples[i] = r.fpr
        fnr_samples[i] = r.fnr

    alpha_tail = (1 - confidence) / 2
    fpr_ci = (
        float(np.quantile(fpr_samples, alpha_tail)),
        float(np.quantile(fpr_samples, 1 - alpha_tail)),
    )
    fnr_ci = (
        float(np.quantile(fnr_samples, alpha_tail)),
        float(np.quantile(fnr_samples, 1 - alpha_tail)),
    )

    # H1 success: BOTH FPR and FNR below thresholds, AND upper CI bound also
    # below threshold (stronger criterion — CI does not cross the threshold).
    fpr_meets = rates.fpr <= FPR_THRESHOLD
    fnr_meets = rates.fnr <= FNR_THRESHOLD
    fpr_ci_tight = fpr_ci[1] <= FPR_THRESHOLD
    fnr_ci_tight = fnr_ci[1] <= FNR_THRESHOLD

    return {
        "n_pairs": int(n),
        "n_transitions": int(y_true.sum()),
        "n_triggered": int(y_pred.sum()),
        "fpr": rates.fpr,
        "fnr": rates.fnr,
        "precision": rates.precision,
        "recall": rates.recall,
        "f1": rates.f1,
        "fpr_ci_95": fpr_ci,
        "fnr_ci_95": fnr_ci,
        "fpr_threshold": FPR_THRESHOLD,
        "fnr_threshold": FNR_THRESHOLD,
        "fpr_meets_threshold": bool(fpr_meets),
        "fnr_meets_threshold": bool(fnr_meets),
        "fpr_ci_strictly_below_threshold": bool(fpr_ci_tight),
        "fnr_ci_strictly_below_threshold": bool(fnr_ci_tight),
        "h1_point_estimate": bool(fpr_meets and fnr_meets),
        "h1_strict_ci": bool(fpr_ci_tight and fnr_ci_tight),
    }


def _run_on_paper3a_csv(
    experiment_key: str, *, n_bootstrap: int = 10_000, seed: int = 42,
) -> Dict[str, Any]:
    """Load Paper 3A's stored gate decisions at θ=0.75 with default W₀ and
    evaluate classification rates vs phase-transition ground truth."""
    series = load_experiment(experiment_key)
    y_true = np.asarray(series.is_transition, dtype=int)
    y_pred = np.asarray(series.default_triggered, dtype=int)

    result = _evaluate_transfer(
        y_true, y_pred, n_bootstrap=n_bootstrap, seed=seed,
    )
    result["experiment_key"] = experiment_key
    result["csv_path"] = series.csv_path
    result["phases_seen"] = list(series.phases_seen)
    return result


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Paper 3B-Cal RQ1 driver — threshold transferability",
    )
    p.add_argument(
        "--workload", default="sdss",
        choices=["sdss", "job_static", "oltp_static", "oltp_execute", "burst_v2"],
        help="Paper 3A experiment key (default: sdss)",
    )
    p.add_argument(
        "--w-min", type=int, default=20,
        help="Minimum window size (default 20, from RQ5 smoke analysis)",
    )
    p.add_argument("--theta", type=float, default=0.75,
                   help="Similarity threshold (default: 0.75, Paper 3A default)")
    p.add_argument("--n-bootstrap", type=int, default=10_000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", required=True, help="Output JSON path")
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv if argv is not None else sys.argv[1:])
    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = _THIS.parents[3] / args.out

    t0 = time.monotonic()
    print(f"[rq1] loading Paper 3A CSV for workload={args.workload!r}...")
    result = _run_on_paper3a_csv(
        args.workload, n_bootstrap=args.n_bootstrap, seed=args.seed,
    )
    elapsed = time.monotonic() - t0

    payload = {
        "workload": args.workload,
        "theta": args.theta,
        "w_min": args.w_min,
        "seed": args.seed,
        "n_bootstrap": args.n_bootstrap,
        "elapsed_seconds": elapsed,
        **result,
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as fh:
        json.dump(payload, fh, indent=2, default=float)

    print()
    print(f"=== RQ1 Transfer Result ({args.workload}, θ={args.theta}) ===")
    print(f"  n_pairs      : {result['n_pairs']}")
    print(f"  n_transitions: {result['n_transitions']}")
    print(f"  FPR = {result['fpr']:.4f}  (95% CI: [{result['fpr_ci_95'][0]:.4f}, {result['fpr_ci_95'][1]:.4f}])  threshold={FPR_THRESHOLD}")
    print(f"  FNR = {result['fnr']:.4f}  (95% CI: [{result['fnr_ci_95'][0]:.4f}, {result['fnr_ci_95'][1]:.4f}])  threshold={FNR_THRESHOLD}")
    print(f"  Precision = {result['precision']:.4f}")
    print(f"  Recall    = {result['recall']:.4f}")
    print(f"  F1        = {result['f1']:.4f}")
    print()
    h1_point = "✅" if result["h1_point_estimate"] else "❌"
    h1_strict = "✅" if result["h1_strict_ci"] else "❌"
    print(f"  H1 (point estimate FPR≤{FPR_THRESHOLD} AND FNR≤{FNR_THRESHOLD}): {h1_point}")
    print(f"  H1 (strict CI upper bound ≤ thresholds):                  {h1_strict}")
    print()
    print(f"[rq1] wrote {out_path} ({elapsed:.1f}s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
