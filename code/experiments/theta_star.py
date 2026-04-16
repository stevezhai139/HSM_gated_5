"""
theta_star.py  --  Closed-form decision threshold per Theorem 3
================================================================

Theorem 3 (Economic Optimality, paper main_article.tex Section IV)
yields the cost-aware optimal gating threshold

        theta*(N, Q) = 1 - Q_min(N) / Q                       (paper Eq. 7)

where

        Q_min(N) = (a * N * log N + b) / (f * N - g * log N)   (paper Eq. 8)

is the break-even query volume at which the advisor's amortised cost equals
its benefit.  Below Q_min(N) the advisor cannot recover its overhead; above
it the gain is positive and proportional to (Q - Q_min).

Constants (a, b, f, g) are workload- and DBMS-specific and are calibrated
from a short profiling run (paper Section IV.B); see calibrate_cost() below.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Tuple


# ---------------------------------------------------------------------------
# Cost model parameters (paper Eq. 8)
# ---------------------------------------------------------------------------
@dataclass
class CostParams:
    """Workload-dependent cost-model coefficients.

    a -- per-row advisor cost coefficient   (sec / row / log row)
    b -- fixed advisor overhead             (sec)
    f -- per-row index benefit coefficient  (sec / row)
    g -- per-row index maintenance cost     (sec / log row)
    """
    a: float
    b: float
    f: float
    g: float


def q_min(n: float, params: CostParams) -> float:
    """Break-even query volume Q_min(N) (paper Eq. 8).

    Q_min(N) = (a*N*log N + b) / (f*N - g*log N).

    Returns +inf when the denominator is non-positive (the advisor cannot
    pay for itself at this cardinality, regardless of Q).
    """
    if n <= 1.0:
        return float("inf")
    log_n = math.log(n)
    denom = params.f * n - params.g * log_n
    if denom <= 0:
        return float("inf")
    return (params.a * n * log_n + params.b) / denom


def theta_star(n: float, q: float, params: CostParams) -> float:
    """Closed-form optimal gating threshold theta*(N, Q) (paper Eq. 7).

    theta*(N, Q) = 1 - Q_min(N) / Q,  clipped to [0, 1].

    When Q <= Q_min(N) the advisor has no positive expected benefit; we
    return 0.0 so the gate never triggers.  When Q is far above the
    break-even point, theta* approaches 1.0 and the gate triggers
    aggressively.
    """
    if q <= 0:
        return 0.0
    qm = q_min(n, params)
    if not math.isfinite(qm) or qm >= q:
        return 0.0
    raw = 1.0 - qm / q
    return max(0.0, min(1.0, raw))


# ---------------------------------------------------------------------------
# Calibration helper
# ---------------------------------------------------------------------------
def calibrate_cost(
    samples: Iterable[Tuple[float, float, float, float]],
) -> CostParams:
    """Fit (a, b, f, g) from a profiling sweep.

    Each sample is (N, advisor_time_s, baseline_query_time_s,
    indexed_query_time_s).  We use ordinary least squares on:

        advisor_time_s = a * N * log(N) + b           (advisor cost)
        baseline - indexed = f * 1 - g * log(N)/N      (per-query gain)

    For the simplest calibration we fit two univariate regressions; richer
    multi-workload calibration (paper Section IV.B) is left to the
    experiment_runner.
    """
    import numpy as np

    samples = list(samples)
    n_arr     = np.array([s[0] for s in samples], dtype=float)
    adv_arr   = np.array([s[1] for s in samples], dtype=float)
    base_arr  = np.array([s[2] for s in samples], dtype=float)
    idx_arr   = np.array([s[3] for s in samples], dtype=float)
    log_n     = np.log(n_arr)

    # Advisor cost regression: adv = a * N log N + b
    x_adv = np.column_stack([n_arr * log_n, np.ones_like(n_arr)])
    a_hat, b_hat = np.linalg.lstsq(x_adv, adv_arr, rcond=None)[0]

    # Per-query gain regression: gain = f - g * log(N) / N
    gain  = base_arr - idx_arr
    x_gain = np.column_stack([np.ones_like(n_arr), -log_n / n_arr])
    f_hat, g_hat = np.linalg.lstsq(x_gain, gain, rcond=None)[0]

    return CostParams(a=float(a_hat), b=float(b_hat),
                      f=float(f_hat), g=float(g_hat))


# ---------------------------------------------------------------------------
# Default parameters used as a placeholder for cold-start scenarios.
# These should be overridden via calibrate_cost() once profiling samples are
# available.  The values below are illustrative only and were chosen so
# that Q_min(1e6) ~ 100, matching the order of magnitude reported in
# Section IV-D of the paper.
# ---------------------------------------------------------------------------
DEFAULT_PARAMS = CostParams(a=1e-7, b=0.5, f=1e-3, g=5e-4)


if __name__ == "__main__":
    for n in (1e3, 1e4, 1e5, 1e6, 1e7):
        for q in (10, 100, 1000, 10000):
            qm = q_min(n, DEFAULT_PARAMS)
            th = theta_star(n, q, DEFAULT_PARAMS)
            print(f"N={n:>9.0f}  Q={q:>6}  Qmin={qm:>10.2f}  theta*={th:.3f}")
