"""Evaluation metrics and statistical tests for Paper 3B-Cal RQs.

Consolidates: classification metrics (RQ1, RQ5), paired non-parametric tests
(RQ2, RQ4), effect-size computations, and bootstrap confidence intervals.

Statistical-test choices are locked in `Paper3B_Cal_RQs_v0.md`:
- RQ1 primary — one-sided McNemar vs. Indexer++ baseline.
- RQ2 primary — one-sided Wilcoxon signed-rank on per-window cost savings.
- RQ3 primary — binomial + Clopper-Pearson CI on near-knee frequency.
- RQ4a primary — paired Wilcoxon.
- RQ4b primary — two-sample Kolmogorov-Smirnov on post-drift F1.
- RQ5 primary — broken-stick regression bootstrap (implemented separately
  in `rq5_boundaries.py`; not generic).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class ClassificationRates:
    """Standard confusion-matrix-derived rates."""

    fpr: float
    fnr: float
    precision: float
    recall: float
    f1: float


def fpr_fnr(y_true: Sequence[int], y_pred: Sequence[int]) -> ClassificationRates:
    """Return FPR, FNR, precision, recall, F1 for binary classification.

    Parameters
    ----------
    y_true, y_pred : sequence of {0, 1}

    Notes
    -----
    For RQ1: y_true is derived from performance-clustering ground truth
    (see `rq1_transfer.py`); y_pred is the gate decision.
    """
    raise NotImplementedError(
        "fpr_fnr: TODO — compute TP/FP/TN/FN and derived rates; guard "
        "against divide-by-zero when a class has no examples."
    )


def mcnemar_test(
    y_true: Sequence[int],
    y_pred_a: Sequence[int],
    y_pred_b: Sequence[int],
    alternative: str = "two-sided",
) -> tuple[float, float]:
    """McNemar test on paired binary predictions.

    Returns
    -------
    statistic, p_value

    Notes
    -----
    Implemented via `statsmodels.stats.contingency_tables.mcnemar` once
    that dependency is added. Exact binomial test is preferred for small
    discordant-cell counts.
    """
    raise NotImplementedError("mcnemar_test: TODO — wrap statsmodels.")


def wilcoxon_paired(
    x: Sequence[float],
    y: Sequence[float],
    alternative: str = "two-sided",
) -> tuple[float, float]:
    """One-sided Wilcoxon signed-rank test on paired samples x - y."""
    raise NotImplementedError("wilcoxon_paired: TODO — wrap scipy.stats.wilcoxon.")


def ks_test(
    sample_a: Sequence[float],
    sample_b: Sequence[float],
) -> tuple[float, float]:
    """Two-sample Kolmogorov-Smirnov test."""
    raise NotImplementedError("ks_test: TODO — wrap scipy.stats.ks_2samp.")


def cohens_d_paired(x: Sequence[float], y: Sequence[float]) -> float:
    """Paired Cohen's d effect size."""
    raise NotImplementedError("cohens_d_paired: TODO")


def bootstrap_ci(
    values: Sequence[float],
    statistic: str = "mean",
    n_resamples: int = 1000,
    confidence_level: float = 0.95,
    seed: int | None = None,
) -> tuple[float, float]:
    """Non-parametric bootstrap percentile CI for a 1-D statistic.

    Parameters
    ----------
    values : observations
    statistic : {'mean', 'median', 'std'}
    n_resamples : int, default 1000
    confidence_level : float, default 0.95

    Returns
    -------
    (lower, upper) — percentile CI.
    """
    raise NotImplementedError("bootstrap_ci: TODO — use numpy RNG + np.percentile")


def clopper_pearson_ci(
    successes: int,
    trials: int,
    confidence_level: float = 0.95,
) -> tuple[float, float]:
    """Exact Clopper-Pearson binomial CI (RQ3 near-knee rate reporting)."""
    raise NotImplementedError("clopper_pearson_ci: TODO — use scipy.stats.beta")


__all__ = [
    "ClassificationRates",
    "fpr_fnr",
    "mcnemar_test",
    "wilcoxon_paired",
    "ks_test",
    "cohens_d_paired",
    "bootstrap_ci",
    "clopper_pearson_ci",
]
