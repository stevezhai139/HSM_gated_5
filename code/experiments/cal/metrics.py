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
from typing import Callable, Sequence, Union

import numpy as np


@dataclass(frozen=True)
class ClassificationRates:
    """Standard confusion-matrix-derived rates.

    All rates are in ``[0, 1]``. When a class is absent from ``y_true`` the
    corresponding rate is undefined in the textbook sense; we return ``0.0``
    with the convention that "no positives" implies no recall to lose
    (``fnr = 0``) and "no negatives" implies no false-alarm room
    (``fpr = 0``). Precision/F1 also default to ``0.0`` when no predictions
    are positive, matching scikit-learn's ``zero_division=0`` behaviour.
    """

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
        Must be equal-length; values outside ``{0, 1}`` raise ``ValueError``.

    Notes
    -----
    - **RQ1 usage:** ``y_true`` derives from performance-clustering ground
      truth (see ``rq1_transfer.py``); ``y_pred`` is the gate decision
      ``1 - G_{w,θ}(W_i, W_{i+1})`` (gate = 1 means "invoke advisor", i.e.
      "potential shift" — matches RQ1 definition in ``Paper3B_Cal_RQs_v0.md``
      §RQ1 "Operational definitions").
    - **RQ5 usage:** ``y_true`` / ``y_pred`` are per-pair labels across a
      single window-size sweep (see ``rq5_boundaries.py``).
    - Confusion-matrix convention (Paper 3B-Cal §Notation):
      positive class = "shift", so TP = correctly flagged shift.
    - Divide-by-zero guards follow scikit-learn's ``zero_division=0``
      default — see the ``ClassificationRates`` docstring.
    """
    a = np.asarray(y_true, dtype=int)
    b = np.asarray(y_pred, dtype=int)
    if a.shape != b.shape:
        raise ValueError(
            f"y_true and y_pred must be the same length (got {a.shape} vs {b.shape})"
        )
    if a.ndim != 1:
        raise ValueError(f"y_true/y_pred must be 1-D (got ndim={a.ndim})")
    if a.size == 0:
        return ClassificationRates(0.0, 0.0, 0.0, 0.0, 0.0)
    unique = set(np.unique(a).tolist()) | set(np.unique(b).tolist())
    if not unique.issubset({0, 1}):
        raise ValueError(f"y_true/y_pred must be binary {{0,1}}; got values {unique}")

    tp = int(np.sum((a == 1) & (b == 1)))
    fp = int(np.sum((a == 0) & (b == 1)))
    tn = int(np.sum((a == 0) & (b == 0)))
    fn = int(np.sum((a == 1) & (b == 0)))

    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )
    return ClassificationRates(
        fpr=float(fpr),
        fnr=float(fnr),
        precision=float(precision),
        recall=float(recall),
        f1=float(f1),
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


_BUILTIN_STATS: dict[str, Callable[[np.ndarray], float]] = {
    "mean": lambda x: float(np.mean(x)),
    "median": lambda x: float(np.median(x)),
    "std": lambda x: float(np.std(x, ddof=1)) if x.size > 1 else 0.0,
}


def bootstrap_ci(
    values: Sequence[float],
    statistic: Union[str, Callable[[np.ndarray], float]] = "mean",
    n_resamples: int = 1000,
    confidence_level: float = 0.95,
    seed: int | None = None,
) -> tuple[float, float]:
    """Non-parametric bootstrap percentile CI for a 1-D statistic.

    Parameters
    ----------
    values : observations
        Any sequence castable to a 1-D ``float`` array.
    statistic : {'mean', 'median', 'std'} or callable, default 'mean'
        When a string is given, uses the corresponding built-in
        (``std`` uses ``ddof=1``). When a callable is given, it must accept
        a 1-D ``np.ndarray`` and return a scalar; this callable form is
        what ``rq5_boundaries.py`` uses to bootstrap a W_min estimator.
    n_resamples : int, default 1000
        Number of bootstrap resamples (RQ1 locks in 1000 — see
        ``Paper3B_Cal_RQs_v0.md`` §RQ1 "Measurement procedure").
    confidence_level : float, default 0.95
        Two-sided percentile CI at this confidence level.
    seed : int or None, default None
        Seed for ``numpy.random.default_rng`` to make the CI reproducible.

    Returns
    -------
    (lower, upper) : tuple of float
        Percentile CI at ``confidence_level``.

    Notes
    -----
    - Uses the percentile bootstrap (Efron 1979). BCa is intentionally *not*
      implemented here — the RQs that need formal coverage guarantees
      (RQ1, RQ2, RQ4) use ``statsmodels``/``scipy`` dedicated tests; this
      helper is for ad-hoc CI reporting (RQ5 ``W_min``, RQ1 FPR/FNR CIs).
    - Raises ``ValueError`` on empty input or unknown statistic string;
      raises ``ValueError`` if ``confidence_level`` is not in ``(0, 1)``.
    """
    arr = np.asarray(list(values), dtype=float)
    if arr.size == 0:
        raise ValueError("bootstrap_ci: values is empty")
    if arr.ndim != 1:
        raise ValueError(f"bootstrap_ci: values must be 1-D (got ndim={arr.ndim})")
    if not (0.0 < confidence_level < 1.0):
        raise ValueError(
            f"bootstrap_ci: confidence_level must be in (0, 1) (got {confidence_level})"
        )
    if n_resamples < 1:
        raise ValueError(f"bootstrap_ci: n_resamples must be >= 1 (got {n_resamples})")

    if isinstance(statistic, str):
        if statistic not in _BUILTIN_STATS:
            raise ValueError(
                f"bootstrap_ci: unknown statistic {statistic!r}; "
                f"use one of {sorted(_BUILTIN_STATS)} or pass a callable"
            )
        stat_fn = _BUILTIN_STATS[statistic]
    else:
        stat_fn = statistic

    rng = np.random.default_rng(seed)
    n = arr.size
    # Vectorised resample-indices (n_resamples, n) keeps this fast for the
    # n_resamples=1000 default without blowing memory on typical RQ sizes.
    idx = rng.integers(0, n, size=(n_resamples, n))
    replicates = np.empty(n_resamples, dtype=float)
    for r in range(n_resamples):
        replicates[r] = stat_fn(arr[idx[r]])

    alpha = 1.0 - confidence_level
    lo = float(np.percentile(replicates, 100.0 * (alpha / 2.0)))
    hi = float(np.percentile(replicates, 100.0 * (1.0 - alpha / 2.0)))
    return lo, hi


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
