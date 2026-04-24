"""θ-sweep analysis on a Paper 3A PairSeries.

For each θ in a grid, compute gate predictions (``y_pred = 1 iff K < θ``)
against the ground-truth transition labels (``y_true = is_transition``) and
derive classification rates. Identify θ* that optimises a chosen
objective and report the exact set of false-positive / false-negative
pair indices so downstream analysis can inspect WHICH pairs disagree.

Convention (Paper3B_Cal_Theoretical_Foundations §4.1):
positive class = "shift occurs"; gate prediction = G directly (no flip).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .paper3a_loader import PairSeries


@dataclass(frozen=True)
class ClassificationRates:
    """Per-θ confusion matrix and derived rates."""

    theta: float
    tp: int
    fp: int
    tn: int
    fn: int
    precision: float
    recall: float
    f1: float
    fpr: float
    fnr: float
    n_triggered: int
    fp_indices: Tuple[int, ...]
    fn_indices: Tuple[int, ...]


@dataclass(frozen=True)
class ThetaSweepResult:
    """Full sweep result across a θ grid."""

    experiment: str
    grid: Tuple[float, ...]
    rates: Tuple[ClassificationRates, ...]
    theta_star_by_metric: Dict[str, float] = field(default_factory=dict)
    theta_star_exact_count: Optional[float] = None   # θ where triggers == n_transitions
    theta_star_rates: Dict[str, "ClassificationRates"] = field(default_factory=dict)


def _rates_at_theta(
    scores: np.ndarray,
    y_true: np.ndarray,
    theta: float,
) -> ClassificationRates:
    y_pred = (scores < theta).astype(int)
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
    fp_idx = tuple(int(i) for i in np.where((y_true == 0) & (y_pred == 1))[0])
    fn_idx = tuple(int(i) for i in np.where((y_true == 1) & (y_pred == 0))[0])
    return ClassificationRates(
        theta=float(theta),
        tp=tp,
        fp=fp,
        tn=tn,
        fn=fn,
        precision=float(precision),
        recall=float(recall),
        f1=float(f1),
        fpr=float(fpr),
        fnr=float(fnr),
        n_triggered=tp + fp,
        fp_indices=fp_idx,
        fn_indices=fn_idx,
    )


def sweep(
    series: PairSeries,
    theta_grid: Sequence[float] = tuple(round(0.50 + 0.01 * k, 2) for k in range(46)),
) -> ThetaSweepResult:
    """Sweep θ over the grid and compute rates per θ.

    The default grid spans [0.50, 0.95] in 0.01 steps (46 points).

    Also identifies:
    - θ* minimising FP + FN (best total error)
    - θ* maximising F1
    - θ* such that number of triggers exactly matches number of transitions
      (if multiple such θ exist, returns the midpoint of the range)
    """
    scores = np.asarray(series.scores, dtype=float)
    y_true = np.asarray(series.is_transition, dtype=int)

    rates = tuple(_rates_at_theta(scores, y_true, t) for t in theta_grid)

    # Best by total-error (FP + FN).
    total_err = np.array([r.fp + r.fn for r in rates])
    best_err_idx = int(np.argmin(total_err))
    # Best by F1.
    f1_arr = np.array([r.f1 for r in rates])
    best_f1_idx = int(np.argmax(f1_arr))
    # Exact-count θ*: triggers == n_transitions
    n_trans = series.n_transitions
    matching = [r for r in rates if r.n_triggered == n_trans]
    theta_star_exact = (
        float(np.mean([r.theta for r in matching])) if matching else None
    )

    theta_star_by_metric = {
        "min_total_error": float(rates[best_err_idx].theta),
        "max_f1": float(rates[best_f1_idx].theta),
        "exact_trigger_count": theta_star_exact if theta_star_exact is not None else float("nan"),
    }
    theta_star_rates = {
        "min_total_error": rates[best_err_idx],
        "max_f1": rates[best_f1_idx],
    }
    if matching:
        # Pick the matching rate closest to the mean θ for reporting.
        closest = min(matching, key=lambda r: abs(r.theta - theta_star_exact))
        theta_star_rates["exact_trigger_count"] = closest

    return ThetaSweepResult(
        experiment=series.experiment,
        grid=tuple(theta_grid),
        rates=rates,
        theta_star_by_metric=theta_star_by_metric,
        theta_star_exact_count=theta_star_exact,
        theta_star_rates=theta_star_rates,
    )


def result_to_json_dict(
    series: PairSeries,
    sweep_result: ThetaSweepResult,
) -> Dict[str, object]:
    """Serialise a sweep result (with per-θ breakdown) to a JSON-safe dict."""
    return {
        "series": {
            "experiment": series.experiment,
            "csv_path": series.csv_path,
            "n_pairs": series.n_pairs,
            "n_transitions": series.n_transitions,
            "paper3a_default_trigger_count": series.paper3a_default_trigger_count,
            "phases_seen": list(series.phases_seen),
            "score_stats": {
                "min": min(series.scores),
                "max": max(series.scores),
                "mean": sum(series.scores) / len(series.scores),
            },
            "transition_pair_indices": [
                i for i, t in enumerate(series.is_transition) if t == 1
            ],
            "transition_pair_scores": [
                series.scores[i]
                for i, t in enumerate(series.is_transition) if t == 1
            ],
        },
        "grid": list(sweep_result.grid),
        "rates": [
            {
                "theta": r.theta,
                "tp": r.tp, "fp": r.fp, "tn": r.tn, "fn": r.fn,
                "precision": r.precision, "recall": r.recall, "f1": r.f1,
                "fpr": r.fpr, "fnr": r.fnr,
                "n_triggered": r.n_triggered,
                "fp_indices": list(r.fp_indices),
                "fn_indices": list(r.fn_indices),
            }
            for r in sweep_result.rates
        ],
        "theta_star_by_metric": sweep_result.theta_star_by_metric,
        "theta_star_rates": {
            k: {
                "theta": v.theta,
                "tp": v.tp, "fp": v.fp, "tn": v.tn, "fn": v.fn,
                "precision": v.precision, "recall": v.recall, "f1": v.f1,
                "fpr": v.fpr, "fnr": v.fnr,
                "n_triggered": v.n_triggered,
                "fp_indices": list(v.fp_indices),
                "fn_indices": list(v.fn_indices),
            }
            for k, v in sweep_result.theta_star_rates.items()
        },
    }


__all__ = [
    "ClassificationRates",
    "ThetaSweepResult",
    "sweep",
    "result_to_json_dict",
]
