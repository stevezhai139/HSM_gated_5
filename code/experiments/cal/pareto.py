"""Pareto-efficient-set + knee-point finders for RQ3 and RQ4.

For Paper 3B-Cal, the two objectives are:
  - maximise CostSavings (skipped advisor invocations × per-invoke cost)
  - minimise QualityLoss (expected index quality degradation on skipped windows)

RQ3 uses `pareto_front` + `knee_point` to locate the frontier and its knee.
RQ4 uses the same to assess whether BO-calibrated (w*, θ*) Pareto-dominates
the default (W₀, 0.75).

Convention: all objectives are stored as "lower is better" in internal
computations. Wrap with (- CostSavings, QualityLoss) before calling.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence


@dataclass(frozen=True)
class ParetoPoint:
    """A single Pareto-annotated configuration point.

    Attributes
    ----------
    label : hashable identifier (e.g., theta value, BO evaluation index)
    objectives : tuple of floats (all "lower is better" after conversion)
    is_efficient : set by `pareto_front`
    """

    label: object
    objectives: tuple[float, ...]
    is_efficient: bool = False


def pareto_front(points: Sequence[ParetoPoint]) -> list[ParetoPoint]:
    """Return the Pareto-efficient subset of `points`.

    Uses the standard O(n²) sweep which is fine for the grid sizes in
    Paper 3B-Cal (≤ 50 points per cell).
    """
    raise NotImplementedError(
        "pareto_front: TODO — mark each point efficient iff no other point "
        "strictly dominates it (lower or equal in all objectives, strictly "
        "lower in at least one)."
    )


def knee_point(efficient_points: Sequence[ParetoPoint]) -> ParetoPoint:
    """Return the knee of a Pareto-efficient set.

    Definition (matches RQ3 §operational defs): min-max normalise each
    objective over the efficient set, then return the point maximising
    `sum(1 - obj_norm)` across objectives. Ties broken by lexicographic
    order of labels.
    """
    raise NotImplementedError(
        "knee_point: TODO — min-max normalise; then argmax(sum(1 - obj_norm))."
    )


def is_dominated(
    candidate: tuple[float, ...], reference: tuple[float, ...]
) -> bool:
    """Does `reference` dominate `candidate`? (lower = better convention)."""
    raise NotImplementedError("is_dominated: TODO — O(k) component-wise compare.")


def normalised_distance_to_knee(
    point: tuple[float, ...],
    knee: tuple[float, ...],
    front_extent: Iterable[tuple[float, float]],
) -> float:
    """Min-max-normalised L2 distance from `point` to `knee`.

    `front_extent` gives (min, max) per objective over the efficient set
    — used for normalisation.
    """
    raise NotImplementedError("normalised_distance_to_knee: TODO")


__all__ = [
    "ParetoPoint",
    "pareto_front",
    "knee_point",
    "is_dominated",
    "normalised_distance_to_knee",
]
