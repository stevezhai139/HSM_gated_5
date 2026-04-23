"""Online EMA drift-tracker for θ — Stage 2 of the Hybrid BO+EMA
calibration (design doc §5.1).

Weights are LOCKED from offline Stage 1 calibration (BO). Only θ adapts
online. This intentional asymmetry keeps online cost at < 1 ms per window
(weights adaptation would require re-evaluating cost-benefit per window,
which is expensive in deployment).

Update rule (simplified — see `Paper3B_Cal_RQs_v0.md` RQ4b for the full form):

    θ_t ← (1 − α) · θ_{t−1} + α · θ_target(W_t)
    θ_t ← clip(θ_t, θ_{t−1} − δ, θ_{t−1} + δ)    # monotone-rate guard

where θ_target is the Pareto-knee θ computed on a rolling window of the
last `rolling_window` gate outcomes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Protocol


class ThetaTargetFn(Protocol):
    """Given the most recent gate outcomes, return the target θ value that
    would maximise cost-benefit over that window."""

    def __call__(self, outcomes: list["GateOutcome"]) -> float:
        ...


@dataclass(frozen=True)
class GateOutcome:
    """A single gate decision + its realised outcome (for post-hoc
    reinforcement of θ_target)."""

    window_index: int
    similarity: float
    gate_decision: int
    realised_cost_saving: float
    realised_quality_loss: float


@dataclass
class EMATracker:
    """Online θ tracker with EMA + monotone-rate guard."""

    alpha: float = 0.1
    max_step: float = 0.02
    rolling_window: int = 20
    theta_target_fn: ThetaTargetFn | None = None
    _history: list[GateOutcome] = field(default_factory=list)
    _current_theta: float = 0.75

    def update(self, outcome: GateOutcome) -> float:
        """Record an outcome and return the updated θ."""
        raise NotImplementedError(
            "EMATracker.update: TODO — append outcome, recompute target via "
            "theta_target_fn on the last `rolling_window` items, apply EMA, "
            "clip by max_step, store as _current_theta, return."
        )

    @property
    def current_theta(self) -> float:
        return self._current_theta


@dataclass
class NoUpdateBaseline:
    """Baseline for RQ4b — θ frozen at BO-calibrated value throughout."""

    theta: float = 0.75

    def update(self, outcome: GateOutcome) -> float:
        return self.theta

    @property
    def current_theta(self) -> float:
        return self.theta


def default_theta_target(outcomes: list[GateOutcome]) -> float:
    """Default implementation: fit a one-parameter θ that minimises a
    scalarised cost-benefit on the given outcomes (post-hoc)."""
    raise NotImplementedError(
        "default_theta_target: TODO — closed-form or 1-D grid search; must be "
        "fast enough for per-window online execution (< 1 ms target)."
    )


__all__ = [
    "EMATracker",
    "NoUpdateBaseline",
    "GateOutcome",
    "ThetaTargetFn",
    "default_theta_target",
]
