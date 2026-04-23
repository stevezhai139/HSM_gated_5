"""Offline Bayesian-Optimisation calibrator for (w*, θ*) — Stage 1 of the
Hybrid BO+EMA calibration from design doc §5.1.

Primary implementation: `skopt.gp_minimize` for single-objective BO on the
6-D search space Δ⁴ × [0.5, 0.95]. For multi-objective Pareto exploration
(RQ3/RQ4 use-case), fall back to BoTorch's qEHVI acquisition in CPU mode.

Secondary baseline: CMA-ES at larger evaluation budget — tests whether BO
actually wins on sample efficiency at this problem's dimensionality.
See `Paper3B_Cal_SOTA_v0.md` §2.7 for the methodology citations.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Protocol, Sequence

from .gate import GateConfig


class ObjectiveFn(Protocol):
    """A callable that evaluates (weights, theta) on a held-out trace and
    returns (cost_savings, quality_loss)."""

    def __call__(
        self, weights: Sequence[float], theta: float
    ) -> tuple[float, float]:
        ...


@dataclass
class CalibrationResult:
    """Outcome of a single calibrator run."""

    config: GateConfig
    eval_history: list[tuple[tuple[float, ...], float, tuple[float, float]]] = field(
        default_factory=list
    )  # [(weights, theta, (cs, ql)), ...]
    n_evals: int = 0
    convergence_curve: list[float] = field(default_factory=list)  # hypervolume


class BOCalibrator:
    """Bayesian Optimisation calibrator for (w, θ).

    Defaults: 50 evaluations, GP with Matern-5/2 kernel, single-objective
    via scalarisation (tunable via `alpha`), OR multi-objective via qEHVI
    when `multi_objective=True`.
    """

    def __init__(
        self,
        n_evals: int = 50,
        alpha: float = 5.0,
        multi_objective: bool = False,
        seed: int | None = None,
    ) -> None:
        self.n_evals = n_evals
        self.alpha = alpha  # weight on quality-loss in scalarisation
        self.multi_objective = multi_objective
        self.seed = seed

    def run(self, objective: ObjectiveFn) -> CalibrationResult:
        """Run BO and return the best-found configuration.

        Search space:
          weights ∈ Δ⁴ (4 free dims via simplex parametrisation)
          theta   ∈ [0.5, 0.95]

        Scalarised objective (single-obj mode):
          f(w, θ) = - cost_savings + alpha * quality_loss
        (minimised).

        Multi-objective mode: minimises (-cost_savings, quality_loss) via
        BoTorch qEHVI; returns the Pareto-front hypervolume-maximising
        configuration as the "best" point.
        """
        raise NotImplementedError(
            "BOCalibrator.run: TODO — implement skopt.gp_minimize path first; "
            "add BoTorch qEHVI multi-objective path when RQ3 needs it."
        )


class CMAESCalibrator:
    """CMA-ES secondary baseline — published wisdom says BO wins at this
    dim/budget, but we need the comparison for RQ4a secondary success."""

    def __init__(
        self,
        n_evals: int = 200,
        population_size: int = 8,
        alpha: float = 5.0,
        seed: int | None = None,
    ) -> None:
        self.n_evals = n_evals
        self.population_size = population_size
        self.alpha = alpha
        self.seed = seed

    def run(self, objective: ObjectiveFn) -> CalibrationResult:
        """Run CMA-ES and return the best-found configuration."""
        raise NotImplementedError(
            "CMAESCalibrator.run: TODO — wrap `cma.CMAEvolutionStrategy`; "
            "apply the same scalarisation as BOCalibrator for parity."
        )


def simplex_to_free(weights: Sequence[float]) -> tuple[float, ...]:
    """Map a 5-point simplex vector to 4 free parameters (for BO search space)."""
    raise NotImplementedError("simplex_to_free: TODO — standard softmax parametrisation")


def free_to_simplex(free: Sequence[float]) -> tuple[float, float, float, float, float]:
    """Inverse of simplex_to_free."""
    raise NotImplementedError("free_to_simplex: TODO")


__all__ = [
    "BOCalibrator",
    "CMAESCalibrator",
    "CalibrationResult",
    "ObjectiveFn",
    "simplex_to_free",
    "free_to_simplex",
]
