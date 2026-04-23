"""Gate implementation G_{w,θ} for HSM-Cal.

G_{w,θ}(W_ref, W_new) = 1 iff K_w(W_ref, W_new) < θ else 0

Gate output 1 = "invoke the downstream advisor" (similarity is LOW, workload
may have shifted). Gate output 0 = "skip the advisor" (similarity HIGH,
current index is probably still fine).

Paper 3A's HSM kernel is imported from `code.experiments.hsm_v2_kernel` and
is not modified. If the kernel API changes in Paper 3A revisions, a shim
layer goes in this file, not in the kernel module.

See `Paper3B_Cal_RQs_v0.md` §0 Notation for the formal definition.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Protocol, Sequence

# TODO: once Paper 3A's HSM kernel API is finalised on main, import directly.
# For now this is a forward-reference via a Protocol to avoid hard-coupling.
# from ..hsm_v2_kernel import hsm_score  # noqa: E402 — future import


class HSMKernel(Protocol):
    """Protocol for the HSM kernel that Paper 3A exposes.

    Any callable with this signature is acceptable. Production implementation
    is in ``code.experiments.hsm_v2_kernel``; test doubles may be injected
    in the unit tests.
    """

    def __call__(
        self,
        w_ref: object,
        w_new: object,
        weights: Sequence[float],
    ) -> float:
        """Return similarity in [0, 1]."""
        ...


@dataclass(frozen=True)
class GateConfig:
    """Immutable gate configuration.

    Attributes
    ----------
    weights : tuple of 5 floats on Δ⁴ (sum = 1, all ≥ 0)
        Dimensional weights (w_R, w_V, w_T, w_A, w_P).
    theta : float in [0.5, 0.95]
        Similarity threshold. Gate fires (emits 1) when similarity < theta.
    """

    weights: tuple[float, float, float, float, float]
    theta: float

    def __post_init__(self) -> None:
        if not (abs(sum(self.weights) - 1.0) < 1e-6):
            raise ValueError(f"weights must sum to 1 (got {sum(self.weights)})")
        if any(w < 0 for w in self.weights):
            raise ValueError(f"weights must be non-negative (got {self.weights})")
        if not (0.5 <= self.theta <= 0.95):
            raise ValueError(f"theta must lie in [0.5, 0.95] (got {self.theta})")


# Paper 3A default configuration (locked by TKDE submission).
DEFAULT_CONFIG = GateConfig(
    weights=(0.25, 0.20, 0.20, 0.20, 0.15),  # W0 from Paper 3A
    theta=0.75,
)


class Gate:
    """Binary gate wrapping an HSM kernel.

    The gate is side-effect-free — state (history, EMA tracker, etc.) lives
    outside in ``ema_tracker.py``.
    """

    def __init__(self, kernel: HSMKernel, config: GateConfig = DEFAULT_CONFIG) -> None:
        self.kernel = kernel
        self.config = config

    def decide(self, w_ref: object, w_new: object) -> int:
        """Return 1 if advisor should be invoked, 0 if skipped."""
        similarity = self.kernel(w_ref, w_new, self.config.weights)
        return int(similarity < self.config.theta)

    def similarity(self, w_ref: object, w_new: object) -> float:
        """Return the raw similarity (debug / RQ3 sweep usage)."""
        return self.kernel(w_ref, w_new, self.config.weights)

    def with_theta(self, theta: float) -> "Gate":
        """Return a new gate with the same weights and a different theta."""
        return Gate(self.kernel, GateConfig(self.config.weights, theta))

    def with_config(self, config: GateConfig) -> "Gate":
        """Return a new gate with a new configuration (weights + theta)."""
        return Gate(self.kernel, config)


__all__ = ["Gate", "GateConfig", "DEFAULT_CONFIG", "HSMKernel"]
