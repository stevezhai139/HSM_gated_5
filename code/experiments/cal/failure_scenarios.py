"""RQ5 failure-scenario generators.

Two scenarios defined in `Paper3B_Cal_RQs_v0.md` RQ5:

F1. Micro-shift granularity: within-window per-query parametric drift at
    50 % rate. Exercises whether the gate fails when the signal-to-noise
    ratio drops inside a window.

F2. Syntactic-semantic mismatch: paired workloads with identical SQL
    templates but different target tables. Expected gate output: "shift"
    (invoke advisor), because HSM's Access (S_A) and Periodicity (S_P)
    dimensions should fire even though syntax is identical.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class ScenarioWindows:
    """A (baseline, perturbed) pair of windows for controlled comparison."""

    baseline: object
    perturbed: object
    scenario_name: str
    expected_gate_decision: int  # 0 = skip, 1 = invoke


def micro_shift_scenario(
    windows: Sequence[object],
    fraction_perturbed: float = 0.5,
    *,
    seed: int | None = None,
) -> list[ScenarioWindows]:
    """F1 — replace `fraction_perturbed` of queries in each window with
    parametrically-drifted variants (different predicate constants, same
    templates, same tables)."""
    raise NotImplementedError(
        "micro_shift_scenario: TODO — need a query parametric-drift helper "
        "that rewrites literals/parameters without changing structure."
    )


def syntactic_semantic_mismatch_scenario(
    templates: Sequence[str],
    table_pairs: Sequence[tuple[str, str]],
    *,
    seed: int | None = None,
) -> list[ScenarioWindows]:
    """F2 — for each template and each (tableA, tableB) pair, produce a
    window-A using the template on tableA and a window-B using the same
    template on tableB. Expected gate decision: 1 (invoke advisor)."""
    raise NotImplementedError(
        "syntactic_semantic_mismatch_scenario: TODO — template instantiator + "
        "table substitution; verify that S_A and S_P actually differ."
    )


__all__ = [
    "ScenarioWindows",
    "micro_shift_scenario",
    "syntactic_semantic_mismatch_scenario",
]
