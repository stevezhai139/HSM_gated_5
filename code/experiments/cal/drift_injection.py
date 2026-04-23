"""Synthetic drift generators for RQ4b.

Primary scenario: mix-shift drift — at a specified window index, re-sample a
fraction of queries from a *different* workload distribution. Creates a
clean, reproducible drift event at a known time that the EMA tracker must
respond to.

See `Paper3B_Cal_RQs_v0.md` RQ4b for the 30 %-magnitude default.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class DriftSchedule:
    """Description of a drift-injection schedule."""

    drift_start_window: int
    mix_fraction: float           # fraction of queries replaced, 0..1
    source_workload_label: str    # e.g., "sdss_dr18"
    foreign_workload_label: str   # e.g., "stackoverflow"


def inject_mix_shift(
    windows: Sequence[object],
    foreign_windows: Sequence[object],
    schedule: DriftSchedule,
    *,
    seed: int | None = None,
) -> list[object]:
    """Return a new sequence of windows with drift injected per schedule.

    For window index `i < schedule.drift_start_window`, the window is
    returned unchanged. For `i >= drift_start_window`, replace
    `mix_fraction` of the queries in the window with random queries
    sampled from `foreign_windows`.

    The injection is deterministic given `seed`.
    """
    raise NotImplementedError(
        "inject_mix_shift: TODO — implement per-window query replacement with "
        "reproducible RNG; preserve window metadata (time, DB snapshot)."
    )


__all__ = ["DriftSchedule", "inject_mix_shift"]
