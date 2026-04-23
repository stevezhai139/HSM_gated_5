"""Tests for ema_tracker.py — marked xfail on NotImplementedError."""

from __future__ import annotations

import pytest

from cal import ema_tracker


pytestmark = pytest.mark.xfail(
    raises=NotImplementedError,
    reason="Paper 3B-Cal scaffolding — implementations pending",
)


class TestNoUpdateBaseline:
    def test_theta_never_changes(self):
        baseline = ema_tracker.NoUpdateBaseline(theta=0.75)
        outcome = ema_tracker.GateOutcome(
            window_index=0, similarity=0.5, gate_decision=1,
            realised_cost_saving=1.0, realised_quality_loss=0.1,
        )
        assert baseline.update(outcome) == 0.75
        assert baseline.update(outcome) == 0.75


class TestEMATracker:
    def test_initial_theta_preserved_before_any_update(self):
        tr = ema_tracker.EMATracker(_current_theta=0.75)
        assert tr.current_theta == 0.75

    def test_monotone_step_guard_limits_change(self):
        tr = ema_tracker.EMATracker(
            alpha=1.0,  # fully follow target
            max_step=0.02,
            rolling_window=1,
            theta_target_fn=lambda outs: 0.95,  # wants to jump far
            _current_theta=0.75,
        )
        outcome = ema_tracker.GateOutcome(
            window_index=0, similarity=0.3, gate_decision=1,
            realised_cost_saving=0.0, realised_quality_loss=0.0,
        )
        new_theta = tr.update(outcome)
        assert abs(new_theta - 0.75) <= 0.02 + 1e-9
