"""Tests for pareto.py — marked xfail on NotImplementedError."""

from __future__ import annotations

import pytest

from cal import pareto


pytestmark = pytest.mark.xfail(
    raises=NotImplementedError,
    reason="Paper 3B-Cal scaffolding — implementations pending",
)


class TestParetoFront:
    def test_single_point_is_efficient(self):
        p = pareto.ParetoPoint(label="a", objectives=(0.0, 0.0))
        front = pareto.pareto_front([p])
        assert len(front) == 1
        assert front[0].is_efficient

    def test_dominated_point_excluded(self):
        a = pareto.ParetoPoint(label="a", objectives=(0.0, 0.0))
        b = pareto.ParetoPoint(label="b", objectives=(1.0, 1.0))  # dominated by a
        front = pareto.pareto_front([a, b])
        labels = {p.label for p in front}
        assert labels == {"a"}

    def test_two_non_dominated_both_kept(self):
        a = pareto.ParetoPoint(label="a", objectives=(0.0, 1.0))
        b = pareto.ParetoPoint(label="b", objectives=(1.0, 0.0))
        front = pareto.pareto_front([a, b])
        assert len(front) == 2


class TestKneePoint:
    def test_single_point_is_knee(self):
        a = pareto.ParetoPoint(label="a", objectives=(0.3, 0.4))
        assert pareto.knee_point([a]).label == "a"

    def test_trades_off_center_is_knee(self):
        # Three efficient points: (0,1), (0.5,0.5), (1,0). Knee should be middle.
        a = pareto.ParetoPoint(label="a", objectives=(0.0, 1.0))
        b = pareto.ParetoPoint(label="b", objectives=(0.5, 0.5))
        c = pareto.ParetoPoint(label="c", objectives=(1.0, 0.0))
        assert pareto.knee_point([a, b, c]).label == "b"


class TestIsDominated:
    def test_strict_dominance(self):
        assert pareto.is_dominated((1.0, 1.0), (0.0, 0.0))
        assert not pareto.is_dominated((0.0, 0.0), (1.0, 1.0))

    def test_equal_is_not_dominated(self):
        assert not pareto.is_dominated((1.0, 1.0), (1.0, 1.0))
