"""Tests for bo_calibrator.py — marked xfail on NotImplementedError."""

from __future__ import annotations

import pytest

from cal import bo_calibrator


pytestmark = pytest.mark.xfail(
    raises=NotImplementedError,
    reason="Paper 3B-Cal scaffolding — implementations pending",
)


def _stub_objective(weights, theta):
    """Objective with a known optimum near theta=0.65, uniform weights."""
    w_dev = sum((w - 0.2) ** 2 for w in weights)
    theta_dev = (theta - 0.65) ** 2
    cost_savings = 1.0 - w_dev - theta_dev
    quality_loss = w_dev + theta_dev
    return cost_savings, quality_loss


class TestBOCalibrator:
    def test_runs_with_small_budget(self):
        calibrator = bo_calibrator.BOCalibrator(n_evals=10, seed=42)
        result = calibrator.run(_stub_objective)
        assert result.n_evals <= 10
        # Near optimum
        assert abs(result.config.theta - 0.65) < 0.1

    def test_multi_objective_returns_pareto_family(self):
        calibrator = bo_calibrator.BOCalibrator(
            n_evals=15, multi_objective=True, seed=42
        )
        result = calibrator.run(_stub_objective)
        assert len(result.convergence_curve) > 0


class TestCMAESCalibrator:
    def test_runs_with_small_budget(self):
        calibrator = bo_calibrator.CMAESCalibrator(n_evals=20, seed=42)
        result = calibrator.run(_stub_objective)
        assert result.n_evals <= 25  # small population overshoot tolerated


class TestSimplexReparam:
    def test_roundtrip(self):
        original = (0.25, 0.20, 0.20, 0.20, 0.15)
        free = bo_calibrator.simplex_to_free(original)
        back = bo_calibrator.free_to_simplex(free)
        for a, b in zip(original, back):
            assert abs(a - b) < 1e-9
