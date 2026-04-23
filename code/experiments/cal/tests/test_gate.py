"""Tests for gate.py."""

from __future__ import annotations

import pytest

from cal.gate import DEFAULT_CONFIG, Gate, GateConfig


def _stub_kernel(similarity: float):
    """Return a stub kernel that always reports a fixed similarity."""

    def kernel(w_ref, w_new, weights):
        return similarity

    return kernel


class TestGateConfig:
    def test_default_is_W0_and_075(self):
        assert DEFAULT_CONFIG.weights == (0.25, 0.20, 0.20, 0.20, 0.15)
        assert DEFAULT_CONFIG.theta == 0.75

    def test_weights_must_sum_to_one(self):
        with pytest.raises(ValueError, match="sum to 1"):
            GateConfig(weights=(0.5, 0.5, 0.5, 0.5, 0.5), theta=0.75)

    def test_weights_must_be_nonneg(self):
        with pytest.raises(ValueError, match="non-negative"):
            GateConfig(weights=(-0.1, 0.3, 0.3, 0.3, 0.2), theta=0.75)

    def test_theta_in_bounds(self):
        with pytest.raises(ValueError, match="0.5, 0.95"):
            GateConfig(weights=(0.2, 0.2, 0.2, 0.2, 0.2), theta=0.3)
        with pytest.raises(ValueError, match="0.5, 0.95"):
            GateConfig(weights=(0.2, 0.2, 0.2, 0.2, 0.2), theta=1.0)


class TestGateDecision:
    def test_fires_when_similarity_below_theta(self):
        gate = Gate(_stub_kernel(0.5), DEFAULT_CONFIG)
        assert gate.decide(None, None) == 1  # 0.5 < 0.75 → invoke

    def test_skips_when_similarity_above_theta(self):
        gate = Gate(_stub_kernel(0.9), DEFAULT_CONFIG)
        assert gate.decide(None, None) == 0

    def test_boundary_equal_theta_means_skip(self):
        # definition: G = 1 iff similarity < theta (strict)
        gate = Gate(_stub_kernel(0.75), DEFAULT_CONFIG)
        assert gate.decide(None, None) == 0

    def test_with_theta_returns_new_gate(self):
        gate = Gate(_stub_kernel(0.8), DEFAULT_CONFIG)
        gate2 = gate.with_theta(0.85)
        assert gate2.decide(None, None) == 1
        assert gate.decide(None, None) == 0  # original unchanged
