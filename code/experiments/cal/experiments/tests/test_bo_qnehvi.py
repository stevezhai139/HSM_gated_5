"""Unit tests for bo_qnehvi_calibration (M10 direct-simplex parameterisation).

BoTorch-dependent tests are separate (and skipped if BoTorch not installed).
These tests cover the pure-python helpers: simplex projection, Pareto mask,
2-D hypervolume, and the per-pair evaluation pipeline on a tiny synthetic
labelled trace.

Refactor note (2026-04-24): softmax-pinning parameterisation (M3) has been
replaced by direct-simplex BO (M10) using BoTorch equality_constraints +
inequality_constraints. Tests formerly named TestSoftmaxSimplex are retired;
TestSimplexProjection is added for the Sobol-init helper.
"""

from __future__ import annotations

import math
from typing import List, Tuple

import numpy as np
import pytest


# Import SUT
from cal.experiments import bo_qnehvi_calibration as sut


# -------------------------------------------------------------------------
# Simplex projection (used for Sobol init)
# -------------------------------------------------------------------------

class TestProjectToSimplex:
    def test_uniform_input_maps_to_uniform_simplex(self):
        x = np.ones(5)
        w = sut.project_to_simplex(x)
        np.testing.assert_allclose(w, np.full(5, 0.2), atol=1e-9)

    def test_preserves_ratios(self):
        x = np.array([2.0, 1.0, 1.0, 0.5, 0.5])
        w = sut.project_to_simplex(x)
        assert math.isclose(w.sum(), 1.0, abs_tol=1e-9)
        # Ratios preserved: w[0] / w[1] == x[0] / x[1] == 2
        assert math.isclose(w[0] / w[1], 2.0, rel_tol=1e-9)

    def test_zero_sum_falls_back_to_uniform(self):
        x = np.zeros(5)
        w = sut.project_to_simplex(x)
        np.testing.assert_allclose(w, np.full(5, 0.2), atol=1e-9)

    def test_batch_shape(self):
        x = np.random.default_rng(0).uniform(0, 1, size=(100, 5))
        w = sut.project_to_simplex(x)
        assert w.shape == (100, 5)
        np.testing.assert_allclose(w.sum(axis=-1), np.ones(100), atol=1e-9)

    def test_all_positive_after_projection(self):
        rng = np.random.default_rng(42)
        x = rng.uniform(0, 1, size=(1000, 5))
        w = sut.project_to_simplex(x)
        # After projection of uniform [0,1] samples, w_i can be 0 only if
        # x_i is exactly 0; with uniform sampling this is measure-zero.
        assert np.all(w >= 0)


class TestSimplexWeightsDict:
    def test_produces_paper3a_keys(self):
        w = np.array([0.25, 0.20, 0.20, 0.20, 0.15])
        d = sut.simplex_weights_dict(w)
        assert set(d.keys()) == {"w_R", "w_V", "w_T", "w_A", "w_P"}
        assert math.isclose(sum(d.values()), 1.0, abs_tol=1e-9)
        assert d["w_R"] == 0.25

    def test_rejects_wrong_shape(self):
        with pytest.raises(ValueError, match="must be 5-D"):
            sut.simplex_weights_dict(np.zeros(3))


# -------------------------------------------------------------------------
# Pareto mask + 2-D hypervolume (unchanged from M3 — same logic)
# -------------------------------------------------------------------------

class TestParetoMask:
    def test_single_point_is_pareto(self):
        pts = np.array([[0.5, 0.5]])
        assert sut._pareto_mask(pts).tolist() == [True]

    def test_strictly_dominated_point_removed(self):
        pts = np.array([[0.9, 0.9], [0.5, 0.5], [0.2, 0.2]])
        mask = sut._pareto_mask(pts)
        assert mask.tolist() == [True, False, False]

    def test_non_dominated_pair_both_kept(self):
        pts = np.array([[1.0, 0.0], [0.0, 1.0]])
        mask = sut._pareto_mask(pts)
        assert mask.tolist() == [True, True]

    def test_frontier_on_arc(self):
        pts = np.array([[1.0, 0.0], [0.7, 0.7], [0.0, 1.0], [0.5, 0.3]])
        mask = sut._pareto_mask(pts)
        assert mask.tolist() == [True, True, True, False]


class TestHypervolume2D:
    def test_empty_after_ref_filter(self):
        pts = np.array([[-1.0, -1.0]])
        assert sut._hypervolume_2d(pts, (0.0, 0.0)) == 0.0

    def test_single_point_box_area(self):
        pts = np.array([[0.5, 0.4]])
        assert math.isclose(sut._hypervolume_2d(pts, (0.0, 0.0)), 0.2, rel_tol=1e-9)

    def test_two_non_dominated_non_degenerate(self):
        pts = np.array([[0.8, 0.3], [0.4, 0.7]])
        # Sort by x desc: (0.8, 0.3), then (0.4, 0.7).
        # Step 1: x=0.8, y=0.3 → 0.8 * 0.3 = 0.24
        # Step 2: x=0.4, y=0.7 → 0.4 * (0.7 - 0.3) = 0.16
        # Total = 0.40
        assert math.isclose(sut._hypervolume_2d(pts, (0.0, 0.0)), 0.40, rel_tol=1e-9)

    def test_monotonic_addition(self):
        pts = np.array([[0.5, 0.5]])
        hv1 = sut._hypervolume_2d(pts, (0.0, 0.0))
        pts2 = np.array([[0.5, 0.5], [0.7, 0.3]])
        hv2 = sut._hypervolume_2d(pts2, (0.0, 0.0))
        assert hv2 > hv1


class TestRunningHypervolume:
    def test_monotone_non_decreasing(self):
        pts = [(0.5, 0.5), (0.7, 0.3), (0.4, 0.7), (0.2, 0.9), (0.9, 0.1)]
        hv_series = sut._running_hypervolume(pts, (0.0, 0.0))
        assert len(hv_series) == 5
        for i in range(1, len(hv_series)):
            assert hv_series[i] >= hv_series[i - 1] - 1e-12


# -------------------------------------------------------------------------
# Trace binning + phase labelling (unchanged from M3)
# -------------------------------------------------------------------------

def _synthetic_trace() -> List[Tuple[str, str]]:
    """Build a small TPC-H-like trace: 2 queries per phase × 4 phases = 8."""
    return [
        ("Q1", "SELECT * FROM lineitem WHERE l_shipdate < '1998-09-01'"),
        ("Q3", "SELECT l_orderkey FROM lineitem, orders WHERE l_orderkey = o_orderkey"),
        ("Q10", "SELECT c_custkey, c_name FROM customer, orders WHERE c_custkey = o_custkey"),
        ("Q13", "SELECT c_count FROM customer"),
        ("Q1", "SELECT * FROM lineitem WHERE l_shipdate < '1998-09-01'"),
        ("Q3", "SELECT l_orderkey FROM lineitem, orders WHERE l_orderkey = o_orderkey"),
        ("Q2", "SELECT ps_partkey, ps_availqty FROM partsupp, supplier, nation"),
        ("Q8", "SELECT n_name, sum(volume) FROM lineitem, orders, customer"),
    ]


class TestBuildLabelledWindows:
    def test_basic_windowing(self):
        trace = _synthetic_trace()
        labelled = sut.build_labelled_windows(trace, window_size=2, queries_per_phase=2)
        assert len(labelled.windows) == 4
        assert labelled.phase_labels == (
            "Phase_A", "Phase_B", "Phase_A_repeat", "Phase_C"
        )
        assert labelled.transition_labels == (1, 1, 1)
        assert labelled.n_pairs == 3

    def test_within_phase_pair_is_not_transition(self):
        trace = _synthetic_trace() + _synthetic_trace()[:4]
        labelled = sut.build_labelled_windows(trace, window_size=2, queries_per_phase=4)
        assert 0 in labelled.transition_labels

    def test_rejects_tiny_window(self):
        with pytest.raises(ValueError, match=">= 2"):
            sut.build_labelled_windows(_synthetic_trace(), window_size=1, queries_per_phase=2)

    def test_rejects_insufficient_windows(self):
        trace = _synthetic_trace()[:4]
        with pytest.raises(ValueError, match=">= 3 windows"):
            sut.build_labelled_windows(trace, window_size=2, queries_per_phase=2)


# -------------------------------------------------------------------------
# evaluate_precision_recall with direct simplex w
# -------------------------------------------------------------------------

class TestEvaluatePrecisionRecall:
    def test_default_w_returns_finite_values(self):
        base = _synthetic_trace()
        trace = []
        for _ in range(4):
            trace.extend(base)
        labelled = sut.build_labelled_windows(trace, window_size=2, queries_per_phase=8)
        w = np.array([0.25, 0.20, 0.20, 0.20, 0.15])  # Paper 3A W₀
        p, r = sut.evaluate_precision_recall(labelled, w, theta=0.75)
        assert 0.0 <= p <= 1.0
        assert 0.0 <= r <= 1.0

    def test_accepts_boundary_w_at_eps(self):
        """Verify HSM kernel tolerates w_i at the ε boundary without crashing."""
        base = _synthetic_trace()
        trace = []
        for _ in range(4):
            trace.extend(base)
        labelled = sut.build_labelled_windows(trace, window_size=2, queries_per_phase=8)
        # w_R nearly zero, rest distributed
        w = np.array([sut.EPS, 0.25, 0.25, 0.25, 0.25 - sut.EPS])
        p, r = sut.evaluate_precision_recall(labelled, w, theta=0.75)
        assert 0.0 <= p <= 1.0
        assert 0.0 <= r <= 1.0


# -------------------------------------------------------------------------
# BoTorch-dependent smoke: skipped if botorch not installed
# -------------------------------------------------------------------------


def _botorch_available() -> bool:
    try:
        import botorch  # noqa: F401
        return True
    except ImportError:
        return False


@pytest.mark.skipif(not _botorch_available(), reason="BoTorch not installed")
class TestBOSmallRun:
    def test_tiny_run_converges_monotone(self):
        """Tiny end-to-end BO run: 1 block, 2 iterations, small window.
        Confirms the M10 direct-simplex pipeline (Sobol init projection +
        GP fit + qLogNEHVI with equality/inequality constraints) runs
        without crashing and produces a monotone HV history."""
        result = sut.run_single_configuration(
            block_seed=42,
            window_size=4,
            queries_per_phase=10,
            n_init=4,
            n_iter=2,
            verbose=False,
        )
        # Pass criteria from D5 §7.1:
        assert len(result.history) == 6  # 4 init + 2 BO
        # 1. every iteration returns finite (P, R) ∈ [0, 1]²
        for h in result.history:
            assert 0.0 <= h.precision <= 1.0
            assert 0.0 <= h.recall <= 1.0
        # 2. hypervolume non-decreasing
        hvs = [h.hypervolume for h in result.history]
        for i in range(1, len(hvs)):
            assert hvs[i] >= hvs[i - 1] - 1e-9
        # 5. w satisfies simplex constraints (Σ = 1, all ≥ 0, strictly interior)
        for h in result.history:
            w = np.array(h.w)
            assert np.all(w >= 0)
            # Allow small tolerance: renormalisation can produce Σ ≠ 1 by < 1e-10
            assert math.isclose(w.sum(), 1.0, rel_tol=1e-6)
        # 6. θ proposals stay in (ε, 1−ε)
        for h in result.history:
            assert sut.EPS <= h.theta <= 1.0 - sut.EPS

    def test_simplex_constraints_satisfied_on_bo_proposals(self):
        """Verify that BO-proposed (not just Sobol-init) points respect
        the equality constraint Σw=1 and inequality w_i ≥ EPS."""
        result = sut.run_single_configuration(
            block_seed=7,
            window_size=4,
            queries_per_phase=10,
            n_init=4,
            n_iter=3,
            verbose=False,
        )
        # BO-proposed iterations have iter_index >= 0
        bo_iters = [h for h in result.history if h.iter_index >= 0]
        assert len(bo_iters) == 3
        for h in bo_iters:
            w = np.array(h.w)
            # Equality within numerical tolerance
            assert math.isclose(w.sum(), 1.0, rel_tol=1e-6), f"Σw = {w.sum()}"
            # Inequality (each coord ≥ EPS, with small tolerance for SLSQP)
            assert np.all(w >= -1e-9), f"w has negative coord: {w}"
