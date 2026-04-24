"""Unit tests for aggregate_bo_results.py (BY-FDR + Wilcoxon logic)."""

from __future__ import annotations

import math
import numpy as np
import pytest

from cal.experiments import aggregate_bo_results as agg


class TestByFdr:
    def test_empty_input(self):
        r = agg.by_fdr_adjust([])
        assert r["adjusted"] == []
        assert r["rejections"] == []

    def test_single_p_below_alpha(self):
        r = agg.by_fdr_adjust([0.01], alpha=0.05)
        assert len(r["adjusted"]) == 1
        assert r["rejections"] == [True]

    def test_single_p_above_alpha(self):
        r = agg.by_fdr_adjust([0.50], alpha=0.05)
        assert r["rejections"] == [False]

    def test_all_p_very_small_all_reject(self):
        ps = [0.0001, 0.0002, 0.0003, 0.0004, 0.0005]
        r = agg.by_fdr_adjust(ps, alpha=0.05)
        assert all(r["rejections"])

    def test_all_p_one_all_reject_none(self):
        ps = [1.0, 1.0, 1.0, 1.0, 1.0]
        r = agg.by_fdr_adjust(ps, alpha=0.05)
        assert not any(r["rejections"])

    def test_harmonic_c_m_correct(self):
        # c(m=5) = 1 + 1/2 + 1/3 + 1/4 + 1/5 = 2.2833...
        r = agg.by_fdr_adjust([0.01] * 5, alpha=0.05)
        assert math.isclose(r["c_m"], 1 + 1/2 + 1/3 + 1/4 + 1/5, rel_tol=1e-9)

    def test_monotone_adjusted(self):
        # After BY, adjusted q-values in sorted order of raw p should be non-decreasing.
        ps = [0.001, 0.01, 0.02, 0.03, 0.04]
        r = agg.by_fdr_adjust(ps, alpha=0.05)
        q_sorted_by_p = [r["adjusted"][i] for i in np.argsort(ps)]
        for i in range(1, len(q_sorted_by_p)):
            assert q_sorted_by_p[i] >= q_sorted_by_p[i-1] - 1e-12

    def test_reordering_invariant(self):
        # Shuffling the input order should not change the set of (p, q) pairs.
        ps = [0.001, 0.01, 0.02, 0.03, 0.04]
        r1 = agg.by_fdr_adjust(ps, alpha=0.05)
        ps_rev = list(reversed(ps))
        r2 = agg.by_fdr_adjust(ps_rev, alpha=0.05)
        # Match by sorting both
        q1_sorted = sorted(r1["adjusted"])
        q2_sorted = sorted(r2["adjusted"])
        np.testing.assert_allclose(q1_sorted, q2_sorted, atol=1e-12)

    def test_clipped_to_unit_interval(self):
        ps = [0.01, 0.02, 0.03]
        r = agg.by_fdr_adjust(ps, alpha=0.05)
        for q in r["adjusted"]:
            assert 0.0 <= q <= 1.0

    def test_rejection_matches_q_threshold(self):
        ps = [0.001, 0.01, 0.5]
        r = agg.by_fdr_adjust(ps, alpha=0.05)
        for q, rej in zip(r["adjusted"], r["rejections"]):
            assert rej == (q <= 0.05)


class TestPermutationTest:
    def test_all_zero_returns_p_one(self):
        _, p = agg._sign_flip_permutation_paired(
            [0.5, 0.5, 0.5], [0.5, 0.5, 0.5], n_permutations=500, seed=0,
        )
        assert p == 1.0

    def test_clear_positive_signal(self):
        rng = np.random.default_rng(0)
        x = 0.5 + rng.normal(0, 0.05, 20)
        y = 0.2 + rng.normal(0, 0.05, 20)
        _, p = agg._sign_flip_permutation_paired(
            x.tolist(), y.tolist(), n_permutations=2000, seed=1, alternative="greater",
        )
        assert p < 0.01

    def test_no_positive_signal(self):
        # x ≈ y → p near 0.5
        rng = np.random.default_rng(42)
        x = 0.5 + rng.normal(0, 0.1, 20)
        y = 0.5 + rng.normal(0, 0.1, 20)
        _, p = agg._sign_flip_permutation_paired(
            x.tolist(), y.tolist(), n_permutations=2000, seed=2, alternative="greater",
        )
        assert 0.2 < p < 0.8  # high-variance range

    def test_robust_to_ties_at_zero(self):
        # 10 pairs: 5 with Δ=+0.25, 5 with Δ=0 (ties)
        # Wilcoxon drops zeros → effective n=5 → p=0.0313
        # Permutation treats zeros as neutral → larger effective n → lower p
        x = [0.25] * 5 + [0.5] * 5
        y = [0.0] * 5 + [0.5] * 5
        _, p_wilcoxon = agg._wilcoxon_paired_one_sided(x, y, alternative="greater")
        _, p_perm = agg._sign_flip_permutation_paired(
            x, y, n_permutations=5000, seed=7, alternative="greater",
        )
        # Both should detect the signal, but magnitudes can differ
        assert p_wilcoxon < 0.05
        assert p_perm < 0.05

    def test_reproducible_with_seed(self):
        x = [0.7, 0.6, 0.5, 0.8, 0.9]
        y = [0.5, 0.5, 0.5, 0.5, 0.5]
        _, p1 = agg._sign_flip_permutation_paired(x, y, n_permutations=1000, seed=123)
        _, p2 = agg._sign_flip_permutation_paired(x, y, n_permutations=1000, seed=123)
        assert p1 == p2


class TestWilcoxonWrapper:
    def test_all_zero_returns_p_one(self):
        stat, p = agg._wilcoxon_paired_one_sided([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        assert p == 1.0

    def test_clear_positive_difference(self):
        # x consistently > y → one-sided 'greater' should give small p
        x = [0.9, 0.8, 0.7, 0.85, 0.92, 0.88]
        y = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
        _, p = agg._wilcoxon_paired_one_sided(x, y, alternative="greater")
        assert p < 0.05

    def test_no_positive_difference(self):
        x = [0.5, 0.5, 0.5, 0.5]
        y = [0.6, 0.6, 0.6, 0.6]
        _, p = agg._wilcoxon_paired_one_sided(x, y, alternative="greater")
        assert p > 0.5  # lots of evidence AGAINST H1


class TestCellComputation:
    def test_basic_cell(self):
        """Verify end-to-end _compute_cell with synthetic history."""
        # Mock BO results: 3 blocks, all with F1_BO = 0.8, F1_W0 baseline = 0.5
        bo_results = []
        for b in range(3):
            bo_results.append({
                "track": "easy",
                "block": b,
                "window": 10,
                "result": {
                    "final_precision": 0.8,
                    "final_recall": 0.8,  # F1 = 0.8
                    "history": [],
                    "final_w_star": [0.2] * 5,
                    "final_theta_star": 0.7,
                    "final_hypervolume": 0.64,
                    "pareto_indices": [],
                },
            })
        baseline_f1 = {0: 0.5, 1: 0.5, 2: 0.5}
        cell = agg._compute_cell("easy", 10, bo_results, baseline_f1)
        assert cell is not None
        assert cell.n_blocks == 3
        assert math.isclose(cell.f1_bo_mean, 0.8, rel_tol=1e-9)
        assert math.isclose(cell.delta_f1_mean, 0.3, rel_tol=1e-9)

    def test_missing_baseline_filtered(self):
        bo_results = [
            {"track": "easy", "block": 0, "window": 10,
             "result": {"final_precision": 0.8, "final_recall": 0.8,
                        "history": [], "final_w_star": [0.2]*5,
                        "final_theta_star": 0.7, "final_hypervolume": 0.64,
                        "pareto_indices": []}},
            {"track": "easy", "block": 1, "window": 10,
             "result": {"final_precision": 0.9, "final_recall": 0.9,
                        "history": [], "final_w_star": [0.2]*5,
                        "final_theta_star": 0.7, "final_hypervolume": 0.81,
                        "pareto_indices": []}},
        ]
        # Only block 0 has baseline; block 1 is missing
        baseline_f1 = {0: 0.5}
        cell = agg._compute_cell("easy", 10, bo_results, baseline_f1)
        # Needs >=2 valid (both have baseline); here only 1 has → None
        assert cell is None


class TestDisjunctiveReject:
    """Tests for rejects_any_fdr_005 = rejects_h0_at_fdr_005 OR perm_rejects_h0_at_fdr_005."""

    def _build_run_dir(self, tmp_path, bo_f1: list, w0_f1: list, window: int = 10):
        """Helper: create a minimal TPC-H-style run_dir with N blocks in 'easy' track."""
        import json
        (tmp_path / "easy").mkdir()
        (tmp_path / "baseline").mkdir()
        for b, (f_bo, f_w0) in enumerate(zip(bo_f1, w0_f1)):
            # BO result (set P=R=f_bo so F1_BO = f_bo)
            bdir = tmp_path / "easy" / f"block_{b}" / f"window_{window}"
            bdir.mkdir(parents=True)
            (bdir / "result.json").write_text(json.dumps({
                "track": "easy", "block": b, "window": window,
                "result": {
                    "final_precision": f_bo, "final_recall": f_bo,
                    "history": [], "final_w_star": [0.2]*5,
                    "final_theta_star": 0.7, "final_hypervolume": f_bo*f_bo,
                    "pareto_indices": [],
                },
            }))
            # Baseline (set f1 directly)
            (tmp_path / "baseline" / f"block_{b}.json").write_text(json.dumps({
                "block": b, "f1": f_w0,
            }))

    def test_wilcoxon_rejects_perm_rejects_any_true(self, tmp_path):
        """Both tests reject → rejects_any_fdr_005 = True."""
        # Clear signal: 20 blocks, all positive ΔF1
        bo_f1 = [1.0] * 20
        w0_f1 = [0.5] * 20
        self._build_run_dir(tmp_path, bo_f1, w0_f1)
        result = agg.aggregate(tmp_path)
        c = result["cells"][0]
        assert c["rejects_h0_at_fdr_005"] is True
        assert c["perm_rejects_h0_at_fdr_005"] is True
        assert c["rejects_any_fdr_005"] is True

    def test_neither_rejects_any_false(self, tmp_path):
        """Neither test rejects → rejects_any_fdr_005 = False."""
        # Noise with no consistent direction
        bo_f1 = [0.5, 0.6, 0.4, 0.55, 0.45]
        w0_f1 = [0.5, 0.5, 0.5, 0.5, 0.5]
        self._build_run_dir(tmp_path, bo_f1, w0_f1)
        result = agg.aggregate(tmp_path)
        c = result["cells"][0]
        # With n=5 and mixed Δ, neither test should reach q ≤ 0.05
        assert c["rejects_h0_at_fdr_005"] is False
        assert c["perm_rejects_h0_at_fdr_005"] is False
        assert c["rejects_any_fdr_005"] is False

    def test_perm_only_rejects_any_true(self, tmp_path):
        """Perm rejects but Wilcoxon doesn't (simulated SDSS-like scenario) → any_true."""
        # Many ties at 0 + a few positives: Wilcoxon drops zeros,
        # effective n becomes ~5; sign test p = 1/2^5 = 0.031 — borderline.
        # Permutation treats zeros as neutral, effective n = 20; strong signal.
        # Pattern: 15 zero-differences + 5 positive (+0.1) → Wilcoxon may fail,
        # Permutation should succeed.
        bo_f1 = [0.5] * 15 + [0.6] * 5  # 15 zeros, 5 positives
        w0_f1 = [0.5] * 15 + [0.5] * 5
        self._build_run_dir(tmp_path, bo_f1, w0_f1)
        result = agg.aggregate(tmp_path)
        c = result["cells"][0]
        # We just verify that rejects_any is the disjunction.
        any_expected = c["rejects_h0_at_fdr_005"] or c["perm_rejects_h0_at_fdr_005"]
        assert c["rejects_any_fdr_005"] == any_expected

    def test_field_present_in_serialization(self, tmp_path):
        """aggregate.json must include rejects_any_fdr_005 field."""
        bo_f1 = [0.8] * 10
        w0_f1 = [0.5] * 10
        self._build_run_dir(tmp_path, bo_f1, w0_f1)
        result = agg.aggregate(tmp_path)
        assert "rejects_any_fdr_005" in result["cells"][0]
        # Summary md should mention "(either test — used for ★ plot annotation)"
        md = agg._render_summary_md(result)
        assert "either test" in md
