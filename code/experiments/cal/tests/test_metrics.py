"""Tests for metrics.py.

Phase A primitives (``fpr_fnr`` and ``bootstrap_ci``) are implemented and
expected to PASS. The remaining RQ-specific wrappers (McNemar, Wilcoxon,
KS, Clopper-Pearson, Cohen's d) are still ``NotImplementedError`` stubs
and are marked xfail individually. When each stub is filled in, remove
the corresponding xfail mark.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from cal import metrics


# ------------------------------- fpr_fnr (implemented) ---------------------

class TestFprFnr:
    def test_all_correct_gives_zero_rates(self):
        rates = metrics.fpr_fnr([1, 0, 1, 0], [1, 0, 1, 0])
        assert rates.fpr == 0.0
        assert rates.fnr == 0.0
        assert rates.f1 == 1.0

    def test_all_wrong_gives_unity_rates(self):
        rates = metrics.fpr_fnr([1, 0, 1, 0], [0, 1, 0, 1])
        assert rates.fpr == 1.0
        assert rates.fnr == 1.0

    def test_mixed_case_numeric(self):
        # y_true=[1,1,0,0,1,0], y_pred=[1,0,0,1,1,0]
        # TP=2 (idx 0,4) FN=1 (idx 1) TN=2 (idx 2,5) FP=1 (idx 3)
        rates = metrics.fpr_fnr([1, 1, 0, 0, 1, 0], [1, 0, 0, 1, 1, 0])
        assert rates.fpr == pytest.approx(1 / 3)
        assert rates.fnr == pytest.approx(1 / 3)
        assert rates.precision == pytest.approx(2 / 3)
        assert rates.recall == pytest.approx(2 / 3)
        assert rates.f1 == pytest.approx(2 / 3)

    def test_rejects_non_binary(self):
        with pytest.raises(ValueError, match="binary"):
            metrics.fpr_fnr([0, 1, 2], [0, 1, 0])

    def test_rejects_mismatched_lengths(self):
        with pytest.raises(ValueError, match="same length"):
            metrics.fpr_fnr([0, 1], [0, 1, 0])

    def test_empty_input_returns_zeros(self):
        rates = metrics.fpr_fnr([], [])
        assert rates.fpr == 0.0
        assert rates.fnr == 0.0
        assert rates.f1 == 0.0

    def test_all_negatives_no_predicted_positives(self):
        # y_true=[0,0,0], y_pred=[0,0,0] — no positives at all
        rates = metrics.fpr_fnr([0, 0, 0], [0, 0, 0])
        assert rates.fpr == 0.0
        assert rates.fnr == 0.0
        assert rates.precision == 0.0
        assert rates.recall == 0.0
        assert rates.f1 == 0.0


# ------------------------------- bootstrap_ci (implemented) ----------------

class TestBootstrapCi:
    def test_returns_interval(self):
        lo, hi = metrics.bootstrap_ci([1.0, 2.0, 3.0, 4.0, 5.0], seed=42)
        assert lo <= hi

    def test_reproducible_with_seed(self):
        vals = [0.1, 0.5, 0.3, 0.9, 0.2, 0.7]
        lo1, hi1 = metrics.bootstrap_ci(vals, n_resamples=500, seed=17)
        lo2, hi2 = metrics.bootstrap_ci(vals, n_resamples=500, seed=17)
        assert (lo1, hi1) == (lo2, hi2)

    def test_ci_contains_sample_mean_for_mean_statistic(self):
        rng = np.random.default_rng(7)
        vals = rng.normal(loc=5.0, scale=1.0, size=200)
        lo, hi = metrics.bootstrap_ci(vals, statistic="mean", n_resamples=1000, seed=1)
        sample_mean = float(vals.mean())
        assert lo <= sample_mean <= hi

    def test_median_statistic(self):
        lo, hi = metrics.bootstrap_ci(
            [1.0, 2.0, 3.0, 4.0, 5.0],
            statistic="median",
            n_resamples=500,
            seed=3,
        )
        assert lo <= 3.0 <= hi

    def test_callable_statistic(self):
        # Trivial custom statistic: range (max - min)
        lo, hi = metrics.bootstrap_ci(
            [0.0, 10.0, 5.0, 3.0, 7.0],
            statistic=lambda x: float(x.max() - x.min()),
            n_resamples=300,
            seed=5,
        )
        assert 0.0 <= lo <= hi
        assert hi <= 10.0  # range cannot exceed max - min of the sample

    def test_confidence_bounds(self):
        rng = np.random.default_rng(9)
        vals = rng.normal(size=100)
        lo90, hi90 = metrics.bootstrap_ci(
            vals, n_resamples=1000, confidence_level=0.90, seed=11
        )
        lo99, hi99 = metrics.bootstrap_ci(
            vals, n_resamples=1000, confidence_level=0.99, seed=11
        )
        # Wider confidence level → wider interval (point-estimate property).
        assert (hi99 - lo99) >= (hi90 - lo90)

    def test_rejects_empty(self):
        with pytest.raises(ValueError, match="empty"):
            metrics.bootstrap_ci([], seed=0)

    def test_rejects_unknown_statistic(self):
        with pytest.raises(ValueError, match="unknown statistic"):
            metrics.bootstrap_ci([1.0, 2.0], statistic="not_a_statistic", seed=0)

    def test_rejects_bad_confidence_level(self):
        with pytest.raises(ValueError, match="confidence_level"):
            metrics.bootstrap_ci([1.0, 2.0], confidence_level=1.5, seed=0)


# ------------------------------- Still-stubbed helpers (xfail) -------------

_STUB_XFAIL = pytest.mark.xfail(
    raises=NotImplementedError,
    reason="Paper 3B-Cal: wrapper over statsmodels/scipy still stubbed (filled in per-RQ).",
    strict=False,
)


@_STUB_XFAIL
class TestMcNemar:
    def test_returns_statistic_and_pvalue(self):
        stat, p = metrics.mcnemar_test([1, 0, 1, 0], [1, 1, 0, 0], [1, 0, 1, 0])
        assert isinstance(stat, float)
        assert 0.0 <= p <= 1.0


@_STUB_XFAIL
class TestWilcoxon:
    def test_paired(self):
        stat, p = metrics.wilcoxon_paired([1.0, 2.0, 3.0], [0.5, 1.5, 2.5])
        assert 0.0 <= p <= 1.0


@_STUB_XFAIL
class TestKs:
    def test_two_sample(self):
        stat, p = metrics.ks_test([1.0, 2.0], [1.5, 2.5])
        assert 0.0 <= p <= 1.0


@_STUB_XFAIL
class TestClopperPearson:
    def test_returns_interval(self):
        lo, hi = metrics.clopper_pearson_ci(2, 4, confidence_level=0.95)
        assert 0.0 <= lo <= 0.5 <= hi <= 1.0
