"""Tests for metrics.py — marked xfail on NotImplementedError until filled in."""

from __future__ import annotations

import pytest

from cal import metrics


pytestmark = pytest.mark.xfail(
    raises=NotImplementedError,
    reason="Paper 3B-Cal scaffolding — implementations pending",
)


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


class TestMcNemar:
    def test_returns_statistic_and_pvalue(self):
        stat, p = metrics.mcnemar_test([1, 0, 1, 0], [1, 1, 0, 0], [1, 0, 1, 0])
        assert isinstance(stat, float)
        assert 0.0 <= p <= 1.0


class TestWilcoxon:
    def test_paired(self):
        stat, p = metrics.wilcoxon_paired([1.0, 2.0, 3.0], [0.5, 1.5, 2.5])
        assert 0.0 <= p <= 1.0


class TestKs:
    def test_two_sample(self):
        stat, p = metrics.ks_test([1.0, 2.0], [1.5, 2.5])
        assert 0.0 <= p <= 1.0


class TestBootstrapCi:
    def test_returns_interval(self):
        lo, hi = metrics.bootstrap_ci([1.0, 2.0, 3.0, 4.0, 5.0], seed=42)
        assert lo <= hi


class TestClopperPearson:
    def test_returns_interval(self):
        lo, hi = metrics.clopper_pearson_ci(2, 4, confidence_level=0.95)
        assert 0.0 <= lo <= 0.5 <= hi <= 1.0
