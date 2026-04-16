"""
test_metric_properties.py  --  Verifies Lemma 1 (HSM Metric Properties)
========================================================================
Lemma 1 of the supplementary proves that d_HSM(W_A, W_B) = 1 - S_HSM is a
metric, i.e. it satisfies

    P1 (non-negativity)        d(A, B) >= 0
    P2 (identity of indiscern) d(A, A) = 0  and  d(A, B) = 0 <=> A ~ B
    P3 (symmetry)              d(A, B) = d(B, A)
    P4 (triangle inequality)   d(A, C) <= d(A, B) + d(B, C)

This test suite verifies these properties on randomly generated synthetic
windows.  It is a smoke test, not a substitute for the analytic proof, but
it would surface implementation drift between code and theory.

Run with:    pytest -q test_metric_properties.py
"""

from __future__ import annotations

import math
import os
import random
import sys
from typing import List

import numpy as np
import pytest

# allow tests to be run from repo root or from tests/ directly
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, os.pardir, "experiments"))

from hsm_similarity import (             # noqa: E402
    DEFAULT_WEIGHTS,
    WorkloadWindow,
    build_window,
    hsm_score,
    s_a, s_p, s_r, s_t, s_v,
)

# Direct kernel imports so Lemma 1 is verified against hsm_v2_kernel,
# which is what every validation script invokes through
# ``hsm_score_from_features``.  After the 2026-04-14 consolidation
# ``hsm_similarity`` is itself a thin wrapper around these symbols, so
# both call paths should produce identical scores.
from hsm_v2_kernel import (                # noqa: E402
    W0 as KERNEL_W0,
    hsm_score_from_features,
    sa_v2 as kernel_s_a,
    sp_v2 as kernel_s_p,
    sr_v2 as kernel_s_r,
    st_v2 as kernel_s_t,
    sv_v2 as kernel_s_v,
)


# ---------------------------------------------------------------------------
# Synthetic workload generator
# ---------------------------------------------------------------------------
TEMPLATES = [
    "select * from lineitem where l_shipdate < ?",
    "select count(*) from orders where o_orderdate between ? and ?",
    "select c_custkey, sum(l_extendedprice) from customer, orders, lineitem"
    " where c_custkey = o_custkey and l_orderkey = o_orderkey group by c_custkey",
    "select n_name, sum(l_extendedprice) from nation, supplier, lineitem"
    " where s_nationkey = n_nationkey group by n_name",
    "select p_brand, p_size from part where p_container = ?",
    "insert into orders values (?, ?, ?, ?, ?, ?, ?, ?, ?)",
    "update lineitem set l_discount = ? where l_orderkey = ?",
    "delete from customer where c_acctbal < ?",
]


def _rand_window(rng: random.Random, n: int = 32, dur: float = 60.0) -> WorkloadWindow:
    sqls = [rng.choice(TEMPLATES) for _ in range(n)]
    timestamps = sorted(rng.uniform(0.0, dur) for _ in range(n))
    return build_window(sqls, timestamps=timestamps,
                        window_id=rng.randint(0, 1_000_000), duration_s=dur)


# ---------------------------------------------------------------------------
# Property tests
# ---------------------------------------------------------------------------
N_TRIPLES = 60   # number of random triples to test

@pytest.fixture(scope="module")
def rng() -> random.Random:
    return random.Random(2026)


@pytest.fixture(scope="module")
def windows(rng) -> List[WorkloadWindow]:
    return [_rand_window(rng, n=rng.randint(16, 96)) for _ in range(N_TRIPLES * 3)]


def _d(w_a: WorkloadWindow, w_b: WorkloadWindow) -> float:
    score, _ = hsm_score(w_a, w_b)
    return 1.0 - score


def test_p1_non_negativity(windows):
    """d(A, B) >= 0 for all pairs."""
    for a in windows[:10]:
        for b in windows[:10]:
            d = _d(a, b)
            assert d >= -1e-9, f"non-negativity violated: d={d}"


def test_p2a_self_distance_zero(windows):
    """d(A, A) = 0.

    A small floating-point slack is allowed because the angular-distance
    component S_T rounds a cosine through arccos; on self-pairs the
    argument lands on 1.0 only up to ~1e-15 error, which is amplified
    by arccos's infinite derivative near 1.  1e-6 is well below the
    discrimination scale of d_HSM.
    """
    for w in windows[:20]:
        assert _d(w, w) == pytest.approx(0.0, abs=1e-6)


def test_p3_symmetry(windows):
    """d(A, B) = d(B, A)."""
    for a, b in zip(windows[:25], windows[25:50]):
        d_ab = _d(a, b)
        d_ba = _d(b, a)
        assert d_ab == pytest.approx(d_ba, abs=1e-9), \
            f"symmetry violated: d_ab={d_ab}, d_ba={d_ba}"


def test_p4_triangle_inequality(windows):
    """d(A, C) <= d(A, B) + d(B, C) for all triples (mod numerical slack).

    Note: FastDTW is not a strict metric in general, so a tiny tolerance
    is allowed for the S_P component.  An eps of 1e-6 is well within
    floating-point noise for the band-weighted average defined in Eq. 5.
    """
    eps = 1e-6
    triples = [(windows[3*i], windows[3*i+1], windows[3*i+2])
               for i in range(N_TRIPLES)]
    for a, b, c in triples:
        d_ab = _d(a, b)
        d_bc = _d(b, c)
        d_ac = _d(a, c)
        assert d_ac <= d_ab + d_bc + eps, (
            f"triangle inequality violated: "
            f"d(A,C)={d_ac:.6f} > d(A,B)+d(B,C)={d_ab + d_bc:.6f}"
        )


def test_score_in_unit_interval(windows):
    """0 <= S_HSM <= 1 (Eq. 6 + bounded components)."""
    for a in windows[:10]:
        for b in windows[:10]:
            score, dims = hsm_score(a, b)
            assert -1e-9 <= score <= 1.0 + 1e-9, f"score out of range: {score}"
            for k, v in dims.items():
                assert -1e-9 <= v <= 1.0 + 1e-9, f"{k} out of range: {v}"


def test_default_weights_sum_to_one():
    s = sum(DEFAULT_WEIGHTS.values())
    assert s == pytest.approx(1.0, abs=1e-9), f"weights sum to {s}, not 1"


# ---------------------------------------------------------------------------
# Component-level metric checks
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("fn", [s_r, s_v, s_t, s_a])
def test_components_in_unit_interval(windows, fn):
    for a in windows[:8]:
        for b in windows[:8]:
            v = fn(a, b)
            assert -1e-9 <= v <= 1.0 + 1e-9, \
                f"{fn.__name__} out of range: {v}"


def test_s_v_log_ratio_property():
    """S_V = min(Q_a, Q_b) / max(Q_a, Q_b) by definition."""
    rng = random.Random(0)
    for _ in range(50):
        n_a = rng.randint(10, 1000)
        n_b = rng.randint(10, 1000)
        # Build minimal windows with the requested COUNT (paper §III-B
        # "Volume Q = query count per window"; duration_s is irrelevant
        # to S_V after the 2026-04-14 paper-alignment fix).
        w_a = WorkloadWindow(queries=[], window_id=0, duration_s=1.0)
        w_b = WorkloadWindow(queries=[], window_id=0, duration_s=1.0)
        from hsm_similarity import QueryFeatures
        w_a.queries = [QueryFeatures(raw_sql="x") for _ in range(n_a)]
        w_b.queries = [QueryFeatures(raw_sql="x") for _ in range(n_b)]
        expected = min(n_a, n_b) / max(n_a, n_b)
        assert s_v(w_a, w_b) == pytest.approx(expected, abs=1e-9)


# ---------------------------------------------------------------------------
# Direct hsm_v2_kernel verification
# ---------------------------------------------------------------------------
# Lemma 1 (P1--P4) is verified on the SAME kernel that every validation
# script invokes through ``hsm_score_from_features``.  These tests build
# feature dicts directly so they exercise the kernel's own code paths,
# bypassing the WorkloadWindow wrapper.
def _rand_feature_dict(rng: random.Random, n: int = 32, dur: float = 60.0) -> dict:
    """Synthetic feature dict mirroring what validation scripts produce."""
    sqls = [rng.choice(TEMPLATES) for _ in range(n)]
    template_counts: dict = {}
    tables: set = set()
    cols:   set = set()
    for sql in sqls:
        tpl = sql  # SQL strings here are already unique-per-template
        template_counts[tpl] = template_counts.get(tpl, 0) + 1
        for tbl in ("lineitem", "orders", "customer", "nation",
                    "supplier", "part"):
            if tbl in sql:
                tables.add(tbl)
        for prefix in ("l_", "o_", "c_", "n_", "s_", "p_"):
            for tok in sql.split():
                if tok.startswith(prefix) and "." not in tok:
                    cols.add(tok)
    times = np.array(sorted(rng.uniform(0.0, dur) for _ in range(n)))
    return {
        "freq_map": template_counts,
        "n":        n,
        "tables":   tables,
        "cols":     cols,
        "times":    times,
        "qset":     set(template_counts.keys()),
    }


@pytest.fixture(scope="module")
def feature_dicts(rng) -> List[dict]:
    return [_rand_feature_dict(rng, n=rng.randint(16, 96)) for _ in range(N_TRIPLES * 3)]


def _dk(fa: dict, fb: dict) -> float:
    score, _ = hsm_score_from_features(fa, fb)
    return 1.0 - score


def test_kernel_p1_non_negativity(feature_dicts):
    for a in feature_dicts[:10]:
        for b in feature_dicts[:10]:
            assert _dk(a, b) >= -1e-9


def test_kernel_p2_self_distance_zero(feature_dicts):
    for f in feature_dicts[:20]:
        assert _dk(f, f) == pytest.approx(0.0, abs=1e-6)


def test_kernel_p3_symmetry(feature_dicts):
    for a, b in zip(feature_dicts[:25], feature_dicts[25:50]):
        assert _dk(a, b) == pytest.approx(_dk(b, a), abs=1e-9)


def test_kernel_p4_triangle_inequality(feature_dicts):
    eps = 1e-6
    triples = [(feature_dicts[3*i], feature_dicts[3*i+1], feature_dicts[3*i+2])
               for i in range(N_TRIPLES)]
    for a, b, c in triples:
        d_ab = _dk(a, b)
        d_bc = _dk(b, c)
        d_ac = _dk(a, c)
        assert d_ac <= d_ab + d_bc + eps, (
            f"kernel triangle inequality violated: "
            f"d(A,C)={d_ac:.6f} > d(A,B)+d(B,C)={d_ab + d_bc:.6f}"
        )


def test_kernel_score_in_unit_interval(feature_dicts):
    for a in feature_dicts[:10]:
        for b in feature_dicts[:10]:
            score, dims = hsm_score_from_features(a, b)
            assert -1e-9 <= score <= 1.0 + 1e-9
            for k, v in dims.items():
                assert -1e-9 <= v <= 1.0 + 1e-9, f"{k} out of range: {v}"


def test_kernel_default_weights_sum_to_one():
    s = sum(KERNEL_W0.values())
    assert s == pytest.approx(1.0, abs=1e-9), f"W0 sums to {s}, not 1"


def test_kernel_s_v_count_based():
    """Kernel S_V = min(n_a, n_b) / max(n_a, n_b) -- count, not qps."""
    rng = random.Random(7)
    for _ in range(50):
        n_a = rng.randint(1, 1000)
        n_b = rng.randint(1, 1000)
        expected = min(n_a, n_b) / max(n_a, n_b)
        assert kernel_s_v(n_a, n_b) == pytest.approx(expected, abs=1e-9)


def test_wrapper_matches_kernel(feature_dicts):
    """The hsm_similarity wrapper must agree with the kernel exactly,
    so there is one source of truth across the codebase."""
    for fa, fb in zip(feature_dicts[:8], feature_dicts[8:16]):
        # Build wrappers that mirror the feature dict so both code paths
        # see identical content.
        from hsm_similarity import build_window
        wa_sqls = [k for k, c in fa["freq_map"].items() for _ in range(c)]
        wb_sqls = [k for k, c in fb["freq_map"].items() for _ in range(c)]
        ts_a = list(np.linspace(0.0, 60.0, max(len(wa_sqls), 1)))
        ts_b = list(np.linspace(0.0, 60.0, max(len(wb_sqls), 1)))
        w_a = build_window(wa_sqls, timestamps=ts_a, duration_s=60.0)
        w_b = build_window(wb_sqls, timestamps=ts_b, duration_s=60.0)
        score_wrap, _ = hsm_score(w_a, w_b)
        # Kernel via feature dict (the path used by validation scripts):
        # build matching feature dicts from the wrapper so wrapper and
        # kernel see the same vocabulary.
        from collections import Counter
        fa_kernel = {
            "freq_map": dict(w_a.template_freq),
            "n":        w_a.count,
            "tables":   w_a.tables,
            "cols":     w_a.columns,
            "times":    w_a.arrival_series(n_buckets=64),
            "qset":     set(w_a.template_freq.keys()),
        }
        fb_kernel = {
            "freq_map": dict(w_b.template_freq),
            "n":        w_b.count,
            "tables":   w_b.tables,
            "cols":     w_b.columns,
            "times":    w_b.arrival_series(n_buckets=64),
            "qset":     set(w_b.template_freq.keys()),
        }
        score_kernel, _ = hsm_score_from_features(fa_kernel, fb_kernel)
        assert score_wrap == pytest.approx(score_kernel, abs=1e-9)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
