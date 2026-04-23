"""
hsm_bridge.py — Single source of truth for the HSM detector across engines.

Imports `hsm_v2` and the 5 dimensional primitives directly from
`experiments/v2_10seed/hsm_v2_core.py` so the cross-engine adaptation loops
(Mongo, MySQL fallback) use the EXACT same DWT(db4,L=3)+SAX(α=4)+FastDTW(r=3)
pipeline plus identical W0 weights as the Postgres step 5 experiment.

Semantics (matched to Postgres `07_adaptation_comparison.py`):
  • `hsm_v2(...)` returns SIMILARITY in [0,1]: 1.0 = identical windows.
  • Gating decision is `score < THETA`  → invoke advisor when SIMILARITY drops.
  • THETA is the SIMILARITY threshold (Postgres uses θ = 0.75).
  • The full 5-D weighted score is used (R=0.25, V=0.20, T=0.20, A=0.20, P=0.15).

This is the ONLY module that should import sp_v2 / hsm_v2 / W0.  All engine
runners (12_*, 13_*) must go through `compute_window_hsm_breakdown()` and
`compute_window_hsm()` so any future tweak to the detector lands in exactly
one place and is replicated identically across engines.
"""
from __future__ import annotations

import os
import sys
from typing import Dict

import numpy as np

# ── Locate vendored V3 HSM kernel on the path ─────────────────────────
# Vendored layout: cross_engine/_v3_hsm/hsm_v2_core.py is the V3 kernel
# (verbatim snapshot used to produce the cross-engine results in §V).
# It must remain V3's hsm_v2_core, NOT V5's hsm_v2_kernel — the two have
# different sp_v2 signatures and W0 is only exported by V3's version.
_HERE = os.path.dirname(os.path.abspath(__file__))
_V2_DIR = os.path.abspath(os.path.join(_HERE, "..", "_v3_hsm"))
if _V2_DIR not in sys.path:
    sys.path.insert(0, _V2_DIR)

try:
    from hsm_v2_core import (
        hsm_v2,
        sp_v2,
        sv_v2,
        sr_v2,
        st_v2,
        sa_v2,
        W0,
    )
    _AVAILABLE = True
    _IMPORT_ERR = None
except Exception as e:  # pragma: no cover
    _AVAILABLE = False
    _IMPORT_ERR = e
    W0 = {"R": 0.25, "V": 0.20, "T": 0.20, "A": 0.20, "P": 0.15}


def is_available() -> bool:
    """True iff `v2_10seed/hsm_v2_core.py` imported cleanly."""
    return _AVAILABLE


def get_w0() -> Dict[str, float]:
    """Return a copy of the canonical W0 weight dict (R/V/T/A/P)."""
    return dict(W0)


# ── Window-feature contract ──────────────────────────────────────────
# Engine runners must produce window-feature dicts with EXACTLY these keys
# so that `compute_window_hsm_*` can call hsm_v2() positionally:
#
#   {
#     "freq"   : np.ndarray of length len(ALL_QIDS_SORTED) (one-hot frequency)
#     "n"      : int — number of queries in the window
#     "tables" : Set[str] — table/doc-type surrogate (S_A input)
#     "cols"   : Set[str] — column/field surrogate (S_A input)
#     "times"  : np.ndarray — exec_ms time series (S_P input)
#     "qset"   : Set[str]  — distinct qids in the window (S_P fallback)
#   }
#
# Helper: see common/window_features.py.


def compute_window_hsm_breakdown(w_a: dict, w_b: dict) -> Dict[str, float]:
    """Compute the 5-D HSM breakdown {S_R, S_V, S_T, S_A, S_P, HSM} between
    two window-feature dicts. Wraps `hsm_v2_core.hsm_v2` directly.
    """
    if not _AVAILABLE:
        return {"S_R": 0.0, "S_V": 0.0, "S_T": 0.0, "S_A": 0.0, "S_P": 0.0, "HSM": 0.0}
    return hsm_v2(
        w_a["freq"], w_b["freq"],
        w_a["n"],    w_b["n"],
        w_a["tables"], w_b["tables"],
        w_a["cols"],   w_b["cols"],
        w_a["times"],  w_b["times"],
        w_a["qset"],   w_b["qset"],
    )


def compute_window_hsm(w_a: dict, w_b: dict) -> float:
    """Scalar HSM SIMILARITY in [0,1] (1.0 = identical windows).

    Use this directly in the gating predicate:
        if compute_window_hsm(prev, curr) < THETA: invoke_advisor()
    matching the Postgres convention in 07_adaptation_comparison.py:289.
    """
    return float(compute_window_hsm_breakdown(w_a, w_b)["HSM"])


# ── Self-check ────────────────────────────────────────────────────────

def _self_check() -> None:
    print(f"hsm_v2_core available: {_AVAILABLE}")
    if not _AVAILABLE:
        print(f"  import error: {_IMPORT_ERR}")
        return
    print(f"  W0 weights: {W0}")

    # Build two synthetic window-feature dicts that differ on every dim,
    # then verify HSM(identical) = 1.0 and HSM(different) < HSM(identical).
    rng = np.random.default_rng(0)
    N = 40
    K_QIDS = 5

    def make_window(qid_choice, time_pattern):
        freq = np.zeros(K_QIDS, dtype=float)
        for q in qid_choice:
            freq[q] += 1
        if freq.sum() > 0:
            freq /= freq.sum()
        return {
            "freq":   freq,
            "n":      len(qid_choice),
            "tables": {f"t{q}" for q in set(qid_choice)},
            "cols":   {f"c{q}" for q in set(qid_choice)},
            "times":  time_pattern.astype(float),
            "qset":   {f"Q{q}" for q in set(qid_choice)},
        }

    t = np.linspace(0, 4 * np.pi, N)
    flat_a    = make_window([0, 0, 0, 0, 0],
                            np.full(N, 100.0) + rng.normal(0, 1, N))
    flat_b    = make_window([0, 0, 0, 0, 0],
                            np.full(N, 100.0) + rng.normal(0, 1, N))
    different = make_window([1, 2, 3, 4, 1],
                            (100 + 50 * np.sin(t)).astype(float))

    sim_same = compute_window_hsm(flat_a, flat_b)
    sim_diff = compute_window_hsm(flat_a, different)
    bd_same  = compute_window_hsm_breakdown(flat_a, flat_b)
    bd_diff  = compute_window_hsm_breakdown(flat_a, different)
    print(f"  HSM(flat,flat) = {sim_same:.4f}  (should be ~1.0)")
    print(f"    breakdown   : {bd_same}")
    print(f"  HSM(flat,diff) = {sim_diff:.4f}  (should be < HSM(flat,flat))")
    print(f"    breakdown   : {bd_diff}")
    assert sim_same > sim_diff, \
        f"identical windows ({sim_same}) should score higher than different ({sim_diff})"
    assert 0.0 <= sim_diff <= 1.0
    assert 0.0 <= sim_same <= 1.0
    print("OK")


if __name__ == "__main__":
    _self_check()
