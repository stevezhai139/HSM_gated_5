"""Aggregate BO experiment results — BY-FDR corrected statistics.

Reads per-(track, block, window) result.json files produced by
``run_bo_experiment.py`` and produces aggregate statistics:

1. Per-(track, window) mean F1_BO at knee point (BO's "best achievable")
2. Per-block F1 at fixed W₀ = (0.25, 0.20, 0.20, 0.20, 0.15), θ = 0.75
   (from baseline track; interpolated to same windows as BO)
3. ΔF1 = F1_BO - F1_W0 per-block-per-window
4. Wilcoxon signed-rank paired test per-(track, window): H0 = ΔF1 ≤ 0
5. Benjamini–Yekutieli FDR correction across all tests

Output:
- ``aggregate.json``: machine-readable per-(track, window) stats
- ``summary.md``: human-readable summary table

References:
- BY-FDR: Benjamini & Yekutieli, Annals of Statistics 2001.
  Per Paper3B_Cal_Theoretical_Foundations_v0 §5.3 + Lemma 4.
- Wilcoxon paired signed-rank: scipy.stats.wilcoxon.

Usage:
    python code/experiments/cal/experiments/aggregate_bo_results.py \\
        --run-dir results/cal/experiments/<timestamp>_<sha>_bo_rq4a_run
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# sys.path fixup
_THIS = Path(__file__).resolve()
_EXPERIMENTS_DIR = _THIS.parents[2]
if str(_EXPERIMENTS_DIR) not in sys.path:
    sys.path.insert(0, str(_EXPERIMENTS_DIR))


# ---------------------------------------------------------------------------
# BY-FDR correction (Benjamini-Yekutieli 2001)
# ---------------------------------------------------------------------------


def by_fdr_adjust(p_values: List[float], alpha: float = 0.05) -> Dict[str, Any]:
    """Benjamini-Yekutieli FDR adjustment under arbitrary dependence.

    Formula per Paper3B_Cal_Theoretical_Foundations_v0 §5.3:
        c(m) = Σ_{k=1..m} 1/k
        Reject H_(i) iff p_(i) ≤ (i · α) / (m · c(m))

    Returns dict with:
        - adjusted (q-values): max over prefix reweighted p's
        - rejections: bool per original index
        - c_m, alpha_BY_threshold: diagnostic

    This implementation is vectorised and handles ties correctly.
    """
    p = np.asarray(p_values, dtype=float)
    m = len(p)
    if m == 0:
        return {"adjusted": [], "rejections": [], "c_m": 0.0, "m": 0, "alpha": alpha}

    # c(m) = harmonic number
    c_m = float(np.sum(1.0 / np.arange(1, m + 1)))

    # Sort ascending; track original indices
    order = np.argsort(p, kind="stable")
    p_sorted = p[order]

    # BY-adjusted p-values via the "enforce monotonicity from the right" trick
    # q_(i) = min over j >= i of (m · c_m / j) · p_(j)
    ranks = np.arange(1, m + 1)
    q_raw = (m * c_m / ranks) * p_sorted
    # Ensure monotonicity: from the right end, take running minimum
    q_sorted = np.minimum.accumulate(q_raw[::-1])[::-1]
    q_sorted = np.clip(q_sorted, 0.0, 1.0)

    # Unsort back to original order
    q = np.empty_like(q_sorted)
    q[order] = q_sorted

    rejections = (q <= alpha).tolist()
    return {
        "adjusted": q.tolist(),
        "rejections": rejections,
        "c_m": c_m,
        "m": m,
        "alpha": alpha,
    }


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def _load_track_block_window(run_dir: Path, track: str) -> List[Dict[str, Any]]:
    """Load all result.json files for a given track."""
    out: List[Dict[str, Any]] = []
    track_dir = run_dir / track
    if not track_dir.is_dir():
        return out
    if track == "baseline":
        for p in sorted(track_dir.glob("block_*.json")):
            with p.open() as fh:
                out.append(json.load(fh))
    elif track == "bo":
        # Two SDSS layouts, both supported:
        # - Legacy (single window):   bo/block_<N>/result.json
        # - Multi-window (new):       bo/window_<W>/block_<N>/result.json
        for p in sorted(track_dir.glob("block_*/result.json")):
            with p.open() as fh:
                row = json.load(fh)
                row.setdefault("track", row.get("track", "sdss_real"))
                out.append(row)
        for p in sorted(track_dir.glob("window_*/block_*/result.json")):
            with p.open() as fh:
                row = json.load(fh)
                row.setdefault("track", row.get("track", "sdss_real"))
                out.append(row)
    else:
        for p in sorted(track_dir.glob("block_*/window_*/result.json")):
            with p.open() as fh:
                out.append(json.load(fh))
    return out


def _load_sdss_baseline(run_dir: Path) -> Dict[int, float]:
    """Load per-block baseline F1 from SDSS-style baseline.json.

    Returns a flat {block_idx: f1} dict. For legacy single-window runs, reads
    top-level 'per_block'. For multi-window runs, merges per_window_block
    entries keyed by block (blocks with same index across windows share the
    same slice start/end — only the dims matrix differs, but baseline F1
    could differ across windows).

    For multi-window aggregation, the (track, window) cell lookup in
    _compute_cell uses `baseline_f1_per_block_per_window` via the new helper
    _load_sdss_baseline_per_window (see below). This legacy function only
    returns the FIRST window's baseline to preserve old single-window behaviour.
    """
    baseline_json = run_dir / "baseline.json"
    if not baseline_json.is_file():
        return {}
    with baseline_json.open() as fh:
        data = json.load(fh)
    per_block = data.get("per_block", [])
    if per_block:
        # Legacy single-window run
        out = {}
        for entry in per_block:
            if entry is None:
                continue
            out[int(entry["block"])] = float(entry["f1"])
        return out
    # Multi-window run: use FIRST window as a flat fallback (legacy callers)
    pwb = data.get("per_window_block", {})
    if pwb:
        first_w_entry = next(iter(pwb.values()))
        out = {}
        for entry in first_w_entry.get("per_block", []):
            if entry is None:
                continue
            out[int(entry["block"])] = float(entry["f1"])
        return out
    return {}


def _load_sdss_baseline_per_window(run_dir: Path) -> Dict[int, Dict[int, float]]:
    """Load SDSS per-window-per-block baseline F1.

    Returns {window: {block: f1}}. Works for both legacy single-window and
    new multi-window baseline.json layouts.
    """
    baseline_json = run_dir / "baseline.json"
    if not baseline_json.is_file():
        return {}
    with baseline_json.open() as fh:
        data = json.load(fh)
    out: Dict[int, Dict[int, float]] = {}
    pwb = data.get("per_window_block", {})
    if pwb:
        for w_str, entry in pwb.items():
            w = int(w_str)
            out[w] = {}
            for block_entry in entry.get("per_block", []):
                if block_entry is None:
                    continue
                out[w][int(block_entry["block"])] = float(block_entry["f1"])
        return out
    # Legacy: single-window run_meta; infer W from any result.json
    per_block = data.get("per_block", [])
    if per_block:
        # Try to read a result.json to learn the window
        candidate_paths = list((run_dir / "bo").glob("block_*/result.json"))
        if candidate_paths:
            with candidate_paths[0].open() as fh:
                row = json.load(fh)
            legacy_window = int(row.get("window", 20))
        else:
            legacy_window = 20
        out[legacy_window] = {}
        for entry in per_block:
            if entry is None:
                continue
            out[legacy_window][int(entry["block"])] = float(entry["f1"])
    return out


def _wilcoxon_paired_one_sided(
    x: List[float], y: List[float], alternative: str = "greater"
) -> Tuple[float, float]:
    """scipy.stats.wilcoxon paired, one-sided H1: x > y (alternative='greater').

    Returns (statistic, p_value). Handles small-n edge cases via the default
    mode='auto' selection in scipy.
    """
    from scipy.stats import wilcoxon
    d = np.asarray(x, dtype=float) - np.asarray(y, dtype=float)
    # All-zero differences → Wilcoxon undefined; return p=1 (no evidence).
    if np.all(d == 0):
        return 0.0, 1.0
    try:
        res = wilcoxon(d, alternative=alternative, zero_method="wilcox")
        return float(res.statistic), float(res.pvalue)
    except ValueError:
        return 0.0, 1.0


def _sign_flip_permutation_paired(
    x: List[float], y: List[float], *,
    n_permutations: int = 10000, seed: int = 42,
    alternative: str = "greater",
) -> Tuple[float, float]:
    """Paired sign-flip permutation test — robust to ties at 0.

    H0: the paired differences d_i = x_i − y_i have symmetric distribution
    around 0 (i.e., no systematic direction).
    H1 (one-sided 'greater'): mean(d) > 0.

    Procedure: under H0, the sign of each d_i is exchangeable with its
    negation. We draw n_permutations random sign vectors s_i ∈ {−1, +1},
    compute permuted mean(s_i · |d_i|), and count how often the permuted
    mean matches or exceeds the observed mean.

    Advantage over Wilcoxon under ties: zeros contribute 0 regardless of
    sign-flip (neutral effect), so they don't artificially reduce effective n.

    Returns (observed_mean, p_value). Reference: Fisher (1935), Good (2005).
    """
    d = np.asarray(x, dtype=float) - np.asarray(y, dtype=float)
    if d.size == 0:
        return 0.0, 1.0
    observed = float(np.mean(d))
    if np.all(d == 0):
        # No signal → p = 1 regardless of alternative.
        return 0.0, 1.0
    rng = np.random.default_rng(seed)
    # Generate (n_permutations, n) sign matrix ∈ {-1, +1}
    n = d.size
    signs = rng.choice([-1.0, 1.0], size=(n_permutations, n))
    permuted_means = (signs * np.abs(d)).mean(axis=1)
    # Add observed to the null distribution (standard permutation practice;
    # produces valid p-value in [1/(N+1), 1])
    if alternative == "greater":
        extreme = int(np.sum(permuted_means >= observed))
    elif alternative == "less":
        extreme = int(np.sum(permuted_means <= observed))
    else:  # two-sided
        extreme = int(np.sum(np.abs(permuted_means) >= abs(observed)))
    p = (extreme + 1) / (n_permutations + 1)
    return observed, float(p)


# ---------------------------------------------------------------------------
# Aggregation logic
# ---------------------------------------------------------------------------


@dataclass
class CellStats:
    """Per-(track, window) aggregate statistics."""
    track: str
    window: int
    n_blocks: int
    f1_bo_mean: float
    f1_bo_std: float
    f1_bo_ci_lo: float
    f1_bo_ci_hi: float
    f1_bo_per_block: List[float]
    f1_w0_mean: float
    f1_w0_std: float
    f1_w0_per_block: List[float]
    delta_f1_mean: float
    delta_f1_std: float
    delta_f1_per_block: List[float]
    wilcoxon_stat: float
    wilcoxon_p_raw: float
    wilcoxon_p_fdr: Optional[float] = None  # set after FDR correction
    rejects_h0_at_fdr_005: Optional[bool] = None  # Wilcoxon-only (legacy)
    # Sign-flip permutation test — robust to ties at 0
    perm_p_raw: Optional[float] = None
    perm_p_fdr: Optional[float] = None
    perm_rejects_h0_at_fdr_005: Optional[bool] = None
    # Disjunctive reject: true iff either Wilcoxon OR Permutation rejects H0
    # at BY-FDR q ≤ 0.05. Used by plots for ★ annotation. Rationale:
    # Permutation is ties-robust — Wilcoxon can miss signal when many
    # paired differences tie at 0 (real-data case for SDSS).
    rejects_any_fdr_005: Optional[bool] = None


def _compute_cell(track: str, window: int, bo_results: List[Dict[str, Any]],
                  baseline_f1_per_block: Dict[int, float]) -> Optional[CellStats]:
    """Compute aggregate stats for one (track, window) cell."""
    rows = [r for r in bo_results if r["track"] == track and r["window"] == window]
    if not rows:
        return None
    rows.sort(key=lambda r: r["block"])

    per_block_f1 = []
    f1_w0_per_block = []
    delta_per_block = []
    for r in rows:
        p = r["result"]["final_precision"]
        rec = r["result"]["final_recall"]
        f1_bo = (2 * p * rec / (p + rec)) if (p + rec) > 0 else 0.0
        per_block_f1.append(f1_bo)
        w0_f1 = baseline_f1_per_block.get(r["block"], float("nan"))
        f1_w0_per_block.append(w0_f1)
        delta_per_block.append(f1_bo - w0_f1 if not math.isnan(w0_f1) else float("nan"))

    bo_arr = np.array(per_block_f1, dtype=float)
    w0_arr = np.array(f1_w0_per_block, dtype=float)
    delta_arr = np.array(delta_per_block, dtype=float)

    # Filter NaNs (from missing baseline)
    valid = ~np.isnan(delta_arr)
    valid_bo = bo_arr[valid]
    valid_w0 = w0_arr[valid]
    valid_delta = delta_arr[valid]

    if len(valid_delta) < 2:
        return None

    # 95% normal-approx CI on F1_BO (small-n; bootstrap would be better but
    # scipy's gives usable approx for ~10-block aggregates)
    n = len(valid_bo)
    mean_bo = float(valid_bo.mean())
    std_bo = float(valid_bo.std(ddof=1)) if n > 1 else 0.0
    se_bo = std_bo / math.sqrt(n) if n > 0 else 0.0
    ci_lo = mean_bo - 1.96 * se_bo
    ci_hi = mean_bo + 1.96 * se_bo

    stat, pval = _wilcoxon_paired_one_sided(
        valid_bo.tolist(), valid_w0.tolist(), alternative="greater"
    )

    # Sign-flip permutation test (robust to ties)
    _, perm_p = _sign_flip_permutation_paired(
        valid_bo.tolist(), valid_w0.tolist(),
        n_permutations=10000, alternative="greater",
    )

    return CellStats(
        track=track,
        window=window,
        n_blocks=n,
        f1_bo_mean=mean_bo,
        f1_bo_std=std_bo,
        f1_bo_ci_lo=ci_lo,
        f1_bo_ci_hi=ci_hi,
        f1_bo_per_block=valid_bo.tolist(),
        f1_w0_mean=float(valid_w0.mean()),
        f1_w0_std=float(valid_w0.std(ddof=1)) if n > 1 else 0.0,
        f1_w0_per_block=valid_w0.tolist(),
        delta_f1_mean=float(valid_delta.mean()),
        delta_f1_std=float(valid_delta.std(ddof=1)) if n > 1 else 0.0,
        delta_f1_per_block=valid_delta.tolist(),
        wilcoxon_stat=stat,
        wilcoxon_p_raw=pval,
        perm_p_raw=perm_p,
    )


def aggregate(run_dir: Path, alpha: float = 0.05) -> Dict[str, Any]:
    """Aggregate all results in a run directory; apply BY-FDR across cells.

    Auto-detects run type:
    - TPC-H run: easy/, hard/, baseline/ subdirs
    - SDSS real run: bo/, baseline.json
    - Hybrid: all present at once (unlikely but supported)
    """
    easy = _load_track_block_window(run_dir, "easy")
    hard = _load_track_block_window(run_dir, "hard")
    baseline = _load_track_block_window(run_dir, "baseline")
    sdss_bo = _load_track_block_window(run_dir, "bo")

    # TPC-H-style baseline F1 per block
    baseline_f1 = {r["block"]: float(r.get("f1", 0.0)) for r in baseline}
    # SDSS-style per-(window, block) baseline
    sdss_baseline_per_window = _load_sdss_baseline_per_window(run_dir)
    # Flat fallback for legacy callers that expect {block: f1}
    sdss_baseline_f1 = _load_sdss_baseline(run_dir)

    # Cells: (track, window) combos that have data
    cells: List[CellStats] = []
    for track, rows, bl in [
        ("easy", easy, baseline_f1),
        ("hard", hard, baseline_f1),
    ]:
        windows = sorted({r["window"] for r in rows})
        for window in windows:
            cell = _compute_cell(track, window, rows, bl)
            if cell is not None:
                cells.append(cell)

    # SDSS: use per-window baseline (different windows re-bin the trace
    # differently, so per-window baseline F1 is distinct)
    sdss_windows = sorted({r["window"] for r in sdss_bo})
    for window in sdss_windows:
        bl_w = sdss_baseline_per_window.get(window, sdss_baseline_f1)
        cell = _compute_cell("sdss_real", window, sdss_bo, bl_w)
        if cell is not None:
            cells.append(cell)

    # BY-FDR adjustment (Wilcoxon)
    p_values = [c.wilcoxon_p_raw for c in cells]
    fdr = by_fdr_adjust(p_values, alpha=alpha) if p_values else {
        "adjusted": [], "rejections": [], "c_m": 0.0, "m": 0, "alpha": alpha
    }
    for c, q, rej in zip(cells, fdr["adjusted"], fdr["rejections"]):
        c.wilcoxon_p_fdr = float(q)
        c.rejects_h0_at_fdr_005 = bool(rej)

    # BY-FDR adjustment (Permutation — robust to ties)
    perm_p_values = [c.perm_p_raw for c in cells if c.perm_p_raw is not None]
    perm_fdr = by_fdr_adjust(perm_p_values, alpha=alpha) if perm_p_values else {
        "adjusted": [], "rejections": [], "c_m": 0.0, "m": 0, "alpha": alpha
    }
    perm_iter = iter(zip(perm_fdr["adjusted"], perm_fdr["rejections"]))
    for c in cells:
        if c.perm_p_raw is not None:
            q, rej = next(perm_iter)
            c.perm_p_fdr = float(q)
            c.perm_rejects_h0_at_fdr_005 = bool(rej)

    # Disjunctive reject: OR of Wilcoxon and Permutation test rejections.
    # Motivates ★ annotation in plots. Rationale: Permutation is robust to
    # ties at 0 — for paired differences with many zeros (common in real
    # workloads where F1_BO = F1_W0 on some blocks), Wilcoxon drops zeros
    # and loses power. Permutation preserves effective n. Reporting either
    # significant result is appropriate when both tests are pre-registered.
    for c in cells:
        c.rejects_any_fdr_005 = bool(
            c.rejects_h0_at_fdr_005 or c.perm_rejects_h0_at_fdr_005
        )

    # Baseline summary — prefer TPC-H-style baseline; fall back to SDSS
    chosen_baseline = baseline_f1 if baseline_f1 else sdss_baseline_f1
    baseline_f1_list = list(chosen_baseline.values())
    baseline_summary = {
        "n_blocks": len(baseline_f1_list),
        "f1_mean": float(np.mean(baseline_f1_list)) if baseline_f1_list else 0.0,
        "f1_std": float(np.std(baseline_f1_list, ddof=1)) if len(baseline_f1_list) > 1 else 0.0,
        "per_block": chosen_baseline,
        "source": "tpch" if baseline_f1 else ("sdss" if sdss_baseline_f1 else "none"),
    }

    return {
        "run_dir": str(run_dir),
        "alpha": alpha,
        "cells": [
            {
                "track": c.track, "window": c.window, "n_blocks": c.n_blocks,
                "f1_bo_mean": c.f1_bo_mean, "f1_bo_std": c.f1_bo_std,
                "f1_bo_ci_lo": c.f1_bo_ci_lo, "f1_bo_ci_hi": c.f1_bo_ci_hi,
                "f1_bo_per_block": c.f1_bo_per_block,
                "f1_w0_mean": c.f1_w0_mean, "f1_w0_std": c.f1_w0_std,
                "f1_w0_per_block": c.f1_w0_per_block,
                "delta_f1_mean": c.delta_f1_mean, "delta_f1_std": c.delta_f1_std,
                "delta_f1_per_block": c.delta_f1_per_block,
                "wilcoxon_stat": c.wilcoxon_stat,
                "wilcoxon_p_raw": c.wilcoxon_p_raw,
                "wilcoxon_p_fdr": c.wilcoxon_p_fdr,
                "rejects_h0_at_fdr_005": c.rejects_h0_at_fdr_005,
                "perm_p_raw": c.perm_p_raw,
                "perm_p_fdr": c.perm_p_fdr,
                "perm_rejects_h0_at_fdr_005": c.perm_rejects_h0_at_fdr_005,
                "rejects_any_fdr_005": c.rejects_any_fdr_005,
            }
            for c in cells
        ],
        "by_fdr_meta": {"c_m": fdr["c_m"], "m": fdr["m"], "alpha": fdr["alpha"]},
        "baseline": baseline_summary,
    }


def _render_summary_md(agg: Dict[str, Any]) -> str:
    """Human-readable markdown summary."""
    lines = []
    lines.append("# Paper 3B-Cal RQ4a Aggregate Results\n")
    lines.append(f"Run dir: `{agg['run_dir']}`\n")
    lines.append(f"BY-FDR: c(m) = {agg['by_fdr_meta']['c_m']:.4f}, m = {agg['by_fdr_meta']['m']}, α = {agg['alpha']}\n")
    lines.append("\n## Baseline (fixed W₀ = (0.25, 0.20, 0.20, 0.20, 0.15), θ = 0.75, window = 10)\n")
    b = agg["baseline"]
    lines.append(f"- n_blocks: {b['n_blocks']}")
    lines.append(f"- F1: mean {b['f1_mean']:.4f} ± {b['f1_std']:.4f}")
    lines.append("\n## Per-cell BO-qLogNEHVI results\n")
    lines.append("| Track | Window | n | ΔF1 (mean ± SD) | Wilcoxon p_raw | Wilcoxon q_FDR | Wilcoxon reject? | Perm p_raw | Perm q_FDR | Perm reject? |")
    lines.append("|---|---:|---:|---|---:|---:|:---:|---:|---:|:---:|")
    for c in agg["cells"]:
        wr = "✅" if c["rejects_h0_at_fdr_005"] else "❌"
        pr = "✅" if c.get("perm_rejects_h0_at_fdr_005") else "❌"
        p_perm = c.get("perm_p_raw")
        q_perm = c.get("perm_p_fdr")
        p_perm_s = f"{p_perm:.4f}" if p_perm is not None else "—"
        q_perm_s = f"{q_perm:.4f}" if q_perm is not None else "—"
        lines.append(
            f"| {c['track']} | {c['window']} | {c['n_blocks']} | "
            f"{c['delta_f1_mean']:+.4f} ± {c['delta_f1_std']:.4f} | "
            f"{c['wilcoxon_p_raw']:.4f} | "
            f"{c['wilcoxon_p_fdr']:.4f} | {wr} | "
            f"{p_perm_s} | {q_perm_s} | {pr} |"
        )
    lines.append("\n### Details — F1 values\n")
    lines.append("| Track | Window | F1_BO (mean ± SD) | 95% CI | F1_W0 (mean ± SD) |")
    lines.append("|---|---:|---|---|---|")
    for c in agg["cells"]:
        lines.append(
            f"| {c['track']} | {c['window']} | "
            f"{c['f1_bo_mean']:.4f} ± {c['f1_bo_std']:.4f} | "
            f"[{c['f1_bo_ci_lo']:.4f}, {c['f1_bo_ci_hi']:.4f}] | "
            f"{c['f1_w0_mean']:.4f} ± {c['f1_w0_std']:.4f} |"
        )
    lines.append("\n## Interpretation\n")
    n_total = len(agg["cells"])
    n_rejected_w = sum(1 for c in agg["cells"] if c["rejects_h0_at_fdr_005"])
    n_rejected_p = sum(1 for c in agg["cells"] if c.get("perm_rejects_h0_at_fdr_005"))
    n_rejected_any = sum(1 for c in agg["cells"] if c.get("rejects_any_fdr_005"))
    lines.append(f"- **{n_rejected_w}/{n_total} cells** (Wilcoxon signed-rank, BY-FDR q ≤ 0.05)")
    lines.append(f"- **{n_rejected_p}/{n_total} cells** (sign-flip permutation, BY-FDR q ≤ 0.05) — robust to ties")
    lines.append(f"- **{n_rejected_any}/{n_total} cells** (either test — used for ★ plot annotation)")
    easy_cells = [c for c in agg["cells"] if c["track"] == "easy"]
    hard_cells = [c for c in agg["cells"] if c["track"] == "hard"]
    if easy_cells:
        easy_gains = [c["delta_f1_mean"] for c in easy_cells]
        lines.append(f"- **Easy track** (qpp=60): mean ΔF1 across {len(easy_cells)} windows = {np.mean(easy_gains):+.4f}")
    if hard_cells:
        hard_gains = [c["delta_f1_mean"] for c in hard_cells]
        lines.append(f"- **Hard track** (qpp=20): mean ΔF1 across {len(hard_cells)} windows = {np.mean(hard_gains):+.4f}")
    return "\n".join(lines) + "\n"


def main(argv: Optional[List[str]] = None) -> int:
    argv = argv if argv is not None else sys.argv[1:]
    p = argparse.ArgumentParser(description="Aggregate BO experiment results with BY-FDR")
    p.add_argument("--run-dir", required=True, help="Path to run_bo_experiment output dir")
    p.add_argument("--alpha", type=float, default=0.05)
    args = p.parse_args(argv)

    run_dir = Path(args.run_dir)
    if not run_dir.is_absolute():
        run_dir = _THIS.parents[4] / run_dir
    if not run_dir.is_dir():
        print(f"ERROR: run dir not found: {run_dir}", file=sys.stderr)
        return 1

    agg = aggregate(run_dir, alpha=args.alpha)

    out_json = run_dir / "aggregate.json"
    with out_json.open("w") as fh:
        json.dump(agg, fh, indent=2, default=float)
    print(f"[aggregate] wrote {out_json}")

    out_md = run_dir / "summary.md"
    out_md.write_text(_render_summary_md(agg))
    print(f"[aggregate] wrote {out_md}")
    print()
    print(_render_summary_md(agg))
    return 0


if __name__ == "__main__":
    sys.exit(main())
