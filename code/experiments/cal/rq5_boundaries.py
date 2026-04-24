"""RQ5 — Boundary conditions and failure analysis (foundational RQ).

This RQ runs FIRST because it establishes W_min which every other RQ uses.

Usage (smoke — runs on the in-repo ``SkyLog_Workload.csv`` stand-in that
Paper 3A already shipped; DR18 loader lands when access is granted):

    # Window-size sweep to locate W_min
    python code/experiments/cal/rq5_boundaries.py \\
        --mode window_sweep --workload sdss_sim \\
        --window-grid 5,10,20,50,100 \\
        --out results/cal/rq5_window_sweep_sdss.json

    # Failure scenarios (Phase C — not implemented yet)
    python code/experiments/cal/rq5_boundaries.py \\
        --mode scenarios --workload sdss_sim \\
        --scenarios micro_shift,syntactic_semantic_mismatch \\
        --out results/cal/rq5_scenarios_sdss.json

Design notes (see ``Paper3B_Cal_RQs_v0.md`` §RQ5 for full spec):

* **Ground truth for smoke runs.** RQ5/RQ1 define the ground-truth shift
  label via k-means(k=2) on per-query *index-quality* vectors. We do not
  have Postgres index-quality data in the smoke dataset, so this driver
  uses a structural-feature surrogate (template / table / column
  histograms per window) — same k-means(k=2) recipe, different feature
  space. When DR18 + index-quality pipeline lands, swap the
  ``ground_truth_labels`` function; the rest of the pipeline does not
  change. The JSON output records ``ground_truth_source`` so downstream
  analysis can tell smoke runs apart from definitive runs.
* **Broken-stick regression.** Inline helper (``fit_broken_stick``). Grid-
  search over candidate break-points in ``log(|W|)`` space; pick the one
  minimising SSR. Bootstrap CI via resampling (|W|, F1-per-bootstrap)
  pairs across both the bootstrap replicates at each window size *and*
  the window sizes themselves.
* **Integrity rules.** No file under Paper 3A's ``v5.0.0-tkde-submission``
  tag is modified. We only *read* the SkyLog CSV and the HSM kernel;
  everything else is defined here under ``code/experiments/cal/``.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import Counter
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

# Make the cal package importable when this script is run as a file
# (matches Paper 3A's ``python code/experiments/xxx.py`` convention).
_THIS = Path(__file__).resolve()
_EXPERIMENTS_DIR = _THIS.parents[1]
if str(_EXPERIMENTS_DIR) not in sys.path:
    sys.path.insert(0, str(_EXPERIMENTS_DIR))

# VENDORING (Paper 3B-Cal): kernel is imported from cal/vendored/ — an
# immutable snapshot of Paper 3A's v5.0.0-tkde-submission artefacts.
# See cal/vendored/README.md for rationale + sync protocol.
# Intentionally NOT importing from the live ``code/experiments/hsm_similarity``
# so Paper 3B remains isolated from any Paper 3A revision-branch edits.
from cal.vendored.hsm_similarity import (  # noqa: E402
    WorkloadWindow,
    build_window,
    hsm_score,
)

from cal.gate import DEFAULT_CONFIG, Gate, GateConfig  # noqa: E402
from cal.metrics import (  # noqa: E402
    ClassificationRates,
    bootstrap_ci,
    fpr_fnr,
)


SCENARIOS = ["micro_shift", "syntactic_semantic_mismatch"]
DEFAULT_WINDOW_GRID = (5, 10, 20, 50, 100)

# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class WindowSweepResult:
    """Per-window-size metrics from the sweep."""

    window_size: int
    n_windows: int
    n_pairs: int
    f1: float
    fpr: float
    fnr: float
    precision: float
    recall: float
    bootstrap_f1: List[float]  # sample distribution used for broken-stick CI


@dataclass(frozen=True)
class BrokenStickFit:
    """Result of the broken-stick regression on F1 ~ log(|W|)."""

    w_min: float                 # point estimate (queries)
    w_min_ci_lo: float
    w_min_ci_hi: float
    slope_left: float            # slope in log|W| below break-point
    slope_right: float           # slope in log|W| above break-point
    intercept: float             # F1 at |W| = 1 (exp(0))
    ssr: float                   # sum of squared residuals at the fit
    confidence_level: float


# ---------------------------------------------------------------------------
# SkyLog loader (smoke stand-in for SDSS DR18)
# ---------------------------------------------------------------------------

# Regex copied (read-only) from ``hsm_sdss_validation.py`` so smoke loading
# matches Paper 3A's parsing without importing the script (which runs side
# effects at module-import time).
_SKYLOG_RECORD_END = re.compile(
    r",(\d+/\d+/\d{4}\s[\d:]+\s[AP]M),([\d.]+),([\d.]+),(\d+),([^,]*),(\d+)\s*$"
)


def _resolve_skylog_path() -> Path:
    """Locate ``SkyLog_Workload.csv`` — same candidate list as Paper 3A."""
    env = os.environ.get("HSM_SDSS_CSV")
    if env and Path(env).is_file():
        return Path(env)
    repo_root = _EXPERIMENTS_DIR.parents[1]  # HSM_gated/
    candidates = [
        repo_root / "code" / "data" / "sdss" / "SkyLog_Workload.csv",
        repo_root / "data" / "SkyLog_Workload.csv",
    ]
    for c in candidates:
        if c.is_file():
            return c
    raise FileNotFoundError(
        "SkyLog_Workload.csv not found; set HSM_SDSS_CSV or place it in "
        "code/data/sdss/. Tried:\n    "
        + "\n    ".join(str(c) for c in candidates)
    )


def load_sdss_sim_trace(
    max_records: int = 2000,
    path: Optional[Path] = None,
) -> List[Tuple[str, float]]:
    """Load (sql, elapsed_ms) pairs from the SkyLog CSV.

    The SkyLog ``theTime`` column is coarse (seconds resolution). For
    windowing we use a virtual serial clock: cumulative ``elapsed_ms`` →
    arrival seconds. This mirrors ``hsm_v2_kernel.build_qps_series`` and
    keeps the smoke data consistent with Paper 3A's S_P pipeline.
    """
    p = path or _resolve_skylog_path()
    out: List[Tuple[str, float]] = []
    current_lines: List[str] = []
    with open(p, "r", encoding="utf-8", errors="replace") as fh:
        next(fh, None)  # header
        for line in fh:
            m = _SKYLOG_RECORD_END.search(line.rstrip())
            if m:
                sql_part = line[: m.start()]
                current_lines.append(sql_part)
                sql = " ".join(current_lines).strip()
                try:
                    elapsed_ms = float(m.group(2))
                except ValueError:
                    elapsed_ms = 0.0
                # SkyLog error column — skip obvious syntax-error rows so
                # the HSM kernel's template extractor does not blow up.
                err_col = m.group(6).strip()
                if err_col == "0" and sql:
                    out.append((sql, elapsed_ms))
                current_lines = []
                if len(out) >= max_records:
                    break
            else:
                current_lines.append(line.rstrip())
    if not out:
        raise RuntimeError(f"SkyLog parser produced 0 records from {p}")
    return out


# ---------------------------------------------------------------------------
# Windowing + feature engineering
# ---------------------------------------------------------------------------


def _bin_trace_into_windows(
    trace: Sequence[Tuple[str, float]], window_size: int
) -> List[WorkloadWindow]:
    """Slice a (sql, elapsed_ms) trace into non-overlapping windows.

    Each window's ``duration_s`` is the sum of its queries' ``elapsed_ms``
    / 1000 (serial execution model; matches ``hsm_v2_kernel``).
    """
    if window_size < 2:
        raise ValueError(f"window_size must be >= 2 (got {window_size})")
    windows: List[WorkloadWindow] = []
    # Pre-compute cumulative arrival seconds across the whole trace so the
    # bucketed q(t) series for S_P has meaningful timestamps.
    elapsed_ms = np.array([e for _, e in trace], dtype=float)
    arrival_s = np.cumsum(np.maximum(elapsed_ms, 0.0)) / 1000.0
    n = len(trace)
    n_windows = n // window_size
    for i in range(n_windows):
        start = i * window_size
        end = start + window_size
        sqls = [s for s, _ in trace[start:end]]
        ts_window = arrival_s[start:end]
        ts_window = ts_window - ts_window[0]
        duration_s = float(ts_window[-1]) if window_size > 1 else 1.0
        if duration_s <= 0:
            duration_s = 1.0
        w = build_window(
            sqls,
            timestamps=list(ts_window),
            window_id=i,
            duration_s=duration_s,
        )
        windows.append(w)
    return windows


def _window_feature_vector(
    win: WorkloadWindow,
    template_axis: Sequence[str],
    table_axis: Sequence[str],
    column_axis: Sequence[str],
) -> np.ndarray:
    """Per-window feature vector used for the smoke ground-truth clusterer.

    Concatenates L1-normalised histograms over a shared vocabulary so
    k-means on the resulting vectors is meaningful across windows.
    """
    tf = win.template_freq
    total = sum(tf.values()) or 1
    template_hist = np.array(
        [tf.get(t, 0) / total for t in template_axis], dtype=float
    )
    tables = win.tables
    table_hist = np.array([1.0 if t in tables else 0.0 for t in table_axis], dtype=float)
    if table_hist.sum() > 0:
        table_hist = table_hist / table_hist.sum()
    cols = win.columns
    col_hist = np.array([1.0 if c in cols else 0.0 for c in column_axis], dtype=float)
    if col_hist.sum() > 0:
        col_hist = col_hist / col_hist.sum()
    return np.concatenate([template_hist, table_hist, col_hist])


def _shared_vocabulary(
    windows: Sequence[WorkloadWindow],
) -> Tuple[List[str], List[str], List[str]]:
    templates: Counter = Counter()
    tables: set = set()
    columns: set = set()
    for w in windows:
        templates.update(w.template_freq)
        tables |= w.tables
        columns |= w.columns
    return (
        [t for t, _ in templates.most_common()],
        sorted(tables),
        sorted(columns),
    )


def structural_ground_truth_labels(
    windows: Sequence[WorkloadWindow],
    *,
    seed: int = 42,
) -> Tuple[List[int], Dict[str, float]]:
    """Surrogate ground-truth for smoke runs.

    Procedure (matches the RQ1/RQ5 shape but with structural features):

    1. Build per-window feature vectors on the trace's shared vocabulary
       (templates + tables + columns).
    2. k-means(k=2) over all windows; assignment gives a cluster per
       window.
    3. Label for adjacent pair (W_i, W_{i+1}) is 1 iff the clusters
       differ, else 0.
    4. Diagnostics dict reports silhouette (-1..1) of the k=2 clustering
       so downstream reports can guard against "clustering was noise".

    Returns
    -------
    labels : list of ``{0, 1}`` length ``len(windows) - 1``
    diagnostics : dict with ``silhouette`` and ``cluster_sizes``
    """
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score

    if len(windows) < 3:
        # Not enough adjacency pairs to be informative; degenerate case.
        return [0] * max(0, len(windows) - 1), {
            "silhouette": float("nan"),
            "cluster_sizes": [len(windows)],
        }
    t_axis, tb_axis, c_axis = _shared_vocabulary(windows)
    X = np.stack(
        [_window_feature_vector(w, t_axis, tb_axis, c_axis) for w in windows]
    )
    # Guard: if feature matrix is all-zeros or constant, clustering is
    # meaningless — return all-zero labels.
    if np.allclose(X.std(axis=0), 0):
        return [0] * (len(windows) - 1), {
            "silhouette": float("nan"),
            "cluster_sizes": [len(windows)],
        }
    km = KMeans(n_clusters=2, n_init=10, random_state=seed)
    clusters = km.fit_predict(X)
    try:
        sil = float(silhouette_score(X, clusters)) if len(set(clusters)) == 2 else float("nan")
    except Exception:
        sil = float("nan")
    sizes = np.bincount(clusters, minlength=2).tolist()
    labels = [int(clusters[i] != clusters[i + 1]) for i in range(len(windows) - 1)]
    return labels, {"silhouette": sil, "cluster_sizes": sizes}


# ---------------------------------------------------------------------------
# Gate predictions and metric aggregation per window size
# ---------------------------------------------------------------------------


def gate_pair_predictions(
    windows: Sequence[WorkloadWindow],
    gate_config: GateConfig,
) -> Tuple[List[int], List[float]]:
    """Apply the gate to every adjacent pair.

    Gate output convention (RQ1 §Operational definitions): the *prediction*
    we compare against the shift label is ``1 - G_{w,θ}`` inverted — i.e.
    ``1`` when the gate flagged a shift (similarity < θ). For RQ5 we keep
    it simple: "predicted shift" = similarity below θ, same as the gate's
    raw output. Returns both the prediction list and the raw similarity
    list for reporting.
    """
    weights_map = {
        "w_R": gate_config.weights[0],
        "w_V": gate_config.weights[1],
        "w_T": gate_config.weights[2],
        "w_A": gate_config.weights[3],
        "w_P": gate_config.weights[4],
    }
    preds: List[int] = []
    sims: List[float] = []
    for a, b in zip(windows[:-1], windows[1:]):
        score, _ = hsm_score(a, b, weights=weights_map)
        sims.append(float(score))
        preds.append(int(score < gate_config.theta))
    return preds, sims


def compute_pairwise_metrics(
    y_true: Sequence[int],
    y_pred: Sequence[int],
) -> ClassificationRates:
    return fpr_fnr(y_true, y_pred)


def _bootstrap_pair_f1(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    n_resamples: int,
    seed: int,
) -> List[float]:
    """Bootstrap distribution of F1 for a single window size."""
    a = np.asarray(y_true, dtype=int)
    b = np.asarray(y_pred, dtype=int)
    if a.size == 0:
        return [0.0] * n_resamples
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, a.size, size=(n_resamples, a.size))
    out: List[float] = []
    for r in range(n_resamples):
        ib = idx[r]
        out.append(float(fpr_fnr(a[ib].tolist(), b[ib].tolist()).f1))
    return out


# ---------------------------------------------------------------------------
# Broken-stick regression
# ---------------------------------------------------------------------------


def _fit_broken_stick_once(
    log_w: np.ndarray,
    f1: np.ndarray,
    candidate_taus: np.ndarray,
) -> Tuple[float, float, float, float, float]:
    """Fit F1 = β0 + β1·log|W| + β2·max(0, log|W| - τ).

    Chooses τ from ``candidate_taus`` by minimum SSR. Returns
    ``(tau, slope_left, slope_right, intercept, ssr)``.

    ``slope_right = slope_left + β2`` so the break-point represents a
    *change* in slope rather than a discontinuity.
    """
    best: Optional[Tuple[float, float, float, float, float]] = None
    for tau in candidate_taus:
        hinge = np.maximum(0.0, log_w - tau)
        X = np.column_stack([np.ones_like(log_w), log_w, hinge])
        # Solve least squares; fall back gracefully if rank-deficient.
        try:
            beta, *_ = np.linalg.lstsq(X, f1, rcond=None)
        except np.linalg.LinAlgError:
            continue
        resid = f1 - X @ beta
        ssr = float(np.sum(resid ** 2))
        slope_left = float(beta[1])
        slope_right = float(beta[1] + beta[2])
        intercept = float(beta[0])
        if best is None or ssr < best[4]:
            best = (float(tau), slope_left, slope_right, intercept, ssr)
    if best is None:
        # Degenerate — fall back to the mean and zero slopes.
        return (
            float(log_w.mean()),
            0.0,
            0.0,
            float(f1.mean()) if f1.size else 0.0,
            float(np.sum((f1 - f1.mean()) ** 2)) if f1.size else 0.0,
        )
    return best


def fit_broken_stick(
    window_sizes: Sequence[int],
    f1_per_size: Sequence[float],
    *,
    bootstrap_f1: Optional[Sequence[Sequence[float]]] = None,
    n_tau_candidates: int = 100,
    confidence_level: float = 0.95,
    seed: int = 42,
) -> BrokenStickFit:
    """Fit broken-stick regression F1 ~ log(|W|) and bootstrap the break.

    Parameters
    ----------
    window_sizes : grid of ``|W|`` values (>= 2)
    f1_per_size : observed F1 at each window size
    bootstrap_f1 : optional list of bootstrap F1 distributions per size.
        When supplied, the break-point CI comes from resampling one F1
        value per size per replicate and re-fitting. Mirrors the RQ5
        design's "bootstrap windows → refit" protocol.
    n_tau_candidates : grid density for break-point search
    confidence_level : two-sided percentile CI on W_min

    Returns
    -------
    BrokenStickFit
    """
    sizes = np.asarray(window_sizes, dtype=float)
    if sizes.size < 3:
        raise ValueError(
            f"broken-stick fit needs >= 3 grid points (got {sizes.size})"
        )
    if np.any(sizes < 2):
        raise ValueError("window_sizes must all be >= 2")
    f1 = np.asarray(f1_per_size, dtype=float)
    if f1.shape != sizes.shape:
        raise ValueError("window_sizes and f1_per_size must be same length")

    log_w = np.log(sizes)
    # Candidate break-points: a dense grid across the interior of log|W|.
    lo, hi = float(log_w.min()), float(log_w.max())
    eps = (hi - lo) * 0.01
    candidate_taus = np.linspace(lo + eps, hi - eps, n_tau_candidates)

    tau, sl, sr, intercept, ssr = _fit_broken_stick_once(log_w, f1, candidate_taus)
    w_min_point = float(np.exp(tau))

    # Bootstrap the break-point.
    if bootstrap_f1 is not None:
        boot_matrix = np.asarray(
            [list(b) for b in bootstrap_f1], dtype=float
        )  # shape (n_sizes, n_bootstrap)
        if boot_matrix.shape[0] != sizes.size:
            raise ValueError(
                f"bootstrap_f1 must have one row per size "
                f"(got {boot_matrix.shape[0]} vs {sizes.size})"
            )
        n_bootstrap = boot_matrix.shape[1]
        rng = np.random.default_rng(seed)
        # For each bootstrap replicate pick one F1 per size (with the
        # column-index drawn from the replicate column itself), then
        # refit.
        tau_samples = np.empty(n_bootstrap, dtype=float)
        for r in range(n_bootstrap):
            # Random index within each size's bootstrap column so
            # replicates can mix across columns — this matches the
            # independence assumption across sizes.
            col_idx = rng.integers(0, n_bootstrap, size=sizes.size)
            f1_r = boot_matrix[np.arange(sizes.size), col_idx]
            tau_r, *_ = _fit_broken_stick_once(log_w, f1_r, candidate_taus)
            tau_samples[r] = tau_r
        alpha = 1.0 - confidence_level
        tau_lo = float(np.percentile(tau_samples, 100.0 * alpha / 2))
        tau_hi = float(np.percentile(tau_samples, 100.0 * (1.0 - alpha / 2)))
        w_min_lo = float(np.exp(tau_lo))
        w_min_hi = float(np.exp(tau_hi))
    else:
        w_min_lo = w_min_hi = w_min_point

    return BrokenStickFit(
        w_min=w_min_point,
        w_min_ci_lo=w_min_lo,
        w_min_ci_hi=w_min_hi,
        slope_left=sl,
        slope_right=sr,
        intercept=intercept,
        ssr=ssr,
        confidence_level=confidence_level,
    )


# ---------------------------------------------------------------------------
# Workload-loader registry
# ---------------------------------------------------------------------------

_WorkloadLoader = Callable[[], List[Tuple[str, float]]]


def _workload_loader(name: str, max_records: int) -> _WorkloadLoader:
    name = name.lower()
    if name in {"sdss_sim", "sdss", "sky_log", "skylog"}:
        return lambda: load_sdss_sim_trace(max_records=max_records)
    if name == "sdss_dr18":
        raise NotImplementedError(
            "sdss_dr18 loader not implemented yet — requires DR18 access "
            "(design doc §9 item 5). Use --workload sdss_sim for smoke runs."
        )
    raise ValueError(
        f"Unknown workload {name!r}. Supported: sdss_sim (smoke), "
        "sdss_dr18 (pending DR18 access)."
    )


# ---------------------------------------------------------------------------
# Main driver
# ---------------------------------------------------------------------------


def run_window_sweep(
    trace: Sequence[Tuple[str, float]],
    window_grid: Sequence[int] = DEFAULT_WINDOW_GRID,
    gate_config: GateConfig = DEFAULT_CONFIG,
    *,
    n_bootstrap: int = 1000,
    confidence_level: float = 0.95,
    seed: int = 42,
    ground_truth_fn: Callable[
        [Sequence[WorkloadWindow]], Tuple[List[int], Dict[str, float]]
    ] = None,
) -> Dict[str, object]:
    """End-to-end window-size sweep → per-size metrics → broken-stick fit.

    Returned dict shape::

        {
          "per_size": [WindowSweepResult.__dict__, ...],
          "broken_stick": BrokenStickFit.__dict__,
          "ground_truth_diagnostics": {...},
          "gate_config": {...},
          "meta": {...},
        }
    """
    gt_fn = ground_truth_fn or structural_ground_truth_labels

    per_size: List[WindowSweepResult] = []
    gt_diag_by_size: Dict[int, Dict[str, float]] = {}
    bootstrap_f1_by_size: List[List[float]] = []

    for size in window_grid:
        windows = _bin_trace_into_windows(trace, size)
        if len(windows) < 3:
            # Not enough windows to form pairs + clustering — record a
            # degenerate row so the caller can see what happened.
            per_size.append(
                WindowSweepResult(
                    window_size=size,
                    n_windows=len(windows),
                    n_pairs=max(0, len(windows) - 1),
                    f1=0.0,
                    fpr=0.0,
                    fnr=0.0,
                    precision=0.0,
                    recall=0.0,
                    bootstrap_f1=[0.0],
                )
            )
            gt_diag_by_size[size] = {"silhouette": float("nan"), "cluster_sizes": [0, 0]}
            bootstrap_f1_by_size.append([0.0])
            continue
        y_true, gt_diag = gt_fn(windows)
        y_pred, _sims = gate_pair_predictions(windows, gate_config)
        rates = compute_pairwise_metrics(y_true, y_pred)
        boot_f1 = _bootstrap_pair_f1(
            y_true, y_pred, n_resamples=n_bootstrap, seed=seed + size
        )
        per_size.append(
            WindowSweepResult(
                window_size=size,
                n_windows=len(windows),
                n_pairs=len(y_true),
                f1=rates.f1,
                fpr=rates.fpr,
                fnr=rates.fnr,
                precision=rates.precision,
                recall=rates.recall,
                bootstrap_f1=boot_f1,
            )
        )
        gt_diag_by_size[size] = gt_diag
        bootstrap_f1_by_size.append(boot_f1)

    # Broken-stick fit across sizes that yielded real data.
    usable = [r for r in per_size if r.n_pairs > 0]
    if len(usable) >= 3:
        sizes = [r.window_size for r in usable]
        f1_obs = [r.f1 for r in usable]
        boot_matrix = [r.bootstrap_f1 for r in usable]
        fit = fit_broken_stick(
            sizes,
            f1_obs,
            bootstrap_f1=boot_matrix,
            confidence_level=confidence_level,
            seed=seed,
        )
    else:
        fit = None

    return {
        "per_size": [
            {
                "window_size": r.window_size,
                "n_windows": r.n_windows,
                "n_pairs": r.n_pairs,
                "f1": r.f1,
                "fpr": r.fpr,
                "fnr": r.fnr,
                "precision": r.precision,
                "recall": r.recall,
                # Keep bootstrap list compact in JSON — mean + std only
                # unless caller asked for the full distribution later.
                "bootstrap_f1_mean": (
                    float(np.mean(r.bootstrap_f1)) if r.bootstrap_f1 else 0.0
                ),
                "bootstrap_f1_std": (
                    float(np.std(r.bootstrap_f1, ddof=1))
                    if len(r.bootstrap_f1) > 1
                    else 0.0
                ),
            }
            for r in per_size
        ],
        "broken_stick": asdict(fit) if fit is not None else None,
        "ground_truth_diagnostics": {
            str(k): v for k, v in gt_diag_by_size.items()
        },
        "gate_config": {
            "weights": list(gate_config.weights),
            "theta": gate_config.theta,
        },
        "meta": {
            "window_grid": list(window_grid),
            "n_bootstrap": n_bootstrap,
            "confidence_level": confidence_level,
            "seed": seed,
            "n_trace_records": len(trace),
        },
    }


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Paper 3B-Cal RQ5 driver")
    p.add_argument("--mode", required=True, choices=["window_sweep", "scenarios"])
    p.add_argument("--workload", required=True)
    p.add_argument(
        "--window-grid",
        default=",".join(str(x) for x in DEFAULT_WINDOW_GRID),
        help="Comma-separated window sizes (queries) for W_min sweep",
    )
    p.add_argument(
        "--scenarios",
        default=",".join(SCENARIOS),
        help="Comma-separated failure scenarios to run (mode=scenarios)",
    )
    p.add_argument("--out", required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--max-records",
        type=int,
        default=2000,
        help="Cap records loaded from the workload source (smoke default 2000)",
    )
    p.add_argument(
        "--n-bootstrap",
        type=int,
        default=1000,
        help="Bootstrap resamples for F1 and W_min CI (default 1000)",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.mode == "window_sweep":
        grid = tuple(int(x) for x in args.window_grid.split(","))
        loader = _workload_loader(args.workload, max_records=args.max_records)
        trace = loader()
        result = run_window_sweep(
            trace,
            window_grid=grid,
            n_bootstrap=args.n_bootstrap,
            seed=args.seed,
        )
        result["meta"]["workload"] = args.workload
        result["meta"]["ground_truth_source"] = (
            "structural_kmeans_k2_smoke"
            if args.workload in {"sdss_sim", "sdss", "sky_log", "skylog"}
            else "tbd"
        )
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w") as fh:
            json.dump(result, fh, indent=2, default=float)
        print(f"[rq5] wrote window-sweep JSON to {out_path}")
        return 0

    # mode == "scenarios"
    raise NotImplementedError(
        f"rq5_boundaries.main: --mode scenarios still scaffolding; "
        f"failure_scenarios.py F1/F2 generators are not yet implemented. "
        f"Args parsed: {args}"
    )


if __name__ == "__main__":
    sys.exit(main())
