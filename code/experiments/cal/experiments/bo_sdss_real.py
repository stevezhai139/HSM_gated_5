"""BO-qLogNEHVI on SDSS real-workload data (harder workload for BO-gain).

Why a separate driver from ``run_bo_experiment.py``:

TPC-H phase-shift (synthetic) workloads have disjoint phase tables, so HSM
can separate phases at nearly any w (baseline F1 = 1.0 on our smoke).
This saturates the objective and leaves no room for BO to improve over W₀.
SDSS SkyLog real queries have overlapping table/column vocabularies across
phases, making separation harder — Paper 3A Supplementary §XIV reports
Youden-optimal F1 ≈ 0.62 on SDSS at fixed W₀, so there IS room for BO.

Fast-path design:
Per-pair HSM per-dimension scores (S_R, S_V, S_T, S_A, S_P) depend ONLY on
the window pair, not on the weight w. So we pre-compute a
(n_pairs, 5) matrix once per SDSS subsample, then per BO iteration
evaluate:
    composite[i] = matrix[i] @ w            # 5-D dot product
    gate[i]     = 𝟙[ composite[i] < θ ]
    (P, R)      = fpr_fnr(y_true, gate)
All vectorised numpy. Per BO iter is <1 ms after pre-compute.

Usage:
    python code/experiments/cal/experiments/bo_sdss_real.py --all
    python code/experiments/cal/experiments/bo_sdss_real.py --smoke

Output tree:
    results/cal/experiments/<ts>_<sha>_bo_sdss_real/
      run_meta.json
      precomputed_dims.npz          (shared across blocks)
      bo/block_<0..9>/result.json   (10 blocks)
      baseline.json                 (W₀, θ=0.75 evaluation)
      progress.log
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# sys.path fixup
_THIS = Path(__file__).resolve()
_EXPERIMENTS_DIR = _THIS.parents[2]
if str(_EXPERIMENTS_DIR) not in sys.path:
    sys.path.insert(0, str(_EXPERIMENTS_DIR))

from cal.validation.paper3a_loader import load_experiment as load_sdss  # noqa: E402
from cal.metrics import fpr_fnr  # noqa: E402
from cal.experiments import bo_qnehvi_calibration as bq  # noqa: E402
from cal.validation import _run_meta  # noqa: E402


EPS = bq.EPS
N_DIMS_HSM = bq.N_DIMS_HSM


# ---------------------------------------------------------------------------
# Pre-computation of per-pair 5-dim HSM scores on SDSS
# ---------------------------------------------------------------------------


def _recompute_dims_from_skylog(window_size: int = 20) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Re-bin SkyLog_Workload.csv into windows of the given size and compute
    per-pair 5-dim HSM scores from the vendored kernel.

    This is the expensive path — called once per run, cached in npz.

    Returns
    -------
    dims_matrix : (n_pairs, 5) float — [S_R, S_V, S_T, S_A, S_P] per adjacent pair
    y_true      : (n_pairs,) int {0, 1} — 1 iff phase changes between pair
    window_idx  : (n_pairs,) int — the window_idx aligning with stored CSV
    """
    # Use Paper 3A's stored trigger_timeseries to get y_true (phase transitions)
    series = load_sdss("sdss")
    y_true = np.asarray(series.is_transition, dtype=int)
    window_idx = np.asarray(series.window_idx, dtype=int)

    # For the dims matrix we need raw SkyLog queries re-binned the same way
    # Paper 3A did (window_size=20). We use the vendored hsm_similarity
    # functions to construct WorkloadWindows from SkyLog SQL strings.
    from cal.rq5_boundaries import load_sdss_sim_trace  # re-use existing loader
    # load_sdss_sim_trace returns list of (sql, elapsed_ms); Paper 3A's
    # window_size is 20. We re-bin the same trace.
    trace = load_sdss_sim_trace(max_records=200_000)  # cap to avoid OOM

    from cal.vendored.hsm_similarity import build_window, hsm_score

    # Re-bin: non-overlapping windows of size `window_size`
    n_q = len(trace)
    n_windows = n_q // window_size
    # Cap windows to match stored series.n_pairs + 1
    max_windows = series.n_pairs + 1
    n_windows = min(n_windows, max_windows)

    windows = []
    elapsed_ms = [t[1] for t in trace[: n_windows * window_size]]
    arrival_s = np.cumsum(np.maximum(np.asarray(elapsed_ms, dtype=float), 0.0)) / 1000.0
    for i in range(n_windows):
        start = i * window_size
        end = start + window_size
        sqls = [s for s, _ in trace[start:end]]
        ts_rel = arrival_s[start:end] - arrival_s[start]
        duration_s = float(ts_rel[-1]) if ts_rel[-1] > 0 else 1.0
        w = build_window(sqls, timestamps=ts_rel.tolist(), window_id=i, duration_s=duration_s)
        windows.append(w)

    # Per-pair HSM per-dimension scores
    n_pairs = len(windows) - 1
    dims = np.zeros((n_pairs, 5), dtype=float)
    # Uniform reference weights during per-dim extraction; dims don't depend on w
    ref_w = {"w_R": 0.2, "w_V": 0.2, "w_T": 0.2, "w_A": 0.2, "w_P": 0.2}
    for i in range(n_pairs):
        _, d = hsm_score(windows[i], windows[i + 1], weights=ref_w)
        dims[i] = [d["S_R"], d["S_V"], d["S_T"], d["S_A"], d["S_P"]]

    # Align y_true to our re-binned pair count
    y_true_aligned = y_true[:n_pairs]
    window_idx_aligned = window_idx[:n_pairs]

    return dims, y_true_aligned, window_idx_aligned


def _cache_path(run_dir: Path, window_size: int = 20) -> Path:
    """Per-window dims cache. Legacy single-window name 'precomputed_dims.npz'
    is preserved for window_size=20 for backward-compat with existing runs."""
    if window_size == 20:
        legacy = run_dir / "precomputed_dims.npz"
        if legacy.is_file():
            return legacy
    return run_dir / f"precomputed_dims_W{window_size}.npz"


def _load_or_compute_dims(
    run_dir: Path, window_size: int, verbose: bool
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load cached dims matrix if present, else compute and cache."""
    cache = _cache_path(run_dir, window_size)
    if cache.is_file():
        if verbose:
            print(f"[precompute] cache hit: {cache}")
        data = np.load(cache)
        return data["dims"], data["y_true"], data["window_idx"]
    if verbose:
        print(f"[precompute] computing dims from SkyLog at W={window_size} "
              f"(this takes several minutes)...")
    t0 = time.monotonic()
    dims, y_true, widx = _recompute_dims_from_skylog(window_size=window_size)
    elapsed = time.monotonic() - t0
    np.savez(cache, dims=dims, y_true=y_true, window_idx=widx)
    if verbose:
        print(f"[precompute] done in {elapsed:.1f}s → {cache} "
              f"(shape dims={dims.shape}, n_transitions={int(y_true.sum())})")
    return dims, y_true, widx


# ---------------------------------------------------------------------------
# Fast-path BO on pre-computed dims matrix
# ---------------------------------------------------------------------------


def _eval_fast(
    dims: np.ndarray, y_true: np.ndarray, w: np.ndarray, theta: float
) -> Tuple[float, float]:
    """Evaluate (Precision, Recall) via pre-computed dims matrix.

    composite[i] = dims[i] @ w; y_pred = 1 iff composite < theta; positive = shift.
    """
    composite = dims @ w
    y_pred = (composite < theta).astype(int)
    rates = fpr_fnr(y_true.tolist(), y_pred.tolist())
    return float(rates.precision), float(rates.recall)


def run_bo_on_sdss_slice(
    dims: np.ndarray,
    y_true: np.ndarray,
    block_seed: int,
    slice_indices: np.ndarray,
    *,
    n_init: int = 10,
    n_iter: int = 50,
    ref_point: Tuple[float, float] = (0.0, 0.0),
    verbose: bool = False,
) -> bq.BOQnehviResult:
    """Run BO-qLogNEHVI on a SDSS data slice using pre-computed dims matrix.

    Adapts the core BO loop to use the fast _eval_fast path instead of
    re-running the HSM kernel per iteration.
    """
    bt = bq._import_botorch()
    torch = bt["torch"]

    # Subset the dims + y_true to this block's slice
    d_slice = dims[slice_indices]
    y_slice = y_true[slice_indices]

    if int(y_slice.sum()) < 2 or int((1 - y_slice).sum()) < 2:
        raise ValueError(
            f"Block {block_seed}: too few transitions ({int(y_slice.sum())}) "
            f"or stable pairs ({int((1 - y_slice).sum())}) for meaningful F1"
        )

    torch.manual_seed(block_seed + 2000)
    np.random.seed(block_seed)

    bounds = torch.tensor(
        [[0.0, 0.0, 0.0, 0.0, 0.0, EPS],
         [1.0, 1.0, 1.0, 1.0, 1.0, 1.0 - EPS]],
        dtype=torch.double,
    )
    equality_constraints = [
        (torch.tensor([0, 1, 2, 3, 4], dtype=torch.long),
         torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0], dtype=torch.double),
         1.0),
    ]
    inequality_constraints = [
        (torch.tensor([i], dtype=torch.long),
         torch.tensor([1.0], dtype=torch.double),
         EPS)
        for i in range(5)
    ]

    def eval_candidate(x_raw: "torch.Tensor") -> Tuple[float, float]:
        x = x_raw.detach().cpu().numpy().flatten()
        w = x[:N_DIMS_HSM]
        w = w / w.sum() if w.sum() > 0 else np.full(N_DIMS_HSM, 1.0 / N_DIMS_HSM)
        theta = float(np.clip(x[N_DIMS_HSM], EPS, 1.0 - EPS))
        return _eval_fast(d_slice, y_slice, w, theta)

    # Sobol init
    init_X = bq._make_simplex_init(bt, n_init, block_seed)
    init_Y_list: List[List[float]] = []
    history: List[bq.BOIterationRecord] = []
    for j in range(n_init):
        p, r = eval_candidate(init_X[j])
        init_Y_list.append([p, r])
        x_np = init_X[j].detach().cpu().numpy()
        w = x_np[:N_DIMS_HSM]
        w = w / w.sum() if w.sum() > 0 else np.full(N_DIMS_HSM, 1.0 / N_DIMS_HSM)
        theta = float(np.clip(x_np[N_DIMS_HSM], EPS, 1.0 - EPS))
        history.append(bq.BOIterationRecord(
            iter_index=-(n_init - j),
            w=w.tolist(), theta=theta,
            precision=p, recall=r, hypervolume=0.0,
        ))
    running_hv = bq._running_hypervolume([(h.precision, h.recall) for h in history], ref_point)
    for j, hv in enumerate(running_hv):
        history[j] = bq.BOIterationRecord(
            iter_index=history[j].iter_index,
            w=history[j].w, theta=history[j].theta,
            precision=history[j].precision, recall=history[j].recall,
            hypervolume=hv,
        )

    train_X = init_X.clone()
    train_Y = torch.tensor(init_Y_list, dtype=torch.double)

    for i in range(n_iter):
        gps = []
        for obj_idx in range(2):
            y = train_Y[..., obj_idx:obj_idx + 1]
            gp = bt["SingleTaskGP"](
                train_X.double(), y.double(),
                input_transform=bt["Normalize"](d=train_X.shape[-1], bounds=bounds),
            )
            gps.append(gp)
        model = bt["ModelListGP"](*gps)
        mll = bt["SumMLL"](model.likelihood, model)
        bt["fit_mll"](mll)
        sampler = bt["SobolQMCNormalSampler"](sample_shape=torch.Size([64]))
        acq = bt["qLogNEHVI"](
            model=model, ref_point=list(ref_point),
            X_baseline=train_X.double(), sampler=sampler, prune_baseline=True,
        )
        candidates, _ = bt["optimize_acqf"](
            acq_function=acq, bounds=bounds, q=1,
            num_restarts=10, raw_samples=128,
            equality_constraints=equality_constraints,
            inequality_constraints=inequality_constraints,
            sequential=True,
        )
        new_x = candidates.detach()
        p, r = eval_candidate(new_x[0])
        train_X = torch.cat([train_X, new_x], dim=0)
        train_Y = torch.cat([train_Y, torch.tensor([[p, r]], dtype=torch.double)], dim=0)

        x_np = new_x[0].detach().cpu().numpy()
        w = x_np[:N_DIMS_HSM]
        w = w / w.sum() if w.sum() > 0 else np.full(N_DIMS_HSM, 1.0 / N_DIMS_HSM)
        theta = float(np.clip(x_np[N_DIMS_HSM], EPS, 1.0 - EPS))
        obj_list = [(h.precision, h.recall) for h in history] + [(p, r)]
        hv = bq._running_hypervolume(obj_list, ref_point)[-1]
        history.append(bq.BOIterationRecord(
            iter_index=i, w=w.tolist(), theta=theta,
            precision=p, recall=r, hypervolume=hv,
        ))
        if verbose:
            print(f"    [iter {i+1}/{n_iter}] P={p:.3f} R={r:.3f} HV={hv:.4f}")

    # Pareto + knee
    objs = np.array([[h.precision, h.recall] for h in history])
    pareto_mask = bq._pareto_mask(objs)
    pareto_indices = [int(i) for i, m in enumerate(pareto_mask) if m]
    pareto_points = objs[pareto_mask]
    knee_local = int(np.argmax(pareto_points.sum(axis=1)))
    knee_idx = pareto_indices[knee_local]
    knee = history[knee_idx]

    return bq.BOQnehviResult(
        block_seed=block_seed, window_size=20, queries_per_phase=0,
        n_init=n_init, n_iter=n_iter, history=history,
        final_w_star=knee.w, final_theta_star=knee.theta,
        final_precision=knee.precision, final_recall=knee.recall,
        final_hypervolume=knee.hypervolume, pareto_indices=pareto_indices,
    )


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def _make_block_slices(n_pairs: int, n_blocks: int = 10) -> List[np.ndarray]:
    """Split [0, n_pairs) into n_blocks non-overlapping chronological slices."""
    boundaries = np.linspace(0, n_pairs, n_blocks + 1, dtype=int)
    return [np.arange(boundaries[i], boundaries[i + 1]) for i in range(n_blocks)]


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="BO-qLogNEHVI on SDSS real workload")
    p.add_argument("--all", action="store_true", help="Run n_blocks per window + baseline")
    p.add_argument("--smoke", action="store_true", help="1 block × 5 iters (first window only)")
    p.add_argument("--out-root", default="results/cal/experiments")
    # Either --window-size (single) or --windows (multi). Kept both for backward compat.
    p.add_argument("--window-size", type=int, default=None,
                   help="Single window size. Equivalent to --windows N. "
                        "Default: 20 if neither --window-size nor --windows is set.")
    p.add_argument("--windows", type=str, default=None,
                   help="Comma-separated list of window sizes, e.g. '20,50,100'. "
                        "Creates one cell per window; BY-FDR family m = len(windows).")
    p.add_argument("--n-blocks", type=int, default=10)
    p.add_argument("--resume-dir", default=None,
                   help="Reuse existing SDSS run directory (skips cached dims + result.json)")
    return p


def _parse_windows(args: argparse.Namespace) -> List[int]:
    """Coalesce --window-size and --windows into a canonical list."""
    if args.windows:
        return [int(w.strip()) for w in args.windows.split(",") if w.strip()]
    if args.window_size is not None:
        return [int(args.window_size)]
    return [20]


def _repo_root() -> Path:
    return _THIS.parents[4]


def main(argv: Optional[List[str]] = None) -> int:
    argv = argv if argv is not None else sys.argv[1:]
    args = build_parser().parse_args(argv)

    if not (args.all or args.smoke):
        print("Use --all or --smoke")
        return 1

    windows = _parse_windows(args)
    if args.smoke and len(windows) > 1:
        print(f"[info] --smoke restricting to first window only: W={windows[0]}")
        windows = windows[:1]

    meta = _run_meta.capture(
        experiment="bo_sdss_real",
        cli_args=argv,
        seed=0,
        extra={"windows": windows, "smoke": args.smoke},
    )

    # Resume-dir support (mirror run_bo_experiment.py semantics)
    if args.resume_dir:
        resume_path = Path(args.resume_dir)
        if not resume_path.is_absolute():
            resume_path = _repo_root() / args.resume_dir
        if not resume_path.is_dir():
            print(f"ERROR: --resume-dir not found: {resume_path}", file=sys.stderr)
            return 1
        run_dir = resume_path
        print(f"[bo_sdss_real] resuming existing run_dir (checkpoint-reuse): {run_dir}")
    else:
        slug = _run_meta.slug_for_filename(meta)
        run_dir = _repo_root() / args.out_root / f"{slug}_sdss"
        run_dir.mkdir(parents=True, exist_ok=True)

    with (run_dir / "run_meta.json").open("w") as fh:
        json.dump(meta, fh, indent=2, default=str)

    progress_path = run_dir / "progress.log"
    progress_fh = progress_path.open("a")
    progress_fh.write(f"\n==== SDSS run start: {meta['timestamp_utc']} "
                      f"windows={windows} ====\n")

    t_total0 = time.monotonic()

    # Baseline container: per window, per block
    # Keeps backward compat: single-W run still writes 'per_block' at top level.
    W0 = np.array([0.25, 0.20, 0.20, 0.20, 0.15])
    baseline_all: Dict[str, Any] = {
        "w": W0.tolist(), "theta": 0.75,
        "per_window_block": {},   # {str(W): [per_block_entries]}
    }

    n_blocks = 1 if args.smoke else args.n_blocks
    n_init = 4 if args.smoke else 10
    n_iter = 5 if args.smoke else 50

    for window_size in windows:
        print(f"\n===== SDSS window={window_size} =====")
        progress_fh.write(f"\n===== window={window_size} =====\n"); progress_fh.flush()

        # Pre-compute per-pair dims for THIS window size (cached per W)
        print(f"[pre-compute] W={window_size}: loading SkyLog + computing 5-dim HSM scores...")
        dims, y_true, widx = _load_or_compute_dims(run_dir, window_size, verbose=True)
        progress_fh.write(f"[pre-compute] W={window_size} dims shape {dims.shape}, "
                          f"transitions {int(y_true.sum())}/{len(y_true)}\n")

        # Full-dataset baseline for this window
        p_w0, r_w0 = _eval_fast(dims, y_true, W0, 0.75)
        f1_w0 = (2 * p_w0 * r_w0 / (p_w0 + r_w0)) if (p_w0 + r_w0) > 0 else 0.0
        print(f"[baseline] W={window_size} full: P={p_w0:.4f} R={r_w0:.4f} F1={f1_w0:.4f}")

        slices = _make_block_slices(dims.shape[0], n_blocks)
        baseline_per_block: List[Optional[Dict[str, Any]]] = []
        for b, idx in enumerate(slices):
            if int(y_true[idx].sum()) < 2 or int((1 - y_true[idx]).sum()) < 2:
                baseline_per_block.append(None)
                continue
            p, r = _eval_fast(dims[idx], y_true[idx], W0, 0.75)
            f1 = (2 * p * r / (p + r)) if (p + r) > 0 else 0.0
            baseline_per_block.append({"block": b, "n_pairs": int(len(idx)),
                                       "precision": p, "recall": r, "f1": f1})
        baseline_all["per_window_block"][str(window_size)] = {
            "precision_full": p_w0, "recall_full": r_w0, "f1_full": f1_w0,
            "n_pairs": int(dims.shape[0]),
            "per_block": baseline_per_block,
        }
        # Backward-compat: for single-window runs, also write flat per_block/f1_full
        if len(windows) == 1:
            baseline_all["precision_full"] = p_w0
            baseline_all["recall_full"] = r_w0
            baseline_all["f1_full"] = f1_w0
            baseline_all["n_pairs"] = int(dims.shape[0])
            baseline_all["per_block"] = baseline_per_block
        with (run_dir / "baseline.json").open("w") as fh:
            json.dump(baseline_all, fh, indent=2)

        # BO per block for THIS window
        for b, idx in enumerate(slices):
            # Output path: single-W keeps legacy bo/block_<b>/; multi-W uses bo/window_<W>/block_<b>/
            if len(windows) == 1:
                out_path = run_dir / "bo" / f"block_{b}" / "result.json"
            else:
                out_path = run_dir / "bo" / f"window_{window_size}" / f"block_{b}" / "result.json"
            if out_path.is_file():
                print(f"[skip] W={window_size}/block_{b} cached at {out_path}")
                continue
            if baseline_per_block[b] is None:
                print(f"[skip] W={window_size}/block_{b} — too few transitions or stable pairs")
                continue
            print(f"[bo] W={window_size}/block_{b} (slice {idx[0]}..{idx[-1]}, n_pairs={len(idx)})")
            progress_fh.write(f"[bo] W={window_size} block_{b} start\n"); progress_fh.flush()
            t0 = time.monotonic()
            result = run_bo_on_sdss_slice(
                dims=dims, y_true=y_true, block_seed=b,
                slice_indices=idx, n_init=n_init, n_iter=n_iter, verbose=False,
            )
            elapsed = time.monotonic() - t0
            out_path.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "track": "sdss_real", "block": b, "window": window_size,
                "slice_start": int(idx[0]), "slice_end": int(idx[-1]),
                "n_pairs": int(len(idx)), "n_init": n_init, "n_iter": n_iter,
                "elapsed_seconds": elapsed,
                "result": result.as_dict(),
                "baseline_f1": baseline_per_block[b]["f1"],
            }
            with out_path.open("w") as fh:
                json.dump(payload, fh, indent=2, default=float)
            msg = (f"    → HV={result.final_hypervolume:.4f} "
                   f"(P, R)=({result.final_precision:.3f}, {result.final_recall:.3f}) "
                   f"θ*={result.final_theta_star:.4f} "
                   f"baseline_F1={baseline_per_block[b]['f1']:.4f} "
                   f"{elapsed:.1f}s")
            print(msg)
            progress_fh.write(msg + "\n"); progress_fh.flush()

    t_total = time.monotonic() - t_total0
    msg = f"\n=== SDSS total elapsed: {t_total:.1f}s ({t_total/60:.2f} min) ==="
    print(msg); progress_fh.write(msg + "\n"); progress_fh.close()

    log_path = _repo_root() / args.out_root / "EXPERIMENT_LOG.md"
    baseline_f1_summary = next(
        (entry["f1_full"] for entry in baseline_all["per_window_block"].values()),
        0.0,
    )
    summary = (f"sdss_real; windows={windows}; n_blocks={n_blocks}; smoke={args.smoke}; "
               f"baseline_F1_first_W={baseline_f1_summary:.4f}; elapsed={t_total:.1f}s; "
               f"dir=`{run_dir.relative_to(_repo_root())}`")
    _run_meta.append_experiment_log(log_path, meta, summary)

    print(f"[bo_sdss_real] done. Artifacts: {run_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
