"""Real BO-qLogNEHVI experiment driver — full RQ4a run.

Per Paper3B_Cal_Experiment_Plan_v0.docx §5.3, with supplementary hard case
per Day 2 review. Three tracks:

- **Easy track**: 10 blocks × 4 windows {5, 10, 20, 50} × queries_per_phase=60
                  × 50 BO iters (plus 10 Sobol init). Total: 40 configs.
- **Hard track**: 10 blocks × 2 windows {5, 10} × queries_per_phase=20
                  × 50 BO iters. Total: 20 configs. Lower signal-to-noise.
- **Baseline track**: 10 blocks × 1 window (10) × fixed W₀ = (0.25, 0.20,
                      0.20, 0.20, 0.15), θ = 0.75. No BO; single evaluation.
                      Total: 10 configs. Paper 3A default baseline.

Usage (from HSM_gated repo root):

    # Full run (easy + hard + baseline)
    python code/experiments/cal/experiments/run_bo_experiment.py --all

    # Subset (for testing / staged runs)
    python code/experiments/cal/experiments/run_bo_experiment.py --easy
    python code/experiments/cal/experiments/run_bo_experiment.py --hard
    python code/experiments/cal/experiments/run_bo_experiment.py --baseline

    # Dry-run smoke (1 block × 1 window per track, 5 BO iters)
    python code/experiments/cal/experiments/run_bo_experiment.py --smoke

Output tree:

    results/cal/experiments/<timestamp>_<sha>_bo_rq4a_run/
      run_meta.json
      easy/block_<0..9>/window_<5,10,20,50>/result.json
      hard/block_<0..9>/window_<5,10>/result.json
      baseline/block_<0..9>/result.json
      progress.log

Checkpointing: per-(track, block, window) result.json is written immediately
after completion. If the script is interrupted and rerun with the SAME output
directory, existing result.json files are detected and those configs are
skipped. This enables recovery from crashes without restarting the whole run.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# sys.path fixup
_THIS = Path(__file__).resolve()
_EXPERIMENTS_DIR = _THIS.parents[2]
if str(_EXPERIMENTS_DIR) not in sys.path:
    sys.path.insert(0, str(_EXPERIMENTS_DIR))

from cal.experiments import bo_qnehvi_calibration as bq  # noqa: E402
from cal.validation import _run_meta  # noqa: E402


# ---------------------------------------------------------------------------
# Experiment configuration (per Plan §5.3 + Day 2 hard-case extension)
# ---------------------------------------------------------------------------

DEFAULT_N_BLOCKS = 10  # default count; --n-blocks overrides without cap

EASY_CONFIG = {
    "queries_per_phase": 60,
    "windows": (5, 10, 20, 50),
    "n_init": 10,
    "n_iter": 50,
}

HARD_CONFIG = {
    "queries_per_phase": 20,
    "windows": (5, 10),  # window=20 impossible with qpp=20 (only 4 windows)
    "n_init": 10,
    "n_iter": 50,
}

BASELINE_CONFIG = {
    "queries_per_phase": 60,
    "windows": (10,),  # single window for baseline reproducibility
    # Paper 3A default: W₀ = (0.25, 0.20, 0.20, 0.20, 0.15), θ = 0.75
    "W0": np.array([0.25, 0.20, 0.20, 0.20, 0.15]),
    "theta": 0.75,
}

SMOKE_CONFIG = {
    "n_init": 4,
    "n_iter": 5,
    "queries_per_phase_easy": 30,
    "queries_per_phase_hard": 20,
    "window_easy": 10,
    "window_hard": 5,
    "n_blocks": 1,
}


# ---------------------------------------------------------------------------
# Track runners
# ---------------------------------------------------------------------------


def _result_path(run_dir: Path, track: str, block: int, window: int) -> Path:
    if track == "baseline":
        return run_dir / track / f"block_{block}.json"
    return run_dir / track / f"block_{block}" / f"window_{window}" / "result.json"


def _save_result_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fh:
        json.dump(payload, fh, indent=2, default=float)


def _run_bo_config(
    track: str,
    block: int,
    window: int,
    qpp: int,
    n_init: int,
    n_iter: int,
    run_dir: Path,
    progress_fh,
) -> Dict[str, Any]:
    """Run one BO configuration (one block × one window) with checkpointing."""
    out_path = _result_path(run_dir, track, block, window)
    if out_path.is_file():
        msg = f"  [skip] {track}/block_{block}/window_{window} — cached at {out_path}"
        print(msg); progress_fh.write(msg + "\n"); progress_fh.flush()
        with out_path.open() as fh:
            return json.load(fh)

    msg = f"  [run]  {track}/block_{block}/window_{window} qpp={qpp} (n_init={n_init}, n_iter={n_iter})"
    print(msg); progress_fh.write(msg + "\n"); progress_fh.flush()

    t0 = time.monotonic()
    result = bq.run_single_configuration(
        block_seed=block,
        window_size=window,
        queries_per_phase=qpp,
        n_init=n_init,
        n_iter=n_iter,
        verbose=False,
    )
    elapsed = time.monotonic() - t0

    payload = {
        "track": track,
        "block": block,
        "window": window,
        "queries_per_phase": qpp,
        "n_init": n_init,
        "n_iter": n_iter,
        "elapsed_seconds": elapsed,
        "result": result.as_dict(),
    }
    _save_result_json(out_path, payload)
    msg_done = f"    → HV={result.final_hypervolume:.4f}, (P, R)=({result.final_precision:.3f}, {result.final_recall:.3f}), θ*={result.final_theta_star:.4f}, {elapsed:.1f}s"
    print(msg_done); progress_fh.write(msg_done + "\n"); progress_fh.flush()
    return payload


def _run_baseline_config(
    block: int,
    run_dir: Path,
    progress_fh,
) -> Dict[str, Any]:
    """Baseline: single (w, θ) = (W₀, 0.75) evaluation per block."""
    out_path = _result_path(run_dir, "baseline", block, 10)
    if out_path.is_file():
        msg = f"  [skip] baseline/block_{block} — cached"
        print(msg); progress_fh.write(msg + "\n"); progress_fh.flush()
        with out_path.open() as fh:
            return json.load(fh)

    from cal.vendored.workload_generator import get_workload_trace
    msg = f"  [run]  baseline/block_{block} (fixed W₀, θ=0.75, window=10)"
    print(msg); progress_fh.write(msg + "\n"); progress_fh.flush()

    t0 = time.monotonic()
    trace = get_workload_trace(queries_per_phase=BASELINE_CONFIG["queries_per_phase"], seed=block)
    labelled = bq.build_labelled_windows(
        trace,
        window_size=BASELINE_CONFIG["windows"][0],
        queries_per_phase=BASELINE_CONFIG["queries_per_phase"],
    )
    w = BASELINE_CONFIG["W0"]
    theta = BASELINE_CONFIG["theta"]
    p, r = bq.evaluate_precision_recall(labelled, w, theta)
    elapsed = time.monotonic() - t0

    payload = {
        "track": "baseline",
        "block": block,
        "window": 10,
        "queries_per_phase": BASELINE_CONFIG["queries_per_phase"],
        "w": w.tolist(),
        "theta": theta,
        "precision": float(p),
        "recall": float(r),
        "f1": float(2 * p * r / (p + r)) if (p + r) > 0 else 0.0,
        "elapsed_seconds": elapsed,
    }
    _save_result_json(out_path, payload)
    msg_done = f"    → (P, R)=({p:.3f}, {r:.3f}), F1={payload['f1']:.3f}, {elapsed:.1f}s"
    print(msg_done); progress_fh.write(msg_done + "\n"); progress_fh.flush()
    return payload


# ---------------------------------------------------------------------------
# Main driver
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Paper 3B-Cal RQ4a full BO experiment driver",
    )
    p.add_argument("--easy", action="store_true", help="Run easy track (qpp=60, 4 windows)")
    p.add_argument("--hard", action="store_true", help="Run hard track (qpp=20, 2 windows)")
    p.add_argument("--baseline", action="store_true", help="Run baseline track (fixed W₀, 1 window)")
    p.add_argument("--all", action="store_true", help="Run all tracks (easy + hard + baseline)")
    p.add_argument("--smoke", action="store_true",
                   help="Quick smoke: 1 block × 1 window per track, fewer iters")
    p.add_argument("--out-root", default="results/cal/experiments",
                   help="Root for versioned output (rel. to repo root)")
    p.add_argument("--n-blocks", type=int, default=DEFAULT_N_BLOCKS,
                   help=f"Number of blocks (default {DEFAULT_N_BLOCKS}); no upper cap")
    p.add_argument("--start-block", type=int, default=0,
                   help="First block seed (default 0); use with --n-blocks to run a subset")
    p.add_argument("--resume-dir", default=None,
                   help="Reuse an existing run directory instead of creating a timestamped one "
                        "(required for incremental runs that extend a prior invocation)")
    return p


def _repo_root() -> Path:
    return _THIS.parents[4]


def main(argv: Optional[List[str]] = None) -> int:
    argv = argv if argv is not None else sys.argv[1:]
    args = build_parser().parse_args(argv)

    tracks_enabled = {
        "easy": args.easy or args.all,
        "hard": args.hard or args.all,
        "baseline": args.baseline or args.all,
    }
    if not any(tracks_enabled.values()) and not args.smoke:
        print("No tracks enabled. Use --easy, --hard, --baseline, --all, or --smoke.")
        return 1

    if args.smoke:
        tracks_enabled = {"easy": True, "hard": True, "baseline": True}

    repo_root = _repo_root()
    out_root = repo_root / args.out_root

    meta = _run_meta.capture(
        experiment="bo_rq4a_full",
        cli_args=argv,
        seed=args.start_block,
        extra={"tracks": [k for k, v in tracks_enabled.items() if v],
               "smoke": args.smoke, "start_block": args.start_block,
               "n_blocks": args.n_blocks},
    )
    if args.resume_dir:
        run_dir = Path(args.resume_dir)
        if not run_dir.is_absolute():
            run_dir = repo_root / run_dir
        if not run_dir.is_dir():
            print(f"ERROR: --resume-dir not found: {run_dir}")
            return 1
        print(f"[run_bo_experiment] resuming existing run_dir (checkpoint-reuse): {run_dir}")
    else:
        slug = _run_meta.slug_for_filename(meta)
        run_dir = out_root / slug
    run_dir.mkdir(parents=True, exist_ok=True)

    # Save run_meta.json
    with (run_dir / "run_meta.json").open("w") as fh:
        json.dump(meta, fh, indent=2, default=str)

    # Open progress log (append-mode for resume safety)
    progress_path = run_dir / "progress.log"
    progress_fh = progress_path.open("a")
    progress_fh.write(f"\n==== Run start: {meta['timestamp_utc']} ====\n")
    progress_fh.flush()

    blocks = list(range(args.start_block, args.start_block + args.n_blocks))
    if args.smoke:
        blocks = blocks[:SMOKE_CONFIG["n_blocks"]]

    print(f"[run_bo_experiment] output dir: {run_dir}")
    print(f"[run_bo_experiment] tracks enabled: {[k for k, v in tracks_enabled.items() if v]}")
    print(f"[run_bo_experiment] blocks: {blocks}")
    print(f"[run_bo_experiment] smoke mode: {args.smoke}")
    print()

    t_total0 = time.monotonic()

    # ---------------------- Easy track -------------------------------------
    if tracks_enabled["easy"]:
        print("=== Easy track ===")
        progress_fh.write("=== Easy track ===\n"); progress_fh.flush()
        cfg = EASY_CONFIG
        windows = [SMOKE_CONFIG["window_easy"]] if args.smoke else cfg["windows"]
        qpp = SMOKE_CONFIG["queries_per_phase_easy"] if args.smoke else cfg["queries_per_phase"]
        n_init = SMOKE_CONFIG["n_init"] if args.smoke else cfg["n_init"]
        n_iter = SMOKE_CONFIG["n_iter"] if args.smoke else cfg["n_iter"]
        for block in blocks:
            for window in windows:
                _run_bo_config("easy", block, window, qpp, n_init, n_iter, run_dir, progress_fh)
        print()

    # ---------------------- Hard track -------------------------------------
    if tracks_enabled["hard"]:
        print("=== Hard track ===")
        progress_fh.write("=== Hard track ===\n"); progress_fh.flush()
        cfg = HARD_CONFIG
        windows = [SMOKE_CONFIG["window_hard"]] if args.smoke else cfg["windows"]
        qpp = SMOKE_CONFIG["queries_per_phase_hard"] if args.smoke else cfg["queries_per_phase"]
        n_init = SMOKE_CONFIG["n_init"] if args.smoke else cfg["n_init"]
        n_iter = SMOKE_CONFIG["n_iter"] if args.smoke else cfg["n_iter"]
        for block in blocks:
            for window in windows:
                _run_bo_config("hard", block, window, qpp, n_init, n_iter, run_dir, progress_fh)
        print()

    # ---------------------- Baseline track ---------------------------------
    if tracks_enabled["baseline"]:
        print("=== Baseline track (fixed W₀, θ=0.75) ===")
        progress_fh.write("=== Baseline track ===\n"); progress_fh.flush()
        for block in blocks:
            _run_baseline_config(block, run_dir, progress_fh)
        print()

    t_total = time.monotonic() - t_total0
    msg = f"\n=== Total elapsed: {t_total:.1f}s ({t_total/60:.2f} min) ==="
    print(msg); progress_fh.write(msg + "\n"); progress_fh.close()

    # Append to global experiment log
    log_path = out_root / "EXPERIMENT_LOG.md"
    summary = (
        f"tracks={[k for k, v in tracks_enabled.items() if v]}; "
        f"n_blocks={len(blocks)}; smoke={args.smoke}; "
        f"elapsed={t_total:.1f}s; dir=`{run_dir.relative_to(repo_root)}`"
    )
    _run_meta.append_experiment_log(log_path, meta, summary)

    print(f"[run_bo_experiment] done. Run artifacts: {run_dir}")
    print(f"[run_bo_experiment] next: python code/experiments/cal/experiments/aggregate_bo_results.py --run-dir {run_dir.relative_to(repo_root)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
