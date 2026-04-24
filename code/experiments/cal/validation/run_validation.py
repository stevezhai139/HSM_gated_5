"""CLI driver for Paper 3B-Cal empirical validation runs.

Invoked by the researcher manually:

    python code/experiments/cal/validation/run_validation.py \\
        --experiment job_static \\
        --out-root results/cal/validation \\
        --seed 42

Produces, per run, a versioned result directory::

    results/cal/validation/<experiment>/<YYYYMMDD-HHMMSS>_<sha>_<experiment>_seed<N>/
        result.json
        scores_per_pair.png
        trigger_vs_theta.png
        f1_vs_theta.png
        run_meta.json

and appends a row to
``results/cal/validation/EXPERIMENT_LOG.md``.

All paths are resolved relative to the HSM_gated repo root.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Make the cal.validation package importable when run as a script.
_THIS = Path(__file__).resolve()
_EXPERIMENTS = _THIS.parents[2]   # code/experiments/
if str(_EXPERIMENTS) not in sys.path:
    sys.path.insert(0, str(_EXPERIMENTS))

from cal.validation import _run_meta  # noqa: E402
from cal.validation.paper3a_loader import (  # noqa: E402
    list_experiments,
    load_experiment,
)
from cal.validation.theta_sweep import result_to_json_dict, sweep  # noqa: E402
from cal.validation.scenario_classifier import classify  # noqa: E402


def _repo_root() -> Path:
    return _THIS.parents[4]


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Paper 3B-Cal θ-sweep validation driver"
    )
    p.add_argument(
        "--experiment", required=True,
        choices=list_experiments() + ["all"],
        help="Paper 3A experiment key (or 'all')",
    )
    p.add_argument(
        "--out-root", default="results/cal/validation",
        help="Root directory for versioned outputs (relative to repo root)",
    )
    p.add_argument("--seed", type=int, default=42,
                   help="RNG seed (unused for deterministic sweep, recorded for provenance)")
    p.add_argument(
        "--theta-min", type=float, default=0.50,
        help="Grid lower bound (default 0.50)",
    )
    p.add_argument(
        "--theta-max", type=float, default=0.95,
        help="Grid upper bound (default 0.95)",
    )
    p.add_argument(
        "--theta-step", type=float, default=0.01,
        help="Grid step (default 0.01 → 46 points)",
    )
    p.add_argument(
        "--no-plots", action="store_true",
        help="Skip matplotlib figures (useful if matplotlib is unavailable)",
    )
    return p


def _build_grid(lo: float, hi: float, step: float) -> list[float]:
    if step <= 0:
        raise ValueError("--theta-step must be positive")
    out = []
    x = lo
    while x <= hi + 1e-9:
        out.append(round(x, 6))
        x += step
    return out


def _run_one(args, experiment: str, out_root: Path, cli_args: list[str]) -> dict:
    meta = _run_meta.capture(
        experiment=f"{experiment}_theta_sweep",
        cli_args=cli_args,
        seed=args.seed,
    )
    slug = _run_meta.slug_for_filename(meta)
    run_dir = out_root / experiment / slug
    run_dir.mkdir(parents=True, exist_ok=True)

    print(f"[run_validation] experiment={experiment} → {run_dir}")
    series = load_experiment(experiment, repo_root=_repo_root())
    print(f"  loaded: n_pairs={series.n_pairs}, n_transitions={series.n_transitions}, "
          f"phases={list(series.phases_seen)}")
    print(f"  paper3a_default_trigger_count={series.paper3a_default_trigger_count}")

    grid = _build_grid(args.theta_min, args.theta_max, args.theta_step)
    sweep_result = sweep(series, theta_grid=grid)
    scenario = classify(series, sweep_result)
    print(f"  scenario = {scenario.scenario}")
    print(f"  {scenario.justification}")

    # Emit result.json
    result = {
        "meta": meta,
        "scenario": {
            "letter": scenario.scenario,
            "justification": scenario.justification,
            "evidence": scenario.evidence,
        },
        "sweep": result_to_json_dict(series, sweep_result),
    }

    result_path = run_dir / "result.json"
    with result_path.open("w") as fh:
        json.dump(result, fh, indent=2, default=float)

    # Run meta side file for quick audit.
    with (run_dir / "run_meta.json").open("w") as fh:
        json.dump(meta, fh, indent=2, default=str)

    # Plots
    plot_paths: dict = {}
    if not args.no_plots:
        try:
            from cal.validation.plots import save_all_plots
            plot_paths = save_all_plots(series, sweep_result, run_dir)
            print(f"  plots written: {list(plot_paths.keys())}")
        except Exception as exc:  # pragma: no cover
            print(f"  WARNING: plot generation failed: {exc!s}")

    # One-line summary for EXPERIMENT_LOG.md
    best_f1 = sweep_result.theta_star_rates.get("max_f1")
    best_f1_summary = (
        f"θ*(F1)={best_f1.theta:.2f} F1={best_f1.f1:.3f} TP={best_f1.tp} "
        f"FP={best_f1.fp} FN={best_f1.fn}"
        if best_f1 is not None
        else "n/a"
    )
    summary = (
        f"scenario={scenario.scenario}; n_pairs={series.n_pairs} "
        f"n_transitions={series.n_transitions} "
        f"default_trig_count={series.paper3a_default_trigger_count}; "
        f"{best_f1_summary}; dir=`{run_dir.relative_to(_repo_root())}`"
    )
    _run_meta.append_experiment_log(
        out_root / "EXPERIMENT_LOG.md",
        meta,
        summary,
    )

    return {
        "experiment": experiment,
        "run_dir": str(run_dir),
        "scenario": scenario.scenario,
        "summary": summary,
    }


def main(argv: list[str] | None = None) -> int:
    argv = argv if argv is not None else sys.argv[1:]
    args = build_parser().parse_args(argv)
    out_root = _repo_root() / args.out_root

    targets = (
        list_experiments() if args.experiment == "all" else [args.experiment]
    )

    print(f"[run_validation] targets: {targets}")
    print(f"[run_validation] out_root: {out_root}")
    print(f"[run_validation] θ-grid: "
          f"[{args.theta_min}, {args.theta_max}] step {args.theta_step}")

    any_failed = False
    for experiment in targets:
        try:
            _run_one(args, experiment, out_root, list(argv))
            print("")
        except Exception as exc:
            any_failed = True
            print(f"ERROR running {experiment}: {exc!s}")

    return 0 if not any_failed else 1


if __name__ == "__main__":
    sys.exit(main())
