"""Smoke-test for BO-qNEHVI calibration per D5 §7.1.

Runs the exact configuration specified in the Execution Plan:
    1 block × 10 BO iterations × window = 10

and evaluates all 6 pass criteria. Prints a clear PASS/FAIL verdict.

Usage (from HSM_gated repo root):

    python code/experiments/cal/experiments/smoke_test_bo_qnehvi.py

Output: verdict on stdout + detailed trace JSON at:
    results/cal/experiments/smoke_<timestamp>_<sha>/smoke_result.json

Exit code 0 = all 6 criteria pass (OK to proceed to overnight run).
Exit code 1 = any criterion failed (abandon per D5 §7.2).
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

# sys.path fixup so `cal.*` resolves
_THIS = Path(__file__).resolve()
_EXPERIMENTS_DIR = _THIS.parents[2]
if str(_EXPERIMENTS_DIR) not in sys.path:
    sys.path.insert(0, str(_EXPERIMENTS_DIR))

import numpy as np

from cal.experiments import bo_qnehvi_calibration as bq
from cal.validation import _run_meta


# D5 §7.1 — smoke-test configuration (do not override without updating the plan)
SMOKE_BLOCK_SEED = 42
SMOKE_N_INIT = 10
SMOKE_N_ITER = 10
SMOKE_WINDOW_SIZE = 10
SMOKE_QUERIES_PER_PHASE = 60  # matches overnight config
EPS = bq.EPS


def _check_all_finite_pr(result: bq.BOQnehviResult) -> Tuple[bool, str]:
    for h in result.history:
        if not (math.isfinite(h.precision) and math.isfinite(h.recall)):
            return False, f"iter {h.iter_index}: non-finite (P, R) = ({h.precision}, {h.recall})"
        if not (0.0 <= h.precision <= 1.0 and 0.0 <= h.recall <= 1.0):
            return False, f"iter {h.iter_index}: (P, R) out of [0, 1]² = ({h.precision}, {h.recall})"
    return True, f"all {len(result.history)} iterations have finite (P, R) ∈ [0, 1]²"


def _check_hv_non_decreasing(result: bq.BOQnehviResult) -> Tuple[bool, str]:
    hvs = [h.hypervolume for h in result.history]
    for i in range(1, len(hvs)):
        if hvs[i] < hvs[i - 1] - 1e-9:
            return False, f"HV decreased at iter {i}: {hvs[i-1]:.6f} → {hvs[i]:.6f}"
    return True, f"HV history monotone non-decreasing (final = {hvs[-1]:.4f})"


def _check_pareto_has_2plus(result: bq.BOQnehviResult) -> Tuple[bool, str]:
    if len(result.pareto_indices) < 2:
        return False, f"Pareto front has {len(result.pareto_indices)} point(s), need ≥ 2"
    return True, f"Pareto front has {len(result.pareto_indices)} non-dominated points"


def _check_proposal_variation(result: bq.BOQnehviResult) -> Tuple[bool, str]:
    """Gate is learning if proposed (w, θ) differ across iterations.

    Under M10 direct-simplex parameterisation, BOIterationRecord stores
    w (5-D simplex) and theta (1-D) directly; there is no latent z.
    """
    iterations = [h for h in result.history if h.iter_index >= 0]
    if len(iterations) < 2:
        return True, "only 1 BO iteration — cannot check variation"
    # Compare pairwise L2 distance in (w, θ) space (6-D total)
    max_dist = 0.0
    for i in range(len(iterations)):
        for j in range(i + 1, len(iterations)):
            p_i = np.array(iterations[i].w + [iterations[i].theta])
            p_j = np.array(iterations[j].w + [iterations[j].theta])
            d = float(np.linalg.norm(p_i - p_j))
            max_dist = max(max_dist, d)
    if max_dist < 1e-4:
        return False, f"BO proposals near-identical (max L2 = {max_dist:.2e}) — GP not learning"
    return True, f"BO proposals show variation (max pairwise L2 = {max_dist:.3f})"


def _check_simplex_interior(result: bq.BOQnehviResult) -> Tuple[bool, str]:
    for h in result.history:
        w = np.array(h.w)
        if not np.all(w > 0):
            return False, f"iter {h.iter_index}: w has non-positive component: {w}"
        if not math.isclose(w.sum(), 1.0, abs_tol=1e-6):
            return False, f"iter {h.iter_index}: w sum = {w.sum()} ≠ 1"
    return True, "all w ∈ Δ⁴ strictly interior (Σw_i = 1 ± 1e-6)"


def _check_theta_bounded(result: bq.BOQnehviResult) -> Tuple[bool, str]:
    for h in result.history:
        if not (EPS <= h.theta <= 1.0 - EPS):
            return False, f"iter {h.iter_index}: θ = {h.theta} breaches (ε, 1−ε)"
    return True, f"all θ in [ε, 1−ε] = [{EPS}, {1-EPS:.6f}]"


CRITERIA = [
    ("1. All iterations return finite (P, R) ∈ [0, 1]²", _check_all_finite_pr),
    ("2. Hypervolume history non-decreasing",          _check_hv_non_decreasing),
    ("3. Final Pareto front has ≥ 2 non-dominated points", _check_pareto_has_2plus),
    ("4. Proposed (z, θ) sequence shows variation (GP is learning)", _check_proposal_variation),
    ("5. Softmax produces w ∈ Δ⁴ strictly interior",  _check_simplex_interior),
    ("6. θ proposals stay in (ε, 1−ε)",                _check_theta_bounded),
]


def main() -> int:
    print("=" * 70)
    print("BO-qNEHVI smoke-test — D5 §7.1")
    print("=" * 70)
    print(f"Config: block_seed={SMOKE_BLOCK_SEED}, window_size={SMOKE_WINDOW_SIZE}, "
          f"queries_per_phase={SMOKE_QUERIES_PER_PHASE}, "
          f"n_init={SMOKE_N_INIT}, n_iter={SMOKE_N_ITER}")
    print("")
    print("Running BO... (this may take 1-3 minutes)")

    cli_args = [
        f"--block-seed={SMOKE_BLOCK_SEED}",
        f"--window-size={SMOKE_WINDOW_SIZE}",
        f"--queries-per-phase={SMOKE_QUERIES_PER_PHASE}",
        f"--n-init={SMOKE_N_INIT}",
        f"--n-iter={SMOKE_N_ITER}",
    ]
    meta = _run_meta.capture(
        experiment="bo_qnehvi_smoke",
        cli_args=cli_args,
        seed=SMOKE_BLOCK_SEED,
    )

    try:
        result = bq.run_single_configuration(
            block_seed=SMOKE_BLOCK_SEED,
            window_size=SMOKE_WINDOW_SIZE,
            queries_per_phase=SMOKE_QUERIES_PER_PHASE,
            n_init=SMOKE_N_INIT,
            n_iter=SMOKE_N_ITER,
            verbose=True,
        )
    except Exception as exc:
        print("")
        print("❌ SMOKE-TEST FAILED: run_single_configuration raised exception")
        print(f"   {type(exc).__name__}: {exc}")
        import traceback
        traceback.print_exc()
        return 1

    print("")
    print("-" * 70)
    print("Evaluating 6 pass criteria (D5 §7.1):")
    print("-" * 70)

    verdicts: List[Dict[str, Any]] = []
    all_pass = True
    for label, check in CRITERIA:
        ok, msg = check(result)
        mark = "✅" if ok else "❌"
        print(f"{mark} {label}")
        print(f"   {msg}")
        verdicts.append({"criterion": label, "pass": ok, "detail": msg})
        if not ok:
            all_pass = False

    print("")
    print("-" * 70)
    print("Final summary:")
    print(f"  Iterations run: {len(result.history)} "
          f"({result.n_init} Sobol init + {result.n_iter} BO)")
    print(f"  Final (w*, θ*): w = {[f'{x:.3f}' for x in result.final_w_star]}, "
          f"θ = {result.final_theta_star:.4f}")
    print(f"  Final Pareto front size: {len(result.pareto_indices)}")
    print(f"  Final hypervolume: {result.final_hypervolume:.4f}")
    print(f"  Final (P, R) at knee: ({result.final_precision:.3f}, "
          f"{result.final_recall:.3f})")
    print("")
    if all_pass:
        print("✅ ALL 6 CRITERIA PASSED — safe to proceed to overnight run.")
    else:
        print("❌ AT LEAST ONE CRITERION FAILED — abandon per D5 §7.2; do NOT overnight-run.")
    print("-" * 70)

    # Persist result artifact
    slug = _run_meta.slug_for_filename(meta)
    out_dir = _EXPERIMENTS_DIR.parents[1] / "results" / "cal" / "experiments" / f"smoke_{slug}"
    out_dir.mkdir(parents=True, exist_ok=True)
    artifact = {
        "meta": meta,
        "config": {
            "block_seed": SMOKE_BLOCK_SEED,
            "window_size": SMOKE_WINDOW_SIZE,
            "queries_per_phase": SMOKE_QUERIES_PER_PHASE,
            "n_init": SMOKE_N_INIT,
            "n_iter": SMOKE_N_ITER,
        },
        "result": result.as_dict(),
        "criteria_verdicts": verdicts,
        "all_pass": all_pass,
    }
    with (out_dir / "smoke_result.json").open("w") as fh:
        json.dump(artifact, fh, indent=2, default=float)
    print(f"\nArtifact written to: {out_dir}/smoke_result.json")

    # Append to experiment log
    summary_line = (
        f"pass={all_pass}; iters={len(result.history)}; "
        f"final_hv={result.final_hypervolume:.4f}; "
        f"final_theta_star={result.final_theta_star:.4f}; "
        f"pareto_size={len(result.pareto_indices)}"
    )
    log_path = _EXPERIMENTS_DIR.parents[1] / "results" / "cal" / "experiments" / "EXPERIMENT_LOG.md"
    _run_meta.append_experiment_log(log_path, meta, summary_line)

    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
