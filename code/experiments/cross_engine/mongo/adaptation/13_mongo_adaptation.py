#!/usr/bin/env python3
"""
13_mongo_adaptation.py — Gate G3 paired-RCB adaptation comparison for Mongo.

Strict mirror of `experiments/v2_10seed/07_adaptation_comparison.py` with the
engine plug swapped Postgres→Mongo. Constants, window-0 init phase, paired
RCB seed sharing, gating semantics, and output schema are all maintained at
1:1 parity so cross-engine comparison is meaningful.

Parity contract with Postgres step 5
────────────────────────────────────
  STRATEGIES        = ['no_advisor','always_on','periodic','hsm_gated']
  N_BLOCKS          = 10
  N_WINDOWS         = 24       (4 phases × 6 windows)
  WIN_PER_PH        = 6
  QUERIES_PW        = 20
  THETA             = 0.75     (SIMILARITY threshold; invoke when score < θ)
  K_PERIODIC        = 3
  W0                = {R:0.25, V:0.20, T:0.20, A:0.20, P:0.15}  (from hsm_v2_core)
  detector          = full hsm_v2 (R+V+T+A+P)  via common.hsm_bridge
  window-0 init     = shared init window; all advisor-using strategies invoke
                       advisor exactly once at win=0; no_advisor stays clean
  workload_fp       = sha256 over the same qid-stream that the queries see,
                       so paired strategies in the same block share fp
  RCB seed offset   = block_seed XOR 0xA5A5 → param sampler RNG;
                       all strategies in the same block see identical
                       concrete pipelines.

Engine-specific deltas (intentional)
────────────────────────────────────
  - source: mydb_p3a.combined_clean (built by build_experiment_db.py)
  - advisor: createIndex / dropIndex on the candidate_index of the worst
             qid in the most-recent window; cleanup = drop all advisor
             indexes at end of block (markovian state per block)
  - phase schedule: edge → geo → text → review (24 windows, 6 each)
  - per-strategy seed family: BASE_SEED=9000 (distinct from Postgres 1000-band)

Outputs
───────
  results/cross_engine/mongo/adaptation/<timestamp>/
    block_metrics.csv          # one row per (strategy, block)
    breakdown_per_window.csv   # one row per (strategy, block, window) with
                               #   S_R, S_V, S_T, S_A, S_P, HSM, invoked,
                               #   drift_truth, exec_ms_window
    run_meta.json              # provenance bundle for reproducibility

Parallelism: DO NOT RUN during Postgres experiments.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

# ── Local imports ─────────────────────────────────────────────────────
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "workload"))
sys.path.insert(0, os.path.join(HERE, "..", "..", "common"))
from templates import (  # noqa: E402
    ALL_TEMPLATES,
    ALL_PHASES,
    ALL_QIDS_SORTED,
)
from param_sampler import materialize_pipeline  # noqa: E402
from window_features import make_window_features  # noqa: E402
from hsm_bridge import (  # noqa: E402
    compute_window_hsm,
    compute_window_hsm_breakdown,
    get_w0,
    is_available as hsm_available,
)


# ── Parity constants (mirror Postgres step 5) ─────────────────────────
STRATEGIES = ["no_advisor", "always_on", "periodic", "hsm_gated"]
N_BLOCKS = 10           # parity with Postgres step 5
N_WINDOWS = 24
WIN_PER_PH = 6
QUERIES_PW = 20         # parity with Postgres step 5
THETA = 0.75            # SIMILARITY threshold; invoke when hsm < THETA
K_PERIODIC = 3          # invoke every K-th window
BASE_SEED = 9000        # distinct from Postgres seed-band 1000-7000

# Phase schedule for 24 windows: edge × 6, geo × 6, text × 6, review × 6
PHASE_SCHEDULE = (
    ["edge"] * 6 + ["geo"] * 6 + ["text"] * 6 + ["review"] * 6
)
assert len(PHASE_SCHEDULE) == N_WINDOWS

SOURCE_DB = "mydb_p3a"
SOURCE_COLL = "combined_clean"

RESULTS_ROOT = os.path.abspath(
    os.path.join(HERE, "..", "..", "..", "..", "results", "cross_engine", "mongo", "adaptation")
)


def log(m: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {m}", flush=True)


# ───────────────────────────────────────────────────────────────────────
# Workload generator — paired-RCB deterministic per (block, seed)
# ───────────────────────────────────────────────────────────────────────

def generate_window(phase_name: str, n: int, rng: random.Random) -> list[str]:
    """Sample n qids from a phase mix using the phase weight vector."""
    phase = ALL_PHASES[phase_name]
    qids = list(phase["mix"].keys())
    weights = [phase["mix"][q] for q in qids]
    return rng.choices(qids, weights=weights, k=n)


def generate_block_workload(block_seed: int) -> list[list[str]]:
    """Generate a deterministic 24-window qid sequence for a single block.

    Includes a window-0 init draw so all 4 strategies share the same initial
    workload AND the same hsm baseline. Window 0 is from the FIRST phase
    (edge) by convention.
    """
    rng = random.Random(block_seed)
    windows: list[list[str]] = []
    # window 0 = init (always edge phase; matches Postgres ph0 = PHASES[0])
    windows.append(generate_window(PHASE_SCHEDULE[0], QUERIES_PW, rng))
    # windows 1..N_WINDOWS = measured
    for w_idx in range(1, N_WINDOWS + 1):
        ph = PHASE_SCHEDULE[w_idx - 1]
        windows.append(generate_window(ph, QUERIES_PW, rng))
    return windows  # length = N_WINDOWS + 1


def fingerprint(windows: list[list[str]]) -> str:
    """sha256 of all qids across all windows (incl. window 0). Cumulative,
    matching the Postgres `workload_fp.update(...)` style."""
    h = hashlib.sha256()
    for w in windows:
        h.update(("|".join(w) + "\n").encode())
    return h.hexdigest()[:16]


# ───────────────────────────────────────────────────────────────────────
# Mongo engine adapter
# ───────────────────────────────────────────────────────────────────────

def connect(uri: str):
    from pymongo import MongoClient
    c = MongoClient(uri, serverSelectionTimeoutMS=5000)
    c.admin.command("ping")
    return c


def ensure_backbone_indexes(coll) -> None:
    """Create the always-on backbone indexes (cheap, no workload signal).

    Backbone = the minimal set every strategy starts with. It does not by
    itself resolve any witness query, so the advisor still has meaningful
    work to do.
    """
    coll.create_index([("type", 1)], name="bb_type")
    coll.create_index([("label", 1)], name="bb_label")


def drop_advisor_indexes(coll) -> int:
    """Drop every index that is NOT _id_ and NOT a backbone (bb_*).

    Used at the start of every block (markovian per-block state) and at
    end-of-block cleanup. Returns count dropped.
    """
    n = 0
    for name in list(coll.index_information().keys()):
        if name == "_id_" or name.startswith("bb_"):
            continue
        try:
            coll.drop_index(name)
            n += 1
        except Exception as e:
            log(f"  drop_index({name}) failed: {e}")
    return n


def _index_name_for(cand_index: tuple) -> str:
    parts = []
    for k, d in cand_index:
        if d == 1:
            parts.append(f"{k}a")
        elif d == -1:
            parts.append(f"{k}d")
        else:
            parts.append(f"{k}{str(d)[:1]}")
    return "adv_" + "_".join(parts)


def invoke_advisor(coll, recent_exec_ms: list[float], recent_qids: list[str]) -> tuple[int, float]:
    """Pick the candidate index of the slowest qid in the most recent window
    and create it if missing. Returns (n_created, wall_ms_overhead).

    The "advisor" here is intentionally simple: it ranks qids by mean
    exec_ms within the recent window and creates up to 3 missing
    candidate indexes. Real index advisors (e.g., Dexter, AutoIndex) can
    be plugged in by replacing this single function.
    """
    if not recent_exec_ms:
        return (0, 0.0)
    from collections import defaultdict
    agg: dict[str, list[float]] = defaultdict(list)
    for ms, qid in zip(recent_exec_ms, recent_qids):
        agg[qid].append(ms)
    means = sorted(((sum(v) / len(v), q) for q, v in agg.items()), reverse=True)
    existing = set(coll.index_information().keys())
    n_created = 0
    t0 = time.perf_counter()
    for _, qid in means[:3]:
        cand = ALL_TEMPLATES[qid].candidate_index
        if not cand:
            continue
        name = _index_name_for(cand)
        if name in existing:
            continue
        try:
            coll.create_index(list(cand), name=name)
            n_created += 1
        except Exception as e:
            log(f"  create_index({name}) failed: {e}")
    overhead_ms = (time.perf_counter() - t0) * 1000
    return (n_created, overhead_ms)


def run_window(coll, qids: list[str], rng: random.Random) -> tuple[list[float], int]:
    """Execute a window of queries; return (per-query exec_ms list, n_ok)."""
    exec_ms: list[float] = []
    n_ok = 0
    for qid in qids:
        tmpl = ALL_TEMPLATES[qid]
        pipeline = materialize_pipeline(tmpl, rng)
        t0 = time.perf_counter()
        try:
            list(coll.aggregate(pipeline, allowDiskUse=True))
            n_ok += 1
        except Exception as e:
            log(f"  query {qid} failed: {e}")
        exec_ms.append((time.perf_counter() - t0) * 1000)
    return exec_ms, n_ok


# ───────────────────────────────────────────────────────────────────────
# Run one (strategy, block) cell
# ───────────────────────────────────────────────────────────────────────

def should_invoke(strategy: str, win_1based: int, hsm_score: float) -> bool:
    """Strategy decision (mirrors Postgres should_invoke).

    win_1based is 1..N_WINDOWS (window 0 is init, never decided here).
    Periodic: invoke at windows 1, 1+K, 1+2K, ...
    HSM-gated: invoke when SIMILARITY < THETA.
    """
    if strategy == "no_advisor":
        return False
    if strategy == "always_on":
        return True
    if strategy == "periodic":
        return ((win_1based - 1) % K_PERIODIC == 0)
    if strategy == "hsm_gated":
        return hsm_score < THETA
    return False


def run_block(client, strategy: str, block_idx: int, block_seed: int) -> dict:
    """Run one (strategy, block) cell of the paired-RCB design."""
    coll = client[SOURCE_DB][SOURCE_COLL]

    # Reset advisor state, keep backbone
    drop_advisor_indexes(coll)
    ensure_backbone_indexes(coll)

    # Build the deterministic qid stream for this block (incl. window 0)
    windows = generate_block_workload(block_seed)
    wp_fp = fingerprint(windows)

    # Param-sampling RNG: same offset for all strategies → identical concrete
    # pipelines across paired strategies in the same block.
    param_rng = random.Random(block_seed ^ 0xA5A5)

    # ── Window 0: shared init (parity with Postgres DESIGN FIX 2026-04-09) ──
    # All strategies execute window-0 queries to seed `prev_features`.
    # Only advisor-using strategies provision the advisor at window 0.
    init_qids = windows[0]
    init_exec_ms, _ = run_window(coll, init_qids, param_rng)
    prev_features = make_window_features(init_qids, init_exec_ms)

    if strategy != "no_advisor":
        invoke_advisor(coll, init_exec_ms, init_qids)

    # ── Windows 1..N_WINDOWS: measured ──
    advisor_calls = 0
    T_A_total_ms = 0.0
    queries_total = 0
    queries_ok = 0
    hsm_scores: list[float] = []
    breakdown_rows: list[dict] = []

    wall_start = time.perf_counter()

    for w_1based in range(1, N_WINDOWS + 1):
        qids = windows[w_1based]
        phase_name = PHASE_SCHEDULE[w_1based - 1]

        exec_ms, n_ok = run_window(coll, qids, param_rng)
        queries_total += len(qids)
        queries_ok += n_ok

        cur_features = make_window_features(qids, exec_ms)
        breakdown = compute_window_hsm_breakdown(prev_features, cur_features)
        hsm_score = breakdown["HSM"]
        hsm_scores.append(hsm_score)

        invoked = False
        if should_invoke(strategy, w_1based, hsm_score):
            _, t_a = invoke_advisor(coll, exec_ms, qids)
            advisor_calls += 1
            T_A_total_ms += t_a
            invoked = True

        # Drift truth label: True iff phase_name differs from previous window's phase.
        # Window 1 has no measured predecessor, so drift_truth = 0 by definition
        # (the init pass at w_1based=0 is treated as the same phase as w_1based=1).
        if w_1based == 1:
            drift_truth = False
        else:
            drift_truth = (PHASE_SCHEDULE[w_1based - 1] != PHASE_SCHEDULE[w_1based - 2])

        breakdown_rows.append({
            "block": block_idx,
            "block_seed": block_seed,
            "strategy": strategy,
            "window": w_1based,
            "phase": phase_name,
            "drift_truth": int(drift_truth),
            "S_R": breakdown["S_R"],
            "S_V": breakdown["S_V"],
            "S_T": breakdown["S_T"],
            "S_A": breakdown["S_A"],
            "S_P": breakdown["S_P"],
            "HSM": breakdown["HSM"],
            "invoked": int(invoked),
            "n_queries": len(qids),
            "n_ok": n_ok,
            "exec_ms_window_sum": round(sum(exec_ms), 3),
        })

        prev_features = cur_features

    wall_time_s = time.perf_counter() - wall_start
    wall_qps = queries_total / wall_time_s if wall_time_s > 0 else 0.0

    # End-of-block cleanup (markovian state)
    drop_advisor_indexes(coll)

    # Drift-boundary precision/recall (TPR/TNR computed offline by T4 post-processor)
    drift_windows = {b["window"] for b in breakdown_rows if b["drift_truth"] == 1}
    invoked_windows = {b["window"] for b in breakdown_rows if b["invoked"] == 1}
    n_drift = len(drift_windows)
    if strategy == "hsm_gated" and len(invoked_windows) > 0:
        tp = len(invoked_windows & drift_windows)
        fp = len(invoked_windows - drift_windows)
        fn = len(drift_windows - invoked_windows)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    else:
        precision = float("nan")
        recall = float("nan")

    return {
        "block_metrics": {
            "block": block_idx,
            "block_seed": block_seed,
            "strategy": strategy,
            "workload_fp": wp_fp,
            "wall_qps": round(wall_qps, 4),
            "wall_time_s": round(wall_time_s, 3),
            "queries_total": queries_total,
            "queries_ok": queries_ok,
            "advisor_calls": advisor_calls,
            "T_A_total_ms": round(T_A_total_ms, 2),
            "p_advisor": round(advisor_calls / N_WINDOWS, 4),
            "mean_hsm": round(float(np.mean(hsm_scores)), 4),
            "hsm_below_theta": int(sum(1 for h in hsm_scores if h < THETA)),
            "n_drift_points": n_drift,
            "precision": round(precision, 4) if precision == precision else float("nan"),
            "recall": round(recall, 4) if recall == recall else float("nan"),
            "hsm_series": ",".join(f"{x:.4f}" for x in hsm_scores),
            "phase_series": "|".join(PHASE_SCHEDULE),
            "theta": THETA,
            "k_periodic": K_PERIODIC,
        },
        "breakdown_rows": breakdown_rows,
    }


# ───────────────────────────────────────────────────────────────────────
# Output helpers
# ───────────────────────────────────────────────────────────────────────

def _git_sha() -> str:
    try:
        import subprocess
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=HERE, stderr=subprocess.DEVNULL,
        ).decode().strip()
        dirty = subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=HERE, stderr=subprocess.DEVNULL,
        ).decode().strip() != ""
        return f"{out}{'-dirty' if dirty else ''}"
    except Exception:
        return "unknown"


def _file_sha(p: str) -> str:
    try:
        with open(p, "rb") as f:
            return hashlib.sha256(f.read()).hexdigest()[:16]
    except Exception:
        return "unknown"


def write_outputs(outdir: Path, all_metrics: list[dict],
                  all_breakdowns: list[dict], meta: dict) -> None:
    outdir.mkdir(parents=True, exist_ok=True)

    bm_path = outdir / "block_metrics.csv"
    if all_metrics:
        with open(bm_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(all_metrics[0].keys()))
            w.writeheader()
            w.writerows(all_metrics)
    log(f"  wrote {bm_path}")

    bw_path = outdir / "breakdown_per_window.csv"
    if all_breakdowns:
        with open(bw_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(all_breakdowns[0].keys()))
            w.writeheader()
            w.writerows(all_breakdowns)
    log(f"  wrote {bw_path}")

    meta_path = outdir / "run_meta.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2, default=str)
    log(f"  wrote {meta_path}")


# ───────────────────────────────────────────────────────────────────────
# Driver
# ───────────────────────────────────────────────────────────────────────

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--uri", default="mongodb://localhost:27017")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--blocks", type=int, default=N_BLOCKS)
    ap.add_argument("--strategies", nargs="+", default=STRATEGIES)
    args = ap.parse_args()

    log(f"13_mongo_adaptation.py dry_run={args.dry_run} blocks={args.blocks}")
    log(f"  source        : {SOURCE_DB}.{SOURCE_COLL}")
    log(f"  strategies    : {args.strategies}")
    log(f"  N_WINDOWS     : {N_WINDOWS} (init=window0 + measured 1..{N_WINDOWS})")
    log(f"  QUERIES_PW    : {QUERIES_PW}")
    log(f"  THETA         : {THETA}  (similarity; invoke when score < θ)")
    log(f"  K_PERIODIC    : {K_PERIODIC}")
    log(f"  W0            : {get_w0()}")
    log(f"  hsm_v2 avail  : {hsm_available()}")
    log(f"  phase schedule: {PHASE_SCHEDULE}")

    if args.dry_run:
        log("DRY-RUN: validating workload generator + features (no mongod contact)")
        for b in range(min(3, args.blocks)):
            seed = BASE_SEED + b * 100
            ws = generate_block_workload(seed)
            fp = fingerprint(ws)
            log(f"  block {b:02d} seed={seed} fp={fp} "
                f"win0={ws[0][:5]}… win1={ws[1][:5]}… "
                f"len(windows)={len(ws)}")
            # Validate feature builder runs end-to-end with synthetic times
            fake_ms = [10.0] * QUERIES_PW
            f0 = make_window_features(ws[0], fake_ms)
            f1 = make_window_features(ws[1], fake_ms)
            sim = compute_window_hsm(f0, f1)
            bd = compute_window_hsm_breakdown(f0, f1)
            log(f"    sim(w0,w1)={sim:.4f}  breakdown={bd}")
        # Verify paired-RCB invariant: same seed → same fingerprint
        s = BASE_SEED + 0
        fp_a = fingerprint(generate_block_workload(s))
        fp_b = fingerprint(generate_block_workload(s))
        assert fp_a == fp_b, f"non-deterministic fingerprint: {fp_a} vs {fp_b}"
        log("  paired-RCB invariant OK (same seed → same fingerprint)")
        log("DRY-RUN OK")
        return 4

    # ── Real run ──
    if not hsm_available():
        log("FATAL: hsm_v2_core unavailable; cannot run real adaptation")
        return 2

    client = connect(args.uri)
    started_at = datetime.now().isoformat(timespec="seconds")
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    outdir = Path(RESULTS_ROOT) / ts

    all_metrics: list[dict] = []
    all_breakdowns: list[dict] = []

    try:
        for b in range(args.blocks):
            block_seed = BASE_SEED + b * 100
            for strategy in args.strategies:
                log(f"  block={b:02d} seed={block_seed} strategy={strategy}")
                result = run_block(client, strategy, b, block_seed)
                all_metrics.append(result["block_metrics"])
                all_breakdowns.extend(result["breakdown_rows"])
    finally:
        client.close()
        ended_at = datetime.now().isoformat(timespec="seconds")
        meta = {
            "engine": "mongo",
            "git_sha": _git_sha(),
            "n_blocks": args.blocks,
            "strategies": args.strategies,
            "constants": {
                "N_WINDOWS": N_WINDOWS,
                "WIN_PER_PH": WIN_PER_PH,
                "QUERIES_PW": QUERIES_PW,
                "THETA": THETA,
                "K_PERIODIC": K_PERIODIC,
                "BASE_SEED": BASE_SEED,
                "W0": get_w0(),
            },
            "phase_schedule": PHASE_SCHEDULE,
            "source": f"{SOURCE_DB}.{SOURCE_COLL}",
            "started_at": started_at,
            "ended_at": ended_at,
            "templates_sha": _file_sha(os.path.join(HERE, "..", "workload", "templates.py")),
            "hsm_v2_core_sha": _file_sha(os.path.join(HERE, "..", "..", "..", "v2_10seed", "hsm_v2_core.py")),
            "param_sampler_sha": _file_sha(os.path.join(HERE, "..", "..", "common", "param_sampler.py")),
            "window_features_sha": _file_sha(os.path.join(HERE, "..", "..", "common", "window_features.py")),
            "hsm_bridge_sha": _file_sha(os.path.join(HERE, "..", "..", "common", "hsm_bridge.py")),
            "command_line": " ".join(sys.argv),
        }
        write_outputs(outdir, all_metrics, all_breakdowns, meta)

    log(f"DONE. Results: {outdir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
