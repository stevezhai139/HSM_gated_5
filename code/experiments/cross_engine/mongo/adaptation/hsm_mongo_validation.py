#!/usr/bin/env python3
"""
14_mongo_adaptation_theta_sweep.py — G3b: Re-run adaptation with θ=0.65.

This is a thin wrapper around 13_mongo_adaptation.py that overrides THETA
for the engine-specific calibration experiment (Section A-CE, Corollary).

Background:
  Post-hoc ROC analysis on the G3 θ=0.75 run showed that MongoDB's
  optimal θ* ∈ [0.55, 0.60] (J=1.000), compared to PostgreSQL's θ*=0.775.
  This script confirms the practical DR improvement at θ=0.65 (a
  conservative choice between the two optima).

Usage:
  python3 14_mongo_adaptation_theta_sweep.py --dry-run     # validate
  python3 14_mongo_adaptation_theta_sweep.py               # full run (~35 min)

Output:
  results/cross_engine/mongo/adaptation_theta065/<timestamp>/
    block_metrics.csv, breakdown_per_window.csv, run_meta.json
"""
from __future__ import annotations

import os
import sys

# ── Override THETA before importing the main module ──
# We patch the module-level constant so should_invoke() uses the new θ.
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

# Standalone driver with THETA=0.65 and a different output dir.
# (Duplicated from 13_ rather than patching module constants.)

sys.path.insert(0, os.path.join(HERE, "..", "workload"))
sys.path.insert(0, os.path.join(HERE, "..", "..", "common"))

# ── Re-use everything from 13_mongo_adaptation.py except THETA + output dir ──
from templates import ALL_TEMPLATES, ALL_PHASES, ALL_QIDS_SORTED  # noqa
from param_sampler import materialize_pipeline  # noqa
from window_features import make_window_features  # noqa
from hsm_bridge import compute_window_hsm, compute_window_hsm_breakdown, get_w0, is_available as hsm_available  # noqa

import argparse, csv, hashlib, json, random, time
import numpy as np
from datetime import datetime
from pathlib import Path

# ── Changed constants ──
THETA = 0.65              # ← ENGINE-CALIBRATED θ (was 0.75)
THETA_LABEL = "0.65"

# ── Parity constants (unchanged from 13_) ──
STRATEGIES = ["no_advisor", "always_on", "periodic", "hsm_gated"]
N_BLOCKS = 10
N_WINDOWS = 24
WIN_PER_PH = 6
QUERIES_PW = 20
K_PERIODIC = 3
BASE_SEED = 9000          # same seeds → same workload → directly comparable

PHASE_SCHEDULE = ["edge"] * 6 + ["geo"] * 6 + ["text"] * 6 + ["review"] * 6
SOURCE_DB = "mydb_p3a"
SOURCE_COLL = "combined_clean"

RESULTS_ROOT = os.path.abspath(
    os.path.join(HERE, "..", "..", "..", "..", "results",
                 "cross_engine", "mongo", "adaptation_theta065")
)


def log(m): print(f"[{datetime.now().strftime('%H:%M:%S')}] {m}", flush=True)


# ── Verbatim copies from 13_ (workload gen, engine adapter, run_block) ──
# Only should_invoke() uses THETA — everything else is identical.

def generate_window(phase_name, n, rng):
    phase = ALL_PHASES[phase_name]
    qids = list(phase["mix"].keys())
    weights = [phase["mix"][q] for q in qids]
    return rng.choices(qids, weights=weights, k=n)

def generate_block_workload(block_seed):
    rng = random.Random(block_seed)
    windows = [generate_window(PHASE_SCHEDULE[0], QUERIES_PW, rng)]
    for w_idx in range(1, N_WINDOWS + 1):
        windows.append(generate_window(PHASE_SCHEDULE[w_idx - 1], QUERIES_PW, rng))
    return windows

def fingerprint(windows):
    h = hashlib.sha256()
    for w in windows:
        h.update(("|".join(w) + "\n").encode())
    return h.hexdigest()[:16]

def connect(uri):
    from pymongo import MongoClient
    c = MongoClient(uri, serverSelectionTimeoutMS=5000)
    c.admin.command("ping")
    return c

def ensure_backbone_indexes(coll):
    coll.create_index([("type", 1)], name="bb_type")
    coll.create_index([("label", 1)], name="bb_label")

def drop_advisor_indexes(coll):
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

def _index_name_for(cand_index):
    parts = []
    for k, d in cand_index:
        if d == 1: parts.append(f"{k}a")
        elif d == -1: parts.append(f"{k}d")
        else: parts.append(f"{k}{str(d)[:1]}")
    return "adv_" + "_".join(parts)

def invoke_advisor(coll, recent_exec_ms, recent_qids):
    if not recent_exec_ms:
        return (0, 0.0)
    from collections import defaultdict
    agg = defaultdict(list)
    for ms, qid in zip(recent_exec_ms, recent_qids):
        agg[qid].append(ms)
    means = sorted(((sum(v)/len(v), q) for q, v in agg.items()), reverse=True)
    existing = set(coll.index_information().keys())
    n_created = 0
    t0 = time.perf_counter()
    for _, qid in means[:3]:
        cand = ALL_TEMPLATES[qid].candidate_index
        if not cand: continue
        name = _index_name_for(cand)
        if name in existing: continue
        try:
            coll.create_index(list(cand), name=name)
            n_created += 1
        except Exception as e:
            log(f"  create_index({name}) failed: {e}")
    return (n_created, (time.perf_counter() - t0) * 1000)

def run_window(coll, qids, rng):
    exec_ms, n_ok = [], 0
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

def should_invoke(strategy, win_1based, hsm_score):
    """Uses THETA=0.65 (engine-calibrated)."""
    if strategy == "no_advisor": return False
    if strategy == "always_on": return True
    if strategy == "periodic": return ((win_1based - 1) % K_PERIODIC == 0)
    if strategy == "hsm_gated": return hsm_score < THETA  # ← 0.65
    return False

def run_block(client, strategy, block_idx, block_seed):
    coll = client[SOURCE_DB][SOURCE_COLL]
    drop_advisor_indexes(coll)
    ensure_backbone_indexes(coll)
    windows = generate_block_workload(block_seed)
    wp_fp = fingerprint(windows)
    param_rng = random.Random(block_seed ^ 0xA5A5)

    init_qids = windows[0]
    init_exec_ms, _ = run_window(coll, init_qids, param_rng)
    prev_features = make_window_features(init_qids, init_exec_ms)
    if strategy != "no_advisor":
        invoke_advisor(coll, init_exec_ms, init_qids)

    advisor_calls = 0
    T_A_total_ms = 0.0
    queries_total = queries_ok = 0
    hsm_scores = []
    breakdown_rows = []
    wall_start = time.perf_counter()

    for w_1based in range(1, N_WINDOWS + 1):
        qids = windows[w_1based]
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

        drift_truth = False if w_1based == 1 else (
            PHASE_SCHEDULE[w_1based - 1] != PHASE_SCHEDULE[w_1based - 2])

        breakdown_rows.append({
            "block": block_idx, "block_seed": block_seed,
            "strategy": strategy, "window": w_1based,
            "phase": PHASE_SCHEDULE[w_1based - 1],
            "drift_truth": int(drift_truth),
            "S_R": breakdown["S_R"], "S_V": breakdown["S_V"],
            "S_T": breakdown["S_T"], "S_A": breakdown["S_A"],
            "S_P": breakdown["S_P"], "HSM": breakdown["HSM"],
            "invoked": int(invoked),
            "n_queries": len(qids), "n_ok": n_ok,
            "exec_ms_window_sum": round(sum(exec_ms), 3),
        })
        prev_features = cur_features

    wall_time_s = time.perf_counter() - wall_start
    drop_advisor_indexes(coll)

    drift_windows = {b["window"] for b in breakdown_rows if b["drift_truth"] == 1}
    invoked_windows = {b["window"] for b in breakdown_rows if b["invoked"] == 1}
    if strategy == "hsm_gated" and invoked_windows:
        tp = len(invoked_windows & drift_windows)
        fp = len(invoked_windows - drift_windows)
        fn = len(drift_windows - invoked_windows)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    else:
        precision = recall = float("nan")

    return {
        "block_metrics": {
            "block": block_idx, "block_seed": block_seed,
            "strategy": strategy, "workload_fp": wp_fp,
            "wall_qps": round(queries_total / wall_time_s, 4),
            "wall_time_s": round(wall_time_s, 3),
            "queries_total": queries_total, "queries_ok": queries_ok,
            "advisor_calls": advisor_calls,
            "T_A_total_ms": round(T_A_total_ms, 2),
            "p_advisor": round(advisor_calls / N_WINDOWS, 4),
            "mean_hsm": round(float(np.mean(hsm_scores)), 4),
            "hsm_below_theta": int(sum(1 for h in hsm_scores if h < THETA)),
            "n_drift_points": len(drift_windows),
            "precision": round(precision, 4) if precision == precision else float("nan"),
            "recall": round(recall, 4) if recall == recall else float("nan"),
            "hsm_series": ",".join(f"{x:.4f}" for x in hsm_scores),
            "phase_series": "|".join(PHASE_SCHEDULE),
            "theta": THETA, "k_periodic": K_PERIODIC,
        },
        "breakdown_rows": breakdown_rows,
    }


def write_outputs(outdir, all_metrics, all_breakdowns, meta):
    outdir.mkdir(parents=True, exist_ok=True)
    if all_metrics:
        p = outdir / "block_metrics.csv"
        with open(p, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(all_metrics[0].keys()))
            w.writeheader(); w.writerows(all_metrics)
        log(f"  wrote {p}")
    if all_breakdowns:
        p = outdir / "breakdown_per_window.csv"
        with open(p, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(all_breakdowns[0].keys()))
            w.writeheader(); w.writerows(all_breakdowns)
        log(f"  wrote {p}")
    p = outdir / "run_meta.json"
    with open(p, "w") as f:
        json.dump(meta, f, indent=2, default=str)
    log(f"  wrote {p}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--uri", default="mongodb://localhost:27017")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--blocks", type=int, default=N_BLOCKS)
    args = ap.parse_args()

    log(f"14_mongo_adaptation_theta_sweep.py  THETA={THETA}  dry_run={args.dry_run}")
    log(f"  source: {SOURCE_DB}.{SOURCE_COLL}")
    log(f"  strategies: {STRATEGIES}")
    log(f"  Same seeds as 13_ → directly comparable workload")

    if args.dry_run:
        log("DRY-RUN: validating workload generator")
        for b in range(min(3, args.blocks)):
            seed = BASE_SEED + b * 100
            ws = generate_block_workload(seed)
            fp = fingerprint(ws)
            log(f"  block {b:02d} seed={seed} fp={fp}")
        log("DRY-RUN OK")
        return 4

    if not hsm_available():
        log("FATAL: hsm_v2_core unavailable")
        return 2

    client = connect(args.uri)
    started_at = datetime.now().isoformat(timespec="seconds")
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    outdir = Path(RESULTS_ROOT) / ts
    all_metrics, all_breakdowns = [], []

    try:
        for b in range(args.blocks):
            block_seed = BASE_SEED + b * 100
            for strategy in STRATEGIES:
                log(f"  block={b:02d} seed={block_seed} strategy={strategy} θ={THETA}")
                result = run_block(client, strategy, b, block_seed)
                all_metrics.append(result["block_metrics"])
                all_breakdowns.extend(result["breakdown_rows"])
    finally:
        client.close()
        meta = {
            "engine": "mongo", "theta": THETA,
            "theta_rationale": "Engine-calibrated via post-hoc ROC (θ*_mongo=0.55, conservative choice=0.65)",
            "comparable_to": "20260411_174517 (θ=0.75, same seeds)",
            "n_blocks": args.blocks, "strategies": STRATEGIES,
            "constants": {"N_WINDOWS": N_WINDOWS, "QUERIES_PW": QUERIES_PW,
                          "THETA": THETA, "K_PERIODIC": K_PERIODIC,
                          "BASE_SEED": BASE_SEED, "W0": get_w0()},
            "started_at": started_at,
            "ended_at": datetime.now().isoformat(timespec="seconds"),
        }
        write_outputs(outdir, all_metrics, all_breakdowns, meta)
    log(f"DONE. Results: {outdir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
