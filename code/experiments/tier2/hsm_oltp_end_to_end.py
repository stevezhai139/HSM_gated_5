"""
hsm_oltp_end_to_end.py — Tier-2 end-to-end advisor study on pgbench (OLTP)
==========================================================================

Measures HSM-gated index advising against two production-grade advisors
(Dexter and Supabase index_advisor) on a write-heavy OLTP workload
(pgbench TPC-B schema).  Complements the TPC-H end-to-end study in
``experiment_runner.py``:

    ``experiment_runner.py``      — OLAP, TPC-H, stub advisor (column-map)
    ``hsm_oltp_end_to_end.py``    — OLTP, pgbench TPC-B, real advisors
    ``hsm_burst_end_to_end.py``   — burst_v2 synthetic, real advisors

DESIGN
──────
Workload
    4 phases × 30 queries = 120 queries, window size W=5 → 24 windows.
    Phase order: ReadOnly → WriteHeavy → Mixed → BulkRead
    Phase transitions at windows {6, 12, 18}.
    Query pool is the same as ``hsm_oltp_validation.py`` (inlined below so
    this file stays import-safe).

Conditions  (7 unique runs per block)
    baseline                    — no advisor; only the FK/PK indexes pgbench
                                  creates at init time
    always_on      × dexter     — advisor called after every window
    always_on      × supabase
    hsm_gated      × dexter     — advisor called iff HSM(w_{i-1}, w_i) < θ
    hsm_gated      × supabase
    periodic       × dexter     — advisor called every K=3 windows
    periodic       × supabase

Block design
    10 blocks, randomised condition order per block (seed=block).
    Each block regenerates a trace with seed=block so parameter variants
    differ across blocks (removes page-cache reuse while keeping query
    structure identical).

Per-run metrics recorded
    wall_time_s, total_query_time_s, total_advisor_time_s
    num_queries, errors, qps, wall_qps
    advisor_calls, trigger_rate, trigger_precision (fraction of triggers
      that land at a real phase transition)
    per-query latency (ms)                    → p50, p95, p99
    per-window: HSM score, triggered flag, advisor_wall_s, phase
    latency_on_skip_ms  (mean in-window latency when the gate skipped)
    latency_on_trigger_ms

Database
    Docker PostgreSQL 16 (custom image hsm-postgres:16-hypopg) on port 5433.
    Database: ``oltp``  (precondition — see docs/Tier2_SETUP.md).
    Extensions required: hypopg, index_advisor.
    pgbench schema at scale 10 (≈1M accounts).

Usage
    source .env
    # initialise the OLTP database once:
    #     createdb -h localhost -p 5433 -U postgres oltp
    #     pgbench -i -s 10 -h localhost -p 5433 -U postgres oltp
    #     psql -h localhost -p 5433 -U postgres oltp \
    #          -c "CREATE EXTENSION IF NOT EXISTS hypopg;" \
    #          -c "CREATE EXTENSION IF NOT EXISTS index_advisor;"
    python hsm_oltp_end_to_end.py --smoke           # 1 block, 1 advisor — wiring test
    python hsm_oltp_end_to_end.py                   # full 10 blocks × 7 conditions
    python hsm_oltp_end_to_end.py --blocks 3        # shorter run
    python hsm_oltp_end_to_end.py --advisor dexter  # restrict to one advisor
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import random
import re
import statistics
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Iterable

# psycopg2 is optional at import-time so CPU-only smoke scripts can load
# this module for its shared helpers; DB entrypoints check _HAS_PSYCOPG2.
try:
    import psycopg2
    import psycopg2.extensions
    _HAS_PSYCOPG2 = True
except ImportError:
    psycopg2 = None  # type: ignore[assignment]
    _HAS_PSYCOPG2 = False

# ─── Path bootstrap so we can import from code/experiments/ ───────────────────
_HERE = Path(__file__).resolve().parent
_EXPS = _HERE.parent                          # code/experiments
_REPO = _HERE.parents[2]                      # HSM_gated repo root
sys.path.insert(0, str(_EXPS))
sys.path.insert(0, str(_HERE))

from hsm_similarity import build_window, should_trigger_advisor, DEFAULT_THETA  # noqa: E402
from advisor_wrappers import (                                                   # noqa: E402
    DexterAdvisor, SupabaseIndexAdvisor, apply_recommendations,
)

# ─── Configuration ────────────────────────────────────────────────────────────
DOCKER_HOST = os.environ.get("HSM_DOCKER_HOST", "localhost")
DOCKER_PORT = int(os.environ.get("HSM_DOCKER_PORT", "5433"))
DOCKER_USER = os.environ.get("HSM_DOCKER_USER", "postgres")
DOCKER_PASS = os.environ.get("HSM_DOCKER_PASSWORD", "postgres")
DOCKER_DB   = os.environ.get("HSM_DOCKER_OLTP_DB", "oltp")

QUERIES_PER_PHASE = 30
PHASE_ORDER       = ["ReadOnly", "WriteHeavy", "Mixed", "BulkRead"]
WINDOW_SIZE       = 5        # 120 / 5 = 24 windows; transitions at {6,12,18}
THETA             = DEFAULT_THETA    # 0.75
PERIODIC_K        = 3
N_BLOCKS          = 10
STATEMENT_TIMEOUT_MS = 30_000
SETTLE_SEC        = 2

ADVISORS    = ["dexter", "supabase"]
POLICIES    = ["always_on", "hsm_gated", "periodic"]

# A baseline "no-advisor" run (policy=baseline, advisor=none) is also collected
# once per block so each block has a fair reference point that shares the same
# trace seed.  All advisor × policy combinations are then compared to this.

# ─── Logging ───────────────────────────────────────────────────────────────────
RESULTS_DIR = _REPO / "results" / "tier2_oltp"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(RESULTS_DIR / "oltp_end_to_end.log"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger("tier2.oltp")

# Silence the advisor wrapper's info lines inside the per-window hot loop —
# we log our own line per window that rolls up the same information.
logging.getLogger("hsm.tier2.advisors").setLevel(logging.WARNING)


# =============================================================================
# 1. Query pool  (mirrors hsm_oltp_validation.py — same phase semantics)
# =============================================================================
AIDS    = [1, 42, 1000, 5000, 10000, 50000, 100000,
           250000, 500000, 750000, 900000, 999999]
BIDS    = list(range(1, 11))
TIDS    = [1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
DELTAS  = [100, 250, 500, 1000, 2500]
RANGES  = [(1, 100000), (100001, 200000), (200001, 400000),
           (400001, 600000), (600001, 800000), (800001, 1000000)]


def build_query_pool() -> list[tuple[str, str, str, str]]:
    """Return list of (qid, phase, qtype, sql).  See hsm_oltp_validation."""
    pool: list[tuple[str, str, str, str]] = []
    n = [0]

    def add(phase: str, qtype: str, sql: str) -> None:
        n[0] += 1
        pool.append((f"{phase[:2]}{n[0]:03d}", phase, qtype, sql))

    # ReadOnly
    for aid in AIDS:
        add("ReadOnly", "SELECT",
            f"SELECT aid, bid, abalance FROM pgbench_accounts WHERE aid = {aid}")
    for tid in TIDS[:6]:
        add("ReadOnly", "SELECT",
            f"SELECT tid, bid, tbalance FROM pgbench_tellers WHERE tid = {tid}")
    for bid in BIDS[:5]:
        add("ReadOnly", "SELECT",
            f"SELECT bid, bbalance FROM pgbench_branches WHERE bid = {bid}")
    for aid in AIDS[:7]:
        add("ReadOnly", "SELECT",
            f"SELECT a.aid, a.abalance, b.bbalance "
            f"FROM pgbench_accounts a JOIN pgbench_branches b ON a.bid = b.bid "
            f"WHERE a.aid = {aid}")

    # WriteHeavy
    for aid, d in zip(AIDS, DELTAS * 3):
        add("WriteHeavy", "UPDATE",
            f"UPDATE pgbench_accounts SET abalance = abalance + {d} "
            f"WHERE aid = {aid}")
    for tid, d in zip(TIDS[:8], DELTAS * 2):
        add("WriteHeavy", "UPDATE",
            f"UPDATE pgbench_tellers SET tbalance = tbalance + {d} "
            f"WHERE tid = {tid}")
    for bid, d in zip(BIDS[:5], DELTAS):
        add("WriteHeavy", "UPDATE",
            f"UPDATE pgbench_branches SET bbalance = bbalance + {d} "
            f"WHERE bid = {bid}")
    for i, (aid, d) in enumerate(zip(AIDS[:8], DELTAS * 2)):
        tid, bid = TIDS[i % len(TIDS)], BIDS[i % len(BIDS)]
        add("WriteHeavy", "INSERT",
            f"INSERT INTO pgbench_history(tid, bid, aid, delta, mtime) "
            f"VALUES ({tid}, {bid}, {aid}, {d}, NOW())")
    for aid in AIDS[:3]:
        add("WriteHeavy", "SELECT",
            f"SELECT abalance FROM pgbench_accounts WHERE aid = {aid}")

    # Mixed  — must total 30 queries
    for aid, d in zip(AIDS[:8], DELTAS * 2):
        add("Mixed", "SELECT",
            f"SELECT abalance FROM pgbench_accounts WHERE aid = {aid}")
        add("Mixed", "UPDATE",
            f"UPDATE pgbench_accounts SET abalance = abalance + {d} "
            f"WHERE aid = {aid}")
    for tid in TIDS[:5]:
        add("Mixed", "SELECT",
            f"SELECT tbalance FROM pgbench_tellers WHERE tid = {tid}")
    for i, aid in enumerate(AIDS[:5]):
        tid, bid = TIDS[i % len(TIDS)], BIDS[i % len(BIDS)]
        add("Mixed", "INSERT",
            f"INSERT INTO pgbench_history(tid, bid, aid, delta, mtime) "
            f"VALUES ({tid}, {bid}, {aid}, 500, NOW())")
    # Padding: 4 more balanced read-then-update pairs on branches/tellers so
    # the phase has exactly 30 queries and the phase transition lands at w18.
    for tid in TIDS[5:7]:
        add("Mixed", "SELECT",
            f"SELECT tbalance FROM pgbench_tellers WHERE tid = {tid}")
        add("Mixed", "UPDATE",
            f"UPDATE pgbench_tellers SET tbalance = tbalance + 100 "
            f"WHERE tid = {tid}")

    # BulkRead
    for bid in BIDS:
        add("BulkRead", "SELECT",
            f"SELECT COUNT(*), SUM(abalance), AVG(abalance), MAX(abalance) "
            f"FROM pgbench_accounts WHERE bid = {bid}")
    for lo, hi in RANGES:
        add("BulkRead", "SELECT",
            f"SELECT COUNT(*), SUM(abalance) "
            f"FROM pgbench_accounts WHERE aid BETWEEN {lo} AND {hi}")
    for _ in range(3):
        add("BulkRead", "SELECT",
            "SELECT bid, COUNT(*), SUM(abalance), AVG(abalance) "
            "FROM pgbench_accounts GROUP BY bid ORDER BY bid")
    for top_n in (10, 50, 100):
        add("BulkRead", "SELECT",
            f"SELECT aid, abalance FROM pgbench_accounts "
            f"ORDER BY abalance DESC LIMIT {top_n}")
    for bid in BIDS[:5]:
        add("BulkRead", "SELECT",
            f"SELECT a.bid, COUNT(a.aid), SUM(a.abalance), b.bbalance "
            f"FROM pgbench_accounts a JOIN pgbench_branches b ON a.bid = b.bid "
            f"WHERE a.bid = {bid} GROUP BY a.bid, b.bbalance")
    # Padding: 3 more heavy aggregates so the phase totals exactly 30.
    for lo, hi in RANGES[:3]:
        add("BulkRead", "SELECT",
            f"SELECT AVG(abalance), MIN(abalance), MAX(abalance) "
            f"FROM pgbench_accounts WHERE aid BETWEEN {lo} AND {hi}")

    return pool


def build_trace(pool: list[tuple[str, str, str, str]], seed: int) \
        -> list[tuple[str, str, str, str]]:
    """Concatenate phase blocks in fixed order, shuffling members per phase."""
    rng = random.Random(seed)
    trace: list[tuple[str, str, str, str]] = []
    for phase in PHASE_ORDER:
        members = [r for r in pool if r[1] == phase]
        rng.shuffle(members)
        # Clip to QUERIES_PER_PHASE so the total is deterministic = 120.
        trace.extend(members[:QUERIES_PER_PHASE])
    return trace


def split_windows(trace: list, w: int = WINDOW_SIZE) -> list[list]:
    """Contiguous, non-overlapping windows."""
    return [trace[i:i + w] for i in range(0, len(trace), w) if len(trace[i:i + w]) == w]


# Window transitions (window_idx → True when the window crosses a phase
# boundary compared to the previous window).  With W=5 and 30 queries/phase,
# phase boundaries land exactly at window starts 6, 12, 18.
TRUE_TRANSITION_WINDOWS = {6, 12, 18}


# =============================================================================
# 2. Database helpers
# =============================================================================
def get_connection(dbname: str = DOCKER_DB, autocommit: bool = True):
    conn = psycopg2.connect(
        host=DOCKER_HOST, port=DOCKER_PORT, dbname=dbname,
        user=DOCKER_USER, password=DOCKER_PASS, connect_timeout=10,
        options=f"-c statement_timeout={STATEMENT_TIMEOUT_MS}",
    )
    if autocommit:
        conn.set_session(autocommit=True)
    with conn.cursor() as cur:
        cur.execute("SET lock_timeout = '10s'")
        cur.execute("SET work_mem = '64MB'")
        cur.execute("SET random_page_cost = 1.1")
    return conn


def check_prerequisites(conn) -> None:
    """Verify schema + extensions are present; raise with a clear message if not."""
    with conn.cursor() as cur:
        cur.execute(
            "SELECT to_regclass('public.pgbench_accounts')::text, "
            "       to_regclass('public.pgbench_history')::text"
        )
        accounts, history = cur.fetchone()
        if accounts is None or history is None:
            raise RuntimeError(
                f"pgbench schema missing in database {DOCKER_DB!r}. "
                f"Run:  pgbench -i -s 10 -h {DOCKER_HOST} -p {DOCKER_PORT} "
                f"-U {DOCKER_USER} {DOCKER_DB}"
            )
        cur.execute("SELECT extname FROM pg_extension")
        exts = {r[0] for r in cur.fetchall()}
        for req in ("hypopg", "index_advisor"):
            if req not in exts:
                raise RuntimeError(
                    f"extension {req!r} not enabled in {DOCKER_DB!r}. "
                    f"Run:  psql -h {DOCKER_HOST} -p {DOCKER_PORT} "
                    f"-U {DOCKER_USER} -d {DOCKER_DB} "
                    f"-c 'CREATE EXTENSION IF NOT EXISTS {req};'"
                )
        cur.execute("SELECT count(*) FROM pgbench_accounts")
        n_acc = cur.fetchone()[0]
        log.info("prereqs OK — pgbench_accounts=%d rows, exts=%s",
                 n_acc, sorted(exts & {"hypopg", "index_advisor"}))


# ─── Index reset: keep only the indexes pgbench creates at init time ──────────
# pgbench -i builds PRIMARY KEY on (aid), (bid), (tid) and no other indexes on
# pgbench_history.  We mirror that minimal baseline so the advisor has real
# room to recommend new filter/range indexes.
PGBENCH_BASE_INDEXES: set[str] = {
    "pgbench_accounts_pkey",
    "pgbench_branches_pkey",
    "pgbench_tellers_pkey",
}


def reset_indexes(conn) -> None:
    """Drop every non-PK index, leaving the pgbench initial state intact."""
    with conn.cursor() as cur:
        cur.execute("""
            SELECT i.indexname
            FROM pg_indexes i
            WHERE i.schemaname = 'public'
              AND NOT EXISTS (
                SELECT 1 FROM information_schema.table_constraints tc
                WHERE tc.constraint_name = i.indexname
                  AND tc.constraint_type IN ('PRIMARY KEY', 'UNIQUE')
              )
        """)
        candidates = [r[0] for r in cur.fetchall()]
    dropped = 0
    for idx in candidates:
        if idx in PGBENCH_BASE_INDEXES:
            continue
        try:
            with conn.cursor() as cur:
                cur.execute(f'DROP INDEX IF EXISTS "{idx}"')
            dropped += 1
        except Exception as e:
            log.debug("could not drop %s: %s", idx, e)
    log.debug("reset_indexes: dropped %d non-base indexes", dropped)


# =============================================================================
# 3. Query execution + timing
# =============================================================================
def execute_window(conn, items: list[tuple[str, str, str, str]]) -> dict:
    """Execute one window's queries.  Returns per-query latency + errors."""
    latencies_ms: list[float] = []
    errors = 0
    with conn.cursor() as cur:
        for _qid, _phase, _qtype, sql in items:
            t0 = time.perf_counter()
            try:
                cur.execute(sql)
                if cur.description is not None:
                    cur.fetchall()
            except psycopg2.errors.QueryCanceled:
                errors += 1
            except Exception as e:
                errors += 1
                log.debug("query error (%s): %.80s", type(e).__name__, sql)
            latencies_ms.append((time.perf_counter() - t0) * 1000.0)
    return {"latencies_ms": latencies_ms, "errors": errors}


def make_advisor(name: str):
    if name == "dexter":
        return DexterAdvisor.from_env("docker")
    if name == "supabase":
        return SupabaseIndexAdvisor()
    raise ValueError(f"unknown advisor: {name!r}")


def call_advisor(advisor, conn, recent_sqls: list[str]) -> dict:
    """Invoke advisor → apply_recommendations; return timing + counts."""
    t0 = time.perf_counter()
    if isinstance(advisor, SupabaseIndexAdvisor):
        ddl = advisor.recommend(recent_sqls, conn=conn)
    else:
        ddl = advisor.recommend(recent_sqls)
    t_reco = time.perf_counter() - t0

    t1 = time.perf_counter()
    stats = apply_recommendations(conn, ddl, drop_existing=True,
                                  preserve=PGBENCH_BASE_INDEXES,
                                  max_create=5)
    t_apply = time.perf_counter() - t1

    return {
        "recommend_s": t_reco,
        "apply_s":     t_apply,
        "wall_s":      t_reco + t_apply,
        "n_ddl":       len(ddl),
        "dropped":     stats["dropped"],
        "created":     stats["created"],
    }


# =============================================================================
# 4. Gating policies
# =============================================================================
def run_condition(
    conn,
    advisor,
    policy: str,                     # baseline | always_on | hsm_gated | periodic
    trace: list[tuple[str, str, str, str]],
    theta: float = THETA,
    periodic_k: int = PERIODIC_K,
) -> dict:
    """
    Run a full 120-query trace under the given gating policy, returning a
    rich dict with per-window breakdown.
    """
    windows = split_windows(trace, WINDOW_SIZE)
    n_windows = len(windows)

    per_window: list[dict] = []
    all_latencies: list[float] = []
    total_query_s  = 0.0
    total_advisor_s = 0.0
    total_errors   = 0
    advisor_calls  = 0
    hit_transitions = 0

    prev_sqls: list[str] | None = None
    prev_win  = None
    wall0 = time.perf_counter()

    for w_idx, win in enumerate(windows):
        sqls  = [it[3] for it in win]
        phase = Counter(it[1] for it in win).most_common(1)[0][0]

        # ── 1. Execute the window's queries ────────────────────────────────
        t0 = time.perf_counter()
        r = execute_window(conn, win)
        dt_query = time.perf_counter() - t0
        total_query_s += dt_query
        total_errors  += r["errors"]
        all_latencies.extend(r["latencies_ms"])

        # ── 2. Decide whether to call the advisor ───────────────────────────
        trigger = False
        hsm_score = 1.0
        if policy == "baseline" or advisor is None:
            trigger = False
        elif policy == "always_on":
            trigger = True
        elif policy == "periodic":
            trigger = (w_idx % periodic_k == 0 and w_idx > 0)
            # We skip w_idx==0 because the workload has not started drifting
            # yet and always_on already covers the "first-window" case.
        elif policy == "hsm_gated":
            curr_win = build_window(sqls, window_id=w_idx)
            if prev_win is None:
                trigger = False                                 # Fix 6 parity
            else:
                trigger, hsm_score, _dims = should_trigger_advisor(
                    prev_win, curr_win, theta)
            prev_win = curr_win
        else:
            raise ValueError(f"unknown policy: {policy!r}")

        # ── 3. Call the advisor, if triggered ──────────────────────────────
        adv_stats = None
        if trigger and advisor is not None:
            recent = (prev_sqls or []) + sqls         # ≤ 2 windows of context
            adv_stats = call_advisor(advisor, conn, recent)
            total_advisor_s += adv_stats["wall_s"]
            advisor_calls   += 1
            if w_idx in TRUE_TRANSITION_WINDOWS:
                hit_transitions += 1

        per_window.append({
            "window_idx":     w_idx,
            "phase":          phase,
            "hsm_score":      float(hsm_score),
            "triggered":      int(bool(trigger)),
            "is_true_trans":  int(w_idx in TRUE_TRANSITION_WINDOWS),
            "query_time_s":   dt_query,
            "advisor_wall_s": adv_stats["wall_s"] if adv_stats else 0.0,
            "n_ddl":          adv_stats["n_ddl"]  if adv_stats else 0,
            "dropped":        adv_stats["dropped"] if adv_stats else 0,
            "created":        adv_stats["created"] if adv_stats else 0,
            "errors":         r["errors"],
            "latencies_ms":   list(r["latencies_ms"]),
        })
        prev_sqls = sqls

        log.info(
            "    w%02d %-10s  q=%6.2fs  adv=%5.2fs  hsm=%.3f  %s",
            w_idx, phase, dt_query,
            (adv_stats["wall_s"] if adv_stats else 0.0),
            hsm_score,
            "TRIG" if trigger else "skip",
        )

    wall = time.perf_counter() - wall0

    # ── Aggregate latency stats (quantiles + skip/trigger split) ─────────────
    def _q(xs, p):
        if not xs:
            return 0.0
        s = sorted(xs)
        k = min(int(round(p * (len(s) - 1))), len(s) - 1)
        return s[k]

    lat_skip    = [l for w in per_window if not w["triggered"] for l in w["latencies_ms"]]
    lat_trigger = [l for w in per_window if     w["triggered"] for l in w["latencies_ms"]]

    n_ok = len(all_latencies) - total_errors
    qps = n_ok / total_query_s if total_query_s > 0 else 0.0

    return {
        # headline
        "wall_time_s":         wall,
        "total_query_time_s":  total_query_s,
        "total_advisor_time_s": total_advisor_s,
        "num_queries":         len(all_latencies),
        "errors":              total_errors,
        "qps":                 qps,
        "wall_qps":            n_ok / wall if wall > 0 else 0.0,
        # advisor
        "advisor_calls":       advisor_calls,
        "trigger_rate":        advisor_calls / n_windows if n_windows else 0.0,
        "trigger_precision":   (hit_transitions / advisor_calls) if advisor_calls else 0.0,
        "transitions_hit":     hit_transitions,
        "transitions_total":   len(TRUE_TRANSITION_WINDOWS),
        # latency
        "latency_p50_ms":      _q(all_latencies, 0.50),
        "latency_p95_ms":      _q(all_latencies, 0.95),
        "latency_p99_ms":      _q(all_latencies, 0.99),
        "latency_mean_skip_ms":    statistics.mean(lat_skip)    if lat_skip    else 0.0,
        "latency_mean_trigger_ms": statistics.mean(lat_trigger) if lat_trigger else 0.0,
        # detail
        "per_window":          per_window,
    }


# =============================================================================
# 5. Persistence
# =============================================================================
RAW_FIELDS = [
    "timestamp", "block", "order_in_block", "policy", "advisor",
    "wall_time_s", "total_query_time_s", "total_advisor_time_s",
    "num_queries", "errors", "qps", "wall_qps",
    "advisor_calls", "trigger_rate", "trigger_precision",
    "transitions_hit", "transitions_total",
    "latency_p50_ms", "latency_p95_ms", "latency_p99_ms",
    "latency_mean_skip_ms", "latency_mean_trigger_ms",
]


def save_raw(block: int, order: int, policy: str, advisor_name: str, res: dict,
             raw_file: Path) -> None:
    new = not raw_file.exists()
    with raw_file.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=RAW_FIELDS)
        if new:
            w.writeheader()
        w.writerow({
            "timestamp":      datetime.now().isoformat(timespec="seconds"),
            "block":          block + 1,
            "order_in_block": order + 1,
            "policy":         policy,
            "advisor":        advisor_name,
            **{k: round(res[k], 6) if isinstance(res[k], float) else res[k]
               for k in RAW_FIELDS if k in res},
        })


def save_detail(block: int, policy: str, advisor_name: str, res: dict,
                detail_dir: Path) -> None:
    detail_dir.mkdir(parents=True, exist_ok=True)
    tag = f"block{block + 1:02d}_{policy}_{advisor_name}"
    out = detail_dir / f"{tag}.json"
    payload = {k: v for k, v in res.items() if k != "per_window"}
    payload["per_window"] = [
        {**w, "latencies_ms": [round(x, 3) for x in w["latencies_ms"]]}
        for w in res["per_window"]
    ]
    with out.open("w") as f:
        json.dump(payload, f, indent=2, default=float)


# =============================================================================
# 6. Main loop
# =============================================================================
def build_condition_list(policies: list[str], advisors: list[str]) \
        -> list[tuple[str, str | None]]:
    """(policy, advisor_name) pairs.  baseline ignores advisor_name."""
    out: list[tuple[str, str | None]] = [("baseline", None)]
    for p in policies:
        for a in advisors:
            out.append((p, a))
    return out


def run_block(block: int, pool: list, raw_file: Path, detail_dir: Path,
              policies: list[str], advisors: list[str]) -> None:
    trace = build_trace(pool, seed=block)
    conds = build_condition_list(policies, advisors)

    # Randomise condition order within the block (reproducible per block).
    rng = random.Random(block ^ 0xC0FFEE)
    rng.shuffle(conds)

    log.info("\n── Block %d/%d  order: %s",
             block + 1, N_BLOCKS,
             " → ".join(f"{p}/{a or '—'}" for p, a in conds))

    for order, (policy, advisor_name) in enumerate(conds):
        log.info("  [%d/%d] %s × %s", order + 1, len(conds),
                 policy, advisor_name or "—")
        conn = get_connection()
        try:
            reset_indexes(conn)
            advisor = make_advisor(advisor_name) if advisor_name else None
            res = run_condition(conn, advisor, policy, trace)
        finally:
            conn.close()
        save_raw(block, order, policy, advisor_name or "none", res, raw_file)
        save_detail(block, policy, advisor_name or "none", res, detail_dir)
        log.info(
            "    → wall=%.2fs  qps=%.2f  calls=%d  trig_rate=%.2f  "
            "precision=%.2f  p95=%.1fms",
            res["wall_time_s"], res["qps"], res["advisor_calls"],
            res["trigger_rate"], res["trigger_precision"],
            res["latency_p95_ms"],
        )
        time.sleep(SETTLE_SEC)


def main() -> int:
    global N_BLOCKS
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--blocks",   type=int, default=N_BLOCKS,
                    help=f"Number of blocks (default {N_BLOCKS}).")
    ap.add_argument("--policy",   choices=POLICIES, nargs="+", default=POLICIES,
                    help="Subset of gating policies (default all).")
    ap.add_argument("--advisor",  choices=ADVISORS, nargs="+", default=ADVISORS,
                    help="Subset of advisors (default all).")
    ap.add_argument("--smoke",    action="store_true",
                    help="Run 1 block × 1 advisor × {baseline, always_on, hsm_gated} "
                         "as a wiring test.")
    ap.add_argument("--tag",      default="",
                    help="Optional suffix appended to the raw CSV name.")
    args = ap.parse_args()

    if args.smoke:
        args.blocks   = 1
        args.advisor  = ["dexter"]
        args.policy   = ["always_on", "hsm_gated"]     # + baseline, always added

    N_BLOCKS = args.blocks

    tag_sfx = f"_{args.tag}" if args.tag else ""
    raw_file   = RESULTS_DIR / f"oltp_tier2_raw{tag_sfx}.csv"
    detail_dir = RESULTS_DIR / f"detail{tag_sfx}"

    log.info("=" * 72)
    log.info("  HSM Tier-2 end-to-end OLTP runner (pgbench TPC-B, scale 10)")
    log.info("=" * 72)
    log.info("  DB          : %s on %s:%d as %s",
             DOCKER_DB, DOCKER_HOST, DOCKER_PORT, DOCKER_USER)
    log.info("  Blocks      : %d", N_BLOCKS)
    log.info("  Policies    : %s", args.policy)
    log.info("  Advisors    : %s", args.advisor)
    log.info("  Window W    : %d  →  %d windows per trace",
             WINDOW_SIZE, (QUERIES_PER_PHASE * len(PHASE_ORDER)) // WINDOW_SIZE)
    log.info("  θ (HSM)     : %.3f", THETA)
    log.info("  K (periodic): %d", PERIODIC_K)
    log.info("  Transitions : %s", sorted(TRUE_TRANSITION_WINDOWS))
    log.info("  Raw CSV     : %s", raw_file)
    log.info("  Details     : %s/", detail_dir)

    # One-off prerequisites check + schema sanity.
    conn = get_connection()
    try:
        check_prerequisites(conn)
    finally:
        conn.close()

    pool = build_query_pool()
    phase_counts = Counter(p for _, p, _, _ in pool)
    log.info("  Query pool  : %s", dict(phase_counts))

    for block in range(N_BLOCKS):
        run_block(block, pool, raw_file, detail_dir,
                  policies=args.policy, advisors=args.advisor)

    log.info("\nDone.  Raw results → %s", raw_file)
    log.info("        Per-window detail → %s/", detail_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
