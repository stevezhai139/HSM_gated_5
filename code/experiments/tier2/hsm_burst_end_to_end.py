"""
hsm_burst_end_to_end.py — Tier-2 end-to-end advisor study on a burst-v2-style
synthetic workload against pgbench tables.
==================================================================================

Companion runner to ``hsm_oltp_end_to_end.py`` — same factorial design against
real advisors (Dexter, Supabase index_advisor), but the workload exercises
rapid *burst* patterns in predicate columns rather than OLTP read/write mixes.

    ``experiment_runner.py``      — OLAP, TPC-H, stub advisor
    ``hsm_oltp_end_to_end.py``    — OLTP, pgbench TPC-B, real advisors
    ``hsm_burst_end_to_end.py``   — burst_v2 synthetic on pgbench, real advisors

DESIGN  (post-review; differs intentionally from OLTP)
──────
Workload
    3 phases × 35 queries = 105 queries, window size W=5 → 21 windows.
    Phase order: Steady_Point → Burst_Alt → Burst_Grp
    Phase transitions at windows {7, 14}  (design chosen so neither 7 nor 14
    is a multiple of 2, 3 or 4 — avoids periodic-K alignment luck).
    Phase 4 (Return_Steady) intentionally omitted: advisor_wrappers'
    apply_recommendations only CREATEs indexes, it cannot DROP stale indexes,
    so a return-to-steady phase would measure index persistence, not advisor
    quality.

    Phase 1  Steady_Point : 35 point reads on pgbench_accounts(aid)  [PK fast]
    Phase 2  Burst_Alt    : stable 3 PK + 2 AGG_BID per window × 7 windows
                            (21 point + 14 bid aggregates; fixed intra-window
                             order PK,AGG,PK,AGG,PK)
    Phase 3  Burst_Grp    : stable 3 PK + 2 AGG_ABAL per window × 7 windows
                            (21 point + 14 range aggregates on abalance; same
                             intra-window order)

    Composition rationale (design review v2)
        Phase 2 and Phase 3 are intentionally symmetric in *size* (3/2 mix)
        but differ in *aggregate template* (bid filter vs abalance range).
        This isolates the HSM signal to pure template-set difference at
        phase boundaries — Jaccard 1/2 at P1→P2, Jaccard 1/3 at P2→P3 —
        and keeps intra-phase Jaccard at 1.0 so HSM only fires at true
        transitions.  An earlier "strict period-2 at query level" design
        caused window composition to alternate 3P2A ↔ 2P3A, producing
        bigger intra-phase drift than the phase boundary and making HSM
        fire inside the phase instead of at the boundary (unfair to HSM).

Conditions  (11 runs per block when K-sweep is enabled)
    baseline                        — no advisor
    always_on       × {dexter, supabase}
    hsm_gated       × {dexter, supabase}         — θ = 0.75
    periodic_K2     × {dexter, supabase}
    periodic_K3     × {dexter, supabase}
    periodic_K4     × {dexter, supabase}

Block design
    10 blocks, randomised condition order per block (seed=block).
    Each block regenerates a trace with seed=block (phase-internal shuffle).

Per-run metrics recorded
    wall_time_s, total_query_time_s, total_advisor_time_s
    num_queries, errors, qps, wall_qps
    advisor_calls, trigger_rate,
    trigger_precision_strict (triggered window ∈ TRUE_TRANSITIONS),
    trigger_precision_relaxed (triggered window ∈ TRUE_TRANSITIONS ± 1),
    per-query latency (ms) → p50, p95, p99
    per-window: HSM score, triggered flag, advisor_wall_s, phase
    latency_on_skip_ms, latency_on_trigger_ms

Database
    Docker PostgreSQL 16 (hsm-postgres:16-hypopg) on port 5433.
    Reuses database ``oltp`` (pgbench scale 10) from Phase 2a.
    Extensions required: hypopg, index_advisor.

Usage
    source .env
    python hsm_burst_end_to_end.py --smoke          # 1 block, wiring test
    python hsm_burst_end_to_end.py                  # full 10 × 11 = 110 runs
    python hsm_burst_end_to_end.py --blocks 3       # shorter run
    python hsm_burst_end_to_end.py --advisor dexter # restrict to one advisor
    python hsm_burst_end_to_end.py --no-k-sweep     # only K=3, 7 conditions
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import random
import statistics
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

# psycopg2 is optional: the module is imported by CPU-only smoke scripts
# (e.g. b2_kernel_ablation, b4_inphase_perturbation) that only use
# build_query_pool / build_trace and never open a DB connection. We fall
# back to None here and let the DB-touching functions raise at call time.
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

QUERIES_PER_PHASE = 35
PHASE_ORDER       = ["Steady_Point", "Burst_Alt", "Burst_Grp"]
WINDOW_SIZE       = 5        # 105 / 5 = 21 windows
THETA             = DEFAULT_THETA    # 0.75
PERIODIC_K_DEFAULT = 3
PERIODIC_K_SWEEP  = [2, 3, 4]
N_BLOCKS          = 10
STATEMENT_TIMEOUT_MS = 30_000
SETTLE_SEC        = 2

# Range of aid values to sample from (matches pgbench scale 10 = 1M accounts)
N_ACCOUNTS = 1_000_000
N_BRANCHES = 10               # one branch per 100 000 accounts
ABALANCE_RANGES = [
    (-1_000, 1_000),
    (1_000, 5_000),
    (5_000, 10_000),
    (-500, 500),
    (0, 2_000),
    (2_000, 8_000),
    (-2_000, 0),
    (500, 1_500),
    (1_000, 10_000),
    (0, 5_000),
]

ADVISORS    = ["dexter", "supabase"]
# We expose logical "policy" labels — K-sweep variants each map to
# policy=periodic with a distinct K captured in the condition name.
POLICIES_BASE = ["always_on", "hsm_gated"]

# ─── Logging ───────────────────────────────────────────────────────────────────
RESULTS_DIR = _REPO / "results" / "burst_end_to_end"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(RESULTS_DIR / "burst_end_to_end.log"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger("tier2.burst")
logging.getLogger("hsm.tier2.advisors").setLevel(logging.WARNING)


# =============================================================================
# 1. Query pool — burst_v2 style synthetic on pgbench_accounts
# =============================================================================
def _rand_aids(rng: random.Random, n: int) -> list[int]:
    return [rng.randint(1, N_ACCOUNTS) for _ in range(n)]


def _rand_bids(rng: random.Random, n: int) -> list[int]:
    return [rng.randint(1, N_BRANCHES) for _ in range(n)]


def build_query_pool(seed: int = 0) -> list[tuple[str, str, str, str]]:
    """
    Return a list of (qid, phase, qtype, sql) of length 3 × QUERIES_PER_PHASE.
    The pool is built deterministically from ``seed`` — each block gets a fresh
    pool but with the same phase structure.
    """
    rng = random.Random(seed)
    pool: list[tuple[str, str, str, str]] = []
    counter = [0]

    def add(phase: str, qtype: str, sql: str) -> None:
        counter[0] += 1
        pool.append((f"{phase[:2]}{counter[0]:03d}", phase, qtype, sql))

    # ── Phase 1: Steady_Point — 35 point reads on aid (PK) ──────────────────
    for aid in _rand_aids(rng, QUERIES_PER_PHASE):
        add("Steady_Point", "SELECT_PK",
            f"SELECT abalance FROM pgbench_accounts WHERE aid = {aid}")

    # ── Phase 2: Burst_Alt — stable per-window mix of PK + AGG_BID ───────────
    # Every window contains exactly 3 PK + 2 AGG_BID queries in a fixed
    # intra-window order (PK, AGG, PK, AGG, PK).  This keeps intra-phase
    # composition constant so HSM sees template drift ONLY at phase
    # boundaries, not between windows that share a phase.  Design review
    # v2 fix: the previous "strict period-2 at query level" caused window
    # composition to alternate 3P2A ↔ 2P3A, producing bigger intra-phase
    # drift than the actual phase boundary and making HSM fire inside the
    # phase instead of at the boundary.
    n_windows_p2 = QUERIES_PER_PHASE // WINDOW_SIZE   # 7
    for _ in range(n_windows_p2):
        aids_w = _rand_aids(rng, 3)
        bids_w = _rand_bids(rng, 2)
        slots = [
            ("PK",      aids_w[0]),
            ("AGG_BID", bids_w[0]),
            ("PK",      aids_w[1]),
            ("AGG_BID", bids_w[1]),
            ("PK",      aids_w[2]),
        ]
        for qtype, val in slots:
            if qtype == "PK":
                add("Burst_Alt", "SELECT_PK",
                    f"SELECT abalance FROM pgbench_accounts "
                    f"WHERE aid = {val}")
            else:
                add("Burst_Alt", "SELECT_AGG_BID",
                    f"SELECT COUNT(*), SUM(abalance) FROM pgbench_accounts "
                    f"WHERE bid = {val}")

    # ── Phase 3: Burst_Grp — stable per-window mix of PK + AGG_ABAL ──────────
    # Every window contains exactly 3 PK + 2 AGG_ABAL queries with the same
    # intra-window ordering as Phase 2.  Phase 2 and Phase 3 therefore have
    # IDENTICAL composition sizes (3/2) but DIFFERENT aggregate templates
    # (bid filter → abalance range).  This isolates the HSM signal to pure
    # template-set difference (Jaccard 1/3 at P2→P3 boundary), which is
    # precisely what the advisor needs to notice when the index-needy column
    # changes.  Range ratio = 2/5 = 40 % per phase (14 range / 35).
    n_windows_p3 = QUERIES_PER_PHASE // WINDOW_SIZE   # 7
    for _ in range(n_windows_p3):
        aids_w = _rand_aids(rng, 3)
        ranges_w = rng.choices(ABALANCE_RANGES, k=2)
        slots = [
            ("PK",       aids_w[0]),
            ("AGG_ABAL", ranges_w[0]),
            ("PK",       aids_w[1]),
            ("AGG_ABAL", ranges_w[1]),
            ("PK",       aids_w[2]),
        ]
        for qtype, val in slots:
            if qtype == "PK":
                add("Burst_Grp", "SELECT_PK",
                    f"SELECT abalance FROM pgbench_accounts "
                    f"WHERE aid = {val}")
            else:
                lo, hi = val
                add("Burst_Grp", "SELECT_AGG_ABAL",
                    f"SELECT COUNT(*), AVG(abalance) FROM pgbench_accounts "
                    f"WHERE abalance BETWEEN {lo} AND {hi}")

    return pool


def build_trace(pool: list[tuple[str, str, str, str]], seed: int) \
        -> list[tuple[str, str, str, str]]:
    """
    Preserve strict phase order — Phase 1 then Phase 2 then Phase 3 — so the
    transition windows {7, 14} are deterministic.  Within a phase the order
    is already fixed by build_query_pool (Burst_Alt relies on the period-2
    pattern, so we do NOT shuffle inside a phase).  The ``seed`` argument is
    kept for symmetry with the OLTP runner; it drives build_query_pool via
    the caller.
    """
    _ = seed  # already baked into the pool via build_query_pool(seed)
    trace: list[tuple[str, str, str, str]] = []
    for phase in PHASE_ORDER:
        members = [r for r in pool if r[1] == phase]
        trace.extend(members[:QUERIES_PER_PHASE])
    return trace


def split_windows(trace: list, w: int = WINDOW_SIZE) -> list[list]:
    return [trace[i:i + w] for i in range(0, len(trace), w) if len(trace[i:i + w]) == w]


# With 3 phases × 35 queries and W=5: phase 1 = windows 0-6, phase 2 = 7-13,
# phase 3 = 14-20.  Transitions land at the first window of each new phase.
TRUE_TRANSITION_WINDOWS = {7, 14}
RELAXED_DELTA = 1   # ±1 window counts as a relaxed-precision hit


def is_relaxed_hit(w_idx: int) -> bool:
    return any(abs(w_idx - t) <= RELAXED_DELTA for t in TRUE_TRANSITION_WINDOWS)


# =============================================================================
# 2. Database helpers  (identical to OLTP runner — shared schema)
# =============================================================================
def get_connection(dbname: str = DOCKER_DB, autocommit: bool = True):
    if not _HAS_PSYCOPG2:
        raise RuntimeError(
            "psycopg2 is required for DB-connected experiments. "
            "Install it with: pip install psycopg2-binary"
        )
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
    with conn.cursor() as cur:
        cur.execute("SELECT to_regclass('public.pgbench_accounts')::text")
        (accounts,) = cur.fetchone()
        if accounts is None:
            raise RuntimeError(
                f"pgbench schema missing in {DOCKER_DB!r}.  "
                f"Run pgbench -i first — see docs/Tier2_SETUP.md"
            )
        cur.execute("SELECT extname FROM pg_extension")
        exts = {r[0] for r in cur.fetchall()}
        for req in ("hypopg", "index_advisor"):
            if req not in exts:
                raise RuntimeError(
                    f"extension {req!r} not enabled in {DOCKER_DB!r}"
                )
        cur.execute("SELECT count(*) FROM pgbench_accounts")
        n_acc = cur.fetchone()[0]
        log.info("prereqs OK — pgbench_accounts=%d rows, exts=%s",
                 n_acc, sorted(exts & {"hypopg", "index_advisor"}))


PGBENCH_BASE_INDEXES: set[str] = {
    "pgbench_accounts_pkey",
    "pgbench_branches_pkey",
    "pgbench_tellers_pkey",
}


def reset_indexes(conn) -> None:
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
# 3. Query execution + advisor invocation
# =============================================================================
def execute_window(conn, items: list[tuple[str, str, str, str]]) -> dict:
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
# 4. Condition runner
# =============================================================================
def run_condition(
    conn,
    advisor,
    policy: str,                     # baseline | always_on | hsm_gated | periodic
    trace: list[tuple[str, str, str, str]],
    theta: float | None = None,
    periodic_k: int = PERIODIC_K_DEFAULT,
) -> dict:
    if theta is None:
        theta = THETA
    windows = split_windows(trace, WINDOW_SIZE)
    n_windows = len(windows)

    per_window: list[dict] = []
    all_latencies: list[float] = []
    total_query_s  = 0.0
    total_advisor_s = 0.0
    total_recommend_s = 0.0   # TCO-A: pure advisor "think" cost
    total_apply_s     = 0.0   # TCO-B: index DROP+CREATE cost
    total_created     = 0
    total_dropped     = 0
    total_errors   = 0
    advisor_calls  = 0
    hit_strict     = 0
    hit_relaxed    = 0

    prev_sqls: list[str] | None = None
    prev_win  = None
    wall0 = time.perf_counter()

    for w_idx, win in enumerate(windows):
        sqls  = [it[3] for it in win]
        phase = Counter(it[1] for it in win).most_common(1)[0][0]

        t0 = time.perf_counter()
        r = execute_window(conn, win)
        dt_query = time.perf_counter() - t0
        total_query_s += dt_query
        total_errors  += r["errors"]
        all_latencies.extend(r["latencies_ms"])

        trigger = False
        hsm_score = 1.0
        if policy == "baseline" or advisor is None:
            trigger = False
        elif policy == "always_on":
            trigger = True
        elif policy == "periodic":
            trigger = (w_idx % periodic_k == 0 and w_idx > 0)
        elif policy == "hsm_gated":
            curr_win = build_window(sqls, window_id=w_idx)
            if prev_win is None:
                trigger = False
            else:
                trigger, hsm_score, _dims = should_trigger_advisor(
                    prev_win, curr_win, theta)
            prev_win = curr_win
        else:
            raise ValueError(f"unknown policy: {policy!r}")

        adv_stats = None
        if trigger and advisor is not None:
            recent = (prev_sqls or []) + sqls
            adv_stats = call_advisor(advisor, conn, recent)
            total_advisor_s   += adv_stats["wall_s"]
            total_recommend_s += adv_stats["recommend_s"]
            total_apply_s     += adv_stats["apply_s"]
            total_created     += adv_stats["created"]
            total_dropped     += adv_stats["dropped"]
            advisor_calls     += 1
            if w_idx in TRUE_TRANSITION_WINDOWS:
                hit_strict += 1
            if is_relaxed_hit(w_idx):
                hit_relaxed += 1

        per_window.append({
            "window_idx":     w_idx,
            "phase":          phase,
            "hsm_score":      float(hsm_score),
            "triggered":      int(bool(trigger)),
            "is_true_trans":  int(w_idx in TRUE_TRANSITION_WINDOWS),
            "is_relaxed":     int(is_relaxed_hit(w_idx)),
            "query_time_s":   dt_query,
            "advisor_wall_s":  adv_stats["wall_s"]       if adv_stats else 0.0,
            "advisor_recommend_s": adv_stats["recommend_s"] if adv_stats else 0.0,
            "advisor_apply_s":     adv_stats["apply_s"]     if adv_stats else 0.0,
            "n_ddl":          adv_stats["n_ddl"]  if adv_stats else 0,
            "dropped":        adv_stats["dropped"] if adv_stats else 0,
            "created":        adv_stats["created"] if adv_stats else 0,
            "errors":         r["errors"],
            "latencies_ms":   list(r["latencies_ms"]),
        })
        prev_sqls = sqls

        log.info(
            "    w%02d %-14s  q=%6.2fs  adv=%5.2fs  hsm=%.3f  %s",
            w_idx, phase, dt_query,
            (adv_stats["wall_s"] if adv_stats else 0.0),
            hsm_score,
            "TRIG" if trigger else "skip",
        )

    wall = time.perf_counter() - wall0

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

    # TCO decomposition (paper §V.D, Theorem 3 empirical receipt):
    #   TCO = query_cost + advisor_cost + index_churn_cost
    # where wall_time_s captures the user-perceived total.  The three
    # component totals below allow Phase 2c to answer "did advisor calls
    # + index rebuilds pay for themselves in query savings?".
    return {
        "wall_time_s":         wall,
        "total_query_time_s":  total_query_s,
        "total_advisor_time_s": total_advisor_s,
        "total_recommend_time_s": total_recommend_s,
        "total_apply_time_s":     total_apply_s,
        "tco_time_s":          total_query_s + total_advisor_s,
        "indexes_created":     total_created,
        "indexes_dropped":     total_dropped,
        "num_queries":         len(all_latencies),
        "errors":              total_errors,
        "qps":                 qps,
        "wall_qps":            n_ok / wall if wall > 0 else 0.0,
        "advisor_calls":       advisor_calls,
        "trigger_rate":        advisor_calls / n_windows if n_windows else 0.0,
        "trigger_precision_strict":  (hit_strict  / advisor_calls) if advisor_calls else 0.0,
        "trigger_precision_relaxed": (hit_relaxed / advisor_calls) if advisor_calls else 0.0,
        "transitions_hit_strict":    hit_strict,
        "transitions_hit_relaxed":   hit_relaxed,
        "transitions_total":         len(TRUE_TRANSITION_WINDOWS),
        "latency_p50_ms":      _q(all_latencies, 0.50),
        "latency_p95_ms":      _q(all_latencies, 0.95),
        "latency_p99_ms":      _q(all_latencies, 0.99),
        "latency_mean_skip_ms":    statistics.mean(lat_skip)    if lat_skip    else 0.0,
        "latency_mean_trigger_ms": statistics.mean(lat_trigger) if lat_trigger else 0.0,
        "per_window":          per_window,
    }


# =============================================================================
# 5. Persistence
# =============================================================================
RAW_FIELDS = [
    "timestamp", "block", "order_in_block", "condition", "policy", "advisor",
    "periodic_k",
    "wall_time_s", "total_query_time_s", "total_advisor_time_s",
    # TCO decomposition (added 2026-04-15 for C2 economic analysis):
    #   total_recommend_time_s : advisor "think" cost (Dexter/Supabase API)
    #   total_apply_time_s     : index DROP+CREATE cost (real DDL time)
    #   tco_time_s             : query + advisor (composite cost proxy)
    #   indexes_{created,dropped} : DDL count per policy per block
    "total_recommend_time_s", "total_apply_time_s", "tco_time_s",
    "indexes_created", "indexes_dropped",
    "num_queries", "errors", "qps", "wall_qps",
    "advisor_calls", "trigger_rate",
    "trigger_precision_strict", "trigger_precision_relaxed",
    "transitions_hit_strict", "transitions_hit_relaxed", "transitions_total",
    "latency_p50_ms", "latency_p95_ms", "latency_p99_ms",
    "latency_mean_skip_ms", "latency_mean_trigger_ms",
]


def save_raw(block: int, order: int, condition: str, policy: str,
             advisor_name: str, periodic_k: int | None, res: dict,
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
            "condition":      condition,
            "policy":         policy,
            "advisor":        advisor_name,
            "periodic_k":     periodic_k if periodic_k is not None else "",
            **{k: round(res[k], 6) if isinstance(res[k], float) else res[k]
               for k in RAW_FIELDS if k in res},
        })


def save_detail(block: int, condition: str, res: dict, detail_dir: Path) -> None:
    detail_dir.mkdir(parents=True, exist_ok=True)
    tag = f"block{block + 1:02d}_{condition}"
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
def build_condition_list(
    policies: list[str],
    advisors: list[str],
    periodic_ks: list[int],
) -> list[tuple[str, str, str | None, int | None]]:
    """
    Return list of (condition_name, policy, advisor_name, periodic_k).
    ``condition_name`` is used for file naming + CSV dedup keys.
    """
    out: list[tuple[str, str, str | None, int | None]] = [("baseline", "baseline", None, None)]
    for p in policies:
        for a in advisors:
            out.append((f"{p}_{a}", p, a, None))
    for k in periodic_ks:
        for a in advisors:
            out.append((f"periodic_K{k}_{a}", "periodic", a, k))
    return out


def run_block(block: int, raw_file: Path, detail_dir: Path,
              policies: list[str], advisors: list[str],
              periodic_ks: list[int]) -> None:
    # Each block gets its own pool (different random aid/bid/range choices)
    pool = build_query_pool(seed=block)
    trace = build_trace(pool, seed=block)
    conds = build_condition_list(policies, advisors, periodic_ks)

    rng = random.Random(block ^ 0xC0FFEE)
    rng.shuffle(conds)

    log.info("\n── Block %d/%d  order: %s",
             block + 1, N_BLOCKS,
             " → ".join(c[0] for c in conds))

    for order, (cond_name, policy, advisor_name, pk) in enumerate(conds):
        log.info("  [%d/%d] %s", order + 1, len(conds), cond_name)
        conn = get_connection()
        try:
            reset_indexes(conn)
            advisor = make_advisor(advisor_name) if advisor_name else None
            k = pk if pk is not None else PERIODIC_K_DEFAULT
            res = run_condition(conn, advisor, policy, trace, periodic_k=k)
        finally:
            conn.close()
        save_raw(block, order, cond_name, policy,
                 advisor_name or "none", pk, res, raw_file)
        save_detail(block, cond_name, res, detail_dir)
        log.info(
            "    → wall=%.2fs  qps=%.2f  calls=%d  trig_rate=%.2f  "
            "P_strict=%.2f  P_relaxed=%.2f  p95=%.1fms",
            res["wall_time_s"], res["qps"], res["advisor_calls"],
            res["trigger_rate"],
            res["trigger_precision_strict"],
            res["trigger_precision_relaxed"],
            res["latency_p95_ms"],
        )
        time.sleep(SETTLE_SEC)


def main() -> int:
    global N_BLOCKS, QUERIES_PER_PHASE, WINDOW_SIZE, TRUE_TRANSITION_WINDOWS, THETA
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--blocks",   type=int, default=N_BLOCKS,
                    help=f"Number of blocks (default {N_BLOCKS}).")
    ap.add_argument("--policy",   choices=POLICIES_BASE, nargs="+",
                    default=POLICIES_BASE,
                    help="Subset of non-periodic policies (default all).")
    ap.add_argument("--advisor",  choices=ADVISORS, nargs="+", default=ADVISORS,
                    help="Subset of advisors (default all).")
    ap.add_argument("--no-k-sweep", action="store_true",
                    help="Disable K-sweep — only run periodic with K=3.")
    ap.add_argument("--smoke",    action="store_true",
                    help="Run 1 block × 1 advisor × {baseline, always_on, hsm_gated} "
                         "as a wiring test.")
    ap.add_argument("--tag",      default="",
                    help="Optional suffix appended to the raw CSV name.")
    ap.add_argument("--queries-per-phase", type=int, default=QUERIES_PER_PHASE,
                    help=f"Queries per phase (default {QUERIES_PER_PHASE}). "
                         "Large-Q variant: use 500 to cross Q_min and force HSM triggers.")
    ap.add_argument("--window-size", type=int, default=WINDOW_SIZE,
                    help=f"Window size in queries (default {WINDOW_SIZE}).")
    ap.add_argument("--theta", type=float, default=THETA,
                    help=f"HSM trigger threshold (default {THETA}). "
                         "Use optimal_theta(N, Q) from hsm_similarity for Theorem-3 calibration.")
    args = ap.parse_args()

    # Allow CLI override of phase length / window size / theta; recompute transitions.
    QUERIES_PER_PHASE = args.queries_per_phase
    WINDOW_SIZE       = args.window_size
    THETA             = args.theta
    _wpp              = QUERIES_PER_PHASE // WINDOW_SIZE
    TRUE_TRANSITION_WINDOWS = {_wpp, 2 * _wpp}

    if args.smoke:
        args.blocks   = 1
        args.advisor  = ["dexter"]
        args.policy   = ["always_on", "hsm_gated"]
        periodic_ks   = [3]
    else:
        periodic_ks = [PERIODIC_K_DEFAULT] if args.no_k_sweep else list(PERIODIC_K_SWEEP)

    N_BLOCKS = args.blocks

    # Formal output filenames: burst_<tag>_results.csv / detail_<tag>/
    # e.g. --tag small_q -> burst_small_q_results.csv, detail_small_q/
    raw_file   = (
        RESULTS_DIR / f"burst_{args.tag}_results.csv" if args.tag
        else RESULTS_DIR / "burst_results.csv"
    )
    detail_dir = (
        RESULTS_DIR / f"detail_{args.tag}" if args.tag
        else RESULTS_DIR / "detail"
    )

    log.info("=" * 72)
    log.info("  HSM Tier-2 end-to-end burst_v2 runner (pgbench backing store)")
    log.info("=" * 72)
    log.info("  DB          : %s on %s:%d as %s",
             DOCKER_DB, DOCKER_HOST, DOCKER_PORT, DOCKER_USER)
    log.info("  Blocks      : %d", N_BLOCKS)
    log.info("  Policies    : %s", args.policy)
    log.info("  Advisors    : %s", args.advisor)
    log.info("  Periodic K  : %s", periodic_ks)
    log.info("  Window W    : %d  →  %d windows per trace",
             WINDOW_SIZE, (QUERIES_PER_PHASE * len(PHASE_ORDER)) // WINDOW_SIZE)
    log.info("  θ (HSM)     : %.3f", THETA)
    log.info("  Transitions : strict=%s  relaxed=±%d window",
             sorted(TRUE_TRANSITION_WINDOWS), RELAXED_DELTA)
    log.info("  Raw CSV     : %s", raw_file)
    log.info("  Details     : %s/", detail_dir)

    conn = get_connection()
    try:
        check_prerequisites(conn)
    finally:
        conn.close()

    # Sanity-print the pool shape for block 0 so Phase counts are visible.
    pool0 = build_query_pool(seed=0)
    phase_counts = Counter(p for _, p, _, _ in pool0)
    log.info("  Query pool  : %s  (example, block 0)", dict(phase_counts))

    for block in range(N_BLOCKS):
        run_block(block, raw_file, detail_dir,
                  policies=args.policy, advisors=args.advisor,
                  periodic_ks=periodic_ks)

    log.info("\nDone.  Raw results → %s", raw_file)
    log.info("        Per-window detail → %s/", detail_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
