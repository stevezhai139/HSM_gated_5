"""
advisor_wrappers.py — Phase 1, Tier 2 end-to-end advisor adapters
==================================================================

Provides two advisors that replace the hardcoded TPC-H column-map stub
(`run_index_advisor` in experiment_runner.py) with real production-grade
advisors:

    DexterAdvisor         — Ankane/pgdexter (log-driven HypoPG what-if)
    SupabaseIndexAdvisor  — Supabase index_advisor (SQL-driven HypoPG)

Both implement the common interface:

    class Advisor:
        name: str
        def recommend(self, queries: list[str]) -> list[str]:
            '''Return DDL statements (CREATE INDEX …) recommended for queries.'''

    def apply_recommendations(conn, ddl: list[str],
                              *, drop_existing: bool = True,
                              preserve: set[str] = FK_BASE_INDEXES) -> dict:
        '''Apply DDL with the same DROP-before-CREATE discipline used in
        experiment_runner.run_index_advisor (Fix 7).'''

Design notes
────────────
•  **Equivalence with stub.** Both advisors return a list of DDL strings.
   The caller (experiment runner) handles DROP/CREATE/timing exactly as in
   the stub path, so wall-clock accounting is consistent across advisors.
•  **Dexter** works by parsing query logs, so we pipe recent SQL via stdin
   to `dexter --input-format=sql`. It prints recommendations to stdout.
•  **Supabase index_advisor** is an in-engine SQL extension:
   `SELECT * FROM index_advisor('<sql>')` per query, aggregated.
•  **Timing.** Advisor wall time is measured in the runner wrapper, not here.
   `recommend()` is synchronous and blocks.

This module is import-safe: it does NOT touch the database at import time.
"""

from __future__ import annotations

import logging
import os
import re
import shutil
import subprocess
import time
from dataclasses import dataclass, field
from typing import Iterable

log = logging.getLogger("hsm.tier2.advisors")

# ─── Shared FK/join index preservation list (mirrors experiment_runner) ────────
FK_BASE_INDEXES: set[str] = {
    "idx_lineitem_orderkey", "idx_lineitem_partkey", "idx_lineitem_suppkey",
    "idx_orders_custkey",
    "idx_partsupp_partkey", "idx_partsupp_suppkey",
    "idx_customer_nationkey", "idx_supplier_nationkey",
    "idx_nation_regionkey",
}


# =============================================================================
# 1. Dexter (Ankane/pgdexter)
# =============================================================================
@dataclass
class DexterAdvisor:
    """
    Thin wrapper around the `dexter` CLI.

    Dexter reads queries from a log file or stdin and uses HypoPG to simulate
    each candidate index. Output is parsed to produce CREATE INDEX DDL.

    Requires:
      * `dexter` on PATH (installed via `gem install pgdexter`)
      * HypoPG extension enabled in the target database

    Connection defaults are read from the same env vars the rest of HSM_gated
    uses. Tier-2 OLTP + Burst experiments run against the Docker instance, so
    the defaults match `.env.example`'s `HSM_DOCKER_*` block (port 5433).
    Pass `DexterAdvisor.from_env("native")` to switch to `HSM_DB_*` (port 5432),
    or construct with explicit fields to override.

    Reference: https://github.com/ankane/dexter
    """
    name: str = "dexter"
    host: str = field(default_factory=lambda: os.environ.get("HSM_DOCKER_HOST", "localhost"))
    port: int = field(default_factory=lambda: int(os.environ.get("HSM_DOCKER_PORT", "5433")))
    dbname: str = field(default_factory=lambda: os.environ.get("HSM_DOCKER_DB", "postgres"))
    user: str = field(default_factory=lambda: os.environ.get("HSM_DOCKER_USER", "postgres"))
    password: str = field(default_factory=lambda: os.environ.get("HSM_DOCKER_PASSWORD", "postgres"))
    min_cost_savings_pct: int = 50        # Dexter's default threshold
    timeout_s: float = 30.0

    @classmethod
    def from_env(cls, mode: str = "docker", **overrides) -> "DexterAdvisor":
        """Build from `.env` vars; mode ∈ {'docker','native'}."""
        mode = mode.lower()
        if mode == "docker":
            base = dict(
                host=os.environ.get("HSM_DOCKER_HOST", "localhost"),
                port=int(os.environ.get("HSM_DOCKER_PORT", "5433")),
                dbname=os.environ.get("HSM_DOCKER_DB", "postgres"),
                user=os.environ.get("HSM_DOCKER_USER", "postgres"),
                password=os.environ.get("HSM_DOCKER_PASSWORD", "postgres"),
            )
        elif mode == "native":
            base = dict(
                host=os.environ.get("HSM_DB_HOST", "localhost"),
                port=int(os.environ.get("HSM_DB_PORT", "5432")),
                dbname=os.environ.get("HSM_DB_NAME", "tpch"),
                user=os.environ.get("HSM_DB_USER", "postgres"),
                password=os.environ.get("HSM_DB_PASSWORD", ""),
            )
        else:
            raise ValueError(f"mode must be 'docker' or 'native'; got {mode!r}")
        base.update(overrides)
        return cls(**base)

    # Pattern emitted by dexter for each recommendation
    _idx_re = re.compile(
        r"CREATE\s+INDEX\s+CONCURRENTLY\s+ON\s+[\"\w.]+\s*\([^)]*\)",
        re.IGNORECASE,
    )

    def _verify_binary(self) -> str:
        path = shutil.which("dexter")
        if not path:
            raise RuntimeError(
                "dexter CLI not found on PATH. "
                "Install with: sudo gem install pgdexter"
            )
        return path

    def _connection_args(self) -> list[str]:
        args = ["--host", self.host, "--port", str(self.port),
                "--dbname", self.dbname, "--username", self.user]
        return args

    def recommend(self, queries: Iterable[str]) -> list[str]:
        """Return a list of CREATE INDEX DDL strings recommended by Dexter."""
        queries = [q.strip().rstrip(";") for q in queries if q and q.strip()]
        if not queries:
            return []

        dexter_bin = self._verify_binary()
        stdin_payload = "\n".join(q + ";" for q in queries)

        env = os.environ.copy()
        if self.password:
            env["PGPASSWORD"] = self.password

        # Dexter 0.6.x requires an explicit source flag (--stdin, --pg-stat-statements,
        # --pg-stat-activity, or a path).  We pipe queries via stdin from Python,
        # so --stdin is the right choice.  Older 0.5.x used --input-format sql;
        # that flag was removed in 0.6.0.
        cmd = [
            dexter_bin, *self._connection_args(),
            "--stdin",
            "--min-cost-savings-pct", str(self.min_cost_savings_pct),
        ]
        log.debug("dexter cmd: %s", " ".join(cmd))

        try:
            proc = subprocess.run(
                cmd,
                input=stdin_payload,
                capture_output=True,
                text=True,
                timeout=self.timeout_s,
                env=env,
                check=False,
            )
        except subprocess.TimeoutExpired:
            log.warning("dexter timed out after %.1fs — returning empty recs", self.timeout_s)
            return []

        if proc.returncode != 0:
            log.warning("dexter exited %d; stderr=%s", proc.returncode, proc.stderr.strip())
            # Continue — partial stdout may still contain recs.

        ddl = [m.group(0) for m in self._idx_re.finditer(proc.stdout)]
        # Drop CONCURRENTLY because the runner wraps the call in a transaction.
        ddl = [re.sub(r"\s+CONCURRENTLY\b", "", d, flags=re.IGNORECASE) for d in ddl]
        log.info("dexter recommended %d index(es)", len(ddl))
        return ddl


# =============================================================================
# 2. Supabase index_advisor  (in-engine PostgreSQL extension)
# =============================================================================
@dataclass
class SupabaseIndexAdvisor:
    """
    Wrapper around the Supabase `index_advisor` PostgreSQL extension.

    Usage: `SELECT index_statements FROM index_advisor($sql);` returns a
    text array of CREATE INDEX statements for a single query.  We aggregate
    across the window's query set and de-duplicate.

    Requires:
      * index_advisor and hypopg extensions enabled in the target DB

    Reference: https://github.com/supabase/index_advisor
    """
    name: str = "supabase_index_advisor"
    per_query_timeout_ms: int = 5000

    def recommend(self, queries: Iterable[str], *, conn=None) -> list[str]:
        """
        Return de-duplicated CREATE INDEX DDL.

        `conn` is a psycopg2 connection; required.  We accept it as a kwarg
        so callers can reuse the experiment connection.
        """
        if conn is None:
            raise ValueError("SupabaseIndexAdvisor.recommend requires conn=<psycopg2.connection>")

        queries = [q.strip().rstrip(";") for q in queries if q and q.strip()]
        if not queries:
            return []

        ddl: list[str] = []
        seen: set[str] = set()

        with conn.cursor() as cur:
            cur.execute(f"SET LOCAL statement_timeout = {int(self.per_query_timeout_ms)}")
            for q in queries:
                try:
                    cur.execute("SELECT index_statements FROM index_advisor(%s)", (q,))
                    row = cur.fetchone()
                except Exception as e:
                    conn.rollback()
                    log.debug("index_advisor failed on one query (%s): %.80s",
                              type(e).__name__, q)
                    continue
                if not row or not row[0]:
                    continue
                for stmt in row[0]:                # text[] of DDL
                    key = re.sub(r"\s+", " ", stmt).strip().lower()
                    if key not in seen:
                        seen.add(key)
                        ddl.append(stmt.strip())
        conn.commit()
        log.info("supabase_index_advisor recommended %d unique index(es)", len(ddl))
        return ddl


# =============================================================================
# 3. Apply recommendations — shared DROP/CREATE path
# =============================================================================
def apply_recommendations(
    conn,
    ddl: list[str],
    *,
    drop_existing: bool = True,
    preserve: set[str] = FK_BASE_INDEXES,
    max_create: int = 5,
) -> dict:
    """
    Apply DDL recommendations with the same discipline as
    experiment_runner.run_index_advisor (Fix 7: DROP-before-CREATE to prevent
    phase contamination).

    Returns a dict:
        { 'dropped': int, 'created': int, 'wall_time_s': float }
    """
    t0 = time.perf_counter()
    dropped = 0
    created = 0

    if drop_existing:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT i.indexname
                FROM pg_indexes i
                WHERE i.schemaname = 'public'
                  AND NOT EXISTS (
                    SELECT 1 FROM information_schema.table_constraints tc
                    WHERE tc.constraint_name = i.indexname
                      AND tc.constraint_type IN ('PRIMARY KEY', 'UNIQUE')
                  )
                """
            )
            all_idx = [r[0] for r in cur.fetchall()]
        advisor_idx = [i for i in all_idx if i not in preserve]
        with conn.cursor() as cur:
            for idx in advisor_idx:
                try:
                    cur.execute(f'DROP INDEX IF EXISTS "{idx}"')
                    dropped += 1
                except Exception:
                    conn.rollback()
        conn.commit()

    with conn.cursor() as cur:
        for stmt in ddl[:max_create]:
            try:
                cur.execute(stmt)
                created += 1
            except Exception as e:
                conn.rollback()
                log.debug("CREATE INDEX failed: %s — %s", stmt[:80], e)
    conn.commit()

    return {
        "dropped":     dropped,
        "created":     created,
        "wall_time_s": time.perf_counter() - t0,
    }


# =============================================================================
# 4. Factory
# =============================================================================
def make_advisor(name: str):
    """Return an advisor instance by name: 'dexter' or 'supabase'."""
    name = name.lower()
    if name == "dexter":
        return DexterAdvisor()
    if name in ("supabase", "supabase_index_advisor", "index_advisor"):
        return SupabaseIndexAdvisor()
    raise ValueError(f"unknown advisor: {name!r}")
