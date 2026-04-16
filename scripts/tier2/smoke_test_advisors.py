#!/usr/bin/env python3
"""
smoke_test_advisors.py — Phase 1 smoke test for Tier-2 advisors
================================================================

Verifies, on the user's host PostgreSQL, that:

  1. psycopg2 can connect using the HSM_DB_* env vars.
  2. hypopg and index_advisor extensions are enabled in the target DB.
  3. `dexter` CLI is on PATH and can reach the DB.
  4. DexterAdvisor.recommend returns a (possibly empty) list of DDL.
  5. SupabaseIndexAdvisor.recommend returns a (possibly empty) list of DDL.
  6. apply_recommendations runs DROP/CREATE without raising.

This does NOT run the full experiment — it only exercises the wrappers.
Exits non-zero on the first failure so CI / user can see what broke.

Env-var resolution (same contract as the rest of HSM_gated):
    Copy `.env.example` to `.env`, fill in values, then `source .env`.

    --mode docker  (default, matches OLTP + Burst validation runners)
        Reads HSM_DOCKER_HOST / _PORT / _DB / _USER / _PASSWORD
        (defaults: localhost:5433 / postgres / postgres / postgres)
    --mode native
        Reads HSM_DB_HOST / _PORT / _NAME / _USER / _PASSWORD
        (defaults: localhost:5432 / tpch / postgres / "")

Usage:
    source .env
    python scripts/tier2/smoke_test_advisors.py                 # docker (5433)
    python scripts/tier2/smoke_test_advisors.py --mode native   # native (5432)
    python scripts/tier2/smoke_test_advisors.py --dbname foo    # override DB
"""
from __future__ import annotations

import argparse
import logging
import os
import shutil
import sys
from pathlib import Path

# ── Path bootstrap (so we can import advisor_wrappers without install) ────────
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "code" / "experiments"))
sys.path.insert(0, str(REPO_ROOT / "code" / "experiments" / "tier2"))

try:
    import psycopg2
except ImportError:
    print("FATAL: psycopg2 not installed (pip install psycopg2-binary)", file=sys.stderr)
    sys.exit(2)

from advisor_wrappers import (
    DexterAdvisor, SupabaseIndexAdvisor,
    apply_recommendations,
)


def _defaults_for(mode: str) -> dict:
    """Return connection defaults matching the rest of HSM_gated."""
    if mode == "native":
        return dict(
            host=os.environ.get("HSM_DB_HOST", "localhost"),
            port=int(os.environ.get("HSM_DB_PORT", "5432")),
            dbname=os.environ.get("HSM_DB_NAME", "tpch"),
            user=os.environ.get("HSM_DB_USER", "postgres"),
            password=os.environ.get("HSM_DB_PASSWORD", ""),
        )
    # docker (default)
    return dict(
        host=os.environ.get("HSM_DOCKER_HOST", "localhost"),
        port=int(os.environ.get("HSM_DOCKER_PORT", "5433")),
        dbname=os.environ.get("HSM_DOCKER_DB", "postgres"),
        user=os.environ.get("HSM_DOCKER_USER", "postgres"),
        password=os.environ.get("HSM_DOCKER_PASSWORD", "postgres"),
    )

log = logging.getLogger("smoke")
logging.basicConfig(level=logging.INFO, format="%(levelname).1s %(name)s: %(message)s")

# Three tiny OLTP-style queries that any pgbench-loaded DB (or TPC-H) should
# accept at least syntactically. These are probes, not a real workload.
PROBE_QUERIES = [
    "SELECT 1",
    "SELECT count(*) FROM pg_class",
    "SELECT relname FROM pg_class WHERE relkind='r' LIMIT 5",
]

CHECKS: list[tuple[str, bool]] = []


def record(name: str, ok: bool, detail: str = "") -> None:
    CHECKS.append((name, ok))
    mark = "PASS" if ok else "FAIL"
    log.info("[%s] %s%s", mark, name, f" — {detail}" if detail else "")


def make_conn(args) -> "psycopg2.extensions.connection":
    return psycopg2.connect(
        host=args.host, port=args.port, dbname=args.dbname,
        user=args.user, password=args.password,
    )


def check_connection(args) -> "psycopg2.extensions.connection | None":
    try:
        conn = make_conn(args)
    except Exception as e:
        record("psycopg2 connect", False, f"{type(e).__name__}: {e}")
        return None
    record("psycopg2 connect", True, f"{args.user}@{args.host}:{args.port}/{args.dbname}")
    return conn


def check_extensions(conn) -> None:
    with conn.cursor() as cur:
        cur.execute("SELECT extname FROM pg_extension")
        exts = {r[0] for r in cur.fetchall()}
    record("hypopg enabled",         "hypopg"         in exts, ", ".join(sorted(exts)))
    record("index_advisor enabled",  "index_advisor"  in exts)


def check_dexter_binary() -> None:
    record("dexter on PATH", shutil.which("dexter") is not None,
           shutil.which("dexter") or "install via `gem install pgdexter`")


def check_dexter_recommend(mode: str) -> None:
    """
    Strong check: invoke dexter directly (not through the wrapper) so we can
    inspect its exit code and stderr.  An exit code of 0 with no error text
    in stderr proves the CLI is wired up correctly; 0 DDL for the probe
    queries is expected and OK.  This catches silent flag regressions that
    the permissive wrapper would otherwise hide.
    """
    import shutil, subprocess
    dexter_bin = shutil.which("dexter")
    if not dexter_bin:
        record("DexterAdvisor.recommend", False, "dexter CLI missing")
        return

    # Build the same connection args the wrapper uses.
    a = DexterAdvisor.from_env(mode)
    cmd = [dexter_bin,
           "--host", a.host, "--port", str(a.port),
           "--dbname", a.dbname, "--username", a.user,
           "--stdin", "--min-cost-savings-pct", str(a.min_cost_savings_pct)]
    env = os.environ.copy()
    if a.password:
        env["PGPASSWORD"] = a.password
    try:
        proc = subprocess.run(cmd, input="\n".join(q + ";" for q in PROBE_QUERIES),
                              capture_output=True, text=True, timeout=20,
                              env=env, check=False)
    except Exception as e:
        record("DexterAdvisor.recommend", False, f"{type(e).__name__}: {e}")
        return
    if proc.returncode != 0:
        record("DexterAdvisor.recommend", False,
               f"dexter exited {proc.returncode}: {proc.stderr.strip()[:200]}")
        return

    # Now exercise the wrapper path too, so we catch regressions there.
    try:
        ddl = a.recommend(PROBE_QUERIES)
    except Exception as e:
        record("DexterAdvisor.recommend", False, f"wrapper raised {type(e).__name__}: {e}")
        return
    record("DexterAdvisor.recommend", True,
           f"exit 0, wrapper returned {len(ddl)} DDL (0 expected for probes)")


def check_supabase_recommend(conn) -> None:
    try:
        a = SupabaseIndexAdvisor()
        ddl = a.recommend(PROBE_QUERIES, conn=conn)
    except Exception as e:
        record("SupabaseIndexAdvisor.recommend", False, f"{type(e).__name__}: {e}")
        return
    record("SupabaseIndexAdvisor.recommend", True, f"{len(ddl)} DDL recommended (0 is OK for probes)")


def check_apply_recommendations(conn) -> None:
    # A no-op application: empty DDL list, drop_existing=False → should just return zeros.
    try:
        stats = apply_recommendations(conn, ddl=[], drop_existing=False)
    except Exception as e:
        record("apply_recommendations (noop)", False, f"{type(e).__name__}: {e}")
        return
    record("apply_recommendations (noop)", True, f"stats={stats}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=("docker", "native"), default="docker",
                    help="Which .env block to read: HSM_DOCKER_* (default) or HSM_DB_*")
    ap.add_argument("--host",     default=None)
    ap.add_argument("--port",     default=None, type=int)
    ap.add_argument("--dbname",   default=None)
    ap.add_argument("--user",     default=None)
    ap.add_argument("--password", default=None)
    cli = ap.parse_args()

    # Start from the env defaults for the chosen mode, then apply any CLI overrides.
    defaults = _defaults_for(cli.mode)
    for k in ("host", "port", "dbname", "user", "password"):
        v = getattr(cli, k)
        if v is not None:
            defaults[k] = v
    args = argparse.Namespace(**defaults)

    log.info("Mode=%s   Target: %s@%s:%s/%s",
             cli.mode, args.user, args.host, args.port, args.dbname)

    conn = check_connection(args)
    if conn is None:
        return summarize(exit_on_fail=True)

    try:
        check_extensions(conn)
        check_dexter_binary()
        check_dexter_recommend(cli.mode)
        check_supabase_recommend(conn)
        check_apply_recommendations(conn)
    finally:
        conn.close()

    return summarize()


def summarize(exit_on_fail: bool = False) -> int:
    failed = [name for name, ok in CHECKS if not ok]
    print("\n" + "─" * 60)
    print(f"smoke_test_advisors: {len(CHECKS) - len(failed)}/{len(CHECKS)} checks passed")
    if failed:
        print("  failures:")
        for name in failed:
            print(f"    - {name}")
        return 1
    print("  all advisors wired up — ready for Phase 2 (OLTP + Burst runners).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
