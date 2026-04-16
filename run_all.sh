#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────────────
# HSM_gated — master v2 sweep driver
# ──────────────────────────────────────────────────────────────────────────────
# Runs the full HSM v2 validation pipeline end-to-end:
#
#   1. Bring up Docker PostgreSQL 16 (8 GB RAM cap, port 5433)
#   2. Load TPC-B (pgbench) and optionally IMDB (JOB) data
#   3. Run the seven per-workload validation scripts
#   4. Regenerate all ten paper figures
#
# All scripts import the canonical v2 kernel in
#   code/experiments/hsm_v2_kernel.py
# which implements the paper §III formulas (Spearman / log-ratio / angular
# distance / dual Jaccard / DWT-db4+SAX+FastDTW) — there are no legacy
# Jaccard / db2 paths left in the pipeline.
#
# USAGE
#   ./run_all.sh            # start Docker + all 7 validations + figures
#   ./run_all.sh --smoke    # smoke-only (no --execute); fast sanity pass
#   ./run_all.sh --skip-docker   # skip docker-compose up (DB already running)
#   ./run_all.sh --oltp-only     # only OLTP + burst (no IMDB dependency)
#
# EXIT CODES
#   0  all phases completed
#   1  Docker bring-up or health check failed
#   2  at least one validation script failed  (rest still run, code=2 returned)
# ──────────────────────────────────────────────────────────────────────────────
set -o pipefail

HERE="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$HERE"

MODE=execute
SKIP_DOCKER=0
OLTP_ONLY=0
for arg in "$@"; do
  case $arg in
    --smoke)        MODE=smoke        ;;
    --skip-docker)  SKIP_DOCKER=1     ;;
    --oltp-only)    OLTP_ONLY=1       ;;
    -h|--help)      grep '^#' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *)              echo "unknown arg: $arg" >&2; exit 1 ;;
  esac
done

EXP_DIR="$HERE/code/experiments"
FIG_DIR="$HERE/code/figures"
DOCKER_DIR="$HERE/code/docker"
LOG_DIR="$HERE/results/_run_logs"
mkdir -p "$LOG_DIR"

TS=$(date +%Y%m%d_%H%M%S)
RUN_LOG="$LOG_DIR/run_all_${TS}.log"
: > "$RUN_LOG"
echo "[run_all] $(date)  mode=$MODE  skip-docker=$SKIP_DOCKER  oltp-only=$OLTP_ONLY" | tee -a "$RUN_LOG"

# ── Phase 1: Docker PostgreSQL 16 ────────────────────────────────────────────
if [ "$SKIP_DOCKER" -eq 0 ]; then
  echo "[phase 1] starting Docker PostgreSQL (port 5433) ..." | tee -a "$RUN_LOG"
  ( cd "$DOCKER_DIR" && docker compose up -d 2>&1 | tee -a "$RUN_LOG" ) \
    || { echo "[phase 1] docker compose up failed"; exit 1; }

  # Wait until healthy (up to 60 s)
  for _ in $(seq 1 12); do
    status=$(docker inspect -f '{{.State.Health.Status}}' tpch_docker 2>/dev/null || echo "starting")
    if [ "$status" = "healthy" ]; then break; fi
    sleep 5
  done
  if [ "$status" != "healthy" ]; then
    echo "[phase 1] container did not reach healthy state" | tee -a "$RUN_LOG"
    exit 1
  fi
  echo "[phase 1] PostgreSQL healthy" | tee -a "$RUN_LOG"
fi

# ── Phase 2: pgbench TPC-B data (used by OLTP + burst scripts) ───────────────
echo "[phase 2] ensuring pgbench 'oltp' database exists ..." | tee -a "$RUN_LOG"
PGPASSWORD=${HSM_DOCKER_PASSWORD:-postgres} psql -h ${HSM_DOCKER_HOST:-localhost} -p ${HSM_DOCKER_PORT:-5433} \
  -U ${HSM_DOCKER_USER:-postgres} -tc "SELECT 1 FROM pg_database WHERE datname='oltp'" \
  2>>"$RUN_LOG" | grep -q 1 || \
  PGPASSWORD=${HSM_DOCKER_PASSWORD:-postgres} psql -h ${HSM_DOCKER_HOST:-localhost} -p ${HSM_DOCKER_PORT:-5433} \
  -U ${HSM_DOCKER_USER:-postgres} -c "CREATE DATABASE oltp" 2>>"$RUN_LOG" || true

# Populate pgbench if pgbench_accounts is empty
ACC_ROWS=$(PGPASSWORD=${HSM_DOCKER_PASSWORD:-postgres} psql -h ${HSM_DOCKER_HOST:-localhost} -p ${HSM_DOCKER_PORT:-5433} \
  -U ${HSM_DOCKER_USER:-postgres} -d oltp -tc "SELECT count(*) FROM pgbench_accounts" 2>/dev/null | tr -d ' ')
if [ -z "$ACC_ROWS" ] || [ "$ACC_ROWS" -lt 10000 ]; then
  echo "[phase 2] initialising pgbench TPC-B (scale 10) ..." | tee -a "$RUN_LOG"
  PGPASSWORD=${HSM_DOCKER_PASSWORD:-postgres} pgbench -h ${HSM_DOCKER_HOST:-localhost} -p ${HSM_DOCKER_PORT:-5433} \
    -U ${HSM_DOCKER_USER:-postgres} -i -s 10 oltp 2>&1 | tee -a "$RUN_LOG"
fi

# ── Phase 3: run per-workload validation scripts ─────────────────────────────
FAIL=0
run_script () {
  local name="$1"; shift
  local script="$EXP_DIR/$name"
  local log="$LOG_DIR/${name%.py}_${TS}.log"
  echo "[phase 3] $name ..." | tee -a "$RUN_LOG"
  python3 "$script" "$@" > "$log" 2>&1 && \
    echo "  -> OK  ($(wc -l < "$log") lines)" | tee -a "$RUN_LOG" || {
    echo "  -> FAIL  see $log" | tee -a "$RUN_LOG"
    FAIL=1
  }
}

FLAGS=""
[ "$MODE" = "execute" ] && FLAGS="--execute"
[ "$MODE" = "smoke"   ] && FLAGS="--smoke"

run_script hsm_oltp_validation.py              $FLAGS
run_script hsm_burst_validation.py             $FLAGS
run_script hsm_burst_v2_validation.py          $FLAGS
run_script hsm_burst_v3_validation.py          $FLAGS

if [ "$OLTP_ONLY" -eq 0 ]; then
  # JOB needs IMDB data separately
  run_script hsm_job_validation.py             $FLAGS
  run_script hsm_job_complexity_validation.py  $FLAGS
  # SDSS reads a CSV log file, not Docker
  run_script hsm_sdss_validation.py
fi

# ── Phase 3b: aggregate per-workload CSVs for paper table + figure inputs ───
echo "[phase 3b] building summary_all.csv and figure CSVs ..." | tee -a "$RUN_LOG"
( cd "$HERE" && python3 scripts/build_summary_all.py 2>&1 | tee -a "$RUN_LOG" ) \
  || echo "[phase 3b] summary_all.csv build reported missing inputs (non-fatal)" | tee -a "$RUN_LOG"
( cd "$HERE" && python3 scripts/build_figure_csvs.py 2>&1 | tee -a "$RUN_LOG" ) \
  || echo "[phase 3b] figure CSV build reported missing inputs (non-fatal)" | tee -a "$RUN_LOG"
( cd "$HERE" && python3 scripts/build_gate_calibration_csvs.py 2>&1 | tee -a "$RUN_LOG" ) \
  || echo "[phase 3b] gate/theta calibration build reported missing inputs (non-fatal)" | tee -a "$RUN_LOG"
( cd "$HERE" && python3 scripts/bench_complexity.py 2>&1 | tee -a "$RUN_LOG" ) \
  || echo "[phase 3b] complexity benchmark reported failure (non-fatal)" | tee -a "$RUN_LOG"
( cd "$HERE" && python3 scripts/bench_overhead.py 2>&1 | tee -a "$RUN_LOG" ) \
  || echo "[phase 3b] overhead benchmark reported failure (non-fatal)" | tee -a "$RUN_LOG"
( cd "$HERE" && python3 scripts/bench_noise_sensitivity.py 2>&1 | tee -a "$RUN_LOG" ) \
  || echo "[phase 3b] noise sensitivity reported failure (non-fatal)" | tee -a "$RUN_LOG"
( cd "$HERE" && python3 scripts/bench_window_sweep.py 2>&1 | tee -a "$RUN_LOG" ) \
  || echo "[phase 3b] window-size sweep reported failure (non-fatal)" | tee -a "$RUN_LOG"
( cd "$HERE" && python3 scripts/aggregate_throughput.py 2>&1 | tee -a "$RUN_LOG" ) \
  || echo "[phase 3b] throughput aggregation skipped (raw_results.csv not found)" | tee -a "$RUN_LOG"

# ── Phase 4: regenerate all figures ──────────────────────────────────────────
echo "[phase 4] regenerating figures ..." | tee -a "$RUN_LOG"
( cd "$HERE" && python3 code/figures/regenerate_all_figures.py . 2>&1 | tee -a "$RUN_LOG" ) \
  || { echo "[phase 4] figure regeneration failed"; FAIL=1; }

# ── Done ─────────────────────────────────────────────────────────────────────
if [ "$FAIL" -eq 0 ]; then
  echo "[run_all] all phases completed successfully  (log: $RUN_LOG)"
  exit 0
else
  echo "[run_all] completed with some failures       (log: $RUN_LOG)" >&2
  exit 2
fi
