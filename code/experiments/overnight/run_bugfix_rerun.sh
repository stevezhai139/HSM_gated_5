#!/usr/bin/env bash
# run_bugfix_rerun.sh — Re-runs only the 3 failed steps from 2026-04-16 overnight:
#
#   B2: Kernel ablation (F3)            — CPU-only, ~5–15 min
#   B5: Noise × weight grid              — CPU-only, ~1.5 h
#   B3-B: MongoDB θ=0.775                — Mongo-bound, ~35 min
#
# B1, B3-A, B4 are already clean and are NOT re-run here.
#
# Before running each step, stale outputs from the previous (buggy) runs are
# deleted so CSVs do not double-append. Nothing outside
#   results/overnight_2026-04-16/{b2_kernel_ablation,b5_noise_weight}
# and the Mongo adaptation_theta0775 dir is touched.
#
# Usage (from any dir):
#   bash code/experiments/overnight/run_bugfix_rerun.sh
#
# Log: results/overnight_2026-04-16/bugfix_rerun.log

set -u
set -o pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "$HERE/../../.." && pwd)"
cd "$REPO"

# shellcheck disable=SC1091
[[ -f .env ]] && source .env

LOG_DIR="$REPO/results/overnight_2026-04-16"
LOG="$LOG_DIR/bugfix_rerun.log"
mkdir -p "$LOG_DIR"

stamp () { date -u '+%Y-%m-%dT%H:%M:%SZ'; }
log   () { echo "[$(stamp)] $*" | tee -a "$LOG"; }

STEP_TAGS=()
STEP_STATUSES=()

run_step () {
    local tag="$1"
    local cmd="$2"
    STEP_TAGS+=("$tag")
    log "=== $tag START ==="
    if bash -c "$cmd"; then
        STEP_STATUSES+=("ok")
        log "=== $tag OK ==="
    else
        STEP_STATUSES+=("fail")
        log "=== $tag FAIL ==="
    fi
}

# ── Clean stale outputs from the buggy 06:38 run ────────────────────────────
log "Cleaning stale B2 + B5 + B3-B output dirs"
rm -rf "$LOG_DIR/b2_kernel_ablation" \
       "$LOG_DIR/b5_noise_weight"
# B3 Pass B — wipe only adaptation_theta0775 (it was never produced; safe either way)
V3_ADAPT_ROOT="$REPO/../../Version 3/code/results/cross_engine/mongo/adaptation_theta0775"
if [[ -d "$V3_ADAPT_ROOT" ]]; then
    log "Cleaning stale V3 adaptation_theta0775 dir"
    rm -rf "$V3_ADAPT_ROOT"
fi

# ── B2: Kernel ablation ─────────────────────────────────────────────────────
run_step "B2" "python3 \"$HERE/b2_kernel_ablation.py\" 2>&1 | tee -a \"$LOG_DIR/b2_kernel_ablation_rerun.log\""

# ── B5: Noise × weight grid ─────────────────────────────────────────────────
run_step "B5" "python3 \"$HERE/b5_noise_weight_grid.py\" 2>&1 | tee -a \"$LOG_DIR/b5_noise_weight_rerun.log\""

# ── B3 Pass B: Mongo at θ=0.775 ─────────────────────────────────────────────
V3_MONGO_SCRIPT="$REPO/../../Version 3/code/experiments/cross_engine/mongo/adaptation/14_mongo_adaptation_theta_sweep.py"
if [[ ! -f "$V3_MONGO_SCRIPT" ]]; then
    log "B3-B SKIP: V3 Mongo script not found at $V3_MONGO_SCRIPT"
    STEP_TAGS+=("B3-B"); STEP_STATUSES+=("skip")
else
    run_step "B3-B" "python3 \"$HERE/b3_mongo_theta0775.py\" 2>&1 | tee -a \"$LOG_DIR/b3_pass_b_rerun.log\""
fi

# ── Summary ─────────────────────────────────────────────────────────────────
log ""
log "============ bugfix rerun summary ============"
i=0
for t in "${STEP_TAGS[@]}"; do
    log "  $t: ${STEP_STATUSES[$i]}"
    i=$((i + 1))
done
log "=============================================="
log "Next: python3 $HERE/analyze_overnight.py"
