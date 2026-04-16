#!/usr/bin/env bash
# b1_burst_large_q.sh — Burst large-Q (Q=3000/phase, 9000 total) at two thetas.
#
#   Pass A (clamped, paper-recommended):  theta = 0.875 = S_floor, all 11 conds
#   Pass B (Theorem-3-naive control):     theta = 0.959 = 1 - 122/3000, hsm_gated only
#
# Demonstrates: Theorem 3 closed-form theta* exceeds noise floor at large Q,
# so θ_eff = min(θ_econ, S_floor) is essential. Pass B exposes the over-firing
# that the clamp prevents.
#
# Output:
#   results/burst_end_to_end/burst_large_q_results.csv     (Pass A, 110 runs)
#   results/burst_end_to_end/detail_large_q/               (Pass A per-window)
#   results/burst_end_to_end/burst_large_q_naive_results.csv (Pass B, 10 runs)
#   results/burst_end_to_end/detail_large_q_naive/
#
# Est. runtime: ~4.5 h (Pass A) + ~25 min (Pass B) = ~5 h.

set -euo pipefail

REPO="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$REPO"

# shellcheck disable=SC1091
[[ -f .env ]] && source .env

LOG_DIR="$REPO/results/overnight_2026-04-16"
mkdir -p "$LOG_DIR"

echo "[$(date -u '+%Y-%m-%dT%H:%M:%SZ')] B1 Pass A (θ=0.875 clamp, 11 conds × 10 blocks × Q=3000)" | tee -a "$LOG_DIR/overnight.log"

python3 code/experiments/tier2/hsm_burst_end_to_end.py \
    --tag large_q \
    --queries-per-phase 3000 \
    --theta 0.875 \
    --blocks 10 \
    2>&1 | tee -a "$LOG_DIR/b1_pass_a.log"

echo "[$(date -u '+%Y-%m-%dT%H:%M:%SZ')] B1 Pass A done" | tee -a "$LOG_DIR/overnight.log"
echo "[$(date -u '+%Y-%m-%dT%H:%M:%SZ')] B1 Pass B (θ=0.959 naive, hsm_gated only)" | tee -a "$LOG_DIR/overnight.log"

python3 code/experiments/tier2/hsm_burst_end_to_end.py \
    --tag large_q_naive \
    --queries-per-phase 3000 \
    --theta 0.959 \
    --blocks 10 \
    --policy hsm_gated \
    --advisor dexter supabase \
    --no-k-sweep \
    2>&1 | tee -a "$LOG_DIR/b1_pass_b.log"

echo "[$(date -u '+%Y-%m-%dT%H:%M:%SZ')] B1 complete" | tee -a "$LOG_DIR/overnight.log"
