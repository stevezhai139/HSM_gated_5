#!/usr/bin/env bash
# b3_theta_transfer.sh — Bidirectional θ-transfer (F1 receipt).
#
#   Pass A: PostgreSQL burst-mid-Q at θ = 0.55  (Mongo-optimal θ applied to PG)
#   Pass B: MongoDB A-CE workload at θ = 0.775 (PG-optimal θ applied to Mongo)
#
# Shows: if you deploy a foreign engine's θ, the kernel still behaves predictably
# (PG over-fires because 0.55 is below S_floor by a wide margin; Mongo under-fires
# because 0.775 is above its within-phase distribution tail). Kernel transfers,
# θ is what needs calibrating — consistent with Theorem's ROC framework.
#
# Pass B requires the V3 MongoDB infrastructure (adaptation scripts). If the
# V3 path is not reachable, Pass B is skipped gracefully — Pass A is the
# cheaper, more reproducible half of the receipt.
#
# Output:
#   Pass A: results/burst_end_to_end/burst_theta055_results.csv
#           results/burst_end_to_end/detail_theta055/
#   Pass B: results/overnight_2026-04-16/b3_theta_transfer/mongo_theta077/
#
# Est. runtime: Pass A ≈ 25 min (hsm_gated only, 10 blocks × 2 advisors);
#               Pass B ≈ 35 min (full Mongo adaptation with θ=0.775).

set -euo pipefail

REPO="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$REPO"

# shellcheck disable=SC1091
[[ -f .env ]] && source .env

LOG_DIR="$REPO/results/overnight_2026-04-16"
mkdir -p "$LOG_DIR"

# ── Pass A: PG at θ=0.55 (Mongo-optimal) ────────────────────────────────────
echo "[$(date -u '+%Y-%m-%dT%H:%M:%SZ')] B3 Pass A  PG at θ=0.55" | tee -a "$LOG_DIR/overnight.log"

python3 code/experiments/tier2/hsm_burst_end_to_end.py \
    --tag theta055 \
    --queries-per-phase 700 \
    --theta 0.55 \
    --blocks 10 \
    --policy hsm_gated \
    --advisor dexter supabase \
    --no-k-sweep \
    2>&1 | tee -a "$LOG_DIR/b3_pass_a.log"

echo "[$(date -u '+%Y-%m-%dT%H:%M:%SZ')] B3 Pass A done" | tee -a "$LOG_DIR/overnight.log"

# ── Pass B: MongoDB at θ=0.775 (PG-optimal) ─────────────────────────────────
# REPO = .../Paper 3A/Version 5/HSM_gated → V3 script lives two dirs up under Version 3/.
V3_MONGO_SCRIPT="$REPO/../../Version 3/code/experiments/cross_engine/mongo/adaptation/14_mongo_adaptation_theta_sweep.py"
if [[ ! -f "$V3_MONGO_SCRIPT" ]]; then
    echo "[$(date -u '+%Y-%m-%dT%H:%M:%SZ')] B3 Pass B SKIP: V3 Mongo script not found at $V3_MONGO_SCRIPT" \
        | tee -a "$LOG_DIR/overnight.log"
    exit 0
fi

echo "[$(date -u '+%Y-%m-%dT%H:%M:%SZ')] B3 Pass B  Mongo at θ=0.775" | tee -a "$LOG_DIR/overnight.log"

# Wrapper script overrides THETA=0.775 and writes to a fresh output dir.
python3 "$(dirname "$0")/b3_mongo_theta0775.py" \
    2>&1 | tee -a "$LOG_DIR/b3_pass_b.log"

echo "[$(date -u '+%Y-%m-%dT%H:%M:%SZ')] B3 complete" | tee -a "$LOG_DIR/overnight.log"
