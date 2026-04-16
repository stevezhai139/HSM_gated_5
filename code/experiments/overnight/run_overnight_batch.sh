#!/usr/bin/env bash
# run_overnight_batch.sh — Serial orchestrator for B1–B5.
#
#   B1  burst large-Q (Q=3000) clamp + naive        ~5.0 h   [DB-heavy]
#   B2  angular/cosine/euclid kernel ablation       ~5 min   [CPU-only]
#   B3  bidirectional θ-transfer (PG 0.55 + Mongo)  ~1.0 h   [DB-heavy]
#   B4  in-phase amplitude perturbation             ~1 min   [CPU-only]
#   B5  noise × weight 2-D grid                     ~1.5 h   [CPU-only]
#
# Total wall clock: ~7.5 h. Order chosen so long DB-bound runs finish first
# (early feedback if anything dies); CPU-only runs fill the tail.
#
# USAGE:
#   cd HSM_gated
#   bash code/experiments/overnight/run_overnight_batch.sh
#
# Progress is tee'd to:
#   results/overnight_2026-04-16/overnight.log
#   results/overnight_2026-04-16/b{1..5}_*.log
#
# A manifest with git SHA + start/stop times + CSV checksums is written to
#   results/overnight_2026-04-16/overnight_manifest.json
# when the batch finishes (even partially — on failure, the manifest records
# which steps completed).

set -u   # -e NOT set: we want partial completion to still be summarised.

REPO="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$REPO"

# shellcheck disable=SC1091
[[ -f .env ]] && source .env

OUT_DIR="$REPO/results/overnight_2026-04-16"
mkdir -p "$OUT_DIR"

OVERNIGHT_LOG="$OUT_DIR/overnight.log"
MANIFEST="$OUT_DIR/overnight_manifest.json"

HERE="$(dirname "$0")"

# Bash 3.2-compatible git SHA lookup: try the repo root first, then walk up
# (macOS default Bash is 3.2; the associative-array syntax below is also 3.2-safe).
GIT_SHA="unknown"
for d in "$REPO" "$REPO/.." "$REPO/../.." "$REPO/../../.."; do
    if [[ -d "$d/.git" ]]; then
        GIT_SHA="$(git -C "$d" rev-parse --short HEAD 2>/dev/null || echo 'unknown')"
        [[ "$GIT_SHA" != "unknown" ]] && break
    fi
done
START_ISO=$(date -u '+%Y-%m-%dT%H:%M:%SZ')

echo "════════════════════════════════════════════════════════════════════" | tee -a "$OVERNIGHT_LOG"
echo "  HSM_gated  Overnight Batch B1–B5" | tee -a "$OVERNIGHT_LOG"
echo "  start=$START_ISO  git=$GIT_SHA" | tee -a "$OVERNIGHT_LOG"
echo "════════════════════════════════════════════════════════════════════" | tee -a "$OVERNIGHT_LOG"

# Parallel indexed arrays (Bash 3.2-compatible, no associative arrays).
STEP_TAGS=()
STEP_STATUSES=()

run_step () {
    local tag="$1" cmd="$2"
    echo "----" | tee -a "$OVERNIGHT_LOG"
    echo "[$(date -u '+%H:%M:%S')] START $tag" | tee -a "$OVERNIGHT_LOG"
    STEP_TAGS+=("$tag")
    if bash -c "$cmd"; then
        STEP_STATUSES+=("ok")
        echo "[$(date -u '+%H:%M:%S')] OK    $tag" | tee -a "$OVERNIGHT_LOG"
    else
        local rc=$?
        STEP_STATUSES+=("fail")
        echo "[$(date -u '+%H:%M:%S')] FAIL  $tag (rc=$rc)" | tee -a "$OVERNIGHT_LOG"
    fi
}

run_step B1 "bash '$HERE/b1_burst_large_q.sh'"
run_step B2 "python3 '$HERE/b2_kernel_ablation.py'"
run_step B3 "bash '$HERE/b3_theta_transfer.sh'"
run_step B4 "python3 '$HERE/b4_inphase_perturbation.py'"
run_step B5 "python3 '$HERE/b5_noise_weight_grid.py'"

END_ISO=$(date -u '+%Y-%m-%dT%H:%M:%SZ')

# ── Status lookup (linear; 5 entries, fine) ─────────────────────────────────
lookup_status () {
    local tag="$1" i=0
    for t in "${STEP_TAGS[@]}"; do
        if [[ "$t" == "$tag" ]]; then echo "${STEP_STATUSES[$i]}"; return; fi
        i=$((i + 1))
    done
    echo "skip"
}
S_B1=$(lookup_status B1); S_B2=$(lookup_status B2); S_B3=$(lookup_status B3)
S_B4=$(lookup_status B4); S_B5=$(lookup_status B5)

# ── Manifest ────────────────────────────────────────────────────────────────
python3 - "$MANIFEST" "$START_ISO" "$END_ISO" "$GIT_SHA" \
    "$S_B1" "$S_B2" "$S_B3" "$S_B4" "$S_B5" "$OUT_DIR" <<'PYEOF'
import json, os, sys, hashlib
from pathlib import Path

fp, start, end, sha, *rest = sys.argv[1:]
s1, s2, s3, s4, s5, out_dir = rest
out_dir = Path(out_dir)

def h(p):
    p = Path(p)
    if not p.exists() or p.is_dir():
        return None
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()[:16]

artefacts = {
    "B1": [
        "results/burst_end_to_end/burst_large_q_results.csv",
        "results/burst_end_to_end/burst_large_q_naive_results.csv",
    ],
    "B2": [
        "results/overnight_2026-04-16/b2_kernel_ablation/kernel_ablation_summary.csv",
        "results/overnight_2026-04-16/b2_kernel_ablation/kernel_ablation_scores.csv",
    ],
    "B3": [
        "results/burst_end_to_end/burst_theta055_results.csv",
        # Mongo output path includes a timestamp — record directory only.
    ],
    "B4": [
        "results/overnight_2026-04-16/b4_inphase/inphase_amplitude.csv",
        "results/overnight_2026-04-16/b4_inphase/inphase_summary.csv",
    ],
    "B5": [
        "results/overnight_2026-04-16/b5_noise_weight/noise_weight_grid.csv",
    ],
}
repo = out_dir.parents[1]
artefact_hashes = {
    step: {path: h(repo / path) for path in paths}
    for step, paths in artefacts.items()
}

manifest = {
    "git_sha": sha, "start_utc": start, "end_utc": end,
    "statuses": {"B1": s1, "B2": s2, "B3": s3, "B4": s4, "B5": s5},
    "artefacts": artefact_hashes,
    "host": os.uname().nodename,
}
Path(fp).write_text(json.dumps(manifest, indent=2))
print(f"manifest written: {fp}")
PYEOF

echo "════════════════════════════════════════════════════════════════════" | tee -a "$OVERNIGHT_LOG"
echo "  Overnight batch finished at $END_ISO" | tee -a "$OVERNIGHT_LOG"
i=0
for tag in "${STEP_TAGS[@]}"; do
    echo "    $tag : ${STEP_STATUSES[$i]}" | tee -a "$OVERNIGHT_LOG"
    i=$((i + 1))
done
echo "  manifest: $MANIFEST" | tee -a "$OVERNIGHT_LOG"
echo "  morning analyzer:  python3 code/experiments/overnight/analyze_overnight.py" | tee -a "$OVERNIGHT_LOG"
echo "════════════════════════════════════════════════════════════════════" | tee -a "$OVERNIGHT_LOG"
