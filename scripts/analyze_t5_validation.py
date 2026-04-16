#!/usr/bin/env python3
"""
analyze_t5_validation.py
=========================
Cross-workload empirical validation of Theorem 5 (Operational Speedup with
Phase-Mix Prior).

For each workload, we read the time-ordered HSM trigger timeseries and
compute:
  - p_stable_hat = (# windows with gate_triggered=0) / total windows
  - predicted speedup bound  : 1 / (1 - p_stable_hat)
  - HSM advisor invocations  : sum(gate_triggered)
  - always-on invocations    : total windows
  - periodic K=3 invocations : ceil(total / 3)
  - invocation savings vs always-on / vs periodic
  - realized invocation ratio (always_on / hsm_gated) — the empirical
    counterpart to the asymptotic Theorem 5 bound; equals predicted bound
    in expectation when HSM perfectly identifies stable windows.

Inputs (relative to repo root /code/results/):
  TPC-H   : ../../results/sf{0.2,1.0,3.0}/raw_results.csv
            (uses advisor_calls per condition × block × SF; 24 windows/block)
  OLTP    : oltp_validation/oltp_hsm_execute_trigger_timeseries.csv
  Burst   : burst_v2_validation/burst_v2_hsm_trigger_timeseries.csv
  JOB     : job_validation/job_hsm_static_trigger_timeseries.csv
  SDSS    : sdss_validation/sdss_hsm_trigger_timeseries.csv

Output:
  results/t5_validation.csv  (paper-ready summary table)
"""

import csv
import math
from pathlib import Path

REPO_ROOT  = Path(__file__).resolve().parents[1]
CODE_RES   = REPO_ROOT / "code" / "results"
PAPER_RES  = REPO_ROOT / "results"
OUT_FILE   = PAPER_RES / "t5_validation.csv"

# Per-workload θ* sources (already published in repo / paper). The paper's
# Theorem 3 says θ* is workload-specific (closed form θ*(N,Q) = 1 − Q_min/Q),
# so we MUST re-decide the gate at each workload's own θ*, not at the global
# default θ=0.75 (which is PostgreSQL/TPC-H specific). Two pre-existing CSVs
# carry these calibrations:
#   - results/theta_optimal_per_workload.csv: Youden-J optima from
#     within/cross HSM-pair labels (TPC-H, SDSS, JOB, JOB-complexity).
#   - results/theta_calibration.csv: per-θ realised-cost sweep with 95% CIs
#     (OLTP, burst, burst_v2, burst_v3, job, job_complexity, sdss).
# We treat the Youden file as authoritative where available, otherwise we
# pick the cost-minimising θ from the calibration sweep.
THETA_PUBLISHED   = PAPER_RES / "theta_optimal_per_workload.csv"
THETA_CALIBRATION = PAPER_RES / "theta_calibration.csv"


def load_published_theta():
    """workload (lowered) -> theta_star, plus J* and AUC."""
    out = {}
    if THETA_PUBLISHED.exists():
        with open(THETA_PUBLISHED, "r") as f:
            for r in csv.DictReader(f):
                out[r["workload"].lower()] = {
                    "theta_star": float(r["theta_star"]),
                    "j_star":     float(r["J_star"]),
                    "auc":        float(r["AUC"]),
                    "source":     "Youden J on within/cross labels",
                }
    return out


def load_costmin_theta():
    """workload (lowered) -> theta with smallest cost_mean in calibration."""
    if not THETA_CALIBRATION.exists():
        return {}
    by_w = {}
    with open(THETA_CALIBRATION, "r") as f:
        for r in csv.DictReader(f):
            w = r["workload"].lower()
            by_w.setdefault(w, []).append(
                (float(r["theta"]), float(r["cost_mean"])))
    out = {}
    for w, rows in by_w.items():
        # Tie-break toward higher θ to favour fewer false alarms.
        rows.sort(key=lambda x: (x[1], -x[0]))
        theta_min, cost_min = rows[0]
        out[w] = {
            "theta_star": theta_min,
            "cost_min":   cost_min,
            "source":     f"cost-min argθ in theta_calibration.csv (cost={cost_min:.3f})",
        }
    return out


def resolve_theta_star(workload_key):
    """Look up workload-specific θ*. Prefer the published Youden table;
    fall back to the cost-minimising θ from theta_calibration.csv."""
    pub = load_published_theta()
    cal = load_costmin_theta()
    k   = workload_key.lower()
    if k in pub:
        rec = pub[k]
        return rec["theta_star"], rec["source"]
    if k in cal:
        rec = cal[k]
        return rec["theta_star"], rec["source"]
    return None, "no calibration on file"


def load_trigger_timeseries(path):
    """Return list of {window_idx, score, gate_triggered, phase_a, phase_b,
    is_cross}. is_cross is the ground truth (phase_a != phase_b)."""
    rows = []
    with open(path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            pa = r.get("phase_a", "")
            pb = r.get("phase_b", "")
            rows.append({
                "window_idx":     int(r["window_idx"]),
                "score":          float(r["score"]),
                "gate_triggered": int(r["gate_triggered"]),
                "phase_a":        pa,
                "phase_b":        pb,
                "is_cross":       (pa != pb),
            })
    return rows


def compute_t5_metrics(rows, periodic_k=3, theta=None):
    """If theta is None, use the gate_triggered field already in rows
    (encoded at theta=0.75 by the validation scripts). Otherwise re-decide
    using `score < theta`.

    Also computes ground-truth detection metrics when phase_a/phase_b are
    available: TP/FP/FN/TN, TPR (drift detection), FPR (false alarms).
    """
    n_total   = len(rows)
    if theta is None:
        n_trig = sum(r["gate_triggered"] for r in rows)
        triggered = [r["gate_triggered"] == 1 for r in rows]
    else:
        triggered = [r["score"] < theta for r in rows]
        n_trig = sum(triggered)
    n_stable  = n_total - n_trig

    # T5 quantity: p_stable based on HSM's own decisions
    p_stable  = n_stable / n_total if n_total else 0.0
    bound     = 1.0 / (1.0 - p_stable) if p_stable < 1.0 else float("inf")

    # Ground-truth p_stable (independent of HSM's decisions): fraction of
    # consec pairs that are within-phase. Available only when phase labels
    # are present.
    has_labels = any(r.get("phase_a") for r in rows)
    if has_labels:
        n_cross_true = sum(1 for r in rows if r["is_cross"])
        n_within_true = n_total - n_cross_true
        p_stable_true = n_within_true / n_total if n_total else 0.0
        # Confusion matrix (positive = drift = should-trigger)
        tp = sum(1 for r, t in zip(rows, triggered) if t and r["is_cross"])
        fp = sum(1 for r, t in zip(rows, triggered) if t and not r["is_cross"])
        fn = sum(1 for r, t in zip(rows, triggered) if not t and r["is_cross"])
        tn = sum(1 for r, t in zip(rows, triggered) if not t and not r["is_cross"])
        tpr = tp / (tp + fn) if (tp + fn) else float("nan")
        fpr = fp / (fp + tn) if (fp + tn) else float("nan")
    else:
        n_cross_true = n_within_true = -1
        p_stable_true = float("nan")
        tp = fp = fn = tn = -1
        tpr = fpr = float("nan")

    n_periodic = math.ceil(n_total / periodic_k)
    save_vs_always   = (n_total - n_trig) / n_total if n_total else 0.0
    save_vs_periodic = (n_periodic - n_trig) / n_periodic if n_periodic else 0.0

    realized_vs_always   = n_total / n_trig    if n_trig else float("inf")
    realized_vs_periodic = n_periodic / n_trig if n_trig else float("inf")

    return {
        "n_windows":              n_total,
        "n_triggered":            n_trig,
        "p_stable_hat":           p_stable,
        "p_stable_true":          p_stable_true,
        "bound_speedup":          bound,
        "always_on_calls":        n_total,
        "periodic_calls":         n_periodic,
        "hsm_gated_calls":        n_trig,
        "savings_vs_always":      save_vs_always,
        "savings_vs_periodic":    save_vs_periodic,
        "realized_vs_always":     realized_vs_always,
        "realized_vs_periodic":   realized_vs_periodic,
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "tpr": tpr, "fpr": fpr,
        "n_cross_true": n_cross_true, "n_within_true": n_within_true,
    }


# ─────────────────────────────────────────────────────────────────────────────
# TPC-H is a special case: we already ran a 10-block × 4-condition RCB and
# stored advisor_calls per block. Aggregate to one summary row per SF.
# ─────────────────────────────────────────────────────────────────────────────
def tpch_metrics(sf):
    raw = REPO_ROOT / "results" / f"sf{sf}" / "raw_results.csv"
    if not raw.exists():
        return None

    by_cond = {}
    with open(raw, "r") as f:
        for r in csv.DictReader(f):
            cond = r["condition"]
            by_cond.setdefault(cond, []).append(int(r["advisor_calls"]))

    # 24 windows per block by experiment design
    n_per_block = 24
    n_blocks    = max(len(v) for v in by_cond.values())
    n_total     = n_per_block * n_blocks

    n_always   = sum(by_cond.get("always_on",  [0]))
    n_periodic = sum(by_cond.get("periodic",   [0]))
    n_hsm      = sum(by_cond.get("hsm_gated",  [0]))

    # n_always should be 24 × n_blocks; periodic should be 8 × n_blocks
    p_stable = (n_total - n_hsm) / n_total if n_total else 0.0
    bound    = 1.0 / (1.0 - p_stable) if p_stable < 1.0 else float("inf")

    return {
        "n_windows":              n_total,
        "n_triggered":            n_hsm,
        "p_stable_hat":           p_stable,
        "p_stable_true":          float("nan"),  # no per-row labels in raw_results.csv
        "bound_speedup":          bound,
        "always_on_calls":        n_always,
        "periodic_calls":         n_periodic,
        "hsm_gated_calls":        n_hsm,
        "savings_vs_always":      (n_always - n_hsm) / n_always   if n_always else 0.0,
        "savings_vs_periodic":    (n_periodic - n_hsm) / n_periodic if n_periodic else 0.0,
        "realized_vs_always":     n_always / n_hsm   if n_hsm else float("inf"),
        "realized_vs_periodic":   n_periodic / n_hsm if n_hsm else float("inf"),
        "tp": -1, "fp": -1, "fn": -1, "tn": -1,
        "tpr": float("nan"), "fpr": float("nan"),
        "n_cross_true": -1, "n_within_true": -1,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Workload registry
# ─────────────────────────────────────────────────────────────────────────────
WORKLOADS = [
    # (display_name, kind, source_or_sf, pair_scores_path_or_None,
    #  theta_lookup_key)
    ("TPC-H SF=0.2",  "tpch",        "0.2",                                   None,
        "tpc-h"),
    ("TPC-H SF=1.0",  "tpch",        "1.0",                                   None,
        "tpc-h"),
    ("TPC-H SF=3.0",  "tpch",        "3.0",                                   None,
        "tpc-h"),
    ("OLTP (pgbench)", "trig",
        CODE_RES / "oltp_validation" / "oltp_hsm_execute_trigger_timeseries.csv",
        CODE_RES / "oltp_validation" / "oltp_hsm_execute_pair_scores.csv",
        "oltp"),
    ("Burst",          "trig",
        CODE_RES / "burst_v2_validation" / "burst_v2_hsm_trigger_timeseries.csv",
        CODE_RES / "burst_v2_validation" / "burst_v2_hsm_pair_scores.csv",
        "burst_v2"),
    ("JOB / IMDB",     "trig",
        CODE_RES / "job_validation"  / "job_hsm_static_trigger_timeseries.csv",
        CODE_RES / "job_validation"  / "job_hsm_static_pair_scores.csv",
        "job"),
    ("SDSS",           "trig",
        CODE_RES / "sdss_validation" / "sdss_hsm_trigger_timeseries.csv",
        CODE_RES / "sdss_validation" / "sdss_hsm_pair_scores.csv",
        "sdss"),
]


def fmt_float(x, n=3):
    if x == float("inf"):
        return "inf"
    return f"{x:.{n}f}"


def youden_optimal_theta(pair_scores_path):
    """
    Find theta* maximising Youden J = TPR - FPR on the within/cross labels
    in *_pair_scores.csv. Returns (theta_star, J_star, n_within, n_cross).
    Convention: 'within' = stable (gate should NOT trigger),
                'cross'  = drift (gate SHOULD trigger).
    Gate fires when score < theta. So:
        TPR (drift caught)  = P(score < theta | cross)
        FPR (false alarms)  = P(score < theta | within)
    """
    within, cross = [], []
    with open(pair_scores_path, "r") as f:
        for r in csv.DictReader(f):
            s = float(r["score"])
            if r["group"] == "within":
                within.append(s)
            else:
                cross.append(s)
    if not within or not cross:
        return None, None, len(within), len(cross)

    # Sweep candidate thresholds drawn from the pooled score grid.
    grid = sorted(set(within + cross))
    best_theta, best_j = None, -1.0
    for t in grid:
        tpr = sum(1 for s in cross  if s < t) / len(cross)
        fpr = sum(1 for s in within if s < t) / len(within)
        j   = tpr - fpr
        if j > best_j:
            best_j, best_theta = j, t
    return best_theta, best_j, len(within), len(cross)


def main():
    PAPER_RES.mkdir(parents=True, exist_ok=True)
    out_rows = []

    print(f"\n{'Workload':<22} {'N':>5}  {'theta':>5} "
          f"{'p_st_T':>6} {'Bnd_T':>5} "
          f"{'HSM':>4} {'TPR':>5} {'FPR':>5} "
          f"{'sav_A':>6} {'sav_P':>6}")
    print("─" * 90)

    for entry in WORKLOADS:
        name, kind   = entry[0], entry[1]
        theta_key    = entry[4]
        theta_star, theta_src = resolve_theta_star(theta_key)

        if kind == "tpch":
            sf = entry[2]
            m = tpch_metrics(sf)
            if m is None:
                print(f"{name:<18}  (raw_results.csv missing — skipped)")
                continue
            # TPC-H raw_results was already gated at default θ=0.75 inside
            # the runner; re-decoding here is impossible because we don't
            # store per-window scores for TPC-H. Report as-is and label
            # what θ would be if we recomputed.
            m["theta_used"]   = 0.75
            m["theta_source"] = "default (gated in-runner)"
            m["theta_star_published"]        = theta_star if theta_star is not None else float("nan")
            m["theta_star_published_source"] = theta_src
            m["workload"]     = name
            out_rows.append(m)
            print(f"{name:<22} {m['n_windows']:>5}  {0.75:>5.3f} "
                  f"{'  --  ':>6} {'  -- ':>5} "
                  f"{m['hsm_gated_calls']:>4} {' --':>5} {' --':>5} "
                  f"{fmt_float(100*m['savings_vs_always'], 1)+'%':>6} "
                  f"{fmt_float(100*m['savings_vs_periodic'], 1)+'%':>6}")
            continue

        # trig-mode workloads: report at default AND at workload-specific θ*
        src, pair_src = entry[2], entry[3]
        if not src.exists():
            print(f"{name:<18}  (missing {src.name} — skipped)")
            continue
        rows = load_trigger_timeseries(src)

        def _pr(m, label, theta_show):
            bound_t = m["bound_speedup"] if m["p_stable_true"] != m["p_stable_true"] else (
                1.0 / (1.0 - m["p_stable_true"]) if m["p_stable_true"] < 1.0 else float("inf")
            )
            tpr_s = "  --" if m["tpr"] != m["tpr"] else f"{m['tpr']:.2f}"
            fpr_s = "  --" if m["fpr"] != m["fpr"] else f"{m['fpr']:.2f}"
            pst_s = "  --" if m["p_stable_true"] != m["p_stable_true"] else f"{m['p_stable_true']:.3f}"
            bnd_s = "  -- " if bound_t != bound_t else (
                "  inf" if bound_t == float("inf") else f"{bound_t:5.2f}"
            )
            print(f"{label:<22} {m['n_windows']:>5}  {theta_show:>5.3f} "
                  f"{pst_s:>6} {bnd_s:>5} "
                  f"{m['hsm_gated_calls']:>4} {tpr_s:>5} {fpr_s:>5} "
                  f"{fmt_float(100*m['savings_vs_always'], 1)+'%':>6} "
                  f"{fmt_float(100*m['savings_vs_periodic'], 1)+'%':>6}")

        # (i) default theta = 0.75 (PostgreSQL/TPC-H tuned, shown for context)
        m_def = compute_t5_metrics(rows)
        m_def["workload"]     = name
        m_def["theta_used"]   = 0.75
        m_def["theta_source"] = "default (PostgreSQL/TPC-H)"
        out_rows.append(m_def)
        _pr(m_def, name, 0.75)

        # (ii) workload-specific θ* — looked up from pre-existing files
        # (theta_optimal_per_workload.csv or theta_calibration.csv). This is
        # the value the paper's Theorem 3 prescribes for each workload, NOT a
        # post-hoc recomputation tailored to this analysis run.
        if theta_star is not None:
            m_cal = compute_t5_metrics(rows, theta=theta_star)
            m_cal["workload"]     = name + " (θ*)"
            m_cal["theta_used"]   = theta_star
            m_cal["theta_source"] = theta_src
            out_rows.append(m_cal)
            _pr(m_cal, "  ↳ workload θ*", theta_star)
        else:
            print(f"  ↳ no published θ* for key '{theta_key}' — skipped")

    # Dump CSV
    cols = ["workload", "theta_used", "theta_source",
            "n_windows", "n_triggered", "p_stable_hat", "p_stable_true",
            "bound_speedup", "always_on_calls", "periodic_calls",
            "hsm_gated_calls", "savings_vs_always", "savings_vs_periodic",
            "realized_vs_always", "realized_vs_periodic",
            "tp", "fp", "fn", "tn", "tpr", "fpr",
            "n_cross_true", "n_within_true"]
    with open(OUT_FILE, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in out_rows:
            w.writerow({c: r[c] for c in cols})

    print(f"\nWrote {OUT_FILE}")


if __name__ == "__main__":
    main()
