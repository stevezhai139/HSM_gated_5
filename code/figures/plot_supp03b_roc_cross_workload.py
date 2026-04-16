"""
plot_supp03b_roc_cross_workload.py
----------------------------------
Cross-workload ROC analysis: per-workload Youden-optimal theta* derived from
within/cross pair-scores (TPC-H SF=0.2, SDSS, JOB, JOB-complexity).

Detector convention:  trigger when score < theta  (low HSM similarity =>
phase-change predicted).  Cross-pairs are positives (label=1), within-pairs
are negatives (label=0).

Outputs:
  - paper/supplementary/figures/supp03b_roc_cross_workload.pdf  (ROC overlay)
  - results/theta_optimal_per_workload.csv                      (theta*, J*, AUC)
"""

from __future__ import annotations

import os
import sys
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from _placeholder import placeholder
from _style import PALETTE, apply


OUT_FIG = "paper/supplementary/figures/supp03b_roc_cross_workload.pdf"
OUT_CSV = "results/theta_optimal_per_workload.csv"

# Map plot label -> (csv path relative to repo root, workload key).
# NB: code/results/within_cross_phase.csv contains only SDSS rows; the file's
# rows tagged ``tpch'' are duplicates of the JOB pair-scores (data-pipeline
# bug that mislabeled JOB as TPC-H) and are therefore NOT used here.  The
# canonical TPC-H all-pairs scores live in code/results/score_distribution.csv
# (no workload column).
WORKLOADS: Dict[str, Tuple[str, str | None]] = {
    "TPC-H":            ("code/results/score_distribution.csv",                None),
    "SDSS":             ("code/results/sdss_validation/sdss_hsm_pair_scores.csv", None),
    "JOB":              ("code/results/job_validation/job_hsm_execute_pair_scores.csv", None),
    "JOB-complexity":   ("code/results/job_validation/job_hsm_complexity_execute_pair_scores.csv", None),
}


def load_pairs(repo_root: str, csv_rel: str, wl_key: str | None) -> pd.DataFrame:
    """Load a pair-scores CSV and return a frame with columns score, group."""
    p = os.path.join(repo_root, csv_rel)
    df = pd.read_csv(p)
    if wl_key is not None and "workload" in df.columns:
        df = df[df["workload"] == wl_key]
    if "score" not in df.columns or "group" not in df.columns:
        raise ValueError(f"{csv_rel}: missing score/group columns")
    df = df[df["group"].isin(["within", "cross"])][["score", "group"]].copy()
    return df.reset_index(drop=True)


def roc_youden(df: pd.DataFrame, theta_grid: np.ndarray) -> Tuple[
        np.ndarray, np.ndarray, np.ndarray, float, float, float, float]:
    """Sweep theta on a fixed grid; return TPR, FPR, J arrays plus theta*, J*, AUC, n.

    Decision rule: predict positive (cross) when score < theta.
    """
    cross_scores  = df.loc[df["group"] == "cross",  "score"].to_numpy()
    within_scores = df.loc[df["group"] == "within", "score"].to_numpy()
    n_pos, n_neg = len(cross_scores), len(within_scores)

    tpr = np.array([(cross_scores  < t).mean() if n_pos else 0.0 for t in theta_grid])
    fpr = np.array([(within_scores < t).mean() if n_neg else 0.0 for t in theta_grid])
    j   = tpr - fpr

    # Trapezoidal AUC over (FPR, TPR), sorted by FPR ascending
    order = np.argsort(fpr)
    fpr_s, tpr_s = fpr[order], tpr[order]
    auc = float(np.trapz(tpr_s, fpr_s))

    idx_star = int(np.argmax(j))
    return tpr, fpr, j, float(theta_grid[idx_star]), float(j[idx_star]), auc, n_pos + n_neg


def main(repo_root: str = ".") -> None:
    apply()
    fig_out = os.path.join(repo_root, OUT_FIG)
    csv_out = os.path.join(repo_root, OUT_CSV)
    os.makedirs(os.path.dirname(csv_out), exist_ok=True)
    os.makedirs(os.path.dirname(fig_out), exist_ok=True)

    theta_grid = np.linspace(0.0, 1.0, 401)  # step 0.0025

    rows = []
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.6))

    for i, (label, (csv_rel, wl_key)) in enumerate(WORKLOADS.items()):
        try:
            df = load_pairs(repo_root, csv_rel, wl_key)
        except FileNotFoundError:
            print(f"[skip] {label}: csv not found ({csv_rel})")
            continue
        if df.empty:
            print(f"[skip] {label}: no within/cross rows")
            continue

        tpr, fpr, j, theta_star, j_star, auc, n_pairs = roc_youden(df, theta_grid)
        n_within = int((df["group"] == "within").sum())
        n_cross  = int((df["group"] == "cross").sum())
        rows.append({
            "workload":  label,
            "n_within":  n_within,
            "n_cross":   n_cross,
            "n_pairs":   n_pairs,
            "theta_star": round(theta_star, 4),
            "J_star":     round(j_star, 4),
            "AUC":        round(auc, 4),
        })

        c = PALETTE[i % len(PALETTE)]
        # Panel A: ROC curve
        order = np.argsort(fpr)
        axes[0].plot(fpr[order], tpr[order], color=c, label=f"{label} (AUC={auc:.3f})")
        # Mark theta* operating point
        idx_star = int(np.argmax(j))
        axes[0].scatter([fpr[idx_star]], [tpr[idx_star]], marker="*", s=55,
                        color=c, edgecolor="black", linewidth=0.5, zorder=5)

        # Panel B: J vs theta
        axes[1].plot(theta_grid, j, color=c, label=f"{label}")
        axes[1].scatter([theta_star], [j_star], marker="*", s=55,
                        color=c, edgecolor="black", linewidth=0.5, zorder=5)

    # Save table
    out_df = pd.DataFrame(rows)
    out_df.to_csv(csv_out, index=False)
    print(f"[csv]  {csv_out}")
    print(out_df.to_string(index=False))

    # Cosmetics
    axes[0].plot([0, 1], [0, 1], "--", color="grey", linewidth=0.6)
    axes[0].set_xlim(0, 1); axes[0].set_ylim(0, 1.02)
    axes[0].set_xlabel("FPR"); axes[0].set_ylabel("TPR")
    axes[0].set_title("(A) ROC by workload")
    axes[0].legend(loc="lower right", fontsize=6)

    axes[1].set_xlim(0, 1); axes[1].set_ylim(-0.05, 1.05)
    axes[1].axhline(0.0, color="grey", linewidth=0.4)
    axes[1].set_xlabel("$\\theta$ candidate"); axes[1].set_ylabel("Youden $J = $ TPR $-$ FPR")
    axes[1].set_title("(B) Youden $J$ vs $\\theta$ (\\star = $\\theta^*$)")
    axes[1].legend(loc="lower center", fontsize=6, ncol=2)

    fig.tight_layout()
    fig.savefig(fig_out)
    plt.close(fig)
    print(f"[fig]  {fig_out}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else ".")
