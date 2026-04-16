"""
plot_supp03_theta_calibration.py
---------------------------------
theta calibration curve: cost (mean +/- 95% CI) for each candidate threshold
theta in [0, 1].  The empirical optimum is highlighted.

Reads results/theta_calibration.csv with columns
    theta, cost_mean, ci_low, ci_high, workload
"""

from __future__ import annotations

import os
import sys

import matplotlib.pyplot as plt

from _placeholder import placeholder
from _style import PALETTE, apply

OUT_REL = "paper/supplementary/figures/supp03_theta_calibration.pdf"
CSV_REL = "results/theta_calibration.csv"


def main(repo_root: str = ".") -> None:
    apply()
    out  = os.path.join(repo_root, OUT_REL)
    csv  = os.path.join(repo_root, CSV_REL)

    if not os.path.exists(csv):
        placeholder(out,
                    xlabel="\u03b8 candidate",
                    ylabel="Cost (mean \u00b1 95% CI)",
                    title="\u03b8 calibration curve",
                    kind="line")
        print(f"[placeholder] {out}  (csv not found: {csv})")
        return

    import pandas as pd
    df = pd.read_csv(csv)
    fig, ax = plt.subplots(figsize=(3.4, 2.4))
    for i, wl in enumerate(sorted(df["workload"].unique())):
        sub = df[df["workload"] == wl].sort_values("theta")
        ax.plot(sub["theta"], sub["cost_mean"], marker="o",
                color=PALETTE[i % len(PALETTE)], label=wl)
        ax.fill_between(sub["theta"], sub["ci_low"], sub["ci_high"],
                        color=PALETTE[i % len(PALETTE)], alpha=0.18)
        # Highlight the empirical optimum (minimum cost).
        opt = sub.loc[sub["cost_mean"].idxmin()]
        ax.scatter([opt["theta"]], [opt["cost_mean"]],
                   marker="*", s=60,
                   color=PALETTE[i % len(PALETTE)], edgecolor="black",
                   linewidth=0.6, zorder=5)
    ax.set_xlabel("\u03b8 candidate")
    ax.set_ylabel("Cost (mean \u00b1 95% CI)")
    ax.legend(loc="best")
    fig.savefig(out)
    plt.close(fig)
    print(f"[real]        {out}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else ".")
