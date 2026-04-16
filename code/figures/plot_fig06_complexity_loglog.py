"""
plot_fig06_complexity_loglog.py
--------------------------------
Linear-time complexity verification: HSM scoring time T(n) vs window size n
on log-log axes.  A reference line of slope 1 is overlaid.

Reads results/complexity.csv with columns
    n_pts, time_s, method        (method e.g. {"HSM","baseline"})
"""

from __future__ import annotations

import os
import sys

import matplotlib.pyplot as plt
import numpy as np

from _placeholder import placeholder
from _style import PALETTE, apply

OUT_REL = "paper/main/figures/fig06_complexity_loglog.pdf"
CSV_REL = "results/complexity.csv"


def main(repo_root: str = ".") -> None:
    apply()
    out  = os.path.join(repo_root, OUT_REL)
    csv  = os.path.join(repo_root, CSV_REL)

    if not os.path.exists(csv):
        placeholder(out,
                    xlabel="Window size n_pts",
                    ylabel="Time (s)",
                    title="Linear-time complexity verification",
                    kind="loglog")
        print(f"[placeholder] {out}  (csv not found: {csv})")
        return

    import pandas as pd
    df = pd.read_csv(csv)
    fig, ax = plt.subplots(figsize=(3.4, 2.4))
    for i, method in enumerate(sorted(df["method"].unique())):
        sub = df[df["method"] == method].sort_values("n_pts")
        ax.loglog(sub["n_pts"], sub["time_s"], marker="o",
                  color=PALETTE[i % len(PALETTE)], label=method)
    # Reference slope-1 line
    n = np.array([df["n_pts"].min(), df["n_pts"].max()], dtype=float)
    ref = (df["time_s"].min() / n[0]) * n
    ax.loglog(n, ref, ls="--", color="grey", lw=0.8, label="O(n)")
    ax.set_xlabel("Window size n_pts")
    ax.set_ylabel("Time (s)")
    ax.legend(loc="best")
    fig.savefig(out)
    plt.close(fig)
    print(f"[real]        {out}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else ".")
