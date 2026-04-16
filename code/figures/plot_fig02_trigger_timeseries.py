"""
plot_fig02_trigger_timeseries.py
---------------------------------
Adaptation trigger timeseries: HSM score per window with the gate threshold
overlaid as a horizontal line.

Reads results/trigger_timeseries.csv with columns
    window_idx, score, gate_triggered    (gate_triggered in {0, 1})
"""

from __future__ import annotations

import os
import sys

import matplotlib.pyplot as plt

from _placeholder import placeholder
from _style import PALETTE, apply

OUT_REL = "paper/main/figures/fig02_trigger_timeseries.pdf"
CSV_REL = "results/trigger_timeseries.csv"


def main(repo_root: str = ".") -> None:
    apply()
    out  = os.path.join(repo_root, OUT_REL)
    csv  = os.path.join(repo_root, CSV_REL)

    if not os.path.exists(csv):
        placeholder(out,
                    xlabel="Window index",
                    ylabel="HSM score",
                    title="Adaptation trigger across windows",
                    kind="line")
        print(f"[placeholder] {out}  (csv not found: {csv})")
        return

    import pandas as pd
    df = pd.read_csv(csv)
    fig, ax = plt.subplots(figsize=(3.4, 2.4))

    # HSM score line
    ax.plot(df["window_idx"], df["score"], color=PALETTE[0], lw=1.0,
            label="HSM score")

    # Gate trigger markers (below theta)
    triggered = df[df["gate_triggered"] == 1]
    not_trig  = df[df["gate_triggered"] == 0]
    if len(triggered):
        ax.scatter(triggered["window_idx"], triggered["score"],
                   s=22, color=PALETTE[1], zorder=3, label="trigger")
    if len(not_trig):
        ax.scatter(not_trig["window_idx"], not_trig["score"],
                   s=14, facecolors="white", edgecolors=PALETTE[0],
                   linewidths=0.8, zorder=3, label="no trigger")

    # Use the paper default theta = 0.75 (§IV gate threshold) for visual
    # consistency across all figures and runs.  The underlying CSV encodes
    # the same threshold in its `gate_triggered` column.
    theta = 0.75
    ax.axhline(theta, color="gray", lw=0.8, ls="--",
               label=f"$\\theta$ = {theta:.2f}")

    ax.set_xlabel("Window index")
    ax.set_ylabel("HSM score")
    ax.legend(loc="lower right", fontsize=7)
    fig.savefig(out)
    plt.close(fig)
    print(f"[real]        {out}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else ".")
