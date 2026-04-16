"""
plot_fig03_within_cross_phase.py
---------------------------------
Within- vs cross-phase score density for two workloads (TPC-H, SDSS).

Reads results/within_cross_phase.csv with columns
    workload, group, score              (workload in {"tpch","sdss"};
                                         group    in {"within","cross"})
"""

from __future__ import annotations

import os
import sys

import matplotlib.pyplot as plt
import numpy as np

from _placeholder import placeholder
from _style import PALETTE, apply

OUT_REL = "paper/main/figures/fig03_within_cross_phase.pdf"
CSV_REL = "results/within_cross_phase.csv"


def main(repo_root: str = ".") -> None:
    apply()
    out  = os.path.join(repo_root, OUT_REL)
    csv  = os.path.join(repo_root, CSV_REL)

    if not os.path.exists(csv):
        placeholder(out,
                    xlabel="HSM composite score",
                    ylabel="Density",
                    title="TPC-H vs SDSS within/cross-phase",
                    kind="hist")
        print(f"[placeholder] {out}  (csv not found: {csv})")
        return

    import pandas as pd
    df = pd.read_csv(csv)
    fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.4), sharey=True)
    bins = np.linspace(0.0, 1.0, 31)
    for ax, wl in zip(axes, ("tpch", "sdss")):
        for i, group in enumerate(("within", "cross")):
            sub = df[(df["workload"] == wl) & (df["group"] == group)]["score"]
            ax.hist(sub, bins=bins, alpha=0.55, density=True,
                    color=PALETTE[i], label=group)
        ax.set_title(wl.upper())
        ax.set_xlabel("HSM composite score")
    axes[0].set_ylabel("Density")
    axes[1].legend(loc="upper center")
    fig.savefig(out)
    plt.close(fig)
    print(f"[real]        {out}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else ".")
