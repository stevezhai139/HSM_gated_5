"""
plot_supp02_gate_sensitivity.py
--------------------------------
Gate sensitivity: fraction of windows that trigger the gate as the trigger
threshold z (in units of sigma) varies.

Reads results/gate_sensitivity.csv with columns
    z, trigger_fraction, workload
"""

from __future__ import annotations

import os
import sys

import matplotlib.pyplot as plt

from _placeholder import placeholder
from _style import PALETTE, apply

OUT_REL = "paper/supplementary/figures/supp02_gate_sensitivity.pdf"
CSV_REL = "results/gate_sensitivity.csv"


def main(repo_root: str = ".") -> None:
    apply()
    out  = os.path.join(repo_root, OUT_REL)
    csv  = os.path.join(repo_root, CSV_REL)

    if not os.path.exists(csv):
        placeholder(out,
                    xlabel="Threshold z",
                    ylabel="Gate trigger fraction",
                    title="Gate sensitivity (z>3\u03c3)",
                    kind="line")
        print(f"[placeholder] {out}  (csv not found: {csv})")
        return

    import pandas as pd
    df = pd.read_csv(csv)
    fig, ax = plt.subplots(figsize=(3.4, 2.4))
    for i, wl in enumerate(sorted(df["workload"].unique())):
        sub = df[df["workload"] == wl].sort_values("z")
        ax.plot(sub["z"], sub["trigger_fraction"], marker="o",
                color=PALETTE[i % len(PALETTE)], label=wl)
    ax.set_xlabel("Threshold z (\u03c3)")
    ax.set_ylabel("Gate trigger fraction")
    ax.legend(loc="best")
    fig.savefig(out)
    plt.close(fig)
    print(f"[real]        {out}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else ".")
