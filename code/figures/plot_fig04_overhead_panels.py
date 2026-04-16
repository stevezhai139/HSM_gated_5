"""
plot_fig04_overhead_panels.py
------------------------------
HSM overhead breakdown across three panels:
    (A) per-component time (ms)
    (B) memory footprint (KB)
    (C) end-to-end advisor time vs window size

Reads results/overhead.csv with columns
    panel, label, value, err
where panel in {"time","mem","advisor"}.
"""

from __future__ import annotations

import os
import sys

import matplotlib.pyplot as plt
import numpy as np

from _placeholder import placeholder
from _style import PALETTE, apply

OUT_REL = "paper/main/figures/fig04_overhead_panels.pdf"
CSV_REL = "results/overhead.csv"


def main(repo_root: str = ".") -> None:
    apply()
    out  = os.path.join(repo_root, OUT_REL)
    csv  = os.path.join(repo_root, CSV_REL)

    if not os.path.exists(csv):
        placeholder(out,
                    xlabel="Component",
                    ylabel="Time (ms) / Footprint",
                    title="HSM overhead",
                    kind="panels3")
        print(f"[placeholder] {out}  (csv not found: {csv})")
        return

    import pandas as pd
    df = pd.read_csv(csv)
    fig, axes = plt.subplots(1, 3, figsize=(6.4, 2.4))
    panels   = (("time",    "Time (ms)",     True),   # use log-y
                ("mem",     "Footprint (KB)", False),
                ("advisor", "Advisor (s)",    False))
    for ax, (panel, ylab, use_log) in zip(axes, panels):
        sub = df[df["panel"] == panel].copy()
        # For log-scale, clamp zero/near-zero values so bars remain visible.
        vals = sub["value"].astype(float).to_numpy()
        if use_log:
            floor = max(vals[vals > 0].min() * 0.5, 1e-4) if (vals > 0).any() else 1e-4
            vals = np.where(vals <= 0, floor, vals)
        ax.bar(range(len(sub)), vals, yerr=sub.get("err"),
               color=PALETTE[: len(sub)])
        ax.set_xticks(range(len(sub)))
        ax.set_xticklabels(sub["label"], rotation=30, ha="right")
        ax.set_ylabel(ylab)
        if use_log:
            ax.set_yscale("log")
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    print(f"[real]        {out}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else ".")
