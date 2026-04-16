"""
plot_fig01_score_distribution.py
---------------------------------
Within-phase vs cross-phase HSM composite score distribution.

Reads results/score_distribution.csv with columns
    score, group       (group in {"within", "cross"})

If the CSV is missing, falls back to the watermarked placeholder so the
paper still compiles cleanly.
"""

from __future__ import annotations

import os
import sys

import matplotlib.pyplot as plt
import numpy as np

from _placeholder import placeholder
from _style import PALETTE, apply

OUT_REL = "paper/main/figures/fig01_score_distribution.pdf"
CSV_REL = "results/score_distribution.csv"


def main(repo_root: str = ".") -> None:
    apply()
    out  = os.path.join(repo_root, OUT_REL)
    csv  = os.path.join(repo_root, CSV_REL)

    if not os.path.exists(csv):
        placeholder(out,
                    xlabel="HSM composite score",
                    ylabel="Frequency",
                    title="Within-phase vs cross-phase score distribution",
                    kind="hist")
        print(f"[placeholder] {out}  (csv not found: {csv})")
        return

    import pandas as pd
    df = pd.read_csv(csv)
    fig, ax = plt.subplots(figsize=(3.4, 2.4))
    bins = np.linspace(0.0, 1.0, 31)
    for i, group in enumerate(("within", "cross")):
        sub = df[df["group"] == group]["score"].to_numpy()
        ax.hist(sub, bins=bins, alpha=0.55, label=group, color=PALETTE[i])
    ax.set_xlabel("HSM composite score")
    ax.set_ylabel("Frequency")
    ax.legend(loc="upper center")
    fig.savefig(out)
    plt.close(fig)
    print(f"[real]        {out}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else ".")
