"""
plot_fig05_noise_sensitivity.py
--------------------------------
Discrimination ratio (DR) under +/-60% noise injected in N1..N5 conditions.

Reads results/noise_sensitivity.csv with columns
    noise_level, condition, dr        (condition in {"N1","N2","N3","N4","N5"})
"""

from __future__ import annotations

import os
import sys

import matplotlib.pyplot as plt

from _placeholder import placeholder
from _style import PALETTE, apply

OUT_REL = "paper/main/figures/fig05_noise_sensitivity.pdf"
CSV_REL = "results/noise_sensitivity.csv"


def main(repo_root: str = ".") -> None:
    apply()
    out  = os.path.join(repo_root, OUT_REL)
    csv  = os.path.join(repo_root, CSV_REL)

    if not os.path.exists(csv):
        placeholder(out,
                    xlabel="Noise level",
                    ylabel="Discrimination ratio (DR)",
                    title="Noise sensitivity (\u00b160% N1\u2013N5)",
                    kind="line")
        print(f"[placeholder] {out}  (csv not found: {csv})")
        return

    import pandas as pd
    df = pd.read_csv(csv)
    fig, ax = plt.subplots(figsize=(3.4, 2.4))
    for i, cond in enumerate(sorted(df["condition"].unique())):
        sub = df[df["condition"] == cond].sort_values("noise_level")
        ax.plot(sub["noise_level"], sub["dr"], marker="o",
                color=PALETTE[i % len(PALETTE)], label=cond)
    ax.set_xlabel("Noise level")
    ax.set_ylabel("Discrimination ratio (DR)")
    ax.legend(loc="best", ncol=2)
    fig.savefig(out)
    plt.close(fig)
    print(f"[real]        {out}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else ".")
