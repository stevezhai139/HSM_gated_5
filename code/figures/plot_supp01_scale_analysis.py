"""
plot_supp01_scale_analysis.py
------------------------------
Scale analysis: advisor time T_A(N) versus cardinality N on log-log axes.

Reads results/scale_analysis.csv with columns
    N, advisor_time_s, model
where `model` distinguishes empirical samples from the analytic fit
"a*N*log(N) + b" (paper supplementary Eq. supp-nstar-implicit).
"""

from __future__ import annotations

import os
import sys

import matplotlib.pyplot as plt

from _placeholder import placeholder
from _style import PALETTE, apply

OUT_REL = "paper/supplementary/figures/supp01_scale_analysis.pdf"
CSV_REL = "results/scale_analysis.csv"


def main(repo_root: str = ".") -> None:
    apply()
    out  = os.path.join(repo_root, OUT_REL)
    csv  = os.path.join(repo_root, CSV_REL)

    if not os.path.exists(csv):
        placeholder(out,
                    xlabel="Cardinality N",
                    ylabel="Advisor time T_A(N)",
                    title="Scale analysis",
                    kind="loglog")
        print(f"[placeholder] {out}  (csv not found: {csv})")
        return

    import pandas as pd
    df = pd.read_csv(csv)
    fig, ax = plt.subplots(figsize=(3.4, 2.4))
    for i, model in enumerate(sorted(df["model"].unique())):
        sub = df[df["model"] == model].sort_values("N")
        ax.loglog(sub["N"], sub["advisor_time_s"], marker="o",
                  color=PALETTE[i % len(PALETTE)], label=model)
    ax.set_xlabel("Cardinality N")
    ax.set_ylabel("Advisor time T_A(N)")
    ax.legend(loc="best")
    fig.savefig(out)
    plt.close(fig)
    print(f"[real]        {out}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else ".")
