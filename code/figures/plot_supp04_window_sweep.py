"""
plot_supp04_window_sweep.py
---------------------------
Window-size sweep: empirical validation of Theorem 6 (optimal $\\Nstar$)
and its Corollary (cross-scale consistency).

Two-panel figure:
    (a) normalised cost $\\mathcal{C}(N) = \\lambda\\,T_\\mathrm{norm}(N) +
        (1{-}\\lambda)\\,\\varepsilon(N)$ on log-$N$ axis; interior
        minimum marks $\\Nstar$.
    (b) $\\Nstar$ vs. data scale (events/phase); a flat curve supports
        the corollary.

Reads:
    results/window_sweep.csv  (N, DR, advisor_time_s, cost, workload)
    results/cross_scale.csv   (data_scale, N, DR, advisor_time_s, cost, N_star)
"""

from __future__ import annotations

import os
import sys

import matplotlib.pyplot as plt

from _placeholder import placeholder
from _style import PALETTE, apply

OUT_REL   = "paper/supplementary/figures/supp04_window_sweep.pdf"
CSV_PRIM  = "results/window_sweep.csv"
CSV_CROSS = "results/cross_scale.csv"


def main(repo_root: str = ".") -> None:
    apply()
    out   = os.path.join(repo_root, OUT_REL)
    csv_p = os.path.join(repo_root, CSV_PRIM)
    csv_c = os.path.join(repo_root, CSV_CROSS)

    if not os.path.exists(csv_p) or not os.path.exists(csv_c):
        placeholder(out,
                    xlabel="Window size N",
                    ylabel="Cost C(N)",
                    title="Window-size sweep",
                    kind="line")
        print(f"[placeholder] {out}  (missing csv)")
        return

    import numpy as np
    import pandas as pd

    df_p = pd.read_csv(csv_p).sort_values("N")
    df_c = pd.read_csv(csv_c).sort_values(["data_scale", "N"])

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(7.0, 2.6))

    # --- Panel (a): cost curve with interior minimum ------------------
    axL.plot(df_p["N"], df_p["cost"], marker="o", color=PALETTE[0],
             lw=1.2, label="cost C(N)")
    n_star = int(df_p.iloc[df_p["cost"].values.argmin()]["N"])
    c_star = float(df_p["cost"].min())
    axL.axvline(n_star, color=PALETTE[1], lw=0.9, ls="--",
                label=f"$N^*$ = {n_star}")
    axL.scatter([n_star], [c_star], s=28, color=PALETTE[1], zorder=4)
    axL.set_xscale("log")
    axL.set_xlabel("Window size $N$")
    axL.set_ylabel("Normalised cost $\\mathcal{C}(N)$")
    axL.set_title("(a) Interior optimum (Thm 6)")
    axL.legend(loc="best", fontsize=7)

    # --- Panel (b): N* invariance across scales -----------------------
    scales = sorted(df_c["data_scale"].unique())
    nstars = [int(df_c[df_c["data_scale"] == s]["N_star"].iloc[0])
              for s in scales]
    for i, s in enumerate(scales):
        sub = df_c[df_c["data_scale"] == s]
        axR.plot(sub["N"], sub["cost"], marker="o",
                 color=PALETTE[i % len(PALETTE)], lw=1.0,
                 label=f"scale {s:,}")
    # shared N* marker
    if len(set(nstars)) == 1:
        axR.axvline(nstars[0], color="gray", lw=0.9, ls="--",
                    label=f"$N^*$ = {nstars[0]} (all scales)")
    axR.set_xscale("log")
    axR.set_xlabel("Window size $N$")
    axR.set_ylabel("Normalised cost $\\mathcal{C}(N)$")
    axR.set_title("(b) Cross-scale consistency (Cor.)")
    axR.legend(loc="best", fontsize=7)

    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    print(f"[real]        {out}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else ".")
