"""
plot_fig07_throughput_comparison.py
------------------------------------
End-to-end throughput comparison across DBA-advisor configurations,
normalized within-block by the no_advise baseline (paired ratio).

Reads results/throughput.csv with columns
    sf, config, qps_raw, qps_norm, ci_low_norm, ci_high_norm, n

Produces a grouped-bar chart: 3 advisor configs (periodic, always_on,
HSM_gated) × all evaluated SFs.  A horizontal dashed line at y=1.0
marks the no_advise reference (each bar is the fraction of no_advise
throughput retained at that SF).
"""

from __future__ import annotations

import os
import sys

import matplotlib.pyplot as plt
import numpy as np

from _placeholder import placeholder
from _style import PALETTE, apply

OUT_REL = "paper/main/figures/fig07_throughput_comparison.pdf"
CSV_REL = "results/throughput.csv"

# Conditions to plot (baseline omitted — used only as the y=1 reference).
PLOT_ORDER = ["periodic", "always_on", "HSM_gated"]


def main(repo_root: str = ".") -> None:
    apply()
    out = os.path.join(repo_root, OUT_REL)
    csv = os.path.join(repo_root, CSV_REL)

    if not os.path.exists(csv):
        placeholder(out,
                    xlabel="Scale factor (SF)",
                    ylabel="Throughput / no_advise",
                    title="Throughput comparison (10-block RCB)",
                    kind="bar")
        print(f"[placeholder] {out}  (csv not found: {csv})")
        return

    import pandas as pd
    df = pd.read_csv(csv)

    # Backward-compat: support old single-axis schema by falling through.
    required = {"sf", "config", "qps_norm", "ci_low_norm", "ci_high_norm"}
    if not required.issubset(df.columns):
        placeholder(out,
                    xlabel="Scale factor (SF)",
                    ylabel="Throughput / no_advise",
                    title="Throughput comparison (10-block RCB)",
                    kind="bar")
        print(f"[placeholder] {out}  (csv missing columns: "
              f"{required - set(df.columns)})")
        return

    sfs = sorted(df["sf"].unique())
    n_groups = len(sfs)
    n_bars   = len(PLOT_ORDER)

    # Bar geometry.
    group_width = 0.8
    bar_width   = group_width / n_bars
    x_centers   = np.arange(n_groups)
    bar_offsets = (np.arange(n_bars) - (n_bars - 1) / 2) * bar_width

    # Map each plotted condition to a fixed PALETTE slot for visual
    # continuity with the legacy figure (skip baseline at index 0).
    cond_color = {cond: PALETTE[i + 1] for i, cond in enumerate(PLOT_ORDER)}

    fig, ax = plt.subplots(figsize=(3.6, 2.4))

    # Reference line at 1.0 (no_advise).
    ax.axhline(1.0, color="#444444", linestyle="--", linewidth=0.7,
               zorder=0)
    ax.text(n_groups - 0.5, 1.02, "no_advise",
            fontsize=6, ha="right", va="bottom", color="#444444")

    for i, cond in enumerate(PLOT_ORDER):
        rows = df[df["config"] == cond].set_index("sf").reindex(sfs)
        means = rows["qps_norm"].to_numpy(dtype=float)
        lo    = rows["ci_low_norm"].to_numpy(dtype=float)
        hi    = rows["ci_high_norm"].to_numpy(dtype=float)
        yerr  = np.vstack([means - lo, hi - means])
        ax.bar(
            x_centers + bar_offsets[i],
            means,
            width=bar_width,
            yerr=yerr,
            color=cond_color[cond],
            label=cond,
            error_kw={"linewidth": 0.8, "capsize": 2.0},
        )

    ax.set_xticks(x_centers)
    ax.set_xticklabels([f"SF={sf:g}" for sf in sfs])
    ax.set_xlabel("Scale factor")
    ax.set_ylabel("Wall-clock QPS / no_advise")
    ax.set_ylim(0, 1.15)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.grid(axis="y", linewidth=0.3, alpha=0.5, zorder=0)
    ax.legend(loc="upper right", frameon=False, fontsize=6,
              handlelength=1.2, columnspacing=0.8, ncol=3,
              bbox_to_anchor=(1.0, 1.18))

    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    print(f"[real]        {out}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else ".")
