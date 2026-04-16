"""Placeholder PDF generator (option b: real-looking axes + watermark).

Usage:
    from _placeholder import placeholder
    placeholder("paper/main/figures/fig01_score_distribution.pdf",
                xlabel="HSM score", ylabel="Frequency",
                title="HSM composite score across window pairs",
                kind="hist")

The generated PDF has a publication-grade matplotlib axes with axis labels,
title, and a large grey diagonal "PLACEHOLDER" watermark across the plot
area, but no real data.  Replace by re-running the corresponding plot
script once v2 experimental results land in the results/ directory.
"""

from __future__ import annotations

import os
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np

from _style import PALETTE, WATERMARK_KW, apply


def _watermark(ax) -> None:
    ax.text(0.5, 0.5, transform=ax.transAxes, **WATERMARK_KW)


def placeholder(
    path: str,
    xlabel: str = "x",
    ylabel: str = "y",
    title: Optional[str] = None,
    kind: str = "line",
    figsize: tuple = (3.4, 2.4),
) -> None:
    """Create one placeholder PDF.

    kind:
        "line"    -- single empty line panel
        "hist"    -- histogram-shaped axes
        "bar"     -- grouped-bar layout
        "panels3" -- three-panel side-by-side
        "loglog"  -- log-log axes
        "box"     -- box-plot axes
        "scatter" -- empty scatter
    """
    apply()
    os.makedirs(os.path.dirname(path), exist_ok=True)

    if kind == "panels3":
        fig, axes = plt.subplots(1, 3, figsize=(figsize[0] * 1.8, figsize[1]))
        for ax, sub in zip(axes, ["A", "B", "C"]):
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            ax.set_title(f"({sub})")
            _watermark(ax)
        if title:
            fig.suptitle(title, y=1.02)
    else:
        fig, ax = plt.subplots(figsize=figsize)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        if title:
            ax.set_title(title)
        if kind == "loglog":
            ax.set_xscale("log")
            ax.set_yscale("log")
        # Synthetic axis range so the figure is not blank
        ax.set_xlim(0, 1) if kind != "loglog" else ax.set_xlim(1e1, 1e6)
        ax.set_ylim(0, 1) if kind != "loglog" else ax.set_ylim(1e-3, 1e1)
        _watermark(ax)

    fig.savefig(path)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Catalogue of all paper figures (single source of truth)
# ---------------------------------------------------------------------------
CATALOGUE = [
    # (output path relative to repo, xlabel, ylabel, title, kind)
    ("paper/main/figures/fig01_score_distribution.pdf",
     "HSM composite score", "Frequency",
     "Within-phase vs.\u00a0cross-phase score distribution", "hist"),
    ("paper/main/figures/fig02_trigger_timeseries.pdf",
     "Window index", "HSM score",
     "Adaptation trigger across windows", "line"),
    ("paper/main/figures/fig03_within_cross_phase.pdf",
     "HSM composite score", "Density",
     "TPC-H vs.\u00a0SDSS within/cross-phase", "hist"),
    ("paper/main/figures/fig04_overhead_panels.pdf",
     "Component", "Time (ms) / Footprint",
     "HSM overhead", "panels3"),
    ("paper/main/figures/fig05_noise_sensitivity.pdf",
     "Noise level", "Discrimination ratio (DR)",
     "Noise sensitivity (\u00b160% N1\u2013N5)", "line"),
    ("paper/main/figures/fig06_complexity_loglog.pdf",
     "Window size n_pts", "Time (s)",
     "Linear-time complexity verification", "loglog"),
    ("paper/main/figures/fig07_throughput_comparison.pdf",
     "Configuration", "Wall-clock QPS",
     "Throughput comparison (10-block RCB)", "bar"),
    ("paper/supplementary/figures/supp01_scale_analysis.pdf",
     "Cardinality N", "Advisor time T_A(N)",
     "Scale analysis", "loglog"),
    ("paper/supplementary/figures/supp02_gate_sensitivity.pdf",
     "Threshold z", "Gate trigger fraction",
     "Gate sensitivity (z>3\u03c3)", "line"),
    ("paper/supplementary/figures/supp03_theta_calibration.pdf",
     "\u03b8 candidate", "Cost (mean \u00b1 95% CI)",
     "\u03b8 calibration curve", "line"),
]


def regenerate_all(repo_root: str = ".") -> None:
    """(Re)create every placeholder figure listed in CATALOGUE."""
    for rel, xl, yl, title, kind in CATALOGUE:
        out = os.path.join(repo_root, rel)
        placeholder(out, xlabel=xl, ylabel=yl, title=title, kind=kind)
        print(f"  wrote  {out}")


if __name__ == "__main__":
    import sys
    root = sys.argv[1] if len(sys.argv) > 1 else "."
    regenerate_all(root)
