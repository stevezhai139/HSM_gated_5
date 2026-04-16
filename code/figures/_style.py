"""Common matplotlib style for HSM v2 figures."""

import matplotlib as mpl

# IEEE-friendly style: serif, small fonts, tight margins.
RC: dict = {
    "figure.dpi":      150,
    "savefig.dpi":     300,
    "savefig.bbox":    "tight",
    "savefig.format":  "pdf",
    "font.family":     "serif",
    "font.serif":      ["Times New Roman", "DejaVu Serif"],
    "font.size":       8,
    "axes.labelsize":  8,
    "axes.titlesize":  9,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "axes.linewidth":  0.6,
    "lines.linewidth": 1.2,
    "lines.markersize": 4,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":         True,
    "grid.alpha":        0.25,
    "grid.linewidth":    0.4,
    "legend.frameon":    False,
}

# Colour-blind safe palette, ordered for grouped bar / line plots.
PALETTE = [
    "#0072B2",  # blue
    "#D55E00",  # vermillion
    "#009E73",  # bluish green
    "#CC79A7",  # purple
    "#E69F00",  # orange
    "#56B4E9",  # sky blue
    "#F0E442",  # yellow
]

WATERMARK_KW = dict(
    s="PLACEHOLDER",
    color="#888888",
    alpha=0.18,
    fontsize=42,
    rotation=30,
    ha="center",
    va="center",
    fontfamily="sans-serif",
    fontweight="bold",
)


def apply():
    """Apply the HSM matplotlib style globally."""
    mpl.rcParams.update(RC)
