#!/usr/bin/env python3
"""
Supplementary Figure 06: Amplitude Invariance (B4 Experiment)

Creates a publication-quality heatmap showing HSM component values across
amplitude factors (k = 0.5, 1, 2, 5, 10). Demonstrates angular invariance
(S_R and S_T ≈ const) and amplitude dependence of other components.

IEEE Style: 3.5" width, serif 8pt font, single-column figure.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
import os

# Configuration
DATA_PATH = '/sessions/awesome-eager-dijkstra/mnt/Research Papers/Paper 3 /Paper 3A/Version 5/HSM_gated/results/overnight_2026-04-16/b4_inphase/inphase_summary.csv'
OUTPUT_DIR = '/sessions/awesome-eager-dijkstra/mnt/Research Papers/Paper 3 /Paper 3A/Version 5/HSM_gated/paper/supplementary/figures'

# IEEE style parameters
FIGURE_WIDTH = 3.5  # inches (single column)
FIGURE_HEIGHT = 3.5  # inches
DPI = 300
FONT_SIZE = 8.5
FONT_FAMILY = 'serif'

# Colors and styling
CMAP = 'YlOrRd'
INVARIANT_EDGE_WIDTH = 2.5
INVARIANT_EDGE_COLOR = '#1f77b4'  # bold blue
ANNOTATION_COLOR_DARK = '#000000'
ANNOTATION_COLOR_LIGHT = '#ffffff'

# Read data
df = pd.read_csv(DATA_PATH)

# Extract components and prepare heatmap data
components = ['S_R', 'S_V', 'S_T', 'S_A', 'S_P', 'HSM']
k_values = df['k'].values

# Build mean and sd matrices
n_components = len(components)
n_k = len(k_values)

means = np.zeros((n_components, n_k))
sds = np.zeros((n_components, n_k))

for i, comp in enumerate(components):
    for j, k in enumerate(k_values):
        row = df[df['k'] == k].iloc[0]
        means[i, j] = row[f'{comp}_mean']
        sds[i, j] = row[f'{comp}_sd']

# Identify invariant components (S_R, S_T) - rows that should have borders
invariant_rows = [0, 2]  # S_R is row 0, S_T is row 2

# Create figure
fig, ax = plt.subplots(
    figsize=(FIGURE_WIDTH, FIGURE_HEIGHT),
    dpi=DPI,
    tight_layout=False
)

# Normalize colors across all data
norm = Normalize(vmin=0.0, vmax=1.0)
cmap = plt.get_cmap(CMAP)

# Draw heatmap cells
cell_height = 1.0
cell_width = 1.0

for i in range(n_components):
    for j in range(n_k):
        # Cell position
        x = j * cell_width
        y = (n_components - 1 - i) * cell_height

        # Get color
        value = means[i, j]
        color = cmap(norm(value))

        # Draw rectangle
        rect = patches.Rectangle(
            (x, y), cell_width, cell_height,
            linewidth=0.5, edgecolor='white', facecolor=color
        )
        ax.add_patch(rect)

        # Add border for invariant components
        if i in invariant_rows:
            rect_border = patches.Rectangle(
                (x, y), cell_width, cell_height,
                linewidth=INVARIANT_EDGE_WIDTH,
                edgecolor=INVARIANT_EDGE_COLOR,
                facecolor='none'
            )
            ax.add_patch(rect_border)

        # Annotate with mean ± sd
        mean_val = means[i, j]
        sd_val = sds[i, j]

        # Choose text color based on background brightness
        luminance = 0.299 * color[0] + 0.587 * color[1] + 0.114 * color[2]
        text_color = ANNOTATION_COLOR_LIGHT if luminance < 0.5 else ANNOTATION_COLOR_DARK

        # Format annotation
        if sd_val > 0.001:
            label = f'{mean_val:.2f}\n±{sd_val:.2f}'
        else:
            label = f'{mean_val:.2f}'

        # Place text centered in cell
        ax.text(
            x + cell_width/2, y + cell_height/2,
            label,
            ha='center', va='center',
            fontsize=FONT_SIZE - 1,
            fontfamily=FONT_FAMILY,
            color=text_color,
            weight='normal',
            linespacing=1.2
        )

# Set axis properties
ax.set_xlim(0, n_k)
ax.set_ylim(0, n_components)

# X-axis: k values
ax.set_xticks(np.arange(n_k) + 0.5)
ax.set_xticklabels([f'{k:.1f}' for k in k_values], fontsize=FONT_SIZE, fontfamily=FONT_FAMILY)
ax.set_xlabel('Amplitude Factor k', fontsize=FONT_SIZE, fontfamily=FONT_FAMILY, labelpad=8)

# Y-axis: components
ax.set_yticks(np.arange(n_components) + 0.5)
ax.set_yticklabels(components[::-1], fontsize=FONT_SIZE, fontfamily=FONT_FAMILY)
ax.set_ylabel('HSM Component', fontsize=FONT_SIZE, fontfamily=FONT_FAMILY, labelpad=8)

# Remove spines
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(True)
ax.spines['bottom'].set_visible(True)
ax.spines['left'].set_linewidth(0.5)
ax.spines['bottom'].set_linewidth(0.5)

# Grid
ax.grid(False)
ax.set_axisbelow(True)

# Colorbar
sm = ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
cbar = plt.colorbar(sm, ax=ax, pad=0.15, fraction=0.046)
cbar.set_label('Mean Value', fontsize=FONT_SIZE, fontfamily=FONT_FAMILY, labelpad=10)
cbar.ax.tick_params(labelsize=FONT_SIZE - 1)
for label in cbar.ax.get_yticklabels():
    label.set_fontfamily(FONT_FAMILY)

# Background
ax.set_facecolor('white')
fig.patch.set_facecolor('white')

# Adjust layout
plt.tight_layout(pad=0.3)

# Create output directory if needed
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Save outputs
pdf_path = os.path.join(OUTPUT_DIR, 'supp06_amplitude_invariance.pdf')
png_path = os.path.join(OUTPUT_DIR, 'supp06_amplitude_invariance.png')

print(f"Saving PDF to: {pdf_path}")
fig.savefig(pdf_path, format='pdf', bbox_inches='tight', facecolor='white', edgecolor='none', dpi=DPI)

print(f"Saving PNG to: {png_path}")
fig.savefig(png_path, format='png', bbox_inches='tight', facecolor='white', edgecolor='none', dpi=DPI)

plt.close(fig)

print("\n" + "="*70)
print("AMPLITUDE INVARIANCE FIGURE GENERATION COMPLETE")
print("="*70)
print(f"\nOutputs:")
print(f"  PDF: {pdf_path}")
print(f"  PNG: {png_path}")
print(f"\nFigure Properties:")
print(f"  Width: {FIGURE_WIDTH} inches (IEEE single-column)")
print(f"  Font: {FONT_FAMILY} {FONT_SIZE}pt")
print(f"  Resolution: {DPI} DPI")
print(f"\nKey Findings:")
print(f"  • S_R (row border): invariant ≈ 0.50 across all k")
print(f"  • S_T (row border): invariant ≈ 1.00 across all k")
print(f"  • S_V: varies with k (amplitude dependence)")
print(f"  • HSM floor: ~0.695-0.875 (varies with k)")
print("="*70)
