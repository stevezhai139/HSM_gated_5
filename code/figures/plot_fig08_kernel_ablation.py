#!/usr/bin/env python3
"""
Publication-quality figure for TKDE paper: Kernel Ablation Results
Creates a grouped bar chart comparing three metrics across three kernel types.

Figure specifications:
- Width: 3.5 inches (single column IEEE TKDE)
- Font: Times New Roman, 8-9pt
- Colors: Colorblind-friendly palette (blue, orange, green)
- Output: PDF and PNG
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Read the data
csv_path = "/sessions/awesome-eager-dijkstra/mnt/Research Papers/Paper 3 /Paper 3A/Version 5/HSM_gated/results/overnight_2026-04-16/b2_kernel_ablation/kernel_ablation_summary.csv"
df = pd.read_csv(csv_path)

# Extract unique kernels and metrics
kernels = df['kernel'].unique()
metrics = ['theta_star', 'J_star', 'AUC']
metric_labels = ['θ*', 'J*', 'AUC']

# Prepare data for grouped bar chart
data_dict = {kernel: [df[df['kernel'] == kernel][metric].values[0] for metric in metrics]
             for kernel in kernels}

# Colorblind-friendly palette: blue, orange, green
colors = ['#1f77b4', '#ff7f0e', '#2ca02c']  # Blue, Orange, Green (Paul Tol palette)

# Set up the figure with IEEE TKDE specifications
fig_width = 3.5  # inches
fig_height = 3.5 * 0.75  # aspect ratio for single column
dpi = 300

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif']
plt.rcParams['font.size'] = 8

fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=dpi)

# Set up positions for grouped bars
x = np.arange(len(kernels))
width = 0.25  # width of each bar

# Plot bars
bars = []
for idx, metric_label in enumerate(metric_labels):
    offset = (idx - 1) * width
    values = [data_dict[kernel][idx] for kernel in kernels]
    bar = ax.bar(x + offset, values, width, label=metric_label, color=colors[idx],
                 edgecolor='black', linewidth=0.5)
    bars.append(bar)

    # Add value labels on top of bars
    for b in bar:
        height = b.get_height()
        ax.text(b.get_x() + b.get_width()/2., height,
                f'{height:.3f}',
                ha='center', va='bottom', fontsize=7, fontweight='normal')

# Customize axes
ax.set_ylabel('Metric Value', fontsize=8, fontstyle='normal')
ax.set_xlabel('Kernel Type', fontsize=8, fontstyle='normal')
ax.set_xticks(x)
ax.set_xticklabels([k.capitalize() for k in kernels], fontsize=8)
ax.set_ylim(0.0, 1.0)
ax.tick_params(axis='y', labelsize=8)
ax.tick_params(axis='x', labelsize=8)

# Add horizontal grid lines only
ax.grid(axis='y', linestyle='-', alpha=0.3, linewidth=0.5, color='gray')
ax.set_axisbelow(True)

# Remove top and right spines
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_linewidth(0.5)
ax.spines['bottom'].set_linewidth(0.5)

# Add legend inside figure, top-right
ax.legend(loc='upper right', fontsize=8, frameon=True, fancybox=False,
          edgecolor='black', framealpha=0.95, labelspacing=0.3)

# Set white background
fig.patch.set_facecolor('white')
ax.set_facecolor('white')

# Tight layout to minimize whitespace
plt.tight_layout()

# Output paths
pdf_path = "/sessions/awesome-eager-dijkstra/mnt/Research Papers/Paper 3 /Paper 3A/Version 5/HSM_gated/paper/main/figures/fig08_kernel_ablation.pdf"
png_path = "/sessions/awesome-eager-dijkstra/mnt/Research Papers/Paper 3 /Paper 3A/Version 5/HSM_gated/paper/main/figures/fig08_kernel_ablation.png"

# Save as PDF and PNG
fig.savefig(pdf_path, format='pdf', bbox_inches='tight', dpi=dpi)
fig.savefig(png_path, format='png', bbox_inches='tight', dpi=dpi)

print(f"✓ PDF saved to: {pdf_path}")
print(f"✓ PNG saved to: {png_path}")

plt.close()
