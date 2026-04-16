#!/usr/bin/env python3
"""
Supplementary Figure 7: Noise × Weight Robustness Grid (B5 Experiment)
Publication-quality heatmaps showing θ* and AUC across noise and weight perturbation.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from pathlib import Path

# Set up matplotlib for publication quality
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.size'] = 8
mpl.rcParams['axes.labelsize'] = 8
mpl.rcParams['axes.titlesize'] = 9
mpl.rcParams['xtick.labelsize'] = 8
mpl.rcParams['ytick.labelsize'] = 8
mpl.rcParams['legend.fontsize'] = 8
mpl.rcParams['figure.facecolor'] = 'white'
mpl.rcParams['axes.facecolor'] = 'white'

# Load data
data_path = Path('/sessions/awesome-eager-dijkstra/mnt/Research Papers/Paper 3 /Paper 3A/Version 5/HSM_gated/results/overnight_2026-04-16/b5_noise_weight/noise_weight_grid.csv')
df = pd.read_csv(data_path)

# Extract unique z (noise) and delta (weight) values
z_values = sorted(df['z'].unique())
delta_values = sorted(df['delta'].unique())

# Create 5x5 matrices for θ* and AUC
theta_star_matrix = np.zeros((len(z_values), len(delta_values)))
auc_matrix = np.zeros((len(z_values), len(delta_values)))

for i, z in enumerate(z_values):
    for j, delta in enumerate(delta_values):
        row = df[(df['z'] == z) & (df['delta'] == delta)]
        if not row.empty:
            theta_star_matrix[i, j] = row['theta_star'].values[0]
            auc_matrix[i, j] = row['AUC'].values[0]

# Create figure with 1x2 subplots
fig, axes = plt.subplots(1, 2, figsize=(7, 3.5))
fig.patch.set_facecolor('white')

# ===== LEFT PANEL: θ* heatmap =====
ax_theta = axes[0]
im_theta = ax_theta.imshow(theta_star_matrix, cmap='Blues', aspect='auto',
                           vmin=0.882, vmax=0.920, interpolation='nearest')

# Annotate cells with θ* values
for i in range(len(z_values)):
    for j in range(len(delta_values)):
        text = ax_theta.text(j, i, f'{theta_star_matrix[i, j]:.3f}',
                           ha="center", va="center", color="black", fontsize=7)

ax_theta.set_xticks(np.arange(len(delta_values)))
ax_theta.set_yticks(np.arange(len(z_values)))
ax_theta.set_xticklabels([f'{d:.3f}' for d in delta_values])
ax_theta.set_yticklabels([f'{z:.1f}' for z in z_values])
ax_theta.set_xlabel('Weight perturbation (δ)', fontsize=8)
ax_theta.set_ylabel('Noise (z)', fontsize=8)
ax_theta.set_title('(a)', fontsize=9, loc='left', pad=5)

# Colorbar for θ*
cbar_theta = plt.colorbar(im_theta, ax=ax_theta, label='θ*', fraction=0.046, pad=0.04)
cbar_theta.ax.tick_params(labelsize=7)

# ===== RIGHT PANEL: AUC heatmap =====
ax_auc = axes[1]
im_auc = ax_auc.imshow(auc_matrix, cmap='Greens', aspect='auto',
                       vmin=0.833, vmax=0.934, interpolation='nearest')

# Annotate cells with AUC values
for i in range(len(z_values)):
    for j in range(len(delta_values)):
        text = ax_auc.text(j, i, f'{auc_matrix[i, j]:.3f}',
                          ha="center", va="center", color="black", fontsize=7)

ax_auc.set_xticks(np.arange(len(delta_values)))
ax_auc.set_yticks(np.arange(len(z_values)))
ax_auc.set_xticklabels([f'{d:.3f}' for d in delta_values])
ax_auc.set_yticklabels([f'{z:.1f}' for z in z_values])
ax_auc.set_xlabel('Weight perturbation (δ)', fontsize=8)
ax_auc.set_ylabel('Noise (z)', fontsize=8)
ax_auc.set_title('(b)', fontsize=9, loc='left', pad=5)

# Colorbar for AUC
cbar_auc = plt.colorbar(im_auc, ax=ax_auc, label='AUC', fraction=0.046, pad=0.04)
cbar_auc.ax.tick_params(labelsize=7)

# Adjust layout
plt.tight_layout(pad=0.5)

# Save outputs
output_dir_pdf = Path('/sessions/awesome-eager-dijkstra/mnt/Research Papers/Paper 3 /Paper 3A/Version 5/HSM_gated/paper/supplementary/figures')
output_dir_pdf.mkdir(parents=True, exist_ok=True)

pdf_path = output_dir_pdf / 'supp07_noise_weight_grid.pdf'
png_path = output_dir_pdf / 'supp07_noise_weight_grid.png'

plt.savefig(pdf_path, dpi=300, bbox_inches='tight', facecolor='white')
plt.savefig(png_path, dpi=300, bbox_inches='tight', facecolor='white')

print(f"✓ PDF saved: {pdf_path}")
print(f"✓ PNG saved: {png_path}")
print(f"\nFigure summary:")
print(f"  θ* range: {theta_star_matrix.min():.4f} – {theta_star_matrix.max():.4f}")
print(f"  AUC range: {auc_matrix.min():.4f} – {auc_matrix.max():.4f}")
print(f"  Noise levels (z): {z_values}")
print(f"  Weight perturbations (δ): {delta_values}")

plt.close()
