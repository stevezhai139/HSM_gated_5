"""Plots for BO experiment results.

Hero plots per Deliverable 5 §4.2:

1. ``hypervolume_convergence_<track>.png``: mean + CI of HV vs BO iteration
   across 10 blocks per window size (shows BO convergence behaviour).
2. ``delta_f1_histogram.png``: ΔF1 = F1_BO − F1_W0 distribution per cell
   (easy vs hard side-by-side, 10 blocks per bar).
3. ``w_star_distribution.png``: per-block final w* on a 5-simplex projection
   (shows per-workload adjustment — the RQ4a hero claim).
4. ``pareto_frontier_overlay.png``: 10 blocks' final Pareto fronts
   overlaid per window (shows cross-block stability).

Usage:
    python code/experiments/cal/experiments/plot_bo_results.py \\
        --run-dir results/cal/experiments/<run_dir>
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

_THIS = Path(__file__).resolve()
_EXPERIMENTS_DIR = _THIS.parents[2]
if str(_EXPERIMENTS_DIR) not in sys.path:
    sys.path.insert(0, str(_EXPERIMENTS_DIR))


def _load_all_bo(run_dir: Path) -> List[Dict[str, Any]]:
    """Auto-detect all BO tracks: easy/, hard/ (TPC-H) or bo/ (SDSS real)."""
    out = []
    # TPC-H-style: track_dir / block_* / window_* / result.json
    for track in ("easy", "hard"):
        track_dir = run_dir / track
        if not track_dir.is_dir():
            continue
        for p in sorted(track_dir.glob("block_*/window_*/result.json")):
            with p.open() as fh:
                out.append(json.load(fh))
    # SDSS-style: bo/ block_* / result.json (no window dir)
    bo_dir = run_dir / "bo"
    if bo_dir.is_dir():
        for p in sorted(bo_dir.glob("block_*/result.json")):
            with p.open() as fh:
                row = json.load(fh)
                row.setdefault("track", "sdss_real")
                out.append(row)
    return out


def _setup_mpl():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def plot_hv_convergence(all_results: List[Dict[str, Any]], out_dir: Path) -> None:
    plt = _setup_mpl()
    tracks = sorted({r["track"] for r in all_results})
    for track in tracks:
        rows = [r for r in all_results if r["track"] == track]
        windows = sorted({r["window"] for r in rows})
        fig, axes = plt.subplots(1, len(windows), figsize=(4.2 * len(windows), 3.8),
                                 sharey=True, squeeze=False)
        for j, win in enumerate(windows):
            ax = axes[0, j]
            win_rows = [r for r in rows if r["window"] == win]
            # Collect hypervolume series per block
            hv_series = []
            for r in win_rows:
                hv = [h["hypervolume"] for h in r["result"]["history"]]
                hv_series.append(hv)
            if not hv_series:
                continue
            # Align lengths (they should all be n_init + n_iter)
            n_steps = min(len(h) for h in hv_series)
            hv_array = np.array([h[:n_steps] for h in hv_series])  # (n_blocks, n_steps)
            x = np.arange(n_steps)
            mean = hv_array.mean(axis=0)
            std = hv_array.std(axis=0)
            ax.plot(x, mean, color="#1f77b4", linewidth=1.6, label="mean HV")
            ax.fill_between(x, mean - std, mean + std, alpha=0.2, color="#1f77b4",
                            label=f"± 1 SD across {hv_array.shape[0]} blocks")
            ax.set_title(f"window = {win}", fontsize=10)
            ax.set_xlabel("BO iteration (incl. Sobol init)")
            if j == 0:
                ax.set_ylabel("Hypervolume")
            ax.grid(True, alpha=0.3)
            ax.set_ylim(0, 1.05)
            ax.legend(loc="lower right", fontsize=8)
        fig.suptitle(f"HV convergence — track: {track}", fontsize=11)
        fig.tight_layout()
        p = out_dir / f"hypervolume_convergence_{track}.png"
        fig.savefig(p, dpi=150)
        plt.close(fig)


def plot_delta_f1_histogram(agg: Dict[str, Any], out_dir: Path) -> None:
    plt = _setup_mpl()
    cells = agg.get("cells", [])
    if not cells:
        return
    fig, ax = plt.subplots(figsize=(9, 4.2))
    labels = []
    deltas_list = []
    colors = []
    for c in cells:
        labels.append(f"{c['track']}/W={c['window']}")
        deltas_list.append(c["delta_f1_per_block"])
        colors.append("#2ca02c" if c["track"] == "easy" else "#d62728")

    positions = np.arange(len(labels))
    bp = ax.boxplot(deltas_list, positions=positions, patch_artist=True, widths=0.6)
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)

    # Overlay individual block points
    for i, (d, c) in enumerate(zip(deltas_list, colors)):
        xs = np.full(len(d), positions[i]) + np.random.uniform(-0.15, 0.15, size=len(d))
        ax.scatter(xs, d, c=c, alpha=0.7, s=18, zorder=3, edgecolors="black", linewidths=0.5)

    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel("ΔF1 = F1_BO − F1_W0")
    # Use the first cell's n_blocks for the title (all cells share the same n per experiment)
    n_blocks_per_cell = len(deltas_list[0]) if deltas_list else 0
    ax.set_title(
        f"Per-block BO-gain vs Paper 3A default W₀ ({n_blocks_per_cell} blocks per cell)"
    )
    ax.grid(True, alpha=0.3, axis="y")

    # Annotate FDR-significant cells. Use disjunctive reject (Wilcoxon OR
    # Permutation) so ties-robust Permutation signals are not hidden by
    # Wilcoxon's tie-dropping behaviour. Falls back to Wilcoxon-only for
    # backward compatibility with older aggregate.json files.
    # Compute global y_max across all cells so we can extend ylim safely;
    # otherwise small-range plots (e.g., SDSS ΔF1 ≈ 0.1) clip star annotation.
    global_y_max = max(max(d) for d in deltas_list) if deltas_list else 0.0
    global_y_min = min(min(d) for d in deltas_list) if deltas_list else 0.0
    star_pad = max(0.02, 0.04 * (global_y_max - global_y_min))
    for i, c in enumerate(cells):
        reject = c.get("rejects_any_fdr_005", c.get("rejects_h0_at_fdr_005"))
        if reject:
            y_max = max(c["delta_f1_per_block"])
            ax.annotate("★", (positions[i], y_max + star_pad),
                        ha="center", fontsize=14, color="#2ca02c")
    # Extend ylim to ensure stars and legend text are visible
    y_lo = global_y_min - star_pad
    y_hi = global_y_max + 3 * star_pad  # room for stars + legend above
    ax.set_ylim(y_lo, y_hi)
    ax.text(0.02, 0.98, "★ = BY-FDR q ≤ 0.05 (Wilcoxon or Permutation)",
            transform=ax.transAxes, fontsize=9, va="top")
    fig.tight_layout()
    fig.savefig(out_dir / "delta_f1_histogram.png", dpi=150)
    plt.close(fig)


def plot_w_star_distribution(all_results: List[Dict[str, Any]], out_dir: Path) -> None:
    """Per-block w* on parallel-coordinates plot (5 dims on x-axis, w value on y-axis).

    Shows whether BO converges to different w* per block (per-workload adjustment).
    """
    plt = _setup_mpl()
    # Group by (track, window)
    cells = {}
    for r in all_results:
        key = (r["track"], r["window"])
        cells.setdefault(key, []).append(r)

    if not cells:
        return

    n = len(cells)
    cols = min(n, 3)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 3.5 * rows),
                             squeeze=False)
    axes = axes.flatten()

    dim_names = ["w_R", "w_V", "w_T", "w_A", "w_P"]
    for idx, ((track, win), rs) in enumerate(sorted(cells.items())):
        ax = axes[idx]
        # Collect final w* per block
        w_stars = np.array([r["result"]["final_w_star"] for r in rs])  # (n_blocks, 5)
        if w_stars.size == 0:
            continue
        x = np.arange(5)
        for j, w in enumerate(w_stars):
            ax.plot(x, w, alpha=0.5, marker="o", markersize=5,
                    label=f"block {j}" if idx == 0 else None)
        # Paper 3A W₀ reference
        w0 = np.array([0.25, 0.20, 0.20, 0.20, 0.15])
        ax.plot(x, w0, color="black", linewidth=2.5, linestyle="--", marker="s",
                markersize=7, label="Paper 3A W₀" if idx == 0 else None, zorder=10)
        ax.set_xticks(x)
        ax.set_xticklabels(dim_names)
        ax.set_ylabel("weight")
        ax.set_title(f"{track} / window={win} — BO w* per block")
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-0.05, 1.0)
        if idx == 0:
            ax.legend(fontsize=7, loc="upper right", ncol=2)

    # Hide unused axes
    for k in range(n, len(axes)):
        axes[k].axis("off")

    fig.suptitle("Per-block final w* (per-workload adjustment evidence)", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_dir / "w_star_distribution.png", dpi=150)
    plt.close(fig)


def plot_pareto_overlay(all_results: List[Dict[str, Any]], out_dir: Path) -> None:
    """Overlay final Pareto fronts across 10 blocks per (track, window)."""
    plt = _setup_mpl()
    cells = {}
    for r in all_results:
        key = (r["track"], r["window"])
        cells.setdefault(key, []).append(r)

    if not cells:
        return

    n = len(cells)
    cols = min(n, 3)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4.5 * cols, 4.2 * rows),
                             squeeze=False)
    axes = axes.flatten()

    for idx, ((track, win), rs) in enumerate(sorted(cells.items())):
        ax = axes[idx]
        for j, r in enumerate(rs):
            pareto_idx = r["result"].get("pareto_indices", [])
            history = r["result"]["history"]
            if not pareto_idx:
                continue
            P = [history[i]["precision"] for i in pareto_idx]
            R = [history[i]["recall"] for i in pareto_idx]
            # Sort by P descending for clean line
            order = np.argsort([-p for p in P])
            P_sorted = [P[k] for k in order]
            R_sorted = [R[k] for k in order]
            ax.plot(P_sorted, R_sorted, marker="o", markersize=4, alpha=0.5, linewidth=1,
                    label=f"block {j}" if idx == 0 else None)
        ax.set_xlabel("Precision")
        ax.set_ylabel("Recall")
        ax.set_title(f"{track} / window={win}", fontsize=10)
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, alpha=0.3)
        if idx == 0:
            ax.legend(fontsize=7, loc="lower left", ncol=2)

    for k in range(n, len(axes)):
        axes[k].axis("off")

    fig.suptitle("Final Pareto fronts per block overlay", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_dir / "pareto_frontier_overlay.png", dpi=150)
    plt.close(fig)


def main(argv: Optional[List[str]] = None) -> int:
    argv = argv if argv is not None else sys.argv[1:]
    p = argparse.ArgumentParser(description="Plot BO experiment results")
    p.add_argument("--run-dir", required=True)
    args = p.parse_args(argv)

    run_dir = Path(args.run_dir)
    if not run_dir.is_absolute():
        run_dir = _THIS.parents[4] / run_dir

    plots_dir = run_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    all_results = _load_all_bo(run_dir)
    agg_path = run_dir / "aggregate.json"
    agg = {}
    if agg_path.is_file():
        with agg_path.open() as fh:
            agg = json.load(fh)

    print(f"[plot] {len(all_results)} BO result files loaded")

    if all_results:
        print("[plot] hypervolume_convergence_<track>.png...")
        plot_hv_convergence(all_results, plots_dir)
        print("[plot] w_star_distribution.png...")
        plot_w_star_distribution(all_results, plots_dir)
        print("[plot] pareto_frontier_overlay.png...")
        plot_pareto_overlay(all_results, plots_dir)

    if agg:
        print("[plot] delta_f1_histogram.png...")
        plot_delta_f1_histogram(agg, plots_dir)
    else:
        print("[plot] skip delta_f1_histogram (run aggregate_bo_results.py first)")

    print(f"[plot] done. Figures in {plots_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
