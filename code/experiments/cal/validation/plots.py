"""Matplotlib figures for θ-sweep results.

Three figures per run:

1. ``scores_per_pair.png``: HSM K per pair-index with transition pairs
   highlighted (red markers) and θ reference lines drawn (default 0.75,
   θ* if distinct).
2. ``trigger_vs_theta.png``: trigger count as a function of θ with the
   n_transitions line drawn for reference.
3. ``f1_vs_theta.png``: F1, precision, recall as functions of θ with
   the θ* markers annotated.

Import policy: matplotlib is imported lazily inside each function so the
rest of the validation harness can run on machines without matplotlib.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from .paper3a_loader import PairSeries
from .theta_sweep import ThetaSweepResult


def _setup_axes(ax, title: str, xlabel: str, ylabel: str):
    ax.set_title(title, fontsize=11)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)


def plot_scores_per_pair(
    series: PairSeries,
    sweep_result: ThetaSweepResult,
    out_path: Path,
    *,
    default_theta: float = 0.75,
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 4.2))
    n = series.n_pairs
    xs = list(range(n))
    # Stable and transition points split for color coding.
    stable_x = [i for i, t in enumerate(series.is_transition) if t == 0]
    stable_y = [series.scores[i] for i in stable_x]
    trans_x = [i for i, t in enumerate(series.is_transition) if t == 1]
    trans_y = [series.scores[i] for i in trans_x]

    ax.plot(xs, series.scores, color="0.6", linewidth=0.8, alpha=0.6,
            label="HSM similarity K (all pairs)")
    ax.scatter(stable_x, stable_y, s=18, c="#1f77b4", label="stable pair", zorder=3)
    ax.scatter(trans_x, trans_y, s=50, c="#d62728", marker="D",
               label="phase transition", zorder=4, edgecolor="black", linewidths=0.5)

    # Reference θ lines
    ax.axhline(default_theta, color="orange", linestyle=":", linewidth=1.2,
               label=f"Paper 3A default θ={default_theta}")
    best_f1 = sweep_result.theta_star_rates.get("max_f1")
    if best_f1 is not None and abs(best_f1.theta - default_theta) > 0.005:
        ax.axhline(best_f1.theta, color="green", linestyle="--", linewidth=1.2,
                   label=f"θ* max-F1 = {best_f1.theta:.2f}")
    _setup_axes(ax, f"HSM similarity per pair — {series.experiment}",
                "pair index (adjacent window pair)",
                "HSM similarity K ∈ [0, 1]")
    ax.set_ylim(0.0, 1.0)
    ax.legend(loc="lower left", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_trigger_vs_theta(
    series: PairSeries,
    sweep_result: ThetaSweepResult,
    out_path: Path,
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 4.2))
    thetas = [r.theta for r in sweep_result.rates]
    n_trig = [r.n_triggered for r in sweep_result.rates]

    ax.plot(thetas, n_trig, color="#1f77b4", linewidth=1.6, marker="o",
            markersize=3, label="triggers at θ")
    ax.axhline(series.n_transitions, color="#d62728", linestyle="--",
               label=f"n_transitions = {series.n_transitions}")
    ax.axvline(0.75, color="orange", linestyle=":",
               label="Paper 3A default θ=0.75")
    best_f1 = sweep_result.theta_star_rates.get("max_f1")
    if best_f1 is not None:
        ax.axvline(best_f1.theta, color="green", linestyle="--",
                   label=f"θ* max-F1 = {best_f1.theta:.2f}")

    _setup_axes(ax, f"Trigger count vs θ — {series.experiment}",
                "threshold θ", "number of triggered pairs")
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_f1_vs_theta(
    series: PairSeries,
    sweep_result: ThetaSweepResult,
    out_path: Path,
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 4.2))
    thetas = [r.theta for r in sweep_result.rates]

    ax.plot(thetas, [r.f1 for r in sweep_result.rates],
            color="#2ca02c", linewidth=1.6, marker="o", markersize=3, label="F1")
    ax.plot(thetas, [r.precision for r in sweep_result.rates],
            color="#1f77b4", linewidth=1.0, linestyle="--", label="Precision")
    ax.plot(thetas, [r.recall for r in sweep_result.rates],
            color="#ff7f0e", linewidth=1.0, linestyle="--", label="Recall")
    ax.axvline(0.75, color="orange", linestyle=":",
               label="Paper 3A default θ=0.75")
    best_f1 = sweep_result.theta_star_rates.get("max_f1")
    if best_f1 is not None:
        ax.axvline(best_f1.theta, color="green", linestyle="--",
                   label=f"θ* max-F1 = {best_f1.theta:.2f}")

    _setup_axes(ax, f"F1 / Precision / Recall vs θ — {series.experiment}",
                "threshold θ", "score ∈ [0, 1]")
    ax.set_ylim(-0.02, 1.02)
    ax.legend(loc="lower left", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def save_all_plots(
    series: PairSeries,
    sweep_result: ThetaSweepResult,
    out_dir: Path,
) -> dict:
    """Save all three figures into ``out_dir``; return dict of filename → path."""
    out_dir.mkdir(parents=True, exist_ok=True)
    p_scores = out_dir / "scores_per_pair.png"
    p_trig = out_dir / "trigger_vs_theta.png"
    p_f1 = out_dir / "f1_vs_theta.png"
    plot_scores_per_pair(series, sweep_result, p_scores)
    plot_trigger_vs_theta(series, sweep_result, p_trig)
    plot_f1_vs_theta(series, sweep_result, p_f1)
    return {
        "scores_per_pair": str(p_scores),
        "trigger_vs_theta": str(p_trig),
        "f1_vs_theta": str(p_f1),
    }


__all__ = [
    "plot_scores_per_pair",
    "plot_trigger_vs_theta",
    "plot_f1_vs_theta",
    "save_all_plots",
]
