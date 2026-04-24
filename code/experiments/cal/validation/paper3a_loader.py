"""Read Paper 3A's stored trigger-timeseries CSVs — READ-ONLY.

Paper 3A's ``code/results/<experiment>/*_trigger_timeseries.csv`` files
are the authoritative per-pair HSM similarity scores at the default
θ = 0.75 gate, along with phase labels that define ground truth.

This loader NEVER modifies those files — only reads them. It produces
a normalised in-memory ``PairSeries`` object with:

- ``scores``: per-pair HSM similarity K (floats in [0, 1])
- ``phase_a``, ``phase_b``: phase label of the two windows being compared
- ``is_transition``: derived — True iff phase_a != phase_b
- ``default_triggered``: Paper 3A's gate decision at θ = 0.75 (from CSV)

The derived ``is_transition`` is our ground-truth label for θ-sweep
scoring: y_true = 1 iff is_transition else 0, under the convention
fixed in Paper3B_Cal_Theoretical_Foundations_v0 §4.1.

Usage::

    from cal.validation.paper3a_loader import load_experiment, list_experiments
    series = load_experiment("job_static")
    print(series.n_pairs, series.n_transitions)
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple


# Registry mapping logical experiment name → CSV path relative to repo root.
# Extend this when new Paper 3A results are added; do not add entries that
# modify Paper 3A files.
_CSV_REGISTRY: Dict[str, str] = {
    "job_static": "code/results/job_validation/job_hsm_static_trigger_timeseries.csv",
    "oltp_static": "code/results/oltp_validation/oltp_hsm_static_trigger_timeseries.csv",
    "oltp_execute": "code/results/oltp_validation/oltp_hsm_execute_trigger_timeseries.csv",
    "burst_v2": "code/results/burst_v2_validation/burst_v2_hsm_trigger_timeseries.csv",
    "sdss": "code/results/sdss_validation/sdss_hsm_trigger_timeseries.csv",
}


@dataclass(frozen=True)
class PairSeries:
    """A normalised adjacent-pair timeseries extracted from Paper 3A CSV."""

    experiment: str
    csv_path: str
    window_idx: Tuple[int, ...]
    scores: Tuple[float, ...]
    default_triggered: Tuple[int, ...]
    phase_a: Tuple[str, ...]
    phase_b: Tuple[str, ...]
    is_transition: Tuple[int, ...]  # 1 iff phase_a != phase_b
    phases_seen: Tuple[str, ...] = field(default=())

    @property
    def n_pairs(self) -> int:
        return len(self.scores)

    @property
    def n_transitions(self) -> int:
        return int(sum(self.is_transition))

    @property
    def paper3a_default_trigger_count(self) -> int:
        return int(sum(self.default_triggered))

    def summary(self) -> Dict[str, object]:
        """One-line summary for the experiment log."""
        return {
            "experiment": self.experiment,
            "n_pairs": self.n_pairs,
            "n_transitions": self.n_transitions,
            "paper3a_default_trigger_count": self.paper3a_default_trigger_count,
            "score_min": min(self.scores),
            "score_max": max(self.scores),
            "score_mean": sum(self.scores) / len(self.scores),
            "phases_seen": list(self.phases_seen),
        }


def list_experiments() -> List[str]:
    """Return the registered experiment names."""
    return sorted(_CSV_REGISTRY.keys())


def resolve_csv_path(experiment: str, repo_root: Path) -> Path:
    """Return absolute path to the CSV for a given experiment key."""
    if experiment not in _CSV_REGISTRY:
        raise KeyError(
            f"unknown experiment {experiment!r}; known: {list_experiments()}"
        )
    return repo_root / _CSV_REGISTRY[experiment]


def load_experiment(
    experiment: str,
    *,
    repo_root: Path | None = None,
) -> PairSeries:
    """Load a Paper 3A trigger timeseries into a ``PairSeries``.

    Parameters
    ----------
    experiment : one of ``list_experiments()``.
    repo_root : HSM_gated repo root; inferred from this file's location if omitted.
    """
    if repo_root is None:
        # this file lives at <root>/code/experiments/cal/validation/paper3a_loader.py
        repo_root = Path(__file__).resolve().parents[4]
    csv_path = resolve_csv_path(experiment, repo_root)
    if not csv_path.is_file():
        raise FileNotFoundError(f"Paper 3A CSV not found: {csv_path}")

    window_idx: List[int] = []
    scores: List[float] = []
    trig: List[int] = []
    pa: List[str] = []
    pb: List[str] = []
    phases: List[str] = []
    with csv_path.open("r", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        required_cols = {"window_idx", "score", "gate_triggered", "phase_a", "phase_b"}
        missing = required_cols - set(reader.fieldnames or ())
        if missing:
            raise ValueError(
                f"{csv_path.name} missing columns: {sorted(missing)}"
            )
        for row in reader:
            window_idx.append(int(row["window_idx"]))
            scores.append(float(row["score"]))
            trig.append(int(row["gate_triggered"]))
            pa.append(row["phase_a"])
            pb.append(row["phase_b"])
            for p in (row["phase_a"], row["phase_b"]):
                if p not in phases:
                    phases.append(p)
    is_trans = tuple(1 if a != b else 0 for a, b in zip(pa, pb))
    return PairSeries(
        experiment=experiment,
        csv_path=str(csv_path),
        window_idx=tuple(window_idx),
        scores=tuple(scores),
        default_triggered=tuple(trig),
        phase_a=tuple(pa),
        phase_b=tuple(pb),
        is_transition=is_trans,
        phases_seen=tuple(phases),
    )


__all__ = ["PairSeries", "list_experiments", "resolve_csv_path", "load_experiment"]
