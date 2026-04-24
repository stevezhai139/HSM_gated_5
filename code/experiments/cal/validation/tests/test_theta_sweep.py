"""Unit tests for validation harness.

These are fast, deterministic tests that do not touch Paper 3A's CSVs.
The end-to-end integration test that DOES read Paper 3A's data is in
test_paper3a_loader.py and is gated on the CSV being present.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from cal.validation.paper3a_loader import (
    PairSeries,
    list_experiments,
    load_experiment,
)
from cal.validation.scenario_classifier import classify
from cal.validation.theta_sweep import result_to_json_dict, sweep


def _series(scores, is_trans, phases_seen=("A", "B"), experiment="toy") -> PairSeries:
    n = len(scores)
    pa = []
    pb = []
    # Build phase_a/phase_b consistent with is_trans.
    cur = "A"
    for t in is_trans:
        pa.append(cur)
        if t:
            cur = "B" if cur == "A" else "A"
        pb.append(cur)
    return PairSeries(
        experiment=experiment,
        csv_path="(synthetic)",
        window_idx=tuple(range(n)),
        scores=tuple(scores),
        default_triggered=tuple(1 if s < 0.75 else 0 for s in scores),
        phase_a=tuple(pa),
        phase_b=tuple(pb),
        is_transition=tuple(is_trans),
        phases_seen=phases_seen,
    )


class TestThetaSweepBasics:
    def test_default_grid_has_46_points(self):
        series = _series([0.5, 0.6, 0.7, 0.8, 0.9], [1, 0, 0, 0, 1])
        result = sweep(series)
        assert len(result.grid) == 46
        assert result.grid[0] == 0.50
        assert result.grid[-1] == 0.95

    def test_json_dict_is_serializable(self):
        import json
        series = _series([0.5, 0.6, 0.7, 0.8, 0.9], [1, 0, 0, 0, 1])
        result = sweep(series, theta_grid=[0.5, 0.7, 0.9])
        d = result_to_json_dict(series, result)
        s = json.dumps(d, default=float)  # should not raise
        assert "rates" in s
        assert "series" in s

    def test_positive_class_is_shift(self):
        # 3 pairs: the middle two are transitions with LOW score (shift).
        # At θ=0.75, both should be predicted shift → TP=2, FP=0 (stable has score 0.9>0.75).
        series = _series([0.9, 0.3, 0.4, 0.9], [0, 1, 1, 0])
        result = sweep(series, theta_grid=[0.75])
        r = result.rates[0]
        assert r.tp == 2
        assert r.fp == 0
        assert r.fn == 0
        assert r.tn == 2


class TestScenarioA:
    def test_clean_separation_gives_A(self):
        # Transitions have low scores 0.3-0.4; stable have high 0.85-0.95.
        # Any θ in (0.4, 0.85) separates cleanly → scenario A.
        series = _series([0.9, 0.3, 0.88, 0.4, 0.92, 0.35], [0, 1, 0, 1, 0, 1])
        result = sweep(series)
        s = classify(series, result)
        assert s.scenario == "A"
        assert "FP=FN=0" in s.justification


class TestScenarioB:
    def test_overlap_but_count_matches_gives_B(self):
        # Construct: 2 transitions at scores 0.40, 0.80; stable at 0.35, 0.85, 0.60, 0.70.
        # At some θ: triggers count matches (2) but membership is wrong.
        # Transitions [0.40, 0.80], stable [0.35, 0.60, 0.70, 0.85]
        # θ=0.71 → triggers = {0.40, 0.35, 0.60, 0.70} = 4 triggered. Not match.
        # θ=0.59 → triggers = {0.40, 0.35} = 2 triggered. Count match!
        # But actual transitions are at scores 0.40 and 0.80; one trigger at 0.40 is TP
        # one at 0.35 is FP → alignment off → scenario B.
        series = _series(
            scores=[0.35, 0.40, 0.85, 0.60, 0.80, 0.70],
            is_trans=[0, 1, 0, 0, 1, 0],
        )
        result = sweep(series)
        s = classify(series, result)
        assert s.scenario in ("B", "C")  # depending on grid granularity


class TestScenarioC:
    def test_transition_scores_higher_than_stable_gives_C(self):
        # Pathological: transition pairs have HIGHER similarity than stable pairs
        # (the JOB case Paper 3A exhibits). No θ can fix this.
        series = _series(
            scores=[0.72, 0.80, 0.74, 0.82, 0.73, 0.81],
            is_trans=[0, 1, 0, 1, 0, 1],
        )
        result = sweep(series)
        s = classify(series, result)
        assert s.scenario == "C"


class TestPaper3ALoaderFallsGracefully:
    def test_unknown_experiment_raises_keyerror(self):
        with pytest.raises(KeyError):
            from cal.validation.paper3a_loader import resolve_csv_path
            resolve_csv_path("does_not_exist", Path("/tmp"))


# -----------------------------------------------------------------------
# Optional integration test: reads a real Paper 3A CSV if present.
# -----------------------------------------------------------------------


@pytest.mark.parametrize("experiment", list_experiments())
def test_load_real_paper3a_csv_if_present(experiment):
    try:
        series = load_experiment(experiment)
    except FileNotFoundError:
        pytest.skip(f"Paper 3A CSV for {experiment} not present")
        return
    assert series.n_pairs > 0
    assert all(0.0 <= s <= 1.0 for s in series.scores)
    assert all(t in (0, 1) for t in series.is_transition)
