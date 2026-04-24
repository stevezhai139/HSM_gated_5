"""Classify each experiment into scenario A / B / C after a θ-sweep.

Scenarios (as negotiated in the pre-approval review):

- **A (best)**: ∃ θ* ∈ grid such that triggers = n_transitions AND
  all triggered pairs are EXACTLY the transition pairs. θ-manipulation
  alone cleanly separates TP from FP. Strongest motivation for BO.

- **B (partial)**: ∃ θ* that gives triggers = n_transitions, but the
  triggered set does not match the transition set (i.e., at that θ
  some transition is missed and some non-transition fires). Counts
  match but labels don't. θ alone is not sufficient; weight
  calibration or multi-signal confirmation is needed.

- **C (worst)**: The ranges of transition-pair scores and
  non-transition-pair scores OVERLAP such that no θ can give count
  match with alignment. θ-only cannot separate. Either re-label FPs
  as legitimate micro-shifts (Paper 3A's math-proof interpretation)
  or redesign the gate.

This classifier returns the scenario letter, a short justification,
and the evidence (score ranges) that drove the decision.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from .paper3a_loader import PairSeries
from .theta_sweep import ThetaSweepResult


@dataclass(frozen=True)
class ScenarioClassification:
    scenario: str  # "A", "B", or "C"
    justification: str
    evidence: Dict[str, object] = field(default_factory=dict)


def classify(
    series: PairSeries,
    sweep_result: ThetaSweepResult,
) -> ScenarioClassification:
    """Classify the experiment into scenario A / B / C per the rubric above."""
    scores = list(series.scores)
    is_trans = list(series.is_transition)

    trans_scores = [s for s, t in zip(scores, is_trans) if t == 1]
    stable_scores = [s for s, t in zip(scores, is_trans) if t == 0]

    trans_min = min(trans_scores) if trans_scores else float("nan")
    trans_max = max(trans_scores) if trans_scores else float("nan")
    stable_min = min(stable_scores) if stable_scores else float("nan")
    stable_max = max(stable_scores) if stable_scores else float("nan")

    evidence: Dict[str, object] = {
        "n_transitions": series.n_transitions,
        "n_stable_pairs": series.n_pairs - series.n_transitions,
        "transition_score_range": [trans_min, trans_max],
        "stable_score_range": [stable_min, stable_max],
    }

    # Separability: strict if MAX of transition scores < MIN of stable scores
    # (lower scores = LESS similarity = SHIFT, so transitions should have
    # LOWER scores than stable pairs for θ-sweep to work).
    if trans_scores and stable_scores:
        separable = trans_max < stable_min
        evidence["separable"] = bool(separable)
        evidence["separation_margin"] = float(stable_min - trans_max)
    else:
        separable = False
        evidence["separable"] = False

    # Check exact-alignment scenario A: is there a θ where FP=FN=0?
    zero_error = [r for r in sweep_result.rates if r.fp == 0 and r.fn == 0]
    if zero_error:
        thetas = [r.theta for r in zero_error]
        evidence["zero_error_theta_range"] = [min(thetas), max(thetas)]
        return ScenarioClassification(
            scenario="A",
            justification=(
                f"There exists θ ∈ [{min(thetas):.2f}, {max(thetas):.2f}] "
                f"with FP=FN=0: triggers align exactly with the "
                f"{series.n_transitions} phase transitions."
            ),
            evidence=evidence,
        )

    # Exact count with SOME true positives → scenario B (partial alignment)
    exact_count = sweep_result.theta_star_rates.get("exact_trigger_count")
    if (
        exact_count is not None
        and (exact_count.fp > 0 or exact_count.fn > 0)
        and exact_count.tp >= 1
    ):
        return ScenarioClassification(
            scenario="B",
            justification=(
                f"At θ={exact_count.theta:.2f} trigger count matches the "
                f"{series.n_transitions} transitions, TP={exact_count.tp} "
                f"transitions detected correctly, but alignment is imperfect: "
                f"FP={exact_count.fp} and FN={exact_count.fn}. "
                f"Score ranges of transitions and stable pairs overlap "
                f"(transitions ∈ [{trans_min:.3f}, {trans_max:.3f}]; "
                f"stable ∈ [{stable_min:.3f}, {stable_max:.3f}])."
            ),
            evidence=evidence,
        )
    # Exact count but ZERO true positives → scenario C (pathological: gate fires
    # on stable pairs only, missing every transition). This happens when
    # transition scores are HIGHER than stable scores (gate inverted).
    if exact_count is not None and exact_count.tp == 0:
        return ScenarioClassification(
            scenario="C",
            justification=(
                f"At θ={exact_count.theta:.2f} trigger count happens to match "
                f"n_transitions={series.n_transitions}, but ZERO transitions "
                f"are correctly detected (TP=0, FP={exact_count.fp}, "
                f"FN={exact_count.fn}). Transition scores are HIGHER than "
                f"stable scores — the gate fires on stable pairs only. "
                f"θ-manipulation cannot separate; transitions ∈ "
                f"[{trans_min:.3f}, {trans_max:.3f}], stable ∈ "
                f"[{stable_min:.3f}, {stable_max:.3f}]."
            ),
            evidence=evidence,
        )

    # Otherwise scenario C — no θ gives count match at all
    best_f1 = sweep_result.theta_star_rates.get("max_f1")
    note = ""
    if best_f1 is not None:
        note = (
            f" Best F1 at θ={best_f1.theta:.2f} is "
            f"{best_f1.f1:.3f} "
            f"(TP={best_f1.tp}, FP={best_f1.fp}, FN={best_f1.fn})."
        )
    return ScenarioClassification(
        scenario="C",
        justification=(
            f"No θ in the sweep grid matches the transition count, and "
            f"score ranges overlap: transitions ∈ "
            f"[{trans_min:.3f}, {trans_max:.3f}], stable ∈ "
            f"[{stable_min:.3f}, {stable_max:.3f}]. θ-manipulation alone "
            f"cannot separate.{note}"
        ),
        evidence=evidence,
    )


__all__ = ["ScenarioClassification", "classify"]
