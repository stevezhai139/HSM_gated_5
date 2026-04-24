"""Paper 3B-Cal empirical validation harness (read-only over Paper 3A results).

This subtree loads Paper 3A's stored HSM pair-score timeseries, sweeps the
similarity threshold θ, and reports whether θ-manipulation alone can reduce
false positives under each experiment's ground-truth phase structure.

Integrity rules: only READ from Paper 3A artefacts under
``code/results/``; never modify. All outputs go to ``results/cal/validation/``
with provenance stamping (git SHA, timestamp, env). Intended to be run
manually by the researcher — no automated commits.
"""

__all__ = [
    "paper3a_loader",
    "theta_sweep",
    "scenario_classifier",
    "plots",
    "_run_meta",
]
