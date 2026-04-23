# `code/experiments/cal/` — Paper 3B-Cal scaffolding

**Paper:** Paper 3B-Cal — *HSM-Cal: Adaptive Threshold and Weight Calibration for Workload-Similarity Gating in Index Maintenance — Validation on Real-World Logs*
**Status:** v0 scaffolding (2026-04-23). API stubs + test skeletons; no working experiments yet.
**Branch:** `paper3b-cal` (cut from `main` at `35b26ce`; `v5.0.0-tkde-submission` tag preserved at `0a2dbaa`).
**Canonical docs (outside this repo):**
- `../../../Paper 3 /Paper 3B/Paper3B_Cal_Design_v0.md` — design decisions, constraints, timeline
- `../../../Paper 3 /Paper 3B/Paper3B_Cal_RQs_v0.md` — formal RQ reformulation (this scaffold's blueprint)
- `../../../Paper 3 /Paper 3B/Paper3B_Cal_SOTA_v0.md` — SOTA landscape

## Integrity rules — do not violate

1. No file in this directory may modify any file at `git tag v5.0.0-tkde-submission`. All imports of Paper 3A code (HSM kernel, feature extraction, etc.) are read-only.
2. New directories under `code/experiments/cal/` only. Do not create siblings elsewhere.
3. Tests for this scaffold live under `code/experiments/cal/tests/`, not in the top-level `code/tests/` (which is Paper 3A's tree).
4. `SUBMISSION_MANIFEST.md` in the TKDE submission bundle remains APPEND-ONLY.

## Layout

```
cal/
├── README.md                  (this file)
├── __init__.py
├── gate.py                    Gate implementation G_{w,θ}
├── metrics.py                 FPR/FNR/F1, McNemar, Wilcoxon, KS helpers
├── pareto.py                  Pareto-efficient-set + knee finder
├── bo_calibrator.py           Offline Bayesian Optimisation (skopt / BoTorch qEHVI)
├── ema_tracker.py             Online EMA drift-tracker for θ
├── advisor_wrappers.py        Black-box wrappers for Dexter + Supabase index_advisor
├── drift_injection.py         Synthetic drift generators
├── failure_scenarios.py       RQ5 failure-scenario generators (F1, F2)
├── rq1_transfer.py            RQ1 driver — θ = 0.75 transferability
├── rq2_cost_benefit.py        RQ2 driver — cost-benefit across advisors × workloads
├── rq3_pareto.py              RQ3 driver — Pareto frontier under θ sweep
├── rq4_calibration.py         RQ4 driver — BO offline + EMA online
├── rq5_boundaries.py          RQ5 driver — W_min + failure modes
└── tests/
    ├── __init__.py
    ├── test_gate.py
    ├── test_metrics.py
    ├── test_pareto.py
    ├── test_ema_tracker.py
    └── test_bo_calibrator.py
```

## Module-to-RQ mapping

| Module | RQs served | Key symbols |
|---|---|---|
| `gate.py` | all | `Gate`, `Gate.decide` |
| `metrics.py` | RQ1, RQ2, RQ4b, RQ5 | `fpr_fnr`, `mcnemar_test`, `wilcoxon_paired`, `ks_test` |
| `pareto.py` | RQ3, RQ4a | `pareto_front`, `knee_point` |
| `bo_calibrator.py` | RQ4a | `BOCalibrator.run`, `CMAESCalibrator.run` |
| `ema_tracker.py` | RQ4b | `EMATracker.update`, `NoUpdateBaseline` |
| `advisor_wrappers.py` | RQ2, RQ3, RQ4 | `DexterWrapper`, `SupabaseIndexAdvisorWrapper` |
| `drift_injection.py` | RQ4b | `inject_mix_shift` |
| `failure_scenarios.py` | RQ5 | `micro_shift_scenario`, `syntactic_semantic_mismatch_scenario` |

## External dependencies (not yet added to `requirements.txt`)

These are the additional packages Paper 3B-Cal will need. Deliberately not
edited into the top-level `requirements.txt` yet — that file ships with the
TKDE submission and should not change without explicit approval. We'll add
a `requirements-cal.txt` when the first experiment actually runs.

```
scikit-optimize>=0.9          # offline BO (skopt.gp_minimize)
botorch>=0.9                  # qEHVI multi-objective acquisition (CPU mode)
cma>=3.3                      # CMA-ES secondary baseline
statsmodels>=0.14             # McNemar, KS, Wilcoxon
```

Existing `requirements.txt` already pins `numpy`, `scipy`, `pandas`,
`psycopg2-binary` (optional), `sqlparse` — all reusable.

## Running the experiments (future)

These entry points do not work yet — they exist as documented CLIs so we can
fill them in iteratively. Paper 3A convention runs scripts by path from the
repo root (not `python -m`), because `code` shadows the Python stdlib
`code` module:

```bash
# cwd = HSM_gated/
# Foundational — must run first (establishes W_min)
python code/experiments/cal/rq5_boundaries.py \
    --mode window_sweep --workload sdss_dr18 \
    --out results/cal/rq5_window_sweep_sdss.json

# Once W_min is known
python code/experiments/cal/rq1_transfer.py \
    --workload sdss_dr18 --w-min 20 \
    --out results/cal/rq1.json

# Serial — rq3 depends on rq2 infrastructure, rq4 on rq3
python code/experiments/cal/rq2_cost_benefit.py --advisor dexter --workload sdss_dr18 --w-min 20 --out results/cal/rq2.json
python code/experiments/cal/rq3_pareto.py       --advisor dexter --workload sdss_dr18 --w-min 20 --out results/cal/rq3.json
python code/experiments/cal/rq4_calibration.py  --mode offline --advisor dexter --workload sdss_dr18 --w-min 20 --out results/cal/rq4a.json
```

## Running the tests

```bash
# cwd = HSM_gated/code/experiments/
python -m pytest cal/tests/ -v
```

Current expected state (2026-04-23 scaffold):
- `test_gate.py` — 8 PASS (Gate logic is real, not stub)
- All others — XFAIL (implementations stubbed)
- 2 XPASS on trivial dataclass properties (expected)

## Timeline anchor

Per design doc §10:
- **May 2026** — SDSS DR18 access + RQ reformulation (RQ doc is done; access TBD)
- **Jun–Jul 2026** — fill in `bo_calibrator.py`, `ema_tracker.py`, run RQ1 + RQ4
- **Aug 2026** — RQ2 + RQ3 using existing `tier2/` advisor pipelines
- **Sep 2026** — RQ5 (boundary analysis); Stack Overflow if accessible
- **Oct–Dec 2026** — draft writing
- **Jan 2027** — submit DASFAA 2027

---

*End of cal/ README.*
