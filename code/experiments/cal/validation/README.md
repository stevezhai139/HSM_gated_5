# Paper 3B-Cal empirical validation harness

**Purpose.** Run a θ-sweep on Paper 3A's stored HSM similarity timeseries to
test whether θ-manipulation alone can reduce false positives under each
experiment's known phase-transition ground truth. Results feed back into
the v0.1 revision of the Theoretical Foundations + GT Refactor
deliverables before any Paper 3B code / docs are committed.

**Companion documents** (read before running):

- `Paper3B_Cal_Theoretical_Foundations_v0.docx` — §1, §4.1 (gate orientation), §5 (modern methods)
- `Paper3B_Cal_RQ5_GT_Refactor_Plan_v0.docx` — §3 (feature-extractor functional equivalence)
- `Paper3B_Cal_RQ1_Orientation_Analysis_v0.docx` — confirms y_pred = G directly

**Integrity rules respected.**

- Paper 3A CSVs under `code/results/` are READ-ONLY inputs; this harness never writes there
- All new code sits under `code/experiments/cal/validation/` (cal/ subtree per design doc §8 constraint)
- All experiment outputs go under `results/cal/validation/` and are timestamped + git-SHA stamped
- `SUBMISSION_MANIFEST.md` untouched; `v5.0.0-tkde-submission` tag untouched

---

## 1. Prerequisites

Python 3.10+ with:

```
numpy          (required)
scikit-learn   (required — pulled in by scenario classifier path dependencies)
matplotlib     (optional — used only for PNG figures; --no-plots skips)
pytest         (required for unit tests)
```

Install (user-local, no root):

```bash
pip install --user --break-system-packages numpy scikit-learn matplotlib pytest
```

Or in a venv:

```bash
python -m venv .venv && source .venv/bin/activate
pip install numpy scikit-learn matplotlib pytest
```

---

## 2. Directory layout

```
code/experiments/cal/validation/
├── README.md                          ← this file
├── __init__.py
├── _run_meta.py                       git SHA + timestamp + env provenance
├── paper3a_loader.py                  read Paper 3A trigger-timeseries CSVs
├── theta_sweep.py                     θ-sweep core (deterministic)
├── scenario_classifier.py             A / B / C classification logic
├── plots.py                           matplotlib figures (lazy-imported)
├── run_validation.py                  CLI driver
└── tests/
    ├── __init__.py
    └── test_theta_sweep.py            12 unit tests; runs in <1s

results/cal/validation/                (gitignored except EXPERIMENT_LOG.md)
├── EXPERIMENT_LOG.md                  append-only ledger, one row per run
└── <experiment>/
    └── <YYYYMMDD-HHMMSS>_<sha>_<exp>_theta_sweep_seed<N>/
        ├── result.json                full sweep + scenario classification
        ├── run_meta.json              git + env provenance (duplicated for audit)
        ├── scores_per_pair.png        K per pair with transition markers
        ├── trigger_vs_theta.png       trigger count as function of θ
        └── f1_vs_theta.png            F1/Precision/Recall as function of θ
```

The `results/cal/validation/` tree is created automatically on first run.

---

## 3. Available experiments (from Paper 3A stored data)

All five come from `code/results/<name>/*_trigger_timeseries.csv`:

| Key | Source experiment | Pairs | Transitions | Paper 3A triggers at θ=0.75 |
|---|---|---|---|---|
| `job_static` | JOB benchmark static analysis | 20 | 3 | 4 |
| `oltp_static` | OLTP schema phases (static) | 37 | 3 | 37 |
| `oltp_execute` | OLTP schema phases (execute) | 37 | 3 | 37 |
| `burst_v2` | Burst injection (v2) | 26 | 2 | 5 |
| `sdss` | SDSS SkyLog real query log | 4954 | 1334 | 3092 |

`job_static` is the closest match to your "3 expected, 4 triggered"
recollection. Start there.

---

## 4. Running the harness

### 4.1 Pre-run sanity: run the unit tests first

```bash
# cwd = HSM_gated/code/experiments/
cd "<your repo root>/code/experiments"
python -m pytest cal/validation/tests/ -v
```

Expected: **12 passed in < 1 s**. If any fail, stop and investigate.

### 4.2 Run a single experiment

```bash
# cwd = HSM_gated/
cd "<your repo root>"
python code/experiments/cal/validation/run_validation.py \
    --experiment job_static \
    --seed 42
```

What you'll see in the terminal (example on current scaffold commit):

```
[run_validation] targets: ['job_static']
[run_validation] out_root: <repo>/results/cal/validation
[run_validation] θ-grid: [0.5, 0.95] step 0.01
[run_validation] experiment=job_static → <repo>/results/cal/validation/job_static/20260423-135034_85b2db6_job_static_theta_sweep_seed42
  loaded: n_pairs=20, n_transitions=3, phases=['actor', 'movie', 'production', 'keyword']
  paper3a_default_trigger_count=4
  scenario = C
  No θ in the sweep grid matches the transition count, and score ranges overlap: transitions ∈ [0.759, 0.883], stable ∈ [0.723, 0.893]. θ-manipulation alone cannot separate. Best F1 at θ=0.80 is 0.333 (TP=2, FP=7, FN=1).
  plots written: ['scores_per_pair', 'trigger_vs_theta', 'f1_vs_theta']
```

The run creates the output directory shown on the third line of the log.
Open the three PNGs to see the plots; open `result.json` for the full
per-θ breakdown.

### 4.3 Run all experiments in one batch

```bash
python code/experiments/cal/validation/run_validation.py \
    --experiment all \
    --seed 42
```

This will run `job_static`, `oltp_static`, `oltp_execute`, `burst_v2`,
`sdss` in sequence — each produces its own versioned output directory
and appends a row to `EXPERIMENT_LOG.md`.

Expected runtime on SDSS (4954 pairs × 46 θ values): ~30–60 s.
Others: under 5 s each.

### 4.4 Customising the θ-grid

```bash
python code/experiments/cal/validation/run_validation.py \
    --experiment job_static \
    --theta-min 0.60 --theta-max 0.90 --theta-step 0.005 \
    --seed 42
```

Smaller step → finer resolution on θ*. Default 0.01 is usually enough.

### 4.5 Skipping plots (CI or headless environments without matplotlib)

```bash
python code/experiments/cal/validation/run_validation.py \
    --experiment job_static --no-plots --seed 42
```

JSON still gets written; only PNGs are suppressed.

---

## 5. Interpreting the scenarios

Each run classifies the experiment into one of three scenarios. The
classification appears in the terminal, in `result.json` (top-level
`scenario.letter`), and in `EXPERIMENT_LOG.md`.

### Scenario A — θ alone can fix false positives

**Condition.** There exists θ ∈ [0.50, 0.95] such that `FP = FN = 0` —
all triggers align exactly with phase transitions.

**Interpretation.** Score ranges of transition and stable pairs are
fully separable. θ-manipulation is sufficient. Paper 3B's RQ3 Pareto
frontier analysis will find a clean trade-off.

**Paper 3B narrative effect.** Motivates the BO + qNEHVI offline
calibration (§2.1 / §5.1 of Theoretical Foundations) with concrete
"before/after" example on real data.

### Scenario B — θ count matches but alignment is imperfect

**Condition.** There exists θ where `n_triggered == n_transitions`, and
`TP ≥ 1`, but `FP > 0` or `FN > 0` — counts match, labels don't.

**Interpretation.** Score ranges overlap partially. θ alone is not
enough; weight calibration or multi-signal confirmation is needed.

**Paper 3B narrative effect.** Strengthens the motivation for the
six-dimensional BO (w calibration, not only θ). The paper can argue
"classical single-parameter tuning hits an empirical ceiling here; BO
over (w, θ) breaks through." RQ4a becomes the hero.

### Scenario C — θ alone cannot separate transitions from stable pairs

**Condition.** Either no θ yields count match, OR the θ that does match
has `TP = 0` (triggers hit only stable pairs — gate is inverted for
this workload).

**Interpretation.** HSM's five-dimensional kernel weights are miscalibrated
for this workload family, OR the "false positives" are legitimate
micro-shifts within a phase that HSM correctly detects but the
construction-based ground truth mislabels.

**Paper 3B narrative effect.** Two defensible paths:
1. **Reframe FPs as legitimate micro-shifts** — aligns with Paper 3A's
   math proof that HSM detects fine-grained behaviour. Adjust RQ1
   success criteria; success now means "HSM finds ALL labelled
   transitions AND additional real-but-unlabelled micro-shifts".
2. **Introduce multi-window confirmation** — require N consecutive
   sub-threshold windows before triggering. This is the F1 scenario
   already anticipated in RQ5.

Either path is a paper contribution, not a failure.

---

## 6. Version-control workflow (pre-commit review)

### 6.1 What to commit

After manual review of the first set of results, these files are
considered committable (subject to integrity-rules check):

```
code/experiments/cal/validation/
    __init__.py
    README.md                        (this file)
    _run_meta.py
    paper3a_loader.py
    theta_sweep.py
    scenario_classifier.py
    plots.py
    run_validation.py
    .gitignore
    tests/__init__.py
    tests/test_theta_sweep.py
results/cal/validation/
    EXPERIMENT_LOG.md                (append-only ledger ONLY)
```

### 6.2 What NOT to commit

```
results/cal/validation/<experiment>/**/*                (run artifacts)
                                                        — these can grow
                                                        into megabytes and
                                                        are regenerable
                                                        from result.json
```

To suppress artefacts from git tracking, add to the **repo-root**
`.gitignore` (NOT to `cal/validation/.gitignore`, which is
documentation-only):

```
# Paper 3B-Cal empirical validation — ignore regenerable run artefacts
results/cal/validation/**/*
!results/cal/validation/EXPERIMENT_LOG.md
!results/cal/validation/.gitkeep
```

(Alternative: keep `results/cal/validation/` untracked entirely until
definitive run day.)

### 6.3 Pre-commit check

```bash
git status    # confirm no file modified under v5.0.0-tkde-submission tag
git diff v5.0.0-tkde-submission -- code/experiments/cal/validation/
# Expected: all changes are ADDITIONS under cal/validation/.
```

### 6.4 Suggested commit message (after Steve approves)

```
Paper 3B-Cal: empirical validation harness (cal/validation/)

Adds a read-only θ-sweep harness over Paper 3A's stored
trigger-timeseries to test whether θ-manipulation alone reduces
false positives under known phase-transition ground truth.

- paper3a_loader.py: read-only CSV loader (5 experiments registered)
- theta_sweep.py: deterministic sweep over θ ∈ [0.50, 0.95]
- scenario_classifier.py: classifies each run into scenario A/B/C
- plots.py: matplotlib figures (lazy-imported; --no-plots skips)
- run_validation.py: CLI driver with provenance-stamped outputs
- tests/: 12 unit tests, run < 1 s

Integrity: cal/validation/ subtree only; no v5.0.0-tkde-submission
file touched; SUBMISSION_MANIFEST unchanged; paper3b-cal branch.

First run outcomes (see results/cal/validation/EXPERIMENT_LOG.md):
  - job_static: scenario C (transitions score higher than stable pairs)
  - oltp_*: [to be determined by Steve]
  - burst_v2: [to be determined by Steve]
  - sdss: [to be determined by Steve]
```

---

## 7. Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `FileNotFoundError: Paper 3A CSV not found` | CSV path moved or repo root mis-detected | Confirm you're running from the repo root, not inside `code/experiments/` |
| `ModuleNotFoundError: cal.validation` | sys.path missing the experiments dir | Use the CLI as shown (don't import as `code.experiments.cal.validation`) |
| Plots not generated | matplotlib missing | `pip install matplotlib` or pass `--no-plots` |
| `git sha` shows as `nosha` in output | Not a git working copy, or git not installed | Harmless; provenance will say `nosha`; timestamp and path still preserved |
| `git dirty = true` shown in log | Uncommitted changes in HSM_gated | Expected during development; commit changes before definitive runs |

---

## 8. What to send back after your run

When you've reviewed results, send me (in chat):

1. The terminal log output (copy-paste)
2. The `EXPERIMENT_LOG.md` content
3. Which experiment(s) you want to treat as the DEFINITIVE validation
4. Your call on scenario per experiment: do you accept my classifier's
   letter (A/B/C) or want to contest it based on the per-pair plot?

Once we agree on the scenario, I revise Deliverable 2 (v0 → v0.1) to
embed the empirical evidence, then proceed with the architecture
refactor (Deliverable 3).

---

*End of README.*
