# HSM_gated — Changelog

All notable changes to this regeneration branch are documented here.
This branch forks from `Paper 3A/Version 4/` after a code-paper drift
audit exposed that the earlier implementation used a 5-Jaccard
composite instead of the Spearman / log-ratio / angular / dual-Jaccard
/ DWT+SAX+FastDTW pipeline defined in §III of the paper.

## [unreleased] — v2 regeneration

### Added (2026-04-14 figure-input aggregators)
- `scripts/build_summary_all.py`: consolidates the seven per-workload
  summary CSVs into a single paper-citable table at
  `code/results/summary_all.csv` with a canonical column order
  (DR_median / DR_mean / DR_CI_lo / DR_CI_hi / MWU_p_median /
  dominant_dim / delta_S_* ).  Chooses the empirical CI (emp) over the
  bootstrap CI when both variants exist because §VI quotes the empirical
  interval.
- `scripts/build_figure_csvs.py`: pools every per-workload pair-score
  file (reference seed) into
    - `code/results/score_distribution.csv` (columns: `score, group`) —
       consumed by `plot_fig01_score_distribution.py`.
    - `code/results/within_cross_phase.csv` (columns:
       `workload, group, score`) — consumed by
       `plot_fig03_within_cross_phase.py`; `job_execute` is relabelled
       `tpch` and `sdss` is kept verbatim to match the figure's
       two-workload view.
- `hsm_v2_kernel.dump_pair_scores_csv(path, within, cross, workload)`:
  helper that every validation script calls at the end of its run to
  persist the reference seed's raw pair scores.  The seed-0 reference
  matches the `within_mean` / `cross_mean` quoted in the per-workload
  summary CSV, so the paper figure and the paper table agree by
  construction.
- `run_all.sh` Phase 3b: invokes both aggregators between validation
  (Phase 3) and figure regeneration (Phase 4), logging any missing
  inputs as non-fatal warnings.

### Changed (2026-04-14 validation scripts emit raw pair scores)
- All seven validation scripts now write a second CSV next to their
  summary CSV:
    - `oltp_hsm_{static,execute}_pair_scores.csv`
    - `burst_hsm_pair_scores.csv`
    - `burst_v2_hsm_pair_scores.csv`
    - `burst_v3_hsm_pair_scores.csv`
    - `job_hsm_execute_pair_scores.csv`
    - `job_hsm_complexity_execute_pair_scores.csv`
    - `sdss_hsm_pair_scores.csv`
  The two burst scripts that previously dropped the pair-score arrays
  from `compute_dr`'s `info` dict now forward them through `seed_results`
  so the reference seed's distribution can be dumped verbatim.
- `hsm_sdss_validation.py` now also writes a summary CSV at
  `code/results/sdss_validation/sdss_hsm.csv` (previously stdout-only),
  unlocking SDSS inclusion in `summary_all.csv`.

### Changed (2026-04-14 S_P input paper-strict alignment)
- Every validation script now passes the **q(t) arrival-count series at
  1-second resolution** (paper §III-A line 291) to the S_P kernel, not
  the per-query latency profile.  Previously `times = [elapsed_ms_1,
  elapsed_ms_2, ...]` was fed into `sp_v2`, which made DWT/SAX/FastDTW
  measure the shape of the *latency distribution* rather than the
  *arrival intensity* that Theorem 2 (sufficiency) and Lemma 4 (DWT
  error bound) are stated over.
- `hsm_v2_kernel.build_qps_series(elapsed_ms, min_bins=16)` constructs
  the q(t) series by placing each query's arrival at its cumulative
  serial-finish time and histogramming arrivals into 1-second bins
  (padded to at least 16 bins so a level-3 db4 DWT is well-defined).
  A companion `arrivals_to_qps_series` variant accepts pre-recorded
  wall-clock arrival timestamps for traces that have them.
- All seven validation scripts replaced their `times_raw = [...]; np.tile(...)`
  latency-series construction with a single call to `build_qps_series`.

### Changed (2026-04-14 paper-strict kernel alignment)
- `hsm_similarity.py` is now a **thin wrapper** around `hsm_v2_kernel.py`.
  Previously this file was a parallel v2 implementation that drifted
  from the paper in two places:
    1. `S_V` used `WorkloadWindow.qps` (queries-per-second), but paper
       §III-B Relational extractor explicitly defines
       "Volume Q = query count per window".
       The wrapper now delegates to `hsm_v2_kernel.sv_v2(n_a, n_b)`
       which is count-based, matching Eq. 2 verbatim.
    2. `test_metric_properties.py` imported from `hsm_similarity`, so
       Lemma 1 was being verified against the legacy module rather than
       the kernel that the seven validation scripts actually call.
       The test suite now imports from `hsm_v2_kernel` directly and
       adds a `test_wrapper_matches_kernel` cross-check that fails if
       the two code paths ever diverge again.
- All five relational validation scripts (OLTP, SDSS, three burst
  scripts) **no longer pass `type_vec` to the kernel**.  Per paper
  §III-B Relational extractor, the type-similarity vocabulary is the
  "set of SQL query templates", so `S_T` uses the template-frequency
  vector aligned on the union vocabulary.  CRUD-tier counts
  (`crud_vec`, `tier_vec`) remain in feature dicts but are
  diagnostic-only and never reach the scoring kernel.  The kernel still
  exposes the `type_vec_*` parameters to support the document
  extractor (MongoDB pipeline categories) — the docstring now states
  this explicitly and warns that relational scripts must not use them.
- `_band_score` in `hsm_v2_kernel.py` short-circuits identical SAX
  encodings to score 1.0.  The `fastdtw` library's recursive coarsening
  can return non-zero distance on byte-identical sequences once they
  are long enough that the coarse path cannot reach all diagonal cells;
  true DTW(s, s) = 0 by definition, and Lemma 1 P2 (self-distance zero)
  requires this short circuit.
- Test suite expanded from 11 to 19 tests, all green.  New tests:
  `test_kernel_p1_non_negativity`, `test_kernel_p2_self_distance_zero`,
  `test_kernel_p3_symmetry`, `test_kernel_p4_triangle_inequality`,
  `test_kernel_score_in_unit_interval`,
  `test_kernel_default_weights_sum_to_one`,
  `test_kernel_s_v_count_based`, `test_wrapper_matches_kernel`.

### Added (2026-04-14 kernel consolidation)
- `code/experiments/hsm_v2_kernel.py`: canonical source-of-truth kernel
  that re-exports the §III formulas from Version 3's reference
  `hsm_v2_core.py` / `measures.py` and adds a single feature-dict
  adapter `hsm_score_from_features(fa, fb, weights=None)` used by every
  validation script.  FastDTW is explicitly symmetrised
  (`0.5 · (fastdtw(a,b) + fastdtw(b,a))`) so Lemma 1 P3 holds up to
  numerical noise.
- `run_all.sh`: master sweep driver (Phase 1 Docker up → Phase 2 pgbench
  TPC-B load → Phase 3 seven validations → Phase 4 figure regeneration)
  with `--smoke`, `--skip-docker`, and `--oltp-only` flags; per-script
  logs under `results/_run_logs/`.

### Changed (2026-04-14 kernel consolidation)
- All seven validation scripts now import `hsm_v2_kernel` and delegate
  the full similarity computation to `hsm_v2(...)` via the feature-dict
  adapter.  Any residual inline Jaccard / db2 / raw-cosine code paths
  left over from the v1 regression have been removed.  Verified:
    - OLTP static mode: DR = 1.152, p = 1.3e-39, S_T dominant
      (paper §A8c prediction).
    - Lemma 1 metric-property suite
      (`code/tests/test_metric_properties.py`): 11 / 11 green.
    - End-to-end adapter sanity: `|HSM(a,b) − HSM(b,a)| = 0` and
      `HSM(a,a) = 1`.

### Added
- `code/experiments/hsm_similarity.py`: complete five-dimension kernel
  with the formulas and parameters specified in paper §III.
    - `S_R`: Spearman rank correlation + arccos → unit interval.
    - `S_V`: log-ratio volume similarity, `exp(-|log(Q_A) - log(Q_B)|)`.
    - `S_T`: angular distance on template-vector cosine (required for
      metric property P4 — triangle inequality — in Lemma 1).
    - `S_A`: dual Jaccard on the table × column attribute sets.
    - `S_P`: DWT (db4, L=3) + SAX (α=4) + FastDTW (Sakoe–Chiba r=3)
      with band weights (0.40, 0.20, 0.20, 0.20).
    - Default weights `w_0 = (0.25, 0.20, 0.20, 0.20, 0.15)`.
- `code/experiments/theta_star.py`: closed-form θ*(N, Q) per Theorem 3
  and a two-regression cost calibrator.
- `code/tests/test_metric_properties.py`: pytest suite verifying Lemma 1
  properties P1–P4 on 60 random-workload triples with eps = 1e-6 slack
  for FastDTW numerical noise.
- `code/figures/_style.py`: IEEE-friendly matplotlib style, colour-blind
  safe palette, watermark kwargs.
- `code/figures/_placeholder.py`: watermarked-PDF generator covering
  `line / hist / bar / panels3 / loglog / box / scatter` layouts.  Used
  as a fallback when the paired CSV in `results/` is missing.
- `code/figures/plot_fig0[1-7]_*.py` and `plot_supp0[1-3]_*.py`:
  ten reusable plot scripts, one per figure in the paper.  No hardcoded
  numerical values and no figure numbers embedded in titles.
- `code/figures/regenerate_all_figures.py`: one-shot driver that runs
  every `plot_*.py` in the directory.
- LaTeX macros `\TBD{label}{v1value}`, `\TBDcell{label}`, `\REMARK{text}`
  plus a `soul`/`xcolor` dependency in both `main_article.tex` and
  `supplementary.tex`, used to flag every v1 numeric awaiting v2 data.

### Changed
- Seven validation scripts upgraded to the v2 module:
  `hsm_oltp_validation.py`, `hsm_burst_validation.py`,
  `hsm_burst_v2_validation.py`, `hsm_burst_v3_validation.py`,
  `hsm_job_validation.py`, `hsm_job_complexity_validation.py`,
  `hsm_sdss_validation.py`.  Parameters moved from `(db2, L=1, r=1)` to
  `(db4, L=3, r=3)` and weights from `(0.2, 0.2, 0.2, 0.2, 0.2)` to
  `w_0 = (0.25, 0.20, 0.20, 0.20, 0.15)`.  Each docstring carries a
  `[HSM v2]` banner.
- `paper/main/main_article.tex`:
    - Abstract DR range corrected from `1.09–2.01` to `1.093–2.008`
      (consistent with the §VI per-workload totals).
    - Composite equation now carries `\label{eq:composite}` so the
      caption of Table~\ref{tab:dims} uses `Eq.~\ref{eq:composite}`
      instead of the hardcoded `Eq.~6`.
    - Hardcoded `Theorem~3` narrative reference replaced with
      `Theorem~\ref{thm:econ}`.
    - All seven `\includegraphics{figures/image*.png}` references
      replaced with the new `fig0[1-7]_*.pdf` series.
- `paper/supplementary/supplementary.tex`:
    - Broken label `eq:supp-optwin-Nstar` corrected to
      `eq:supp-nstar-implicit` inside Corollary~\ref{cor:supp-scale-invariance}.
    - Three `\includegraphics` references (`image8.png`,
      `gate_sensitivity.png`, `theta_calibration.png`) pointed at the
      new `supp0[1-3]_*.pdf` series.
- Added `xcolor` and `soul` to the supplementary preamble to support
  the placeholder macros.

### Deprecated / Flagged
- Every numerical value in Section~VI of the main article (DR, $J$,
  $\beta$, speedup bands, overhead, per-dimension ablation deltas,
  $|r|$, ICC, etc.) is now wrapped in `\TBD{}` or `\TBDcell{}`.
- Discussion, Conclusion, and the Extended Experimental Details
  section in the supplementary each carry a `\REMARK{}` banner
  acknowledging that the v2 numbers will supersede them.

### Not changed
- The six theorems and four lemmas in the paper (main §IV and
  supplementary Sections I–X).  The proofs are unaffected; only the
  empirical validation numbers move.
- Prior-work baselines (`77\%` classification accuracy and
  `4,623×` latency improvement) which come from published v1
  conference papers and are not part of this regeneration.
- `LICENSE`, `.env.example`, `.gitignore` (copied verbatim from the
  Version 4 baseline).
