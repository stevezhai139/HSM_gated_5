# HSM_gated — Changelog

All notable changes to this regeneration branch are documented here.
This branch forks from `Paper 3A/Version 4/` after a code-paper drift
audit exposed that the earlier implementation used a 5-Jaccard
composite instead of the Spearman / log-ratio / angular / dual-Jaccard
/ DWT+SAX+FastDTW pipeline defined in §III of the paper.

## [2026-04-23] — Post-submission repo cleanup

### Replaced (paper/main/ now matches TKDE submission)
- `paper/main/main_article.{tex,pdf}`, `references.bib`, `IEEEtran.cls`:
  Replaced article-class manuscript with the IEEE Computer Society
  `IEEEtran` compsoc version (13 pp) actually submitted to TKDE on
  2026-04-20. Previous `IEEEtran.cls` was a 0-byte placeholder; now
  the full v1.8b CTAN file (281 KB).
- Added `paper/main/IEEEtran.bst` (57 KB, IEEE bibliography style)
  — required to re-compile with `bibtex`.
- `paper/supplementary/{supplementary.tex,pdf,references.bib}`: latest
  versions matching submitted `02_supplementary.pdf`.

### Vendored (cross-engine MongoDB self-containment)
- New directory `code/experiments/cross_engine/` containing a verbatim
  V3 snapshot of the MongoDB cross-engine experiment (8 files):
    - `_v3_hsm/hsm_v2_core.py` (V3's HSM kernel — different signature
      from V5's `hsm_v2_kernel.py`, must remain V3 for numerical
      reproducibility of §V results)
    - `common/{__init__,hsm_bridge,param_sampler,window_features}.py`
    - `mongo/adaptation/{13_mongo_adaptation,hsm_mongo_validation}.py`
      (the latter renamed from V3's `14_mongo_adaptation_theta_sweep.py`
      to match the README traceability table)
    - `mongo/workload/templates.py`
- One-line edit in `cross_engine/common/hsm_bridge.py`: `_V2_DIR` now
  points to vendored `../_v3_hsm/` instead of V3's original
  `v2_10seed/`.
- Refactored `code/experiments/overnight/b3_mongo_theta0775.py` to
  load the vendored `hsm_mongo_validation.py` from within this repo,
  removing the cross-repo `Version 3/` dependency that previously
  caused `FATAL: cannot find ...` for any reviewer cloning only
  `HSM_gated_5`.
- Added `cross_engine/README_VENDOR.md` documenting provenance and
  the rationale for not using V5's refactored kernel.
- Smoke-tested: full import chain (templates → param_sampler →
  window_features → hsm_bridge → V3 hsm_v2_core) resolves with no
  ImportError; `hsm_bridge.is_available()` returns `True` and
  `get_w0()` returns the canonical paper weights.

### Fixed (CITATION.cff metadata)
- Real ORCID `0009-0009-2077-0006` (was placeholder
  `0000-0000-0000-0000`).
- Repository URL `https://github.com/stevezhai139/HSM_gated_5` (was
  `https://github.com/arunr/HSM_gated`, which does not exist).
- Removed Zenodo DOI placeholder `10.5281/zenodo.0000000`; replaced
  with a YAML comment template to fill in once the Zenodo archive is
  minted on paper acceptance.

### Updated (README traceability)
- Page-count corrected: `paper/main/main_article` is **13 pp** (was
  labelled 12 pp).
- Cross-engine traceability row now points to the vendored
  `code/experiments/cross_engine/mongo/adaptation/hsm_mongo_validation.py`
  (was the non-existent `code/experiments/hsm_mongo_validation.py`).

---

## [2026-04-17b] — Second-pass cleanup (TKDE)

### Fixed (Supplementary reference alignment)
- `paper/main/main_article.tex` L622: Theorem 1 (Wavelet Pattern
  Preservation) proof sketch cited `Supplementary~\S III` (which is
  the Lemma 3 / Tight Linear Complexity proof). Corrected to
  `Supplementary~\S V` (Theorem 1 proof). This was a pre-existing
  misreference unrelated to the Lemma swap; caught in the post-swap
  cross-check audit.

### Cleaned (Dead macro removal)
- `paper/main/main_article.tex` preamble: removed the retired
  `\TBD / \TBDcell / \REMARK` macro definitions + the 8-line comment
  block describing why they were kept. Zero live invocations remain
  anywhere in the source; macros are no longer needed.
- `paper/supplementary/supplementary.tex` preamble: same three-macro
  block removed with its two-line comment.

### Rephrased (Theorem 6: "closed-form optimal window size" \u2192 "convex-optimal")
- Seven locations in `main/main_article.tex` where Theorem 6's
  $\Nstar$ was described as "closed-form" have been rewritten to
  reflect the technical reality that Theorem 6 provides a *closed-form
  convex cost model* whose minimiser $\Nstar$ is characterised by an
  *implicit fixed-point equation* (solved numerically). The cost model
  itself is closed form; the minimiser is not.
    - Abstract (L67), Intro contributions (L118), Conclusion (L1505):
      "closed-form optimal window size" \u2192 "convex-optimal window size"
    - Theorem 6 section intro (L544): rephrased as "the convex-optimal
      window size $\Nstar$ that minimises total steady-state cost"
    - Corollary (L953): "closed-form $\Nstar$" \u2192 "convex-optimal $\Nstar$"
    - A17 empirical-match text (L969): "closed-form prediction" \u2192
      "Theorem~6 prediction"
    - A17 summary table row (L1370): "Optimal $\Npts$ closed-form" \u2192
      "Optimal $\Npts$ from convex cost model"
- Remaining "closed-form" uses audited and verified to refer to
  genuinely closed-form objects: Theorem 3's $\thetastar(N,Q)$ (L123,
  L158, L461, L484, L685, L1428); Theorem 1's $\varepsilon(M,\Npts,L)$
  (L605); Theorem 5's speedup interval (L826).

### Verification
- `main_article.pdf` recompiled: 12 pages, no undefined references,
  no errors.
- `supplementary.pdf` recompiled: 14 pages, clean.
- Page-count preserved at 12 after rephrasing by choosing the shorter
  "convex-optimal" in list positions (abstract, intro, conclusion).

## [2026-04-17] — Pre-submission consistency pass (TKDE)

### Fixed (Lemma numbering + citation consistency main↔supp)
- `paper/main/main_article.tex`: physical order of Lemma 3 and Lemma 4
  swapped so both main and supplementary now agree:
    - Lemma 1 = HSM Metric Properties (`lem:metric`)
    - Lemma 2 = DWT Parameter Selection (`lem:dwt`)
    - Lemma 3 = Tight Linear Complexity (`lem:complexity`)   ← was Lemma 4
    - Lemma 4 = Wavelet Coefficient Bound   (`lem:wcb`)      ← was Lemma 3
  All 14 `\ref{lem:complexity}` / `\ref{lem:wcb}` callers auto-renumber
  correctly under the new ordering (verified in `main_article.aux`).
- `paper/main/main_article.tex` (Theorem 6 body, L893 region): the
  `a·Npts·log Npts` DWT/SAX/DTW term is now documented as a *conservative
  upper bound* accommodating kernel variants beyond FastDTW, with the
  tight $\Theta(\Npts)$ of Lemma~\ref{lem:complexity} noted as a strict
  special case. Previously the citation implied Lemma 3 proved the
  `N log N` form, which contradicted the lemma statement.
- `paper/main/main_article.tex` (Corollary, L937 region): rebuild-cost
  claim $\lambda(N)=\Theta(N\log N)$ now cites "the premise
  $T_A(N)=\Omega(N\log N)$ of Theorem~\ref{thm:speedup}" instead of
  Lemma 3 (whose scope is $T_{\mathrm{HSM}}$, not $T_A$).
- `paper/supplementary/supplementary.tex` (L760 region): same cost-model
  reframing as main, citing Lemma 3's Θ(Npts) as the tight special case.

### Fixed (T2.4 — β=1.094 vs Θ(Npts) framing, A12/A14 + Limitation iii)
- A12/A14 row in `main/main_article.tex` distinguishes two regression
  domains: (a) rebuild-cost $T_A$ over $N_{\mathrm{lineitem}}\!\in\!
  [1.2\mathrm{M},18\mathrm{M}]$ gives $\beta=1.177$, consistent with
  $\Theta(N\log N)$; (b) kernel cost $T_{\mathrm{HSM}}$ over
  $\Npts\!\in\![100,30{,}000]$ gives $\beta=1.094$, consistent with
  the tight $\Theta(\Npts)$ of Lemma~\ref{lem:complexity} (the small
  super-linear residual absorbs FastDTW path length and memory-hierarchy
  constant-factor effects; no genuine $\log\Npts$ factor).
- Limitation (iii) rewritten to say "sub-dominant constant-factor
  effects within the finite micro-benchmark range; the asymptotic bound
  is linear" — replacing the earlier text which implied
  $\Theta(\Npts\log\Npts)$ scaling.

### Added (T2.1 — Change-point detection literature in §II.B)
- `paper/main/main_article.tex` §II.B: paragraph positioning HSM against
  classical sequential CPD methods, citing CUSUM (Page, 1954) and
  Bayesian Online Change-Point Detection (Adams & MacKay, 2007).
  Frames HSM as a *window-pair similarity classifier* that is
  complementary (not substitutable) to streaming CPD — streaming CPD
  can supply windows to HSM dynamically; HSM supplies a DBMS-agnostic
  composite similarity single-signal CPD does not provide.
- `paper/main/references.bib`: two new `@article` entries
  (`Page1954CUSUM`, `AdamsMacKay2007BOCPD`).

### Fixed (T3.4 — Theorem 2 global sufficiency vs. workload utility)
- `paper/main/main_article.tex` empirical caveat after Theorem 2
  retitled "global sufficiency vs.\ per-workload utility" and rewritten
  to make explicit that minimal sufficiency is a property of the
  witness construction over the joint workload space; TPC-H's
  $\Delta\SPER\!\approx\!0$ is per-workload utility and does not
  contradict the theorem. Burst workload (A8d, 14% $\SPER$ contribution)
  is invoked as the witness pair on which $\SPER$ is necessary.

### Fixed (T1.1/T1.2/T1.3 — Tier-1 issues from PRE_SUBMIT_NOTES)
- Theorem 6 body now cites $C_{\mathrm{db4}}=\sqrt{2}/2\approx 0.707$
  (matching Lemma 4 / `lem:wcb`); the earlier $\approx 1.5$ value is
  removed.
- Equation (optwin-Nstar) reformulated as an explicit implicit
  fixed-point equation in a single scalar $\Nstar$, with a note that
  bisection converges in $O(\log(1/\epsilon))$.
- Limitation (ii) rewritten to frame the fixed-weights trade-off as a
  deliberate cross-workload-consistency choice (preserving composite
  metric properties P1–P4 and Algorithm~2 diagnosis) rather than a
  weakness.

### Verification
- `main_article.pdf` recompiled (12 pages, no undefined references, no
  LaTeX errors).
- `supplementary.pdf` recompiled (14 pages, no undefined references).
- Cross-document numerical audit passed:
  θ*∈{0.750, 0.6325, 0.7275} match; speedup bounds {5.22×, 5.20×,
  4.01×, 20.0×} match; p̂_stable∈{0.808, 0.751, 0.950, 0.000} match;
  95% certificate $[5.72\times, 23.45\times]$ matches; Youden
  $J^*\!\in\!\{0.9626, 0.9684\}$ matches; savings pairs
  80.8%/42.5% and 80.8%/44.4% match.

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
