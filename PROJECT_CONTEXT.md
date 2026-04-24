# HSM_gated Version 5 — Complete Project Context for Session Handoff

Created: 2026-04-17 by Claude Opus 4.6 session
Purpose: Enable a new Claude session to continue work seamlessly.
User: Steve (Arun Reungsilpkolkarn), arunjai139@gmail.com
Language preference: Thai for conversation, English for paper content

---

## 1. PROJECT OVERVIEW

**Paper title:** HSM: Workload-Similarity Gating for Index-Maintenance Decisions, with Formal Bounds
**Target journal:** IEEE Transactions on Knowledge and Data Engineering (TKDE)
**Author:** Arun Reungsilpkolkarn (single author), Bangkok University
**Status:** Pre-submission. Main paper 12 pages (article.cls mock), Supplementary 14 pages. Needs IEEEtran port before submit.

**Core idea:** HSM is a training-free, DBMS-agnostic decision layer that answers "WHEN should an index advisor be invoked?" — complementary to existing advisors (Dexter, BALANCE, Indexer++) that answer "WHAT indexes to build." HSM quantifies workload similarity along 5 dimensions and gates advisor invocation in Theta(Npts) time via a calibrated threshold theta*(N,Q).

**Prior work (v1):** Two conference papers (Reungsinkonkarn 2026 iEECON, Reungsilpkolkarn 2026 ICICT) validated the HSM concept empirically. This Version 5 adds: (a) formal theoretical core (6 theorems + 4 lemmas), (b) cross-engine validation (PostgreSQL + MongoDB), (c) end-to-end deployment with production advisors.

---

## 2. FILE STRUCTURE (key files only)

### Paper
- `paper/main/main_article.tex` — Main paper (12 pages, article.cls)
- `paper/main/references.bib` — 48 BibTeX entries
- `paper/main/main_article.pdf` — Compiled PDF
- `paper/supplementary/supplementary.tex` — Supplementary (14 pages)
- `paper/supplementary/supplementary.pdf` — Compiled PDF

### Code
- `code/experiments/hsm_v2_kernel.py` — **Core HSM kernel implementation**
  - L18: S_P pipeline = DWT(db4, L=3) -> SAX(alpha=4) -> FastDTW(r=3)
  - L41: `SAX_ALPHA = 4` (hardcoded)
  - L43: `BAND_WEIGHTS = (0.40, 0.20, 0.20, 0.20)` for cA3, cD3, cD2, cD1
  - L113-125: `_sax_encode` uses Gaussian breakpoints via `norm.ppf`
- `code/experiments/theta_star.py` — Theta* calibration and ROC analysis
- `code/experiments/hsm_sdss_validation.py` — SDSS workload validation
- `code/experiments/hsm_job_validation.py` — JOB/IMDB validation
- `code/experiments/hsm_oltp_validation.py` — pgbench OLTP validation
- `code/experiments/hsm_burst_v3_validation.py` — Burst workload validation
- `code/experiments/tier2/hsm_burst_end_to_end.py` — End-to-end with Dexter/Supabase advisors
- `code/experiments/workload_generator.py` — TPC-H workload trace generator

### Results (authoritative CSVs)
- `results/t5_validation.csv` — **Theorem 5 bounds per workload** (p_stable_hat, bound_speedup, TPR, FPR per workload x theta)
- `results/theta_optimal_per_workload.csv` — **Youden-optimal theta* per workload** (TPC-H 0.605, SDSS 0.6325, JOB 0.7275, burst 0.895, MongoDB 0.530)
- `results/scale_analysis.csv` — Kernel-measurement cost vs Npts (beta=1.094, R^2=0.9999)
- `results/throughput.csv` — SF 0.2/1.0/3.0 throughput comparison
- `results/score_distribution.csv` — HSM score distributions
- `results/within_cross_phase.csv` — Within-phase vs cross-phase score data
- `results/burst_end_to_end/` — A13d end-to-end CSV results (small_q, mid_q)
- `results/sf0.2/`, `results/sf1.0/`, `results/sf3.0/` — Per-SF detailed results
- `results/overnight_2026-04-16/` — B-series robustness experiments

### Tracking Documents
- `AUDIT_FINDINGS.md` — Complete audit report (CRITICAL/MAJOR/MINOR) from pre-submission review
- `PRE_SUBMIT_NOTES.md` — **Tier 2/3 pending items + pre-written reviewer responses + submit-day checklist**
- `README.md`, `REPRODUCE.md`, `CHANGELOG.md`, `CITATION.cff` — Standard repo docs

---

## 3. MATHEMATICAL FRAMEWORK

### Five HSM Dimensions
| Dim | Symbol | Formula | Captures |
|-----|--------|---------|----------|
| Ratio | S_R | Spearman rank + arccos angular distance | Query-type frequency ranks |
| Volume | S_V | exp(-|ln Q_A - ln Q_B|) = min/max | Query volume |
| Type | S_T | Angular distance on unit-normalised frequency vectors | Query-template distribution |
| Access | S_A | Average of two Jaccard scores (table + column) | Attribute overlap |
| Pattern | S_P | DWT(db4,L=3) -> SAX(alpha=4) -> FastDTW(r=3), band-weighted | Temporal pattern |

Composite: S_HSM = sum(w_i * S_i), w_i >= 0.05, sum = 1
Default weights: w_R=0.25, w_V=w_T=w_A=0.20, w_P=0.15

### Theorems (6) and Lemmas (4)

**Lemma 1 (Metric Properties):** d_HSM = 1 - S_HSM satisfies P1-P4 (bounded, identity, symmetry, triangle inequality). Each d_i is an analytic metric; convex combination preserves all four.

**Lemma 2 (DWT Parameter Selection):** alpha* = 2k = 8 is SAX-only worst case. Deployed pipeline uses alpha = k = 4 because DWT detail bands (cD1-cD3) absorb burst direction, leaving SAX on cA3 to discriminate only k=4 canonical states. Full remark in supp Section II.

**Lemma 3 (Tight Linear Complexity):** T_HSM(Npts, N) = Theta(Npts), space Theta(Npts). Lower bound by adversary argument. [2026-04-17 swap: was Lemma 4.]

**Lemma 4 (Wavelet Coefficient Bound):** ||q - q_tilde||_2 <= C_db4^L * M * sqrt(Npts), C_db4 = sqrt(2)/2 ~ 0.707. [2026-04-17 swap: was Lemma 3. Main and supplementary now agree.]

**Theorem 1 (Wavelet Pattern Preservation):** epsilon(M, Npts, L) = 2*C_db4^L*M/sqrt(Npts), strictly decreasing in L and Npts. At L=3, Npts=100: epsilon <= 0.071*M.

**Theorem 2 (Minimal Sufficient Dimensions):** All 5 dimensions necessary (witness pairs prove each indispensable), sufficient (span rank/magnitude/angular/attribute/temporal info), minimal (no 4-dim subset suffices).

**Theorem 3 (Economic Optimality of HSM Gating):**
- (i) Q_min(N) = Theta(log N)
- (ii) theta*(N,Q) = 1 - Q_min(N)/Q for Q > Q_min
- (iii) T_HSM = Theta(Npts) regardless of access attribute
- (iv-v) HSM saves Omega(tau * N log N) over tau attribute transitions

**Theorem 4 (Detector-Quality Concentration via Stratified Hoeffding):**
- Stratified sampling: m_w within-phase pairs, m_c cross-phase pairs
- eta(m, delta) = sqrt(log(4/delta)/(2m))
- Simultaneous band on (TPR, TNR) with probability >= 1-delta
- Union bound factor 4/delta (not 2/delta) for two-sided stratification

**Theorem 5 (Operational Speedup with Phase-Mix Prior):**
- p_stable = pi_w*TPR + (1-pi_w)*(1-TNR)
- Speedup -> 1/(1-p_stable) as N -> infinity
- Convergence rate O(Npts/(N log N))
- Deployment certificate: closed-form [lower, upper] speedup bounds at any operator pi_w

**Theorem 6 (Optimal Window Size):**
- Cost function: C(Npts) = a*Npts*log(Npts) + lambda*(f_N - g*log(Npts)) + M^2*C_db4^{2L}/Npts^2
- N* is unique root of dC/dNpts = 0 (implicit fixed-point equation, solve numerically)
- Strictly convex, bisection converges in O(log(1/epsilon))
- N_min^pts >= 8 for statistical validity; round to nearest integer
- Corollary: N* invariant in database size N to leading order

---

## 4. KEY EMPIRICAL RESULTS

### Theorem 5 Bounds (from t5_validation.csv)
| Workload | theta | p_stable_hat | Bound speedup | savings_vs_always | savings_vs_periodic |
|----------|-------|-------------|---------------|-------------------|---------------------|
| TPC-H (all SF) | 0.75 | 0.808 | 5.22x | 80.8% | 42.5% |
| Burst | 0.75 | 0.808 | 5.20x | 80.8% | 44.4% |
| SDSS (theta*) | 0.6325 | 0.751 | 4.01x | 75.1% | 25.2% |
| JOB (theta*) | 0.7275 | 0.95 | 20.0x | 95.0% | 85.7% |
| OLTP | 0.75 | 0.000 | 1.0x | 0% | — (self-deactivates) |

### Scale Analysis (from scale_analysis.csv)
- Rebuild cost: beta=1.177, R^2=0.998 over N_lineitem in [1.2M, 18M] (SF 0.2-3.0)
- Kernel measurement cost: beta=1.094, R^2=0.9999 over Npts in [100, 30000]

### Cross-Engine (MongoDB A-CE)
- A-CE1 (theta=0.75): DR=1.271, J=0.705
- A-CE2 (theta=0.65, calibrated): DR=1.486, J=0.981, 85.8% fewer calls
- theta*_Mongo = 0.55 (J*=1.000) vs theta*_PG = 0.775 (J*=0.965)

### A14: Youden J at every scale
- J >= 0.9626 across SF 0.2, 1.0, 3.0
- 95% deployment certificate: [5.72x, 23.45x] at pi_w=0.95

---

## 5. AUDIT AND FIXES COMPLETED

### Phase A: Critical/Major fixes (all resolved)
- Burst bound: 5.20x (from p_stable_hat=0.808), not 13.0x (which was oracle p_stable_true)
- JOB bound: 20.0x (from p_stable_hat=0.95), corrected from 19x
- SDSS theta*: 0.6325 (Youden), corrected from 0.633
- OLTP: self-deactivates (p_stable_hat=0.000), supp Table XVII regenerated from CSV
- beta conflation: separated rebuild-cost beta=1.177 vs kernel-cost beta=1.094
- REMARK macro: silenced (\newcommand{\REMARK}[1]{})
- Stratified Hoeffding: eta uses log(4/delta) not log(2/delta) for union bound
- Supp Table XVII: fully regenerated from empirical p_stable_hat (not oracle)

### Phase B: Redundancy cleanup
- 80.8%/42.5% reduced from 5 to 4 occurrences
- 16.1x/41.9x reduced from 5+ to 2 occurrences

### Phase C: M7 (SAX alpha notation)
- Supp Lemma 2 Remark expanded: explains alpha*=2k=8 is SAX-only worst case; deployed pipeline offloads burst direction to DWT detail bands; alpha=k=4 sufficient
- Main Lemma 2: disclaimer moved to footnote (saves space)

### Tier 1 Fixes (pre-submission)
- T1.1: C_db4 constant reconciled (was 1.5 in Thm 6, now 0.707 matching Lemma 4 / `lem:wcb`)
- T1.2: Theorem 6 "closed-form" -> "implicit fixed-point equation" + bisection convergence note
- T1.3: Limitation (ii) reframed as deliberate design choice (cross-workload consistency + metric properties + Algorithm 2 diagnosis)
- T1.4: Theorem 6 N_min^pts >= 8 constraint + integer rounding footnote

### Layout fix
- Bibliography compacted (small font + itemsep=0) to fit ref [48] on page 12
- Lemma 2 disclaimer moved to footnote to save vertical space

---

## 6. PENDING TASKS (see PRE_SUBMIT_NOTES.md for details)

### Tier 2 (do during revision or before submit)
- T2.1: Add change-point detection citations (CUSUM, BOCPD) to Section II.B
- T2.2: Consider splitting Theorem 3 -> 3a (break-even) + 3b (access-attribute independence)
- T2.3: Abstract flow cleanup (80.8%/42.5% sentence)
- T2.4: beta=1.094 Theta(N log N) framing (pre-written response ready)

### Tier 3 (pre-written reviewer responses)
- T3.1: "Why only MongoDB?" (document-model boundary is harder test)
- T3.2: "Why no BALANCE/Indexer++ comparison?" (code unavailable, orthogonal layers)
- T3.3: Single-author defense (if asked)
- T3.4: Theorem 2 empirical caveat (sufficiency global, utility workload-specific)
- T3.5: OLTP self-deactivation is a feature

### Submit-day checklist
- [ ] Port to IEEEtran.cls (currently article.cls mock)
- [ ] Write cover letter
- [ ] Archive replication package (Zenodo DOI)
- [ ] Final cold-read of abstract + conclusion + Table 5
- [ ] Attach supplementary as separate PDF
- [ ] Declare AI use per IEEE policy (already in Acknowledgments)

---

## 7. IMPORTANT TECHNICAL NOTES

### p_stable_hat vs p_stable_true
- **p_stable_hat**: empirical estimate from gated-run trigger count. USED IN MAIN PAPER.
- **p_stable_true**: oracle value from ground-truth phase labels. USED ONLY for validation.
- Supp Table XVII was previously using p_stable_true; regenerated to use p_stable_hat.

### Two distinct beta fits
- **Rebuild cost** beta=1.177 (R^2=0.998): OLS on T_A(N) for N_lineitem in [1.2M, 18M]
- **Kernel measurement cost** beta=1.094 (R^2=0.9999): OLS on T_HSM(Npts) for Npts in [100, 30000]
- These are genuinely different fits on different N ranges. Paper explicitly disambiguates.

### SAX alpha=4 vs alpha*=8
- Lemma 2 proves alpha*=2k=8 necessary for SAX-only discrimination of 2k burst regions
- Deployed pipeline: DWT detail bands absorb burst direction; SAX on cA3 needs only k=4 states
- Main paper: footnote on Lemma 2 (L568-571)
- Supplementary: expanded Remark after Lemma 2 proof (L230-247)

### Notation
- S_P (Pattern) used consistently everywhere. S_F does NOT exist in any file.
- \SPER macro renders as S_P in both main and supplementary.

---

## 8. USER PREFERENCES AND COMMUNICATION STYLE

- User communicates in Thai; responds are in Thai for conversation, English for paper edits
- User said: "เราไม่ได้เทียบกับ ML โดยตรง และงานเราไม่ใช่ ML" — DO NOT frame HSM as ML comparison
- User said: "เชื่อผลของ version 5 เท่านั้น" — Trust ONLY Version 5 CSV results
- User said: "ทำตามที่คุณคิดว่าเหมาะสมที่สุด" (for REMARK cleanup) — autonomous judgment OK
- User is emotionally invested — this is a significant research milestone. Be supportive but honest.
- User's fear: "เป็น nobody ในงานวิจัยระดับนี้" — prior assessment: paper quality is mid-to-upper TKDE tier, likely passes editor desk, expect major revision round 1.

---

## 9. HOW TO BOOTSTRAP A NEW SESSION

Tell the new Claude:
```
อ่านไฟล์ PROJECT_CONTEXT.md ใน HSM_gated ก่อน แล้วอ่าน PRE_SUBMIT_NOTES.md
เราจะทำ [specific task] ต่อจาก session ก่อน
```

Key files to read first:
1. This file (PROJECT_CONTEXT.md) — full overview
2. PRE_SUBMIT_NOTES.md — pending tasks + reviewer responses
3. paper/main/main_article.tex — current main paper
4. paper/supplementary/supplementary.tex — current supplementary
5. results/t5_validation.csv — authoritative numbers

---

## 10. 2026-04-17 UPDATE (consistency pass)

Done in this session (see CHANGELOG.md "[2026-04-17]"):
- Main Lemma 3/4 physical order swapped → main and supp now agree
  (Lemma 3 = Tight Linear Complexity; Lemma 4 = Wavelet Coefficient Bound).
- Theorem 6 cost-model + Corollary citations fixed in main + supp
  (a·Npts·log Npts is now labelled as a conservative upper bound;
  Lemma 3's Θ(Npts) noted as the tight special case).
- β=1.094 (kernel) vs β=1.177 (rebuild cost) clearly partitioned in
  A12/A14 + Limitation (iii); no remaining Θ(N log N) claim for the
  HSM kernel.
- CUSUM + BOCPD citations added to §II.B (T2.1).
- Theorem 2 "empirical caveat" rewritten as "global sufficiency vs.
  per-workload utility" (T3.4).
- Both PDFs recompile cleanly (main 12 pp, supp 14 pp, no undefined
  refs).

Still pending (from PRE_SUBMIT_NOTES.md) if Steve wants to address:
- T2.2 split Theorem 3 → 3a + 3b (only if space allows in revision).
- T2.3 abstract flow cosmetic polish of "80.8% vs. always-on / 42.5%
  vs. periodic".
- T3.1–T3.3, T3.5 are pre-written reviewer responses, not paper edits.
- Submit-day checklist (IEEEtran port, TOC, anonymisation, supp PDF
  attachment, replication pointer, AI-use declaration).
