# Pre-Submission Audit — Paper 3A V5 (HSM_gated)

**Audit date:** 2026-04-16
**Scope:** Full audit of `paper/main/main_article.tex` (1,505 lines) and `paper/supplementary/supplementary.tex` (1,775 lines)
**Focus areas (per your selection):** Claim strength · Numerical consistency · Math/proof correctness · Contribution framing C1–C5 · Redundancy

Severity legend:
- **CRITICAL** — will embarrass the paper at review, or contradicts itself on a headline number. Must fix before submission.
- **MAJOR** — reviewer will flag; addressable with text edits.
- **MINOR** — stylistic, cosmetic, or redundancy cleanup.

---

## CRITICAL — fix before submission (6 issues)

### C1. `\REMARK` banner renders in red at the top of Section XI of the supplementary
**File/line:** `supplementary.tex` L890–894; macro defined L71
**Exact text that will appear in the PDF (red italic):**
> *[REMARK: Every numeric in Sections XI–XIII of this supplementary (A1–A14, A-CE1–3, scale-sensitivity tables, gate sensitivity, θ calibration) is a v1 placeholder pending the HSM_gated v2 regeneration. Do not cite these values as final; the proofs in Sections I–X are unaffected.]*

This was presumably an internal note and not meant for the camera-ready. Unlike the main paper (which silences `\REMARK{}` via `\newcommand{\REMARK}[1]{}` at L58), the supplementary defines it as red italic text (L71) and it WILL render. A reviewer opening Section XI will see the author telling them "do not cite these values" — this is effectively a self-withdrawal of §§XI–XV.

**Fix:** either redefine `\REMARK{}` as a no-op in `supplementary.tex` L71 (swap to `\newcommand{\REMARK}[1]{}`), or remove L890–894 outright. The latter is safer if any other `\REMARK{…}` calls exist elsewhere.

### C2. Supplementary title does not match the main paper title
**Main (L61–62):** "HSM: Workload-Similarity Gating for Index-Maintenance Decisions, with Formal Bounds"
**Supplementary (L77–78):** "Multi-Scale Workload Similarity Measurement for Proactive Database Index Tuning Using Hierarchical Signal Processing"

Also — the supplementary file header (L1) is self-labelled "Supplementary Material (Version 4)" and L5 says "Restructured to match Version 4 (6 Theorems + 4 Lemmas) of the main paper." Either the Version 5 retitle never propagated, or the two documents were developed on separate branches. TKDE submission systems display them as a linked pair; mismatched titles are the first thing the EiC's desk-check flags.

**Fix:** update `supplementary.tex` L77–78 (and the header comments L1, L5) to the Version 5 title.

### C3. Headline speedup bound for burst contradicts itself inside the main paper
**Main Table III (L1333, A13c):** "burst (13.0×, 80.8%/44.4%)"
**Main prose (L1108–1112):** "burst attains 80.8%/44.4% savings under a **5.20×** bound"
**Supp Table XVII (L1721):** burst at θ=0.75 → **13.0×** bound

The supplementary and the main-paper table agree (13.0×). The main-paper *prose* appears to have copied TPC-H's 5.20× value (which is TPC-H at θ=0.75; see Table III same line and `results/t5_validation.csv`). This is a direct internal contradiction and a reviewer will catch it on the first read of §V.

**Fix:** main L1110–1112 → replace "5.20× bound" with "13.0× bound". Or, if the prose intended to report burst at the *default* θ=0.75 (where p̂_stable=0.808 → 5.20×), make the θ explicit — but that choice contradicts Table III which uses θ* for burst.

### C4. Headline speedup bound for JOB differs by ~3× between main and supplementary
**Main prose (L1113):** "JOB (n=20, sample-limited) attains 95%/85.7% under a **19×** bound"
**Supp Table XVII (L1723):** JOB at θ*=0.728 → **6.67×** bound

19× ≠ 6.67× is not a rounding or protocol difference — it implies two different p_stable values. Inspecting `results/t5_validation.csv` (JOB at θ*): p_stable≈0.850, 1/(1−0.85)=6.67×. A 19× bound would require p_stable≈0.947, which appears nowhere in the CSV set I examined. The main-paper number appears to be stale.

**Fix:** reconcile. Most likely path: update main L1113 to "6.67× bound" to match supp Table XVII and the CSV of record. If 19× is defensible, add the derivation row to Table XVII.

### C5. Headline speedup bound for SDSS differs between main and supplementary
**Main Table III (L1333) and prose (L1111):** SDSS → **4.01×** bound
**Supp Table XVII (L1722):** SDSS at θ*=0.633 → **3.71×** bound

Less severe than C3/C4 but still a direct numeric mismatch on a value that appears in the abstract-adjacent summary table. Probably θ-rounding (main quotes the θ* from a coarser sweep, supp quotes it from finer-grained CSV). Either way the two documents must match.

**Fix:** pick one value, propagate. If 3.71× is the canonical CSV value, update main Table III L1333 and L1111; otherwise update supp Table XVII L1722.

### C6. OLTP narrative contradicts supplementary Table XVII
**Main prose (L1113–1116) and Table III (L1333):** "OLTP correctly self-deactivates: pgbench drifts on essentially every window, so p_stable→0 and the bound itself collapses to 1×"
**Supp Table XVII (L1724):** OLTP (pgbench) at θ=0.75 → p̂_stable=0.919, bound=**12.3×**, TPR=1.00

The two documents are describing the same experiment but reporting opposite stories:
- `results/t5_validation.csv`, OLTP default θ=0.75: `p_stable_hat=0.0` (empirical from Hoeffding), `p_stable_true=0.919` (oracle), bound=1.0× (using hat).
- Supp Table XVII is reporting the oracle p_stable (0.919) and the resulting 12.3× bound — but the whole point of the self-deactivation narrative is that the *empirical* Hoeffding estimate under-counts stable windows and thus collapses the bound.

**Fix:** supp Table XVII L1724 is the wrong row to show. Either (a) drop the OLTP row and cite the self-deactivation narrative only, or (b) replace the "p̂_stable" cell with the empirical value (0.0) so Bnd reads 1.0× — matching the main text.

---

## MAJOR — reviewer will likely flag (7 issues)

### M1. β disagreement between main (β=1.094) and supplementary (β=1.177) for Θ(N log N) scaling
**Main L1330, L1439, L1475:** β=1.094, R²=0.9999 across Npts∈[100, 30000]
**Supp L1449, L1507 (Fig. caption):** β=1.177, R²=0.998 across same range
**CSV of record (`results/scale_analysis.csv`):** β=1.177, R²=0.998

The supplementary matches the CSV; the main paper's β=1.094 / R²=0.9999 appears to come from a *different* fit (possibly the kernel-only cost excluding wavelet overhead, or an earlier run). In a theory paper where Theorem 5 hinges on this exponent, reviewers will notice.

**Fix:** replace β=1.094 with β=1.177 in main L1330, L1439, L1475, and in Table III. If β=1.094 is the kernel-internal slope and β=1.177 is the kernel-measurement-end-to-end slope, disambiguate explicitly.

### M2. Main paper mis-labels the β=1.094/1.177 scaling as "index-rebuild cost"
**Main L1437–1439:** "OLS regression of index-rebuild cost T_A(N) across N_pts∈[100, 30000] yields β=1.094..."
**Supp and CSV:** the fit is over the *kernel measurement cost* S_HSM(W_A, W_B), not the index rebuild cost. The rebuild cost is `C_create = aN log N` (supp L909) and is not measured in this experiment.
**Main L1473–1475 is correct:** "kernel-measurement cost itself is empirically Θ(N_pts log N_pts)..."

So the main paper contradicts itself three lines apart: L1439 says "rebuild cost", L1473 says "kernel-measurement cost". The CSV supports the latter.

**Fix:** change L1437–1439 "index-rebuild cost T_A(N)" → "kernel-measurement cost T_HSM(N_pts)".

### M3. Hoeffding η formula differs between theorem statement and Notation Summary
**Theorem 4 in supp (L565, L1324, L1490):** η(m,δ) = √(log(4/δ)/(2m)) — correct stratified form
**Notation Summary (supp L1763):** η_{m,δ} = √(log(2/δ)/(2m)) — single-rate form

The stratified version (4/δ) is required to give joint confidence 1−δ over both TPR and TNR estimates via union bound. Notation Summary is a legacy single-rate form.

**Fix:** supp L1763 → √(log(4/δ)/(2m)) to match Theorem 4.

### M4. Abstract "J≥0.96 at every scale" does not disclose the protocol qualifier
**Main L82–84 (abstract):** "HSM attains J≥0.96 at every scale..."
**Main Table III A14 (L1334):** J≥0.9626 across all SF — same protocol
**Supp Table XVIII (θ calibration, per context summary):** all-pairs protocol gives J*=0.33–0.50 across workloads; AUC=0.705–0.810

The abstract and Table III use the *per-window within-phase-vs-cross-phase* Youden J (detection protocol), while the θ calibration table uses *all-pairs* Youden J. These are different quantities but are both labelled "Youden J" in the text. A careful reviewer will notice that the supp Table XVIII numbers look much worse than the abstract claim.

**Fix:** in the abstract and in Table III, qualify: "Youden J≥0.96 in the within-phase vs. cross-phase detection protocol at m_w=m_c=160". Also add a one-sentence disclosure in §V.A that all-pairs Youden J is a different (harder) quantity and is reported in the supplementary.

### M5. C3 ("access-attribute independence") claim is supported but phrased in two different ways
**Main L127–131 (C1–C5 block, §I):** "C3 HSM runs in Θ(N_pts) time independent of table width |A|, index structure, or page-access model"
**Main §III.G (Access-Attribute Dependence sub-section, approx. L680–720):** frames C3 via a closed-form independence proof on the |A|→0 limit
**Supp Theorem 3, Part (ii) (L462–466):** proves T_HSM = Θ(N_pts) regardless of access attributes |A|

C3 is sound, but the main paper should cite "Theorem 3 part (ii)" at the C3 statement in §I (currently it does not). A reviewer will otherwise interpret C3 as an empirical claim rather than a theorem.

**Fix:** at main L130 or equivalent, add "(formalised as Theorem 3(ii), supplementary §VII)".

### M6. "Bound" vs. "limit" terminology in Theorem 5 is ambiguous
**Theorem 5 statement and main conclusion L1456:** "speedup certificate → 1/(1−p_stable)"
**A13c text L1114–1116:** "the bound itself collapses to 1×"
**A13c table caption "Bnd" column:** shows inf for OLTP (θ*)

"Bound" in probability-theory reading is an upper bound, so "bound=1×" means "no more than 1×", which is trivially true for any speedup-vs-always-on. The intended meaning is "*attained* speedup asymptote 1/(1−p_stable)". The current text lets a reviewer interpret Theorem 5 as vacuous.

**Fix:** change "bound" → "attained asymptote" in the main A13c prose; keep "Bnd" in the supp table but add a footnote "Bnd = 1/(1−p̂_stable), not an upper bound on realised speedup".

### M7. L2 "α* = 2k = 8" contradicts the default implementation α = 4
**Supp Lemma 2 proof (§II):** proves α* = 2k where k=4 is the db4 wavelet vanishing-moments count → α* = 8
**Main §III.E (DWT settings):** L=3 decomposition levels, default decimation α=4 (i.e., 2^L with L=3 actually gives 8, but the text conflates "L=3 levels" with "decimation α=4")

If the L2 proof concludes α*=8 is optimal but the implementation uses α=4, then either (a) the implementation is sub-optimal (contradicts the experimental claims) or (b) "α" in L2 and "α" in §III.E are different quantities with the same symbol.

**Fix:** audit §III.E to confirm whether the deployed decimation matches α*=8. If it uses α=4 (as L=3 with decimation-by-2 would yield), clarify the notation clash in L2.

---

## MINOR — cosmetic, redundancy, style (6 issues)

### m1. N_pts defined inconsistently between main and supp
**Supp L100–101:** "N_pts = |W_A| + |W_B| denotes the total query events across both windows"
**Main §III preprocessing (approx. L220):** N_pts is "the number of events in a single window"

The two definitions differ by a factor of 2. All complexity results still hold (Θ(N) is insensitive), but the constant in β and in C_db4^L · M · √N_pts (L4) changes. Pick one and propagate.

### m2. "5.20×" appears 5 times in the main paper
Locations: L1096, L1112, L1333 (two columns), L1459 (conclusion). The SF=3.0 95% certificate [5.72×, 23.45×] also appears 3 times (L1334, L1444, L1460). Readers lose the novelty of the number; tighten to 2 mentions (abstract + §V.A13c table + conclusion should name the interval, not the TPC-H point bound).

### m3. "80.8% / 42.5%" savings-pair appears 5+ times
L84, L1331, L1333, L1459, and supp Table XVII. Redundant with Table III (A13). Delete the abstract-level repetition in the conclusion.

### m4. Theorem statements are verbatim-duplicated between main §IV and supp §§I–X
Theorems 1–6 and Lemmas 1–4 are repeated word-for-word. Standard IEEE/ACM supplementary style is to refer via hyperref ("\Cref{thm:gate}") instead of re-stating. Either (a) delete the Lemma/Theorem boxes from supp §§I–X and begin each section with "Proof of Theorem N (§IV of main).", or (b) leave them but tighten the redundancy disclosure in the supp abstract (L84–88 currently says "Notation follows the main paper exactly" — add "theorem statements are reproduced verbatim for self-containedness").

### m5. "16.1× at π_w=0.95, rising to 41.9× at π_w=0.99" appears 5 times
L1334, L1445–1446, L1459, L1460, and supp (per §XII scale sensitivity). This is the third derivative of the TPC-H worst-case certificate and is reported in at least three prose paragraphs. Pick one.

### m6. `\REMARK{}` macro in main is *defined* but never fired after macro-expansion
Main L58: `\newcommand{\REMARK}[1]{}` silences it. Fine as-is, but leaves dead macro definitions in the preamble. Cosmetic only.

---

## Sanity checks that passed

- Metric properties P1–P4 (Lemma 1) verified against the five-dimensional construction.
- T3 closed-form θ*(N, Q) = 1 − Q_min(N)/Q derived correctly from NetBenefit=0 condition (supp §V).
- T4 stratified Hoeffding union bound (4/δ factor) correctly yields joint-confidence 1−δ over (TPR, TNR).
- T5 Amdahl-style 1/(1−p_stable) asymptote derivation is sound.
- T6 closed-form N*_pts from differentiating per-window cost w.r.t. window size is correct.
- Contributions C1, C2, C4, C5 framings are consistent between main §I and main §VI/VII. Only C3 needs the Theorem 3(ii) citation fix (M5).
- Figure captions in supp §XII match the scale_analysis.csv fit within rounding.

---

## Recommended fix order (lowest risk first)

1. **C1** — delete or silence `\REMARK` (one-line edit).
2. **C2** — retitle supplementary (one-line edit).
3. **M3, m1, m6** — notation/formula consistency fixes (surgical).
4. **M1, M2** — β and cost-label reconciliation (requires a decision: which fit is canonical?).
5. **C3, C4, C5, C6** — numeric reconciliation between main prose, main Table III, supp Table XVII. Ideally re-run the `results/t5_validation.csv` → Table generator and regenerate both tables from one source.
6. **M4** — abstract disclosure; short paragraph edit.
7. **M5, M7** — citation and notation clarification.
8. **Redundancy pass (m2–m5)** — once numbers are locked.

---

## Open questions for you before I start editing

1. Do you want me to silence the supp `\REMARK` macro (preserve the markers for later) or delete the L890–894 block outright?
2. For the β question (M1): is β=1.094 the kernel-internal fit and β=1.177 the kernel-measurement-end-to-end fit, or is one of them simply stale? I can re-fit from `results/scale_analysis.csv` and report.
3. For the JOB/burst/SDSS/OLTP bound reconciliation (C3–C6): should I regenerate supp Table XVII and main Table III from `results/t5_validation.csv` in a single pass, or do you want to inspect the CSV and choose the canonical rows yourself?
4. For the abstract J≥0.96 qualifier (M4): are you OK with the disclosed phrasing "within-phase vs. cross-phase detection protocol"? Some TKDE reviewers prefer the ML term "seen-vs-unseen distributional drift detection"; let me know the register you want.

I'll wait for your direction before editing either `.tex` file.
