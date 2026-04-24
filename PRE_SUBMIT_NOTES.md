# Pre-Submission Notes — TKDE Submission (HSM_gated Version 5)

Created: 2026-04-16
Purpose: Track items that were deferred from the pre-submission Tier 1 fix, so they are not forgotten during revision rounds.

---

## Tier 1: Completed 2026-04-16 (committed to main_article.tex)

### T1.1 — C_db4 constant reconciliation (Theorem 6)
**Before:** Lemma 3 defined `C_db4 = √2/2 ≈ 0.707`. Theorem 6 body used `C_db4 ≈ 1.5`, contradicting Lemma 3.
**After:** Theorem 6 body now cites `C_db4 = √2/2 ≈ 0.707` with explicit reference to Lemma 3.

### T1.2 — Theorem 6 "closed-form" → "implicit fixed-point equation"
**Before:** Eq.~\eqref{eq:optwin-Nstar} labelled "(implicit)" next to a formula with Npts on both sides; confusing whether it is closed-form or not.
**After:** Rewritten as an explicit implicit fixed-point equation in a single scalar N*, with both sides using N* (not Npts), plus a note that bisection / fixed-point iteration converges in O(log(1/ε)).

### T1.3 — Limitation (ii) soft-framed
**Before:** "Fixed weights w_0 do not adapt to workload class: on SDSS-like workloads dominated by S_T, a single-dimension detector can outperform the composite; automated weight calibration would mitigate this."
**After:** Frames the trade-off as deliberate — HSM trades peak-per-workload J for cross-workload consistency + composite metric properties (P1-P4) + Algorithm 2 diagnosis, all of which a single-dim detector forfeits.

---

## Tier 2: To address during revision (if reviewer does not raise, volunteer in response letter)

### T2.1 — Change-point detection literature (§II.B Signal Processing)
Add 2-3 sentences after the DWT/SAX/DTW paragraph citing classical change-point detection:
- CUSUM (Page 1954)
- Bayesian Online Change-Point Detection (Adams & MacKay 2007, arXiv:0710.3742)
- Comment on how HSM-as-detector (Thm 4 reformulation) relates: HSM is a similarity-threshold detector, not a sequential CPD — it decides per-window, not per-stream-position, so the CUSUM/BOCPD tradeoffs (detection delay vs. false alarm rate) apply differently. Reviewer pre-empt: "HSM is a window-pair similarity classifier; it is complementary to streaming CPD methods which could feed HSM windows dynamically — out of scope here."

### T2.2 — Split Theorem 3 into 3a + 3b (if space permits in revision)
Current Theorem 3 bundles:
- (i)(ii) break-even formula Q_min(N), θ*(N,Q)
- (iii) T_HSM = Θ(Npts) regardless of access attribute
- (iv)(v) access-attribute independence saves Ω(τ·N log N)

Clean split:
- **Theorem 3a (Economic Break-Even):** parts (i)(ii) — closed-form θ*(N,Q)
- **Theorem 3b (Access-Attribute Independence):** parts (iii)(iv)(v) — independence + savings

Only split if page budget allows. Otherwise leave as bundled theorem and note in response letter: "Theorem 3 parts (i,ii) and (iii,iv,v) are separable; we bundled them to emphasise that both properties derive from the same cost model."

### T2.3 — Abstract flow cleanup
Current: "$80.8\%$ versus always-on and $42.5\%$ versus periodic" reads like jammed bullets.
Target rewrite: connect the two numbers into one flowing sentence, e.g.: "reducing advisor invocations by 80.8% over always-on and 42.5% over periodic (K=3) rebuild — while retaining phase-boundary Youden J ≥ 0.96."
Priority: low (cosmetic, doesn't affect technical content).

### T2.4 — β=1.094 Θ(N log N) framing
Current (§V L1448, Limitation (iii) L1497): claims `β=1.094, R²=0.9999` supports Θ(Npts log Npts).
Reviewer concern: 1.094 is closer to linear than N log N.
Pre-written response: "Over the operational range Npts ∈ [100, 30000], the additional log N factor is sub-dominant; both Θ(N) and Θ(N log N) fit the data within the residual precision (R² = 0.9999 for both, β-difference within standard error). Theorem 5's amortisation gap is derived from the more conservative Θ(N log N) assumption; Θ(N) would tighten the gap further. See `results/scale_analysis.csv` rows 1-12 for the two fits side by side."

---

## Tier 3: Pre-written reviewer responses (for cover letter or rebuttal)

### T3.1 — "Why only MongoDB cross-engine? Why not MySQL/SQLite?"
Planned response:
- MongoDB was chosen specifically to test the data-model boundary (relational → document), which is a stronger generality claim than relational → relational
- MySQL/SQLite extensions would require only the relational extractor already implemented (Section III-B), with engine-specific vocabulary Q bound to each planner's statement normalisation
- Estimated effort: 1-2 weeks for each additional engine; deferred because A-CE already establishes the kernel-transfer property under a harder test (polymorphic collection vs. relational schema)
- Can add MySQL in the revision if reviewer insists

### T3.2 — "Why no comparison to BALANCE / Indexer++?"
Planned response (already partially addressed in §V.G "Scope vs. ML-Based Advisors"):
- BALANCE (Wang 2024) and Indexer++ (Sharma 2022) public implementations are unavailable as of 2026-04-16
- Both solve the complementary *what*-to-build problem, not *when*-to-invoke; any direct throughput comparison conflates advisor quality with gating quality
- A composition experiment (HSM gating → BALANCE) would quantify Theorem 5's orthogonality claim, but requires BALANCE source not yet released
- Willing to add in revision if source becomes available before decision

### T3.3 — "Single-author paper at TKDE is unusual — is the work sufficiently broad?"
Planned response (only if reviewer raises it explicitly):
- The paper represents a self-contained contribution: one framework (HSM), one theoretical development (Theorems 1-6), one empirical evaluation across 5 workloads + 2 engines + 2 deployed advisors
- Single-authorship does not indicate limited effort: the replication package (available upon acceptance) contains ~15k LoC of experimental infrastructure, 13 result CSVs, and the reproducibility scripts
- Prior work (Reungsinkonkarn 2026, Reungsilpkolkarn 2026) provides the empirical precedent for the HSM concept; the present paper adds the formal theoretical core that was missing

### T3.4 — "Why is the empirical caveat on Theorem 2 not a contradiction?"
Quote: "On TPC-H, ΔS_PER ≈ 0 across all SFs (vs. ΔS_T ≈ 0.53, ΔS_A ≈ 0.56), so (S_T, S_A) dominate phase separation."
Planned response:
- Theorem 2 claims **minimal sufficiency**: no four-dim subset is sufficient for the witness pair. It does NOT claim all five dimensions are equally informative on every workload.
- S_PER's utility is workload-dependent: near-zero on TPC-H (analytical queries with stable long-term patterns), but 14% contribution on burst workload (A8d) where within-window temporal signature is the phase signal.
- The burst workload witness pair proves S_PER is necessary for Theorem 2; the TPC-H caveat shows the bound is not tight on every workload.
- This is the correct relationship: sufficiency is global, utility is workload-specific.

### T3.5 — "Why OLTP self-deactivates is not a limitation but a feature"
Planned response (partially in Limitation iv):
- pgbench OLTP has p_stable → 0 because every window sees unique key distributions (high-drift by construction)
- Theorem 5 bound `1/(1-p_stable) → 1×` correctly reports that no gating policy can save work on a workload that never stabilises
- This is a **correct diagnosis**, not a failure: the framework tells the operator "don't deploy HSM here" with formal justification, rather than silently producing wrong gating decisions
- Gate TPR = 1.00 at default θ = 0.75 on OLTP confirms the gate itself is functioning — it fires on every window, correctly classifying each as a phase change

---

## Submit-Day Checklist
- [ ] Verify 12-page layout in IEEEtran (not article.cls)
- [ ] Regenerate Table of Contents if TKDE uses one
- [ ] Anonymise if TKDE double-blind (check guidelines — believed to be single-blind as of 2026)
- [ ] Attach supplementary as separate PDF
- [ ] Include replication package pointer (Code & Data Availability section)
- [ ] Cover letter: mention prior HSM work (v1) and highlight new contributions (theoretical core + cross-engine)
- [ ] Declare AI use per IEEE policy (already in Acknowledgments)
