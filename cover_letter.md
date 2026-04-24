21 April 2026

**To the Editor-in-Chief**
IEEE Transactions on Knowledge and Data Engineering

**Subject:** Submission of regular paper — *"HSM: Workload-Similarity Gating for Index-Maintenance Decisions, with Formal Bounds"*

Dear Editor,

I am pleased to submit the manuscript titled *"HSM: Workload-Similarity Gating for Index-Maintenance Decisions, with Formal Bounds"* for consideration as a regular paper in *IEEE Transactions on Knowledge and Data Engineering*.

**Contribution.** The manuscript introduces the *"when to invoke"* question as complementary to the classical *"what to build"* question that has dominated three decades of index-advising research. We propose HSM (Hierarchical Similarity Measurement), a training-free, DBMS-agnostic decision layer that gates advisor invocation in Θ(N_pts) time via a calibrated threshold θ*(N,Q). The framework rests on a formal core of six theorems and four lemmas covering metric soundness, a tight Θ(N_pts) complexity bound, economic gating optimality, stratified-Hoeffding detector-quality bounds, a deployment speedup certificate, and a convex-optimal window size. Empirical validation spans five workloads (TPC-H at SF 0.2–3.0, SDSS, JOB/IMDB, pgbench, and a burst workload), two database engines (PostgreSQL 16 and MongoDB 7), and two deployed production advisors (Dexter and Supabase `index_advisor`). On TPC-H, HSM attains Youden *J* ≥ 0.96 at every scale factor and reduces advisor invocations by 80.8% versus always-on and 42.5% versus periodic (*K*=3) rebuilding, at a kernel cost of approximately 1 ms per decision. End-to-end deployment with the two production advisors yields 66–71% wall-clock savings at 100% trigger precision.

**Relation to prior work.** The conceptual basis of HSM appeared in two preliminary conference venues — Reungsinkonkarn 2026 (iEECON; presented March 2026, in press, DOI forthcoming) and Reungsilpkolkarn 2026 (ICICT; presented March 2026, DOI 10.1145/3803291.3803321 assigned, indexing in progress) — which demonstrated classification accuracy and runtime improvements of the similarity concept. Camera-ready PDFs of both prior papers are available on request to assist editorial assessment of the present submission's incremental contribution. The present manuscript extends that prior work substantially in three dimensions: (i) a **formal theoretical core** — six theorems and four lemmas with complete proofs in the supplementary — that was absent from the conference versions; (ii) **cross-engine validation on MongoDB 7**, establishing generality across the relational–document data-model boundary, not merely across engines within the relational family; and (iii) **end-to-end deployment** with two production PostgreSQL advisors, moving the evidence base beyond synthetic comparisons. No proofs, cross-engine results, or end-to-end deployment data from the present submission appear in either conference version.

**Note on author name across publications.** The two prior conference papers cited above appear under the surname transliteration *Reungsinkonkarn*. Current legal spelling per the author's passport is *Reungsilpkolkarn*, used throughout this submission. Both spellings refer to the same person and are linked under ORCID 0009-0009-2077-0006, with the older spelling recorded in the ORCID "also known as" field. This note is provided proactively to avoid any appearance of citation inconsistency.

**Positioning in the TKDE literature.** HSM is explicitly positioned as complementary to — not competing with — existing index-advising systems such as BALANCE (Wang 2024), Indexer++ (Sharma 2022), Dexter, Supabase `index_advisor`, and the classical DTA line. These advisors answer *which* indexes to build; HSM answers *when* the advisor should be invoked. The two families compose: any advisor can be gated by HSM to avoid redundant invocation during stable-workload phases while preserving responsiveness at phase boundaries. Section V.G of the manuscript makes this orthogonality explicit. HSM is not proposed as a replacement for machine-learning-based advisors but as a decision layer that any advisor can plug into without retraining.

**Code and data availability.** A replication package comprising approximately 15,000 lines of experimental infrastructure, the shipped result CSVs that back every figure and table in the paper, a single-command reproducibility script (`run_all.sh`), and the TPC-H / SDSS / JOB / pgbench / burst harnesses is publicly available at <https://github.com/stevezhai139/HSM_gated_5> under a BSD-3-Clause license. The repository includes a five-minute reviewer quickstart (CPU-only path, no database required), the metric-axiom unit tests for Lemma 1, and step-by-step reproduction instructions for the PostgreSQL 16 and MongoDB 7 end-to-end runs in `REPRODUCE.md`. A versioned snapshot will be archived at Zenodo upon acceptance to provide a citable DOI.

**Declarations.**

- *Originality.* This manuscript is not under consideration at any other venue and has not been published elsewhere, except for the preliminary conference reports cited above.
- *Competing interests.* The author declares no competing financial or non-financial interests.
- *Generative-AI assistance.* Consistent with IEEE's policy on generative-AI disclosure, the Acknowledgments section of the manuscript states: *"This work used generative AI tools per IEEE guidelines: Claude (Anthropic) for code review, formula verification, and language checking; SciSpace for literature search and writing polish."* All theorems, proofs, experimental designs, code, and data interpretations are the author's own work.
- *Funding.* No external funding was received for this work.

**Suggested reviewers.** I respectfully defer reviewer selection to the editorial team. If preferred-reviewer suggestions would assist the editorial process, I can provide a short list of researchers active in index advising, workload characterisation, and database self-tuning upon request.

Thank you for considering this submission. I look forward to the editorial and reviewer assessment.

Sincerely,

**Arun Reungsilpkolkarn**
School of Information Technology and Innovation
Bangkok University
Pathum Thani, Thailand
Email: arun.r@bu.ac.th
ORCID: 0009-0009-2077-0006
