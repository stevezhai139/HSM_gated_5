# `cal/vendored/` — vendored Paper 3A kernel for Paper 3B-Cal

**Status.** Frozen snapshot of Paper 3A's kernel at `v5.0.0-tkde-submission`.
Do not edit these files. Sync only via the protocol in §4 below.

**Date vendored.** 2026-04-23  
**Author.** Arun Reungsilpkolkarn (ORCID 0009-0009-2077-0006)  
**Companion docs:**

- `Paper3B_Cal_Design_v0.md` §8 (Integrity rules — Paper 3A isolation)
- `Paper3B_Cal_Theoretical_Foundations_v0.docx` §6 (Paper 3A continuity)
- `code/experiments/cal/vendored/CHECKSUMS.txt` (integrity verification)

---

## 1. Why vendor instead of import live Paper 3A code?

Paper 3A is under TKDE peer review (TKDE-2026-04-1369). If TKDE requests
major revision, Paper 3A's source files may evolve on `revision-r1`,
`revision-r2`, … branches. Three concrete scenarios motivate vendoring:

**S1 (safe).** Reviewer asks to add a new validation script — does not
touch kernel; vendoring not needed.

**S2 (critical).** Reviewer asks to fix a bug in `hsm_v2_kernel.py`
(e.g., a numerical-stability correction in `sp_v2`). Paper 3A's live
kernel changes; Paper 3B's empirical results would change mid-study
unless isolated.

**S3 (data).** Reviewer asks to re-run experiments at a different TPC-H
scale factor; stored result CSVs in `code/results/` change. Paper 3B's
validation harness (which reads these CSVs) would see different data.

Vendoring guards against S2 completely and makes S3 observable
(checksum audit flags drift). Paper 3B's claims remain reproducible
against a fixed kernel regardless of Paper 3A revision cycles.

**Precedent:** Paper 3A itself vendored `cross_engine/` from V3 for its
§V cross-engine validation claim (design doc §14 Traceability). Same
pattern applies here.

---

## 2. What is vendored

Three Python modules, copied verbatim from `v5.0.0-tkde-submission`:

| File | Line count | Purpose | Byte-identical to v5.0.0? |
|---|---|---|---|
| `hsm_v2_kernel.py` | 413 | Canonical 5-dimension HSM kernel | YES |
| `hsm_similarity.py` | 456 | QueryFeatures / WorkloadWindow / hsm_score wrapper | NO — 1-line import rewrite (see §3) |
| `workload_generator.py` | 644 | TPC-H phase-shifted trace generator | YES |

**Not vendored (kept as live references, read-only):**

- `code/data/sdss/SkyLog_Workload.csv` — data file, too large to vendor, read-only by design
- `code/results/**/*.csv` — Paper 3A stored results; validation harness reads these; drift under S3 is DETECTABLE by the harness's own provenance-stamped outputs (future work: add CSV checksum to provenance when S3 becomes a live concern)
- `code/experiments/tier2/advisor_wrappers.py` — vendor when RQ2 pipeline is implemented (deferred per design doc §10)

---

## 3. Modifications applied during vendoring

Only ONE file was modified: `hsm_similarity.py`, a single import line:

```diff
- from hsm_v2_kernel import (W0 as _W0, sa_v2, sp_v2, sr_v2, st_v2, sv_v2)
+ from .hsm_v2_kernel import (W0 as _W0, sa_v2, sp_v2, sr_v2, st_v2, sv_v2)
```

The `.` prefix makes the import package-relative, so `hsm_similarity`
resolves to the VENDORED `hsm_v2_kernel` (not Paper 3A's live one).
Without this edit, Python's import machinery would fall back to the
live Paper 3A file, defeating the entire isolation purpose.

Everything else in `hsm_similarity.py` is byte-identical to the v5.0.0
source. `hsm_v2_kernel.py` and `workload_generator.py` are fully
byte-identical (see `CHECKSUMS.txt`).

---

## 4. Sync protocol (when to re-vendor)

Re-vendoring is a deliberate, reviewed action — NOT automatic.

### 4.1 When to consider re-vendoring

- Paper 3A's TKDE revision merges a fix that Paper 3B semantically
  depends on (e.g., kernel bug fix that affects HSM scores)
- Paper 3A's data pipeline produces updated CSVs that Paper 3B's
  validation harness interprets differently

### 4.2 When NOT to re-vendor

- Paper 3A revision is cosmetic (docstring / typo / paper text)
- Revision adds new files Paper 3B doesn't use
- Paper 3B is close to DASFAA submission (Jan 2027) — freeze instead

### 4.3 Re-vendor procedure

1. **Identify the source tag** (e.g., `v5.0.1-tkde-revision-1`).
2. **Copy each file** from the new tag into `cal/vendored/`:
   ```bash
   for f in hsm_v2_kernel.py hsm_similarity.py workload_generator.py; do
     git show <new_tag>:code/experiments/$f > code/experiments/cal/vendored/$f
   done
   ```
3. **Re-apply the 1-line import fix** in `hsm_similarity.py`
   (see §3 above).
4. **Update `CHECKSUMS.txt`** with new SHA256 values and the new
   source tag name in each row's annotation.
5. **Run the full test suite**:
   ```bash
   cd code/experiments && python -m pytest cal/tests/ cal/validation/tests/ -v
   ```
   All existing tests must still pass. If any fail, that is a signal
   that the revised kernel has semantic differences Paper 3B cares
   about — investigate before accepting.
6. **Run functional-equivalence check** (Deliverable 2 §6.3 spec):
   compare Paper 3B's outputs using vendored old vs vendored new on a
   fixed seed. Report deltas in the commit message.
7. **Commit** with message of the form:
   ```
   Paper 3B-Cal: resync vendored/ with Paper 3A <new_tag>

   - hsm_v2_kernel.py: <describe differences vs previous vendor>
   - hsm_similarity.py: <describe differences>
   - workload_generator.py: <describe differences>

   Functional-equivalence check: <pass/fail, with deltas>
   CHECKSUMS.txt updated.
   ```

### 4.4 Integrity verification (any time)

```bash
cd code/experiments/cal/vendored
sha256sum hsm_v2_kernel.py workload_generator.py
# Expected exactly:
#   e01a9d0ecf9af166348db2d46c102561bff3f6dee986cc23ee634a5a9ee949e9  hsm_v2_kernel.py
#   7a9e558408a119d4560c5446171f95b67680619f97b5e637111bdd8685295a1b  workload_generator.py
sha256sum hsm_similarity.py
# Expected exactly:
#   204891dc734e4a3602112f1b23ec71540c28a750d24a73a89cc61f2cf1947601  hsm_similarity.py
```

If any checksum differs, the vendored copy has drifted. Either:
- Revert local edits (`git checkout cal/vendored/<file>` if committed), or
- Re-vendor from `v5.0.0-tkde-submission` per §4.3.

---

## 5. Using the vendored kernel from Paper 3B code

Import path pattern:

```python
# From any module under cal/ (when sys.path includes code/experiments/):
from cal.vendored.hsm_v2_kernel import hsm_score_from_features, hsm_v2, W0
from cal.vendored.hsm_similarity import (
    WorkloadWindow, build_window, hsm_score, should_trigger_advisor,
    DEFAULT_THETA, DEFAULT_WEIGHTS,
)
from cal.vendored.workload_generator import get_workload_trace, PHASE_A, PHASE_B, PHASE_C
```

Paper 3B code MUST use these import paths. Do NOT write:

```python
# ❌ WRONG — bypasses vendoring, re-introduces Paper 3A coupling
from hsm_similarity import hsm_score
from hsm_v2_kernel import hsm_v2
from workload_generator import get_workload_trace
```

---

## 6. Relationship to Paper 3A source

The vendored files reference source files at:

- `code/experiments/hsm_v2_kernel.py` (Paper 3A, v5.0.0 tag)
- `code/experiments/hsm_similarity.py` (Paper 3A, v5.0.0 tag)
- `code/experiments/workload_generator.py` (Paper 3A, v5.0.0 tag)

Paper 3A's source files remain the **authoritative reference** for its
own submission. Paper 3B's vendored copies are the **stable snapshot**
against which Paper 3B's empirical claims are reproducible.

When Paper 3A is published (post-revision, DOI assigned), the vendored
files here continue to cite the pre-revision v5.0.0 state — this is
the desired behaviour, because Paper 3B's experiments and Lemmas were
derived from exactly that state.

---

*End of cal/vendored/README.md.*
