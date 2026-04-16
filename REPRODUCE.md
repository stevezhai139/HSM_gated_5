# Reproducing the HSM paper

Step-by-step guide for reproducing every claim, figure, and table in
*HSM: Workload-Similarity Gating for Index-Maintenance Decisions, with
Formal Bounds* (Reungsilpkolkarn, TKDE 2026).

All experiments are deterministic; seeds are hard-coded in the runners.

---

## 0. Hardware baseline

Reference hardware used for paper timings:

- 8 × Intel Xeon vCPU @ 3.0 GHz, 32 GB RAM, NVMe SSD
- Ubuntu 22.04.4 LTS, Linux 6.2
- PostgreSQL 16.2 with HypoPG 1.4
- MongoDB 7.0.5
- Python 3.11.7

Smoke-test paths (§2) run comfortably on a 4-core laptop with 16 GB RAM.

---

## 1. Python environment

```bash
python3 -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt

# Unit tests first — if these fail, stop and debug.
pytest -q code/tests/test_metric_properties.py
# Expected: 7 passed in ~5 s
```

---

## 2. CPU-only smoke reproductions (no database needed)

These paths use only the shipped CSV fixtures and validate the
theoretical core (Lemmas 1–4, Theorems 1–2, 4–6 partial).

| Step | Command                                                                   | Wall time | Produces                                                                                          | Paper reference              |
|------|---------------------------------------------------------------------------|-----------|---------------------------------------------------------------------------------------------------|------------------------------|
| 2.1  | `pytest -q code/tests/`                                                   | ~5 s      | 7 assertions for P1–P4 + idempotency                                                              | Lemma 1                      |
| 2.2  | `python code/experiments/overnight/b2_kernel_ablation.py`                 | ~5 min    | `results/overnight_<DATE>/b2_kernel_ablation/kernel_ablation_{scores,summary}.csv`                | Table 5 kernel rows; B2      |
| 2.3  | `python code/experiments/overnight/b4_inphase_perturbation.py`            | ~1 min    | `results/overnight_<DATE>/b4_inphase/inphase_{amplitude,summary}.csv`                             | §A-CE (amplitude invariance) |
| 2.4  | `python code/experiments/overnight/b5_noise_weight_grid.py`               | ~1.5 h    | `results/overnight_<DATE>/b5_noise_weight/noise_weight_grid.csv`                                  | A10, A10b                    |
| 2.5  | `python code/figures/regenerate_all_figures.py .`                         | ~1 min    | `code/figures/output/fig0{1..7}_*.pdf` and `supp0{1..3}_*.pdf`                                    | All figures                  |

**Checkpoint:** Compare 2.2 `kernel_ablation_summary.csv` to the shipped
baseline in `results/overnight_2026-04-16/b2_kernel_ablation/`; Youden J
values should match within the reported CIs.

---

## 3. Full experiment suite (requires PostgreSQL + MongoDB)

### 3.1 Database setup (Docker, ~10 min first time)

```bash
cd code/docker
docker compose up -d               # spins up PostgreSQL 16 + HypoPG 1.4 + MongoDB 7
../setup/00_compile_and_generate.sh    # builds TPC-H dbgen + generates sf0.2/1.0/3.0
../setup/02_load_data.sh               # loads TPC-H into PostgreSQL
bash ./load_imdb.sh                # loads JOB/IMDB into PostgreSQL (~10 min, 4 GB)
```

Copy `.env.example` → `.env` and set database credentials. The default
`docker compose` values work out of the box.

### 3.2 Install deployed advisors

```bash
bash scripts/tier2/install_advisors.sh     # Dexter + Supabase index_advisor
```

### 3.3 Per-workload runners

| ID   | Command                                                                    | Wall time | Produces (in `results/`)                                         | Paper          |
|------|----------------------------------------------------------------------------|-----------|------------------------------------------------------------------|----------------|
| A1–A7| `python code/experiments/experiment_runner.py`                             | ~30 min   | `summary_all.csv`, `complexity.csv`, `overhead.csv`              | §V.A           |
| A8a  | `python code/experiments/hsm_sdss_validation.py`                           | ~10 min   | `sf0.2/sdss_*.csv`                                               | A8 SDSS        |
| A8b  | `python code/experiments/hsm_job_validation.py`                            | ~20 min   | `sf0.2/job_*.csv`                                                | A8 JOB/IMDB    |
| A8c  | `python code/experiments/hsm_oltp_validation.py`                           | ~15 min   | `tier2_oltp/*.csv`                                               | A8 pgbench     |
| A8d  | `python code/experiments/hsm_burst_v2_validation.py`                       | ~5 min    | `sf0.2/burst_v2_*.csv`                                           | A8 burst       |
| A12  | `python scripts/bench_complexity.py`                                       | ~20 min   | `complexity.csv`, `scale_analysis.csv`                           | Theorem 6      |
| A13d | `python code/experiments/tier2/hsm_burst_end_to_end.py`                    | ~2 h      | `burst_end_to_end/burst_{small,mid,large}_q_results.csv`          | Table 3        |
| A-CE | `python code/experiments/hsm_mongo_validation.py`                          | ~15 min   | `sf0.2/mongo_*.csv`                                              | §V cross-eng.  |

### 3.4 Overnight batch (B1–B5)

```bash
bash code/experiments/overnight/run_overnight_batch.sh
# ~7.5 h wall-clock
python code/experiments/overnight/analyze_overnight.py
```

Produces the B1–B5 receipts cited in §V.B and the supplementary.

### 3.5 Regenerate all figures with fresh CSVs

```bash
python code/figures/regenerate_all_figures.py .
ls code/figures/output/    # 10 PDFs matching paper figures 1–7 + S1–S3
```

---

## 4. Expected headline numbers

Cross-check against the shipped CSVs after a full re-run.

| Claim (paper §)                        | Expected                                     | CSV                                                    |
|----------------------------------------|----------------------------------------------|--------------------------------------------------------|
| Youden J across SF 0.2–3.0 (C4)        | `J ≥ 0.9626` every scale                      | `results/scale_analysis.csv`                           |
| Advisor savings vs always-on (A13)     | `80.8%`                                      | `results/throughput.csv`                               |
| Advisor savings vs periodic K=3 (A13)  | `42.5%`                                      | `results/throughput.csv`                               |
| HSM kernel overhead (A9)               | `1.09 ms` per transition                     | `results/overhead.csv`                                 |
| CETS vs always-on (A13d mid-Q, Dexter) | `0.665 [0.64, 0.69]`                         | `results/burst_end_to_end/burst_mid_q_results.csv`     |
| Trigger precision (A13d mid-Q)         | `1.000 [1.000, 1.000]`                       | `results/burst_end_to_end/burst_mid_q_results.csv`     |
| Theorem 5 speedup bound (A13c TPC-H)   | `5.20×`                                      | `results/t5_validation.csv`                            |
| p_stable (measured, TPC-H)             | `0.808`                                      | `results/throughput.csv`                               |
| Noise floor (B1, burst)                | `S_floor ≈ 0.875`                            | `results/theta_optimal_per_workload.csv`               |

Bootstrap CIs should overlap at 95% confidence with the shipped values;
point estimates may drift by `< 1%` due to hardware jitter.

---

## 5. Continuous integration

Every push runs the Lemma 1 metric-property tests plus a CPU-only smoke
path via `.github/workflows/ci.yml`. Green CI on `main` is the baseline
reproduction contract.

---

## 6. Where datasets come from

| Dataset                 | Bundled? | How to fetch                                          |
|-------------------------|----------|-------------------------------------------------------|
| SDSS `SkyLog_Workload`  | ✓ (31 MB) | `code/data/sdss/SkyLog_Workload.csv` (shipped)       |
| JOB benchmark queries   | ✓         | `code/data/job/queries/*.sql` (shipped)              |
| JOB IMDB raw data       | ✗         | `code/docker/load_imdb.sh` pulls from JOB release     |
| TPC-H `sf{0.2,1.0,3.0}` | ✗         | `code/setup/00_compile_and_generate.sh` (uses dbgen)  |
| Burst raw timeseries    | ✗         | Zenodo archive (DOI in `CITATION.cff`, live on accept)|
| pgbench OLTP            | ✗ (synth) | Generated on the fly by `code/docker/load_data.sh`    |

---

## 7. Help and issues

- Inspect `results/_run_logs/` for per-experiment trace logs after a run.
- File an issue with the failing command and the head/tail of the run log.
- For reviewers: a private repository URL will be shared alongside the
  manuscript; the public archive goes live on acceptance with the Zenodo
  DOI.
