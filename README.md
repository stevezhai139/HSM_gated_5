# HSM: Workload-Similarity Gating for Index-Maintenance Decisions

Replication package for the TKDE submission
**"HSM: Workload-Similarity Gating for Index-Maintenance Decisions, with Formal Bounds"** (Arun Reungsilpkolkarn, 2026).

HSM is a training-free, DBMS-agnostic decision layer that gates index-advisor
invocation in `Θ(N_pts)` time. Deployed end-to-end with Dexter and Supabase
`index_advisor`, it delivers 66–71% wall-clock savings versus always-on with
100% trigger precision (§A13d of the paper).

---

## Reviewer Quickstart (5 minutes)

```bash
# 1. Set up Python env
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2. Verify metric-axiom unit tests (Lemma 1)
pytest -q code/tests/test_metric_properties.py         # ~5 s

# 3. Run CPU-only smoke experiments (no DB needed)
python code/experiments/overnight/b2_kernel_ablation.py         # ~5 min
python code/experiments/overnight/b4_inphase_perturbation.py    # ~1 min
python code/experiments/overnight/b5_noise_weight_grid.py       # ~1.5 h

# 4. Regenerate paper figures (uses shipped CSVs)
python code/figures/regenerate_all_figures.py .
```

**Expected outputs:** pytest passes (7 checks for P1–P4 + idempotency);
CSVs land in `results/overnight_<DATE>/`; PDFs land in
`code/figures/output/`.

Full reproduction (including PostgreSQL 16 + MongoDB 7 end-to-end runs)
is documented in [`REPRODUCE.md`](./REPRODUCE.md).

---

## Repository Structure

```
HSM_gated/
├── code/
│   ├── experiments/            # HSM kernel, theta*, per-workload runners
│   │   ├── hsm_similarity.py         # §III: five dims + composite
│   │   ├── hsm_v2_kernel.py          # Optimised DWT+SAX+FastDTW
│   │   ├── theta_star.py             # Theorem 3: θ*(N,Q)
│   │   ├── workload_generator.py     # Synthetic workloads
│   │   ├── experiment_runner.py      # TPC-H harness
│   │   ├── hsm_*_validation.py       # SDSS / JOB / burst / OLTP
│   │   ├── tier2/                    # End-to-end Dexter / index_advisor
│   │   └── overnight/                # B1–B5 long-running receipts
│   ├── figures/                # Plot scripts (7 main + 3 supplementary)
│   ├── tests/                  # Metric-axiom unit tests (Lemma 1)
│   ├── setup/                  # Database + workload setup scripts
│   ├── docker/                 # Postgres 16 / Mongo 7 containers
│   └── data/
│       ├── sdss/               # SkyLog_Workload.csv (31 MB, included)
│       └── job/queries/        # JOB/IMDB benchmark SQL (tracked)
├── paper/
│   ├── main/main_article.{tex,pdf}          # 12-page IEEE TKDE main
│   └── supplementary/supplementary.{tex,pdf}# 14-page appendix
├── results/                    # Shipped CSVs that back the paper tables
├── scripts/                    # Result aggregation helpers
├── .env.example                # DB connection template
├── requirements.txt
├── LICENSE                     # BSD 3-Clause
├── CITATION.cff                # Zenodo-compatible citation
├── REPRODUCE.md                # Step-by-step reproduction guide
└── README.md
```

**Datasets not bundled** (see [`REPRODUCE.md`](./REPRODUCE.md) for fetch
instructions): TPC-H `sf{0.2,1.0,3.0}` (.tbl files, 208 MB+), JOB `imdb`
raw dumps, burst end-to-end raw timeseries (216 MB, Zenodo mirror on
acceptance).

---

## Key Hyperparameters

All parameters are **fixed across every workload**; no per-workload
retraining (see paper §III.F, A10b).

| Parameter                  | Value                           | Paper reference |
|----------------------------|---------------------------------|-----------------|
| DWT wavelet                | `db4`                           | §III.E, Lemma 2 |
| DWT decomposition level    | `L = 3`                         | §III.E          |
| SAX alphabet cardinality   | `α = 4`                         | §III.E          |
| FastDTW Sakoe–Chiba band   | `r = 3`                         | §III.E          |
| Composite weight vector    | `(0.25, 0.20, 0.20, 0.20, 0.15)`| §III.F          |
| Default gating threshold   | `θ = 0.75`                      | §IV.C           |
| Window size (default)      | `N_pts = 100`                   | Theorem 6       |

---

## Paper ↔ Code Traceability

| Paper location          | Implementation                                          |
|-------------------------|---------------------------------------------------------|
| §III, Eq. (1)–(5)       | `code/experiments/hsm_similarity.py`                    |
| Lemma 1 (P1–P4)         | `code/tests/test_metric_properties.py`                  |
| Theorem 3, Eq. (7)–(8)  | `code/experiments/theta_star.py`                        |
| Theorem 4 (Hoeffding)   | `code/experiments/hsm_v2_kernel.py::hoeffding_band()`   |
| Theorem 5 (speedup)     | `scripts/analyze_t5_validation.py`                      |
| Table 3 (A13d)          | `code/experiments/tier2/hsm_burst_end_to_end.py`        |
| §V (A-CE cross-engine)  | `code/experiments/hsm_mongo_validation.py`              |
| Figures 1–7 (main)      | `code/figures/plot_fig0[1-7]_*.py`                      |
| Figures S1–S3 (suppl.)  | `code/figures/plot_supp0[1-3]_*.py`                     |

---

## Environment

- **Python 3.10+** (tested 3.10, 3.11, 3.12)
- **PostgreSQL 16** with HypoPG 1.4 (end-to-end advisor experiments only)
- **MongoDB 7** (A-CE cross-engine validation only)
- CPU-only smoke paths require neither PostgreSQL nor MongoDB.

See [`REPRODUCE.md`](./REPRODUCE.md) for Docker-based one-shot setup.

---

## Anonymised access during review

- Raw-dataset and large-log mirrors are hosted on Zenodo; the DOI is
  embedded in [`CITATION.cff`](./CITATION.cff) and will be activated on
  paper acceptance.
- During single-blind review, the repository is shared privately with
  the program chair; a read-only anonymised mirror is available on
  request.

---

## Reproducibility pledge

- All experiments are **deterministic** with fixed seeds (`0–9`).
- Hyperparameters are **locked** across workloads (see table above).
- CSV provenance: every figure/table in the paper cites the exact CSV
  in `results/` that backs it (§V of the paper).
- Unit tests (`pytest`) assert Lemma 1 metric properties on every
  commit via GitHub Actions (`.github/workflows/ci.yml`).

---

## Citation

```bibtex
@article{reungsilpkolkarn2026hsm,
  title   = {{HSM}: Workload-Similarity Gating for Index-Maintenance
             Decisions, with Formal Bounds},
  author  = {Reungsilpkolkarn, Arun},
  journal = {IEEE Transactions on Knowledge and Data Engineering},
  year    = {2026},
  note    = {Under review}
}
```

A machine-readable citation (CFF 1.2.0) is provided in
[`CITATION.cff`](./CITATION.cff) for GitHub's *Cite this repository*
button and Zenodo auto-archiving.

---

## License

BSD 3-Clause License © 2026 Arun Reungsilpkolkarn. See [`LICENSE`](./LICENSE).

---

## Troubleshooting

- **Figure regeneration says "CSV not found"** → the shipped CSVs in
  `results/` back every figure in the paper. Placeholder PDFs will be
  emitted for any experiment not yet re-run.
- **`pytest` import errors** → confirm Python 3.10+ and `pip install -r
  requirements.txt` inside a fresh virtualenv.
- **Database connection errors** → copy `.env.example` → `.env` and fill
  in local credentials. See `REPRODUCE.md` §3 for Docker-compose shortcuts.
- **`psycopg` not found** → `pip install psycopg[binary] pymongo` (both
  are optional dependencies for DB-backed experiments only).
