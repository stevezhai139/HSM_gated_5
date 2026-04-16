# HSM: Hierarchical Similarity Measurement for Proactive Database Index Tuning

Second-generation implementation of the Hierarchical Similarity Measurement (HSM) framework for proactive database index tuning using multi-scale workload similarity. This repository contains the complete replication package for the TKDE paper, including core algorithms, experimental validation scripts, and result generation pipelines.

> *Multi-Scale Workload Similarity Measurement for Proactive Database Index
> Tuning Using Hierarchical Signal Processing* — Arun Reungsilpkolkarn.

## Repository Structure

```
HSM_gated/
├── code/
│   ├── experiments/               # Core HSM implementation and validation
│   │   ├── hsm_similarity.py      # Paper §III: five dimensions + composite scoring
│   │   ├── hsm_v2_kernel.py       # Optimized signal processing (DWT, SAX, FastDTW)
│   │   ├── theta_star.py          # Theorem 3: closed-form gating threshold θ*(N, Q)
│   │   ├── workload_generator.py  # Synthetic workload synthesis
│   │   ├── experiment_runner.py   # TPC-H validation harness
│   │   ├── hsm_*_validation.py    # Workload-specific runners (SDSS, JOB, burst, OLTP)
│   │   ├── tier2/                 # End-to-end real advisor integration
│   │   │   ├── hsm_burst_end_to_end.py     # Burst-pattern validation vs. Dexter/Supabase
│   │   │   ├── hsm_oltp_end_to_end.py      # OLTP validation vs. real advisors
│   │   │   └── advisor_wrappers.py         # PostgreSQL index advisor APIs
│   │   └── overnight/             # Long-running batch experiments (B1–B5)
│   │       ├── run_overnight_batch.sh      # Orchestrator for all overnight runs
│   │       ├── b1_burst_large_q.sh         # Burst validation with Q=3000
│   │       ├── b2_kernel_ablation.py       # Angular/cosine/euclidean ablation
│   │       ├── b3_theta_transfer.sh        # θ transfer PG↔Mongo validation
│   │       ├── b4_inphase_perturbation.py  # Amplitude perturbation sensitivity
│   │       ├── b5_noise_weight_grid.py     # Noise × weight 2-D grid
│   │       └── analyze_overnight.py        # Result aggregation and reporting
│   ├── figures/                   # Plot scripts and figure generation
│   │   ├── _style.py              # Matplotlib IEEE serif style configuration
│   │   ├── _placeholder.py        # Watermarked placeholder PDF generator
│   │   ├── plot_fig0[1-7]_*.py    # Main paper figures (7 plots)
│   │   ├── plot_supp0[1-3]_*.py   # Supplementary figures (3 plots)
│   │   └── regenerate_all_figures.py
│   ├── tests/                     # Unit and property tests
│   │   └── test_metric_properties.py  # Lemma 1 metric axiom validation
│   ├── setup/                     # Database and workload setup
│   └── docker/                    # Reproducible container images
├── paper/
│   ├── main/main_article.tex      # Main paper LaTeX source
│   └── supplementary/supplementary.tex
├── results/                       # Experiment outputs (CSVs, logs, manifests)
│   ├── *.csv                      # Result tables and summaries
│   ├── burst_end_to_end/          # End-to-end advisor comparison results
│   ├── tier2_oltp/                # OLTP validation results
│   └── overnight_*/               # Timestamped overnight batch artifacts
├── .env.example                   # Environment configuration template
├── requirements.txt               # Python dependencies
├── .gitignore                     # Git ignore rules
├── LICENSE                        # BSD 3-Clause License
└── README.md
```

## Key Hyperparameters

| Parameter                | Value               | Reference                |
|--------------------------|---------------------|--------------------------|
| DWT wavelet              | `db4`               | §III.E, Lemma 2          |
| DWT decomposition level  | `L = 3`             | §III.E                   |
| SAX alphabet cardinality | `α = 4`             | §III.E                   |
| FastDTW Sakoe-Chiba band | `r = 3`             | §III.E                   |
| Composite weight vector  | `(0.25, 0.20, 0.20, 0.20, 0.15)` | §III.F (PostgreSQL) |
| Gating threshold (default)| `θ = 0.75`          | §IV (TPC-H calibration)  |

All parameters are **fixed** across all workloads. No per-workload tuning (see Appendix A8, A10b).

## Prerequisites

- **Python 3.10+**
- **PostgreSQL 16** (for real workload validation; optional for figures-only)
- **MongoDB 7** (for cross-platform gating threshold validation; optional)

## Installation

```bash
# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate  # on Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# (Optional) For database-backed experiments, uncomment psycopg and pymongo in requirements.txt
# pip install psycopg[binary] pymongo
```

## Quick Start

### Verify Installation

```bash
# Run metric property tests (Lemma 1)
pytest -q code/tests/test_metric_properties.py
```

### Regenerate Figures

```bash
# Generate all 10 figures (7 main + 3 supplementary)
# Falls back to watermarked placeholders if CSV results are missing
python code/figures/regenerate_all_figures.py .
```

Outputs appear as PDF files in `code/figures/output/`.

## Reproducing Experiments

All experiments are deterministic (seeded 0–9) and emit CSV results to `results/`.
Plotting scripts automatically fall back to watermarked placeholders if results are unavailable.

### Quick Smoke Tests (CPU-only)

```bash
# Metric property validation (Lemma 1)
pytest -q code/tests/test_metric_properties.py

# Kernel ablation (angular/cosine/euclidean)
python code/experiments/overnight/b2_kernel_ablation.py

# In-phase perturbation sensitivity
python code/experiments/overnight/b4_inphase_perturbation.py

# Noise × weight grid
python code/experiments/overnight/b5_noise_weight_grid.py
```

### Full Experiment Suite (requires PostgreSQL + MongoDB)

Set up environment variables in `.env` (copy from `.env.example`):

```bash
# Configure database connections
export PGHOST=localhost PGPORT=5432 PGUSER=postgres PGPASSWORD=***
export MONGO_URI=mongodb://localhost:27017/hsm

# Run overnight batch (B1–B5, ~7.5 hours total)
bash code/experiments/overnight/run_overnight_batch.sh

# Analyze results
python code/experiments/overnight/analyze_overnight.py
```

### Individual Experiment Runs

| Experiment                     | Script                                          | Est. Time |
|--------------------------------|-------------------------------------------------|-----------|
| **TPC-H end-to-end**           | `python code/experiments/experiment_runner.py`  | ~30 min   |
| **SDSS workload**              | `python code/experiments/hsm_sdss_validation.py`| ~10 min   |
| **JOB queries**                | `python code/experiments/hsm_job_validation.py` | ~20 min   |
| **Burst pattern (v2)**         | `python code/experiments/hsm_burst_v2_validation.py` | ~5 min |
| **OLTP (pgbench)**             | `python code/experiments/hsm_oltp_validation.py` | ~15 min  |
| **Burst end-to-end (advisor)** | `python code/experiments/tier2/hsm_burst_end_to_end.py --smoke` | ~1 min |
| **OLTP end-to-end (advisor)**  | `python code/experiments/tier2/hsm_oltp_end_to_end.py --smoke` | ~2 min |

### Overnight Batch Details

The `run_overnight_batch.sh` orchestrator runs five long-running tasks:

| ID  | Experiment | Script | Time | Type |
|-----|-----------|--------|------|------|
| B1  | Burst with Q=3000 | `b1_burst_large_q.sh` | ~5.0 h | DB-bound |
| B2  | Kernel ablation (3 metrics) | `b2_kernel_ablation.py` | ~5 min | CPU |
| B3  | θ transfer (PG ↔ Mongo) | `b3_theta_transfer.sh` | ~1.0 h | DB-bound |
| B4  | Amplitude perturbation | `b4_inphase_perturbation.py` | ~1 min | CPU |
| B5  | Noise × weight 2-D grid | `b5_noise_weight_grid.py` | ~1.5 h | CPU |

Results and manifest saved to `results/overnight_<DATE>/`.

## Paper–Code Traceability

| Paper Location | Implementation | File |
|---|---|---|
| Equations (1)–(5), §III | HSM five-dimension metric | `code/experiments/hsm_similarity.py` |
| Lemma 1 | Metric properties (symmetry, triangle inequality) | `code/tests/test_metric_properties.py` |
| Theorem 3, Eq. (7)–(8) | Closed-form gating threshold | `code/experiments/theta_star.py` |
| Table 12 (A13d) | End-to-end advisor evaluation | `code/experiments/tier2/hsm_burst_end_to_end.py` |
| Appendix A1–A8 | Workload-specific validation runners | `code/experiments/hsm_*_validation.py` |
| Figures 1–7 | Main paper plots | `code/figures/plot_fig0[1-7]_*.py` |
| Supplementary 1–3 | Appendix plots | `code/figures/plot_supp0[1-3]_*.py` |

## Reproducibility Notes

- **Deterministic:** All experiments use fixed random seeds (0–9) for cross-run comparisons.
- **Parameter lock:** Weights, wavelet, and thresholds are **not** re-tuned across workloads (see Appendix A8, A10b).
- **Container support:** PostgreSQL 16 and MongoDB 7 images available in `code/docker/` for reproducible environments.
- **Result artifacts:** CSV files in `results/` are preserved; logs in `results/overnight_*/` are excluded from commits.

## Troubleshooting

**Figure regeneration fails with "CSV not found"**
→ This is expected if experiments haven't run yet. Placeholders will be used.

**Tests fail on import errors**
→ Verify Python 3.10+ and run `pip install -r requirements.txt`

**Database connection errors**
→ Check `.env` file and PostgreSQL/MongoDB connectivity. See `.env.example` for configuration.

## Citation

If you use this code, please cite:

```bibtex
@article{reungsilpkolkarn2026hsm,
  title={Multi-Scale Workload Similarity Measurement for Proactive Database 
         Index Tuning Using Hierarchical Signal Processing},
  author={Reungsilpkolkarn, Arun},
  journal={IEEE Transactions on Knowledge and Data Engineering},
  year={2026}
}
```

## License

BSD 3-Clause License. See `LICENSE` for details.
