#!/usr/bin/env python3
"""
b3_mongo_theta0775.py — MongoDB at θ = 0.775 (PG-optimal, deliberately mis-applied).

Thin wrapper around the vendored hsm_mongo_validation.py that overrides
THETA to the PostgreSQL-optimal 0.775. Expected outcome: Mongo under-fires
(few triggers, high missed drift) because 0.775 is above MongoDB's within-phase
tail (~0.78). Same seeds as the V3 G3 run so results are directly comparable
to the θ=0.75 and θ=0.65 MongoDB runs already in hand.

Output: results/cross_engine/mongo/adaptation_theta0775/<ts>/
        {block_metrics.csv, breakdown_per_window.csv, run_meta.json}

Est. runtime: ~35 min.

Notes:
- The underlying script `hsm_mongo_validation.py` is a verbatim snapshot of
  V3's `14_mongo_adaptation_theta_sweep.py` (vendored into
  `code/experiments/cross_engine/mongo/adaptation/` for self-contained
  reproducibility — see `cross_engine/README_VENDOR.md`).
"""
from __future__ import annotations

import os
import sys
import importlib.util
from pathlib import Path
from datetime import datetime

HERE = Path(__file__).resolve()
V5_REPO = HERE.parents[3]                         # .../HSM_gated
CE_DIR = V5_REPO / "code" / "experiments" / "cross_engine"
MONGO_DIR = CE_DIR / "mongo" / "adaptation"
MAIN_SCRIPT = MONGO_DIR / "hsm_mongo_validation.py"

if not MAIN_SCRIPT.exists():
    print(f"FATAL: cannot find {MAIN_SCRIPT}", file=sys.stderr)
    sys.exit(2)

# Prepend cross_engine subdirs to sys.path so the vendored script can resolve
# `from templates import ...`, `from param_sampler import ...`, etc.
sys.path.insert(0, str(MONGO_DIR))
sys.path.insert(0, str(CE_DIR / "mongo" / "workload"))
sys.path.insert(0, str(CE_DIR / "common"))

# Dynamically load hsm_mongo_validation as a module, then patch THETA + RESULTS_ROOT before main().
spec = importlib.util.spec_from_file_location("hsm_mongo_validation", str(MAIN_SCRIPT))
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

# ── Override θ and output dir ───────────────────────────────────────────────
mod.THETA = 0.775
mod.THETA_LABEL = "0.775"
new_root = V5_REPO / "results" / "cross_engine" / "mongo" / "adaptation_theta0775"
new_root.mkdir(parents=True, exist_ok=True)
mod.RESULTS_ROOT = str(new_root)

print(f"[{datetime.now():%H:%M:%S}] b3_mongo_theta0775.py  THETA={mod.THETA}")
print(f"  results root: {new_root}")

sys.exit(mod.main())
