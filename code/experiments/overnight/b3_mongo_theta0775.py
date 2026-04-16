#!/usr/bin/env python3
"""
b3_mongo_theta0775.py — MongoDB at θ = 0.775 (PG-optimal, deliberately mis-applied).

Thin wrapper around V3's 14_mongo_adaptation_theta_sweep.py that overrides
THETA to the PostgreSQL-optimal 0.775. Expected outcome: Mongo under-fires
(few triggers, high missed drift) because 0.775 is above MongoDB's within-phase
tail (~0.78). Same seeds as the V3 G3 run so results are directly comparable
to the θ=0.75 and θ=0.65 MongoDB runs already in hand.

Output: Version 3/code/results/cross_engine/mongo/adaptation_theta0775/<ts>/
        {block_metrics.csv, breakdown_per_window.csv, run_meta.json}

Est. runtime: ~35 min.
"""
from __future__ import annotations

import os
import sys
import importlib.util
from pathlib import Path
from datetime import datetime

HERE = Path(__file__).resolve()
V5_REPO = HERE.parents[3]                         # .../Version 5/HSM_gated
V3_REPO = V5_REPO.parent.parent / "Version 3"     # .../Paper 3A/Version 3
V3_MONGO_DIR = V3_REPO / "code" / "experiments" / "cross_engine" / "mongo" / "adaptation"
V3_14_SCRIPT = V3_MONGO_DIR / "14_mongo_adaptation_theta_sweep.py"

if not V3_14_SCRIPT.exists():
    print(f"FATAL: cannot find {V3_14_SCRIPT}", file=sys.stderr)
    sys.exit(2)

# Dynamically load 14_ as a module, then patch THETA + RESULTS_ROOT before main().
spec = importlib.util.spec_from_file_location("v3_mongo_14", str(V3_14_SCRIPT))
mod = importlib.util.module_from_spec(spec)

# Prepend V3 mongo dirs to sys.path so its `from templates import ...` works.
sys.path.insert(0, str(V3_MONGO_DIR))
sys.path.insert(0, str(V3_REPO / "code" / "experiments" / "cross_engine" / "mongo" / "workload"))
sys.path.insert(0, str(V3_REPO / "code" / "experiments" / "cross_engine" / "common"))

spec.loader.exec_module(mod)

# ── Override θ and output dir ───────────────────────────────────────────────
mod.THETA = 0.775
mod.THETA_LABEL = "0.775"
new_root = V3_REPO / "code" / "results" / "cross_engine" / "mongo" / "adaptation_theta0775"
new_root.mkdir(parents=True, exist_ok=True)
mod.RESULTS_ROOT = str(new_root)

print(f"[{datetime.now():%H:%M:%S}] b3_mongo_theta0775.py  THETA={mod.THETA}")
print(f"  results root: {new_root}")

sys.exit(mod.main())
