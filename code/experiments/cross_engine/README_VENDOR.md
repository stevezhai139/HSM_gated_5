# `cross_engine/` — Vendored from V3 Snapshot

This directory is a **verbatim snapshot** of the cross-engine MongoDB
validation pipeline used to produce the §V (A-CE) results in the paper
*HSM: Workload-Similarity Gating for Index-Maintenance Decisions, with
Formal Bounds*.

## Why is it vendored?

The MongoDB cross-engine experiment was developed in an earlier internal
working tree (referred to here as "V3"). When the V5 reference
implementation was split out into the present `HSM_gated_5` repository,
the cross-engine subtree was not initially included because:

1. The V5 PostgreSQL pipeline uses a refactored kernel
   (`code/experiments/hsm_v2_kernel.py`) whose `sp_v2` signature and
   exported symbols differ from V3's `hsm_v2_core.py`.
2. The V5 kernel does **not** re-export `W0`, `extract_windows`,
   `compute_all_pairs`, etc. — symbols that the V3 mongo adaptation
   pipeline depends on.

Substituting the V5 kernel for V3's would change the numerical outputs
of the cross-engine experiment and invalidate the paper's reproducibility
claim. The safest fix is to vendor the V3 mongo subtree along with the
exact V3 kernel that produced those numbers.

## What is here

```
cross_engine/
├── README_VENDOR.md                        ← this file
├── _v3_hsm/
│   └── hsm_v2_core.py                      ← V3 HSM kernel (verbatim)
├── common/
│   ├── __init__.py                         ← verbatim
│   ├── hsm_bridge.py                       ← verbatim except _V2_DIR (now points to ../_v3_hsm)
│   ├── param_sampler.py                    ← verbatim
│   └── window_features.py                  ← verbatim
└── mongo/
    ├── adaptation/
    │   ├── 13_mongo_adaptation.py          ← verbatim (G3 baseline run)
    │   └── hsm_mongo_validation.py         ← renamed from V3's 14_mongo_adaptation_theta_sweep.py
    └── workload/
        └── templates.py                    ← verbatim
```

## How to run

```bash
# From repo root, with MongoDB 7 running locally:
python3 code/experiments/cross_engine/mongo/adaptation/hsm_mongo_validation.py --dry-run    # validate
python3 code/experiments/cross_engine/mongo/adaptation/hsm_mongo_validation.py              # full run (~35 min)

# Or via the convenience wrapper (overrides THETA to 0.775 — PG-optimal mis-applied):
python3 code/experiments/overnight/b3_mongo_theta0775.py
```

## Modifications from V3 originals

| File | Change | Reason |
|------|--------|--------|
| `common/hsm_bridge.py` | Line 30: `_V2_DIR = ... "v2_10seed"` → `... "_v3_hsm"` | Vendored kernel lives at sibling `_v3_hsm/` rather than V3's original `v2_10seed/` |
| `mongo/adaptation/14_mongo_adaptation_theta_sweep.py` | Renamed → `hsm_mongo_validation.py` | Match the README traceability table; signal that this is the canonical cross-engine validation entry-point |
| All other vendored files | None — byte-identical to V3 originals | Preserve numerical reproducibility |

## Provenance

Source tree: `Paper 3A/Version 3/code/experiments/`
- `v2_10seed/hsm_v2_core.py`
- `cross_engine/common/{__init__.py, hsm_bridge.py, param_sampler.py, window_features.py}`
- `cross_engine/mongo/adaptation/{13_mongo_adaptation.py, 14_mongo_adaptation_theta_sweep.py}`
- `cross_engine/mongo/workload/templates.py`

Vendored: 2026-04-23, immediately after TKDE initial submission, to make
the public `HSM_gated_5` repository self-contained for cross-engine
reproduction.
