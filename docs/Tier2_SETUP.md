# Tier 2 End-to-End Setup (Phase 1)

This document covers the host-side installation needed for the Tier 2
end-to-end advisor experiments (OLTP and Burst, Phase 2). Everything in
Phase 1 runs on **your machine** against **your PostgreSQL 16 instance** —
nothing runs inside the sandbox.

## Prerequisites

| Component | Minimum version | How to check |
|---|---|---|
| PostgreSQL server | 14+ (16 recommended) | `psql --version` |
| PostgreSQL dev headers | matching major | `pg_config --version` |
| Ruby | **3.1+** (not macOS system Ruby 2.6) | `ruby --version` |
| build tools | `make`, `gcc`, `git` | `make --version` |
| Python | 3.9+ with `psycopg2-binary` | `python -c "import psycopg2"` |

On Debian/Ubuntu:
```bash
sudo apt-get install -y \
    postgresql-16 postgresql-server-dev-16 postgresql-client-16 \
    build-essential git ruby-full ruby-dev
pip install psycopg2-binary
# If your distro's Ruby is < 3.1, install rbenv or use rvm — see below.
```

On macOS (Apple Silicon): **do not use system Ruby** — it is 2.6.10 and
incompatible with `pgdexter`'s transitive `google-protobuf` gem.
```bash
brew install postgresql@16 ruby
export PATH="/opt/homebrew/opt/ruby/bin:$PATH"    # put Homebrew Ruby first
export PATH="$(ruby -e 'puts Gem.bindir'):$PATH"  # make `dexter` findable
pip install psycopg2-binary
```

## Step 1 — run the installer

```bash
cd HSM_gated
bash scripts/tier2/install_advisors.sh
```

The installer builds and installs:

1. **HypoPG** — `CREATE EXTENSION hypopg;` becomes available cluster-wide.
2. **Supabase `index_advisor`** — `CREATE EXTENSION index_advisor;` becomes
   available. Depends on HypoPG at the SQL level.
3. **Dexter** — `gem install pgdexter` provides the `dexter` CLI.

If you want the installer to also enable the extensions in your target DB,
set `HSM_DB` before running:

```bash
HSM_DB=tpch bash scripts/tier2/install_advisors.sh
```

Otherwise, enable them yourself once:

```bash
psql -d tpch -c "CREATE EXTENSION hypopg; CREATE EXTENSION index_advisor;"
```

## Step 2 — load the existing `.env`

The Tier-2 code reuses the env vars already documented in `.env.example`
at the repo root — you do **not** need to re-export anything.

```bash
cp .env.example .env       # first time only
# edit .env to fill in any passwords / non-default hosts
source .env
```

Which block is used depends on `--mode`:

| Mode | Variables | Port | Used by |
|---|---|---|---|
| `--mode docker` (default) | `HSM_DOCKER_HOST/PORT/DB/USER/PASSWORD` | 5433 | OLTP + Burst runners (matches `hsm_oltp_validation.py`, `hsm_burst_v*_validation.py`) |
| `--mode native` | `HSM_DB_HOST/PORT/NAME/USER/PASSWORD` | 5432 | Native TPC-H path (`experiment_runner.py`) |

Because Tier-2 targets OLTP + Burst, the **default is `docker`** — same as
the existing validation runners.

## Step 3 — run the smoke test

```bash
python scripts/tier2/smoke_test_advisors.py             # docker (5433)
python scripts/tier2/smoke_test_advisors.py --mode native   # if needed
```

It performs six checks and exits 0 only if all pass:

| # | Check | What it proves |
|---|---|---|
| 1 | `psycopg2` connect | env vars resolve to a reachable server |
| 2 | `hypopg` enabled in DB | `CREATE EXTENSION` ran |
| 3 | `index_advisor` enabled in DB | extension installed and enabled |
| 4 | `dexter` CLI on PATH | Ruby gem installed and visible |
| 5 | `DexterAdvisor.recommend` returns a list | CLI runs against the DB |
| 6 | `SupabaseIndexAdvisor.recommend` returns a list | extension usable via SQL |
| 7 | `apply_recommendations` runs | DROP/CREATE path intact |

For the probe queries used here (`SELECT 1`, etc.), it is expected that the
advisors return **zero** recommendations — the check verifies that the
machinery is wired, not that the recommendations are useful.

## Next steps

Once the smoke test is green, move to Phase 2.

### Phase 2a — OLTP end-to-end  (`hsm_oltp_end_to_end.py`)

Before the first run, initialise the `oltp` database inside the Docker
container with the pgbench TPC-B schema at scale 10 (~1 M accounts):

```bash
# once, from the repo root after `source .env`
createdb   -h "$HSM_DOCKER_HOST" -p "$HSM_DOCKER_PORT" \
           -U "$HSM_DOCKER_USER" oltp
pgbench -i -s 10 \
           -h "$HSM_DOCKER_HOST" -p "$HSM_DOCKER_PORT" \
           -U "$HSM_DOCKER_USER" oltp
psql       -h "$HSM_DOCKER_HOST" -p "$HSM_DOCKER_PORT" \
           -U "$HSM_DOCKER_USER" -d oltp \
           -c "CREATE EXTENSION IF NOT EXISTS hypopg;" \
           -c "CREATE EXTENSION IF NOT EXISTS index_advisor;"
```

Then:

```bash
# wiring test — 1 block, Dexter only, ~3 min
python code/experiments/tier2/hsm_oltp_end_to_end.py --smoke

# full run — 10 blocks × 7 conditions (baseline + 3 policies × 2 advisors)
python code/experiments/tier2/hsm_oltp_end_to_end.py
```

Raw results land in `results/tier2_oltp/oltp_tier2_raw.csv` and per-window
detail JSON files in `results/tier2_oltp/detail/`.

### Phase 2b — Burst end-to-end  (`hsm_burst_end_to_end.py`)

Same factorial on a burst\_v2-style synthetic workload that reuses the
pgbench `oltp` database from Phase 2a — no additional schema initialisation
is required if Phase 2a ran successfully.

**Design:** 3 phases × 35 queries (Steady\_Point → Burst\_Alt → Burst\_Grp),
window size W=5 → 21 windows, true transitions at windows {7, 14}.  Periodic
policy is swept over K ∈ {2, 3, 4} to demonstrate that no single K lucks
into aligning with the transitions, so results do not depend on a chosen K.

Full factorial:

| # | Condition                   | Policy     | Advisor       |
|--:|-----------------------------|------------|---------------|
| 1 | baseline                    | baseline   | —             |
| 2 | always\_on\_dexter          | always\_on | dexter        |
| 3 | always\_on\_supabase        | always\_on | supabase      |
| 4 | hsm\_gated\_dexter          | hsm\_gated | dexter        |
| 5 | hsm\_gated\_supabase        | hsm\_gated | supabase      |
| 6–11 | periodic\_K{2,3,4}\_{dexter,supabase} | periodic | 2 advisors × 3 K  |

= 11 conditions × 10 blocks = **110 runs** per full execution.

```bash
# wiring test — 1 block × Dexter × {baseline, always_on, hsm_gated}, ~1 min
python code/experiments/tier2/hsm_burst_end_to_end.py --smoke

# full run — 10 blocks × 11 conditions (baseline + 2 policies × 2 advisors
# + periodic K-sweep {2,3,4} × 2 advisors)
python code/experiments/tier2/hsm_burst_end_to_end.py

# lighter variant — drop the K-sweep (only K=3), 7 conditions per block
python code/experiments/tier2/hsm_burst_end_to_end.py --no-k-sweep
```

Raw results land in `results/tier2_burst/burst_tier2_raw.csv` and per-window
detail JSON files in `results/tier2_burst/detail/`.

#### Extractor patches for pgbench (2026-04-15)

Three extractor-side changes were made so `hsm_similarity.extract_features`
would produce non-degenerate `{table, column}` sets on pgbench TPC-B
queries, closing a drift vs paper §III-B "relational extractor":

1. **`KNOWN_TABLES`** now includes pgbench tables (`pgbench_accounts`,
   `pgbench_branches`, `pgbench_tellers`, `pgbench_history`).
2. **`KNOWN_COLUMNS`** adds a whole-word match layer for the pgbench
   column vocabulary (`aid`, `bid`, `tid`, `abalance`, `bbalance`,
   `tbalance`, `delta`, `mtime`, `filler`).  TPC-H/JOB regex paths
   remain unchanged so prior validation scripts are unaffected.
3. **`canonicalize()`** now normalises signed numeric literals
   (`-1000` → `?`) so `WHERE x BETWEEN -500 AND 500` and
   `WHERE x BETWEEN 500 AND 1000` canonicalise to the same template.

Paired runner change in `hsm_burst_end_to_end.py`: the PK probe uses the
realistic OLTP pattern `SELECT abalance FROM pgbench_accounts WHERE
aid = ?` (instead of `SELECT aid, bid, abalance ...`) so the column
set genuinely differs across phases and `S_A` becomes informative.

#### Expected behaviour & C2 interpretation (Theorem 3 receipt)

With the patches above, HSM sees the Phase 2b transitions clearly — at
the P1→P2 boundary HSM = 0.892, at the P2→P3 boundary HSM = 0.781 (vs
1.000 intra-phase) — but does **not** cross θ = 0.75.  This is by
design, and matches the closed-form θ*(N, Q) from paper Theorem 3:

| Per-phase Q | Q_min(10⁶) | θ*(N,Q) | Advisor decision |
|---:|---:|---:|:---|
| 5   | ~122 | 0.00 | retain (always) |
| 35  | ~122 | 0.00 | retain (always) |
| 500 | ~122 | 0.76 | fires only if HSM < 0.76 |
| 5000| ~122 | 0.98 | fires only if HSM < 0.98 |

Phase 2b's per-phase budget (35 queries) is below Q_min, so every rebuild
is economically unamortisable.  HSM_gated correctly retaining is the
optimal decision — this is reported as a **positive result** in the
paper, not a miss.  The TCO receipt is visible in
`burst_tier2_raw*.csv`:

* `total_advisor_time_s = 0` for hsm_gated (zero wasted advisor calls)
* `total_advisor_time_s > 0` for always_on (real overhead, no p95 win)
* `indexes_created ≥ 1` per trigger for always_on (pure cost since
  Q < Q_min)

To analyse post-hoc optimality, call
`hsm_similarity.optimal_theta(N, Q, a, b, f, g)` after calibrating
`a, b, f, g` from a standalone run — see the Phase 2c analysis
notebook.

## Troubleshooting

* `Ruby X.Y is too old for pgdexter` → on macOS the system Ruby (2.6.10)
  cannot satisfy `google-protobuf >= 3.25`. Run `brew install ruby` then
  `export PATH="/opt/homebrew/opt/ruby/bin:$PATH"` and re-run the installer.
* `google-protobuf requires Ruby version >= 3.1` (from `gem install pgdexter`)
  → same root cause as above; fix the Ruby version before retrying.
* `pg_config not found` → install `postgresql-server-dev-<major>`.
* `dexter: command not found` after install → check `gem environment gemdir`;
  add `$(ruby -e 'puts Gem.bindir')` to PATH.
* `hypopg.control missing` after install → the gem/extension install landed
  in a non-default prefix; re-run `make install` with the right `PG_CONFIG`
  or `PATH=/usr/lib/postgresql/16/bin:$PATH`.
* `SupabaseIndexAdvisor` errors with `function index_advisor(text) does not exist`
  → you installed the extension but forgot `CREATE EXTENSION index_advisor;`
  in the specific database.
* Dexter CLI works standalone but `DexterAdvisor.recommend` returns empty
  → Dexter requires `hypopg` in the target DB; re-run `CREATE EXTENSION hypopg;`
  in the database that `HSM_DB_NAME` points to.
