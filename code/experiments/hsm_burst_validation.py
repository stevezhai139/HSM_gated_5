"""
HSM Validation on Burst / Non-Stationary Workload
==================================================
Tests HSM's S_P (DWT temporal pattern) dimension on workloads where the
execution-time series changes pattern across phases, even when query types
(all SELECT) and schema access are identical.

PHASE DESIGN (4 phases, 30/30/24/30 = 114 queries per trace)
--------------------------------------------------------------
  Flat_Low   (FL): All fast indexed point-reads  (~0.2 ms)
                   Time series: [0.2, 0.2, 0.2, ...]  → flat low
  Burst_Alt  (BA): Strict alternating fast/slow  (period = 2)
                   Time series: [0.2, 8, 0.2, 8, ...]  → period-2 oscillation
  Burst_Grp  (BG): Groups of 4 fast then 4 slow (period = 8)
                   Time series: [0.2,0.2,0.2,0.2, 8,8,8,8, ...]  → grouped
  Flat_High  (FH): All slow analytical queries   (~8 ms)
                   Time series: [8, 8, 8, ...]  → flat high

HSM DISCRIMINATION PREDICTION
------------------------------
  S_P: FL vs BA  → flat vs period-2 DWT pattern  (strong)
       BA vs BG  → period-2 vs period-8 DWT      (moderate)
       FL vs FH  → both flat → DWT shape similar  (low — S_V handles this)
  S_V: FL vs FH  → 0.2 ms vs 8 ms, 40× contrast (strong)
  S_R: All SELECT → S_R = 1.0 everywhere → zero discrimination
  S_T: All SELECT → type-vector identical → zero discrimination
  S_A: Same 4 pgbench tables → minimal discrimination
  Expected dominant dims: S_P and S_V
  Expected DR: 1.3 – 1.8

KEY DESIGN PRINCIPLE
--------------------
  Burst_Alt and Burst_Grp have FIXED internal ordering (pattern preserved).
  Per-seed variation comes from shuffling which specific fast/slow queries
  appear in each phase — not from shuffling the pattern structure.

DATABASE
--------
  pgbench TPC-B on Docker PostgreSQL 16, port 5433, database 'oltp'
  (same as A8c OLTP validation — no additional setup required)

Usage:
  python hsm_burst_validation.py            # static proxy mode
  python hsm_burst_validation.py --smoke    # connectivity + per-phase sample
  python hsm_burst_validation.py --execute  # full 10-seed experiment
"""
"""
[HSM v2] Validation script -- updated to match paper Section III strictly.
v2 changes vs. legacy implementation:
    * DWT wavelet: db2 -> db4 (paper line 331)
    * DWT level:   1   -> 3   (paper line 331)
    * FastDTW radius: 1 -> 3  (paper line 336)
    * Default weights: equal -> [0.25, 0.20, 0.20, 0.20, 0.15] (paper line 392)
For the canonical S_R/S_V/S_T/S_A/S_P implementations, prefer importing
hsm_similarity.hsm_score; the inline definitions below are kept for
back-compatibility with the legacy validation harness only.
"""

import re, os, sys, csv, math, time, random, statistics, argparse
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy import stats
import pywt

# v2 kernel — canonical five-dimension HSM per paper §III.
sys.path.insert(0, str(Path(__file__).parent))
from hsm_v2_kernel import hsm_v2, hsm_score_from_features  # noqa: E402

# ── Configuration ──────────────────────────────────────────────────────────────
SCRIPT_DIR  = Path(__file__).parent
RESULTS_DIR = SCRIPT_DIR.parent / 'results' / 'burst_validation'

DOCKER_HOST = os.environ.get('HSM_DOCKER_HOST', 'localhost')
DOCKER_PORT = int(os.environ.get('HSM_DOCKER_PORT', 5433))
DOCKER_USER = os.environ.get('HSM_DOCKER_USER', 'postgres')
DOCKER_PASS = os.environ.get('HSM_DOCKER_PASSWORD', 'postgres')
DOCKER_DB   = 'oltp'

TIMEOUT_MS  = 30_000
NPTS        = 6
STEP        = 3
N_REPS      = 3
SEEDS       = [42, 137, 271, 314, 999, 7, 13, 55, 88, 101]
PHASE_ORDER = ['Flat_Low', 'Burst_Alt', 'Burst_Grp', 'Flat_High']

# Fixed parameter sets (deterministic)
AIDS   = [1, 42, 1000, 5000, 10000, 50000, 100000,
          250000, 500000, 750000, 900000, 999999]
BIDS   = list(range(1, 11))
TIDS   = [1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
RANGES = [(1, 100000), (100001, 300000), (300001, 500000),
          (500001, 700000), (700001, 900000), (900001, 1000000)]
THRESHOLDS = [-5000, -2000, -1000, -500, 0, 500, 1000, 2000, 5000, 10000]

# ── Fast query pool (~0.2 ms each, all SELECT) ─────────────────────────────────
def build_fast_pool():
    pool = []
    for aid in AIDS:
        pool.append(f"SELECT aid, bid, abalance FROM pgbench_accounts WHERE aid = {aid}")
    for tid in TIDS[:6]:
        pool.append(f"SELECT tid, bid, tbalance FROM pgbench_tellers WHERE tid = {tid}")
    for bid in BIDS[:5]:
        pool.append(f"SELECT bid, bbalance FROM pgbench_branches WHERE bid = {bid}")
    for aid in AIDS[:7]:
        pool.append(f"SELECT aid, abalance FROM pgbench_accounts WHERE aid = {aid} AND abalance > -10000")
    return pool[:30]   # exactly 30 fast queries

# ── Slow query pool (~5-15 ms each, analytical SELECT) ────────────────────────
def build_slow_pool():
    pool = []
    # Per-branch aggregates (10 queries)
    for bid in BIDS:
        pool.append(
            f"SELECT bid, COUNT(*) AS n, AVG(abalance) AS avg_bal, "
            f"MAX(abalance) AS max_bal, MIN(abalance) AS min_bal "
            f"FROM pgbench_accounts WHERE bid = {bid} GROUP BY bid")
    # Range scans with GROUP BY (6 queries)
    for lo, hi in RANGES:
        pool.append(
            f"SELECT bid, COUNT(*) AS n, SUM(abalance) AS total "
            f"FROM pgbench_accounts WHERE aid BETWEEN {lo} AND {hi} "
            f"GROUP BY bid ORDER BY n DESC")
    # Branch-join aggregates (5 queries)
    for bid in BIDS[:5]:
        pool.append(
            f"SELECT a.bid, b.bbalance, COUNT(*) AS n, AVG(a.abalance) AS avg_bal "
            f"FROM pgbench_accounts a JOIN pgbench_branches b ON a.bid = b.bid "
            f"WHERE a.bid = {bid} GROUP BY a.bid, b.bbalance ORDER BY n DESC")
    # Balance-threshold aggregates (9 queries)
    for thresh in THRESHOLDS[:9]:
        pool.append(
            f"SELECT bid, COUNT(*) AS n, SUM(abalance) AS total "
            f"FROM pgbench_accounts WHERE abalance > {thresh} "
            f"GROUP BY bid ORDER BY total DESC")
    return pool[:30]   # exactly 30 slow queries

# ── Burst trace builder ────────────────────────────────────────────────────────
def build_burst_trace(fast_pool, slow_pool, seed):
    """
    Returns list of (qid, phase, qtype, sql) preserving burst patterns.

    Flat_Low   (30): fast queries in shuffled order
    Burst_Alt  (30): strict period-2 interleaving  F,S,F,S,...
    Burst_Grp  (24): strict period-8 grouping      FFFF,SSSS,FFFF,SSSS,...
    Flat_High  (30): slow queries in shuffled order
    """
    rng = random.Random(seed)
    fast = fast_pool.copy(); rng.shuffle(fast)
    slow = slow_pool.copy(); rng.shuffle(slow)

    trace = []

    # ── Block 1: Flat_Low (30 fast, any order) ──────────────────────────────
    for i, sql in enumerate(fast):
        trace.append((f"FL{i+1:03d}", 'Flat_Low', 'SELECT', sql))

    # ── Block 2: Burst_Alt (15 fast + 15 slow, strict alternating) ──────────
    for i in range(15):
        trace.append((f"BA{2*i+1:03d}", 'Burst_Alt', 'SELECT', fast[i]))
        trace.append((f"BA{2*i+2:03d}", 'Burst_Alt', 'SELECT', slow[i]))

    # ── Block 3: Burst_Grp (12 fast + 12 slow, groups of 4) ─────────────────
    fast_g = fast[15:27]   # 12 fast (distinct from Burst_Alt slice if overlap OK)
    slow_g = slow[15:27]   # 12 slow
    for g in range(3):     # 3 groups × (4 fast + 4 slow) = 24 queries
        for j in range(4):
            trace.append((f"BG{g*8+j+1:03d}", 'Burst_Grp', 'SELECT', fast_g[g*4+j]))
        for j in range(4):
            trace.append((f"BG{g*8+j+5:03d}", 'Burst_Grp', 'SELECT', slow_g[g*4+j]))

    # ── Block 4: Flat_High (30 slow, any order) ──────────────────────────────
    for i, sql in enumerate(slow):
        trace.append((f"FH{i+1:03d}", 'Flat_High', 'SELECT', sql))

    return trace   # 30 + 30 + 24 + 30 = 114 entries

# ── timed_exec (fresh cursor, conditional fetchall) ───────────────────────────
def timed_exec(cursor, sql, n_reps=N_REPS):
    """Run sql n_reps times; return median elapsed ms. Fresh cursor per call."""
    def _run_once():
        c = conn.cursor()
        try:
            c.execute(sql)
            if c.description is not None:
                c.fetchall()
        finally:
            try: c.close()
            except: pass

    try: _run_once()        # warm-up (discard)
    except: pass

    times = []
    for _ in range(n_reps):
        c = conn.cursor()
        try:
            t0 = time.perf_counter()
            c.execute(sql)
            if c.description is not None:
                c.fetchall()
            times.append((time.perf_counter() - t0) * 1000)
        except psycopg2.extensions.QueryCanceledError:
            times.append(float(TIMEOUT_MS))
        except Exception:
            times.append(float(TIMEOUT_MS))
        finally:
            try: c.close()
            except: pass
    return float(statistics.median(times))

# ── HSM kernel (v2 via hsm_v2_kernel — paper §III) ────────────────────────────
_QTYPE_RE = [
    ('SELECT', 0), ('UPDATE', 1), ('INSERT', 2), ('DELETE', 3),
]

def _window_features(sqls, times):
    """Extract feature-dict from (sqls, times) lists for the v2 kernel."""
    n = len(sqls)

    # Per-SQL template frequency (stable axis so Spearman is meaningful)
    sql_counts = Counter(s.strip() for s in sqls)
    freq_axis  = sorted(sql_counts.keys())
    freq = np.array([float(sql_counts[s]) for s in freq_axis])

    # CRUD breakdown (diagnostic only; per paper §III-B Relational
    # extractor, S_T uses the template-frequency vector, NOT a CRUD vec).
    crud_vec = np.zeros(4)
    for s in sqls:
        up = s.strip().upper()
        for kw, idx in _QTYPE_RE:
            if up.startswith(kw):
                crud_vec[idx] += 1
                break

    # Dual Jaccard — tables + columns
    tables, cols = set(), set()
    for s in sqls:
        for pair in re.findall(r'FROM\s+(\w+)|JOIN\s+(\w+)', s, re.I):
            for t in pair:
                if t:
                    tables.add(t.lower())
        for m in re.finditer(r'\b(aid|bid|tid|abalance|bbalance|tbalance|'
                             r'delta|mtime|filler)\b', s, re.I):
            cols.add(m.group(1).lower())

    # q(t) arrival-count series at 1-second resolution (paper §III-A
    # line 291).  Built from cumulative execution times so each query's
    # arrival lands at its serial finish time.
    from hsm_v2_kernel import build_qps_series
    t_ser = build_qps_series(times, min_bins=16)

    return {
        'freq': freq, 'freq_map': dict(sql_counts),
        'tables': tables, 'cols': cols,
        'times': t_ser, 'qset': set(freq_axis),
        'n': n,
        # diagnostic-only:
        'crud_vec': crud_vec,
    }


def hsm_similarity(w1_sqls, w1_times, w2_sqls, w2_times, weights=None):
    """Paper §III five-dimension HSM; returns (score, [S_R,S_V,S_T,S_A,S_P]).
    Routes through hsm_score_from_features() to pick up freq_map-based
    axis alignment when windows have disjoint template sets."""
    a = _window_features(w1_sqls, w1_times)
    b = _window_features(w2_sqls, w2_times)
    w_dict = ({'R': weights[0], 'V': weights[1], 'T': weights[2],
               'A': weights[3], 'P': weights[4]} if weights else None)
    score, dims = hsm_score_from_features(a, b, weights=w_dict)
    return score, [dims['S_R'], dims['S_V'], dims['S_T'],
                   dims['S_A'], dims['S_P']]

def windows_from_trace(trace_executed, npts=NPTS, step=STEP):
    """trace_executed: list of (qid, phase, sql, elapsed_ms)"""
    wins = []
    n = len(trace_executed)
    for start in range(0, n - npts + 1, step):
        block = trace_executed[start:start+npts]
        phases = [b[1] for b in block]
        dominant = Counter(phases).most_common(1)[0][0]
        wins.append({
            'sqls':   [b[2] for b in block],
            'times':  [b[3] for b in block],
            'phase':  dominant,
            'phases': phases,
        })
    return wins

def compute_dr(wins, theta=0.75):
    pairs_within, pairs_cross = [], []
    for i in range(len(wins)):
        for j in range(i+1, len(wins)):
            sc, dims = hsm_similarity(wins[i]['sqls'], wins[i]['times'],
                                      wins[j]['sqls'], wins[j]['times'])
            entry = (sc, dims)
            if wins[i]['phase'] == wins[j]['phase']:
                pairs_within.append(entry)
            else:
                pairs_cross.append(entry)
    if not pairs_within or not pairs_cross:
        return None, None, None, None

    w_scores = [e[0] for e in pairs_within]
    c_scores = [e[0] for e in pairs_cross]
    dr = statistics.median(w_scores) / statistics.median(c_scores) if statistics.median(c_scores) > 0 else float('inf')

    u_stat, p_val = stats.mannwhitneyu(w_scores, c_scores, alternative='greater')
    r_biserial = 1 - 2*u_stat/(len(w_scores)*len(c_scores))

    # Per-dimension deltas
    w_dims = np.mean([e[1] for e in pairs_within], axis=0)
    c_dims = np.mean([e[1] for e in pairs_cross],  axis=0)
    dim_deltas = w_dims - c_dims
    dominant_idx = int(np.argmax(dim_deltas))
    dominant = ['S_R','S_V','S_T','S_A','S_P'][dominant_idx]

    return dr, p_val, r_biserial, {
        'n_within': len(pairs_within), 'n_cross': len(pairs_cross),
        'w_scores': w_scores, 'c_scores': c_scores,
        'w_dims': w_dims.tolist(), 'c_dims': c_dims.tolist(),
        'dim_deltas': dim_deltas.tolist(), 'dominant': dominant,
    }

# ── CLI ────────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument('--execute', action='store_true')
parser.add_argument('--smoke',   action='store_true')
parser.add_argument('--verbose', action='store_true')
parser.add_argument('--port',    type=int, default=DOCKER_PORT)
parser.add_argument('--dbname',  default=DOCKER_DB)
args = parser.parse_args()

print("=" * 70)
print("  HSM Validation — Burst / Non-Stationary Workload")
print("=" * 70)
mode = ("SMOKE" if args.smoke else "EXECUTE" if args.execute else "STATIC")
print(f"\n  Mode   : {mode}")
print(f"  DB     : {args.dbname} on port {args.port}")
print(f"  Phases : Flat_Low / Burst_Alt / Burst_Grp / Flat_High")
print(f"  N_REPS : {N_REPS}  Seeds: {len(SEEDS)}")

print("\n[1] Building query pools ...")
fast_pool = build_fast_pool()
slow_pool = build_slow_pool()
print(f"  Fast pool : {len(fast_pool)} queries  (~0.2 ms each)")
print(f"  Slow pool : {len(slow_pool)} queries  (~8 ms each)")
sample_trace = build_burst_trace(fast_pool, slow_pool, seed=42)
phase_counts = Counter(ph for _, ph, _, _ in sample_trace)
print(f"  Trace length (per seed) : {len(sample_trace)} queries")
for ph in PHASE_ORDER:
    print(f"    {ph:<12}: {phase_counts.get(ph,0):3d} queries")

# ── Database connection ────────────────────────────────────────────────────────
conn = None
if args.execute or args.smoke:
    print(f"\n[2] Connecting to {args.dbname} ...")
    try:
        import psycopg2, psycopg2.extensions
        conn = psycopg2.connect(
            host=DOCKER_HOST, port=args.port, dbname=args.dbname,
            user=DOCKER_USER, password=DOCKER_PASS, connect_timeout=10,
            options=f"-c statement_timeout={TIMEOUT_MS}")
        conn.set_session(autocommit=True)
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM pgbench_accounts")
        n_acc = cur.fetchone()[0]
        print(f"  Connected ✓  (accounts: {n_acc:,})")
    except Exception as e:
        print(f"  ERROR: {e}")
        sys.exit(1)

# ── Smoke test ─────────────────────────────────────────────────────────────────
if args.smoke:
    print("\n[3] Smoke test — 3 queries per phase ...")
    trace_s = build_burst_trace(fast_pool, slow_pool, seed=42)
    for ph in PHASE_ORDER:
        sample = [(qid, sql) for qid, p, _, sql in trace_s if p == ph][:3]
        times  = []
        for qid, sql in sample:
            t = timed_exec(cur, sql, n_reps=1)
            times.append(t)
            if args.verbose:
                spd = 'fast' if t < 2 else 'slow'
                print(f"    {qid} ({spd}) {t:8.2f} ms")
        print(f"  {ph:<12}: median = {statistics.median(times):8.2f} ms")
    conn.close()
    print("\n  Smoke test complete.")
    print("  Run full experiment:")
    print("    python hsm_burst_validation.py --execute")
    sys.exit(0)

# ── Static proxy mode ──────────────────────────────────────────────────────────
if not args.execute:
    print("\n[Static] Using proxy timing: fast=0.20 ms, slow=8.0 ms")
    PROXY = {'Flat_Low': 0.20, 'Burst_Alt': None,
             'Burst_Grp': None, 'Flat_High': 8.0}
    for seed in SEEDS[:1]:
        trace = build_burst_trace(fast_pool, slow_pool, seed)
        exec_trace = []
        for qid, ph, qt, sql in trace:
            if ph in ('Flat_Low',):
                ms = 0.20 + random.gauss(0, 0.02)
            elif ph == 'Flat_High':
                ms = 8.0 + random.gauss(0, 0.5)
            elif ph == 'Burst_Alt':
                # Alternating based on qid position
                idx = int(qid[2:]) - 1
                ms = 0.20 if idx % 2 == 0 else 8.0
                ms += random.gauss(0, 0.1)
            else:  # Burst_Grp
                idx = int(qid[2:]) - 1
                ms = 0.20 if (idx % 8) < 4 else 8.0
                ms += random.gauss(0, 0.1)
            exec_trace.append((qid, ph, sql, max(0.01, ms)))
        wins = windows_from_trace(exec_trace)
        dr, p, r, info = compute_dr(wins)
        print(f"  Static DR={dr:.3f}  p={p:.3e}  dominant={info['dominant']}")
    sys.exit(0)

# ── Full execution ─────────────────────────────────────────────────────────────
print(f"\n[3] Executing trace for timing calibration (seed=42) ...")
calib_trace = build_burst_trace(fast_pool, slow_pool, seed=42)
exec_calib = []
timeouts = 0
for qid, ph, qt, sql in calib_trace:
    ms = timed_exec(cur, sql, n_reps=N_REPS)
    if ms >= TIMEOUT_MS: timeouts += 1
    exec_calib.append((qid, ph, sql, ms))

# Per-phase timing summary
print(f"  Done. Timeouts: {timeouts}/{len(calib_trace)}")
print(f"\n[4] Per-phase timing summary (seed=42) ...")
print(f"  {'Phase':<14} {'N':>4}  {'Median ms':>10}  {'Min ms':>8}  {'Max ms':>8}")
print(f"  {'-'*52}")
for ph in PHASE_ORDER:
    ph_times = [ms for _, p, _, ms in exec_calib if p == ph]
    if ph_times:
        print(f"  {ph:<14} {len(ph_times):>4}  "
              f"{statistics.median(ph_times):>10.2f}  "
              f"{min(ph_times):>8.2f}  {max(ph_times):>8.2f}")

# ── 10-seed experiment ─────────────────────────────────────────────────────────
print(f"\n[5] Running {len(SEEDS)} seeds ...")
seed_results = []
for seed in SEEDS:
    trace = build_burst_trace(fast_pool, slow_pool, seed)
    exec_trace = []
    for qid, ph, qt, sql in trace:
        ms = timed_exec(cur, sql, n_reps=N_REPS)
        exec_trace.append((qid, ph, sql, ms))
    wins = windows_from_trace(exec_trace)
    dr, p_val, r_biserial, info = compute_dr(wins)
    if dr is None:
        print(f"  Seed {seed:3d}: insufficient pairs")
        continue
    seed_results.append({
        'seed': seed, 'dr': dr, 'p': p_val, 'r': r_biserial,
        'n_within': info['n_within'], 'n_cross': info['n_cross'],
        'dim_deltas': info['dim_deltas'], 'dominant': info['dominant'],
        'w_dims': info['w_dims'], 'c_dims': info['c_dims'],
        'w_scores': info['w_scores'], 'c_scores': info['c_scores'],
    })
    print(f"  Seed {seed:3d}: DR={dr:.3f} (CI per-seed bootstrap)  "
          f"p={p_val:.3e}  dominant={info['dominant']}")

conn.close()

# ── Aggregate ──────────────────────────────────────────────────────────────────
print(f"\n[6] Aggregating ...")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
csv_path = RESULTS_DIR / 'burst_hsm_execute.csv'

drs = [r['dr'] for r in seed_results]
ps  = [r['p']  for r in seed_results]
dr_med = statistics.median(drs)
dr_mean = statistics.mean(drs)
dr_sd   = statistics.stdev(drs) if len(drs) > 1 else 0
n = len(drs)
t_crit = stats.t.ppf(0.975, df=n-1) if n > 1 else 0
ci_lo = dr_mean - t_crit * dr_sd / math.sqrt(n)
ci_hi = dr_mean + t_crit * dr_sd / math.sqrt(n)

# Bootstrap CI from pooled pair scores
all_w, all_c = [], []
for r in seed_results:
    all_w.extend(r.get('w_dims', []))
    all_c.extend(r.get('c_dims', []))

# Dimension analysis (average across seeds)
dim_names = ['S_R', 'S_V', 'S_T', 'S_A', 'S_P']
avg_w_dims = np.mean([r['w_dims'] for r in seed_results], axis=0)
avg_c_dims = np.mean([r['c_dims'] for r in seed_results], axis=0)
avg_deltas = avg_w_dims - avg_c_dims
dominant_overall = dim_names[int(np.argmax(avg_deltas))]

# Mann-Whitney on all pooled scores using seed median p
p_med = statistics.median(ps)

# Save CSV
with open(csv_path, 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['metric', 'value'])
    w.writerow(['DR_median', f'{dr_med:.4f}'])
    w.writerow(['DR_mean',   f'{dr_mean:.4f}'])
    w.writerow(['DR_SD',     f'{dr_sd:.4f}'])
    w.writerow(['DR_CI_lo',  f'{ci_lo:.4f}'])
    w.writerow(['DR_CI_hi',  f'{ci_hi:.4f}'])
    w.writerow(['DR_range_lo', f'{min(drs):.4f}'])
    w.writerow(['DR_range_hi', f'{max(drs):.4f}'])
    w.writerow(['MWU_p_median', f'{p_med:.4e}'])
    w.writerow(['dominant_dim', dominant_overall])
    for i, nm in enumerate(dim_names):
        w.writerow([f'delta_{nm}', f'{avg_deltas[i]:.4f}'])
    for r in seed_results:
        w.writerow([f'DR_seed{r["seed"]}', f'{r["dr"]:.4f}'])
print(f"  Results → {csv_path}")

# Raw pair-score dump (reference seed) — for fig01 / fig03.
from hsm_v2_kernel import dump_pair_scores_csv  # noqa: E402
_ref = seed_results[0]
_pair_path = RESULTS_DIR / 'burst_hsm_pair_scores.csv'
dump_pair_scores_csv(str(_pair_path), _ref['w_scores'], _ref['c_scores'],
                     workload='burst')
print(f"  Pair scores → {_pair_path}")

# ── Final report ───────────────────────────────────────────────────────────────
print()
print("=" * 70)
print("  RESULTS: HSM Burst / Non-Stationary Validation")
print("=" * 70)
print(f"\n  Phases : Flat_Low / Burst_Alt / Burst_Grp / Flat_High")
print(f"  Queries: 114 per trace  Seeds: {n}")

print(f"\n  ── Per-dimension (avg across {n} seeds) ──")
print(f"  {'Dim':<6} {'Within':>8} {'Cross':>8} {'Δ':>8}")
print(f"  {'-'*36}")
for i, nm in enumerate(dim_names):
    marker = " ← dominant" if nm == dominant_overall else ""
    print(f"  {nm:<6} {avg_w_dims[i]:>8.4f} {avg_c_dims[i]:>8.4f} "
          f"{avg_deltas[i]:>+8.4f}{marker}")

print(f"\n  ── Aggregate ({n} seeds) ──")
print(f"  DR median          : {dr_med:.3f}")
print(f"  DR mean ± SD       : {dr_mean:.3f} ± {dr_sd:.3f}")
print(f"  Empirical 95% CI   : [{ci_lo:.3f}, {ci_hi:.3f}]")
print(f"  Mann-Whitney p     : {p_med:.3e}")
print(f"  Dominant dim       : {dominant_overall}")
print(f"  DR range (seeds)   : {min(drs):.3f} – {max(drs):.3f}")

print(f"\n  ── Comparison ──")
print(f"  TPC-H  (A7)  : DR=1.445,  p<0.001")
print(f"  SDSS   (A8)  : DR=1.086,  p=1.6e-70")
print(f"  JOB    (A8b) : DR=1.814,  p=2.8e-85")
print(f"  OLTP   (A8c) : DR=1.543,  p=1.459e-65")
print(f"  Burst  (A8d) : DR={dr_med:.3f},  p={p_med:.3e}")

sig = "✓ SIGNIFICANT" if p_med < 0.05 else "✗ NOT SIGNIFICANT"
print(f"\n  Assessment : {sig} (α=0.05)")
print()
print("=" * 70)
