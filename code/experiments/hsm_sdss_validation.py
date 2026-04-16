"""
HSM Validation on Real SDSS SkyServer Query Logs
=================================================
Replicates A8 validation using real query logs (upgrading from simulated SDSS data).
Computes HSM 5-dimensional similarity across real workload windows and reports
discrimination ratio (within-phase vs cross-phase), Mann-Whitney U, and ICC(2,1).

HSM Dimensions (per paper Section 3):
  S_R : SELECT-ratio similarity      = 1 - |ratio_A - ratio_B|
  S_V : Volume similarity            = exp(-|log(QPS_A) - log(QPS_B)|)
  S_T : Type angular similarity      = cosine([n_sel,n_upd,n_ins,n_del])
  S_A : Access-attribute overlap     = 0.5*Jaccard(tables) + 0.5*Jaccard(columns)
  S_P : Temporal pattern similarity  = DWT(db4,L=3) + SAX(α=4) + FastDTW(r=3)
        via wavedec → SAX encode each band → FastDTW distance → weighted avg

HSM = 0.2*S_R + 0.2*S_V + 0.2*S_T + 0.2*S_A + 0.2*S_P
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

import re, os, sys, csv, math, statistics, random, warnings
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np
from scipy import stats
import pywt
from fastdtw import fastdtw

# v2 kernel — canonical five-dimension HSM per paper §III.
sys.path.insert(0, str(Path(__file__).parent))
from hsm_v2_kernel import hsm_score_from_features  # noqa: E402

# ── Config ────────────────────────────────────────────────────────────────────
NPTS          = 20          # queries per window (≈ TPC-H mean 19.5)
RANDOM_SEED   = 42
HSM_WEIGHTS   = [0.25, 0.20, 0.20, 0.20, 0.15]   # w_R, w_V, w_T, w_A, w_P
DWT_WAVELET   = 'db4'
DWT_LEVEL     = 3
BAND_WEIGHTS  = [0.40, 0.60]   # [cA1, cD1]
SAX_ALPHA     = 4
FASTDTW_R     = 3
N_SLOTS       = 8
# Resolve SkyLog_Workload.csv relative to this script, with a few fallbacks
# so the same code runs in the HSM_gated repo, an earlier Experimental Code
# sibling tree, or a path provided via the HSM_SDSS_CSV environment variable.
def _resolve_skylog_path() -> str:
    here = os.path.dirname(os.path.abspath(__file__))
    repo_root   = os.path.normpath(os.path.join(here, os.pardir, os.pardir))
    version4    = os.path.normpath(os.path.join(repo_root, os.pardir))          # .../Version 4
    paper3a     = os.path.normpath(os.path.join(version4, os.pardir))           # .../Paper 3A
    candidates  = [
        os.environ.get('HSM_SDSS_CSV'),
        os.path.join(repo_root, 'code', 'data', 'sdss', 'SkyLog_Workload.csv'),
        os.path.join(repo_root, 'data', 'SkyLog_Workload.csv'),
        os.path.join(paper3a, 'Experimental Code', 'data', 'SkyLog_Workload.csv'),
    ]
    for c in candidates:
        if c and os.path.isfile(c):
            return c
    tried = "\n    ".join(c for c in candidates if c)
    raise FileNotFoundError(
        "SkyLog_Workload.csv not found.  Set HSM_SDSS_CSV or place it in "
        "code/data/sdss/.  Tried:\n    " + tried
    )

FILE_PATH = _resolve_skylog_path()

# ── Step 1: Parse CSV (multi-line SQL) ────────────────────────────────────────
print("=" * 70)
print("  HSM Validation — Real SDSS SkyServer Query Logs")
print("=" * 70)
print("\n[1] Parsing SkyLog_Workload.csv ...")

record_end = re.compile(
    r',(\d+/\d+/\d{4}\s[\d:]+\s[AP]M),([\d.]+),([\d.]+),(\d+),([^,]*),(\d+)\s*$'
)

records = []
current_lines = []

with open(FILE_PATH, 'r', encoding='utf-8', errors='replace') as f:
    next(f)  # skip header
    for line in f:
        m = record_end.search(line.rstrip())
        if m:
            sql_part = line[:m.start()]
            current_lines.append(sql_part)
            full_sql = ' '.join(current_lines).strip()
            records.append({
                'sql'    : full_sql,
                'time'   : m.group(1),
                'elapsed': float(m.group(2)),
                'rows'   : int(m.group(4)),
                'dbname' : m.group(5).strip(),
                'error'  : m.group(6)
            })
            current_lines = []
        else:
            current_lines.append(line.rstrip())

print(f"  Parsed {len(records):,} query records")

# ── Step 2: Extract query features per record ─────────────────────────────────
print("\n[2] Extracting query features ...")

def extract_tables(sql):
    """Extract table/function names from FROM and JOIN clauses."""
    sql_upper = sql.upper()
    tables = set()
    for m in re.finditer(r'\bFROM\s+([A-Za-z][A-Za-z0-9_]*)', sql, re.I):
        tables.add(m.group(1).lower())
    for m in re.finditer(r'\bJOIN\s+([A-Za-z][A-Za-z0-9_]*)', sql, re.I):
        tables.add(m.group(1).lower())
    return tables

def extract_columns(sql):
    """Extract column names from WHERE / ORDER BY clauses (heuristic)."""
    cols = set()
    for m in re.finditer(r'\bWHERE\s+(\w+)\s*[=<>!]', sql, re.I):
        cols.add(m.group(1).lower())
    for m in re.finditer(r'\bAND\s+(\w+)\s*[=<>!]', sql, re.I):
        cols.add(m.group(1).lower())
    for m in re.finditer(r'\bORDER\s+BY\s+(\w+)', sql, re.I):
        cols.add(m.group(1).lower())
    return cols

def get_qtype(sql):
    s = sql.strip().upper()
    for t in ['SELECT', 'INSERT', 'UPDATE', 'DELETE', 'EXEC']:
        if s.startswith(t):
            return t
    return 'OTHER'

for r in records:
    r['qtype']   = get_qtype(r['sql'])
    r['tables']  = extract_tables(r['sql'])
    r['columns'] = extract_columns(r['sql'])
    r['access']  = r['tables'] | r['columns']

print(f"  Features extracted for {len(records):,} records")

# ── Step 3: Define phases based on dominant table patterns ─────────────────────
print("\n[3] Identifying workload phases ...")

# Classify each query into a workload theme
PHASE_MAP = {
    'spatial'      : {'fgetnearbyobjeq', 'fgetnearbyapogeestareq',
                      'fgetnearbyprimobj', 'fgetnearestspec'},
    'photometric'  : {'photoobjall', 'photoprimary', 'phototag',
                      'photoobj', 'photoz', 'photozrf', 'field',
                      'galaxy', 'star'},
    'spectroscopic': {'apogeestar', 'aspcapstar', 'specobjall',
                      'sppparams', 'galspecline', 'galspecinfo'},
    'metadata'     : {'dbobjects', 'indexmap', 'dbo', 'systables'},
}

def classify_phase(tables):
    scores = {ph: len(tables & kw) for ph, kw in PHASE_MAP.items()}
    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else 'mixed'

for r in records:
    r['phase'] = classify_phase(r['tables'])

phase_counts = Counter(r['phase'] for r in records)
print("  Phase distribution:")
for ph, cnt in phase_counts.most_common():
    print(f"    {ph:15s}: {cnt:6,} ({100*cnt/len(records):.1f}%)")

# ── Step 4: Build windows ─────────────────────────────────────────────────────
print(f"\n[4] Building windows (Npts={NPTS} queries each) ...")

# Use only SELECT queries (99.1%) for cleaner analysis
select_records = [r for r in records if r['qtype'] == 'SELECT']
print(f"  Using {len(select_records):,} SELECT queries")

# Chunk into windows of NPTS queries
windows = []
for i in range(0, len(select_records) - NPTS, NPTS):
    chunk = select_records[i:i + NPTS]
    windows.append(chunk)

print(f"  Created {len(windows):,} windows of {NPTS} queries each")

# Assign window phase = majority phase in that window
def window_phase(w):
    phases = Counter(r['phase'] for r in w)
    return phases.most_common(1)[0][0]

for w in windows:
    pass  # phase assigned per-pair below

# ── Step 5: Compute HSM dimensions per window ─────────────────────────────────
print("\n[5] Computing HSM window features ...")

def compute_window_features(chunk):
    """v2: emit all keys needed by hsm_v2_kernel.hsm_score_from_features."""
    n = len(chunk)
    elapsed = [r['elapsed'] for r in chunk]

    # Per-query-type counts (SDSS is mostly SELECT; kept for diagnostic only)
    n_sel = sum(1 for r in chunk if r['qtype'] == 'SELECT')
    n_upd = sum(1 for r in chunk if r['qtype'] == 'UPDATE')
    n_ins = sum(1 for r in chunk if r['qtype'] == 'INSERT')
    n_del = sum(1 for r in chunk if r['qtype'] == 'DELETE')

    # S_R input: per-table frequency vector so Spearman operates over the
    # rank distribution of accessed objects (paper §III.A generic form).
    table_counts = Counter()
    for r in chunk:
        for t in r['tables']:
            table_counts[t] += 1
    freq_axis = sorted(table_counts.keys())
    freq = np.array([float(table_counts[t]) for t in freq_axis])

    # S_A: dual Jaccard on tables + columns
    table_set = set()
    col_set   = set()
    for r in chunk:
        table_set.update(r['tables'])
        col_set.update(r['columns'])

    # S_P: q(t) arrival-count series at 1-second resolution (paper §III-A
    # line 291).  Built from cumulative execution times.
    from hsm_v2_kernel import build_qps_series
    times = build_qps_series(elapsed, min_bins=16)

    qset = set(freq_axis) | {r['qtype'] for r in chunk}

    return {
        # v2 kernel inputs (paper §III-B Relational extractor)
        'freq'      : freq,
        'freq_map'  : dict(table_counts),
        'tables'    : table_set,
        'cols'      : col_set,
        'times'     : times,
        'qset'      : qset,
        'n'         : n,
        # diagnostic-only fields (NOT inputs to the kernel)
        'crud_vec'  : np.array([n_sel, n_upd, n_ins, n_del], dtype=float),
        'ratio_sel' : n_sel / n,
        'qps'       : n / max(sum(max(e, 0.001) for e in elapsed), 1e-9),
        'columns'   : col_set,
        'phase'     : window_phase(chunk),
    }

win_features = [compute_window_features(w) for w in windows]
print(f"  Computed features for {len(win_features)} windows")

# ── Step 6: HSM similarity computation ────────────────────────────────────────
print("\n[6] Computing pairwise HSM scores ...")

def hsm(fa, fb):
    """Paper §III five-dimension HSM (canonical v2 kernel)."""
    return hsm_score_from_features(fa, fb)

# Compute consecutive window pairs
within_scores, cross_scores = [], []
within_dims  = defaultdict(list)
cross_dims   = defaultdict(list)
consec_rows  = []  # T5 trigger_timeseries: (window_idx, score, phase_a, phase_b)

n_pairs = 0
for i in range(len(win_features) - 1):
    fa = win_features[i]
    fb = win_features[i + 1]
    score, dims = hsm(fa, fb)

    consec_rows.append({
        'window_idx': i + 1,
        'score':      score,
        'phase_a':    fa.get('phase', 'unknown'),
        'phase_b':    fb.get('phase', 'unknown'),
    })

    is_within = (fa['phase'] == fb['phase'])
    if is_within:
        within_scores.append(score)
        for k, v in dims.items():
            within_dims[k].append(v)
    else:
        cross_scores.append(score)
        for k, v in dims.items():
            cross_dims[k].append(v)
    n_pairs += 1

print(f"  Total pairs: {n_pairs}")
print(f"  Within-phase pairs: {len(within_scores)}")
print(f"  Cross-phase pairs:  {len(cross_scores)}")

# ── Step 7: Statistics ─────────────────────────────────────────────────────────
print("\n[7] Computing statistics ...")

w_mean  = statistics.mean(within_scores)
w_std   = statistics.stdev(within_scores)
c_mean  = statistics.mean(cross_scores)
c_std   = statistics.stdev(cross_scores)
dr      = w_mean / c_mean

# Mann-Whitney U
u_stat, p_val = stats.mannwhitneyu(within_scores, cross_scores, alternative='greater')
n1, n2 = len(within_scores), len(cross_scores)
r_biserial = 1 - 2*u_stat / (n1 * n2)

# 95% CI for DR (bootstrap)
rng = random.Random(RANDOM_SEED)
boot_drs = []
for _ in range(2000):
    s_w = [rng.choice(within_scores) for _ in range(n1)]
    s_c = [rng.choice(cross_scores)  for _ in range(n2)]
    m_c = statistics.mean(s_c)
    if m_c > 0:
        boot_drs.append(statistics.mean(s_w) / m_c)
boot_drs.sort()
ci_lo = boot_drs[int(0.025 * len(boot_drs))]
ci_hi = boot_drs[int(0.975 * len(boot_drs))]

# ICC(2,1) approximation
all_scores = within_scores + cross_scores
grand_mean = statistics.mean(all_scores)
ss_total = sum((x - grand_mean)**2 for x in all_scores)
ss_between = (n1*(w_mean - grand_mean)**2 + n2*(c_mean - grand_mean)**2)
ss_within = ss_total - ss_between
ms_between = ss_between / 1
ms_within  = ss_within / (len(all_scores) - 2)
icc = (ms_between - ms_within) / (ms_between + (2-1)*ms_within) if ms_between > ms_within else 0.0

# ── Step 8: Per-dimension analysis ────────────────────────────────────────────
print("\n[8] Per-dimension breakdown ...")

print(f"\n{'Dimension':<8}  {'Within':>8}  {'Cross':>8}  {'Delta':>8}")
print("-" * 45)
for dim in ['S_R', 'S_V', 'S_T', 'S_A', 'S_P']:
    wm = statistics.mean(within_dims[dim]) if within_dims[dim] else 0
    cm = statistics.mean(cross_dims[dim])  if cross_dims[dim]  else 0
    print(f"{dim:<8}  {wm:8.4f}  {cm:8.4f}  {wm-cm:+8.4f}")

# ── Step 9: Final Results ──────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("  RESULTS: Real SDSS SkyServer Query Log Validation")
print("=" * 70)
print(f"\n  Windows         : {len(win_features):,} ({NPTS} queries each)")
print(f"  Within-phase    : {n1:,} pairs  (mean={w_mean:.4f}, σ={w_std:.4f})")
print(f"  Cross-phase     : {n2:,} pairs  (mean={c_mean:.4f}, σ={c_std:.4f})")
print(f"\n  Discrimination Ratio : {dr:.3f}  (95% CI: [{ci_lo:.3f}, {ci_hi:.3f}])")
print(f"  Mann-Whitney p       : {p_val:.3e}")
print(f"  Rank-biserial r      : {r_biserial:.3f}")
print(f"  ICC(2,1)             : {icc:.3f}")
print(f"\n  θ=0.75 separation    : ", end="")
below_thresh = sum(1 for s in cross_scores if s < 0.75)
above_thresh = sum(1 for s in within_scores if s >= 0.75)
print(f"{below_thresh}/{n2} cross-phase below θ  |  "
      f"{above_thresh}/{n1} within-phase above θ")

print(f"\n  Paper (simulated SDSS, A8) : DR=1.199, p=0.002")
print(f"  Real SDSS query logs       : DR={dr:.3f}, p={p_val:.3e}")
improvement = "BETTER" if dr > 1.199 else "COMPARABLE" if dr > 1.0 else "LOWER"
print(f"  Assessment                 : {improvement}")

print("\n" + "=" * 70)
print("  Phase transition examples (cross-phase pairs, lowest HSM score)")
print("=" * 70)
cross_pairs_detail = []
for i in range(len(win_features) - 1):
    fa = win_features[i]
    fb = win_features[i + 1]
    if fa['phase'] != fb['phase']:
        score, _ = hsm(fa, fb)
        cross_pairs_detail.append((score, fa['phase'], fb['phase']))
cross_pairs_detail.sort()
for score, pa, pb in cross_pairs_detail[:5]:
    print(f"  {pa:15s} → {pb:15s}  HSM={score:.4f}")

# ── Step 10: Persist summary + raw pair scores ────────────────────────────────
import csv as _csv  # noqa: E402
from hsm_v2_kernel import dump_pair_scores_csv  # noqa: E402

_RESULTS_DIR = Path(__file__).resolve().parents[1] / 'results' / 'sdss_validation'
_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

_out_summary = _RESULTS_DIR / 'sdss_hsm.csv'
with open(_out_summary, 'w', newline='') as _fh:
    _w = _csv.writer(_fh)
    _w.writerow(['metric', 'value'])
    _w.writerow(['design',         'sdss_skyserver_queries'])
    _w.writerow(['mode',           'execute'])
    _w.writerow(['n_queries',      sum(f.get('n', 0) for f in win_features)])
    _w.writerow(['n_windows_ref',  len(win_features)])
    _w.writerow(['n_within_ref',   n1])
    _w.writerow(['n_cross_ref',    n2])
    _w.writerow(['within_mean',    round(w_mean, 6)])
    _w.writerow(['cross_mean',     round(c_mean, 6)])
    _w.writerow(['DR_median',      round(dr, 4)])
    _w.writerow(['DR_CI_lo',       round(ci_lo, 4)])
    _w.writerow(['DR_CI_hi',       round(ci_hi, 4)])
    _w.writerow(['MWU_p_median',   f'{p_val:.3e}'])
    _w.writerow(['r_biserial',     round(r_biserial, 4)])
    _w.writerow(['ICC_2_1',        round(icc, 4)])
    _dim_delta = {d: (statistics.mean(within_dims[d]) if within_dims[d] else 0.0) -
                    (statistics.mean(cross_dims[d])  if cross_dims[d]  else 0.0)
                  for d in ['S_R', 'S_V', 'S_T', 'S_A', 'S_P']}
    _dom = max(_dim_delta, key=_dim_delta.get)
    _w.writerow(['dominant_dim', _dom])
    for d in ['S_R', 'S_V', 'S_T', 'S_A', 'S_P']:
        _w.writerow([f'delta_{d}', round(_dim_delta[d], 6)])
print(f"\n  Summary    → {_out_summary}")

_pair_file = _RESULTS_DIR / 'sdss_hsm_pair_scores.csv'
dump_pair_scores_csv(str(_pair_file), within_scores, cross_scores,
                     workload='sdss')
print(f"  Pair scores → {_pair_file}")

# T5 trigger timeseries (gate fires when HSM(W_{t-1},W_t) < theta).
_consec_path = _RESULTS_DIR / 'sdss_hsm_trigger_timeseries.csv'
_theta = 0.75  # paper §IV default
import csv as _csv2
with open(_consec_path, 'w', newline='') as _f:
    _cw = _csv2.writer(_f)
    _cw.writerow(['window_idx', 'score', 'gate_triggered',
                  'phase_a', 'phase_b'])
    for _r in consec_rows:
        _t = 1 if _r['score'] < _theta else 0
        _cw.writerow([_r['window_idx'], f"{_r['score']:.6f}", _t,
                      _r['phase_a'], _r['phase_b']])
print(f"  Trigger ts  → {_consec_path}")
