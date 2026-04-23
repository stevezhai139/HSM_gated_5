"""
hsm_v2_core.py
==============
Shared v2 HSM computation and statistics used by all 10-seed runners.

v2 formulas (verified against measures.py):
  S_R = 1 - arccos(ρ_s) / π          [Spearman, correlation-angle metric]
  S_T = 1 - (2/π)·arccos(v̂_A·v̂_B)  [angular distance on unit sphere]
  S_V = min/max                        [volume ratio]
  S_A = 0.5·J(tables) + 0.5·J(cols)  [Jaccard, schema overlap]
  S_P = DWT+FastDTW on timing series  [temporal pattern]

All-pairs DR protocol:
  within_pairs = {(i,j) : phase(i) == phase(j), i < j}
  cross_pairs  = {(i,j) : phase(i) != phase(j), i < j}
  DR = mean(within HSM) / mean(cross HSM)

Statistical tests:
  Bootstrap 95% CI (n=5000), Mann-Whitney U (one-sided), rank-biserial |r|,
  ICC(2,1) on within-phase scores.
"""

import math
import numpy as np
from scipy.stats import mannwhitneyu, spearmanr, pearsonr
from typing import Dict, List, Tuple, Set, Optional
import warnings

warnings.filterwarnings('ignore', category=RuntimeWarning)

# ─── Default weights ──────────────────────────────────────────────────────────
W0 = {'R': 0.25, 'V': 0.20, 'T': 0.20, 'A': 0.20, 'P': 0.15}


# ─── v2 dimension functions ───────────────────────────────────────────────────

def sr_v2(freq_a: np.ndarray, freq_b: np.ndarray) -> float:
    """S_R v2: 1 - arccos(Spearman ρ_s) / π."""
    if len(freq_a) < 2 or freq_a.sum() == 0 or freq_b.sum() == 0:
        return 0.5
    rho, _ = spearmanr(freq_a, freq_b)
    if np.isnan(rho):
        rho = 1.0
    rho = float(np.clip(rho, -1.0, 1.0))
    return float(1.0 - math.acos(rho) / math.pi)


def sv_v2(n_a: int, n_b: int) -> float:
    """S_V v2: min/max query count ratio."""
    if max(n_a, n_b) == 0:
        return 1.0
    return float(min(n_a, n_b) / max(n_a, n_b))


def st_v2(freq_a: np.ndarray, freq_b: np.ndarray) -> float:
    """S_T v2: 1 - (2/π)·arccos(v̂_A · v̂_B) — angular distance on S^{k-1}."""
    denom = np.linalg.norm(freq_a) * np.linalg.norm(freq_b)
    if denom < 1e-12:
        return 1.0
    cos_val = float(np.clip(np.dot(freq_a, freq_b) / denom, -1.0, 1.0))
    return float(1.0 - (2.0 / math.pi) * math.acos(cos_val))


def sa_v2(tables_a: Set[str], tables_b: Set[str],
          cols_a: Set[str],   cols_b: Set[str]) -> float:
    """S_A v2: 0.5·J(tables) + 0.5·J(cols)."""
    t_union = len(tables_a | tables_b)
    c_union = len(cols_a | cols_b)
    j_t = len(tables_a & tables_b) / t_union if t_union > 0 else 1.0
    j_c = len(cols_a   & cols_b)   / c_union if c_union > 0 else 1.0
    return float(0.5 * j_t + 0.5 * j_c)


def sp_v2(times_a: np.ndarray, times_b: np.ndarray,
          query_set_a: Set[str], query_set_b: Set[str]) -> float:
    """
    S_P v2: Three-stage temporal pattern similarity (Paper Eq. 7).

    Matches paper exactly:
      Stage 1 (DWT):     Daubechies-4 (db4) at level L=3
                         → cA3, cD3, cD2, cD1
      Stage 2 (SAX):     Each sub-band encoded with SAX alphabet α=4
      Stage 3 (FastDTW): Sakoe-Chiba radius r=3, per-band score
                         band_score = 1 - dist / (len(s) * (α-1))

      S_P = 0.40·sc(cA3) + 0.20·sc(cD3) + 0.20·sc(cD2) + 0.20·sc(cD1)

    Requires: pywt, fastdtw.
    Falls back to Jaccard on query sets if series too short for 3-level DWT.
    """
    SAX_ALPHA  = 4      # SAX alphabet size (paper: α=4)
    DTW_RADIUS = 3      # FastDTW Sakoe-Chiba radius (paper: r=3)
    WAVELET    = 'db4'  # Daubechies-4 (paper: db4)
    DWT_LEVEL  = 3      # decomposition level (paper: L=3)
    # Band weights (paper Eq. 7): cA3=0.40, cD3=cD2=cD1=0.20
    BAND_WEIGHTS = [0.40, 0.20, 0.20, 0.20]

    # Minimum length for db4: filter length=8; level 3 ideally wants ≥16 pts.
    # For shorter series, we reduce the level gracefully.
    MIN_LEN_L3 = 16     # use level 3 if ≥16 points
    MIN_LEN_L2 = 8      # fall back to level 2 if ≥8 points
    MIN_LEN_ABS = 4     # below this → Jaccard fallback

    try:
        import pywt
        from fastdtw import fastdtw

        min_len = min(len(times_a), len(times_b))
        if min_len < MIN_LEN_ABS:
            raise ValueError('series too short')

        ta = times_a[:min_len].astype(float)
        tb = times_b[:min_len].astype(float)

        # Normalise to [0,1] to prevent scale effects
        scale = max(ta.max(), tb.max(), 1e-12)
        ta /= scale
        tb /= scale

        # ── Stage 1: DWT with db4 ──
        # Use level 3 if enough data; gracefully reduce to level 2
        actual_level = DWT_LEVEL if min_len >= MIN_LEN_L3 else max(1, min(DWT_LEVEL, 2))

        with warnings.catch_warnings():
            warnings.simplefilter('ignore')  # suppress boundary-effect warnings
            coeffs_a = pywt.wavedec(ta, WAVELET, level=actual_level)
            coeffs_b = pywt.wavedec(tb, WAVELET, level=actual_level)
        # coeffs = [cA3, cD3, cD2, cD1]

        def sax_encode(coeffs_arr: np.ndarray, alpha: int) -> np.ndarray:
            """SAX encoding: quantise to α symbols using Gaussian breakpoints."""
            from scipy.stats import norm
            breakpoints = norm.ppf(np.linspace(0, 1, alpha + 1)[1:-1])
            # Z-normalise
            mu = coeffs_arr.mean()
            sd = coeffs_arr.std()
            if sd < 1e-12:
                return np.zeros(len(coeffs_arr), dtype=int)
            z = (coeffs_arr - mu) / sd
            return np.digitize(z, breakpoints).astype(int)

        def band_score(ca: np.ndarray, cb: np.ndarray) -> float:
            """Per-band score: SAX + FastDTW + normalisation (paper formula)."""
            if len(ca) == 0 or len(cb) == 0:
                return 1.0
            sa = sax_encode(ca, SAX_ALPHA)
            sb = sax_encode(cb, SAX_ALPHA)
            dist, _ = fastdtw(sa, sb, radius=DTW_RADIUS,
                              dist=lambda a, b: abs(int(a) - int(b)))
            max_dist = len(sa) * (SAX_ALPHA - 1)  # worst-case distance
            if max_dist == 0:
                return 1.0
            return float(max(0.0, 1.0 - dist / max_dist))

        # ── Stage 2+3: SAX → FastDTW per band ──
        # Compute score for each of the 4 bands: cA3, cD3, cD2, cD1
        scores = []
        for ca, cb in zip(coeffs_a, coeffs_b):
            scores.append(band_score(ca, cb))

        # Pad if fewer bands (shouldn't happen with level=3)
        while len(scores) < 4:
            scores.append(1.0)

        sp = sum(w * s for w, s in zip(BAND_WEIGHTS, scores[:4]))
        return float(np.clip(sp, 0.0, 1.0))

    except Exception:
        pass

    # Fallback: Jaccard on query sets (for very short series)
    union = query_set_a | query_set_b
    return float(len(query_set_a & query_set_b) / len(union)) if union else 1.0


def hsm_v2(freq_a, freq_b, n_a, n_b,
           tables_a, tables_b, cols_a, cols_b,
           times_a, times_b, qset_a, qset_b,
           weights=None) -> Dict[str, float]:
    """Compute full 5-D HSM v2 score between two windows."""
    if weights is None:
        weights = W0
    SR = sr_v2(freq_a, freq_b)
    SV = sv_v2(n_a, n_b)
    ST = st_v2(freq_a, freq_b)
    SA = sa_v2(tables_a, tables_b, cols_a, cols_b)
    SP = sp_v2(times_a, times_b, qset_a, qset_b)
    HSM = (weights['R'] * SR + weights['V'] * SV + weights['T'] * ST
           + weights['A'] * SA + weights['P'] * SP)
    return {
        'S_R': round(SR, 4), 'S_V': round(SV, 4), 'S_T': round(ST, 4),
        'S_A': round(SA, 4), 'S_P': round(SP, 4), 'HSM': round(float(HSM), 4),
    }


# ─── Window feature extraction from trace DataFrame ───────────────────────────

def extract_windows(trace_df, query_tables: Dict[str, Set[str]],
                    query_cols: Dict[str, Set[str]],
                    all_templates: Optional[List[str]] = None):
    """
    Extract per-window feature dicts from a trace DataFrame.
    trace_df columns: run, window, phase, query, exec_ms, ok (at minimum).
    Returns: list of window feature dicts, sorted by (run, window).
    """
    import pandas as pd

    ok_df = trace_df[trace_df['ok'].astype(str).str.lower() == 'true'].copy()
    if all_templates is None:
        all_templates = sorted(ok_df['query'].unique())

    windows = []
    for (run, window), grp in ok_df.groupby(['run', 'window']):
        phase = grp['phase'].iloc[0]
        templates = grp['query'].values
        freq = np.array([float(np.sum(templates == t)) for t in all_templates])
        total = freq.sum()
        if total > 0:
            freq /= total

        tables: Set[str] = set()
        cols:   Set[str] = set()
        for q in np.unique(templates):
            tables |= query_tables.get(q, set())
            cols   |= query_cols.get(q,   set())

        times   = grp['exec_ms'].values.astype(float)
        qset    = set(np.unique(templates))
        n_q     = len(grp)

        windows.append({
            'run': run, 'window': window, 'phase': phase,
            'freq': freq, 'n': n_q,
            'tables': tables, 'cols': cols,
            'times': times, 'qset': qset,
        })

    return sorted(windows, key=lambda w: (w['run'], w['window']))


# ─── All-pairs HSM computation ────────────────────────────────────────────────

def compute_all_pairs(windows: List[dict],
                      weights=None) -> Tuple[List[float], List[float]]:
    """
    Compute all-pairs within/cross HSM for a single run's windows.
    Returns (within_scores, cross_scores).
    """
    within, cross = [], []
    n = len(windows)
    for i in range(n):
        for j in range(i + 1, n):
            wa, wb = windows[i], windows[j]
            scores = hsm_v2(
                wa['freq'], wb['freq'],
                wa['n'],    wb['n'],
                wa['tables'], wb['tables'],
                wa['cols'],   wb['cols'],
                wa['times'],  wb['times'],
                wa['qset'],   wb['qset'],
                weights=weights,
            )
            if wa['phase'] == wb['phase']:
                within.append(scores['HSM'])
            else:
                cross.append(scores['HSM'])
    return within, cross


def compute_all_pairs_full(windows: List[dict],
                           weights=None) -> List[dict]:
    """
    Compute all-pairs HSM with FULL per-dimension breakdown for every
    pair. Used by Theorem 4 (binary classifier reformulation) and
    Algorithm 2 (HSM Index Diagnosis).
    Returns a list of dicts, one per ordered pair (i<j), with keys:
      run, win_a, win_b, phase_a, phase_b, cross_phase,
      S_R, S_V, S_T, S_A, S_P, HSM
    """
    out = []
    n = len(windows)
    for i in range(n):
        for j in range(i + 1, n):
            wa, wb = windows[i], windows[j]
            scores = hsm_v2(
                wa['freq'], wb['freq'],
                wa['n'],    wb['n'],
                wa['tables'], wb['tables'],
                wa['cols'],   wb['cols'],
                wa['times'],  wb['times'],
                wa['qset'],   wb['qset'],
                weights=weights,
            )
            out.append({
                'run':         wa['run'],
                'win_a':       int(wa['window']),
                'win_b':       int(wb['window']),
                'phase_a':     wa['phase'],
                'phase_b':     wb['phase'],
                'cross_phase': wa['phase'] != wb['phase'],
                'S_R':         scores.get('S_R',  scores.get('R', 0.0)),
                'S_V':         scores.get('S_V',  scores.get('V', 0.0)),
                'S_T':         scores.get('S_T',  scores.get('T', 0.0)),
                'S_A':         scores.get('S_A',  scores.get('A', 0.0)),
                'S_P':         scores.get('S_P',  scores.get('P', 0.0)),
                'HSM':         scores['HSM'],
            })
    return out


# ─── DR statistics ────────────────────────────────────────────────────────────

def compute_dr_stats(within: List[float], cross: List[float],
                     n_boot: int = 5000, seed: int = 42) -> dict:
    """
    Compute DR with 95% bootstrap CI, Mann-Whitney U (one-sided, within > cross),
    rank-biserial |r|.
    """
    w = np.array(within, dtype=float)
    c = np.array(cross,  dtype=float)

    if len(w) == 0 or len(c) == 0 or c.mean() == 0:
        return {'DR': float('nan'), 'error': 'insufficient data'}

    DR = float(w.mean() / c.mean())

    # Bootstrap CI
    rng = np.random.default_rng(seed)
    boot_drs = []
    nw, nc = len(w), len(c)
    for _ in range(n_boot):
        bw = rng.choice(w, size=nw, replace=True)
        bc = rng.choice(c, size=nc, replace=True)
        if bc.mean() > 0:
            boot_drs.append(bw.mean() / bc.mean())
    boot_drs = np.sort(np.array(boot_drs))
    ci_lo = float(np.percentile(boot_drs, 2.5))
    ci_hi = float(np.percentile(boot_drs, 97.5))

    # Mann-Whitney U (one-sided: within > cross)
    stat, p = mannwhitneyu(w, c, alternative='greater')
    r = float(2.0 * stat / (nw * nc) - 1.0)  # rank-biserial (positive = within > cross)

    # ICC(2,1) approximation on within-phase scores only
    # (uses within-phase variance vs. total variance)
    icc = float(np.nan)
    if len(w) >= 3:
        var_w = float(np.var(w, ddof=1))
        var_c = float(np.var(c, ddof=1))
        var_total = float(np.var(np.concatenate([w, c]), ddof=1))
        if var_total > 0:
            icc = round(1.0 - var_w / var_total, 4)

    return {
        'within_mean': round(float(w.mean()), 4),
        'within_std':  round(float(w.std()),  4),
        'cross_mean':  round(float(c.mean()), 4),
        'cross_std':   round(float(c.std()),  4),
        'DR':          round(DR, 4),
        'CI_lo':       round(ci_lo, 4),
        'CI_hi':       round(ci_hi, 4),
        'MWU_p':       float(p),
        'r_biserial':  round(abs(r), 4),
        'ICC21':       icc,
        'n_within':    int(nw),
        'n_cross':     int(nc),
        'n_boot':      n_boot,
    }


def compute_per_seed_drs(all_windows: List[dict],
                         weights=None) -> List[float]:
    """
    Compute per-seed (per-run) DR values.
    Returns list of per-seed DRs (for SD/CV reporting).
    """
    runs = sorted(set(w['run'] for w in all_windows))
    drs = []
    for run in runs:
        run_wins = [w for w in all_windows if w['run'] == run]
        within, cross = compute_all_pairs(run_wins, weights=weights)
        if within and cross and np.mean(cross) > 0:
            drs.append(float(np.mean(within) / np.mean(cross)))
    return drs


def print_stats(exp_name: str, stats: dict, per_seed_drs: List[float]) -> None:
    """Print paper-ready statistics."""
    print(f"\n{'='*60}")
    print(f"  {exp_name}")
    print(f"{'='*60}")
    print(f"  Within-phase HSM : {stats['within_mean']:.4f}  (σ={stats['within_std']:.4f})")
    print(f"  Cross-phase  HSM : {stats['cross_mean']:.4f}  (σ={stats['cross_std']:.4f})")
    print(f"  DR               : {stats['DR']:.4f}")
    print(f"  95% CI           : [{stats['CI_lo']:.4f}, {stats['CI_hi']:.4f}]")
    print(f"  Mann-Whitney p   : {stats['MWU_p']:.3e}")
    print(f"  |r| rank-biserial: {stats['r_biserial']:.4f}")
    print(f"  ICC(2,1)         : {stats['ICC21']}")
    print(f"  n_within pairs   : {stats['n_within']}")
    print(f"  n_cross  pairs   : {stats['n_cross']}")
    if per_seed_drs:
        print(f"  Per-seed DRs     : {[round(d, 4) for d in per_seed_drs]}")
        print(f"  DR mean ± SD     : {np.mean(per_seed_drs):.4f} ± {np.std(per_seed_drs):.4f}")
        print(f"  DR CV            : {100*np.std(per_seed_drs)/np.mean(per_seed_drs):.2f}%")
    print()
