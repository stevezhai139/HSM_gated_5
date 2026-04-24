"""Tests for ``rq5_boundaries.py``.

Covers the inline broken-stick regression helper and the end-to-end
pipeline on a tiny synthetic workload. SDSS SkyLog smoke-run is exercised
in a separate conditional test that skips when the CSV is unavailable.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from cal import rq5_boundaries as r5
from cal.gate import DEFAULT_CONFIG


# ------------------------------- broken-stick unit tests -------------------


def _synthetic_broken_stick(
    sizes: np.ndarray,
    true_tau: float,
    slope_left: float,
    slope_right: float,
    intercept: float,
    noise_sd: float = 0.0,
    seed: int = 0,
) -> np.ndarray:
    """Generate F1 values from a known broken-stick model."""
    log_w = np.log(sizes)
    f1 = intercept + slope_left * log_w + (slope_right - slope_left) * np.maximum(
        0.0, log_w - true_tau
    )
    if noise_sd > 0:
        rng = np.random.default_rng(seed)
        f1 = f1 + rng.normal(scale=noise_sd, size=f1.shape)
    return f1


class TestBrokenStick:
    def test_recovers_knee_on_noiseless_data(self):
        # Dense grid so the candidate-tau search has room to find the true
        # break-point.
        sizes = np.array([2, 3, 5, 7, 10, 14, 20, 30, 50, 70, 100, 150, 200])
        true_w_min = 20.0
        true_tau = float(np.log(true_w_min))
        f1 = _synthetic_broken_stick(
            sizes, true_tau, slope_left=0.3, slope_right=0.0, intercept=-0.2
        )
        fit = r5.fit_broken_stick(sizes.tolist(), f1.tolist(), n_tau_candidates=400)
        # Because the candidate grid is dense but finite and the interior
        # of log|W| excludes the endpoints, we allow a ~ half-step
        # tolerance on the recovered W_min.
        assert abs(fit.w_min - true_w_min) / true_w_min < 0.25

    def test_slope_change_sign(self):
        sizes = np.array([5, 10, 20, 50, 100])
        f1 = _synthetic_broken_stick(
            sizes, true_tau=np.log(20.0),
            slope_left=0.4, slope_right=-0.1, intercept=-0.3,
        )
        fit = r5.fit_broken_stick(sizes.tolist(), f1.tolist())
        assert fit.slope_left > 0
        assert fit.slope_right < fit.slope_left

    def test_bootstrap_ci_non_degenerate(self):
        sizes = np.array([5, 10, 20, 50, 100])
        f1 = _synthetic_broken_stick(
            sizes, true_tau=np.log(20.0),
            slope_left=0.3, slope_right=0.05, intercept=-0.2,
        )
        # Construct per-size bootstrap samples by adding Gaussian noise.
        rng = np.random.default_rng(17)
        boot = [rng.normal(loc=f, scale=0.02, size=200).tolist() for f in f1]
        fit = r5.fit_broken_stick(
            sizes.tolist(),
            f1.tolist(),
            bootstrap_f1=boot,
            confidence_level=0.95,
            seed=3,
        )
        assert fit.w_min_ci_lo <= fit.w_min <= fit.w_min_ci_hi
        # With moderate noise the CI should be non-trivial but still bounded.
        assert fit.w_min_ci_hi - fit.w_min_ci_lo > 0.0

    def test_rejects_tiny_grid(self):
        with pytest.raises(ValueError, match=">= 3 grid points"):
            r5.fit_broken_stick([5, 10], [0.6, 0.7])

    def test_rejects_negative_window_sizes(self):
        with pytest.raises(ValueError, match=">= 2"):
            r5.fit_broken_stick([1, 5, 10], [0.3, 0.5, 0.7])


# ------------------------------- end-to-end pipeline ----------------------


def _synthetic_trace_two_phases() -> list[tuple[str, float]]:
    """Build a controlled 2-phase trace so ground truth is meaningful."""
    phase_a = [
        ("SELECT * FROM photoobj WHERE ra BETWEEN 10 AND 20", 2.0),
        ("SELECT ra, dec FROM photoobj WHERE dec > 0", 3.0),
        ("SELECT objid FROM photoobj WHERE u < 18", 2.5),
        ("SELECT u, g, r FROM photoobj WHERE r < 17 AND g > 15", 4.0),
    ]
    phase_b = [
        ("SELECT plate, mjd FROM specobj WHERE class='GALAXY'", 5.0),
        ("SELECT z, zerr FROM specobj WHERE plate BETWEEN 300 AND 400", 4.5),
        ("SELECT fiber, plate FROM specobj WHERE z > 0.5", 6.0),
        ("SELECT ra, dec FROM specobj WHERE class='QSO'", 5.5),
    ]
    trace: list[tuple[str, float]] = []
    # Alternate 40 queries of each phase to get enough windows at size 20.
    for _ in range(40):
        trace.extend(phase_a)
    for _ in range(40):
        trace.extend(phase_b)
    for _ in range(40):
        trace.extend(phase_a)
    return trace


class TestWindowSweepEndToEnd:
    def test_runs_on_synthetic_two_phase_trace(self):
        trace = _synthetic_trace_two_phases()
        result = r5.run_window_sweep(
            trace,
            window_grid=(5, 10, 20, 50),
            gate_config=DEFAULT_CONFIG,
            n_bootstrap=50,
            seed=1,
        )
        # Schema checks
        assert "per_size" in result
        assert "broken_stick" in result
        assert "meta" in result
        assert len(result["per_size"]) == 4
        for row in result["per_size"]:
            assert {
                "window_size",
                "n_windows",
                "n_pairs",
                "f1",
                "fpr",
                "fnr",
                "precision",
                "recall",
                "bootstrap_f1_mean",
                "bootstrap_f1_std",
            }.issubset(row.keys())
            assert 0.0 <= row["f1"] <= 1.0
            assert 0.0 <= row["fpr"] <= 1.0
            assert 0.0 <= row["fnr"] <= 1.0
        # Broken-stick CI must enclose the point estimate.
        bs = result["broken_stick"]
        assert bs is not None
        assert bs["w_min_ci_lo"] <= bs["w_min"] <= bs["w_min_ci_hi"]

    def test_small_grid_produces_null_broken_stick(self):
        trace = _synthetic_trace_two_phases()
        # Only 2 usable grid points → broken-stick fit should be None.
        result = r5.run_window_sweep(
            trace,
            window_grid=(5, 10),
            n_bootstrap=10,
            seed=1,
        )
        assert result["broken_stick"] is None


# ------------------------------- SDSS SkyLog smoke (opt-in) ---------------


@pytest.mark.skipif(
    not (
        Path(
            r5._EXPERIMENTS_DIR.parents[1]
            / "code"
            / "data"
            / "sdss"
            / "SkyLog_Workload.csv"
        ).is_file()
    ),
    reason="SkyLog_Workload.csv not present",
)
def test_sdss_sim_loader_returns_records():
    trace = r5.load_sdss_sim_trace(max_records=200)
    assert len(trace) > 0
    assert all(isinstance(t[0], str) and isinstance(t[1], float) for t in trace)


@pytest.mark.skipif(
    not (
        Path(
            r5._EXPERIMENTS_DIR.parents[1]
            / "code"
            / "data"
            / "sdss"
            / "SkyLog_Workload.csv"
        ).is_file()
    ),
    reason="SkyLog_Workload.csv not present",
)
def test_sdss_sim_smoke_end_to_end(tmp_path):
    trace = r5.load_sdss_sim_trace(max_records=1200)
    result = r5.run_window_sweep(
        trace,
        window_grid=(5, 10, 20, 50),
        n_bootstrap=30,
        seed=42,
    )
    out_path = tmp_path / "rq5_sdss_smoke.json"
    with out_path.open("w") as fh:
        json.dump(result, fh, default=float)
    # Reload to validate JSON-serialisability.
    payload = json.loads(out_path.read_text())
    assert payload["meta"]["n_bootstrap"] == 30
    assert len(payload["per_size"]) == 4
