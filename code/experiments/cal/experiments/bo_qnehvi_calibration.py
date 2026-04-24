"""RQ4a — Offline Bayesian-Optimisation calibration of (w, θ) via qLogNEHVI.

Canonical spec: ``Paper3B_Cal_Experiment_Plan_v0.docx`` §5, updated for M10
direct-simplex parameterisation per 2026-04-24 design review (symmetry in
w coordinates, no pinning bias).

Key design decisions:

* **Parameterisation (M10 — direct simplex, symmetric)**:
    - Weights ``w`` live on the 5-simplex Δ⁴ and are optimised directly.
      BoTorch's ``optimize_acqf`` handles the simplex via
      ``equality_constraints`` (Σ w_i = 1) and ``inequality_constraints``
      (w_i ≥ ε). All 5 coordinates are treated symmetrically — no pinning,
      no softmax, no implicit bias toward any specific w_i.
    - Threshold ``θ`` lives on the OPEN interval ``(ε, 1 − ε)`` with
      ``ε = 1e-6``. No prescriptive [0.5, 0.95] bound (that was the
      EMPIRICAL band Paper 3A Supp §XIV observed, not a BO search-space
      restriction). See Plan §3.
* **Objective**: (Precision, Recall) from gate decisions on a phase-labelled
  TPC-H trace. qLogNEHVI optimises the expected log-hypervolume improvement
  of this 2-D Pareto front against reference point r = (0, 0). See Plan §5.1.
* **Acquisition**: qLogNEHVI (Ament et al. arXiv 2310.20708, 2023) — numerically
  stable log-space reformulation of qNEHVI (Daulton et al. NeurIPS 2021).
  Same underlying hypervolume-improvement method; BoTorch 0.17+ emits a
  deprecation warning if plain qNEHVI is used.
* **Budget**: 10 Sobol init points + 50 BO iterations per (block, window),
  as specified in Plan §5.2–§5.3.
* **Auto-manipulation**: every BO iteration fits a GP posterior on prior
  history and proposes the next (w, θ) via ``optimize_acqf`` maximising
  qLogNEHVI subject to simplex constraints. This is the manipulation
  mechanism (not a grid). See Plan §4.

Integrity rules:
* Uses the vendored Paper 3A kernel via ``cal.vendored.hsm_similarity`` and
  the vendored workload generator via ``cal.vendored.workload_generator``;
  never reaches into the live ``code/experiments/*`` files.
* Results go under ``results/cal/experiments/`` with git-SHA + timestamp
  provenance inherited from ``cal.validation._run_meta``.

Design review changelog:
* 2026-04-23 (v0): softmax pinning parameterisation (M3) — had ~10%
  exploration bias toward the pinned coord (w_P).
* 2026-04-24 (v0.1, this file): refactored to M10 direct-simplex —
  symmetric coverage of all 5 w coordinates via BoTorch equality +
  inequality constraints in optimize_acqf.
"""

from __future__ import annotations

import json
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# sys.path fixup so cal.* resolves regardless of invocation style
# ---------------------------------------------------------------------------
_THIS = Path(__file__).resolve()
_EXPERIMENTS_DIR = _THIS.parents[2]  # code/experiments/
if str(_EXPERIMENTS_DIR) not in sys.path:
    sys.path.insert(0, str(_EXPERIMENTS_DIR))

# Vendored Paper 3A kernel (immutable v5.0.0 snapshot)
from cal.vendored.hsm_similarity import (  # noqa: E402
    WorkloadWindow,
    build_window,
    hsm_score,
)
from cal.vendored.workload_generator import get_workload_trace  # noqa: E402

# Paper 3B primitives
from cal.metrics import fpr_fnr  # noqa: E402


EPS = 1e-6  # numerical open-interval bound for θ ∈ (ε, 1−ε) and w_i ≥ ε
N_DIMS_HSM = 5  # S_R, S_V, S_T, S_A, S_P → w ∈ Δ⁴ has 5 components
N_DIMS_TOTAL = 6  # w (5) + θ (1) = 6-D search space


# ---------------------------------------------------------------------------
# Simplex utilities (used for Sobol init projection)
# ---------------------------------------------------------------------------


def project_to_simplex(x: np.ndarray) -> np.ndarray:
    """Project a non-negative vector onto the simplex via ℓ¹-normalisation.

    For a vector with ``x_i ≥ 0`` and sum > 0, returns ``x / Σ x_i`` which
    lies on the standard simplex Δⁿ⁻¹. Used for projecting Sobol samples
    drawn uniformly in [0, 1]⁵ onto the 5-simplex for BO initialisation.
    """
    x = np.asarray(x, dtype=float)
    s = x.sum(axis=-1, keepdims=True)
    # Numerical safeguard: replace zero-sum with uniform distribution.
    uniform = np.full_like(x, 1.0 / x.shape[-1])
    return np.where(s > 0, x / np.where(s > 0, s, 1.0), uniform)


def simplex_weights_dict(w: np.ndarray) -> Dict[str, float]:
    """Convert w ∈ Δ⁴ (5-component) into the {w_R, w_V, w_T, w_A, w_P} dict
    expected by the vendored ``hsm_score`` call."""
    if w.shape[-1] != N_DIMS_HSM:
        raise ValueError(f"weights must be 5-D (got {w.shape})")
    return {
        "w_R": float(w[..., 0]),
        "w_V": float(w[..., 1]),
        "w_T": float(w[..., 2]),
        "w_A": float(w[..., 3]),
        "w_P": float(w[..., 4]),
    }


# ---------------------------------------------------------------------------
# Workload trace ↔ WorkloadWindow pipeline
# ---------------------------------------------------------------------------


# TPC-H phase labels produced by vendored workload_generator: the phase list
# is [A, B, A_repeat, C]. We encode transitions between phases as the
# ground-truth shift signal.
_PHASE_ORDER = ("Phase_A", "Phase_B", "Phase_A_repeat", "Phase_C")


def _phase_for_query_idx(query_idx: int, queries_per_phase: int) -> str:
    """Return phase name for the given absolute query index in the trace."""
    phase_idx = min(query_idx // queries_per_phase, len(_PHASE_ORDER) - 1)
    return _PHASE_ORDER[phase_idx]


@dataclass(frozen=True)
class PhaseLabelledWindows:
    """A binned phase-labelled trace ready for (w, θ) evaluation."""

    windows: Tuple[WorkloadWindow, ...]
    phase_labels: Tuple[str, ...]  # one per window
    transition_labels: Tuple[int, ...]  # per-pair: 1 if adjacent phases differ

    @property
    def n_pairs(self) -> int:
        return len(self.transition_labels)


def build_labelled_windows(
    trace: Sequence[Tuple[str, str]],
    window_size: int,
    queries_per_phase: int,
    *,
    duration_per_query_s: float = 1.0,
) -> PhaseLabelledWindows:
    """Bin a TPC-H phase trace into windows + per-pair transition labels.

    Parameters
    ----------
    trace : sequence of (query_id, sql_string)
        Output of ``cal.vendored.workload_generator.get_workload_trace``.
    window_size : number of queries per window.
    queries_per_phase : matches the workload_generator's parameter.
    duration_per_query_s : synthetic per-query timing (for S_P series).

    A window's phase label is the phase of its FIRST query (the phase the
    window starts in). A transition pair has ``phase_labels[i] !=
    phase_labels[i+1]``.
    """
    if window_size < 2:
        raise ValueError(f"window_size must be >= 2 (got {window_size})")
    n_queries = len(trace)
    if n_queries == 0:
        raise ValueError("empty trace")

    windows: List[WorkloadWindow] = []
    phase_labels: List[str] = []
    for start in range(0, n_queries - window_size + 1, window_size):
        sqls = [sql for _, sql in trace[start : start + window_size]]
        timestamps = [i * duration_per_query_s for i in range(window_size)]
        w = build_window(
            sqls,
            timestamps=timestamps,
            window_id=len(windows),
            duration_s=max(window_size * duration_per_query_s, 1.0),
        )
        windows.append(w)
        phase_labels.append(_phase_for_query_idx(start, queries_per_phase))
    if len(windows) < 3:
        raise ValueError(
            f"need >= 3 windows for adjacent-pair evaluation "
            f"(got {len(windows)} from trace of {n_queries} queries with "
            f"window={window_size})"
        )
    transitions = tuple(
        1 if phase_labels[i] != phase_labels[i + 1] else 0
        for i in range(len(windows) - 1)
    )
    return PhaseLabelledWindows(
        windows=tuple(windows),
        phase_labels=tuple(phase_labels),
        transition_labels=transitions,
    )


# ---------------------------------------------------------------------------
# Objective: (Precision, Recall) at proposed (w, θ)
# ---------------------------------------------------------------------------


def evaluate_precision_recall(
    labelled: PhaseLabelledWindows,
    w: np.ndarray,
    theta: float,
) -> Tuple[float, float]:
    """Compute (Precision, Recall) for a single (w, θ) candidate.

    Gate prediction per Paper 3B Theoretical Foundations §4.1:
        y_pred = G_{w,θ}(W_i, W_{i+1}) = 𝟙[ K_w(W_i, W_{i+1}) < θ ]
    Positive class = shift = (phase_labels differ).

    Per-pair HSM similarity uses the vendored kernel
    ``cal.vendored.hsm_similarity.hsm_score(w_a, w_b, weights=...)``.
    """
    weights = simplex_weights_dict(w)
    y_true = list(labelled.transition_labels)
    y_pred: List[int] = []
    windows = labelled.windows
    for i in range(len(windows) - 1):
        score, _dims = hsm_score(windows[i], windows[i + 1], weights=weights)
        y_pred.append(1 if score < theta else 0)
    rates = fpr_fnr(y_true, y_pred)
    return float(rates.precision), float(rates.recall)


# ---------------------------------------------------------------------------
# BoTorch BO-qLogNEHVI loop (M10 direct simplex)
# ---------------------------------------------------------------------------


@dataclass
class BOIterationRecord:
    """Per-iteration trace record for the BO run."""

    iter_index: int  # negative for Sobol-init points, >=0 for BO proposals
    w: List[float]  # 5-D simplex weights (Σw=1, w_i≥ε)
    theta: float
    precision: float
    recall: float
    hypervolume: float


@dataclass
class BOQnehviResult:
    """Full BO-qLogNEHVI result for one (block, window) configuration."""

    block_seed: int
    window_size: int
    queries_per_phase: int
    n_init: int
    n_iter: int
    history: List[BOIterationRecord]
    final_w_star: List[float]
    final_theta_star: float
    final_precision: float
    final_recall: float
    final_hypervolume: float
    pareto_indices: List[int]  # indices into history on the final Pareto front

    def as_dict(self) -> Dict[str, Any]:
        return {
            "block_seed": self.block_seed,
            "window_size": self.window_size,
            "queries_per_phase": self.queries_per_phase,
            "n_init": self.n_init,
            "n_iter": self.n_iter,
            "history": [
                {
                    "iter_index": r.iter_index,
                    "w": r.w,
                    "theta": r.theta,
                    "precision": r.precision,
                    "recall": r.recall,
                    "hypervolume": r.hypervolume,
                }
                for r in self.history
            ],
            "final_w_star": self.final_w_star,
            "final_theta_star": self.final_theta_star,
            "final_precision": self.final_precision,
            "final_recall": self.final_recall,
            "final_hypervolume": self.final_hypervolume,
            "pareto_indices": self.pareto_indices,
        }


def _import_botorch():
    """Lazy-import BoTorch so module imports cheaply when BO isn't used."""
    import torch
    from botorch.acquisition.multi_objective.logei import (
        qLogNoisyExpectedHypervolumeImprovement,
    )
    from botorch.fit import fit_gpytorch_mll
    from botorch.models import SingleTaskGP
    from botorch.models.model_list_gp_regression import ModelListGP
    from botorch.models.transforms.input import Normalize
    from botorch.optim.optimize import optimize_acqf
    from botorch.sampling.normal import SobolQMCNormalSampler
    from botorch.utils.multi_objective.box_decompositions.dominated import (
        DominatedPartitioning,
    )
    from botorch.utils.multi_objective.pareto import is_non_dominated
    from botorch.utils.sampling import draw_sobol_samples
    from gpytorch.mlls.sum_marginal_log_likelihood import SumMarginalLogLikelihood

    return {
        "torch": torch,
        "qLogNEHVI": qLogNoisyExpectedHypervolumeImprovement,
        "fit_mll": fit_gpytorch_mll,
        "SingleTaskGP": SingleTaskGP,
        "ModelListGP": ModelListGP,
        "Normalize": Normalize,
        "optimize_acqf": optimize_acqf,
        "SobolQMCNormalSampler": SobolQMCNormalSampler,
        "DominatedPartitioning": DominatedPartitioning,
        "is_non_dominated": is_non_dominated,
        "draw_sobol_samples": draw_sobol_samples,
        "SumMLL": SumMarginalLogLikelihood,
    }


def _make_simplex_init(
    bt: Dict[str, Any],
    n_init: int,
    block_seed: int,
) -> Any:
    """Build Sobol init points in 6-D box then project w-coords to simplex.

    Returns a tensor of shape (n_init, 6) where rows satisfy:
    - sum of first 5 coords ≈ 1 (after simplex projection)
    - each of first 5 coords ≥ EPS (after projection + clipping)
    - last coord (θ) ∈ (ε, 1−ε)
    """
    torch = bt["torch"]
    # Draw in [0, 1]^6 then scale θ into (ε, 1-ε).
    unit_bounds = torch.tensor(
        [[0.0] * 6, [1.0] * 6], dtype=torch.double
    )
    raw = bt["draw_sobol_samples"](bounds=unit_bounds, n=n_init, q=1, seed=block_seed).squeeze(-2)
    # Project first 5 coords to simplex
    w_raw = raw[..., :5].numpy()
    w_simplex = project_to_simplex(w_raw)
    # Clip to ≥ EPS and renormalise (very mild nudge at boundary)
    w_simplex = np.clip(w_simplex, EPS, None)
    w_simplex = w_simplex / w_simplex.sum(axis=-1, keepdims=True)
    # Scale θ from [0, 1] to (ε, 1-ε)
    theta = raw[..., 5].numpy() * (1.0 - 2.0 * EPS) + EPS
    # Assemble
    init = np.concatenate([w_simplex, theta[..., None]], axis=-1)
    return torch.tensor(init, dtype=torch.double)


def run_bo_qnehvi(
    labelled: PhaseLabelledWindows,
    *,
    block_seed: int,
    window_size: int,
    queries_per_phase: int,
    n_init: int = 10,
    n_iter: int = 50,
    ref_point: Tuple[float, float] = (0.0, 0.0),
    verbose: bool = False,
) -> BOQnehviResult:
    """Run BO-qLogNEHVI on one (block, window) configuration (M10 direct simplex).

    Search space: (w ∈ Δ⁴) × (θ ∈ (ε, 1 − ε)), total 6 coords.
    Simplex enforced via BoTorch's ``equality_constraints`` and
    ``inequality_constraints`` arguments to ``optimize_acqf``.

    Returns a ``BOQnehviResult`` with full history + final Pareto front +
    knee point.
    """
    bt = _import_botorch()
    torch = bt["torch"]

    # Seed everything for reproducibility.
    torch.manual_seed(block_seed + 1000)
    np.random.seed(block_seed)

    # 6-D bounds: w_i ∈ [0, 1] for i=0..4, θ ∈ [ε, 1-ε]
    bounds = torch.tensor(
        [[0.0, 0.0, 0.0, 0.0, 0.0, EPS],
         [1.0, 1.0, 1.0, 1.0, 1.0, 1.0 - EPS]],
        dtype=torch.double,
    )

    # Equality constraint: Σ w_i = 1 (indices 0..4)
    equality_constraints = [
        (torch.tensor([0, 1, 2, 3, 4], dtype=torch.long),
         torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0], dtype=torch.double),
         1.0),
    ]
    # Inequality constraints: w_i ≥ EPS for i=0..4
    # Format: (indices, coeffs, rhs) encodes Σ (X[idx]*coef) ≥ rhs
    inequality_constraints = [
        (torch.tensor([i], dtype=torch.long),
         torch.tensor([1.0], dtype=torch.double),
         EPS)
        for i in range(5)
    ]

    def eval_candidate(x_raw: "torch.Tensor") -> Tuple[float, float]:
        """Evaluate one candidate: extract w, θ → (P, R)."""
        x = x_raw.detach().cpu().numpy().flatten()
        w = x[:N_DIMS_HSM]
        # Renormalise just in case numerical drift pushed Σw ≠ 1
        w = w / w.sum() if w.sum() > 0 else np.full(N_DIMS_HSM, 1.0 / N_DIMS_HSM)
        theta = float(np.clip(x[N_DIMS_HSM], EPS, 1.0 - EPS))
        p, r = evaluate_precision_recall(labelled, w, theta)
        return p, r

    # Sobol init projected to simplex
    init_X = _make_simplex_init(bt, n_init, block_seed)
    init_Y_list: List[List[float]] = []
    history: List[BOIterationRecord] = []
    for j in range(n_init):
        p, r = eval_candidate(init_X[j])
        init_Y_list.append([p, r])
        x_np = init_X[j].detach().cpu().numpy()
        w = x_np[:N_DIMS_HSM]
        w = w / w.sum() if w.sum() > 0 else np.full(N_DIMS_HSM, 1.0 / N_DIMS_HSM)
        theta = float(np.clip(x_np[N_DIMS_HSM], EPS, 1.0 - EPS))
        history.append(BOIterationRecord(
            iter_index=-(n_init - j),  # -n_init..-1 for inits
            w=w.tolist(),
            theta=theta,
            precision=p,
            recall=r,
            hypervolume=0.0,  # filled after init batch below
        ))

    # Compute running hypervolume for init batch
    running_hv = _running_hypervolume([(h.precision, h.recall) for h in history], ref_point)
    for j, hv in enumerate(running_hv):
        history[j] = BOIterationRecord(
            iter_index=history[j].iter_index,
            w=history[j].w,
            theta=history[j].theta,
            precision=history[j].precision,
            recall=history[j].recall,
            hypervolume=hv,
        )

    train_X = init_X.clone()
    train_Y = torch.tensor(init_Y_list, dtype=torch.double)

    for i in range(n_iter):
        # Fit GP: one SingleTaskGP per objective, wrapped in ModelListGP.
        # Normalize transform scales the 6-D box to [0, 1]^6 so GP priors
        # operate on the unit cube (BoTorch's canonical assumption).
        gps = []
        for obj_idx in range(2):
            y = train_Y[..., obj_idx:obj_idx + 1]
            gp = bt["SingleTaskGP"](
                train_X.double(),
                y.double(),
                input_transform=bt["Normalize"](d=train_X.shape[-1], bounds=bounds),
            )
            gps.append(gp)
        model = bt["ModelListGP"](*gps)
        mll = bt["SumMLL"](model.likelihood, model)
        bt["fit_mll"](mll)

        # qLogNEHVI acquisition with simplex constraints
        sampler = bt["SobolQMCNormalSampler"](sample_shape=torch.Size([64]))
        acq = bt["qLogNEHVI"](
            model=model,
            ref_point=list(ref_point),
            X_baseline=train_X.double(),
            sampler=sampler,
            prune_baseline=True,
        )
        candidates, _acq_value = bt["optimize_acqf"](
            acq_function=acq,
            bounds=bounds,
            q=1,
            num_restarts=10,
            raw_samples=128,
            equality_constraints=equality_constraints,
            inequality_constraints=inequality_constraints,
            sequential=True,
        )
        new_x = candidates.detach()
        p, r = eval_candidate(new_x[0])
        train_X = torch.cat([train_X, new_x], dim=0)
        train_Y = torch.cat([train_Y, torch.tensor([[p, r]], dtype=torch.double)], dim=0)

        x_np = new_x[0].detach().cpu().numpy()
        w = x_np[:N_DIMS_HSM]
        w = w / w.sum() if w.sum() > 0 else np.full(N_DIMS_HSM, 1.0 / N_DIMS_HSM)
        theta = float(np.clip(x_np[N_DIMS_HSM], EPS, 1.0 - EPS))
        # Compute cumulative hypervolume incl. this new point
        obj_list = [(h.precision, h.recall) for h in history] + [(p, r)]
        hv = _running_hypervolume(obj_list, ref_point)[-1]
        history.append(BOIterationRecord(
            iter_index=i,
            w=w.tolist(),
            theta=theta,
            precision=p,
            recall=r,
            hypervolume=hv,
        ))
        if verbose:
            print(f"  [iter {i+1}/{n_iter}] w={[f'{x:.3f}' for x in w.tolist()]} "
                  f"θ={theta:.4f} P={p:.3f} R={r:.3f} HV={hv:.4f}")

    # Pareto front from final history
    objs = np.array([[h.precision, h.recall] for h in history])
    pareto_mask = _pareto_mask(objs)
    pareto_indices = [int(i) for i, m in enumerate(pareto_mask) if m]
    # Knee = point on the Pareto front with max (P + R) — simple balanced criterion.
    pareto_points = objs[pareto_mask]
    knee_local = int(np.argmax(pareto_points.sum(axis=1)))
    knee_idx = pareto_indices[knee_local]
    knee = history[knee_idx]

    return BOQnehviResult(
        block_seed=block_seed,
        window_size=window_size,
        queries_per_phase=queries_per_phase,
        n_init=n_init,
        n_iter=n_iter,
        history=history,
        final_w_star=knee.w,
        final_theta_star=knee.theta,
        final_precision=knee.precision,
        final_recall=knee.recall,
        final_hypervolume=knee.hypervolume,
        pareto_indices=pareto_indices,
    )


# ---------------------------------------------------------------------------
# Hypervolume helpers (numpy-only — cross-checks BoTorch's internal HV)
# ---------------------------------------------------------------------------


def _pareto_mask(points: np.ndarray) -> np.ndarray:
    """Return a boolean mask of non-dominated points (Pareto-efficient).
    Maximisation convention (both objectives); a point is dominated if
    there exists another point with >= in both coords and > in at least one."""
    n = points.shape[0]
    mask = np.ones(n, dtype=bool)
    for i in range(n):
        if not mask[i]:
            continue
        for j in range(n):
            if i == j:
                continue
            if np.all(points[j] >= points[i]) and np.any(points[j] > points[i]):
                mask[i] = False
                break
    return mask


def _hypervolume_2d(points: np.ndarray, ref_point: Tuple[float, float]) -> float:
    """Exact 2-D hypervolume (area dominated, against reference point).
    Both objectives maximised. ``ref_point`` is the lower-left corner."""
    pts = points[_pareto_mask(points)]
    if pts.size == 0:
        return 0.0
    # Sort by first objective descending.
    order = np.argsort(-pts[:, 0])
    pts = pts[order]
    rx, ry = ref_point
    hv = 0.0
    prev_y = ry
    for x, y in pts:
        if x <= rx or y <= ry:
            continue
        if y > prev_y:
            hv += (x - rx) * (y - prev_y)
            prev_y = y
    return float(hv)


def _running_hypervolume(
    points: Sequence[Tuple[float, float]], ref_point: Tuple[float, float]
) -> List[float]:
    """Return the hypervolume after observing each prefix of ``points``."""
    if not points:
        return []
    arr = np.array(points, dtype=float)
    return [_hypervolume_2d(arr[: i + 1], ref_point) for i in range(len(arr))]


# ---------------------------------------------------------------------------
# Top-level driver entry points
# ---------------------------------------------------------------------------


def run_single_configuration(
    *,
    block_seed: int,
    window_size: int,
    queries_per_phase: int = 60,
    n_init: int = 10,
    n_iter: int = 50,
    verbose: bool = False,
) -> BOQnehviResult:
    """Run one (block, window) BO-qLogNEHVI calibration.

    Combines trace generation, window binning, and the BO loop.
    """
    trace = get_workload_trace(queries_per_phase=queries_per_phase, seed=block_seed)
    labelled = build_labelled_windows(
        trace, window_size=window_size, queries_per_phase=queries_per_phase
    )
    return run_bo_qnehvi(
        labelled,
        block_seed=block_seed,
        window_size=window_size,
        queries_per_phase=queries_per_phase,
        n_init=n_init,
        n_iter=n_iter,
        verbose=verbose,
    )


__all__ = [
    "EPS",
    "N_DIMS_HSM",
    "N_DIMS_TOTAL",
    "project_to_simplex",
    "simplex_weights_dict",
    "PhaseLabelledWindows",
    "build_labelled_windows",
    "evaluate_precision_recall",
    "BOIterationRecord",
    "BOQnehviResult",
    "run_bo_qnehvi",
    "run_single_configuration",
]
