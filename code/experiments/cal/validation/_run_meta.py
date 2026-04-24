"""Provenance capture for Paper 3B-Cal experiment runs.

Emits a single ``RunMeta`` dict that every result artifact embeds.
Captures:
- git SHA (short + full), branch, dirty flag
- UTC timestamp (ISO-8601)
- hostname, platform, Python + key-library versions
- CLI args (caller-provided)

Design: pure-Python stdlib only so this module is always importable, even
when scientific dependencies are not installed. Subprocess calls out to
``git`` read-only; failures degrade gracefully to placeholder strings
rather than raising — a dirty-environment or CI checkout without git
history must still produce a result file.
"""

from __future__ import annotations

import datetime as _dt
import os
import platform as _platform
import socket
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Sequence


def _git(cmd: Sequence[str], cwd: Path) -> Optional[str]:
    """Run ``git`` read-only; return stdout stripped, or None on failure."""
    try:
        r = subprocess.run(
            ["git", *cmd],
            cwd=str(cwd),
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        if r.returncode != 0:
            return None
        return r.stdout.strip() or None
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return None


def _repo_root() -> Path:
    """Locate HSM_gated/ repo root by walking parents of this file."""
    here = Path(__file__).resolve()
    for parent in [here, *here.parents]:
        if (parent / ".git").exists():
            return parent
    # Fallback: assume we are at HSM_gated/code/experiments/cal/validation/
    return here.parents[4]


def capture_git(repo_root: Optional[Path] = None) -> Dict[str, Optional[str]]:
    """Capture git state for provenance. All fields None-safe."""
    root = repo_root or _repo_root()
    sha_full = _git(["rev-parse", "HEAD"], root)
    sha_short = _git(["rev-parse", "--short", "HEAD"], root)
    branch = _git(["rev-parse", "--abbrev-ref", "HEAD"], root)
    # Dirty = uncommitted changes OR untracked-but-tracked files; keep simple.
    status = _git(["status", "--porcelain"], root)
    dirty = bool(status) if status is not None else None
    return {
        "sha_full": sha_full,
        "sha_short": sha_short,
        "branch": branch,
        "dirty": dirty,
        "repo_root": str(root),
    }


def capture_env() -> Dict[str, Any]:
    """Capture environment fingerprint (host, Python, key libraries)."""
    out: Dict[str, Any] = {
        "hostname": socket.gethostname(),
        "platform": _platform.platform(),
        "python_version": sys.version.split()[0],
        "python_executable": sys.executable,
        "user": os.environ.get("USER") or os.environ.get("USERNAME"),
        "cwd": os.getcwd(),
    }
    # Soft-probe scientific libraries; do not import if missing.
    for lib_name in ("numpy", "scipy", "pandas", "sklearn", "matplotlib"):
        try:
            mod = __import__(lib_name)
            out[f"{lib_name}_version"] = getattr(mod, "__version__", None)
        except ImportError:
            out[f"{lib_name}_version"] = None
    return out


def capture(
    *,
    experiment: str,
    cli_args: Sequence[str],
    seed: int,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Full provenance record for a single experiment run.

    Parameters
    ----------
    experiment : str
        Short identifier (e.g., ``"job_theta_sweep"``) — also used for the
        results subdirectory name.
    cli_args : sequence of str
        Argv for this run (excluding the script name) so the run is
        reproducible from the JSON alone.
    seed : int
        RNG seed used by this run.
    extra : dict, optional
        Additional per-run key/values to embed verbatim.

    Returns
    -------
    dict with keys {timestamp_utc, experiment, seed, cli_args, git, env,
                    extra (optional)}.
    """
    ts = _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    meta = {
        "timestamp_utc": ts,
        "experiment": experiment,
        "seed": int(seed),
        "cli_args": list(cli_args),
        "git": capture_git(),
        "env": capture_env(),
    }
    if extra:
        meta["extra"] = dict(extra)
    return meta


def slug_for_filename(meta: Dict[str, Any]) -> str:
    """Produce a filesystem-safe slug: ``YYYYMMDD-HHMMSS_sha_experiment_seedN``."""
    ts_raw = meta["timestamp_utc"]
    ts_compact = ts_raw.replace("-", "").replace(":", "").replace("T", "-").rstrip("Z")
    sha = meta["git"].get("sha_short") or "nosha"
    exp = meta["experiment"].replace(" ", "_")
    seed = meta["seed"]
    return f"{ts_compact}_{sha}_{exp}_seed{seed}"


def append_experiment_log(
    log_path: Path,
    meta: Dict[str, Any],
    summary: str,
) -> None:
    """Append one Markdown row to the running experiment log.

    Creates the file with a header on first use. Each row contains
    timestamp, git SHA, experiment, seed, CLI args, a short one-line
    summary, and a pointer to the result directory (summary-controlled).
    """
    log_path.parent.mkdir(parents=True, exist_ok=True)
    header_needed = not log_path.exists()
    sha = meta["git"].get("sha_short") or "nosha"
    dirty = meta["git"].get("dirty")
    dirty_tag = "+dirty" if dirty else ""
    cli = " ".join(meta["cli_args"])
    ts = meta["timestamp_utc"]
    row = (
        f"| {ts} "
        f"| `{sha}{dirty_tag}` "
        f"| {meta['experiment']} "
        f"| {meta['seed']} "
        f"| `{cli}` "
        f"| {summary} |\n"
    )
    with log_path.open("a") as fh:
        if header_needed:
            fh.write("# Paper 3B-Cal validation — experiment log\n\n")
            fh.write("Append-only. One row per run. Do not hand-edit existing rows.\n\n")
            fh.write("| Timestamp (UTC) | Git SHA | Experiment | Seed | CLI | Summary |\n")
            fh.write("|---|---|---|---|---|---|\n")
        fh.write(row)


__all__ = [
    "capture",
    "capture_env",
    "capture_git",
    "slug_for_filename",
    "append_experiment_log",
]
