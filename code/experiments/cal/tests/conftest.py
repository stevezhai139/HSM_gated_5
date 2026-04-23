"""Pytest path fixup — makes `from cal.xxx import ...` work without relying
on the Python-stdlib-shadowed `code.experiments.cal` path.

Matches Paper 3A's convention where `code/experiments/` is treated as a
script root rather than a package root (see `run_all.sh` comments).
"""

from __future__ import annotations

import sys
from pathlib import Path

# code/experiments/  ← put this on sys.path so `cal.xxx` resolves.
_experiments_dir = Path(__file__).resolve().parents[2]
if str(_experiments_dir) not in sys.path:
    sys.path.insert(0, str(_experiments_dir))
