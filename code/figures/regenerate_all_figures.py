"""
regenerate_all_figures.py
--------------------------
Run every plot_*.py script in this directory.  Each script will produce a
real figure if its input CSV exists under results/, otherwise it will
fall back to the watermarked placeholder so the paper still compiles.

Usage:
    python regenerate_all_figures.py [REPO_ROOT]

REPO_ROOT defaults to the project root (two levels up from this file).
"""

from __future__ import annotations

import importlib.util
import os
import sys
from glob import glob

HERE = os.path.dirname(os.path.abspath(__file__))


def _import(path: str):
    spec = importlib.util.spec_from_file_location("_plot_module", path)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)   # type: ignore[union-attr]
    return mod


def main(repo_root: str) -> None:
    sys.path.insert(0, HERE)        # ensure _style / _placeholder importable
    scripts = sorted(glob(os.path.join(HERE, "plot_*.py")))
    if not scripts:
        print("no plot_*.py scripts found", file=sys.stderr)
        sys.exit(1)
    for s in scripts:
        mod = _import(s)
        if hasattr(mod, "main"):
            mod.main(repo_root)
        else:
            print(f"warning: {os.path.basename(s)} has no main()", file=sys.stderr)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        root = sys.argv[1]
    else:
        # default: ../../  relative to this file (HSM_gated repo root)
        root = os.path.normpath(os.path.join(HERE, os.pardir, os.pardir))
    main(root)
