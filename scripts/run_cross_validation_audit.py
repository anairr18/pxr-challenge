from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run honest Phase 1 audit as the CV gate.")
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--n-boot", type=int, default=5000)
    args = parser.parse_args()

    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "run_phase1_honest_audit.py"),
        "--root",
        str(args.root),
        "--n-boot",
        str(args.n_boot),
    ]
    raise SystemExit(subprocess.call(cmd))


if __name__ == "__main__":
    main()
