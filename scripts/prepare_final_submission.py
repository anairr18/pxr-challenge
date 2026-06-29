from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description="Select final non-contaminated candidate.")
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "submissions" / "activity_predictions_final.csv",
    )
    args = parser.parse_args()

    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "select_best_submission_candidate.py"),
        "--root",
        str(args.root),
        "--out",
        str(args.out),
    ]
    raise SystemExit(subprocess.call(cmd))


if __name__ == "__main__":
    main()
