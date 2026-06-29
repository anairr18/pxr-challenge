from __future__ import annotations

import argparse
from pathlib import Path
import sys
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from openadmet_pxr_repo.audit import build_report_frame
from openadmet_pxr_repo.selection import copy_best_candidate, best_honest_candidate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("/content"))
    parser.add_argument("--report", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    if args.report is not None and args.report.exists():
        report = pd.read_csv(args.report)
    else:
        report = build_report_frame(args.root)["report"]

    best = best_honest_candidate(report)
    out = copy_best_candidate(args.root, report, args.out)
    print("Best honest candidate:", best["file"])
    print("Copied to:", out)


if __name__ == "__main__":
    main()

