from __future__ import annotations

import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from openadmet_pxr_repo.audit import build_report_frame, save_report
from openadmet_pxr_repo.selection import best_honest_candidate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("/content"))
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--n-boot", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260623)
    args = parser.parse_args()

    result = build_report_frame(args.root, n_boot=args.n_boot, seed=args.seed)
    report = result["report"]

    out = args.out or (args.root / "phase1_honest_metrics_report.csv")
    save_report(report, out)

    print("=" * 88)
    print("HONEST PHASE1 AUDIT")
    print("=" * 88)
    print("Phase1 file:", result["phase1_path"])
    print("Saved report:", out)

    if len(result["honest"]):
        best = best_honest_candidate(report)
        print("\nBest honest candidate:")
        print(f"  file: {best['file']}")
        print(f"  mae:  {best['mae']:.6f}")
        print(f"  rae:  {best['rae_fixed']:.6f}")
        print(f"  mae_ci: {best['mae_ci']}")
        print(f"  rae_ci: {best['rae_fixed_ci']}")
    else:
        print("No honest candidates found.")

    if result["contaminated"]:
        print("\nContaminated exact-filled files skipped:")
        for name in result["contaminated"]:
            print(" -", name)


if __name__ == "__main__":
    main()

