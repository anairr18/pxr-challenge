from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from openadmet_pxr_repo.curve_multitask_experiment import run_curve_multitask_experiment


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run the curve-aware multitask experiment. This trains auxiliary "
            "heads for curve-fit, counter-assay, multitask, and single-concentration "
            "endpoints, then tests whether those signals improve the frozen activity anchor "
            "under scaffold-stratified nested validation."
        )
    )
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--n-boot", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260624)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    summary = run_curve_multitask_experiment(
        args.root,
        n_folds=args.folds,
        n_boot=args.n_boot,
        seed=args.seed,
        output_dir=args.output_dir,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
