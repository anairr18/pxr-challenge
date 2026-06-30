from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from openadmet_pxr_repo.sub040_signal_experiment import run_structure_assay_experiment


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run the honest sub-0.40 structure/assay residual experiment. "
            "This writes experimental candidate CSVs and reports, but does not "
            "replace the current final upload file unless a human explicitly chooses to."
        )
    )
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--n-boot", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260623)
    parser.add_argument("--with-3d", action="store_true", help="Add ETKDG/MMFF 3D descriptors.")
    parser.add_argument(
        "--use-single-concentration",
        action="store_true",
        help="Add predicted features from the single-concentration replicate/statistics table.",
    )
    parser.add_argument(
        "--use-mmps",
        action="store_true",
        help="Add train-neighborhood matched-molecular-pair cliff features.",
    )
    parser.add_argument(
        "--use-weighted-aux",
        action="store_true",
        help="Use confidence weights when fitting auxiliary feature heads where available.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Report directory. Defaults to reports/sub040_structure_assay_experiment.",
    )
    args = parser.parse_args()

    summary = run_structure_assay_experiment(
        args.root,
        n_folds=args.folds,
        n_boot=args.n_boot,
        seed=args.seed,
        with_3d=args.with_3d,
        use_single_concentration=args.use_single_concentration,
        use_mmps=args.use_mmps,
        use_weighted_aux=args.use_weighted_aux,
        output_dir=args.output_dir,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
