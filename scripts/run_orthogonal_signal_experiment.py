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
            "Run the cheap orthogonal-signal push: 3D descriptors, "
            "single-concentration assay-statistic heads, weighted auxiliary heads, "
            "and matched-molecular-pair cliff features."
        )
    )
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--n-boot", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260623)
    parser.add_argument(
        "--no-3d",
        action="store_true",
        help="Disable ETKDG/MMFF descriptors for a faster smoke run.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Report directory. Defaults to reports/orthogonal_signal_experiment.",
    )
    args = parser.parse_args()

    output_dir = args.output_dir or (args.root / "reports" / "orthogonal_signal_experiment")
    summary = run_structure_assay_experiment(
        args.root,
        n_folds=args.folds,
        n_boot=args.n_boot,
        seed=args.seed,
        with_3d=not args.no_3d,
        use_single_concentration=True,
        use_mmps=True,
        use_weighted_aux=True,
        output_dir=output_dir,
        oof_candidate_file="orthogonal_signal_oof_candidate.csv",
        upload_candidate_file="orthogonal_signal_upload_candidate.csv",
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
