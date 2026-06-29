from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from openadmet_pxr_repo.sub040_signal_experiment import _stratified_scaffold_folds




def main() -> None:
    parser = argparse.ArgumentParser(description="Create reproducible molecule-level CV folds.")
    parser.add_argument("--input", type=Path, default=ROOT / "data" / "phase1_unblinded.csv")
    parser.add_argument("--out", type=Path, default=ROOT / "data" / "phase1_cv_splits.csv")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260623)
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    required = {"Molecule Name", "SMILES", "pEC50"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{args.input} is missing required columns: {sorted(missing)}")

    fold = _stratified_scaffold_folds(df, args.folds, args.seed)

    out = df[["Molecule Name"]].copy()
    out["fold"] = fold
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)
    print(f"Saved CV splits: {args.out}")


if __name__ == "__main__":
    main()
