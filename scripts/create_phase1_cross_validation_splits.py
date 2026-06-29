from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description="Create reproducible molecule-level CV folds.")
    parser.add_argument("--input", type=Path, default=ROOT / "data" / "phase1_unblinded.csv")
    parser.add_argument("--out", type=Path, default=ROOT / "data" / "phase1_cv_splits.csv")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260623)
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    if "Molecule Name" not in df.columns:
        raise ValueError(f"{args.input} must contain 'Molecule Name'")

    fold = np.full(len(df), -1, dtype=int)
    splitter = KFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    for fold_id, (_, val_idx) in enumerate(splitter.split(df)):
        fold[val_idx] = fold_id

    out = df[["Molecule Name"]].copy()
    out["fold"] = fold
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)
    print(f"Saved CV splits: {args.out}")


if __name__ == "__main__":
    main()
