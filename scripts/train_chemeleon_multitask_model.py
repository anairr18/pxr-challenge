from __future__ import annotations

import argparse
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="CheMeleon training entry point placeholder for the submission repo."
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-lr", type=float, default=3e-4)
    parser.add_argument("--epochs", type=int, default=25)
    args = parser.parse_args()

    out = ROOT / "submissions" / f"chemeleon_multitask_seed{args.seed}_predictions.csv"
    print("CheMeleon training requires the full Chemprop/CheMeleon environment and checkpoint.")
    print("Expected output path:", out)
    print("This lightweight repo keeps the same entry point as the reference repo.")
    print("Add the real training implementation or place the precomputed seed CSV there.")


if __name__ == "__main__":
    main()
