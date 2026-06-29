from __future__ import annotations

import argparse
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Suiren inactive/tail-weight training entry point placeholder."
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--skip-training", action="store_true")
    args = parser.parse_args()

    out = ROOT / "submissions" / "suiren_inactive_tail_weighted_three_seed_predictions.csv"
    print("Suiren training/inference requires the Suiren code and pretrained checkpoint.")
    print("Seeds:", args.seeds)
    print("Skip training:", args.skip_training)
    print("Expected component output:", out)
    print("Place the precomputed Suiren CSV there before building the final ensemble.")


if __name__ == "__main__":
    main()
