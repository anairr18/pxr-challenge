from __future__ import annotations

import argparse
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Conformation-aware predictor training entry point placeholder."
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--skip-training", action="store_true")
    args = parser.parse_args()

    out = ROOT / "submissions" / "conformation_predictor_predictions.csv"
    print("Conformation-aware training/inference requires the external model code and checkpoint.")
    print("Seeds:", args.seeds)
    print("Skip training:", args.skip_training)
    print("Expected component output:", out)
    print("Place the precomputed conformation-predictor CSV there before building the final ensemble.")


if __name__ == "__main__":
    main()
