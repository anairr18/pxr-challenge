from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from openadmet_pxr_repo.asymmetric_graph_experiment import run_asymmetric_graph_experiment


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run a counter-assay-weighted, asymmetric graph neural experiment "
            "against the locked OpenADMET PXR activity anchor."
        )
    )
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--n-boot", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260630)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=192)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--graph-weight", type=float, default=0.12)
    parser.add_argument("--phase-weight", type=float, default=0.75)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--clf-weight", type=float, default=0.20)
    parser.add_argument("--active-cutoff", type=float, default=4.0)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--smoke", action="store_true", help="Short dependency/runtime check.")
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    summary = run_asymmetric_graph_experiment(
        args.root,
        n_folds=args.folds,
        n_boot=args.n_boot,
        seed=args.seed,
        epochs=args.epochs,
        batch_size=args.batch_size,
        hidden_dim=args.hidden_dim,
        depth=args.depth,
        graph_weight=args.graph_weight,
        phase_weight=args.phase_weight,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        patience=args.patience,
        clf_weight=args.clf_weight,
        active_cutoff=args.active_cutoff,
        device=args.device,
        smoke=args.smoke,
        output_dir=args.output_dir,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

