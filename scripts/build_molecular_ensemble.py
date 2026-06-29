from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from openadmet_pxr_repo.metrics import mae, rae_from_mae


SUBMISSIONS = ROOT / "submissions"
DATA = ROOT / "data"
PRED_MIN = 1.0
PRED_MAX = 8.5


def _weight_label(weight: float) -> str:
    return f"{weight:.3f}".replace(".", "p")


def _load_submission(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    expected = {"SMILES", "Molecule Name", "pEC50"}
    missing = expected - set(df.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    return df[["SMILES", "Molecule Name", "pEC50"]].copy()


def _average_graph_seeds(seeds: list[int]) -> pd.DataFrame:
    seed_frames = []
    for seed in seeds:
        path = SUBMISSIONS / f"graph_multitask_seed{seed}_predictions.csv"
        if not path.exists():
            raise FileNotFoundError(
                "Missing "
                f"{path}. Add the per-seed graph predictor CSV or run "
                "train_graph_multitask_predictor.py."
            )
        seed_frames.append(_load_submission(path))

    base = seed_frames[0][["SMILES", "Molecule Name"]].copy()
    base["pEC50"] = np.mean([df["pEC50"].to_numpy(float) for df in seed_frames], axis=0)
    out = SUBMISSIONS / "graph_multitask_ensemble_predictions.csv"
    base.to_csv(out, index=False)
    print(f"Saved graph predictor seed average: {out}")
    return base


def _score_phase1(sub: pd.DataFrame) -> None:
    phase1_path = DATA / "phase1_unblinded.csv"
    if not phase1_path.exists():
        print("No data/phase1_unblinded.csv found; skipping local evaluation.")
        return

    phase1 = pd.read_csv(phase1_path)[["Molecule Name", "pEC50"]].rename(columns={"pEC50": "true"})
    merged = phase1.merge(sub[["Molecule Name", "pEC50"]], on="Molecule Name", how="inner")
    if merged.empty:
        print("No overlap with phase1_unblinded.csv; skipping local evaluation.")
        return

    m = mae(merged["true"], merged["pEC50"])
    print(f"Phase1 MAE: {m:.6f} (n={len(merged)})")
    print(f"Phase1 RAE, fixed leaderboard-style denom: {rae_from_mae(m):.6f}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build molecular ensemble submission CSV."
    )
    parser.add_argument(
        "--w-graph",
        "--w-cm",
        dest="w_graph",
        type=float,
        default=0.325,
        help="Graph predictor blend weight.",
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--evaluate", action="store_true")
    parser.add_argument("--clip-min", type=float, default=PRED_MIN)
    parser.add_argument("--clip-max", type=float, default=PRED_MAX)
    args = parser.parse_args()

    SUBMISSIONS.mkdir(parents=True, exist_ok=True)
    conformation_path = SUBMISSIONS / "conformation_predictor_predictions.csv"
    conformation = _load_submission(conformation_path).set_index("Molecule Name")
    graph = _average_graph_seeds(args.seeds).set_index("Molecule Name")

    common = conformation.index.intersection(graph.index)
    if len(common) != len(conformation):
        raise ValueError(
            "Conformation and graph predictor molecule sets differ: "
            f"common={len(common)}, conformation={len(conformation)}"
        )

    w = float(args.w_graph)
    pred = w * graph.loc[conformation.index, "pEC50"].to_numpy(float)
    pred += (1.0 - w) * conformation["pEC50"].to_numpy(float)
    pred = np.clip(pred, args.clip_min, args.clip_max)

    out = conformation[["SMILES"]].reset_index()[["SMILES", "Molecule Name"]]
    out["pEC50"] = pred
    if abs(w - 0.325) < 1e-12:
        out_path = SUBMISSIONS / "activity_predictions_clean_baseline.csv"
    else:
        out_path = SUBMISSIONS / f"molecular_ensemble_weight_{_weight_label(w)}_predictions.csv"
    out.to_csv(out_path, index=False)
    print(f"Saved ensemble: {out_path}")

    if args.evaluate:
        _score_phase1(out)


if __name__ == "__main__":
    main()
