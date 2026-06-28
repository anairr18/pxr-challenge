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


def _load_submission(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    expected = {"SMILES", "Molecule Name", "pEC50"}
    missing = expected - set(df.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    return df[["SMILES", "Molecule Name", "pEC50"]].copy()


def _average_chemeleon_seeds(seeds: list[int]) -> pd.DataFrame:
    seed_frames = []
    for seed in seeds:
        path = SUBMISSIONS / f"chemeleon_mt_lr3e-04_s{seed}.csv"
        if not path.exists():
            raise FileNotFoundError(
                f"Missing {path}. Add the per-seed CheMeleon CSV or run train_chemeleon.py."
            )
        seed_frames.append(_load_submission(path))

    base = seed_frames[0][["SMILES", "Molecule Name"]].copy()
    base["pEC50"] = np.mean([df["pEC50"].to_numpy(float) for df in seed_frames], axis=0)
    out = SUBMISSIONS / f"chemeleon_mt_lr3e-04_{len(seeds)}seed.csv"
    base.to_csv(out, index=False)
    print(f"Saved CheMeleon seed average: {out}")
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
        description="Build Suiren/CheMeleon ensemble submission CSV."
    )
    parser.add_argument("--w-cm", type=float, default=0.325, help="CheMeleon blend weight.")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--evaluate", action="store_true")
    parser.add_argument("--clip-min", type=float, default=PRED_MIN)
    parser.add_argument("--clip-max", type=float, default=PRED_MAX)
    args = parser.parse_args()

    SUBMISSIONS.mkdir(parents=True, exist_ok=True)
    suiren_path = SUBMISSIONS / "iw2_3seed_ep17-23.csv"
    suiren = _load_submission(suiren_path).set_index("Molecule Name")
    chemeleon = _average_chemeleon_seeds(args.seeds).set_index("Molecule Name")

    common = suiren.index.intersection(chemeleon.index)
    if len(common) != len(suiren):
        raise ValueError(
            f"Suiren and CheMeleon molecule sets differ: common={len(common)}, suiren={len(suiren)}"
        )

    w = float(args.w_cm)
    pred = w * chemeleon.loc[suiren.index, "pEC50"].to_numpy(float)
    pred += (1.0 - w) * suiren["pEC50"].to_numpy(float)
    pred = np.clip(pred, args.clip_min, args.clip_max)

    out = suiren[["SMILES"]].reset_index()[["SMILES", "Molecule Name"]]
    out["pEC50"] = pred
    out_path = SUBMISSIONS / f"ens_cm_lr3e-04_3seed_sur_w{w:.3f}.csv"
    out.to_csv(out_path, index=False)
    print(f"Saved ensemble: {out_path}")

    if args.evaluate:
        _score_phase1(out)


if __name__ == "__main__":
    main()
