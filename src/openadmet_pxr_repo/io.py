from __future__ import annotations

from pathlib import Path
import pandas as pd

SUBMISSION_COLS = ["SMILES", "Molecule Name", "pEC50"]


def load_submission(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = [c for c in SUBMISSION_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"{path}: missing columns {missing}")
    out = df[SUBMISSION_COLS].copy()
    out["SMILES"] = out["SMILES"].astype(str)
    out["Molecule Name"] = out["Molecule Name"].astype(str)
    out["pEC50"] = pd.to_numeric(out["pEC50"], errors="coerce")
    return out


def find_phase1_file(root: Path, filename: str = "phase1_unblinded.csv") -> Path:
    hits = sorted(Path(root).rglob(filename))
    if not hits:
        raise FileNotFoundError(f"Could not find {filename} under {root}")
    return hits[0]


def discover_submission_csvs(root: Path) -> list[Path]:
    root = Path(root)
    paths = []
    for path in sorted(root.rglob("*.csv")):
        try:
            df = pd.read_csv(path, nrows=5)
            if not set(SUBMISSION_COLS).issubset(df.columns):
                continue
            full = pd.read_csv(path, usecols=SUBMISSION_COLS)
            if len(full) != 513:
                continue
            paths.append(path)
        except Exception:
            continue
    return paths


def align_submission_to_phase1(phase1: pd.DataFrame, pred: pd.DataFrame):
    merged = phase1[["Molecule Name", "pEC50"]].merge(
        pred[["Molecule Name", "pEC50"]],
        on="Molecule Name",
        how="left",
        suffixes=("_true", "_pred"),
    )
    if merged["pEC50_pred"].isna().any():
        return None, None
    y_true = merged["pEC50_true"].to_numpy(float)
    y_pred = merged["pEC50_pred"].to_numpy(float)
    return y_true, y_pred

