from __future__ import annotations

from pathlib import Path
import shutil
import pandas as pd


def best_honest_candidate(report: pd.DataFrame) -> pd.Series:
    honest = report[report["status"].eq("honest")].dropna(subset=["mae"])
    if honest.empty:
        raise ValueError("No honest candidates found.")
    return honest.sort_values("mae", ascending=True).iloc[0]


def copy_best_candidate(root: Path, report: pd.DataFrame, out_path: Path | None = None) -> Path:
    root = Path(root)
    best = best_honest_candidate(report)
    matches = [p for p in root.rglob(str(best["file"])) if p.is_file()]
    if not matches:
        raise FileNotFoundError(f"Could not find {best['file']} under {root}")
    matches.sort(key=lambda p: (0 if "submissions" in p.parts else 1, len(str(p))))
    src = matches[0]

    if out_path is None:
        out_path = root / "artifacts" / "best_honest_submission.csv"
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, out_path)
    return out_path

