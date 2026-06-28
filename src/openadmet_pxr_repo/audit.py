from __future__ import annotations

from pathlib import Path
import pandas as pd

from .io import discover_submission_csvs, find_phase1_file, load_submission, align_submission_to_phase1
from .metrics import (
    LEADERBOARD_RAE_DENOM,
    bootstrap_paired_ci,
    ci_summary,
    exact_fill_count,
    mae,
    phase1_denominator,
    rae_from_mae,
)


def _row_from_candidate(path: Path, y_true, y_pred, contaminated: bool, n_boot: int, seed: int):
    exact = exact_fill_count(y_true, y_pred)
    exact_frac = exact / len(y_true)
    if contaminated:
        return {
            "file": path.name,
            "status": "contaminated_exact_fill",
            "exact_matches": exact,
            "exact_frac": exact_frac,
            "mae": float("nan"),
            "rae_fixed": float("nan"),
            "rae_phase1": float("nan"),
            "mae_ci": "",
            "rae_fixed_ci": "",
            "rae_phase1_ci": "",
        }

    m = mae(y_true, y_pred)
    ci = bootstrap_paired_ci(y_true, y_pred, n_boot=n_boot, seed=seed)
    mae_ci = ci_summary(ci["mae"])
    rae_fixed_ci = ci_summary(ci["rae_fixed"])
    rae_phase1_ci = ci_summary(ci["rae_resampled"])

    return {
        "file": path.name,
        "status": "honest",
        "exact_matches": exact,
        "exact_frac": exact_frac,
        "mae": m,
        "rae_fixed": rae_from_mae(m, LEADERBOARD_RAE_DENOM),
        "rae_phase1": rae_from_mae(m, phase1_denominator(y_true)),
        "mae_ci": f"{mae_ci['mid']:.6f}  [{mae_ci['lo']:.6f}, {mae_ci['hi']:.6f}]",
        "rae_fixed_ci": f"{rae_fixed_ci['mid']:.6f}  [{rae_fixed_ci['lo']:.6f}, {rae_fixed_ci['hi']:.6f}]",
        "rae_phase1_ci": f"{rae_phase1_ci['mid']:.6f}  [{rae_phase1_ci['lo']:.6f}, {rae_phase1_ci['hi']:.6f}]",
    }


def build_report_frame(root: Path, n_boot: int = 5000, seed: int = 20260623, exact_fill_threshold: int = 20):
    root = Path(root)
    phase1_path = find_phase1_file(root)
    phase1 = load_submission(phase1_path).dropna(subset=["pEC50"]).copy()
    if len(phase1) != 253:
        raise ValueError(f"Expected 253 Phase1 rows, found {len(phase1)}")

    rows = []
    contaminated = []
    for path in discover_submission_csvs(root):
        if path.name == phase1_path.name:
            continue
        pred = load_submission(path)
        y_true, y_pred = align_submission_to_phase1(phase1, pred)
        if y_true is None:
            rows.append(
                {
                    "file": path.name,
                    "status": "missing_phase1_rows",
                    "exact_matches": float("nan"),
                    "exact_frac": float("nan"),
                    "mae": float("nan"),
                    "rae_fixed": float("nan"),
                    "rae_phase1": float("nan"),
                    "mae_ci": "",
                    "rae_fixed_ci": "",
                    "rae_phase1_ci": "",
                }
            )
            continue
        exact = exact_fill_count(y_true, y_pred)
        contaminated_flag = exact >= exact_fill_threshold or (exact / len(y_true)) >= 0.20
        row = _row_from_candidate(path, y_true, y_pred, contaminated_flag, n_boot=n_boot, seed=seed)
        rows.append(row)
        if contaminated_flag:
            contaminated.append(path.name)

    report = pd.DataFrame(rows)
    honest = report[report["status"].eq("honest")].dropna(subset=["mae"]).sort_values("mae")
    return {
        "phase1_path": phase1_path,
        "phase1": phase1,
        "report": report,
        "honest": honest,
        "contaminated": contaminated,
    }


def save_report(report: pd.DataFrame, path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(path, index=False)
    return path

