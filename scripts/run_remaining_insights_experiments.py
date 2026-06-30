from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import traceback


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from openadmet_pxr_repo.curve_multitask_experiment import run_curve_multitask_experiment
from openadmet_pxr_repo.domain_gate_experiment import run_domain_gate_experiment
from openadmet_pxr_repo.sub040_signal_experiment import FINAL_UPLOAD_FILE, run_structure_assay_experiment


def _run_named(name: str, fn, *args, **kwargs) -> dict:
    print(f"\n=== {name} ===", flush=True)
    try:
        summary = fn(*args, **kwargs)
        print(json.dumps(summary, indent=2), flush=True)
        return {"name": name, "ok": True, "summary": summary}
    except Exception as exc:
        traceback.print_exc()
        return {"name": name, "ok": False, "error": repr(exc)}


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run the remaining high-value signal experiments from the audit: "
            "orthogonal structure/assay features, curve-aware multitask heads, "
            "and applicability-domain/uncertainty gating."
        )
    )
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--n-boot", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260630)
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Fast smoke mode: skip 3D conformer descriptors and use fewer bootstraps.",
    )
    args = parser.parse_args()

    root = args.root
    n_boot = 200 if args.smoke else args.n_boot
    results = []
    results.append(
        _run_named(
            "orthogonal_structure_assay",
            run_structure_assay_experiment,
            root,
            n_folds=args.folds,
            n_boot=n_boot,
            seed=args.seed,
            with_3d=not args.smoke,
            use_single_concentration=True,
            use_mmps=True,
            use_weighted_aux=True,
            output_dir=root / "reports" / "remaining_insights_orthogonal_signal",
            oof_candidate_file="remaining_insights_orthogonal_oof_candidate.csv",
            upload_candidate_file="remaining_insights_orthogonal_upload_candidate.csv",
        )
    )
    results.append(
        _run_named(
            "curve_aware_multitask",
            run_curve_multitask_experiment,
            root,
            n_folds=args.folds,
            n_boot=n_boot,
            seed=args.seed + 1,
            output_dir=root / "reports" / "remaining_insights_curve_multitask",
        )
    )
    results.append(
        _run_named(
            "domain_uncertainty_gate",
            run_domain_gate_experiment,
            root,
            n_folds=args.folds,
            n_boot=n_boot,
            seed=args.seed + 2,
            output_dir=root / "reports" / "remaining_insights_domain_gate",
        )
    )

    candidates = []
    for result in results:
        summary = result.get("summary") if result.get("ok") else None
        if not summary:
            continue
        candidates.append(
            {
                "name": result["name"],
                "decision": summary.get("decision"),
                "mae": summary.get("candidate", {}).get("mae"),
                "rae": summary.get("candidate", {}).get("rae"),
                "folds_improved": summary.get("folds_improved"),
                "upload_candidate_file": summary.get("upload_candidate_file"),
            }
        )
    candidates = sorted(candidates, key=lambda row: (row["mae"] is None, row["mae"] or 99.0))
    passing = [
        row
        for row in candidates
        if row["decision"] in {"REAL_IMPROVEMENT_REVIEW_BEFORE_REPLACEMENT", "SUB040_CREDIBLE_REVIEW_FOR_REPLACEMENT"}
    ]
    combined = {
        "smoke": bool(args.smoke),
        "n_boot": int(n_boot),
        "preserved_final_upload_file": f"submissions/{FINAL_UPLOAD_FILE}",
        "results": results,
        "ranked_candidates": candidates,
        "passing_replacement_candidates": passing,
        "recommendation": (
            passing[0]["upload_candidate_file"]
            if passing
            else f"submissions/{FINAL_UPLOAD_FILE}"
        ),
    }
    out_dir = root / "reports" / "remaining_insights_summary"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "experiment_summary.json").write_text(json.dumps(combined, indent=2), encoding="utf-8")
    print("\n=== Remaining Insights Summary ===", flush=True)
    print(json.dumps(combined, indent=2), flush=True)


if __name__ == "__main__":
    main()
