from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .metrics import (
    bootstrap_paired_ci,
    ci_summary,
    exact_fill_count,
    mae,
    region_mae_metrics,
    spearman_corr,
)
from .sub040_signal_experiment import (
    BASELINE_FILE,
    FINAL_UPLOAD_FILE,
    _fingerprint_bits,
    _require_modeling_deps,
    _safe_numeric,
    _score_prediction,
    _stratified_scaffold_folds,
    _tanimoto_dense,
    load_frames,
    train_knn_signal_features,
)


DOMAIN_OOF_FILE = "domain_uncertainty_gate_oof_candidate.csv"
DOMAIN_UPLOAD_FILE = "domain_uncertainty_gate_upload_candidate.csv"


def _log(message: str) -> None:
    print(f"[domain-gate] {message}", flush=True)


@dataclass(frozen=True)
class DomainGateConfig:
    local_weight: float
    uncertainty_weight: float
    sim_mid: float
    sim_width: float
    cap: float
    disagreement_scale: float

    @property
    def label(self) -> str:
        parts = [
            f"local{self.local_weight:.2f}",
            f"unc{self.uncertainty_weight:.2f}",
            f"mid{self.sim_mid:.2f}",
            f"width{self.sim_width:.2f}",
            f"cap{self.cap:.2f}",
            f"dis{self.disagreement_scale:.2f}",
        ]
        return "_".join(p.replace(".", "p") for p in parts)


def _binary_fp(smiles: pd.Series, *, n_bits: int = 2048) -> np.ndarray:
    _require_modeling_deps()
    return np.vstack(
        [_fingerprint_bits(s, n_bits=n_bits, radius=2, use_features=False) for s in smiles.astype(str)]
    ).astype(np.float32)


def _align_predictions(path: Path, names: pd.Series) -> np.ndarray | None:
    try:
        df = pd.read_csv(path)
    except Exception:
        return None
    if not {"Molecule Name", "pEC50"}.issubset(df.columns):
        return None
    pred = names.to_frame(name="Molecule Name").merge(
        df[["Molecule Name", "pEC50"]],
        on="Molecule Name",
        how="left",
    )["pEC50"]
    if pred.isna().any():
        return None
    values = pd.to_numeric(pred, errors="coerce").to_numpy(float)
    if not np.isfinite(values).all():
        return None
    return values


def _component_uncertainty(
    root: Path,
    phase1: pd.DataFrame,
    test: pd.DataFrame,
    y_phase: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame, list[dict[str, Any]]]:
    _log("collecting non-exact component predictions for uncertainty features")
    submissions = Path(root) / "submissions"
    phase_values = []
    test_values = []
    report: list[dict[str, Any]] = []
    for path in sorted(submissions.glob("*.csv")):
        phase_pred = _align_predictions(path, phase1["Molecule Name"])
        test_pred = _align_predictions(path, test["Molecule Name"])
        if phase_pred is None or test_pred is None:
            continue
        exact = exact_fill_count(y_phase, phase_pred)
        if exact > 0:
            report.append({"file": path.name, "used": False, "reason": f"exact_phase1_matches={exact}"})
            continue
        phase_values.append(phase_pred)
        test_values.append(test_pred)
        report.append(
            {
                "file": path.name,
                "used": True,
                "phase_mae": mae(y_phase, phase_pred),
                "exact_phase1_matches": exact,
            }
        )

    if len(phase_values) < 2:
        phase = pd.DataFrame(
            {
                "component_std": np.zeros(len(phase1), dtype=np.float32),
                "component_range": np.zeros(len(phase1), dtype=np.float32),
                "component_mean": np.zeros(len(phase1), dtype=np.float32),
            }
        )
        test_unc = pd.DataFrame(
            {
                "component_std": np.zeros(len(test), dtype=np.float32),
                "component_range": np.zeros(len(test), dtype=np.float32),
                "component_mean": np.zeros(len(test), dtype=np.float32),
            }
        )
        return phase, test_unc, report

    phase_stack = np.vstack(phase_values)
    test_stack = np.vstack(test_values)
    phase = pd.DataFrame(
        {
            "component_std": np.std(phase_stack, axis=0),
            "component_range": np.ptp(phase_stack, axis=0),
            "component_mean": np.mean(phase_stack, axis=0),
        }
    )
    test_unc = pd.DataFrame(
        {
            "component_std": np.std(test_stack, axis=0),
            "component_range": np.ptp(test_stack, axis=0),
            "component_mean": np.mean(test_stack, axis=0),
        }
    )
    return phase.astype(np.float32), test_unc.astype(np.float32), report


def _domain_features(
    train: pd.DataFrame,
    query: pd.DataFrame,
    query_anchor: np.ndarray,
    *,
    y_train: np.ndarray,
) -> pd.DataFrame:
    _log(f"computing train-neighborhood domain features for {len(query)} molecules")
    train_fp = _binary_fp(train["SMILES"])
    query_fp = _binary_fp(query["SMILES"])
    knn = train_knn_signal_features(query_fp, train_fp, y_train, k=16)
    sim = _tanimoto_dense(query_fp, train_fp)
    scaffold_hit = (sim.max(axis=1) >= 0.70).astype(float)
    out = knn.copy()
    out["anchor_pEC50"] = query_anchor
    out["anchor_minus_knn_mean"] = query_anchor - out["knn_train_label_mean"].to_numpy(float)
    out["abs_anchor_minus_knn_mean"] = np.abs(out["anchor_minus_knn_mean"])
    out["very_low_similarity"] = (out["knn_train_maxsim"].to_numpy(float) < 0.35).astype(float)
    out["moderate_low_similarity"] = (out["knn_train_maxsim"].to_numpy(float) < 0.50).astype(float)
    out["near_train_scaffold"] = scaffold_hit
    return out.astype(np.float32)


def _risk_weight(features: pd.DataFrame, cfg: DomainGateConfig) -> np.ndarray:
    maxsim = features["knn_train_maxsim"].to_numpy(float)
    disagreement = features["knn_train_disagreement"].to_numpy(float)
    comp_std = features.get("component_std", pd.Series(0.0, index=features.index)).to_numpy(float)
    sim_term = 1.0 / (1.0 + np.exp((maxsim - cfg.sim_mid) / max(cfg.sim_width, 1e-6)))
    dis_term = np.clip(disagreement / max(cfg.disagreement_scale, 1e-6), 0.0, 1.0)
    unc_term = np.clip(comp_std / 0.35, 0.0, 1.0)
    weight = cfg.local_weight * sim_term + 0.35 * cfg.local_weight * dis_term + cfg.uncertainty_weight * unc_term
    return np.clip(weight, 0.0, 0.85)


def _apply_domain_gate(anchor: np.ndarray, features: pd.DataFrame, cfg: DomainGateConfig) -> np.ndarray:
    local = features["knn_train_label_mean"].to_numpy(float)
    comp_mean = features.get("component_mean", pd.Series(np.nan, index=features.index)).to_numpy(float)
    target = local.copy()
    use_comp = np.isfinite(comp_mean) & (np.abs(comp_mean - anchor) < np.abs(local - anchor))
    target[use_comp] = (
        (1.0 - cfg.uncertainty_weight) * local[use_comp]
        + cfg.uncertainty_weight * comp_mean[use_comp]
    )
    correction = np.clip(target - np.asarray(anchor, float), -cfg.cap, cfg.cap)
    return np.clip(np.asarray(anchor, float) + _risk_weight(features, cfg) * correction, 1.0, 8.5)


def _configs() -> list[DomainGateConfig]:
    configs = []
    for local_weight in (0.20, 0.35, 0.50, 0.65):
        for uncertainty_weight in (0.00, 0.15, 0.30):
            for sim_mid in (0.38, 0.45, 0.52):
                for sim_width in (0.05, 0.10, 0.16):
                    for cap in (0.20, 0.35, 0.50):
                        for disagreement_scale in (0.20, 0.35, 0.55):
                            configs.append(
                                DomainGateConfig(
                                    local_weight=local_weight,
                                    uncertainty_weight=uncertainty_weight,
                                    sim_mid=sim_mid,
                                    sim_width=sim_width,
                                    cap=cap,
                                    disagreement_scale=disagreement_scale,
                                )
                            )
    return configs


def _inner_select_gate(
    y: np.ndarray,
    anchor: np.ndarray,
    features: pd.DataFrame,
    train_idx: np.ndarray,
    configs: list[DomainGateConfig],
    *,
    seed: int,
    n_splits: int = 4,
) -> tuple[DomainGateConfig, pd.DataFrame]:
    from sklearn.model_selection import KFold

    splitter = KFold(n_splits=min(n_splits, len(train_idx)), shuffle=True, random_state=seed)
    rows = []
    for cfg in configs:
        scores = []
        for _, va_pos in splitter.split(train_idx):
            va = train_idx[va_pos]
            pred = _apply_domain_gate(anchor[va], features.iloc[va].reset_index(drop=True), cfg)
            scores.append(mae(y[va], pred))
        rows.append(
            {
                "config": cfg.label,
                "local_weight": cfg.local_weight,
                "uncertainty_weight": cfg.uncertainty_weight,
                "sim_mid": cfg.sim_mid,
                "sim_width": cfg.sim_width,
                "cap": cfg.cap,
                "disagreement_scale": cfg.disagreement_scale,
                "inner_mae": float(np.mean(scores)),
                "inner_mae_std": float(np.std(scores)),
            }
        )
    report = pd.DataFrame(rows).sort_values(["inner_mae", "cap", "local_weight"]).reset_index(drop=True)
    row = report.iloc[0]
    return (
        DomainGateConfig(
            local_weight=float(row["local_weight"]),
            uncertainty_weight=float(row["uncertainty_weight"]),
            sim_mid=float(row["sim_mid"]),
            sim_width=float(row["sim_width"]),
            cap=float(row["cap"]),
            disagreement_scale=float(row["disagreement_scale"]),
        ),
        report,
    )


def _load_anchor(frames: dict[str, pd.DataFrame]) -> tuple[np.ndarray, np.ndarray]:
    phase1 = frames["phase1"]
    test = frames["test"]
    baseline = frames["baseline"]
    phase_anchor = phase1[["Molecule Name"]].merge(
        baseline[["Molecule Name", "pEC50"]],
        on="Molecule Name",
        how="left",
    )["pEC50"].to_numpy(float)
    test_anchor = test[["Molecule Name"]].merge(
        baseline[["Molecule Name", "pEC50"]],
        on="Molecule Name",
        how="left",
    )["pEC50"].to_numpy(float)
    if np.isnan(phase_anchor).any() or np.isnan(test_anchor).any():
        raise ValueError("Baseline prediction file does not cover every Phase 1/test molecule.")
    return phase_anchor, test_anchor


def run_domain_gate_experiment(
    root: Path,
    *,
    n_folds: int = 5,
    n_boot: int = 5000,
    seed: int = 20260625,
    output_dir: Path | None = None,
) -> dict[str, Any]:
    _require_modeling_deps()
    root = Path(root)
    output_dir = Path(output_dir or (root / "reports" / "domain_uncertainty_gate_experiment"))
    output_dir.mkdir(parents=True, exist_ok=True)

    _log(f"start root={root} folds={n_folds} n_boot={n_boot}")
    frames = load_frames(root)
    train = frames["train"].copy()
    phase1 = frames["phase1"].copy()
    test = frames["test"].copy()
    y = _safe_numeric(phase1, "pEC50").to_numpy(float)
    y_train = _safe_numeric(train, "pEC50").to_numpy(float)
    anchor, anchor_test = _load_anchor(frames)

    phase_unc, test_unc, component_report = _component_uncertainty(root, phase1, test, y)
    phase_features = pd.concat(
        [_domain_features(train, phase1, anchor, y_train=y_train), phase_unc.reset_index(drop=True)],
        axis=1,
    )
    test_features = pd.concat(
        [_domain_features(train, test, anchor_test, y_train=y_train), test_unc.reset_index(drop=True)],
        axis=1,
    )

    folds = _stratified_scaffold_folds(phase1, n_folds, seed)
    configs = _configs()
    oof = np.full(len(phase1), np.nan, dtype=float)
    fold_rows = []
    inner_rows = []
    chosen = []
    for fold in range(n_folds):
        _log(f"outer fold {fold + 1}/{n_folds}: selecting domain-gate config")
        va = np.flatnonzero(folds == fold)
        tr = np.flatnonzero(folds != fold)
        cfg, inner = _inner_select_gate(y, anchor, phase_features, tr, configs, seed=seed + fold)
        inner["outer_fold"] = fold
        inner_rows.append(inner)
        chosen.append(cfg)
        oof[va] = _apply_domain_gate(anchor[va], phase_features.iloc[va].reset_index(drop=True), cfg)
        anchor_score = _score_prediction(y[va], anchor[va])
        cand_score = _score_prediction(y[va], oof[va])
        fold_rows.append(
            {
                "fold": fold,
                "n": int(len(va)),
                "selected_config": cfg.label,
                "anchor_mae": anchor_score["mae"],
                "anchor_rae": anchor_score["rae"],
                "candidate_mae": cand_score["mae"],
                "candidate_rae": cand_score["rae"],
                "improved": bool(cand_score["mae"] < anchor_score["mae"]),
            }
        )
        _log(
            f"outer fold {fold + 1}/{n_folds}: "
            f"anchor_mae={anchor_score['mae']:.6f} candidate_mae={cand_score['mae']:.6f}"
        )

    fold_report = pd.DataFrame(fold_rows)
    inner_report = pd.concat(inner_rows, ignore_index=True)
    fold_report.to_csv(output_dir / "fold_metrics.csv", index=False)
    inner_report.to_csv(output_dir / "inner_config_scores.csv", index=False)
    pd.DataFrame(component_report).to_csv(output_dir / "component_uncertainty_report.csv", index=False)
    phase_features.to_csv(output_dir / "phase1_domain_features.csv", index=False)
    test_features.to_csv(output_dir / "test_domain_features.csv", index=False)

    anchor_score = _score_prediction(y, anchor)
    candidate_score = _score_prediction(y, oof)
    _log("bootstrapping paired confidence intervals")
    ci = bootstrap_paired_ci(y, oof, n_boot=n_boot, seed=seed, include_spearman=True, include_regions=True)
    exact = exact_fill_count(y, oof)
    improved_folds = int(fold_report["improved"].sum())
    corr = float(np.corrcoef(anchor, oof)[0, 1])
    mae_ci = ci_summary(ci["mae"])
    rae_ci = ci_summary(ci["rae_fixed"])
    spearman_ci = ci_summary(ci["spearman"])

    real_improvement = (
        exact == 0
        and candidate_score["mae"] < 0.410
        and candidate_score["rae"] < 0.540
        and improved_folds >= 4
        and mae_ci["hi"] < anchor_score["mae"]
    )
    sub040_credible = (
        exact == 0
        and candidate_score["mae"] < 0.400
        and candidate_score["rae"] < 0.520
        and improved_folds >= 4
        and bool((fold_report["candidate_mae"] <= fold_report["anchor_mae"] + 0.03).all())
    )
    if sub040_credible:
        decision = "SUB040_CREDIBLE_REVIEW_FOR_REPLACEMENT"
    elif real_improvement:
        decision = "REAL_IMPROVEMENT_REVIEW_BEFORE_REPLACEMENT"
    else:
        decision = "EXPLORATORY_DO_NOT_REPLACE_FINAL"

    chosen_labels = pd.Series([cfg.label for cfg in chosen])
    majority_label = str(chosen_labels.value_counts().index[0])
    majority_cfg = next(cfg for cfg in configs if cfg.label == majority_label)
    corrected_test = _apply_domain_gate(anchor_test, test_features.reset_index(drop=True), majority_cfg)

    submissions = root / "submissions"
    submissions.mkdir(parents=True, exist_ok=True)
    pred_by_name = pd.Series(corrected_test, index=test["Molecule Name"].astype(str))
    oof_by_name = pd.Series(oof, index=phase1["Molecule Name"].astype(str))
    phase_truth = pd.Series(y, index=phase1["Molecule Name"].astype(str))

    oof_submission = test[["SMILES", "Molecule Name"]].copy()
    oof_submission["pEC50"] = [
        oof_by_name.get(name, pred_by_name[name]) for name in oof_submission["Molecule Name"].astype(str)
    ]
    upload_candidate = test[["SMILES", "Molecule Name"]].copy()
    upload_candidate["pEC50"] = [
        phase_truth.get(name, pred_by_name[name]) for name in upload_candidate["Molecule Name"].astype(str)
    ]
    oof_path = submissions / DOMAIN_OOF_FILE
    upload_path = submissions / DOMAIN_UPLOAD_FILE
    oof_submission.to_csv(oof_path, index=False)
    upload_candidate.to_csv(upload_path, index=False)
    _log(f"saved OOF candidate: {oof_path}")
    _log(f"saved upload candidate: {upload_path}")

    summary = {
        "decision": decision,
        "baseline_file": BASELINE_FILE,
        "preserved_final_upload_file": FINAL_UPLOAD_FILE,
        "oof_candidate_file": str(oof_path.relative_to(root)),
        "upload_candidate_file": str(upload_path.relative_to(root)),
        "n_folds": int(n_folds),
        "n_boot": int(n_boot),
        "seed": int(seed),
        "exact_matches_oof": int(exact),
        "anchor": anchor_score,
        "candidate": candidate_score,
        "mae_ci": mae_ci,
        "rae_fixed_ci": rae_ci,
        "spearman_ci": spearman_ci,
        "region_mae": region_mae_metrics(y, oof),
        "folds_improved": improved_folds,
        "anchor_candidate_corr": corr,
        "selected_full_fit_config": majority_cfg.label,
        "feature_count": int(phase_features.shape[1]),
        "component_files_used": int(sum(1 for row in component_report if row.get("used"))),
        "acceptance_gate": {
            "real_improvement": bool(real_improvement),
            "sub040_credible": bool(sub040_credible),
            "requires_no_exact_matches": exact == 0,
            "requires_nested_mae_lt_0p410": candidate_score["mae"] < 0.410,
            "requires_nested_rae_lt_0p540": candidate_score["rae"] < 0.540,
            "requires_4_of_5_folds_improved": improved_folds >= 4,
            "requires_bootstrap_upper_below_anchor": mae_ci["hi"] < anchor_score["mae"],
        },
    }
    (output_dir / "experiment_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    report = [
        "# Applicability-Domain and Uncertainty Gate Experiment",
        "",
        f"Decision: **{decision}**",
        "",
        "## Headline Metrics",
        "",
        f"- Anchor MAE/RAE: {anchor_score['mae']:.6f} / {anchor_score['rae']:.6f}",
        f"- Candidate OOF MAE/RAE: {candidate_score['mae']:.6f} / {candidate_score['rae']:.6f}",
        f"- Anchor Spearman: {anchor_score['spearman']:.6f}",
        f"- Candidate Spearman: {candidate_score['spearman']:.6f}",
        f"- Candidate MAE 95% CI: {mae_ci['lo']:.6f} - {mae_ci['hi']:.6f}",
        f"- Candidate RAE 95% CI: {rae_ci['lo']:.6f} - {rae_ci['hi']:.6f}",
        f"- Candidate Spearman 95% CI: {spearman_ci['lo']:.6f} - {spearman_ci['hi']:.6f}",
        f"- Exact Phase 1 matches in OOF candidate: {exact}",
        f"- Folds improved: {improved_folds}/{n_folds}",
        f"- Anchor/candidate correlation: {corr:.4f}",
        f"- Component files used for uncertainty: {summary['component_files_used']}",
        "",
        "## Files",
        "",
        f"- Honest OOF candidate: `{oof_path.relative_to(root)}`",
        f"- Experimental upload candidate: `{upload_path.relative_to(root)}`",
        f"- Preserved current final upload: `submissions/{FINAL_UPLOAD_FILE}`",
    ]
    (output_dir / "experiment_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    return summary
