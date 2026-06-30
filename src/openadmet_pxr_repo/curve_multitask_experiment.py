from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any
import warnings

import numpy as np
import pandas as pd

from .metrics import (
    bootstrap_paired_ci,
    ci_summary,
    exact_fill_count,
    mae,
    rae_from_mae,
    region_mae_metrics,
    spearman_corr,
)
from .sub040_signal_experiment import (
    BASELINE_FILE,
    FINAL_UPLOAD_FILE,
    _apply_residual,
    _clean_matrix,
    _fit_predict_model,
    _fingerprint_bits,
    _inner_select_config,
    _rdkit_descriptors,
    _require_modeling_deps,
    _residual_configs,
    _safe_numeric,
    _score_prediction,
    _stratified_scaffold_folds,
    _tanimoto_dense,
    build_molecular_features,
    load_frames,
    train_knn_signal_features,
)


CURVE_OOF_FILE = "curve_multitask_oof_candidate.csv"
CURVE_UPLOAD_FILE = "curve_multitask_upload_candidate.csv"


@dataclass(frozen=True)
class CurveHeadSpec:
    name: str
    source: str
    target: str
    se_col: str | None = None


def _feature_block_for_smiles(
    smiles: pd.Series,
    *,
    seed: int,
    n_bits: int = 1024,
) -> tuple[np.ndarray, np.ndarray]:
    desc, ecfp, fcfp = build_molecular_features(
        smiles,
        n_bits=n_bits,
        with_3d=False,
        seed=seed,
    )
    x = np.hstack([_clean_matrix(desc.to_numpy(np.float32)), ecfp, fcfp]).astype(np.float32)
    fp = np.hstack([ecfp, fcfp]).astype(np.float32)
    return x, fp


def _sanitize_feature_name(name: str) -> str:
    keep = []
    for char in name:
        if char.isalnum():
            keep.append(char)
        elif char in {"_", "-"}:
            keep.append("_")
        else:
            keep.append("_")
    return "".join(keep).strip("_").lower()


def _exclude_blinded_rows(
    frame: pd.DataFrame,
    phase1: pd.DataFrame,
    test: pd.DataFrame,
) -> pd.DataFrame:
    if frame.empty or "SMILES" not in frame.columns:
        return frame.iloc[0:0].copy()
    blocked_smiles = set(phase1["SMILES"].astype(str)) | set(test["SMILES"].astype(str))
    blocked_names = set(phase1["Molecule Name"].astype(str)) | set(test["Molecule Name"].astype(str))
    out = frame.copy()
    mask = ~out["SMILES"].astype(str).isin(blocked_smiles)
    if "Molecule Name" in out.columns:
        mask &= ~out["Molecule Name"].astype(str).isin(blocked_names)
    return out.loc[mask].copy()


def _source_frames(frames: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    train = frames["train"].copy()
    counter = frames.get("counter", pd.DataFrame()).copy()
    multitask = frames.get("multitask", pd.DataFrame()).copy()
    single = frames.get("single_concentration", pd.DataFrame()).copy()

    if not single.empty:
        single = single.copy()
        numeric_cols = [
            "log2_fc_estimate",
            "log2_fc_stderr",
            "t_statistic",
            "neg_log10_fdr",
            "median_log2_fc",
            "cohens_d",
            "concentration_M",
        ]
        for col in numeric_cols:
            if col in single.columns:
                single[col] = pd.to_numeric(single[col], errors="coerce")
        single["_abs_t"] = single["t_statistic"].abs() if "t_statistic" in single.columns else np.nan
        single["_abs_d"] = single["cohens_d"].abs() if "cohens_d" in single.columns else np.nan
        single = (
            single.groupby("SMILES", as_index=False)
            .agg(
                single_log2fc_mean=("log2_fc_estimate", "mean"),
                single_log2fc_max=("log2_fc_estimate", "max"),
                single_log2fc_min=("log2_fc_estimate", "min"),
                single_abs_t_max=("_abs_t", "max"),
                single_abs_d_max=("_abs_d", "max"),
                single_neglog10fdr_max=("neg_log10_fdr", "max"),
                single_median_log2fc_mean=("median_log2_fc", "mean"),
                single_stderr_mean=("log2_fc_stderr", "mean"),
                single_concentration_count=("concentration_M", "count"),
            )
            .replace([np.inf, -np.inf], np.nan)
        )

    return {
        "train": train,
        "counter": counter,
        "multitask": multitask,
        "single": single,
    }


def _curve_specs(source_frames: dict[str, pd.DataFrame]) -> list[CurveHeadSpec]:
    candidates = [
        CurveHeadSpec("main_pEC50", "train", "pEC50", "pEC50_std.error (-log10(molarity)"),
        CurveHeadSpec(
            "main_emax_log2fc",
            "train",
            "Emax_estimate (log2FC vs. baseline)",
            "Emax_std.error (log2FC vs. baseline)",
        ),
        CurveHeadSpec(
            "main_emax_posctrl",
            "train",
            "Emax.vs.pos.ctrl_estimate (dimensionless)",
            "Emax.vs.pos.ctrl_std.error (dimensionless)",
        ),
        CurveHeadSpec("counter_pEC50", "counter", "pEC50", "pEC50_std.error (-log10(molarity)"),
        CurveHeadSpec(
            "counter_emax_log2fc",
            "counter",
            "Emax_estimate (log2FC vs. baseline)",
            "Emax_std.error (log2FC vs. baseline)",
        ),
        CurveHeadSpec(
            "counter_emax_posctrl",
            "counter",
            "Emax.vs.pos.ctrl_estimate (dimensionless)",
            "Emax.vs.pos.ctrl_std.error (dimensionless)",
        ),
        CurveHeadSpec("mt_pEC50", "multitask", "pEC50", "pEC50_se"),
        CurveHeadSpec("mt_log2fc_8um", "multitask", "log2fc_8um", "log2fc_8um_se"),
        CurveHeadSpec("mt_log2fc_33um", "multitask", "log2fc_33um", "log2fc_33um_se"),
        CurveHeadSpec("single_log2fc_mean", "single", "single_log2fc_mean", "single_stderr_mean"),
        CurveHeadSpec("single_log2fc_max", "single", "single_log2fc_max", None),
        CurveHeadSpec("single_log2fc_min", "single", "single_log2fc_min", None),
        CurveHeadSpec("single_abs_t_max", "single", "single_abs_t_max", None),
        CurveHeadSpec("single_abs_d_max", "single", "single_abs_d_max", None),
        CurveHeadSpec("single_neglog10fdr_max", "single", "single_neglog10fdr_max", None),
    ]
    specs = []
    for spec in candidates:
        frame = source_frames.get(spec.source, pd.DataFrame())
        if not frame.empty and {"SMILES", spec.target}.issubset(frame.columns):
            specs.append(spec)
    return specs


def _fit_head_predictions(
    spec: CurveHeadSpec,
    frame: pd.DataFrame,
    all_x_by_smiles: dict[str, np.ndarray],
    query_x: np.ndarray,
    *,
    seed: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
    from sklearn.linear_model import HuberRegressor, Ridge
    from sklearn.model_selection import KFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    df = frame[["SMILES", spec.target] + ([spec.se_col] if spec.se_col and spec.se_col in frame.columns else [])].copy()
    df[spec.target] = pd.to_numeric(df[spec.target], errors="coerce")
    df = df.dropna(subset=["SMILES", spec.target]).copy()
    if df.empty:
        raise ValueError(f"No labeled rows for {spec.name}")
    df = (
        df.groupby("SMILES", as_index=False)
        .agg(
            target=(spec.target, "median"),
            se=(spec.se_col, "median") if spec.se_col and spec.se_col in df.columns else (spec.target, "size"),
        )
        .copy()
    )
    rows = []
    y_rows = []
    weights = []
    for _, row in df.iterrows():
        smi = str(row["SMILES"])
        if smi not in all_x_by_smiles:
            continue
        y = float(row["target"])
        if not np.isfinite(y):
            continue
        rows.append(all_x_by_smiles[smi])
        y_rows.append(y)
        se = float(row["se"]) if np.isfinite(row["se"]) else np.nan
        if spec.se_col:
            weights.append(float(np.clip(1.0 / max(se * se, 1e-4), 0.2, 200.0)) if np.isfinite(se) else 1.0)
        else:
            weights.append(1.0)
    if len(y_rows) < 100:
        raise ValueError(f"Too few labeled rows for {spec.name}: {len(y_rows)}")

    x = np.vstack(rows).astype(np.float32)
    y = np.asarray(y_rows, dtype=float)
    sample_weight = np.asarray(weights, dtype=float)
    models = [
        make_pipeline(StandardScaler(with_mean=False), Ridge(alpha=20.0)),
        make_pipeline(StandardScaler(with_mean=False), HuberRegressor(alpha=0.003, epsilon=1.4, max_iter=700)),
        ExtraTreesRegressor(
            n_estimators=450,
            min_samples_leaf=4,
            max_features=0.35,
            random_state=seed,
            n_jobs=-1,
        ),
        HistGradientBoostingRegressor(
            loss="absolute_error",
            learning_rate=0.03,
            max_iter=420,
            max_leaf_nodes=15,
            l2_regularization=0.15,
            min_samples_leaf=15,
            random_state=seed,
        ),
    ]

    oof = np.zeros((len(y), len(models)), dtype=np.float32)
    kf = KFold(n_splits=min(5, len(y)), shuffle=True, random_state=seed)
    for j, model in enumerate(models):
        for tr, va in kf.split(x):
            kwargs = {}
            if j in (2,):
                kwargs["sample_weight"] = sample_weight[tr]
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model.fit(x[tr], y[tr], **kwargs)
            oof[va, j] = np.asarray(model.predict(x[va]), dtype=np.float32)

    model_mae = np.mean(np.abs(oof - y[:, None]), axis=0)
    inv = 1.0 / np.maximum(model_mae, 1e-6)
    blend_w = inv / inv.sum()
    pred = np.zeros(query_x.shape[0], dtype=float)
    for j, model in enumerate(models):
        kwargs = {}
        if j in (2,):
            kwargs["sample_weight"] = sample_weight
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(x, y, **kwargs)
        pred += float(blend_w[j]) * np.asarray(model.predict(query_x), dtype=float)

    report = {
        "head": spec.name,
        "source": spec.source,
        "target": spec.target,
        "se_col": spec.se_col,
        "n_labeled": int(len(y)),
        "oof_mae": float(np.mean(np.abs(oof @ blend_w - y))),
        "model_mae": [float(v) for v in model_mae],
        "model_weights": [float(v) for v in blend_w],
        "target_mean": float(np.mean(y)),
        "target_std": float(np.std(y)),
    }
    return pred.astype(np.float32), report


def _nearest_curve_features(
    query_fp: np.ndarray,
    source_fp: np.ndarray,
    source_targets: pd.DataFrame,
    *,
    prefix: str,
    k: int = 12,
) -> pd.DataFrame:
    if source_targets.empty:
        return pd.DataFrame(index=np.arange(query_fp.shape[0]))
    sim = _tanimoto_dense(query_fp, source_fp)
    k_eff = min(k, sim.shape[1])
    idx = np.argpartition(-sim, kth=k_eff - 1, axis=1)[:, :k_eff]
    values = source_targets.to_numpy(float)
    rows = []
    for row_id in range(sim.shape[0]):
        sims = sim[row_id, idx[row_id]]
        neigh = values[idx[row_id]]
        weights = np.maximum(sims, 1e-6) ** 2
        row = {
            f"{prefix}_maxsim": float(np.max(sims)),
            f"{prefix}_meansim": float(np.mean(sims)),
        }
        for col_id, col in enumerate(source_targets.columns):
            vals = neigh[:, col_id]
            finite = np.isfinite(vals)
            if finite.any():
                clean = vals.copy()
                clean[~finite] = float(np.nanmedian(vals[finite]))
                row[f"{prefix}_{col}_wmean"] = float(np.average(clean, weights=weights))
                row[f"{prefix}_{col}_wstd"] = float(np.sqrt(np.average((clean - np.average(clean, weights=weights)) ** 2, weights=weights)))
            else:
                row[f"{prefix}_{col}_wmean"] = 0.0
                row[f"{prefix}_{col}_wstd"] = 0.0
        rows.append(row)
    return pd.DataFrame(rows).astype(np.float32)


def _assemble_curve_signal_matrix(
    root: Path,
    *,
    seed: int,
) -> dict[str, Any]:
    frames = load_frames(root)
    train = frames["train"].copy()
    phase1 = frames["phase1"].copy()
    test = frames["test"].copy()
    baseline = frames["baseline"].copy()

    source_frames = _source_frames(frames)
    for key in list(source_frames):
        source_frames[key] = _exclude_blinded_rows(source_frames[key], phase1, test)

    all_smiles = pd.concat(
        [
            train["SMILES"],
            phase1["SMILES"],
            test["SMILES"],
            *[df["SMILES"] for df in source_frames.values() if not df.empty and "SMILES" in df.columns],
        ],
        ignore_index=True,
    ).astype(str)
    unique_smiles = pd.Series(sorted(set(all_smiles)))
    all_x, all_fp = _feature_block_for_smiles(unique_smiles, seed=seed)
    x_by_smiles = {smi: all_x[i] for i, smi in enumerate(unique_smiles)}
    fp_by_smiles = {smi: all_fp[i] for i, smi in enumerate(unique_smiles)}

    def x_for(smiles: pd.Series) -> np.ndarray:
        return np.vstack([x_by_smiles[str(s)] for s in smiles.astype(str)]).astype(np.float32)

    def fp_for(smiles: pd.Series) -> np.ndarray:
        return np.vstack([fp_by_smiles[str(s)] for s in smiles.astype(str)]).astype(np.float32)

    x_train = x_for(train["SMILES"])
    fp_train = fp_for(train["SMILES"])
    x_phase = x_for(phase1["SMILES"])
    x_test = x_for(test["SMILES"])
    fp_phase = fp_for(phase1["SMILES"])
    fp_test = fp_for(test["SMILES"])

    y_train = _safe_numeric(train, "pEC50").to_numpy(float)
    knn_phase = train_knn_signal_features(fp_phase, fp_train, y_train, k=12)
    knn_test = train_knn_signal_features(fp_test, fp_train, y_train, k=12)

    head_preds_phase = pd.DataFrame(index=np.arange(len(phase1)))
    head_preds_test = pd.DataFrame(index=np.arange(len(test)))
    head_report = []
    specs = _curve_specs(source_frames)
    query_x = np.vstack([x_phase, x_test]).astype(np.float32)
    for spec in specs:
        try:
            query_pred, report = _fit_head_predictions(
                spec,
                source_frames[spec.source],
                x_by_smiles,
                query_x,
                seed=seed + len(head_report),
            )
        except Exception as exc:
            head_report.append({"head": spec.name, "status": "skipped", "reason": repr(exc)})
            continue
        phase_pred = query_pred[: len(phase1)]
        test_pred = query_pred[len(phase1) :]
        safe = _sanitize_feature_name(spec.name)
        head_preds_phase[f"curve_{safe}_pred"] = phase_pred
        head_preds_test[f"curve_{safe}_pred"] = test_pred
        head_report.append({**report, "status": "fit"})

    curve_source = source_frames["multitask"].copy()
    curve_cols = [c for c in ["pEC50", "log2fc_8um", "log2fc_33um"] if c in curve_source.columns]
    if not curve_source.empty and curve_cols:
        curve_source = (
            curve_source[["SMILES"] + curve_cols]
            .assign(**{c: pd.to_numeric(curve_source[c], errors="coerce") for c in curve_cols})
            .groupby("SMILES", as_index=False)
            .median(numeric_only=True)
        )
        source_fp = fp_for(curve_source["SMILES"])
        source_targets = curve_source[curve_cols].reset_index(drop=True)
        nn_phase = _nearest_curve_features(fp_phase, source_fp, source_targets, prefix="mt_nn", k=16)
        nn_test = _nearest_curve_features(fp_test, source_fp, source_targets, prefix="mt_nn", k=16)
    else:
        nn_phase = pd.DataFrame(index=np.arange(len(phase1)))
        nn_test = pd.DataFrame(index=np.arange(len(test)))

    baseline_phase = phase1[["Molecule Name"]].merge(
        baseline[["Molecule Name", "pEC50"]],
        on="Molecule Name",
        how="left",
    )["pEC50"].to_numpy(float)
    baseline_test = test[["Molecule Name"]].merge(
        baseline[["Molecule Name", "pEC50"]],
        on="Molecule Name",
        how="left",
    )["pEC50"].to_numpy(float)

    if np.isnan(baseline_phase).any() or np.isnan(baseline_test).any():
        raise ValueError("Baseline prediction file does not cover every Phase 1/test molecule.")

    from sklearn.decomposition import TruncatedSVD

    svd = TruncatedSVD(n_components=min(48, fp_train.shape[1] - 1, len(train) - 1), random_state=seed)
    svd.fit(fp_train)
    svd_phase = pd.DataFrame(
        svd.transform(fp_phase).astype(np.float32),
        columns=[f"fp_svd_{i:02d}" for i in range(svd.n_components)],
    )
    svd_test = pd.DataFrame(
        svd.transform(fp_test).astype(np.float32),
        columns=[f"fp_svd_{i:02d}" for i in range(svd.n_components)],
    )

    phase_meta = pd.concat(
        [
            pd.DataFrame({"anchor_pEC50": baseline_phase}),
            svd_phase,
            knn_phase.reset_index(drop=True),
            head_preds_phase.reset_index(drop=True),
            nn_phase.reset_index(drop=True),
        ],
        axis=1,
    )
    test_meta = pd.concat(
        [
            pd.DataFrame({"anchor_pEC50": baseline_test}),
            svd_test,
            knn_test.reset_index(drop=True),
            head_preds_test.reset_index(drop=True),
            nn_test.reset_index(drop=True),
        ],
        axis=1,
    )
    return {
        "frames": frames,
        "phase_meta": phase_meta,
        "test_meta": test_meta,
        "phase_anchor": baseline_phase,
        "test_anchor": baseline_test,
        "head_report": head_report,
    }


def run_curve_multitask_experiment(
    root: Path,
    *,
    n_folds: int = 5,
    n_boot: int = 5000,
    seed: int = 20260624,
    output_dir: Path | None = None,
) -> dict[str, Any]:
    _require_modeling_deps()
    root = Path(root)
    output_dir = Path(output_dir or (root / "reports" / "curve_multitask_experiment"))
    output_dir.mkdir(parents=True, exist_ok=True)

    assembled = _assemble_curve_signal_matrix(root, seed=seed)
    frames = assembled["frames"]
    phase1 = frames["phase1"].copy()
    test = frames["test"].copy()
    y = _safe_numeric(phase1, "pEC50").to_numpy(float)
    anchor = assembled["phase_anchor"]
    anchor_test = assembled["test_anchor"]
    x_phase = _clean_matrix(assembled["phase_meta"].to_numpy(np.float32))
    x_test = _clean_matrix(assembled["test_meta"].to_numpy(np.float32))

    folds = _stratified_scaffold_folds(phase1, n_folds, seed)
    configs = _residual_configs(seed)
    oof = np.full(len(phase1), np.nan, dtype=float)
    fold_rows = []
    inner_rows = []
    chosen = []
    for fold in range(n_folds):
        va = np.flatnonzero(folds == fold)
        tr = np.flatnonzero(folds != fold)
        cfg, inner = _inner_select_config(x_phase, y, anchor, tr, configs, seed=seed + fold)
        inner["outer_fold"] = fold
        inner_rows.append(inner)
        chosen.append(cfg)
        pred_res = _fit_predict_model(cfg.name, x_phase[tr], (y - anchor)[tr], x_phase[va], seed=seed + fold)
        oof[va] = _apply_residual(anchor[va], pred_res, cfg)
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

    if np.isnan(oof).any():
        raise RuntimeError("Nested OOF prediction has missing values.")

    fold_report = pd.DataFrame(fold_rows)
    inner_report = pd.concat(inner_rows, ignore_index=True)
    fold_report.to_csv(output_dir / "fold_metrics.csv", index=False)
    inner_report.to_csv(output_dir / "inner_config_scores.csv", index=False)
    pd.DataFrame(assembled["head_report"]).to_csv(output_dir / "curve_head_report.csv", index=False)

    anchor_score = _score_prediction(y, anchor)
    candidate_score = _score_prediction(y, oof)
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
    full_res = _fit_predict_model(majority_cfg.name, x_phase, y - anchor, x_test, seed=seed)
    corrected_test = _apply_residual(anchor_test, full_res, majority_cfg)

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
    oof_path = submissions / CURVE_OOF_FILE
    upload_path = submissions / CURVE_UPLOAD_FILE
    oof_submission.to_csv(oof_path, index=False)
    upload_candidate.to_csv(upload_path, index=False)

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
        "feature_count": int(x_phase.shape[1]),
        "curve_heads_fit": int(sum(1 for row in assembled["head_report"] if row.get("status") == "fit")),
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
        "# Curve-Aware Multitask Experiment",
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
        f"- Curve heads fit: {summary['curve_heads_fit']}",
        "",
        "## Files",
        "",
        f"- Honest OOF candidate: `{oof_path.relative_to(root)}`",
        f"- Experimental upload candidate: `{upload_path.relative_to(root)}`",
        f"- Preserved current final upload: `submissions/{FINAL_UPLOAD_FILE}`",
    ]
    (output_dir / "experiment_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    return summary
