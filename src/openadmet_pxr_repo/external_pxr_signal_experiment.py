from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import time
import urllib.parse
import urllib.request

import numpy as np
import pandas as pd

from .metrics import (
    LEADERBOARD_RAE_DENOM,
    bootstrap_paired_ci,
    ci_summary,
    exact_fill_count,
    mae,
    rae_from_mae,
)
from .sub040_signal_experiment import (
    BASELINE_FILE,
    FINAL_UPLOAD_FILE,
    _fingerprint_bits,
    _rdkit_descriptors,
    _descriptor_names,
    _safe_numeric,
    _scaffold_from_smiles,
    _stratified_scaffold_folds,
)


CHEMBL_PXR_TARGET = "CHEMBL3401"
EXTERNAL_OOF_FILE = "external_chembl_pxr_signal_oof_candidate.csv"
EXTERNAL_UPLOAD_FILE = "external_chembl_pxr_signal_upload_candidate.csv"


@dataclass(frozen=True)
class ExternalResidualConfig:
    model: str
    shrink: float
    cap: float

    @property
    def label(self) -> str:
        return (
            f"{self.model}_shrink{str(self.shrink).replace('.', 'p')}"
            f"_cap{str(self.cap).replace('.', 'p')}"
        )


def _require_deps() -> None:
    missing = []
    try:
        import rdkit  # noqa: F401
    except Exception:
        missing.append("rdkit")
    try:
        import sklearn  # noqa: F401
    except Exception:
        missing.append("scikit-learn")
    if missing:
        raise RuntimeError("Missing dependencies: " + ", ".join(missing))


def _chembl_url(offset: int, limit: int) -> str:
    query = urllib.parse.urlencode(
        {
            "target_chembl_id": CHEMBL_PXR_TARGET,
            "limit": limit,
            "offset": offset,
        }
    )
    return f"https://www.ebi.ac.uk/chembl/api/data/activity.json?{query}"


def download_chembl_pxr_activities(cache_path: Path, *, limit: int = 1000) -> pd.DataFrame:
    """Download public human PXR activity records from ChEMBL.

    ChEMBL target CHEMBL3401 is human NR1I2 / PXR. This function stores the raw
    records so repeated experiments do not keep hitting the API.
    """

    cache_path = Path(cache_path)
    if cache_path.exists():
        return pd.read_csv(cache_path)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    records: list[dict] = []
    offset = 0
    total = None
    while total is None or offset < total:
        url = _chembl_url(offset, limit)
        with urllib.request.urlopen(url, timeout=60) as handle:
            payload = json.loads(handle.read().decode("utf-8"))
        batch = payload.get("activities", [])
        records.extend(batch)
        meta = payload.get("page_meta", {})
        total = int(meta.get("total_count", len(records)))
        offset += limit
        print(f"ChEMBL PXR download: {min(offset, total)}/{total}", flush=True)
        time.sleep(0.15)

    raw = pd.DataFrame(records)
    raw.to_csv(cache_path, index=False)
    return raw


def _pvalue_from_row(row: pd.Series) -> float:
    pchembl = pd.to_numeric(row.get("pchembl_value"), errors="coerce")
    if np.isfinite(pchembl):
        return float(pchembl)

    value = pd.to_numeric(row.get("standard_value"), errors="coerce")
    if not np.isfinite(value) or value <= 0:
        return np.nan
    units = str(row.get("standard_units", "")).lower()
    if units in {"nm", "nanomolar"}:
        molar = value * 1e-9
    elif units in {"um", "micromolar", "\u00b5m"}:
        molar = value * 1e-6
    elif units in {"mm", "millimolar"}:
        molar = value * 1e-3
    elif units in {"m"}:
        molar = value
    else:
        return np.nan
    if molar <= 0:
        return np.nan
    return float(-np.log10(molar))


def curate_chembl_pxr_labels(raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build activation-like and inhibition-like public PXR potency tables."""

    if raw.empty:
        raise RuntimeError("ChEMBL PXR activity table is empty.")

    df = raw.copy()
    df["canonical_smiles"] = df.get("canonical_smiles", "").astype(str)
    df["standard_type"] = df.get("standard_type", "").astype(str).str.upper()
    df["assay_description"] = df.get("assay_description", "").fillna("").astype(str)
    df["pchembl_external"] = df.apply(_pvalue_from_row, axis=1)
    df = df[
        df["canonical_smiles"].str.len().gt(0)
        & np.isfinite(df["pchembl_external"])
        & df["standard_relation"].fillna("=").astype(str).isin(["=", "~"])
    ].copy()

    text = df["assay_description"].str.lower()
    inhibitory = text.str.contains("inhib|antagon|suppress|block", regex=True)
    activation_words = text.str.contains("activ|agon|induc|transactiv|transcription", regex=True)
    ec_like = df["standard_type"].isin(["EC50", "AC50", "MEC"])
    ic_like = df["standard_type"].isin(["IC50", "KI", "KD"])

    activation_mask = ec_like & (~inhibitory | activation_words)
    inhibition_mask = (ic_like | inhibitory) & ~activation_mask
    activation = df[activation_mask].copy()
    inhibition = df[inhibition_mask].copy()

    def aggregate(frame: pd.DataFrame, label: str) -> pd.DataFrame:
        if frame.empty:
            return pd.DataFrame(columns=["SMILES", label, f"{label}_n"])
        grouped = (
            frame.groupby("canonical_smiles", as_index=False)
            .agg(
                value=("pchembl_external", "median"),
                n=("pchembl_external", "size"),
                stype=("standard_type", lambda x: ";".join(sorted(set(map(str, x)))[:8])),
            )
            .rename(columns={"canonical_smiles": "SMILES", "value": label, "n": f"{label}_n"})
        )
        return grouped

    return aggregate(activation, "chembl_pxr_activation_p"), aggregate(inhibition, "chembl_pxr_inhibition_p")


def _build_feature_matrix(smiles: pd.Series, *, n_bits: int = 1024) -> np.ndarray:
    _require_deps()
    desc = np.asarray([_rdkit_descriptors(s) for s in smiles.astype(str)], dtype=np.float32)
    ecfp = np.vstack([_fingerprint_bits(s, n_bits, radius=2, use_features=False) for s in smiles.astype(str)])
    fcfp = np.vstack([_fingerprint_bits(s, n_bits, radius=2, use_features=True) for s in smiles.astype(str)])
    x = np.hstack([desc, ecfp, fcfp]).astype(np.float32)
    return np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)


def _binary_fp_matrix(smiles: pd.Series, *, n_bits: int = 2048) -> np.ndarray:
    return np.vstack([_fingerprint_bits(s, n_bits, radius=2, use_features=False) for s in smiles.astype(str)])


def _tanimoto(query: np.ndarray, ref: np.ndarray) -> np.ndarray:
    query = np.asarray(query, dtype=np.float32)
    ref = np.asarray(ref, dtype=np.float32)
    inter = query @ ref.T
    denom = np.maximum(query.sum(axis=1, keepdims=True) + ref.sum(axis=1, keepdims=True).T - inter, 1e-6)
    return inter / denom


def _fit_external_model(label_df: pd.DataFrame, query_smiles: pd.Series, *, label_col: str, seed: int) -> pd.DataFrame:
    from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
    from sklearn.linear_model import HuberRegressor, Ridge
    from sklearn.model_selection import KFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    out = pd.DataFrame(index=np.arange(len(query_smiles)))
    out[f"{label_col}_available"] = 0.0
    out[f"{label_col}_pred"] = 0.0
    out[f"{label_col}_nn"] = 0.0
    out[f"{label_col}_maxsim"] = 0.0

    if len(label_df) < 100:
        return out

    label_df = label_df.dropna(subset=[label_col]).drop_duplicates("SMILES").reset_index(drop=True)
    y = pd.to_numeric(label_df[label_col], errors="coerce").to_numpy(float)
    keep = np.isfinite(y)
    label_df = label_df.loc[keep].reset_index(drop=True)
    y = y[keep]
    if len(y) < 100:
        return out

    x_ext = _build_feature_matrix(label_df["SMILES"])
    x_query = _build_feature_matrix(query_smiles)

    models = [
        make_pipeline(StandardScaler(with_mean=False), Ridge(alpha=25.0)),
        make_pipeline(StandardScaler(with_mean=False), HuberRegressor(alpha=0.005, epsilon=1.5, max_iter=500)),
        ExtraTreesRegressor(
            n_estimators=450,
            min_samples_leaf=3,
            max_features=0.35,
            random_state=seed,
            n_jobs=-1,
        ),
        HistGradientBoostingRegressor(
            loss="absolute_error",
            learning_rate=0.035,
            max_iter=450,
            max_leaf_nodes=15,
            l2_regularization=0.05,
            random_state=seed,
        ),
    ]

    oof = np.zeros((len(y), len(models)), dtype=np.float32)
    kf = KFold(n_splits=5, shuffle=True, random_state=seed)
    for j, model in enumerate(models):
        for tr, va in kf.split(x_ext):
            model.fit(x_ext[tr], y[tr])
            oof[va, j] = np.asarray(model.predict(x_ext[va]), dtype=np.float32)

    model_mae = np.mean(np.abs(oof - y[:, None]), axis=0)
    weights = 1 / np.maximum(model_mae, 1e-6)
    weights = weights / weights.sum()
    pred_query = np.zeros(len(query_smiles), dtype=float)
    for w, model in zip(weights, models):
        model.fit(x_ext, y)
        pred_query += float(w) * np.asarray(model.predict(x_query), dtype=float)

    fp_ext = _binary_fp_matrix(label_df["SMILES"])
    fp_query = _binary_fp_matrix(query_smiles)
    sims = _tanimoto(fp_query, fp_ext)
    order = np.argsort(-sims, axis=1)[:, :5]
    top_sims = np.take_along_axis(sims, order, axis=1)
    nn_y = y[order]
    nn_weights = np.power(np.maximum(top_sims, 1e-6), 3)
    nn_pred = (nn_y * nn_weights).sum(axis=1) / np.maximum(nn_weights.sum(axis=1), 1e-6)

    out[f"{label_col}_available"] = 1.0
    out[f"{label_col}_pred"] = pred_query
    out[f"{label_col}_nn"] = nn_pred
    out[f"{label_col}_maxsim"] = top_sims[:, 0]
    return out


def _load_frames(root: Path) -> dict[str, pd.DataFrame]:
    data = root / "data"
    subs = root / "submissions"
    return {
        "train": pd.read_csv(data / "pxr-challenge_TRAIN.csv"),
        "phase1": pd.read_csv(data / "phase1_unblinded.csv"),
        "test": pd.read_csv(data / "pxr-challenge_TEST_BLINDED.csv"),
        "anchor": pd.read_csv(subs / BASELINE_FILE),
        "final": pd.read_csv(subs / FINAL_UPLOAD_FILE),
    }


def _align_anchor(phase1: pd.DataFrame, test: pd.DataFrame, anchor: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    anchor = anchor[["Molecule Name", "pEC50"]].copy()
    phase = phase1[["Molecule Name"]].merge(anchor, on="Molecule Name", how="left")["pEC50"].to_numpy(float)
    test_pred = test[["Molecule Name"]].merge(anchor, on="Molecule Name", how="left")["pEC50"].to_numpy(float)
    if not np.isfinite(phase).all() or not np.isfinite(test_pred).all():
        raise RuntimeError(f"{BASELINE_FILE} does not cover all Phase1/test rows.")
    return phase, test_pred


def _residual_configs() -> list[ExternalResidualConfig]:
    configs = []
    for model in ["ridge", "huber", "extra_trees", "hist_gbdt"]:
        for shrink in [0.15, 0.25, 0.35, 0.45]:
            for cap in [0.12, 0.20, 0.30]:
                configs.append(ExternalResidualConfig(model=model, shrink=shrink, cap=cap))
    return configs


def _fit_predict_residual(cfg: ExternalResidualConfig, x_tr: np.ndarray, y_resid: np.ndarray, x_va: np.ndarray, seed: int) -> np.ndarray:
    from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
    from sklearn.linear_model import HuberRegressor, Ridge
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    if cfg.model == "ridge":
        model = make_pipeline(StandardScaler(), Ridge(alpha=15.0))
    elif cfg.model == "huber":
        model = make_pipeline(StandardScaler(), HuberRegressor(alpha=0.01, epsilon=1.35, max_iter=500))
    elif cfg.model == "extra_trees":
        model = ExtraTreesRegressor(
            n_estimators=300,
            min_samples_leaf=6,
            max_features=0.75,
            random_state=seed,
            n_jobs=-1,
        )
    elif cfg.model == "hist_gbdt":
        model = HistGradientBoostingRegressor(
            loss="absolute_error",
            learning_rate=0.025,
            max_iter=250,
            max_leaf_nodes=7,
            l2_regularization=0.2,
            random_state=seed,
        )
    else:
        raise ValueError(cfg.model)
    model.fit(x_tr, y_resid)
    pred = np.asarray(model.predict(x_va), dtype=float)
    return np.clip(pred * cfg.shrink, -cfg.cap, cfg.cap)


def _inner_select(
    x: np.ndarray,
    y: np.ndarray,
    anchor: np.ndarray,
    train_idx: np.ndarray,
    configs: list[ExternalResidualConfig],
    seed: int,
) -> tuple[ExternalResidualConfig, pd.DataFrame]:
    from sklearn.model_selection import KFold

    rows = []
    inner = KFold(n_splits=4, shuffle=True, random_state=seed)
    for cfg in configs:
        scores = []
        for rel_tr, rel_va in inner.split(train_idx):
            tr = train_idx[rel_tr]
            va = train_idx[rel_va]
            resid = y[tr] - anchor[tr]
            corr = _fit_predict_residual(cfg, x[tr], resid, x[va], seed=seed)
            pred = np.clip(anchor[va] + corr, 1.0, 8.5)
            scores.append(mae(y[va], pred))
        rows.append(
            {
                "config": cfg.label,
                "model": cfg.model,
                "shrink": cfg.shrink,
                "cap": cfg.cap,
                "inner_mae": float(np.mean(scores)),
                "inner_mae_std": float(np.std(scores)),
            }
        )
    report = pd.DataFrame(rows).sort_values(["inner_mae", "inner_mae_std"], ascending=True).reset_index(drop=True)
    top = report.iloc[0]
    return ExternalResidualConfig(str(top["model"]), float(top["shrink"]), float(top["cap"])), report


def _feature_block(query: pd.DataFrame, chembl_act: pd.DataFrame, chembl_inh: pd.DataFrame, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    act = _fit_external_model(chembl_act, query["SMILES"], label_col="chembl_pxr_activation_p", seed=seed)
    inh = _fit_external_model(chembl_inh, query["SMILES"], label_col="chembl_pxr_inhibition_p", seed=seed + 13)

    desc = np.asarray([_rdkit_descriptors(s) for s in query["SMILES"].astype(str)], dtype=np.float32)
    desc_df = pd.DataFrame(desc, columns=[f"rdkit_{n}" for n in _descriptor_names()])
    focused_cols = [
        "rdkit_MolWt",
        "rdkit_MolLogP",
        "rdkit_TPSA",
        "rdkit_NumHDonors",
        "rdkit_NumHAcceptors",
        "rdkit_NumRotatableBonds",
        "rdkit_FractionCSP3",
        "rdkit_NumAromaticRings",
        "rdkit_FormalCharge",
    ]
    features = pd.concat([act, inh, desc_df[focused_cols].reset_index(drop=True)], axis=1)
    features = features.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    report = pd.DataFrame(
        [
            {"source": "activation", "rows": len(chembl_act), "unique_smiles": chembl_act["SMILES"].nunique()},
            {"source": "inhibition", "rows": len(chembl_inh), "unique_smiles": chembl_inh["SMILES"].nunique()},
        ]
    )
    return features.astype(float), report


def run_external_pxr_signal_experiment(
    root: Path,
    *,
    n_folds: int = 5,
    n_boot: int = 5000,
    seed: int = 20260628,
    output_dir: Path | None = None,
) -> dict:
    _require_deps()
    root = Path(root)
    output_dir = output_dir or (root / "reports" / "external_chembl_pxr_signal_experiment")
    output_dir.mkdir(parents=True, exist_ok=True)
    (root / "submissions").mkdir(parents=True, exist_ok=True)

    frames = _load_frames(root)
    train = frames["train"].copy()
    phase1 = frames["phase1"].copy()
    test = frames["test"].copy()
    y = _safe_numeric(phase1, "pEC50").to_numpy(float)
    anchor_phase, anchor_test = _align_anchor(phase1, test, frames["anchor"])

    raw = download_chembl_pxr_activities(output_dir / "chembl_pxr_raw_activities.csv")
    chembl_act, chembl_inh = curate_chembl_pxr_labels(raw)
    chembl_act.to_csv(output_dir / "chembl_pxr_activation_curated.csv", index=False)
    chembl_inh.to_csv(output_dir / "chembl_pxr_inhibition_curated.csv", index=False)

    query = pd.concat(
        [
            phase1[["Molecule Name", "SMILES"]].assign(_split="phase1"),
            test[["Molecule Name", "SMILES"]].assign(_split="test"),
        ],
        ignore_index=True,
    )
    x_all, external_report = _feature_block(query, chembl_act, chembl_inh, seed)
    external_report.to_csv(output_dir / "external_signal_source_report.csv", index=False)

    n_phase = len(phase1)
    x_phase = x_all.iloc[:n_phase].to_numpy(float)
    x_test = x_all.iloc[n_phase:].to_numpy(float)
    folds = _stratified_scaffold_folds(phase1, n_folds, seed)
    pd.DataFrame({"Molecule Name": phase1["Molecule Name"], "fold": folds}).to_csv(
        output_dir / "phase1_scaffold_stratified_folds.csv", index=False
    )

    configs = _residual_configs()
    oof = anchor_phase.copy()
    fold_rows = []
    inner_rows = []
    for fold in range(n_folds):
        val_idx = np.flatnonzero(folds == fold)
        train_idx = np.flatnonzero(folds != fold)
        cfg, inner = _inner_select(x_phase, y, anchor_phase, train_idx, configs, seed=seed + fold)
        inner["outer_fold"] = fold
        inner_rows.append(inner)
        residual = y[train_idx] - anchor_phase[train_idx]
        correction = _fit_predict_residual(cfg, x_phase[train_idx], residual, x_phase[val_idx], seed=seed + fold)
        oof[val_idx] = np.clip(anchor_phase[val_idx] + correction, 1.0, 8.5)
        fold_rows.append(
            {
                "fold": fold,
                "n": len(val_idx),
                "selected_config": cfg.label,
                "anchor_mae": mae(y[val_idx], anchor_phase[val_idx]),
                "candidate_mae": mae(y[val_idx], oof[val_idx]),
                "anchor_rae": rae_from_mae(mae(y[val_idx], anchor_phase[val_idx]), LEADERBOARD_RAE_DENOM),
                "candidate_rae": rae_from_mae(mae(y[val_idx], oof[val_idx]), LEADERBOARD_RAE_DENOM),
                "improved": mae(y[val_idx], oof[val_idx]) < mae(y[val_idx], anchor_phase[val_idx]),
            }
        )

    fold_report = pd.DataFrame(fold_rows)
    inner_report = pd.concat(inner_rows, ignore_index=True)
    fold_report.to_csv(output_dir / "fold_metrics.csv", index=False)
    inner_report.to_csv(output_dir / "inner_config_scores.csv", index=False)

    anchor_mae = mae(y, anchor_phase)
    cand_mae = mae(y, oof)
    anchor_score = {"mae": anchor_mae, "rae": rae_from_mae(anchor_mae, LEADERBOARD_RAE_DENOM)}
    candidate_score = {"mae": cand_mae, "rae": rae_from_mae(cand_mae, LEADERBOARD_RAE_DENOM)}
    boot = bootstrap_paired_ci(y, oof, n_boot=n_boot, seed=seed)
    ci = {k: ci_summary(v) for k, v in boot.items()}
    improved_folds = int(fold_report["improved"].sum())
    exact = exact_fill_count(y, oof)
    corr = float(np.corrcoef(anchor_phase, oof)[0, 1])

    if exact == 0 and cand_mae < 0.400 and candidate_score["rae"] < 0.520 and improved_folds >= 4:
        decision = "SUB040_CREDIBLE_REVIEW_FOR_REPLACEMENT"
    elif exact == 0 and cand_mae < 0.410 and candidate_score["rae"] < 0.540 and improved_folds >= 4:
        decision = "REAL_IMPROVEMENT_REVIEW_BEFORE_REPLACEMENT"
    elif cand_mae < anchor_mae and improved_folds >= 3:
        decision = "WEAK_EXPLORATORY_SIGNAL_KEEP_FINAL_LOCKED"
    else:
        decision = "EXPLORATORY_DO_NOT_REPLACE_FINAL"

    oof_path = root / "submissions" / EXTERNAL_OOF_FILE
    oof_df = phase1[["SMILES", "Molecule Name"]].copy()
    oof_df["pEC50"] = oof
    oof_df.to_csv(oof_path, index=False)

    best_cfg = fold_report["selected_config"].mode().iloc[0]
    model_name, shrink_s, cap_s = best_cfg.split("_shrink")
    shrink = float(shrink_s.split("_cap")[0].replace("p", "."))
    cap = float(shrink_s.split("_cap")[1].replace("p", "."))
    cfg = ExternalResidualConfig(model=model_name, shrink=shrink, cap=cap)
    final_resid = y - anchor_phase
    final_correction = _fit_predict_residual(cfg, x_phase, final_resid, x_test, seed=seed + 101)
    candidate_test = np.clip(anchor_test + final_correction, 1.0, 8.5)

    upload_path = root / "submissions" / EXTERNAL_UPLOAD_FILE
    upload = test[["SMILES", "Molecule Name"]].copy()
    phase1_truth = dict(zip(phase1["Molecule Name"].astype(str), y))
    by_name = dict(zip(test["Molecule Name"].astype(str), candidate_test))
    upload["pEC50"] = [
        phase1_truth.get(name, by_name[name]) for name in test["Molecule Name"].astype(str)
    ]
    upload.to_csv(upload_path, index=False)

    summary = {
        "decision": decision,
        "anchor": anchor_score,
        "candidate": candidate_score,
        "candidate_ci": ci,
        "folds_improved": improved_folds,
        "exact_phase1_matches_oof": exact,
        "anchor_candidate_corr": corr,
        "chembl_target": CHEMBL_PXR_TARGET,
        "source_rows": {
            "raw": len(raw),
            "activation_curated": len(chembl_act),
            "inhibition_curated": len(chembl_inh),
        },
        "oof_candidate_file": str(oof_path.relative_to(root)),
        "upload_candidate_file": str(upload_path.relative_to(root)),
    }
    (output_dir / "experiment_summary.json").write_text(json.dumps(summary, indent=2))

    lines = [
        "# External ChEMBL PXR Signal Experiment",
        "",
        f"Decision: **{decision}**",
        "",
        "## Signal",
        f"- ChEMBL target: `{CHEMBL_PXR_TARGET}` / human NR1I2-PXR",
        f"- Raw public PXR records: {len(raw)}",
        f"- Curated activation-like records: {len(chembl_act)}",
        f"- Curated inhibition-like records: {len(chembl_inh)}",
        "",
        "## Metrics",
        f"- Anchor MAE/RAE: {anchor_score['mae']:.6f} / {anchor_score['rae']:.6f}",
        f"- Candidate OOF MAE/RAE: {candidate_score['mae']:.6f} / {candidate_score['rae']:.6f}",
        f"- Candidate MAE 95% CI: {ci['mae']['lo']:.6f} - {ci['mae']['hi']:.6f}",
        f"- Candidate RAE 95% CI: {ci['rae_fixed']['lo']:.6f} - {ci['rae_fixed']['hi']:.6f}",
        f"- Exact Phase1 matches in OOF candidate: {exact}",
        f"- Folds improved: {improved_folds}/{n_folds}",
        f"- Anchor/candidate correlation: {corr:.4f}",
        "",
        "Do not replace the locked final submission unless the decision is a review-for-replacement state.",
    ]
    (output_dir / "experiment_report.md").write_text("\n".join(lines) + "\n")
    return summary
