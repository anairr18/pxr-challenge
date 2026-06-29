from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any
import warnings

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


BASELINE_FILE = "suiren_chemeleon_blend_weight_0p325_predictions.csv"
FINAL_UPLOAD_FILE = "openadmet_pxr_activity_final_submission.csv"
OOF_CANDIDATE_FILE = "structure_assay_residual_oof_candidate.csv"
UPLOAD_CANDIDATE_FILE = "structure_assay_residual_upload_candidate.csv"


@dataclass(frozen=True)
class ResidualConfig:
    name: str
    shrink: float
    cap: float

    @property
    def label(self) -> str:
        cap = str(self.cap).replace(".", "p")
        shrink = str(self.shrink).replace(".", "p")
        return f"{self.name}_shrink{shrink}_cap{cap}"


def _require_modeling_deps():
    missing = []
    try:
        import sklearn  # noqa: F401
    except Exception:
        missing.append("scikit-learn")
    try:
        import rdkit  # noqa: F401
    except Exception:
        missing.append("rdkit")
    if missing:
        raise RuntimeError(
            "Missing modeling dependencies: "
            + ", ".join(missing)
            + ". Install requirements.txt in Colab/Kaggle before running the sub-0.40 experiment."
        )


def _safe_numeric(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        return pd.Series(np.nan, index=df.index, dtype=float)
    return pd.to_numeric(df[col], errors="coerce")


def _load_required_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def load_frames(root: Path) -> dict[str, pd.DataFrame]:
    root = Path(root)
    data = root / "data"
    submissions = root / "submissions"
    frames = {
        "train": _load_required_csv(data / "pxr-challenge_TRAIN.csv"),
        "phase1": _load_required_csv(data / "phase1_unblinded.csv"),
        "test": _load_required_csv(data / "pxr-challenge_TEST_BLINDED.csv"),
        "baseline": _load_required_csv(submissions / BASELINE_FILE),
    }
    optional = {
        "counter": data / "pxr-challenge_counter-assay_TRAIN.csv",
        "multitask": data / "multitask_train.csv",
    }
    for key, path in optional.items():
        frames[key] = pd.read_csv(path) if path.exists() else pd.DataFrame()
    return frames


def _mol_from_smiles(smiles: str):
    from rdkit import Chem

    if not isinstance(smiles, str) or not smiles.strip():
        return None
    return Chem.MolFromSmiles(smiles)


def _scaffold_from_smiles(smiles: str) -> str:
    from rdkit.Chem.Scaffolds import MurckoScaffold

    mol = _mol_from_smiles(smiles)
    if mol is None:
        return f"invalid::{smiles}"
    scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)
    return scaffold or f"acyclic::{smiles}"


def _descriptor_names() -> list[str]:
    return [
        "MolWt",
        "MolLogP",
        "TPSA",
        "NumHDonors",
        "NumHAcceptors",
        "NumRotatableBonds",
        "RingCount",
        "HeavyAtomCount",
        "FractionCSP3",
        "NumAromaticRings",
        "NumAliphaticRings",
        "LabuteASA",
        "BertzCT",
        "MolMR",
        "BalabanJ",
        "Kappa1",
        "Kappa2",
        "Chi0v",
        "Chi1v",
        "FormalCharge",
        "HeteroAtomCount",
    ]


def _rdkit_descriptors(smiles: str) -> list[float]:
    from rdkit import Chem
    from rdkit.Chem import Crippen, Descriptors, Lipinski, rdMolDescriptors

    mol = _mol_from_smiles(smiles)
    if mol is None:
        return [np.nan] * len(_descriptor_names())
    hetero = sum(1 for atom in mol.GetAtoms() if atom.GetAtomicNum() not in (1, 6))
    values = [
        Descriptors.MolWt(mol),
        Crippen.MolLogP(mol),
        rdMolDescriptors.CalcTPSA(mol),
        Lipinski.NumHDonors(mol),
        Lipinski.NumHAcceptors(mol),
        Lipinski.NumRotatableBonds(mol),
        rdMolDescriptors.CalcNumRings(mol),
        mol.GetNumHeavyAtoms(),
        rdMolDescriptors.CalcFractionCSP3(mol),
        rdMolDescriptors.CalcNumAromaticRings(mol),
        rdMolDescriptors.CalcNumAliphaticRings(mol),
        rdMolDescriptors.CalcLabuteASA(mol),
        Descriptors.BertzCT(mol),
        Crippen.MolMR(mol),
        Descriptors.BalabanJ(mol),
        Descriptors.Kappa1(mol),
        Descriptors.Kappa2(mol),
        Descriptors.Chi0v(mol),
        Descriptors.Chi1v(mol),
        Chem.GetFormalCharge(mol),
        hetero,
    ]
    return [float(v) if np.isfinite(v) else np.nan for v in values]


def _fingerprint_bits(smiles: str, n_bits: int, radius: int, use_features: bool) -> np.ndarray:
    from rdkit import DataStructs
    from rdkit.Chem import rdMolDescriptors

    mol = _mol_from_smiles(smiles)
    arr = np.zeros((n_bits,), dtype=np.float32)
    if mol is None:
        return arr
    bitvect = rdMolDescriptors.GetMorganFingerprintAsBitVect(
        mol,
        radius,
        nBits=n_bits,
        useFeatures=use_features,
    )
    DataStructs.ConvertToNumpyArray(bitvect, arr)
    return arr


def _three_d_descriptors(smiles: str, seed: int = 20260623) -> list[float]:
    from rdkit import Chem
    from rdkit.Chem import AllChem, rdMolDescriptors

    mol = _mol_from_smiles(smiles)
    if mol is None:
        return [np.nan, np.nan, np.nan, np.nan, np.nan]
    mol = Chem.AddHs(mol)
    params = AllChem.ETKDGv3()
    params.randomSeed = int(seed) % (2**31 - 1)
    params.numThreads = 0
    status = AllChem.EmbedMolecule(mol, params)
    if status != 0:
        return [np.nan, np.nan, np.nan, np.nan, np.nan]
    try:
        AllChem.MMFFOptimizeMolecule(mol, maxIters=100)
    except Exception:
        pass
    return [
        float(rdMolDescriptors.CalcPBF(mol)),
        float(rdMolDescriptors.CalcPMI1(mol)),
        float(rdMolDescriptors.CalcPMI2(mol)),
        float(rdMolDescriptors.CalcPMI3(mol)),
        float(rdMolDescriptors.CalcRadiusOfGyration(mol)),
    ]


def build_molecular_features(
    smiles: pd.Series,
    *,
    n_bits: int = 1024,
    with_3d: bool = False,
    seed: int = 20260623,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    _require_modeling_deps()

    smiles = smiles.astype(str).reset_index(drop=True)
    desc_rows = [_rdkit_descriptors(s) for s in smiles]
    desc = pd.DataFrame(desc_rows, columns=[f"rdkit_{n}" for n in _descriptor_names()])
    if with_3d:
        d3_rows = [_three_d_descriptors(s, seed=seed) for s in smiles]
        d3 = pd.DataFrame(
            d3_rows,
            columns=[
                "rdkit3d_pbf",
                "rdkit3d_pmi1",
                "rdkit3d_pmi2",
                "rdkit3d_pmi3",
                "rdkit3d_rgyr",
            ],
        )
        desc = pd.concat([desc, d3], axis=1)

    ecfp = np.vstack([_fingerprint_bits(s, n_bits, radius=2, use_features=False) for s in smiles])
    fcfp = np.vstack([_fingerprint_bits(s, n_bits, radius=2, use_features=True) for s in smiles])
    return desc.astype(np.float32), ecfp.astype(np.float32), fcfp.astype(np.float32)


def _clean_matrix(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    return np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)


def _tanimoto_dense(query: np.ndarray, reference: np.ndarray) -> np.ndarray:
    query = np.asarray(query, dtype=np.float32)
    reference = np.asarray(reference, dtype=np.float32)
    inter = query @ reference.T
    qsum = query.sum(axis=1, keepdims=True)
    rsum = reference.sum(axis=1, keepdims=True).T
    denom = np.maximum(qsum + rsum - inter, 1e-6)
    return inter / denom


def train_knn_signal_features(
    query_fp: np.ndarray,
    train_fp: np.ndarray,
    y_train: np.ndarray,
    *,
    k: int = 8,
) -> pd.DataFrame:
    sim = _tanimoto_dense(query_fp, train_fp)
    k_eff = min(k, sim.shape[1])
    idx = np.argpartition(-sim, kth=k_eff - 1, axis=1)[:, :k_eff]
    rows = []
    for row_id in range(sim.shape[0]):
        sims = sim[row_id, idx[row_id]]
        labels = y_train[idx[row_id]]
        order = np.argsort(-sims)
        sims = sims[order]
        labels = labels[order]
        weights = np.maximum(sims, 1e-6) ** 2
        rows.append(
            {
                "knn_train_maxsim": float(sims[0]),
                "knn_train_meansim": float(np.mean(sims)),
                "knn_train_label_mean": float(np.average(labels, weights=weights)),
                "knn_train_label_std": float(np.std(labels)),
                "knn_train_low_frac": float(np.mean(labels < 4.5)),
                "knn_train_active_frac": float(np.mean(labels >= 6.0)),
                "knn_train_disagreement": float(np.std(labels) * (1.0 - sims[0])),
            }
        )
    return pd.DataFrame(rows)


def _stratified_scaffold_folds(phase1: pd.DataFrame, n_splits: int, seed: int) -> np.ndarray:
    from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold

    y = _safe_numeric(phase1, "pEC50").to_numpy(float)
    bins = pd.qcut(y, q=min(5, len(np.unique(y))), labels=False, duplicates="drop")
    scaffolds = phase1["SMILES"].astype(str).map(_scaffold_from_smiles).to_numpy()
    folds = np.full(len(phase1), -1, dtype=int)
    try:
        splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        splits = splitter.split(phase1, bins, groups=scaffolds)
    except Exception:
        splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        splits = splitter.split(phase1, bins)
    for fold, (_, val_idx) in enumerate(splits):
        folds[val_idx] = fold
    if (folds < 0).any():
        raise RuntimeError("Failed to assign every Phase 1 molecule to a CV fold.")
    return folds


def _base_estimators(seed: int) -> dict[str, Any]:
    from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
    from sklearn.linear_model import HuberRegressor, Ridge
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    estimators: dict[str, Any] = {
        "ridge": make_pipeline(StandardScaler(), Ridge(alpha=10.0, random_state=seed)),
        "huber": make_pipeline(
            StandardScaler(),
            HuberRegressor(alpha=0.001, epsilon=1.35, max_iter=1000),
        ),
        "extra_trees": ExtraTreesRegressor(
            n_estimators=450,
            min_samples_leaf=5,
            max_features=0.45,
            random_state=seed,
            n_jobs=-1,
        ),
        "hist_gbdt": HistGradientBoostingRegressor(
            loss="absolute_error",
            learning_rate=0.025,
            max_leaf_nodes=15,
            l2_regularization=0.2,
            min_samples_leaf=12,
            random_state=seed,
        ),
    }
    try:
        from lightgbm import LGBMRegressor

        estimators["lightgbm_l1"] = LGBMRegressor(
            objective="mae",
            n_estimators=450,
            learning_rate=0.025,
            num_leaves=15,
            min_child_samples=12,
            subsample=0.85,
            colsample_bytree=0.55,
            reg_alpha=0.1,
            reg_lambda=1.5,
            random_state=seed,
            n_jobs=-1,
            verbose=-1,
        )
    except Exception:
        pass
    try:
        from xgboost import XGBRegressor

        estimators["xgboost_l1"] = XGBRegressor(
            objective="reg:absoluteerror",
            n_estimators=450,
            learning_rate=0.025,
            max_depth=3,
            min_child_weight=8,
            subsample=0.85,
            colsample_bytree=0.55,
            reg_alpha=0.1,
            reg_lambda=2.0,
            random_state=seed,
            n_jobs=-1,
        )
    except Exception:
        pass
    return estimators


def _fit_predict_model(name: str, x_tr: np.ndarray, y_tr: np.ndarray, x_va: np.ndarray, seed: int) -> np.ndarray:
    model = _base_estimators(seed)[name]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(x_tr, y_tr)
    return np.asarray(model.predict(x_va), dtype=float)


def _residual_configs(seed: int) -> list[ResidualConfig]:
    names = list(_base_estimators(seed).keys())
    configs = []
    for name in names:
        for shrink in (0.25, 0.40, 0.55):
            for cap in (0.15, 0.25, 0.35):
                configs.append(ResidualConfig(name=name, shrink=shrink, cap=cap))
    return configs


def _apply_residual(anchor: np.ndarray, residual: np.ndarray, cfg: ResidualConfig) -> np.ndarray:
    correction = np.clip(np.asarray(residual, float), -cfg.cap, cfg.cap) * cfg.shrink
    return np.clip(np.asarray(anchor, float) + correction, 1.0, 8.5)


def _score_prediction(y_true: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    m = mae(y_true, pred)
    return {"mae": m, "rae": rae_from_mae(m)}


def _inner_select_config(
    x: np.ndarray,
    y: np.ndarray,
    anchor: np.ndarray,
    train_idx: np.ndarray,
    configs: list[ResidualConfig],
    *,
    seed: int,
    n_splits: int = 4,
) -> tuple[ResidualConfig, pd.DataFrame]:
    from sklearn.model_selection import KFold

    residual = y - anchor
    inner = KFold(n_splits=min(n_splits, len(train_idx)), shuffle=True, random_state=seed)
    rows = []
    for cfg in configs:
        fold_scores = []
        for inner_tr_pos, inner_va_pos in inner.split(train_idx):
            tr = train_idx[inner_tr_pos]
            va = train_idx[inner_va_pos]
            pred_res = _fit_predict_model(cfg.name, x[tr], residual[tr], x[va], seed=seed)
            pred = _apply_residual(anchor[va], pred_res, cfg)
            fold_scores.append(mae(y[va], pred))
        rows.append(
            {
                "config": cfg.label,
                "model": cfg.name,
                "shrink": cfg.shrink,
                "cap": cfg.cap,
                "inner_mae": float(np.mean(fold_scores)),
                "inner_mae_std": float(np.std(fold_scores)),
            }
        )
    results = pd.DataFrame(rows).sort_values(["inner_mae", "shrink", "cap"]).reset_index(drop=True)
    best_row = results.iloc[0]
    best = ResidualConfig(
        name=str(best_row["model"]),
        shrink=float(best_row["shrink"]),
        cap=float(best_row["cap"]),
    )
    return best, results


def _fit_auxiliary_predictions(
    train: pd.DataFrame,
    counter: pd.DataFrame,
    multitask: pd.DataFrame,
    x_train: np.ndarray,
    x_query: np.ndarray,
    *,
    seed: int,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    from sklearn.ensemble import ExtraTreesRegressor

    targets = []
    for name, frame, columns in [
        (
            "train",
            train,
            [
                "pEC50",
                "Emax_estimate (log2FC vs. baseline)",
                "Emax.vs.pos.ctrl_estimate (dimensionless)",
            ],
        ),
        (
            "counter",
            counter,
            [
                "pEC50",
                "Emax_estimate (log2FC vs. baseline)",
                "Emax.vs.pos.ctrl_estimate (dimensionless)",
            ],
        ),
        ("multitask", multitask, ["pEC50", "log2fc_8um", "log2fc_33um"]),
    ]:
        if frame.empty:
            continue
        for col in columns:
            if col in frame.columns:
                targets.append((f"aux_{name}_{col}", frame, col))

    out = pd.DataFrame(index=np.arange(x_query.shape[0]))
    report = []
    for feature_name, frame, col in targets:
        y = pd.to_numeric(frame[col], errors="coerce")
        if len(frame) != len(train):
            # The auxiliary frame can contain more rows than the main train set. For now,
            # fit only rows whose SMILES overlap the primary train rows so the feature
            # matrix alignment remains unambiguous.
            aligned = train[["SMILES"]].merge(
                frame[["SMILES", col]],
                on="SMILES",
                how="left",
                suffixes=("", "_aux"),
            )
            y = pd.to_numeric(aligned[col], errors="coerce")
        mask = y.notna().to_numpy()
        if int(mask.sum()) < 100:
            continue
        model = ExtraTreesRegressor(
            n_estimators=350,
            min_samples_leaf=6,
            max_features=0.45,
            random_state=seed,
            n_jobs=-1,
        )
        model.fit(x_train[mask], y.to_numpy(float)[mask])
        safe_name = (
            feature_name.replace(" ", "_")
            .replace("(", "")
            .replace(")", "")
            .replace(".", "")
            .replace("/", "_")
            .replace("-", "_")
        )
        out[safe_name] = model.predict(x_query)
        report.append({"feature": safe_name, "n_labeled": int(mask.sum()), "source_column": col})
    return out.astype(np.float32), report


def _assemble_signal_matrix(
    frames: dict[str, pd.DataFrame],
    *,
    with_3d: bool,
    seed: int,
) -> dict[str, Any]:
    train = frames["train"].copy()
    phase1 = frames["phase1"].copy()
    test = frames["test"].copy()
    baseline = frames["baseline"].copy()

    all_smiles = pd.concat(
        [
            train[["SMILES"]].assign(_block="train"),
            phase1[["SMILES"]].assign(_block="phase1"),
            test[["SMILES"]].assign(_block="test"),
        ],
        ignore_index=True,
    )
    desc, ecfp, fcfp = build_molecular_features(all_smiles["SMILES"], with_3d=with_3d, seed=seed)
    fp = np.hstack([ecfp, fcfp]).astype(np.float32)
    desc_np = _clean_matrix(desc.to_numpy(np.float32))

    n_train = len(train)
    n_phase = len(phase1)
    n_test = len(test)
    sl_train = slice(0, n_train)
    sl_phase = slice(n_train, n_train + n_phase)
    sl_test = slice(n_train + n_phase, n_train + n_phase + n_test)

    y_train = _safe_numeric(train, "pEC50").to_numpy(float)
    knn_phase = train_knn_signal_features(fp[sl_phase], fp[sl_train], y_train)
    knn_test = train_knn_signal_features(fp[sl_test], fp[sl_train], y_train)

    from sklearn.decomposition import TruncatedSVD

    n_components = min(64, max(2, fp[sl_train].shape[0] - 1), fp[sl_train].shape[1] - 1)
    svd = TruncatedSVD(n_components=n_components, random_state=seed)
    svd.fit(fp[sl_train])
    fp_svd_phase = pd.DataFrame(
        svd.transform(fp[sl_phase]).astype(np.float32),
        columns=[f"fp_svd_{i:02d}" for i in range(n_components)],
    )
    fp_svd_test = pd.DataFrame(
        svd.transform(fp[sl_test]).astype(np.float32),
        columns=[f"fp_svd_{i:02d}" for i in range(n_components)],
    )

    x_train_for_aux = np.hstack([desc_np[sl_train], fp[sl_train]]).astype(np.float32)
    x_query_for_aux = np.hstack(
        [
            np.vstack([desc_np[sl_phase], desc_np[sl_test]]),
            np.vstack([fp[sl_phase], fp[sl_test]]),
        ]
    ).astype(np.float32)
    aux, aux_report = _fit_auxiliary_predictions(
        train,
        frames["counter"],
        frames["multitask"],
        x_train_for_aux,
        x_query_for_aux,
        seed=seed,
    )
    aux_phase = aux.iloc[:n_phase].reset_index(drop=True)
    aux_test = aux.iloc[n_phase:].reset_index(drop=True)

    baseline_phase = phase1[["Molecule Name"]].merge(
        baseline[["Molecule Name", "pEC50"]],
        on="Molecule Name",
        how="left",
    )["pEC50"]
    baseline_test = test[["Molecule Name"]].merge(
        baseline[["Molecule Name", "pEC50"]],
        on="Molecule Name",
        how="left",
    )["pEC50"]
    if baseline_phase.isna().any() or baseline_test.isna().any():
        raise ValueError("Baseline prediction file does not cover every Phase 1/test molecule.")

    phase_meta = pd.concat(
        [
            pd.DataFrame(desc_np[sl_phase], columns=desc.columns),
            fp_svd_phase,
            knn_phase.reset_index(drop=True),
            aux_phase,
            pd.DataFrame({"anchor_pEC50": baseline_phase.to_numpy(float)}),
        ],
        axis=1,
    )
    test_meta = pd.concat(
        [
            pd.DataFrame(desc_np[sl_test], columns=desc.columns),
            fp_svd_test,
            knn_test.reset_index(drop=True),
            aux_test,
            pd.DataFrame({"anchor_pEC50": baseline_test.to_numpy(float)}),
        ],
        axis=1,
    )

    return {
        "phase_meta": phase_meta,
        "test_meta": test_meta,
        "phase_fp": fp[sl_phase],
        "test_fp": fp[sl_test],
        "train_fp": fp[sl_train],
        "phase_anchor": baseline_phase.to_numpy(float),
        "test_anchor": baseline_test.to_numpy(float),
        "aux_report": aux_report,
        "feature_columns": list(phase_meta.columns),
    }


def _region_label(y: float) -> str:
    if y < 3.0:
        return "tail_lt_3"
    if y < 4.5:
        return "low_3_4p5"
    if y < 5.5:
        return "mid_4p5_5p5"
    if y < 6.0:
        return "high_5p5_6"
    return "active_ge_6"


def _region_report(y: np.ndarray, anchor: np.ndarray, pred: np.ndarray) -> pd.DataFrame:
    rows = []
    regions = pd.Series([_region_label(v) for v in y])
    for region in ["tail_lt_3", "low_3_4p5", "mid_4p5_5p5", "high_5p5_6", "active_ge_6"]:
        mask = regions.eq(region).to_numpy()
        if not mask.any():
            continue
        rows.append(
            {
                "region": region,
                "n": int(mask.sum()),
                "anchor_mae": mae(y[mask], anchor[mask]),
                "candidate_mae": mae(y[mask], pred[mask]),
                "candidate_bias": float(np.mean(pred[mask] - y[mask])),
            }
        )
    return pd.DataFrame(rows)


def run_structure_assay_experiment(
    root: Path,
    *,
    n_folds: int = 5,
    n_boot: int = 5000,
    seed: int = 20260623,
    with_3d: bool = False,
    output_dir: Path | None = None,
) -> dict[str, Any]:
    _require_modeling_deps()
    root = Path(root)
    output_dir = Path(output_dir or (root / "reports" / "sub040_structure_assay_experiment"))
    output_dir.mkdir(parents=True, exist_ok=True)

    frames = load_frames(root)
    phase1 = frames["phase1"].copy()
    test = frames["test"].copy()
    y = _safe_numeric(phase1, "pEC50").to_numpy(float)
    signals = _assemble_signal_matrix(frames, with_3d=with_3d, seed=seed)
    x_phase = _clean_matrix(signals["phase_meta"].to_numpy(np.float32))
    x_test = _clean_matrix(signals["test_meta"].to_numpy(np.float32))
    anchor = signals["phase_anchor"]
    anchor_test = signals["test_anchor"]

    folds = _stratified_scaffold_folds(phase1, n_folds, seed)
    pd.DataFrame({"Molecule Name": phase1["Molecule Name"], "fold": folds}).to_csv(
        output_dir / "phase1_scaffold_stratified_folds.csv",
        index=False,
    )

    configs = _residual_configs(seed)
    oof = np.full(len(phase1), np.nan, dtype=float)
    fold_rows = []
    inner_rows = []
    chosen_configs: list[ResidualConfig] = []
    for fold in range(n_folds):
        val_idx = np.flatnonzero(folds == fold)
        train_idx = np.flatnonzero(folds != fold)
        cfg, inner = _inner_select_config(x_phase, y, anchor, train_idx, configs, seed=seed + fold)
        inner["outer_fold"] = fold
        inner_rows.append(inner)
        chosen_configs.append(cfg)
        pred_residual = _fit_predict_model(
            cfg.name,
            x_phase[train_idx],
            (y - anchor)[train_idx],
            x_phase[val_idx],
            seed=seed + fold,
        )
        oof[val_idx] = _apply_residual(anchor[val_idx], pred_residual, cfg)
        base_score = _score_prediction(y[val_idx], anchor[val_idx])
        cand_score = _score_prediction(y[val_idx], oof[val_idx])
        fold_rows.append(
            {
                "fold": fold,
                "n": int(len(val_idx)),
                "selected_config": cfg.label,
                "anchor_mae": base_score["mae"],
                "anchor_rae": base_score["rae"],
                "candidate_mae": cand_score["mae"],
                "candidate_rae": cand_score["rae"],
                "improved": bool(cand_score["mae"] < base_score["mae"]),
            }
        )

    if np.isnan(oof).any():
        raise RuntimeError("Nested OOF prediction has missing values.")

    fold_report = pd.DataFrame(fold_rows)
    inner_report = pd.concat(inner_rows, ignore_index=True)
    fold_report.to_csv(output_dir / "fold_metrics.csv", index=False)
    inner_report.to_csv(output_dir / "inner_config_scores.csv", index=False)

    anchor_score = _score_prediction(y, anchor)
    candidate_score = _score_prediction(y, oof)
    exact = exact_fill_count(y, oof)
    ci = bootstrap_paired_ci(y, oof, n_boot=n_boot, seed=seed)
    mae_ci = ci_summary(ci["mae"])
    rae_ci = ci_summary(ci["rae_fixed"])
    corr = float(np.corrcoef(anchor, oof)[0, 1])
    improved_folds = int(fold_report["improved"].sum())
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

    # Select a conservative full-fit configuration by majority vote, then inner score.
    chosen_labels = pd.Series([cfg.label for cfg in chosen_configs])
    majority_label = str(chosen_labels.value_counts().index[0])
    majority_cfg = next(cfg for cfg in configs if cfg.label == majority_label)
    full_residual = _fit_predict_model(
        majority_cfg.name,
        x_phase,
        y - anchor,
        x_test,
        seed=seed,
    )
    corrected_test = _apply_residual(anchor_test, full_residual, majority_cfg)

    oof_submission = test[["SMILES", "Molecule Name"]].copy()
    oof_submission = oof_submission.merge(
        phase1[["Molecule Name"]].assign(_phase1_order=np.arange(len(phase1))),
        on="Molecule Name",
        how="left",
    )
    pred_by_name = pd.Series(corrected_test, index=test["Molecule Name"].astype(str))
    oof_by_name = pd.Series(oof, index=phase1["Molecule Name"].astype(str))
    oof_submission["pEC50"] = [
        oof_by_name.get(name, pred_by_name[name]) for name in oof_submission["Molecule Name"].astype(str)
    ]
    oof_submission = oof_submission[["SMILES", "Molecule Name", "pEC50"]]

    upload_candidate = test[["SMILES", "Molecule Name"]].copy()
    phase1_truth = pd.Series(y, index=phase1["Molecule Name"].astype(str))
    upload_candidate["pEC50"] = [
        phase1_truth.get(name, pred_by_name[name]) for name in upload_candidate["Molecule Name"].astype(str)
    ]

    submissions = root / "submissions"
    oof_path = submissions / OOF_CANDIDATE_FILE
    upload_path = submissions / UPLOAD_CANDIDATE_FILE
    oof_submission.to_csv(oof_path, index=False)
    upload_candidate.to_csv(upload_path, index=False)

    region_report = _region_report(y, anchor, oof)
    region_report.to_csv(output_dir / "region_metrics.csv", index=False)
    pd.DataFrame(signals["aux_report"]).to_csv(output_dir / "auxiliary_feature_report.csv", index=False)

    summary = {
        "decision": decision,
        "baseline_file": BASELINE_FILE,
        "preserved_final_upload_file": FINAL_UPLOAD_FILE,
        "oof_candidate_file": str(oof_path.relative_to(root)),
        "upload_candidate_file": str(upload_path.relative_to(root)),
        "with_3d": bool(with_3d),
        "n_folds": int(n_folds),
        "n_boot": int(n_boot),
        "seed": int(seed),
        "exact_matches_oof": int(exact),
        "anchor": anchor_score,
        "candidate": candidate_score,
        "mae_ci": mae_ci,
        "rae_fixed_ci": rae_ci,
        "folds_improved": improved_folds,
        "anchor_candidate_corr": corr,
        "selected_full_fit_config": majority_cfg.label,
        "feature_count": int(x_phase.shape[1]),
        "feature_columns": signals["feature_columns"],
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

    md = [
        "# Sub-0.40 Structure/Assay Signal Experiment",
        "",
        f"Decision: **{decision}**",
        "",
        "## Headline Metrics",
        "",
        f"- Anchor MAE/RAE: {anchor_score['mae']:.6f} / {anchor_score['rae']:.6f}",
        f"- Candidate OOF MAE/RAE: {candidate_score['mae']:.6f} / {candidate_score['rae']:.6f}",
        f"- Candidate MAE 95% CI: {mae_ci['lo']:.6f} - {mae_ci['hi']:.6f}",
        f"- Candidate RAE 95% CI: {rae_ci['lo']:.6f} - {rae_ci['hi']:.6f}",
        f"- Exact Phase 1 matches in OOF candidate: {exact}",
        f"- Folds improved: {improved_folds}/{n_folds}",
        f"- Anchor/candidate correlation: {corr:.4f}",
        "",
        "## Files",
        "",
        f"- Honest OOF candidate: `{oof_path.relative_to(root)}`",
        f"- Experimental upload candidate: `{upload_path.relative_to(root)}`",
        f"- Preserved current final upload: `submissions/{FINAL_UPLOAD_FILE}`",
        "",
        "The current final upload file is not replaced unless the acceptance gate passes.",
    ]
    (output_dir / "experiment_report.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    return summary
