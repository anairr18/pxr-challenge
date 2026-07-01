from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
import random
from typing import Any

import numpy as np
import pandas as pd

from .metrics import (
    LEADERBOARD_RAE_DENOM,
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
    _descriptor_names,
    _fingerprint_bits,
    _mol_from_smiles,
    _rdkit_descriptors,
    _safe_numeric,
    _scaffold_from_smiles,
    _stratified_scaffold_folds,
)


GRAPH_OOF_FILE = "asymmetric_counter_graph_oof_candidate.csv"
GRAPH_UPLOAD_FILE = "asymmetric_counter_graph_upload_candidate.csv"
GRAPH_CLEAN_UPLOAD_FILE = "asymmetric_counter_graph_clean_upload_candidate.csv"

COMMON_ATOMS = [6, 7, 8, 9, 15, 16, 17, 35, 53, 5, 14]


def _log(message: str) -> None:
    print(f"[asymmetric-graph] {message}", flush=True)


def _require_deps() -> None:
    missing: list[str] = []
    try:
        import rdkit  # noqa: F401
    except Exception:
        missing.append("rdkit")
    try:
        import sklearn  # noqa: F401
    except Exception:
        missing.append("scikit-learn")
    try:
        import torch  # noqa: F401
    except Exception:
        missing.append("torch")
    if missing:
        raise RuntimeError(
            "Missing dependencies: "
            + ", ".join(missing)
            + ". In Colab, run `pip install rdkit scikit-learn torch pandas numpy pyarrow` first."
        )


def _load_required(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def load_frames(root: Path) -> dict[str, pd.DataFrame]:
    root = Path(root)
    data = root / "data"
    submissions = root / "submissions"
    frames = {
        "train": _load_required(data / "pxr-challenge_TRAIN.csv"),
        "phase1": _load_required(data / "phase1_unblinded.csv"),
        "test": _load_required(data / "pxr-challenge_TEST_BLINDED.csv"),
        "baseline": _load_required(submissions / BASELINE_FILE),
    }
    optional = {
        "counter": data / "pxr-challenge_counter-assay_TRAIN.csv",
        "single": data / "pxr-challenge_single_concentration_TRAIN.csv",
        "final": submissions / FINAL_UPLOAD_FILE,
    }
    for key, path in optional.items():
        frames[key] = pd.read_csv(path) if path.exists() else pd.DataFrame()
    return frames


def _find_column(df: pd.DataFrame, required: list[str]) -> str | None:
    lowered = {c: c.lower() for c in df.columns}
    for col, low in lowered.items():
        if all(token.lower() in low for token in required):
            return col
    return None


def _canonical_smiles(smiles: str) -> str:
    from rdkit import Chem

    mol = _mol_from_smiles(smiles)
    if mol is None:
        return str(smiles)
    return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=False)


def _counter_reliability_weights(train: pd.DataFrame, counter: pd.DataFrame) -> pd.DataFrame:
    """Apply the public-report counter-assay selectivity weighting scheme.

    The goal is not to fit Phase 1. It is to down-weight training examples that
    look non-selective against the counter assay and up-weight examples whose
    main PXR activity is separated from the counter-assay confidence interval.
    """

    out = train.copy()
    y = _safe_numeric(out, "pEC50")
    out["_canonical_smiles"] = out["SMILES"].astype(str).map(_canonical_smiles)
    out["assay_weight"] = 0.5
    out["assay_keep"] = np.isfinite(y)
    out["counter_delta"] = np.nan
    if counter.empty:
        return out

    counter = counter.copy()
    counter["_canonical_smiles"] = counter["SMILES"].astype(str).map(_canonical_smiles)
    c_y = _safe_numeric(counter, "pEC50")
    c_low_col = _find_column(counter, ["pEC50", "ci.lower"])
    c_high_col = _find_column(counter, ["pEC50", "ci.upper"])
    c_low = _safe_numeric(counter, c_low_col) if c_low_col else pd.Series(np.nan, index=counter.index)
    c_high = _safe_numeric(counter, c_high_col) if c_high_col else pd.Series(np.nan, index=counter.index)
    counter_summary = (
        pd.DataFrame(
            {
                "_canonical_smiles": counter["_canonical_smiles"],
                "counter_pEC50": c_y,
                "counter_low": c_low,
                "counter_high": c_high,
            }
        )
        .replace([np.inf, -np.inf], np.nan)
        .groupby("_canonical_smiles", as_index=False)
        .median(numeric_only=True)
    )

    low_col = _find_column(out, ["pEC50", "ci.lower"])
    high_col = _find_column(out, ["pEC50", "ci.upper"])
    out_low = _safe_numeric(out, low_col) if low_col else pd.Series(np.nan, index=out.index)
    out_high = _safe_numeric(out, high_col) if high_col else pd.Series(np.nan, index=out.index)

    merged = out[["_canonical_smiles"]].merge(counter_summary, on="_canonical_smiles", how="left")
    delta = y.to_numpy(float) - merged["counter_pEC50"].to_numpy(float)
    out["counter_delta"] = delta
    weights = np.full(len(out), 0.5, dtype=float)
    keep = np.isfinite(y.to_numpy(float))

    has_counter = np.isfinite(merged["counter_pEC50"].to_numpy(float))
    negative_delta = has_counter & np.isfinite(delta) & (delta < 0)
    keep[negative_delta] = False
    weights[negative_delta] = 0.0

    train_low = out_low.to_numpy(float)
    train_high = out_high.to_numpy(float)
    counter_low = merged["counter_low"].to_numpy(float)
    counter_high = merged["counter_high"].to_numpy(float)
    has_ci = (
        has_counter
        & np.isfinite(train_low)
        & np.isfinite(train_high)
        & np.isfinite(counter_low)
        & np.isfinite(counter_high)
    )
    ci_separated = has_ci & (train_low > counter_high)
    ci_overlap = has_ci & ~ci_separated
    weights[ci_separated] = 1.0
    weights[ci_overlap & (delta <= 1.0)] = 0.2
    weights[ci_overlap & (delta > 1.0)] = 0.4

    no_ci = has_counter & ~has_ci & np.isfinite(delta) & ~negative_delta
    weights[no_ci & (delta >= 1.5)] = 1.0
    weights[no_ci & (delta >= 0.5) & (delta < 1.5)] = 0.4
    weights[no_ci & (delta < 0.5)] = 0.2

    out["assay_weight"] = weights
    out["assay_keep"] = keep
    return out


def _single_concentration_pseudo_labels(
    single: pd.DataFrame,
    *,
    excluded_smiles: set[str],
) -> pd.DataFrame:
    if single.empty or "SMILES" not in single.columns:
        return pd.DataFrame(columns=["SMILES", "pEC50", "sample_weight", "source"])

    df = single.copy()
    log2_col = "log2_fc_estimate" if "log2_fc_estimate" in df.columns else _find_column(df, ["log2", "fc"])
    fdr_col = "fdr_bh" if "fdr_bh" in df.columns else _find_column(df, ["fdr"])
    conc_col = "concentration_M" if "concentration_M" in df.columns else _find_column(df, ["concentration"])
    if not log2_col or not fdr_col or not conc_col:
        return pd.DataFrame(columns=["SMILES", "pEC50", "sample_weight", "source"])

    df["log2_fc"] = _safe_numeric(df, log2_col)
    df["fdr"] = _safe_numeric(df, fdr_col)
    df["concentration"] = _safe_numeric(df, conc_col)
    df["_canonical_smiles"] = df["SMILES"].astype(str).map(_canonical_smiles)
    mask = (
        np.isfinite(df["log2_fc"])
        & np.isfinite(df["fdr"])
        & np.isfinite(df["concentration"])
        & df["log2_fc"].gt(0.75)
        & df["fdr"].lt(0.10)
        & df["concentration"].between(20e-6, 50e-6)
        & ~df["_canonical_smiles"].isin(excluded_smiles)
    )
    kept = df.loc[mask].copy()
    if kept.empty:
        return pd.DataFrame(columns=["SMILES", "pEC50", "sample_weight", "source"])

    grouped = (
        kept.groupby("_canonical_smiles", as_index=False)
        .agg(SMILES=("SMILES", "first"), log2_fc=("log2_fc", "median"), fdr=("fdr", "min"))
        .reset_index(drop=True)
    )
    grouped["pEC50"] = 4.48 + 0.15 * np.clip(grouped["log2_fc"].to_numpy(float) - 0.75, 0.0, 4.0)
    grouped["sample_weight"] = np.where(grouped["log2_fc"].to_numpy(float) > 1.5, 0.25, 0.15)
    grouped["source"] = "pseudo_single_concentration"
    return grouped[["SMILES", "pEC50", "sample_weight", "source"]]


@dataclass(frozen=True)
class GraphSample:
    smiles: str
    y: float
    weight: float
    source: str


def _make_samples(
    train: pd.DataFrame,
    counter: pd.DataFrame,
    single: pd.DataFrame,
    phase_train: pd.DataFrame,
    phase1: pd.DataFrame,
    test: pd.DataFrame,
    *,
    phase_weight: float,
) -> list[GraphSample]:
    curated = _counter_reliability_weights(train, counter)
    y_train = _safe_numeric(curated, "pEC50")
    samples: list[GraphSample] = []
    for row_idx, row in curated.iterrows():
        if not bool(row.get("assay_keep", False)):
            continue
        y = float(y_train.loc[row_idx])
        if not np.isfinite(y):
            continue
        samples.append(
            GraphSample(
                smiles=str(row["SMILES"]),
                y=y,
                weight=float(np.clip(row.get("assay_weight", 0.5), 0.05, 1.0)),
                source="curve_fit_counter_weighted",
            )
        )

    excluded = set(train["SMILES"].astype(str).map(_canonical_smiles))
    excluded.update(phase1["SMILES"].astype(str).map(_canonical_smiles))
    excluded.update(test["SMILES"].astype(str).map(_canonical_smiles))
    pseudo = _single_concentration_pseudo_labels(single, excluded_smiles=excluded)
    for _, row in pseudo.iterrows():
        samples.append(
            GraphSample(
                smiles=str(row["SMILES"]),
                y=float(row["pEC50"]),
                weight=float(row["sample_weight"]),
                source="pseudo_single_concentration",
            )
        )

    y_phase = _safe_numeric(phase_train, "pEC50")
    for row_idx, row in phase_train.iterrows():
        y = float(y_phase.loc[row_idx])
        if np.isfinite(y):
            samples.append(
                GraphSample(
                    smiles=str(row["SMILES"]),
                    y=y,
                    weight=phase_weight,
                    source="phase1_fold_train",
                )
            )
    return samples


def _one_hot(value: int, choices: list[int]) -> list[float]:
    return [1.0 if value == choice else 0.0 for choice in choices] + [0.0 if value in choices else 1.0]


def _atom_features(atom: Any) -> list[float]:
    from rdkit import Chem

    hybridization = atom.GetHybridization()
    hyb_choices = [
        Chem.rdchem.HybridizationType.SP,
        Chem.rdchem.HybridizationType.SP2,
        Chem.rdchem.HybridizationType.SP3,
        Chem.rdchem.HybridizationType.SP3D,
    ]
    degree = min(int(atom.GetTotalDegree()), 5)
    hydrogens = min(int(atom.GetTotalNumHs()), 4)
    charge = float(max(-2, min(2, atom.GetFormalCharge()))) / 2.0
    mass = float(atom.GetMass()) / 200.0
    return (
        _one_hot(atom.GetAtomicNum(), COMMON_ATOMS)
        + _one_hot(degree, [0, 1, 2, 3, 4, 5])
        + _one_hot(hydrogens, [0, 1, 2, 3, 4])
        + [1.0 if hybridization == h else 0.0 for h in hyb_choices]
        + [
            1.0 if atom.GetIsAromatic() else 0.0,
            1.0 if atom.IsInRing() else 0.0,
            charge,
            mass,
        ]
    )


def _bond_features(bond: Any) -> list[float]:
    from rdkit import Chem

    btype = bond.GetBondType()
    return [
        1.0 if btype == Chem.rdchem.BondType.SINGLE else 0.0,
        1.0 if btype == Chem.rdchem.BondType.DOUBLE else 0.0,
        1.0 if btype == Chem.rdchem.BondType.TRIPLE else 0.0,
        1.0 if btype == Chem.rdchem.BondType.AROMATIC else 0.0,
        1.0 if bond.GetIsConjugated() else 0.0,
        1.0 if bond.IsInRing() else 0.0,
    ]


def _graph_from_smiles(smiles: str) -> dict[str, np.ndarray]:
    mol = _mol_from_smiles(smiles)
    if mol is None or mol.GetNumAtoms() == 0:
        atom_x = np.zeros((1, len(_atom_features_dummy())), dtype=np.float32)
        edge_index = np.zeros((2, 0), dtype=np.int64)
        edge_attr = np.zeros((0, 6), dtype=np.float32)
        desc = np.zeros((len(_descriptor_names()),), dtype=np.float32)
        return {"atom_x": atom_x, "edge_index": edge_index, "edge_attr": edge_attr, "desc": desc}

    atom_x = np.asarray([_atom_features(atom) for atom in mol.GetAtoms()], dtype=np.float32)
    edges: list[tuple[int, int]] = []
    bond_rows: list[list[float]] = []
    for bond in mol.GetBonds():
        i = int(bond.GetBeginAtomIdx())
        j = int(bond.GetEndAtomIdx())
        feat = _bond_features(bond)
        edges.append((i, j))
        edges.append((j, i))
        bond_rows.append(feat)
        bond_rows.append(feat)
    if edges:
        edge_index = np.asarray(edges, dtype=np.int64).T
        edge_attr = np.asarray(bond_rows, dtype=np.float32)
    else:
        edge_index = np.zeros((2, 0), dtype=np.int64)
        edge_attr = np.zeros((0, 6), dtype=np.float32)
    desc = np.asarray(_rdkit_descriptors(smiles), dtype=np.float32)
    desc = np.nan_to_num(desc, nan=0.0, posinf=0.0, neginf=0.0)
    return {"atom_x": atom_x, "edge_index": edge_index, "edge_attr": edge_attr, "desc": desc}


def _atom_features_dummy() -> list[float]:
    return [0.0] * (len(COMMON_ATOMS) + 1 + 7 + 6 + 4 + 4)


def _build_graph_cache(smiles_values: list[str]) -> dict[str, dict[str, np.ndarray]]:
    cache: dict[str, dict[str, np.ndarray]] = {}
    for idx, smiles in enumerate(dict.fromkeys(map(str, smiles_values))):
        cache[smiles] = _graph_from_smiles(smiles)
        if (idx + 1) % 500 == 0:
            _log(f"graph featurized {idx + 1} molecules")
    return cache


class _GraphDataset:
    def __init__(
        self,
        samples: list[GraphSample],
        graph_cache: dict[str, dict[str, np.ndarray]],
        desc_mean: np.ndarray,
        desc_std: np.ndarray,
    ) -> None:
        self.samples = samples
        self.graph_cache = graph_cache
        self.desc_mean = desc_mean.astype(np.float32)
        self.desc_std = desc_std.astype(np.float32)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        sample = self.samples[idx]
        graph = self.graph_cache[sample.smiles]
        desc = (graph["desc"] - self.desc_mean) / self.desc_std
        return {
            "atom_x": graph["atom_x"],
            "edge_index": graph["edge_index"],
            "edge_attr": graph["edge_attr"],
            "desc": desc.astype(np.float32),
            "y": np.float32(sample.y),
            "weight": np.float32(sample.weight),
            "clf_mask": np.float32(0.0 if sample.source == "pseudo_single_concentration" else 1.0),
        }


def _collate_graph_batch(batch: list[dict[str, Any]]) -> dict[str, Any]:
    import torch

    atom_blocks = []
    edge_indices = []
    edge_attrs = []
    batch_index = []
    desc_rows = []
    ys = []
    weights = []
    clf_masks = []
    atom_offset = 0
    for graph_idx, item in enumerate(batch):
        atom_x = item["atom_x"]
        n_atoms = atom_x.shape[0]
        atom_blocks.append(atom_x)
        batch_index.append(np.full((n_atoms,), graph_idx, dtype=np.int64))
        if item["edge_index"].shape[1] > 0:
            edge_indices.append(item["edge_index"] + atom_offset)
            edge_attrs.append(item["edge_attr"])
        atom_offset += n_atoms
        desc_rows.append(item["desc"])
        ys.append(item["y"])
        weights.append(item["weight"])
        clf_masks.append(item["clf_mask"])

    atom_x = torch.tensor(np.vstack(atom_blocks), dtype=torch.float32)
    if edge_indices:
        edge_index = torch.tensor(np.hstack(edge_indices), dtype=torch.long)
        edge_attr = torch.tensor(np.vstack(edge_attrs), dtype=torch.float32)
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_attr = torch.zeros((0, 6), dtype=torch.float32)
    return {
        "atom_x": atom_x,
        "edge_index": edge_index,
        "edge_attr": edge_attr,
        "batch_index": torch.tensor(np.concatenate(batch_index), dtype=torch.long),
        "desc": torch.tensor(np.vstack(desc_rows), dtype=torch.float32),
        "y": torch.tensor(np.asarray(ys), dtype=torch.float32),
        "weight": torch.tensor(np.asarray(weights), dtype=torch.float32),
        "clf_mask": torch.tensor(np.asarray(clf_masks), dtype=torch.float32),
    }


def _make_model_class(torch: Any):
    class AsymmetricGraphNet(torch.nn.Module):
        def __init__(self, atom_dim: int, bond_dim: int, desc_dim: int, hidden_dim: int, depth: int) -> None:
            super().__init__()
            self.depth = int(depth)
            self.atom_encoder = torch.nn.Sequential(
                torch.nn.Linear(atom_dim, hidden_dim),
                torch.nn.LayerNorm(hidden_dim),
                torch.nn.SiLU(),
            )
            self.bond_encoder = torch.nn.Sequential(
                torch.nn.Linear(bond_dim, hidden_dim),
                torch.nn.SiLU(),
            )
            self.message = torch.nn.ModuleList(
                [
                    torch.nn.Sequential(
                        torch.nn.Linear(hidden_dim * 2, hidden_dim),
                        torch.nn.SiLU(),
                        torch.nn.Linear(hidden_dim, hidden_dim),
                    )
                    for _ in range(depth)
                ]
            )
            self.update = torch.nn.ModuleList(
                [
                    torch.nn.Sequential(
                        torch.nn.Linear(hidden_dim * 2, hidden_dim),
                        torch.nn.LayerNorm(hidden_dim),
                        torch.nn.SiLU(),
                    )
                    for _ in range(depth)
                ]
            )
            self.desc_encoder = torch.nn.Sequential(
                torch.nn.Linear(desc_dim, hidden_dim // 2),
                torch.nn.SiLU(),
                torch.nn.Dropout(0.05),
            )
            self.trunk = torch.nn.Sequential(
                torch.nn.Linear(hidden_dim + hidden_dim // 2, hidden_dim),
                torch.nn.LayerNorm(hidden_dim),
                torch.nn.SiLU(),
                torch.nn.Dropout(0.08),
                torch.nn.Linear(hidden_dim, hidden_dim // 2),
                torch.nn.SiLU(),
            )
            self.regression_head = torch.nn.Linear(hidden_dim // 2, 1)
            self.active_head = torch.nn.Linear(hidden_dim // 2, 1)

        def forward(self, batch: dict[str, Any]) -> tuple[Any, Any]:
            atom_x = batch["atom_x"]
            edge_index = batch["edge_index"]
            edge_attr = batch["edge_attr"]
            batch_index = batch["batch_index"]
            desc = batch["desc"]

            h = self.atom_encoder(atom_x)
            if edge_index.numel() > 0:
                bond_h = self.bond_encoder(edge_attr)
                src = edge_index[0]
                dst = edge_index[1]
                for msg_layer, upd_layer in zip(self.message, self.update):
                    msg = msg_layer(torch.cat([h[src], bond_h], dim=1))
                    agg = torch.zeros_like(h)
                    agg.index_add_(0, dst, msg)
                    h = h + upd_layer(torch.cat([h, agg], dim=1))

            n_graphs = int(batch_index.max().item()) + 1 if batch_index.numel() else desc.shape[0]
            pooled = torch.zeros((n_graphs, h.shape[1]), dtype=h.dtype, device=h.device)
            pooled.index_add_(0, batch_index, h)
            counts = torch.bincount(batch_index, minlength=n_graphs).float().clamp_min(1.0).to(h.device)
            pooled = pooled / counts.unsqueeze(1)

            desc_h = self.desc_encoder(desc)
            z = self.trunk(torch.cat([pooled, desc_h], dim=1))
            return self.regression_head(z).squeeze(1), self.active_head(z).squeeze(1)

    return AsymmetricGraphNet


def _move_to_device(batch: dict[str, Any], device: Any) -> dict[str, Any]:
    return {key: value.to(device) for key, value in batch.items()}


def _weighted_losses(
    pred: Any,
    active_logit: Any,
    y: Any,
    weight: Any,
    clf_mask: Any,
    *,
    active_cutoff: float,
    clf_weight: float,
):
    import torch

    reg = torch.nn.functional.smooth_l1_loss(pred, y, beta=0.35, reduction="none")
    reg_loss = (reg * weight).sum() / weight.sum().clamp_min(1e-6)

    active_label = (y >= active_cutoff).float()
    class_weights = torch.where(active_label > 0.5, torch.full_like(weight, 0.25), torch.ones_like(weight))
    class_weights = class_weights * weight * clf_mask
    if torch.sum(class_weights) > 1e-6:
        bce = torch.nn.functional.binary_cross_entropy_with_logits(active_logit, active_label, reduction="none")
        clf_loss = (bce * class_weights).sum() / class_weights.sum().clamp_min(1e-6)
    else:
        clf_loss = torch.zeros((), dtype=reg_loss.dtype, device=reg_loss.device)
    return reg_loss + clf_weight * clf_loss, reg_loss.detach(), clf_loss.detach()


def _split_internal_validation(samples: list[GraphSample], seed: int) -> tuple[list[GraphSample], list[GraphSample]]:
    labeled = [i for i, s in enumerate(samples) if s.source != "pseudo_single_concentration"]
    rng = random.Random(seed)
    rng.shuffle(labeled)
    n_val = max(32, int(0.10 * len(labeled)))
    val_set = set(labeled[: min(n_val, max(1, len(labeled) // 5))])
    train_samples = [s for i, s in enumerate(samples) if i not in val_set]
    val_samples = [s for i, s in enumerate(samples) if i in val_set]
    return train_samples, val_samples


def _train_graph_predict(
    train_samples: list[GraphSample],
    query_smiles: list[str],
    *,
    seed: int,
    epochs: int,
    batch_size: int,
    hidden_dim: int,
    depth: int,
    learning_rate: float,
    weight_decay: float,
    patience: int,
    clf_weight: float,
    active_cutoff: float,
    device_name: str,
) -> tuple[np.ndarray, dict[str, Any]]:
    import torch
    from torch.utils.data import DataLoader

    if not train_samples:
        raise RuntimeError("No graph training samples were constructed.")

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    all_smiles = [s.smiles for s in train_samples] + list(map(str, query_smiles))
    graph_cache = _build_graph_cache(all_smiles)
    desc_stack = np.vstack([graph_cache[s.smiles]["desc"] for s in train_samples]).astype(np.float32)
    desc_mean = np.nanmean(desc_stack, axis=0).astype(np.float32)
    desc_std = np.nanstd(desc_stack, axis=0).astype(np.float32)
    desc_std = np.where(desc_std < 1e-6, 1.0, desc_std).astype(np.float32)

    fit_samples, val_samples = _split_internal_validation(train_samples, seed)
    train_ds = _GraphDataset(fit_samples, graph_cache, desc_mean, desc_std)
    val_ds = _GraphDataset(val_samples, graph_cache, desc_mean, desc_std)
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=_collate_graph_batch,
        num_workers=0,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=_collate_graph_batch,
        num_workers=0,
    )

    if device_name == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_name)
    model_cls = _make_model_class(torch)
    atom_dim = graph_cache[all_smiles[0]]["atom_x"].shape[1]
    bond_dim = 6
    desc_dim = len(_descriptor_names())
    model = model_cls(atom_dim, bond_dim, desc_dim, hidden_dim, depth).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(epochs, 1))

    best_state = None
    best_val = math.inf
    stale = 0
    best_epoch = 0
    for epoch in range(1, epochs + 1):
        model.train()
        train_losses = []
        for batch in train_loader:
            batch = _move_to_device(batch, device)
            optimizer.zero_grad(set_to_none=True)
            pred, active_logit = model(batch)
            loss, _, _ = _weighted_losses(
                pred,
                active_logit,
                batch["y"],
                batch["weight"],
                batch["clf_mask"],
                active_cutoff=active_cutoff,
                clf_weight=clf_weight,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 3.0)
            optimizer.step()
            train_losses.append(float(loss.detach().cpu()))
        scheduler.step()

        model.eval()
        val_losses = []
        val_abs = []
        with torch.no_grad():
            for batch in val_loader:
                batch = _move_to_device(batch, device)
                pred, active_logit = model(batch)
                loss, _, _ = _weighted_losses(
                    pred,
                    active_logit,
                    batch["y"],
                    batch["weight"],
                    batch["clf_mask"],
                    active_cutoff=active_cutoff,
                    clf_weight=clf_weight,
                )
                val_losses.append(float(loss.detach().cpu()))
                val_abs.extend(torch.abs(pred - batch["y"]).detach().cpu().numpy().tolist())
        val_loss = float(np.mean(val_losses)) if val_losses else float(np.mean(train_losses))
        val_mae = float(np.mean(val_abs)) if val_abs else float("nan")
        if epoch == 1 or epoch % 5 == 0 or epoch == epochs:
            _log(f"epoch {epoch:03d}/{epochs} train_loss={np.mean(train_losses):.4f} val_loss={val_loss:.4f} val_mae={val_mae:.4f}")

        if val_loss < best_val - 1e-5:
            best_val = val_loss
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                _log(f"early stopping at epoch {epoch}; best_epoch={best_epoch}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()

    query_samples = [GraphSample(smiles=str(s), y=0.0, weight=1.0, source="query") for s in query_smiles]
    query_ds = _GraphDataset(query_samples, graph_cache, desc_mean, desc_std)
    query_loader = DataLoader(query_ds, batch_size=batch_size, shuffle=False, collate_fn=_collate_graph_batch, num_workers=0)
    preds: list[float] = []
    active_probs: list[float] = []
    with torch.no_grad():
        for batch in query_loader:
            batch = _move_to_device(batch, device)
            pred, active_logit = model(batch)
            preds.extend(pred.detach().cpu().numpy().astype(float).tolist())
            active_probs.extend(torch.sigmoid(active_logit).detach().cpu().numpy().astype(float).tolist())

    info = {
        "best_epoch": int(best_epoch),
        "best_internal_val_loss": float(best_val),
        "n_train_samples": int(len(train_samples)),
        "n_fit_samples": int(len(fit_samples)),
        "n_internal_val_samples": int(len(val_samples)),
        "mean_active_probability": float(np.mean(active_probs)),
    }
    return np.asarray(preds, dtype=float), info


def _map_predictions(frame: pd.DataFrame, prediction_frame: pd.DataFrame) -> np.ndarray:
    mapping = dict(
        zip(
            prediction_frame["Molecule Name"].astype(str),
            _safe_numeric(prediction_frame, "pEC50").to_numpy(float),
        )
    )
    return frame["Molecule Name"].astype(str).map(mapping).to_numpy(float)


def _score_prediction(y_true: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    m = mae(y_true, pred)
    return {"mae": m, "rae": rae_from_mae(m), "spearman": spearman_corr(y_true, pred)}


def _blend(anchor: np.ndarray, graph_pred: np.ndarray, graph_weight: float) -> np.ndarray:
    pred = (1.0 - graph_weight) * np.asarray(anchor, float) + graph_weight * np.asarray(graph_pred, float)
    return np.clip(pred, 1.5, 7.5)


def _write_candidate_csv(
    test: pd.DataFrame,
    phase1: pd.DataFrame,
    pred_test: np.ndarray,
    pred_phase: np.ndarray,
    path: Path,
    *,
    exact_fill_phase1: bool,
) -> None:
    out = test[["SMILES", "Molecule Name"]].copy()
    out["pEC50"] = np.asarray(pred_test, dtype=float)
    phase_map = dict(zip(phase1["Molecule Name"].astype(str), np.asarray(pred_phase, dtype=float)))
    if exact_fill_phase1:
        phase_map = dict(zip(phase1["Molecule Name"].astype(str), _safe_numeric(phase1, "pEC50").to_numpy(float)))
    mask = out["Molecule Name"].astype(str).isin(phase_map)
    out.loc[mask, "pEC50"] = out.loc[mask, "Molecule Name"].astype(str).map(phase_map).to_numpy(float)
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False)


def _region_table(y: np.ndarray, anchor: np.ndarray, candidate: np.ndarray) -> pd.DataFrame:
    rows = []
    bins = [
        ("tail_lt_3", -np.inf, 3.0),
        ("low_3_4p5", 3.0, 4.5),
        ("mid_4p5_5p5", 4.5, 5.5),
        ("high_5p5_6", 5.5, 6.0),
        ("active_ge_6", 6.0, np.inf),
    ]
    for name, lo, hi in bins:
        mask = (y >= lo) & (y < hi)
        if not mask.any():
            continue
        rows.append(
            {
                "region": name,
                "n": int(mask.sum()),
                "anchor_mae": mae(y[mask], anchor[mask]),
                "candidate_mae": mae(y[mask], candidate[mask]),
                "candidate_bias": float(np.mean(candidate[mask] - y[mask])),
            }
        )
    return pd.DataFrame(rows)


def _safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 2:
        return float("nan")
    return float(np.corrcoef(np.asarray(a, float), np.asarray(b, float))[0, 1])


def _choose_decision(
    anchor_score: dict[str, float],
    candidate_score: dict[str, float],
    folds_improved: int,
    exact_matches: int,
    ci: dict[str, dict[str, float]],
) -> str:
    if exact_matches != 0:
        return "INVALID_EXACT_MATCHES_IN_OOF"
    if (
        candidate_score["mae"] < 0.400
        and candidate_score["rae"] < 0.520
        and folds_improved >= 4
        and ci["mae"]["hi"] < anchor_score["mae"]
    ):
        return "SUB040_CREDIBLE_REVIEW_FOR_REPLACEMENT"
    if (
        candidate_score["mae"] < 0.410
        and candidate_score["rae"] < 0.540
        and folds_improved >= 4
        and ci["mae"]["hi"] < anchor_score["mae"]
    ):
        return "REAL_IMPROVEMENT_REVIEW_BEFORE_REPLACEMENT"
    if candidate_score["mae"] < anchor_score["mae"] and folds_improved >= 4:
        return "EXPLORATORY_REVIEW_BUT_KEEP_FINAL_BY_DEFAULT"
    return "EXPLORATORY_DO_NOT_REPLACE_FINAL"


def run_asymmetric_graph_experiment(
    root: Path,
    *,
    n_folds: int = 5,
    n_boot: int = 5000,
    seed: int = 20260630,
    epochs: int = 80,
    batch_size: int = 128,
    hidden_dim: int = 192,
    depth: int = 4,
    graph_weight: float = 0.12,
    phase_weight: float = 0.75,
    learning_rate: float = 3e-4,
    weight_decay: float = 1e-4,
    patience: int = 12,
    clf_weight: float = 0.20,
    active_cutoff: float = 4.0,
    device: str = "auto",
    smoke: bool = False,
    output_dir: Path | None = None,
) -> dict[str, Any]:
    _require_deps()
    if smoke:
        epochs = min(epochs, 8)
        n_boot = min(n_boot, 500)
        hidden_dim = min(hidden_dim, 96)
        patience = min(patience, 4)

    root = Path(root)
    output_dir = Path(output_dir or (root / "reports" / "asymmetric_graph_experiment"))
    output_dir.mkdir(parents=True, exist_ok=True)
    submissions = root / "submissions"
    submissions.mkdir(parents=True, exist_ok=True)

    frames = load_frames(root)
    train = frames["train"].copy()
    phase1 = frames["phase1"].copy()
    test = frames["test"].copy()
    counter = frames["counter"].copy()
    single = frames["single"].copy()
    baseline = frames["baseline"].copy()

    y = _safe_numeric(phase1, "pEC50").to_numpy(float)
    anchor_phase = _map_predictions(phase1, baseline)
    anchor_test = _map_predictions(test, baseline)
    if not np.isfinite(anchor_phase).all() or not np.isfinite(anchor_test).all():
        raise RuntimeError(f"{BASELINE_FILE} does not cover every Phase 1/test molecule.")

    _log(
        "start "
        f"folds={n_folds} epochs={epochs} graph_weight={graph_weight} "
        f"train={len(train)} phase1={len(phase1)} test={len(test)}"
    )
    folds = _stratified_scaffold_folds(phase1, n_folds, seed)
    pd.DataFrame({"Molecule Name": phase1["Molecule Name"], "fold": folds}).to_csv(
        output_dir / "phase1_scaffold_stratified_folds.csv",
        index=False,
    )

    graph_oof = np.full(len(phase1), np.nan, dtype=float)
    candidate_oof = np.full(len(phase1), np.nan, dtype=float)
    fold_rows: list[dict[str, Any]] = []
    model_rows: list[dict[str, Any]] = []

    for fold in range(n_folds):
        val_idx = np.where(folds == fold)[0]
        train_idx = np.where(folds != fold)[0]
        phase_train = phase1.iloc[train_idx].reset_index(drop=True)
        phase_val = phase1.iloc[val_idx].reset_index(drop=True)
        samples = _make_samples(
            train,
            counter,
            single,
            phase_train,
            phase1,
            test,
            phase_weight=phase_weight,
        )
        _log(f"outer fold {fold + 1}/{n_folds}: samples={len(samples)} val={len(phase_val)}")
        graph_pred, info = _train_graph_predict(
            samples,
            phase_val["SMILES"].astype(str).tolist(),
            seed=seed + fold * 101,
            epochs=epochs,
            batch_size=batch_size,
            hidden_dim=hidden_dim,
            depth=depth,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            patience=patience,
            clf_weight=clf_weight,
            active_cutoff=active_cutoff,
            device_name=device,
        )
        pred = _blend(anchor_phase[val_idx], graph_pred, graph_weight)
        graph_oof[val_idx] = graph_pred
        candidate_oof[val_idx] = pred
        anchor_metrics = _score_prediction(y[val_idx], anchor_phase[val_idx])
        candidate_metrics = _score_prediction(y[val_idx], pred)
        fold_rows.append(
            {
                "fold": fold,
                "n": int(len(val_idx)),
                "anchor_mae": anchor_metrics["mae"],
                "anchor_rae": anchor_metrics["rae"],
                "candidate_mae": candidate_metrics["mae"],
                "candidate_rae": candidate_metrics["rae"],
                "graph_raw_mae": mae(y[val_idx], graph_pred),
                "improved": bool(candidate_metrics["mae"] < anchor_metrics["mae"]),
                "best_epoch": info["best_epoch"],
                "best_internal_val_loss": info["best_internal_val_loss"],
                "n_train_samples": info["n_train_samples"],
            }
        )
        model_rows.append({"fold": fold, **info})
        _log(
            f"fold {fold + 1}: anchor_mae={anchor_metrics['mae']:.6f} "
            f"candidate_mae={candidate_metrics['mae']:.6f} graph_raw_mae={mae(y[val_idx], graph_pred):.6f}"
        )

    if not np.isfinite(candidate_oof).all():
        raise RuntimeError("OOF graph prediction contains missing values.")

    _log("training final graph on train + all revealed Phase 1 labels")
    final_samples = _make_samples(train, counter, single, phase1, phase1, test, phase_weight=phase_weight)
    graph_test, final_info = _train_graph_predict(
        final_samples,
        test["SMILES"].astype(str).tolist(),
        seed=seed + 9001,
        epochs=epochs,
        batch_size=batch_size,
        hidden_dim=hidden_dim,
        depth=depth,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        patience=patience,
        clf_weight=clf_weight,
        active_cutoff=active_cutoff,
        device_name=device,
    )
    candidate_test = _blend(anchor_test, graph_test, graph_weight)

    anchor_score = _score_prediction(y, anchor_phase)
    graph_score = _score_prediction(y, graph_oof)
    candidate_score = _score_prediction(y, candidate_oof)
    folds_improved = int(sum(row["improved"] for row in fold_rows))
    exact_matches = exact_fill_count(y, candidate_oof)
    boot = bootstrap_paired_ci(y, candidate_oof, n_boot=n_boot, seed=seed, include_spearman=True)
    ci = {
        "mae": ci_summary(boot["mae"]),
        "rae_fixed": ci_summary(boot["rae_fixed"]),
        "rae_resampled": ci_summary(boot["rae_resampled"]),
        "spearman": ci_summary(boot["spearman"]),
    }
    decision = _choose_decision(anchor_score, candidate_score, folds_improved, exact_matches, ci)

    oof_path = submissions / GRAPH_OOF_FILE
    upload_path = submissions / GRAPH_UPLOAD_FILE
    clean_upload_path = submissions / GRAPH_CLEAN_UPLOAD_FILE
    _write_candidate_csv(test, phase1, candidate_test, candidate_oof, oof_path, exact_fill_phase1=False)
    _write_candidate_csv(test, phase1, candidate_test, candidate_oof, upload_path, exact_fill_phase1=True)
    _write_candidate_csv(test, phase1, candidate_test, candidate_oof, clean_upload_path, exact_fill_phase1=False)

    fold_df = pd.DataFrame(fold_rows)
    model_df = pd.DataFrame(model_rows)
    region_df = _region_table(y, anchor_phase, candidate_oof)
    fold_df.to_csv(output_dir / "fold_metrics.csv", index=False)
    model_df.to_csv(output_dir / "model_training_info.csv", index=False)
    region_df.to_csv(output_dir / "region_metrics.csv", index=False)

    summary = {
        "decision": decision,
        "anchor": anchor_score,
        "graph_raw_oof": graph_score,
        "candidate": candidate_score,
        "mae_ci": ci["mae"],
        "rae_fixed_ci": ci["rae_fixed"],
        "rae_resampled_ci": ci["rae_resampled"],
        "spearman_ci": ci["spearman"],
        "folds_improved": folds_improved,
        "exact_matches_oof": exact_matches,
        "anchor_candidate_corr": _safe_corr(anchor_phase, candidate_oof),
        "anchor_graph_corr": _safe_corr(anchor_phase, graph_oof),
        "graph_weight": float(graph_weight),
        "phase_weight": float(phase_weight),
        "epochs": int(epochs),
        "hidden_dim": int(hidden_dim),
        "depth": int(depth),
        "clf_weight": float(clf_weight),
        "active_cutoff": float(active_cutoff),
        "final_training_info": final_info,
        "files": {
            "honest_oof_candidate": str(oof_path),
            "exact_filled_upload_candidate": str(upload_path),
            "clean_upload_candidate": str(clean_upload_path),
            "fold_metrics": str(output_dir / "fold_metrics.csv"),
            "region_metrics": str(output_dir / "region_metrics.csv"),
        },
    }
    (output_dir / "experiment_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    report = [
        "# Asymmetric Counter-Weighted Graph Experiment",
        "",
        f"Decision: **{decision}**",
        "",
        "## Metrics",
        f"- Anchor MAE/RAE: {anchor_score['mae']:.6f} / {anchor_score['rae']:.6f}",
        f"- Graph raw OOF MAE/RAE: {graph_score['mae']:.6f} / {graph_score['rae']:.6f}",
        f"- Candidate OOF MAE/RAE: {candidate_score['mae']:.6f} / {candidate_score['rae']:.6f}",
        f"- Candidate MAE 95% CI: {ci['mae']['lo']:.6f} - {ci['mae']['hi']:.6f}",
        f"- Candidate RAE fixed 95% CI: {ci['rae_fixed']['lo']:.6f} - {ci['rae_fixed']['hi']:.6f}",
        f"- Candidate RAE resampled 95% CI: {ci['rae_resampled']['lo']:.6f} - {ci['rae_resampled']['hi']:.6f}",
        f"- Folds improved: {folds_improved}/{n_folds}",
        f"- Anchor/candidate correlation: {summary['anchor_candidate_corr']:.4f}",
        f"- Exact Phase 1 matches in OOF: {exact_matches}",
        "",
        "## Files",
        f"- Honest OOF candidate: `{oof_path}`",
        f"- Exact-filled upload candidate: `{upload_path}`",
        f"- Clean upload candidate: `{clean_upload_path}`",
        "",
        "Keep the locked final submission unless the decision is replacement-grade.",
    ]
    (output_dir / "experiment_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    _log(f"done decision={decision} candidate_mae={candidate_score['mae']:.6f}")
    return summary
