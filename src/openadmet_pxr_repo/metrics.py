from __future__ import annotations

import numpy as np

LEADERBOARD_RAE_DENOM = 0.7576
REGION_BINS = (
    ("tail_lt_3", -np.inf, 3.0),
    ("low_3_4p5", 3.0, 4.5),
    ("mid_4p5_5p5", 4.5, 5.5),
    ("high_5p5_6", 5.5, 6.0),
    ("active_ge_6", 6.0, np.inf),
)


def mae(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, float)
    y_pred = np.asarray(y_pred, float)
    return float(np.mean(np.abs(y_true - y_pred)))


def phase1_denominator(y_true) -> float:
    y_true = np.asarray(y_true, float)
    return float(np.mean(np.abs(y_true - np.mean(y_true))))


def rae_from_mae(mae_value: float, denom: float = LEADERBOARD_RAE_DENOM) -> float:
    return float(mae_value / max(float(denom), 1e-12))


def exact_fill_count(y_true, y_pred) -> int:
    y_true = np.asarray(y_true, float)
    y_pred = np.asarray(y_pred, float)
    return int(np.isclose(y_true, y_pred, atol=1e-12, rtol=0).sum())


def _rankdata_average(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, float)
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x), dtype=float)
    sorted_x = x[order]
    i = 0
    while i < len(x):
        j = i + 1
        while j < len(x) and sorted_x[j] == sorted_x[i]:
            j += 1
        ranks[order[i:j]] = 0.5 * (i + j - 1) + 1.0
        i = j
    return ranks


def spearman_corr(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, float)
    y_pred = np.asarray(y_pred, float)
    if len(y_true) < 2:
        return float("nan")
    rt = _rankdata_average(y_true)
    rp = _rankdata_average(y_pred)
    rt = rt - float(np.mean(rt))
    rp = rp - float(np.mean(rp))
    denom = float(np.sqrt(np.sum(rt * rt) * np.sum(rp * rp)))
    if denom <= 1e-12:
        return float("nan")
    return float(np.sum(rt * rp) / denom)


def region_mae_metrics(y_true, y_pred) -> dict[str, float]:
    y_true = np.asarray(y_true, float)
    y_pred = np.asarray(y_pred, float)
    out: dict[str, float] = {}
    for name, lo, hi in REGION_BINS:
        mask = (y_true >= lo) & (y_true < hi)
        out[f"mae_{name}"] = mae(y_true[mask], y_pred[mask]) if mask.any() else float("nan")
    return out


def bootstrap_paired_ci(
    y_true,
    y_pred,
    n_boot: int = 5000,
    seed: int = 20260623,
    *,
    include_spearman: bool = False,
    include_regions: bool = False,
):
    y_true = np.asarray(y_true, float)
    y_pred = np.asarray(y_pred, float)
    rng = np.random.default_rng(seed)
    n = len(y_true)
    boot_mae = np.empty(n_boot, dtype=float)
    boot_rae_fixed = np.empty(n_boot, dtype=float)
    boot_rae_resampled = np.empty(n_boot, dtype=float)
    boot_spearman = np.empty(n_boot, dtype=float) if include_spearman else None
    boot_regions: dict[str, np.ndarray] = {}
    if include_regions:
        for name, _, _ in REGION_BINS:
            boot_regions[f"mae_{name}"] = np.empty(n_boot, dtype=float)

    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        yt = y_true[idx]
        yp = y_pred[idx]
        mae_i = float(np.mean(np.abs(yt - yp)))
        denom_i = phase1_denominator(yt)
        boot_mae[i] = mae_i
        boot_rae_fixed[i] = rae_from_mae(mae_i, LEADERBOARD_RAE_DENOM)
        boot_rae_resampled[i] = rae_from_mae(mae_i, denom_i)
        if boot_spearman is not None:
            boot_spearman[i] = spearman_corr(yt, yp)
        if boot_regions:
            regions_i = region_mae_metrics(yt, yp)
            for key, values in boot_regions.items():
                values[i] = regions_i[key]

    out = {
        "mae": boot_mae,
        "rae_fixed": boot_rae_fixed,
        "rae_resampled": boot_rae_resampled,
    }
    if boot_spearman is not None:
        out["spearman"] = boot_spearman
    out.update(boot_regions)
    return out


def ci_summary(values):
    values = np.asarray(values, float)
    return {
        "lo": float(np.nanquantile(values, 0.025)),
        "mid": float(np.nanmedian(values)),
        "hi": float(np.nanquantile(values, 0.975)),
    }
