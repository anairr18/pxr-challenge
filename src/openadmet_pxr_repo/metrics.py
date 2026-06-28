from __future__ import annotations

import numpy as np

LEADERBOARD_RAE_DENOM = 0.7576


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


def bootstrap_paired_ci(y_true, y_pred, n_boot: int = 5000, seed: int = 20260623):
    y_true = np.asarray(y_true, float)
    y_pred = np.asarray(y_pred, float)
    rng = np.random.default_rng(seed)
    n = len(y_true)
    boot_mae = np.empty(n_boot, dtype=float)
    boot_rae_fixed = np.empty(n_boot, dtype=float)
    boot_rae_resampled = np.empty(n_boot, dtype=float)

    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        yt = y_true[idx]
        yp = y_pred[idx]
        mae_i = float(np.mean(np.abs(yt - yp)))
        denom_i = phase1_denominator(yt)
        boot_mae[i] = mae_i
        boot_rae_fixed[i] = rae_from_mae(mae_i, LEADERBOARD_RAE_DENOM)
        boot_rae_resampled[i] = rae_from_mae(mae_i, denom_i)

    return {
        "mae": boot_mae,
        "rae_fixed": boot_rae_fixed,
        "rae_resampled": boot_rae_resampled,
    }


def ci_summary(values):
    values = np.asarray(values, float)
    return {
        "lo": float(np.quantile(values, 0.025)),
        "mid": float(np.median(values)),
        "hi": float(np.quantile(values, 0.975)),
    }

