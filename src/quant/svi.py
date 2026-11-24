"""Raw SVI total-variance smiles: evaluation, implied vols and spread-weighted calibration.

w(k) = a + b * (rho * (k - m) + sqrt((k - m)^2 + sigma^2)), with k = ln(K / F); parameters are ordered
(a, b, rho, m, sigma). The calibration evaluates w(k) for any parameters inside the optimiser bounds, including
ones QuantLib's SviSmileSection rejects, so the formula stays vectorised numpy here.
"""
from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
import QuantLib as ql
from scipy.optimize import least_squares
from sklearn.metrics import root_mean_squared_error

SVI_LOWER_BOUNDS = np.array([-0.5, 1e-4, -0.999, -0.5, 5e-3], dtype=np.float64)
SVI_UPPER_BOUNDS = np.array([2.0, 2.0, 0.999, 0.5, 2.0], dtype=np.float64)
W_FLOOR = 1e-8


def svi_total_variance(k, params: Sequence[float]):
    a, b, rho, m, sigma = params
    dk = k - m
    return a + b * (rho * dk + np.sqrt(dk * dk + sigma * sigma))


def svi_implied_vol(k, ttm: float, params: Sequence[float], w_floor: float = W_FLOOR):
    """Vectorised implied vol; total variance is floored at w_floor and ttm at 1e-6."""
    w = np.maximum(svi_total_variance(np.asarray(k, dtype=float), params), w_floor)
    return np.sqrt(w / max(ttm, 1e-6))


def svi_implied_vol_or_none(k: float, ttm: float, params: Sequence[float] | None) -> float | None:
    """Scalar implied vol; None for malformed params, ttm <= 1e-6 or non-positive total variance."""
    if params is None or len(params) != 5 or ttm <= 1e-6:
        return None
    w = float(svi_total_variance(float(k), [float(x) for x in params]))
    return math.sqrt(w / ttm) if w > 0.0 else None


def roger_lee_utilization(params: Sequence[float]) -> float:
    """b * (1 + |rho|) / 2: share of Roger Lee's asymptotic wing-slope bound in use."""
    return float(params[1]) * (1.0 + abs(float(params[2]))) / 2.0


def svi_smile_section(params: Sequence[float], ttm: float, forward: float = 1.0) -> ql.SviSmileSection:
    """QuantLib smile section of raw SVI params (a, b, rho, m, sigma); QuantLib orders them (a, b, sigma, rho, m).

    QuantLib raises RuntimeError unless ttm > 0, b >= 0, |rho| < 1, sigma > 0, a + b sigma sqrt(1 - rho^2) >= 0
    and b (1 + |rho|) <= 4.
    """
    a, b, rho, m, sigma = (float(x) for x in params)
    return ql.SviSmileSection(float(ttm), float(forward), [a, b, sigma, rho, m])


def fit_svi_slice(
    log_m: np.ndarray,
    ivs: np.ndarray,
    weights: np.ndarray,
    ttm: float,
    x0_warm: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Fits spread-weighted Raw SVI total variance with warm-start + cold-start fallback.

    Returns (fitted IVs on log_m, params [a, b, rho, m, sigma], RMSE in vol units).
    """
    w_mkt = (ivs ** 2) * ttm
    w_atm = float(np.interp(0.0, log_m, w_mkt))
    w_norm = np.sqrt(weights / max(float(np.mean(weights)), 1e-12))
    lb = SVI_LOWER_BOUNDS
    ub = SVI_UPPER_BOUNDS

    def residuals(p: np.ndarray) -> np.ndarray:
        a, b, rho, m, sigma = p
        # Soft penalty enforcing Roger Lee wing bound b*(1 + |rho|) <= 2 & positive min variance
        wing_excess = max(0.0, b * (1.0 + abs(rho)) - 1.98)
        min_var = a + b * sigma * np.sqrt(max(1.0 - rho * rho, 1e-6))
        var_deficit = max(0.0, 1e-6 - min_var)
        res = w_norm * (svi_total_variance(log_m, p) - w_mkt)
        return np.append(res, [10.0 * wing_excess, 10.0 * var_deficit])

    def rmse_of(params: np.ndarray) -> tuple[np.ndarray, float]:
        iv_fit = svi_implied_vol(log_m, ttm, params, w_floor=1e-6)
        return iv_fit, root_mean_squared_error(ivs, iv_fit)

    cold_x0 = np.array([max(w_atm * 0.85, 1e-5), 0.08, -0.55, 0.0, 0.08], dtype=np.float64)
    start_x0 = np.clip(x0_warm, lb + 1e-6, ub - 1e-6) if x0_warm is not None else cold_x0

    res = least_squares(residuals, x0=start_x0, bounds=(lb, ub), max_nfev=120)
    iv_fit, rmse_vol = rmse_of(res.x)

    # If warm-start drifted into a poor fit (> 0.5 vol pts RMSE), re-run from cold start
    if x0_warm is not None and rmse_vol > 0.005:
        res_cold = least_squares(residuals, x0=cold_x0, bounds=(lb, ub), max_nfev=150)
        iv_cold, rmse_cold = rmse_of(res_cold.x)
        if rmse_cold < rmse_vol:
            return iv_cold, res_cold.x, rmse_cold

    return iv_fit, res.x, rmse_vol
