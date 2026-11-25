"""Tail-risk measures and tail-budget sizing."""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.stats import norm

from src.common.constants import DAYS_PER_YEAR, DEFAULT_RISK_FREE_RATE, TRADING_DAYS_PER_YEAR


@dataclass(frozen=True)
class TailMetrics:
    mean: float
    prob_profit: float
    var_95: float
    es_95: float
    var_99: float
    es_99: float


def tail_metrics(pnl) -> TailMetrics:
    """Mean, P(P&L > 0), 5%/1% VaR (percentiles) and ES (mean P&L at or below each VaR)."""
    pnl = np.asarray(pnl, dtype=np.float64)
    pnl = pnl[np.isfinite(pnl)]
    if pnl.size == 0:
        return TailMetrics(mean=0.0, prob_profit=0.0, var_95=0.0, es_95=0.0, var_99=0.0, es_99=0.0)

    var_95 = float(np.percentile(pnl, 5.0))
    es_95 = float(np.mean(pnl[pnl <= var_95]))
    var_99 = float(np.percentile(pnl, 1.0))
    es_99 = float(np.mean(pnl[pnl <= var_99]))
    return TailMetrics(
        mean=float(np.mean(pnl)),
        prob_profit=float(np.mean(pnl > 0.0)),
        var_95=var_95,
        es_95=es_95,
        var_99=var_99,
        es_99=es_99,
    )


def tail_loss_per_lot(es_99: float) -> float:
    """Positive tail loss used for sizing; floored at $1 so near-riskless P&L can't divide by zero."""
    if not math.isfinite(es_99):
        return 1.0
    return abs(min(es_99, -1.0))


def contracts_for_tail_budget(net_ev: float, es_99: float, budget_usd: float) -> int:
    """Lots such that lots x tail loss <= budget; zero unless the expected value is positive."""
    if net_ev <= 0.0:
        return 0
    return int(np.floor(budget_usd / tail_loss_per_lot(es_99)))


def portfolio_horizon_var_es(
    legs_spec: list[dict],
    *,
    horizon_days: float = 14.0,
    r: float = DEFAULT_RISK_FREE_RATE,
    n_scenarios: int = 5_000,
    seed: int = 42,
) -> tuple[float, float]:
    """14-day nonlinear portfolio Monte Carlo 99% VaR and Expected Shortfall (both ≥ 0).

    Simulates correlated 1-factor (market) + idiosyncratic spot shocks per underlying
    over ``horizon_days`` (Student-:math:`t_5`, tail correlation 0.85 on market down-shocks)
    and leverage-correlated IV shocks.  Every option leg is repriced nonlinearly via
    Black-Scholes.  Linear legs (shares, futures) are revalued linearly.

    Parameters
    ----------
    legs_spec : list[dict]
        Each dict must have keys ``is_option`` (bool), ``signed_qty`` (float),
        ``multiplier`` (float), ``spot`` (float), and optionally ``strike``, ``ttm``,
        ``q``, ``iv``, ``is_call`` for option legs.
    horizon_days : float
        Risk horizon in calendar days.
    r : float
        Annual risk-free rate.
    n_scenarios : int
        Number of Monte Carlo scenarios.
    seed : int
        RNG seed for CRN.

    Returns
    -------
    var_99, es_99 : tuple[float, float]
        Both are non-negative USD amounts (loss magnitudes).
    """
    if not legs_spec:
        return 0.0, 0.0
    dt_years = horizon_days / DAYS_PER_YEAR

    rng = np.random.default_rng(seed)

    # Identify unique underlying spots (group legs by spot for shock generation)
    spots = np.array([leg["spot"] for leg in legs_spec], dtype=np.float64)
    n_legs = len(legs_spec)

    # Student-t 5 draws, scaled to unit variance sqrt((nu-2)/nu)
    z_market = rng.standard_t(df=5, size=n_scenarios) * np.sqrt(3.0 / 5.0)  # unit variance
    z_idio = rng.standard_t(df=5, size=(n_scenarios, n_legs)) * np.sqrt(3.0 / 5.0)

    # Tail correlation: on market down-shocks (z_market < 0), mix 85% common + 15% idiosyncratic
    rho_tail = 0.85
    tail_mask = z_market < 0.0
    for i in range(n_legs):
        mix = np.where(tail_mask, rho_tail, 0.0)
        z_idio[:, i] = mix * z_market + np.sqrt(1.0 - mix ** 2) * z_idio[:, i]

    # Effective shock per leg: 1-factor loading = 1.0 (all equities), plus idiosyncratic
    z_eff = z_market[:, np.newaxis] + z_idio

    # Annualised vol for scaling (use a default 30% if iv not available)
    vols = np.array(
        [leg.get("iv", 0.30) if leg.get("is_option") else 0.30 for leg in legs_spec],
        dtype=np.float64,
    )

    # Spot simulation: GBM drift
    drift = (r - 0.5 * vols ** 2) * dt_years
    vol_scale = vols * np.sqrt(dt_years)
    spot_sim = spots[np.newaxis, :] * np.exp(drift[np.newaxis, :] + vol_scale[np.newaxis, :] * z_eff)

    # IV shock: leverage effect (β=0.7) + vol-of-vol
    beta_lev = 0.7
    vol_shock = -beta_lev * vol_scale[np.newaxis, :] * z_eff + 0.35 * vols[np.newaxis, :] * np.sqrt(dt_years) * rng.standard_normal(size=(n_scenarios, n_legs))
    vol_sim = np.maximum(vols[np.newaxis, :] + vol_shock, 0.05)

    # Reprice each leg
    from src.quant.black_scholes import bs_greeks

    pnl = np.zeros(n_scenarios, dtype=np.float64)
    for i, leg in enumerate(legs_spec):
        qty = float(leg["signed_qty"])
        mult = float(leg["multiplier"])
        if not leg.get("is_option", False):
            # Linear leg
            pnl += qty * mult * (spot_sim[:, i] - spots[i])
        else:
            # Option leg: reprice via Black-Scholes
            strike = float(leg["strike"])
            ttm = max(float(leg.get("ttm", dt_years)) - dt_years, 1e-4)
            is_call = bool(leg.get("is_call", True))
            q = float(leg.get("q", 0.0))
            theo_price = float(leg.get("theo_price", 0.0))

            scenario_pnl = np.zeros(n_scenarios, dtype=np.float64)
            for j in range(n_scenarios):
                g = bs_greeks(spot_sim[j, i], strike, ttm, r, q, vol_sim[j, i], is_call)
                scenario_pnl[j] = g["price"] - theo_price
            pnl += qty * mult * scenario_pnl

    # Compute VaR and ES from the P&L vector (loss = -pnl)
    loss = -pnl
    var_99 = float(np.percentile(loss, 99.0))
    es_99 = float(np.mean(loss[loss >= var_99])) if np.any(loss >= var_99) else 0.0
    return max(var_99, 0.0), max(es_99, 0.0)


def parametric_book_var_es(
    beta_delta_shares: float,
    net_vega_1v: float,
    spot: float,
    rv_annual: float,
    *,
    trading_days: int = TRADING_DAYS_PER_YEAR,
    vol_move_frac: float = 0.08,
    confidence: float = 0.99,
) -> tuple[float, float]:
    """Approximate 1-day VaR and ES (USD) at `confidence` for normally distributed linear delta and vega P&L.

    One-day spot sigma is spot * rv / sqrt(trading_days); the vol move is vol_move_frac of RV in vol points.
    The VaR multiplier is the normal quantile z and the ES multiplier E[Z | Z > z] = pdf(z) / (1 - confidence).
    """
    if trading_days <= 0:
        raise ValueError("trading_days must be positive")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be between zero and one")
    if not all(math.isfinite(x) for x in (beta_delta_shares, net_vega_1v, spot)):
        raise ValueError("exposure and spot inputs must be finite")
    if spot < 0.0:
        raise ValueError("spot must be non-negative")
    if not math.isfinite(rv_annual) or rv_annual < 0.0:
        raise ValueError("rv_annual must be finite and non-negative")
    if not math.isfinite(vol_move_frac) or vol_move_frac < 0.0:
        raise ValueError("vol_move_frac must be finite and non-negative")

    daily_sigma_usd = spot * (rv_annual / math.sqrt(trading_days))
    lin_std_usd = abs(beta_delta_shares) * daily_sigma_usd + abs(net_vega_1v) * (rv_annual * 100.0 * vol_move_frac)
    z_var = float(norm.ppf(confidence))
    z_es = float(norm.pdf(z_var)) / (1.0 - confidence)
    return round(z_var * lin_std_usd, 2), round(z_es * lin_std_usd, 2)
