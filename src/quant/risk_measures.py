"""Tail-risk measures and tail-budget sizing."""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.stats import norm

from src.common.constants import TRADING_DAYS_PER_YEAR


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
    return abs(min(es_99, -1.0))


def contracts_for_tail_budget(net_ev: float, es_99: float, budget_usd: float) -> int:
    """Lots such that lots x tail loss <= budget; zero unless the expected value is positive."""
    if net_ev <= 0.0:
        return 0
    return int(np.floor(budget_usd / tail_loss_per_lot(es_99)))


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
    daily_sigma_usd = spot * (rv_annual / math.sqrt(trading_days))
    lin_std_usd = abs(beta_delta_shares) * daily_sigma_usd + abs(net_vega_1v) * (rv_annual * 100.0 * vol_move_frac)
    z_var = float(norm.ppf(confidence))
    z_es = float(norm.pdf(z_var)) / (1.0 - confidence)
    return round(z_var * lin_std_usd, 2), round(z_es * lin_std_usd, 2)
