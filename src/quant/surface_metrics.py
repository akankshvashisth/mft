"""Smile shape metrics (25-delta risk reversal and butterfly) and regime classification rules."""
from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

from scipy.stats import norm

from src.quant.svi import svi_implied_vol

BIAS_DEFENSIVE = "DEFENSIVE_OR_LONG_CONVEXITY"
BIAS_SHORT_VRP = "SHORT_VRP_IRON_CONDOR_OR_CREDIT_SPREAD"
BIAS_BULLISH_RR = "BULLISH_RISK_REVERSAL_OR_PUT_SPREAD"
BIAS_SELECTIVE = "SELECTIVE_RELATIVE_VALUE_SPREADS"

# |N^-1(0.25)|: standard-normal distance of a 25-delta strike (0.6745)
Z_25_DELTA = float(norm.ppf(0.75))


def delta25_log_moneyness(atm_iv: float, ttm: float, min_vol: float = 0.10) -> float:
    """Approximate 25-delta log-moneyness: k_25 ~= |N^-1(0.25)| * max(atm_iv, min_vol) * sqrt(T)."""
    return Z_25_DELTA * max(atm_iv, min_vol) * math.sqrt(ttm)


@dataclass(frozen=True)
class SmileWings:
    iv_25p: float
    iv_25c: float
    iv_atm: float

    @property
    def rr_25d(self) -> float:
        return self.iv_25p - self.iv_25c

    @property
    def bf_25d(self) -> float:
        return 0.5 * (self.iv_25p + self.iv_25c) - self.iv_atm


def smile_wings(
    atm_iv: float,
    ttm: float,
    svi_params: Sequence[float] | None = None,
    skew_25d: float = 0.0,
) -> SmileWings:
    """25-delta put/call and ATM vols from SVI params, or from ATM vol and skew when params are missing."""
    if svi_params is not None and len(svi_params) == 5:
        k_25 = delta25_log_moneyness(atm_iv, ttm)
        return SmileWings(
            iv_25p=float(svi_implied_vol(-k_25, ttm, svi_params)),
            iv_25c=float(svi_implied_vol(+k_25, ttm, svi_params)),
            iv_atm=float(svi_implied_vol(0.0, ttm, svi_params)),
        )
    return SmileWings(
        iv_25p=atm_iv + 0.5 * skew_25d,
        iv_25c=max(atm_iv - 0.5 * skew_25d, 0.01),
        iv_atm=atm_iv,
    )


def classify_strategy_bias(vix_ratio: float, vrp_spread: float, bf_25d: float, rr_pct: float) -> str:
    """Structural strategy bias from the VIX term structure, VRP and smile shape (first matching rule wins)."""
    if vix_ratio >= 1.0 or vrp_spread < -0.015:
        return BIAS_DEFENSIVE
    if vrp_spread >= 0.02 and bf_25d >= 0.005:
        return BIAS_SHORT_VRP
    if rr_pct <= 20.0:
        return BIAS_BULLISH_RR
    return BIAS_SELECTIVE


def classify_trend(spot: float, sma_20: float, sma_50: float) -> str:
    if spot > sma_20 > sma_50:
        return "UPTREND"
    if spot < sma_20 < sma_50:
        return "DOWNTREND"
    return "RANGE_OR_TRANSITION"
