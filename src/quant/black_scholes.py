"""Black-Scholes-Merton prices, Greeks and implied vols on QuantLib's Black formula.

QuantLib works in forward terms: an option on a spot S with continuous dividend yield q and rate r is a
Black-76 option on F = S * exp((r - q) T) with standard deviation sigma * sqrt(T) and discount exp(-r T).
"""
from __future__ import annotations

import math

import numpy as np
import QuantLib as ql

from src.common.constants import DAYS_PER_YEAR

MIN_TTM = 1e-6
MIN_VOL = 1e-6
IMPLIED_STDDEV_ACCURACY = 1e-10
IMPLIED_VOL_MAX_ITERATIONS = 100


def option_type(is_call: bool) -> int:
    return ql.Option.Call if is_call else ql.Option.Put


def vanilla_payoff(strike: float, is_call: bool) -> ql.PlainVanillaPayoff:
    return ql.PlainVanillaPayoff(option_type(is_call), float(strike))


def intrinsic_value(spot: float, strike: float, is_call: bool) -> float:
    return vanilla_payoff(strike, is_call)(float(spot))


def extrinsic_value(price: float, spot: float, strike: float, is_call: bool) -> float:
    """Time value of an option price, floored at zero."""
    return max(price - intrinsic_value(spot, strike, is_call), 0.0)


def black_calculator(
    spot: float, strike: float, ttm: float, r: float, q: float, sigma: float, is_call: bool
) -> ql.BlackCalculator:
    forward = spot * math.exp((r - q) * ttm)
    return ql.BlackCalculator(vanilla_payoff(strike, is_call), forward, sigma * math.sqrt(ttm), math.exp(-r * ttm))


def bs_greeks(
    spot: float,
    strike: float,
    ttm: float,
    r: float,
    q: float,
    sigma: float,
    is_call: bool,
) -> dict[str, float]:
    """Black-Scholes-Merton price and Greeks (Delta, Gamma, Vega/1v, Theta/day, Vanna/1v, Charm/day).

    Vega and vanna are per +1 vol point (+0.01 sigma); theta and charm per calendar day (365.25-day years).
    Expired, zero-vol or non-positive inputs are worth their spot intrinsic value.
    """
    if ttm <= MIN_TTM or sigma <= MIN_VOL or spot <= 0.0 or strike <= 0.0:
        delta = (1.0 if spot > strike else 0.0) if is_call else (-1.0 if spot < strike else 0.0)
        return {
            "price": intrinsic_value(spot, strike, is_call),
            "delta": delta,
            "gamma": 0.0,
            "vega_1v": 0.0,
            "theta_day": 0.0,
            "vanna": 0.0,
            "charm_day": 0.0,
        }

    calc = black_calculator(spot, strike, ttm, r, q, sigma, is_call)
    delta = calc.delta(spot)
    gamma = calc.gamma(spot)
    # QuantLib has no charm: dDelta/dt = q * Delta - Gamma * S * (2 (r - q) T - d2 sigma sqrt(T)) / (2 T)
    vol_sqrt_t = sigma * math.sqrt(ttm)
    d2 = (math.log(spot / strike) + (r - q - 0.5 * sigma * sigma) * ttm) / vol_sqrt_t
    charm_ann = q * delta - gamma * spot * (2.0 * (r - q) * ttm - d2 * vol_sqrt_t) / (2.0 * ttm)

    return {
        "price": calc.value(),
        "delta": delta,
        "gamma": gamma,
        "vega_1v": calc.vega(ttm) * 0.01,
        "theta_day": calc.theta(spot, ttm) / DAYS_PER_YEAR,
        "vanna": calc.vanna(spot, ttm) * 0.01,
        "charm_day": charm_ann / DAYS_PER_YEAR,
    }


def black_vega(forward: float, strike: float, ttm: float, sigma: float, discount: float) -> float:
    """dPrice/dsigma (per 1.00 of vol) of an option on `forward` discounted by `discount`; equal for calls and puts."""
    calc = ql.BlackCalculator(vanilla_payoff(strike, True), forward, sigma * math.sqrt(ttm), discount)
    return calc.vega(ttm)


def bsm_prices(S, K, t, r, sigma, flag, q=0.0) -> np.ndarray:
    """Element-wise Black-Scholes-Merton prices (QuantLib blackFormula); fast_vollib-style arguments, flag "c"/"p"."""
    S, K, t, r, sigma, q, is_call = np.broadcast_arrays(
        *(np.asarray(x, dtype=np.float64) for x in (S, K, t, r, sigma, q)), np.asarray(flag) == "c"
    )
    forward = S * np.exp((r - q) * t)
    std_dev = sigma * np.sqrt(t)
    discount = np.exp(-r * t)
    prices = [
        ql.blackFormula(option_type(bool(c)), float(k), float(f), float(s), float(d))
        for c, k, f, s, d in zip(is_call.ravel(), K.ravel(), forward.ravel(), std_dev.ravel(), discount.ravel())
    ]
    return np.asarray(prices, dtype=np.float64).reshape(S.shape)


def _implied_vol(is_call: bool, strike: float, forward: float, price: float, discount: float, ttm: float) -> float:
    if ttm <= 0.0:
        return math.nan
    try:
        std_dev = ql.blackFormulaImpliedStdDev(
            option_type(is_call), strike, forward, price, discount, 0.0, 0.2 * math.sqrt(ttm),
            IMPLIED_STDDEV_ACCURACY, IMPLIED_VOL_MAX_ITERATIONS,
        )
    except RuntimeError:  # QuantLib rejects prices outside the no-arbitrage bounds
        return math.nan
    return std_dev / math.sqrt(ttm)


def bsm_implied_vols(prices, S, K, t, r, flags) -> np.ndarray:
    """Element-wise Black-Scholes implied vols (QuantLib); NaN where no vol reproduces the price.

    Same arguments as the fast_vollib solver: S is the dividend-discounted spot S * exp(-qT) and r the true rate,
    so the forward is S * exp(rT).
    """
    prices, S, K, t, r, is_call = np.broadcast_arrays(
        *(np.asarray(x, dtype=np.float64) for x in (prices, S, K, t, r)), np.asarray(flags) == "c"
    )
    forward = S * np.exp(r * t)
    discount = np.exp(-r * t)
    vols = [
        _implied_vol(bool(c), float(k), float(f), float(p), float(d), float(tt))
        for c, k, f, p, d, tt in zip(
            is_call.ravel(), K.ravel(), forward.ravel(), prices.ravel(), discount.ravel(), t.ravel()
        )
    ]
    return np.asarray(vols, dtype=np.float64).reshape(prices.shape)
