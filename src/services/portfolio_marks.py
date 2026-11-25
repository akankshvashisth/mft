"""Option marks, Greeks exposures and spot x vol stress for the portfolio book (pure functions)."""
from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

from src.common.constants import DAYS_PER_YEAR
from src.common.market import OptionSpec
from src.common.timeutil import days_to_years, years_until
from src.quant.black_scholes import bs_greeks
from src.quant.stress import shock_grid
from src.quant.svi import svi_implied_vol_or_none

GREEK_KEYS = ("delta", "beta_delta", "gamma_1pct", "vega_1v", "theta_day", "vanna", "charm_day")
SPOT_SHOCKS_PCT = (-10.0, -5.0, -2.5, 0.0, 2.5, 5.0, 10.0)
VOL_SHOCKS_PTS = (-5.0, -2.5, 0.0, 2.5, 5.0, 10.0)
STRESS_HORIZON_DAYS = 7.0
MIN_STRESS_VOL = 0.03

Payload = Mapping[str, Any]


def smile_iv(surface: Payload, spot: float, q: float, strike: float, ttm: float, r: float) -> float | None:
    """IV at an option's own strike from the underlying's calibrated SVI slice.

    Same expiry (within a day): log-moneyness against the slice forward. Other expiries: own carry forward,
    with moneyness rescaled by sqrt(T) onto the calibrated slice.
    """
    params = surface.get("svi_params") or []
    t_surf = days_to_years(float(surface.get("ttm_days") or 0.0))
    fwd_surf = float(surface.get("forward") or 0.0)
    if len(params) != 5 or t_surf <= 1e-6 or strike <= 0.0 or spot <= 0.0:
        return None
    if abs(ttm - t_surf) * DAYS_PER_YEAR < 1.0 and fwd_surf > 0.0:
        k = math.log(strike / fwd_surf)
    else:
        fwd = spot * math.exp((r - q) * ttm)
        k = math.log(strike / fwd) * math.sqrt(t_surf / ttm)
    return svi_implied_vol_or_none(k, t_surf, params)


def option_exposure(
    option: OptionSpec,
    multiplier: float,
    qty: float,
    now_ns: int,
    *,
    spot: float,
    surface: Payload,
    rv: Payload,
    bench_spot: float,
    bench_atm_iv: float,
    r: float,
    live_mid: float | None,
) -> dict[str, Any]:
    """Mark, Greeks contribution and stress-leg spec for `qty` signed contracts of one option."""
    q_und = float(surface.get("implied_q") or 0.0)
    strike = float(option.strike)
    ttm = years_until(option.expiry_ns, now_ns, floor=1e-4)
    iv = smile_iv(surface, spot, q_und, strike, ttm, r)
    if iv is None:
        iv = float(surface.get("atm_iv") or rv.get("rv_20d") or bench_atm_iv)

    g = bs_greeks(spot, strike, ttm, r, q_und, iv, option.is_call)
    notional_qty = qty * multiplier
    d_sh = notional_qty * g["delta"]
    return {
        "mark": live_mid if live_mid is not None else g["price"],
        "mark_source": "LIVE_MID" if live_mid is not None else "MODEL",
        "multiplier": multiplier,
        "greeks": {
            "delta": d_sh,
            # Beta-weight delta to the benchmark via spot ratio (beta=1.0 default)
            "beta_delta": d_sh * (spot / bench_spot) if bench_spot > 0 else d_sh,
            "gamma_1pct": notional_qty * g["gamma"] * (0.01 * spot),
            "vega_1v": notional_qty * g["vega_1v"],
            "theta_day": notional_qty * g["theta_day"],
            "vanna": notional_qty * g["vanna"],
            "charm_day": notional_qty * g["charm_day"],
        },
        "stress_leg": {
            "underlying": str(option.underlying),
            "is_option": True,
            "signed_qty": qty,
            "multiplier": multiplier,
            "spot": spot,
            "strike": strike,
            "ttm": ttm,
            "q": q_und,
            "iv": iv,
            "is_call": option.is_call,
            "theo_price": g["price"],
        },
    }


def linear_stress_leg(signed_qty: float, multiplier: float, price: float, underlying: str = "") -> dict[str, Any]:
    return {"underlying": underlying, "is_option": False, "signed_qty": signed_qty, "multiplier": multiplier, "spot": price}


def stress_leg_pnl(leg: Payload, spot_mult: float, vol_shift: float, dt_years: float, r: float) -> float:
    """P&L of one leg after a relative spot move, an absolute vol shift and dt_years of decay."""
    qty = leg["signed_qty"]
    mult = leg["multiplier"]
    s_new = leg["spot"] * spot_mult
    if not leg["is_option"]:
        return qty * mult * (s_new - leg["spot"])
    t_new = max(leg["ttm"] - dt_years, 1e-4)
    sig_new = max(leg["iv"] + vol_shift, MIN_STRESS_VOL)
    g_new = bs_greeks(s_new, leg["strike"], t_new, r, leg["q"], sig_new, leg["is_call"])
    return qty * mult * (g_new["price"] - leg["theo_price"])


def book_stress_grid(
    legs: Sequence[Payload],
    r: float,
    spot_shocks_pct: Sequence[float] = SPOT_SHOCKS_PCT,
    vol_shocks_pts: Sequence[float] = VOL_SHOCKS_PTS,
    horizon_days: float = STRESS_HORIZON_DAYS,
) -> list[list[float]]:
    """Spot x Vol book P&L matrix ($ USD) across all legs; rows are spot shocks, columns vol shocks."""
    if not legs:
        return [[0.0 for _ in vol_shocks_pts] for _ in spot_shocks_pct]
    dt = days_to_years(horizon_days)

    def pnl(spot_pcts, vol_pts):
        return [
            sum(stress_leg_pnl(leg, 1.0 + s_pct / 100.0, v_pts / 100.0, dt, r) for leg in legs)
            for s_pct, v_pts in zip(spot_pcts, vol_pts)
        ]

    return shock_grid(spot_shocks_pct, vol_shocks_pts, pnl)
