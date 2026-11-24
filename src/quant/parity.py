"""Put-call parity forward and implied dividend yield."""
from __future__ import annotations

import math
from collections.abc import Iterable

import numpy as np

from src.quant.quote_hygiene import SliceQuote, is_stale, passes_spread_filter


def estimate_forward_and_q(
    quotes: Iterable[SliceQuote],
    spot: float,
    ttm: float,
    r: float,
    q_default: float,
    now_ns: int,
    *,
    max_age_secs: float,
    min_mid: float,
    max_rel_spread: float,
    atm_band: float = 0.025,
    q_bounds: tuple[float, float] = (-0.05, 0.08),
) -> tuple[float, float, int]:
    """Median forward F and implied dividend yield q across sane near-ATM call/put pairs.

    F = K + exp(rT) * (C - P) per strike. Falls back to (spot * exp((r - q_default) T), q_default, 0)
    when no pair qualifies or the implied q lies outside q_bounds. Returns (F, q, pairs used).
    """
    f_prior = spot * math.exp((r - q_default) * ttm)

    calls_by_k: dict[float, float] = {}
    puts_by_k: dict[float, float] = {}
    for qt in quotes:
        if is_stale(qt.ts_ns, now_ns, max_age_secs):
            continue
        if not passes_spread_filter(qt, min_mid, max_rel_spread):
            continue
        if abs(qt.strike - spot) / spot > atm_band:
            continue
        if qt.is_call:
            calls_by_k[qt.strike] = qt.mid
        else:
            puts_by_k[qt.strike] = qt.mid

    common_strikes = sorted(set(calls_by_k).intersection(puts_by_k))
    if not common_strikes:
        return float(f_prior), float(q_default), 0

    f_estimates = [k + math.exp(r * ttm) * (calls_by_k[k] - puts_by_k[k]) for k in common_strikes]
    f_median = float(np.median(f_estimates))
    q_implied = float(r - math.log(max(f_median, 1e-6) / spot) / ttm)
    if q_bounds[0] <= q_implied <= q_bounds[1]:
        return f_median, q_implied, len(common_strikes)
    return float(f_prior), float(q_default), 0
