"""Option-chain quote hygiene: per-quote filters for SVI slices and array masks for whole chains."""
from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class SliceQuote:
    """Latest two-sided quote of one option in a single-expiry slice."""

    instrument_id: str
    strike: float
    is_call: bool
    bid: float
    ask: float
    mid: float
    ts_ns: int
    open_interest: float = 0.0   # 0 when unknown


@dataclass(frozen=True)
class SliceFilter:
    max_age_secs: float = 90.0
    min_mid: float = 0.15
    max_rel_spread: float = 0.25
    min_open_interest: int = 0
    atm_overlap_pct: float = 0.006


@dataclass
class SliceSelection:
    quotes: list[SliceQuote] = field(default_factory=list)
    spreads: list[float] = field(default_factory=list)   # Bid-ask spread per kept quote, floored at 0.01
    dropped: dict[str, int] = field(
        default_factory=lambda: {"stale": 0, "spread": 0, "intrinsic": 0, "itm": 0, "oi": 0}
    )


def is_stale(ts_ns: int, now_ns: int, max_age_secs: float) -> bool:
    """Quote older than max_age_secs; a non-positive limit disables the check."""
    max_age_ns = int(max_age_secs * 1e9)
    return max_age_ns > 0 and (now_ns - ts_ns) > max_age_ns


def passes_spread_filter(quote: SliceQuote, min_mid: float, max_rel_spread: float) -> bool:
    return quote.mid >= min_mid and (quote.ask - quote.bid) / quote.mid <= max_rel_spread


def select_slice_quotes(
    quotes: Iterable[SliceQuote],
    forward: float,
    disc: float,
    now_ns: int,
    flt: SliceFilter,
) -> SliceSelection:
    """Keeps fresh, tight, liquid out-of-the-money quotes priced above their discounted forward intrinsic."""
    out = SliceSelection()
    for qt in quotes:
        # 1. Quote freshness filter
        if is_stale(qt.ts_ns, now_ns, flt.max_age_secs):
            out.dropped["stale"] += 1
            continue

        # 2. Sane bid/ask & minimum premium filter
        spread = qt.ask - qt.bid
        if not passes_spread_filter(qt, flt.min_mid, flt.max_rel_spread):
            out.dropped["spread"] += 1
            continue

        # 3. Open-interest gate (only when OI metadata is known)
        if flt.min_open_interest > 0 and qt.open_interest > 0 and qt.open_interest < flt.min_open_interest:
            out.dropped["oi"] += 1
            continue

        # 4. Strict ATM + OTM splice filter
        k = qt.strike
        if qt.is_call and k < forward * (1.0 - flt.atm_overlap_pct):
            out.dropped["itm"] += 1
            continue
        if (not qt.is_call) and k > forward * (1.0 + flt.atm_overlap_pct):
            out.dropped["itm"] += 1
            continue

        # 5. Pre-solver intrinsic arbitrage bound (eliminates slow IV fallback spikes)
        intrinsic = disc * max(forward - k, 0.0) if qt.is_call else disc * max(k - forward, 0.0)
        if qt.mid <= intrinsic + 1e-4:
            out.dropped["intrinsic"] += 1
            continue

        out.quotes.append(qt)
        out.spreads.append(max(spread, 0.01))
    return out


def clean_quote_mask(
    S,
    K,
    t,
    r,
    prices,
    flag,
    *,
    bids=None,
    asks=None,
    min_price: float = 1e-3,
    max_abs_log_moneyness: float = 0.45,
    max_rel_spread: float | None = None,
) -> np.ndarray:
    """Boolean mask of chain quotes worth inverting.

    Rejects non-finite or sub-minimum prices, prices at or below spot-vs-discounted-strike intrinsic,
    prices at or above spot, and far wings; optionally zero bids and wide relative spreads.
    """
    S = np.asarray(S, dtype=float)
    K = np.asarray(K, dtype=float)
    t = np.asarray(t, dtype=float)
    r = np.asarray(r, dtype=float)
    prices = np.asarray(prices, dtype=float)
    flag = np.asarray(flag)

    disc = np.exp(-r * t)
    intrinsic = np.where(flag == "c", np.maximum(S - K * disc, 0.0), np.maximum(K * disc - S, 0.0))
    with np.errstate(invalid="ignore"):
        valid = (
            np.isfinite(prices)
            & (prices > np.maximum(intrinsic + 1e-4, min_price))
            & (prices < S - 1e-4)
            & (np.abs(np.log(K / S)) <= max_abs_log_moneyness)
        )
    if bids is not None:
        valid &= np.asarray(bids, dtype=float) > 0.0
    if bids is not None and asks is not None and max_rel_spread is not None:
        spreads = np.asarray(asks, dtype=float) - np.asarray(bids, dtype=float)
        valid &= (spreads / np.maximum(prices, 1e-6)) <= max_rel_spread
    return valid
