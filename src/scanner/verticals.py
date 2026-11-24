"""Adjacent-strike vertical credit spreads, shared by the surface scanner and the VRP strategy.

Leg rows are mappings in the surface "legs" format: instrument_id, strike, right ("P"/"C"), bid, ask, mid,
raw_iv, svi_iv, residual_pts, edge_to_half_spread and (for edge estimates) vega.
"""
from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from statistics import fmean
from typing import Any

from src.scanner.combos import CandidateCombo, CandidateLeg

Row = Mapping[str, Any]

BULL_PUT_SPREAD = "BULL_PUT_SPREAD"
BEAR_CALL_SPREAD = "BEAR_CALL_SPREAD"
MAX_VERTICAL_WIDTH = 15.0


@dataclass(frozen=True)
class VerticalPair:
    kind: str           # BULL_PUT_SPREAD or BEAR_CALL_SPREAD
    short: Row          # Leg sold at entry
    long: Row           # Protective wing bought at entry
    width: float
    mid_credit: float
    nat_credit: float   # Sell the short leg's bid, pay the wing's ask


def split_by_right(rows: Iterable[Row]) -> tuple[list[Row], list[Row]]:
    """(puts, calls), each sorted by strike."""
    rows = list(rows)
    puts = sorted((r for r in rows if r["right"] == "P"), key=lambda r: r["strike"])
    calls = sorted((r for r in rows if r["right"] == "C"), key=lambda r: r["strike"])
    return puts, calls


def _pair(kind: str, short: Row, long: Row, width: float, max_width: float, min_mid_credit: float) -> VerticalPair | None:
    if width <= 0 or width > max_width:
        return None
    mid_credit = short["mid"] - long["mid"]
    if mid_credit <= min_mid_credit:
        return None
    return VerticalPair(kind, short, long, width, mid_credit, short["bid"] - long["ask"])


def adjacent_verticals(
    puts: Sequence[Row],
    calls: Sequence[Row],
    *,
    max_width: float = MAX_VERTICAL_WIDTH,
    min_mid_credit: float = 0.0,
) -> list[VerticalPair]:
    """Bull puts (sell a strike, buy the next lower one), then bear calls (sell a strike, buy the next higher).

    Inputs must be sorted by strike. Pairs need 0 < width <= max_width and a mid credit above min_mid_credit.
    """
    pairs: list[VerticalPair] = []
    for i in range(1, len(puts)):
        short, long = puts[i], puts[i - 1]
        pair = _pair(BULL_PUT_SPREAD, short, long, short["strike"] - long["strike"], max_width, min_mid_credit)
        if pair is not None:
            pairs.append(pair)
    for i in range(0, len(calls) - 1):
        short, long = calls[i], calls[i + 1]
        pair = _pair(BEAR_CALL_SPREAD, short, long, long["strike"] - short["strike"], max_width, min_mid_credit)
        if pair is not None:
            pairs.append(pair)
    return pairs


def vertical_name(symbol: str, pair: VerticalPair) -> str:
    strikes = f"{int(pair.short['strike'])}/{int(pair.long['strike'])}"
    if pair.kind == BULL_PUT_SPREAD:
        return f"{symbol} {strikes}P Bull Put"
    return f"{symbol} {strikes}C Bear Call"


def candidate_leg(row: Row, side: int, qty: int = 1) -> CandidateLeg:
    return CandidateLeg(
        instrument_id=row["instrument_id"],
        strike=float(row["strike"]),
        right=row["right"],
        side=side,
        qty=qty,
        bid=float(row["bid"]),
        ask=float(row["ask"]),
        mid=float(row["mid"]),
        raw_iv=float(row["raw_iv"]),
        svi_iv=float(row["svi_iv"]),
        residual_pts=float(row["residual_pts"]),
        edge_to_half_spread=float(row["edge_to_half_spread"]),
    )


def leg_vol_edge(row: Row) -> float:
    """Model edge per share of selling this leg: (market IV - SVI IV) x vega."""
    return (row["raw_iv"] - row["svi_iv"]) * row["vega"]


def mean_edge_to_half_spread(*rows: Row) -> float:
    return fmean(row["edge_to_half_spread"] for row in rows)


def vertical_combo(
    pair: VerticalPair,
    *,
    name_symbol: str,
    underlying: str,
    spot: float,
    forward: float,
    ttm_days: float,
    risk_free_rate: float,
    dividend_yield: float,
) -> CandidateCombo:
    """CandidateCombo for a vertical: legs [short (side -1), wing (side +1)] with credit, width and edge."""
    return CandidateCombo(
        name=vertical_name(name_symbol, pair),
        strategy_type=pair.kind,
        underlying=underlying,
        spot=spot,
        forward=forward,
        ttm_days=ttm_days,
        risk_free_rate=risk_free_rate,
        dividend_yield=dividend_yield,
        legs=[candidate_leg(pair.short, -1), candidate_leg(pair.long, +1)],
        net_credit=round(pair.mid_credit, 4),
        max_width=round(pair.width, 2),
        theo_edge_per_share=round(leg_vol_edge(pair.short) - leg_vol_edge(pair.long), 4),
        avg_edge_to_half_spread=round(mean_edge_to_half_spread(pair.short, pair.long), 3),
    )
