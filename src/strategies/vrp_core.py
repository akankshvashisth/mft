"""Pure building blocks of the VRP income strategy: spread book, exit/entry rules, candidates and rows.

Spread instruments carry ratio +1 on the leg sold at entry and -1 on the protective wing, so the spread price
is a positive credit: entries SELL it, exits BUY it back.
"""
from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from datetime import date, timedelta
from typing import Any

from src.common.constants import DEFAULT_RISK_FREE_RATE
from src.common.market import OptionSpec, QuoteObs
from src.common.quotes import two_sided_mid
from src.common.timeutil import utc_date
from src.quant.black_scholes import extrinsic_value
from src.quant.risk_measures import tail_loss_per_lot
from src.risk.gate_types import CandidateLegSpec, GateDecision
from src.scanner.verticals import VerticalPair, adjacent_verticals, split_by_right, vertical_combo

ENTRY = "ENTRY"
EXIT = "EXIT"
STRATEGY_MIN_CREDIT = 0.05
DEFAULT_TOB_SIZE = 10
POSITION_TOLERANCE = 1e-6

Trade = dict[str, Any]
StrategyQuote = tuple[float, float, int | None, int | None]   # bid, ask, bid size, ask size


# --- Spread book -----------------------------------------------------------------------------------


def _is_open_trade(trade: Any) -> bool:
    if not isinstance(trade, dict) or not trade.get("legs") or not trade.get("expiry"):
        return False
    try:
        return float(trade.get("open_lots") or 0.0) > 0.0
    except (TypeError, ValueError):
        return False


class SpreadBook:
    """Spread positions opened by the strategy, keyed by spread instrument id."""

    def __init__(self, trades: Mapping[str, Trade] | None = None) -> None:
        self.trades: dict[str, Trade] = dict(trades or {})

    @classmethod
    def from_json(cls, raw: Mapping[str, Any]) -> "SpreadBook":
        """Keeps only well-formed open trades (persisted state may be stale or hand-edited)."""
        return cls({str(key): trade for key, trade in raw.items() if _is_open_trade(trade)})

    def to_json(self) -> dict[str, Trade]:
        return self.trades

    def __len__(self) -> int:
        return len(self.trades)

    def __contains__(self, spread_key: object) -> bool:
        return spread_key in self.trades

    def apply_entry_fill(
        self, spread_key: str, combo: Any, legs: list[dict], expiry: str, price: float, lots: float, now_ns: int
    ) -> Trade:
        """Adds filled lots at a lot-weighted average entry credit (opening the trade if needed)."""
        trade = self.trades.get(spread_key)
        if trade is None:
            trade = {
                "combo": combo,
                "legs": legs,
                "expiry": expiry,
                "open_lots": 0.0,
                "entry_credit": 0.0,
                "opened_ts_ns": now_ns,
            }
            self.trades[spread_key] = trade
        prev_lots = float(trade["open_lots"])
        trade["entry_credit"] = (float(trade["entry_credit"]) * prev_lots + price * lots) / (prev_lots + lots)
        trade["open_lots"] = prev_lots + lots
        return trade

    def apply_exit_fill(self, spread_key: str, lots: float) -> None:
        """Reduces open lots; a fully closed trade leaves the book."""
        trade = self.trades.get(spread_key)
        if trade is None:
            return
        trade["open_lots"] = max(0.0, float(trade["open_lots"]) - lots)
        if trade["open_lots"] <= 1e-9:
            del self.trades[spread_key]

    def expected_leg_positions(self) -> dict[str, float]:
        """Option leg positions the book implies (a short spread holds -lots x ratio of each leg)."""
        expected: dict[str, float] = {}
        for trade in self.trades.values():
            for leg in trade["legs"]:
                leg_id = str(leg["instrument_id"])
                expected[leg_id] = expected.get(leg_id, 0.0) - float(trade["open_lots"]) * int(leg["ratio"])
        return expected

    def rows(self) -> list[dict[str, Any]]:
        return [
            {
                "spread_id": key,
                "combo": trade.get("combo"),
                "open_lots": trade.get("open_lots"),
                "entry_credit": round(float(trade.get("entry_credit") or 0.0), 4),
                "expiry": trade.get("expiry"),
            }
            for key, trade in self.trades.items()
        ]


# --- Exit rules ------------------------------------------------------------------------------------


def exit_reason(
    dte: int,
    mid_debit: float | None,
    entry_credit: float,
    profit_target_fraction: float,
    min_exit_dte: int,
) -> str | None:
    """Time stop inside the exit window, else profit target once the mid debit falls to the target."""
    target_debit = entry_credit * (1.0 - profit_target_fraction)
    if dte <= min_exit_dte:
        return f"TIME_STOP(dte={dte}<={min_exit_dte})"
    if mid_debit is not None and mid_debit <= target_debit:
        return f"PROFIT_TARGET(mid_debit={mid_debit:.2f}<={target_debit:.2f})"
    return None


def leg_mismatches(expected: Mapping[str, float], net_position_qty: Callable[[str], float]) -> set[str]:
    """Option legs whose open position differs from what the book implies."""
    return {
        leg_id
        for leg_id, qty in expected.items()
        if abs(net_position_qty(leg_id) - qty) > POSITION_TOLERANCE
    }


def exit_confirmed(
    spread_key: str,
    trade: Trade,
    mismatches: set[str],
    net_position_qty: Callable[[str], float],
) -> bool:
    """Closing BUYs the spread, so only do it while the cache shows the short exposure (spread or leg level)."""
    if abs(net_position_qty(spread_key) + float(trade["open_lots"])) < POSITION_TOLERANCE:
        return True
    return not any(str(leg["instrument_id"]) in mismatches for leg in trade["legs"])


def exit_signal(
    spread_key: str,
    trade: Trade,
    reason: str,
    open_lots: float,
    entry_credit: float,
    prices: tuple[float, float] | None,
) -> dict[str, Any]:
    return {
        "combo": trade.get("combo"),
        "spread_id": spread_key,
        "reason": reason,
        "open_lots": open_lots,
        "entry_credit": round(entry_credit, 4),
        "close_mid_debit": round(prices[0], 4) if prices is not None else None,
        "close_natural_debit": round(prices[1], 4) if prices is not None else None,
        "status": "SHADOW_SIGNAL",
        "order_id": None,
    }


# --- Entry rules -----------------------------------------------------------------------------------


def entry_block_reason(
    has_working_order: bool,
    spot_mid: float,
    last_fill_wall_ns: int,
    portfolio_ts_ns: int,
) -> str | None:
    if has_working_order:
        return "ORDER_WORKING"
    if spot_mid <= 0.0:
        return "NO_UNDERLYING_QUOTE"
    if last_fill_wall_ns and portfolio_ts_ns <= last_fill_wall_ns:
        # Book ES99 must include the latest fill before more tail budget is committed
        return "WAITING_PORTFOLIO_REFRESH"
    return None


def apply_dte_window(decision: GateDecision, dte: int, min_exit_dte: int) -> GateDecision:
    """A spread the time stop would close right after entry is never approved."""
    if dte <= min_exit_dte:
        decision.reasons.append(f"DTE_INSIDE_EXIT_WINDOW({dte}<={min_exit_dte})")
        decision.approved = False
        decision.approved_qty = 0
    return decision


# --- Candidate legs --------------------------------------------------------------------------------


def normalize_right(raw: Any) -> str:
    right = str(raw or "").upper()
    if right in ("P", "PUT"):
        return "P"
    if right in ("C", "CALL"):
        return "C"
    return right


def _open_interest(raw: Any) -> int | None:
    """Real open interest only; unknown, zero or malformed values stay None (never a placeholder)."""
    try:
        return int(float(raw)) if raw is not None and float(raw) > 0 else None
    except (TypeError, ValueError):
        return None


def enrich_surface_legs(
    raw_legs: Iterable[Mapping[str, Any]],
    atm_iv: float,
    default_expiry: date,
    strategy_quote: Callable[[str], StrategyQuote | None],
    cache_quote: Callable[[str], QuoteObs | None],
    option_spec: Callable[[str], OptionSpec | None],
) -> list[dict[str, Any]]:
    """Surface leg rows overlaid with live quotes, real open interest and the leg's own expiry.

    Quote precedence: the strategy's own quote stream, else the data cache, else the surface snapshot.
    Open interest: the surface table, else instrument metadata, else unknown (None).
    """
    rows: list[dict[str, Any]] = []
    for raw in raw_legs:
        instrument_id = str(raw.get("instrument_id", ""))
        if not instrument_id:
            continue
        bid = float(raw.get("bid") or 0.0)
        ask = float(raw.get("ask") or 0.0)
        bid_size: int | None = None
        ask_size: int | None = None

        own = strategy_quote(instrument_id)
        if own is not None:
            q_bid, q_ask, q_bsz, q_asz = own
            if two_sided_mid(q_bid, q_ask) is not None:
                bid, ask, bid_size, ask_size = q_bid, q_ask, q_bsz, q_asz
        else:
            cached = cache_quote(instrument_id)
            if cached is not None and cached.mid is not None:
                bid, ask = cached.bid, cached.ask
                bid_size = int(cached.bid_size) if cached.bid_size is not None else None
                ask_size = int(cached.ask_size) if cached.ask_size is not None else None

        mid = two_sided_mid(bid, ask)
        if mid is None:
            continue

        spec = option_spec(instrument_id)
        expiry = utc_date(spec.expiry_ns) if spec is not None and spec.expiry_ns > 0 else default_expiry
        oi_raw = raw.get("open_interest")
        if oi_raw is None and spec is not None:
            oi_raw = spec.open_interest

        rows.append(
            {
                "instrument_id": instrument_id,
                "strike": float(raw.get("strike") or 0.0),
                "right": normalize_right(raw.get("right")),
                "bid": bid,
                "ask": ask,
                "mid": mid,
                "bid_size": bid_size,
                "ask_size": ask_size,
                "open_interest": _open_interest(oi_raw),
                "expiry": expiry,
                "raw_iv": float(raw.get("raw_iv") or atm_iv),
                "svi_iv": float(raw.get("svi_iv") or atm_iv),
                "residual_pts": float(raw.get("residual_pts") or 0.0),
                "edge_to_half_spread": float(raw.get("edge_to_half_spread") or 0.0),
                "vega": float(raw.get("vega") or 0.0),
            }
        )
    return rows


def gate_leg(row: Mapping[str, Any], side: str, spot: float) -> CandidateLegSpec:
    """1-lot gate leg; the extrinsic value is the leg's real time value at the current spot."""
    return CandidateLegSpec(
        row["instrument_id"],
        row["right"][:1],
        side,
        row["strike"],
        row["expiry"],
        qty=1,
        bid=row["bid"],
        ask=row["ask"],
        bid_size=row["bid_size"],
        ask_size=row["ask_size"],
        open_interest=row["open_interest"],
        extrinsic_value=extrinsic_value(row["mid"], spot, row["strike"], row["right"] == "C"),
    )


def spread_legs(pair: VerticalPair) -> list[dict[str, Any]]:
    """Spread definition: +1 on the leg sold at entry, -1 on the wing => positive credit price."""
    return [
        {"instrument_id": pair.short["instrument_id"], "ratio": 1},
        {"instrument_id": pair.long["instrument_id"], "ratio": -1},
    ]


def build_candidates(
    surface: Mapping[str, Any],
    *,
    spot_mid: float,
    today: date,
    symbol: str,
    underlying: str,
    strategy_quote: Callable[[str], StrategyQuote | None],
    cache_quote: Callable[[str], QuoteObs | None],
    option_spec: Callable[[str], OptionSpec | None],
) -> list[dict[str, Any]]:
    """Defined-risk bull put / bear call spreads from surface legs overlaid with live quotes."""
    spot = spot_mid or float(surface.get("spot") or 0.0)
    forward = float(surface.get("forward") or spot)
    ttm_days = float(surface.get("ttm_days") or 25.0)
    implied_q = float(surface.get("implied_q") or 0.02)
    atm_iv = float(surface.get("atm_iv") or 0.23)
    raw_legs = surface.get("legs") or []
    if spot <= 0.0 or not raw_legs:
        return []

    default_expiry = today + timedelta(days=max(1, int(round(ttm_days))))
    rows = enrich_surface_legs(raw_legs, atm_iv, default_expiry, strategy_quote, cache_quote, option_spec)
    puts, calls = split_by_right(rows)

    candidates: list[dict[str, Any]] = []
    for pair in adjacent_verticals(puts, calls, min_mid_credit=STRATEGY_MIN_CREDIT):
        combo = vertical_combo(
            pair,
            name_symbol=symbol,
            underlying=underlying,
            spot=spot,
            forward=forward,
            ttm_days=ttm_days,
            risk_free_rate=DEFAULT_RISK_FREE_RATE,
            dividend_yield=implied_q,
        )
        candidates.append(
            {
                "combo": combo,
                "gate_legs": [gate_leg(pair.short, "SELL", spot), gate_leg(pair.long, "BUY", spot)],
                "mid_credit": pair.mid_credit,
                "nat_credit": pair.nat_credit,
                "legs": spread_legs(pair),
                "expiry": pair.short["expiry"],
            }
        )
    return candidates


# --- Evaluation rows -------------------------------------------------------------------------------


def per_lot_risk(risk: Any) -> tuple[float, float, int]:
    """(ES99 tail loss per lot floored at $1, net EV per lot, lots to request from the gate)."""
    return tail_loss_per_lot(risk.es_99_usd), risk.net_ev_usd, max(1, risk.recommended_contracts)


def candidate_row(
    cand: Mapping[str, Any],
    risk: Any,
    decision: GateDecision,
    dte: int,
    per_lot_es99: float,
    per_lot_ev: float,
) -> dict[str, Any]:
    return {
        "combo": cand["combo"].name,
        "mid_credit": round(cand["mid_credit"], 3),
        "nat_credit": round(cand["nat_credit"], 3),
        "dte": dte,
        "net_ev_per_lot": round(per_lot_ev, 2),
        "es99_per_lot": round(per_lot_es99, 2),
        "ev_to_es99": round(per_lot_ev / per_lot_es99, 4),
        "pop_pct": round(risk.prob_profit * 100.0, 1),
        "recommended_qty": risk.recommended_contracts,
        "gate_approved": decision.approved,
        "approved_qty": decision.approved_qty,
        "expected_ev_usd": round(per_lot_ev * decision.approved_qty, 2),
        "gate_reasons": decision.reasons,
        "gate_warnings": decision.warnings,
    }


def rank_key(row: Mapping[str, Any]) -> tuple[bool, float, float]:
    """Tail-budget-aware ranking: approval, expected $EV at the approved size, then EV per $ of ES99."""
    return (row["gate_approved"], row["expected_ev_usd"], row["ev_to_es99"])


def clip_preview_rows(clips: Sequence[Any]) -> list[dict[str, Any]]:
    return [
        {
            "clip_id": c.clip_id,
            "lots": c.lots,
            "mid_credit": c.net_mid_price,
            "natural_credit": c.net_natural_price,
            "passive_limit": c.current_limit_price,
        }
        for c in clips
    ]
