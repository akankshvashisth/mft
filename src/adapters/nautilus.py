"""Conversions from NautilusTrader value types to the plain observations used by the services."""
from __future__ import annotations

from typing import Any

from nautilus_trader.model import (
    BarAggregation,
    BarSpecification,
    BarType,
    InstrumentId,
    OptionContract,
    OptionSpread,
    PriceType,
)

from src.common.market import BarObs, QuoteObs, TradeObs
from src.services.portfolio import OPTION, OPTION_SPREAD, OTHER

MINUTE_LAST = BarSpecification(1, BarAggregation.MINUTE, PriceType.LAST)
DAY_LAST = BarSpecification(1, BarAggregation.DAY, PriceType.LAST)


def to_instrument_id(value: Any) -> InstrumentId:
    return value if isinstance(value, InstrumentId) else InstrumentId.from_str(str(value))


def minute_bar_type(instrument_id: Any) -> BarType:
    return BarType(to_instrument_id(instrument_id), MINUTE_LAST)


def daily_bar_type(instrument_id: Any) -> BarType:
    return BarType(to_instrument_id(instrument_id), DAY_LAST)


def _optional_float(value: Any) -> float | None:
    return float(value) if value is not None else None


def quote_obs(tick: Any) -> QuoteObs:
    return QuoteObs(
        instrument_id=str(tick.instrument_id),
        bid=float(tick.bid_price),
        ask=float(tick.ask_price),
        ts_ns=int(tick.ts_event),
        bid_size=_optional_float(getattr(tick, "bid_size", None)),
        ask_size=_optional_float(getattr(tick, "ask_size", None)),
    )


def trade_obs(tick: Any) -> TradeObs:
    return TradeObs(instrument_id=str(tick.instrument_id), price=float(tick.price), ts_ns=int(tick.ts_event))


def aggregation_name(aggregation: Any) -> str:
    if aggregation == BarAggregation.DAY:
        return "DAY"
    if aggregation == BarAggregation.MINUTE:
        return "MINUTE"
    return str(getattr(aggregation, "name", aggregation))


def bar_obs(bar: Any) -> BarObs:
    spec = bar.bar_type.spec
    return BarObs(
        instrument_id=str(bar.bar_type.instrument_id),
        open=float(bar.open),
        high=float(bar.high),
        low=float(bar.low),
        close=float(bar.close),
        volume=float(bar.volume),
        ts_ns=int(bar.ts_event),
        aggregation=aggregation_name(spec.aggregation),
        step=int(spec.step),
    )


def instrument_kind(instrument: Any) -> str:
    if isinstance(instrument, OptionContract):
        return OPTION
    if isinstance(instrument, OptionSpread):
        return OPTION_SPREAD
    return OTHER
