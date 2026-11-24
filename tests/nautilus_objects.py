"""Real NautilusTrader value objects for the shell tests (import only after importorskip("nautilus_trader"))."""
from __future__ import annotations

from types import SimpleNamespace

from nautilus_trader.model import Bar, BarType, InstrumentId, Price, Quantity, QuoteTick


def iid(text: str) -> InstrumentId:
    return InstrumentId.from_str(text)


def quote_tick(instrument: str, bid: float, ask: float, ts_ns: int, bid_size: int = 10, ask_size: int = 10) -> QuoteTick:
    return QuoteTick(
        instrument_id=iid(instrument),
        bid_price=Price.from_str(f"{bid:.2f}"),
        ask_price=Price.from_str(f"{ask:.2f}"),
        bid_size=Quantity.from_int(bid_size),
        ask_size=Quantity.from_int(ask_size),
        ts_event=ts_ns,
        ts_init=ts_ns,
    )


def bar(bar_type: str, close: float, ts_ns: int, volume: int = 0) -> Bar:
    px = Price.from_str(f"{close:.2f}")
    return Bar(
        bar_type=BarType.from_str(bar_type),
        open=px,
        high=px,
        low=px,
        close=px,
        volume=Quantity.from_int(volume),
        ts_event=ts_ns,
        ts_init=ts_ns,
    )


def timer_event(ts_ns: int) -> SimpleNamespace:
    return SimpleNamespace(ts_event=ts_ns)
