"""Real NautilusTrader value objects for the shell tests (import only after importorskip("nautilus_trader"))."""
from __future__ import annotations

from types import SimpleNamespace

from nautilus_trader.model import (
    AssetClass,
    Bar,
    BarType,
    Currency,
    InstrumentId,
    OptionContract,
    OptionKind,
    Price,
    Quantity,
    QuoteTick,
    Symbol,
)


def iid(text: str) -> InstrumentId:
    return InstrumentId.from_str(text)


def option_contract(instrument_id: str, strike: float, underlying: str = "SPY") -> OptionContract:
    instrument = iid(instrument_id)
    return OptionContract(
        instrument_id=instrument,
        raw_symbol=Symbol.from_str(str(instrument.symbol)),
        asset_class=AssetClass.EQUITY,
        underlying=underlying,
        option_kind=OptionKind.PUT,
        strike_price=Price.from_str(str(strike)),
        currency=Currency.from_str("USD"),
        activation_ns=0,
        expiration_ns=0,
        price_precision=2,
        price_increment=Price.from_str("0.01"),
        multiplier=Quantity.from_int(100),
        lot_size=Quantity.from_int(1),
        ts_event=0,
        ts_init=0,
    )


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
