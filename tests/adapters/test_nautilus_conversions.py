import pytest

pytest.importorskip("nautilus_trader")
pytestmark = pytest.mark.nautilus

from nautilus_trader.model import (  # noqa: E402
    AggressorSide,
    Bar,
    BarType,
    InstrumentId,
    Price,
    Quantity,
    QuoteTick,
    TradeId,
    TradeTick,
)

from src.adapters.nautilus import (  # noqa: E402
    bar_obs,
    daily_bar_type,
    instrument_kind,
    minute_bar_type,
    quote_obs,
    to_instrument_id,
    trade_obs,
)
from src.services.portfolio import OTHER  # noqa: E402
from tests.factories import NOW_NS  # noqa: E402

SPY = InstrumentId.from_str("SPY.ARCA")


def test_quote_tick_becomes_a_quote_observation():
    tick = QuoteTick(
        instrument_id=SPY,
        bid_price=Price.from_str("699.95"),
        ask_price=Price.from_str("700.05"),
        bid_size=Quantity.from_int(10),
        ask_size=Quantity.from_int(20),
        ts_event=NOW_NS,
        ts_init=NOW_NS,
    )
    obs = quote_obs(tick)
    assert (obs.instrument_id, obs.bid, obs.ask, obs.ts_ns) == ("SPY.ARCA", 699.95, 700.05, NOW_NS)
    assert (obs.bid_size, obs.ask_size) == (10.0, 20.0)
    assert obs.mid == pytest.approx(700.0)


def test_trade_tick_becomes_a_trade_observation():
    tick = TradeTick(
        instrument_id=InstrumentId.from_str("MESZ6.CME"),
        price=Price.from_str("7020.25"),
        size=Quantity.from_int(1),
        aggressor_side=AggressorSide.BUYER,
        trade_id=TradeId("T-1"),
        ts_event=NOW_NS,
        ts_init=NOW_NS,
    )
    obs = trade_obs(tick)
    assert (obs.instrument_id, obs.price, obs.ts_ns) == ("MESZ6.CME", 7020.25, NOW_NS)


@pytest.mark.parametrize(
    "bar_type, aggregation",
    [("SPY.ARCA-1-DAY-LAST-EXTERNAL", "DAY"), ("SPY.ARCA-1-MINUTE-LAST-EXTERNAL", "MINUTE")],
)
def test_bar_conversion_detects_daily_bars(bar_type, aggregation):
    bar = Bar(
        bar_type=BarType.from_str(bar_type),
        open=Price.from_str("100.00"),
        high=Price.from_str("101.00"),
        low=Price.from_str("99.00"),
        close=Price.from_str("100.50"),
        volume=Quantity.from_int(1_000),
        ts_event=NOW_NS,
        ts_init=NOW_NS,
    )
    obs = bar_obs(bar)
    assert (obs.instrument_id, obs.open, obs.high, obs.low, obs.close) == ("SPY.ARCA", 100.0, 101.0, 99.0, 100.5)
    assert (obs.volume, obs.ts_ns, obs.aggregation, obs.step) == (1_000.0, NOW_NS, aggregation, 1)
    assert obs.is_daily is (aggregation == "DAY")


def test_bar_types_and_ids():
    assert str(minute_bar_type("SPY.ARCA")) == "SPY.ARCA-1-MINUTE-LAST-EXTERNAL"
    assert str(daily_bar_type(SPY)) == "SPY.ARCA-1-DAY-LAST-EXTERNAL"
    assert to_instrument_id(SPY) is SPY
    assert to_instrument_id("SPY.ARCA") == SPY


def test_anything_but_option_contracts_and_spreads_is_other():
    assert instrument_kind(None) == OTHER
    assert instrument_kind(object()) == OTHER
