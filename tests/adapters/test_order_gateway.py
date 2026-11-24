from types import SimpleNamespace

import pytest

pytest.importorskip("nautilus_trader")
pytestmark = pytest.mark.nautilus

from nautilus_trader.model import OrderSide, TimeInForce  # noqa: E402

from src.adapters.order_gateway import NautilusMarketView, NautilusOrderGateway  # noqa: E402
from src.common.market import QuoteObs  # noqa: E402
from src.strategies.vrp_engine import OrderView  # noqa: E402
from tests.fakes import FakeStrategyPorts  # noqa: E402

P690, P685 = "SPY 261218P00690000.SMART", "SPY 261218P00685000.SMART"
LEGS = [{"instrument_id": P690, "ratio": 1}, {"instrument_id": P685, "ratio": -1}]


@pytest.fixture
def ports():
    return FakeStrategyPorts()


def test_prepare_builds_one_spread_instrument_and_day_limit_orders(ports):
    gateway = NautilusOrderGateway(ports)

    entry = gateway.prepare(LEGS, "CREDIT_OPEN", 3, 0.57)
    exit_ = gateway.prepare(LEGS, "DEBIT_CLOSE", 2, 0.63)

    (spread,) = ports.added_instruments
    assert (entry.order_key, entry.spread_id, exit_.order_key) == ("O-1", str(spread.id), "O-2")
    first, second = ports.limit_orders
    assert (first.side, second.side) == (OrderSide.SELL, OrderSide.BUY)
    assert first.time_in_force == TimeInForce.DAY
    assert (first.quantity, first.price) == (spread.make_qty(3), spread.make_price(0.57))
    assert ports.submitted == []

    gateway.submit(entry)
    assert ports.submitted == [entry.handle]


def test_order_views_modifications_and_cancels(ports):
    gateway = NautilusOrderGateway(ports)
    prepared = gateway.prepare(LEGS, "CREDIT_OPEN", 3, 0.57)
    order = prepared.handle

    assert gateway.order_view("O-1") == OrderView(is_open=True, is_closed=False, status="INITIALIZED")
    assert gateway.order_view("unknown") is None

    assert gateway.modify_price("O-1", 0.52)
    spread = ports.added_instruments[0]
    assert ports.modified == [(order, spread.make_price(0.52))]
    assert not gateway.modify_price("unknown", 0.52)

    gateway.cancel("O-1")
    gateway.cancel("unknown")
    assert ports.cancelled == [order]


def test_modify_needs_the_spread_instrument(ports):
    gateway = NautilusOrderGateway(ports)
    gateway.prepare(LEGS, "CREDIT_OPEN", 3, 0.57)
    ports.instruments_by_id.clear()
    assert not gateway.modify_price("O-1", 0.52)
    assert ports.modified == []


def test_market_view(ports):
    ports.quotes[P690] = SimpleNamespace(instrument_id=P690, bid_price=2.0, ask_price=2.2, bid_size=5, ask_size=None, ts_event=10)
    ports.instruments_by_id[P690] = SimpleNamespace(id=P690)  # Not an OptionContract
    ports.open_positions = [
        SimpleNamespace(instrument_id="S1", signed_qty=-2),
        SimpleNamespace(instrument_id="S1", signed_qty=-1),
        SimpleNamespace(instrument_id="S2", signed_qty=5),
    ]
    market = NautilusMarketView(ports)

    assert market.cache_quote(P690) == QuoteObs(P690, 2.0, 2.2, 10, 5.0, None)
    assert market.cache_quote(P685) is None
    assert market.option_spec(P690) is None
    assert market.net_position_qty("S1") == -3.0

    market.ensure_subscribed(P685)  # Unknown instrument: nothing to subscribe to yet
    market.ensure_subscribed(P690)
    market.ensure_subscribed(P690)
    assert ports.quote_subscriptions == [P690]
