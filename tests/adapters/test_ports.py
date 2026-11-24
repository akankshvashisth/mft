from datetime import timedelta
from types import SimpleNamespace
from unittest import mock

import pytest

pytest.importorskip("nautilus_trader")
pytestmark = pytest.mark.nautilus

from src.adapters.ports import ActorPorts, PortsLog, StrategyPorts  # noqa: E402
from tests.fakes import FakeActorPorts  # noqa: E402


def component():
    """Nautilus component double: calls are recorded by the mock and cache lookups echo their argument."""
    comp = mock.Mock()
    comp.clock.timestamp_ns.return_value = 123
    comp.cache.instrument.side_effect = lambda iid: f"instrument:{iid}"
    comp.cache.instruments.return_value = iter(["a", "b"])
    comp.cache.quote.side_effect = lambda iid: f"quote:{iid}"
    comp.cache.positions_open.side_effect = lambda instrument_id=None: [f"open:{instrument_id}"]
    comp.cache.positions_closed.return_value = ["closed"]
    comp.cache.order.side_effect = lambda coid: f"order:{coid}"
    comp.order_factory.limit.side_effect = lambda **kwargs: kwargs
    return comp


def test_actor_ports_wrap_the_component():
    comp = component()
    ports = ActorPorts(comp)

    assert ports.now_ns() == 123
    assert ports.instrument("SPY.ARCA") == "instrument:SPY.ARCA"
    assert ports.instruments() == ["a", "b"]
    assert ports.quote("SPY.ARCA") == "quote:SPY.ARCA"
    assert ports.positions_open() == ["open:None"]
    assert ports.positions_open("SPY.ARCA") == ["open:SPY.ARCA"]
    assert ports.positions_closed() == ["closed"]

    ports.subscribe_quotes("SPY.ARCA")
    ports.subscribe_bars("bar-type")

    def callback(event):
        return None

    ports.set_timer("eval", 15, callback)
    (instrument_id,) = comp.subscribe_quotes.call_args.args
    assert str(instrument_id) == "SPY.ARCA"
    comp.subscribe_bars.assert_called_once_with("bar-type")
    comp.clock.set_timer.assert_called_once_with("eval", timedelta(seconds=15), callback=callback)

    ports.info("i")
    ports.warning("w")
    ports.error("e")
    assert comp.log.mock_calls == [mock.call.info("i"), mock.call.warning("w"), mock.call.error("e")]


def test_strategy_ports_order_calls():
    comp = component()
    ports = StrategyPorts(comp)

    assert ports.order("O-1") == "order:O-1"
    assert ports.limit_order("SPREAD", "SELL", 3, 0.57, "DAY") == {
        "instrument_id": "SPREAD", "order_side": "SELL", "quantity": 3, "price": 0.57, "time_in_force": "DAY", "post_only": False,
    }
    ports.submit("order")
    ports.modify_price("order", 0.52)
    ports.cancel("order")
    comp.submit_order.assert_called_once_with("order")
    comp.modify_order.assert_called_once_with("order", price=0.52)
    comp.cancel_order.assert_called_once_with("order")


def test_add_instrument_only_once():
    comp = component()
    comp.cache.instrument.side_effect = None
    comp.cache.instrument.return_value = None
    StrategyPorts(comp).add_instrument(SimpleNamespace(id="SPREAD"))
    comp.cache.instrument.return_value = "known"
    StrategyPorts(comp).add_instrument(SimpleNamespace(id="SPREAD"))
    comp.cache.add_instrument.assert_called_once()


def test_ports_log_follows_ports_swaps():
    owner = SimpleNamespace(ports=FakeActorPorts())
    log = PortsLog(owner)
    log.info("before")
    owner.ports = FakeActorPorts()
    log.warning("after")
    log.error("boom")
    assert (owner.ports.infos, owner.ports.warnings, owner.ports.errors) == ([], ["after"], ["boom"])
