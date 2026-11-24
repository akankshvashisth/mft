from datetime import date
from types import SimpleNamespace
from unittest import mock

import pytest

pytest.importorskip("nautilus_trader")
pytestmark = pytest.mark.nautilus

import src.risk.pre_trade_gate as gate_module  # noqa: E402
import src.strategies.vrp_income as shell_module  # noqa: E402
from src.common import keys  # noqa: E402
from tests.factories import NOW_NS, risk_assessment, rv_payload, surface_leg, surface_payload  # noqa: E402
from tests.fakes import FakeGate, FakePricer, FakeStrategyPorts, stub_calendar  # noqa: E402
from tests.nautilus_objects import iid, quote_tick, timer_event  # noqa: E402

SPY = "SPY.ARCA"
P690, P685, P680 = (f"SPY 261218P{k * 1000:08d}.SMART" for k in (690, 685, 680))


@pytest.fixture
def make_strategy(monkeypatch, store):
    store.set_json(
        keys.surface_key(SPY),
        surface_payload(
            legs=[
                surface_leg(P680, 680.0, "P", 1.0, 1.2),
                surface_leg(P685, 685.0, "P", 1.4, 1.6),
                surface_leg(P690, 690.0, "P", 2.0, 2.2),
            ]
        ),
    )
    store.set_json(keys.rv_key(SPY), rv_payload(rv_20d=0.13))
    monkeypatch.setattr(shell_module, "make_store", lambda url: store)
    monkeypatch.setattr(gate_module, "default_event_calendar", stub_calendar)

    def make(paper=False):
        strategy = shell_module.VrpIncomeStrategy(
            shell_module.VrpIncomeStrategyConfig(
                underlying_id=iid(SPY), option_ids=(iid(P690), iid(P685)), paper_routing_enabled=paper
            )
        )
        strategy.ports = FakeStrategyPorts(
            now_ns=NOW_NS, instruments={i: SimpleNamespace(id=i) for i in (P690, P685)}
        )
        strategy.vrp_engine.gate = FakeGate()
        strategy.vrp_engine.pricer = FakePricer(lambda combo: risk_assessment(combo_name=combo.name))
        strategy.wall_clock_ns = lambda: NOW_NS
        return strategy

    return make


def test_start_subscribes_and_schedules(make_strategy):
    strategy = make_strategy()
    strategy.on_start()
    assert strategy.ports.quote_subscriptions == [SPY, P690, P685]
    assert strategy.ports.timers == [("vrp_strategy_eval", 15, strategy.on_timer)]
    assert strategy.ports.infos == [
        f"VrpIncomeStrategy started in SHADOW_DRY_RUN mode watching {SPY} + 2 options (0 open spreads restored)"
    ]


def test_shadow_timer_publishes_with_the_new_york_date(make_strategy, store):
    strategy = make_strategy()
    strategy.on_start()
    strategy.on_quote(quote_tick(SPY, 699.95, 700.05, NOW_NS))

    strategy.on_timer(timer_event(NOW_NS - 2))

    payload = store.get_json_value(keys.VRP_LATEST)
    assert (payload["action"], payload["event_ts_ns"], payload["ts_ns"]) == ("SHADOW_SIGNAL_READY", NOW_NS - 2, NOW_NS)
    assert {call["as_of"] for call in strategy.vrp_engine.gate.calls} == {date(2026, 11, 24)}


def test_live_entry_goes_through_the_nautilus_gateway(make_strategy):
    strategy = make_strategy(paper=True)
    strategy.on_start()
    strategy.on_quote(quote_tick(SPY, 699.95, 700.05, NOW_NS))

    strategy.on_timer(timer_event(NOW_NS))

    (order,) = strategy.ports.submitted
    assert order is strategy.ports.limit_orders[0]
    assert len(strategy.ports.added_instruments) == 1  # The OptionSpread instrument
    assert list(strategy.vrp_engine.working) == [order.client_order_id]


def test_order_events_are_forwarded(make_strategy):
    strategy = make_strategy()
    strategy.vrp_engine = engine = mock.Mock()

    strategy.on_order_filled(SimpleNamespace(client_order_id="O-1", last_qty=2, last_px=0.57, commission=-1.3))
    strategy.on_order_filled(SimpleNamespace(client_order_id="O-1", last_qty=1, last_px=0.57))
    strategy.on_order_canceled(SimpleNamespace(client_order_id="O-1"))
    strategy.on_order_expired(SimpleNamespace(client_order_id="O-2"))
    strategy.on_order_rejected(SimpleNamespace(client_order_id="O-3", reason="margin"))
    strategy.on_order_denied(SimpleNamespace(client_order_id="O-4"))
    strategy.on_order_modify_rejected(SimpleNamespace(client_order_id="O-5"))
    strategy.on_stop()

    assert engine.mock_calls == [
        mock.call.on_fill("O-1", 2.0, 0.57, -1.3),
        mock.call.on_fill("O-1", 1.0, 0.57, None),
        mock.call.on_terminal("O-1", "CANCELLED"),
        mock.call.on_terminal("O-2", "EXPIRED"),
        mock.call.on_terminal("O-3", "REJECTED", "margin"),
        mock.call.on_terminal("O-4", "DENIED", ""),
        mock.call.on_modify_rejected("O-5"),
        mock.call.stop(),
    ]
