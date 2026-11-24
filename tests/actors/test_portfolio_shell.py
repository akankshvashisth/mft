from types import SimpleNamespace

import pytest

pytest.importorskip("nautilus_trader")
pytestmark = pytest.mark.nautilus

import src.actors.portfolio_actor as portfolio_module  # noqa: E402
from src.common import keys  # noqa: E402
from tests.factories import NOW_NS, ago, positions_payload, rv_payload, surface_payload  # noqa: E402
from tests.fakes import FakeActorPorts  # noqa: E402
from tests.nautilus_objects import iid, timer_event  # noqa: E402

SPY = "SPY.ARCA"


@pytest.fixture
def actor(monkeypatch, store):
    store.set_json(keys.surface_key(SPY), surface_payload())
    store.set_json(keys.RV_UNIVERSE, {SPY: rv_payload()})
    store.set_json(keys.POSITIONS_LATEST, positions_payload(realized_total_usd=500.0))
    monkeypatch.setattr(portfolio_module, "make_store", lambda url: store)
    actor = portfolio_module.PortfolioAnalyticsActor(portfolio_module.PortfolioAnalyticsActorConfig(benchmark_id=iid(SPY)))
    actor.ports = FakeActorPorts(
        now_ns=NOW_NS,
        instruments={SPY: SimpleNamespace(id=SPY, multiplier=1)},
        quotes={SPY: SimpleNamespace(instrument_id=SPY, bid_price=699.9, ask_price=700.1, bid_size=1, ask_size=1, ts_event=ago(1.0))},
        positions_open=[SimpleNamespace(instrument_id=SPY, signed_qty=100, avg_px_open=690.0, realized_pnl=0.0)],
        positions_closed=[SimpleNamespace(instrument_id=SPY, signed_qty=0, avg_px_open=0.0, realized_pnl=50.0)],
    )
    actor.wall_clock_ns = lambda: NOW_NS
    return actor


def test_start_restores_the_sleeve(actor):
    actor.on_start()
    assert actor.ports.timers == [("portfolio_analytics_eval", 15, actor.on_timer)]
    assert "realized_carry=$500.00" in actor.ports.infos[0]


def test_timer_marks_cache_positions(actor, store):
    actor.on_start()
    actor.on_timer(timer_event(NOW_NS))

    payload = store.get_json_value(keys.PORTFOLIO_RISK)
    (row,) = payload["positions"]
    assert (row["instrument"], row["mark_source"], row["pnl_usd"]) == (SPY, "LIVE_MID", 1_000.0)
    assert payload["realized_pnl_usd"] == 550.0
    assert store.get_json_value(keys.POSITIONS_LATEST)["equity"] == 101_550.0
