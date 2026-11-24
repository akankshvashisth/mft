from types import SimpleNamespace

import pytest

pytest.importorskip("nautilus_trader")
pytestmark = pytest.mark.nautilus

import src.actors.futures_actor as futures_module  # noqa: E402
from src.common import keys  # noqa: E402
from tests.factories import DAY_NS, NOW_NS, ago, surface_payload  # noqa: E402
from tests.fakes import FakeActorPorts  # noqa: E402
from tests.nautilus_objects import bar, iid, quote_tick, timer_event  # noqa: E402

SPY, MES, SPX, MNQ = "SPY.ARCA", "MESZ6.CME", "^SPX.CBOE", "MNQZ6.CME"


@pytest.fixture
def make_actor(monkeypatch, store):
    store.set_json(keys.surface_key(SPY), surface_payload(SPY))

    def make(cached=(SPY, MES, SPX)):
        monkeypatch.setattr(futures_module, "make_store", lambda url: store)
        actor = futures_module.FuturesContextActor(
            futures_module.FuturesContextActorConfig(spy_id=iid(SPY), mes_id=iid(MES), spx_id=iid(SPX), mnq_id=iid(MNQ))
        )
        instruments = {i: SimpleNamespace(id=i) for i in cached}
        if MES in instruments:
            instruments[MES].expiration_ns = NOW_NS + 30 * DAY_NS
        actor.ports = FakeActorPorts(now_ns=NOW_NS, instruments=instruments)
        actor.wall_clock_ns = lambda: NOW_NS
        return actor

    return make


def test_start_subscribes_cached_instruments_and_index_bars(make_actor):
    actor = make_actor()
    actor.on_start()
    assert actor.ports.quote_subscriptions == [SPY, MES]
    assert actor.ports.bar_subscriptions == [f"{SPX}-1-MINUTE-LAST-EXTERNAL", f"{SPY}-1-MINUTE-LAST-EXTERNAL"]
    assert actor.ports.infos == [f"FuturesContextActor subscribed to {SPY}, {MES}, {SPX} (1-min bars)"]


def test_missing_index_feed_is_a_warning(make_actor):
    actor = make_actor(cached=(SPY, MES))
    actor.on_start()
    assert actor.ports.bar_subscriptions == []
    assert actor.ports.warnings == [
        "FuturesContextActor has no SPX index feed; the parity-vs-futures cross-check will be unavailable"
    ]


def test_market_data_to_cross_check(make_actor, store):
    actor = make_actor()
    actor.on_start()
    actor.on_quote(quote_tick(SPY, 699.99, 700.01, ago(1.0)))
    actor.on_quote(quote_tick(MES, 7019.75, 7020.25, ago(1.0)))
    actor.on_bar(bar(f"{SPX}-1-MINUTE-LAST-EXTERNAL", 7000.0, ago(60.0)))
    actor.on_bar(bar(f"{SPY}-1-MINUTE-LAST-EXTERNAL", 700.0, ago(60.0)))

    actor.on_timer(timer_event(NOW_NS))

    payload = store.get_json_value(keys.FUTURES_LATEST)
    assert (payload["cross_check_valid"], payload["spx_spy_ratio"], payload["mes_ttm_days"]) == (True, 10.0, 30.0)
