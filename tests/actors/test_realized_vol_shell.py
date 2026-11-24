from types import SimpleNamespace

import pytest

pytest.importorskip("nautilus_trader")
pytestmark = pytest.mark.nautilus

import src.actors.realized_vol_actor as rv_module  # noqa: E402
from src.common import keys  # noqa: E402
from tests.factories import NOW_NS, daily_ohlc_frame  # noqa: E402
from tests.fakes import FakeActorPorts  # noqa: E402
from tests.nautilus_objects import iid, timer_event  # noqa: E402

UNIVERSE = ("SPY.ARCA", "QQQ.NASDAQ", "IWM.ARCA")


@pytest.fixture
def actor(monkeypatch, store):
    monkeypatch.setattr(rv_module, "make_store", lambda url: store)
    actor = rv_module.RealizedVolActor(
        rv_module.RealizedVolActorConfig(primary_id=iid("SPY.ARCA"), universe_ids=tuple(iid(u) for u in UNIVERSE))
    )
    history = {"SPY.ARCA": daily_ohlc_frame(n=30)}
    actor.history_loader_factory = lambda catalog_path, log=None: history.get
    actor.ports = FakeActorPorts(now_ns=NOW_NS, instruments={"QQQ.NASDAQ": SimpleNamespace(id="QQQ.NASDAQ")})
    actor.wall_clock_ns = lambda: NOW_NS
    return actor


def test_start_warm_starts_and_subscribes_primary_and_cached_instruments(actor):
    actor.on_start()
    assert actor.ports.quote_subscriptions == ["SPY.ARCA", "QQQ.NASDAQ"]
    assert actor.ports.bar_subscriptions == ["SPY.ARCA-1-MINUTE-LAST-EXTERNAL", "QQQ.NASDAQ-1-MINUTE-LAST-EXTERNAL"]
    assert actor.ports.timers == [("realized_vol_eval", 15, actor.on_timer)]
    assert actor.ports.infos == ["RealizedVolActor warm-started 1/3 universe symbols from catalog"]


def test_timer_publishes_the_primary_snapshot(actor, store):
    actor.on_start()
    actor.on_timer(timer_event(NOW_NS - 3))

    payload = store.get_json_value(keys.rv_key("SPY.ARCA"))
    ((name, snap),) = actor.ports.published
    assert name == "RealizedVolSnapshot"
    assert snap.underlying_id == iid("SPY.ARCA")
    assert snap.rv_20d == pytest.approx(payload["rv_20d"], abs=1e-6)
    assert (snap.ts_event, snap.ts_init) == (NOW_NS - 3, NOW_NS)
