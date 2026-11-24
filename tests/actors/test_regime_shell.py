import pytest

pytest.importorskip("nautilus_trader")
pytestmark = pytest.mark.nautilus

import src.actors.regime_actor as regime_module  # noqa: E402
from src.common import keys  # noqa: E402
from tests.factories import NOW_NS  # noqa: E402
from tests.fakes import FakeActorPorts  # noqa: E402
from tests.nautilus_objects import bar, iid, quote_tick, timer_event  # noqa: E402

VIX_BARS, VIX3M_BARS = "^VIX.CBOE-1-MINUTE-LAST-EXTERNAL", "^VIX3M.CBOE-1-MINUTE-LAST-EXTERNAL"


@pytest.fixture
def make_actor(monkeypatch, store):
    def make(**config):
        monkeypatch.setattr(regime_module, "make_store", lambda url: store)
        actor = regime_module.MacroRegimeActor(
            regime_module.MacroRegimeActorConfig(vix_id=iid("^VIX.CBOE"), vix3m_id=iid("^VIX3M.CBOE"), **config)
        )
        actor.ports = FakeActorPorts(now_ns=NOW_NS + 7)
        actor.wall_clock_ns = lambda: NOW_NS
        return actor

    return make


def test_start_subscribes_per_feed_choice(make_actor):
    actor = make_actor(use_quotes=False)
    actor.on_start()
    assert actor.ports.quote_subscriptions == []
    assert actor.ports.bar_subscriptions == [VIX_BARS, VIX3M_BARS]
    assert actor.ports.timers == [("regime_eval", 15, actor.on_timer)]

    both = make_actor()
    both.on_start()
    assert both.ports.quote_subscriptions == ["^VIX.CBOE", "^VIX3M.CBOE"]


def test_levels_from_bars_and_quotes_publish_a_signal(make_actor, store):
    actor = make_actor()
    actor.on_bar(bar(VIX_BARS, 15.0, NOW_NS - 60_000_000_000))
    actor.on_quote(quote_tick("^VIX3M.CBOE", 16.9, 17.1, NOW_NS - 1_000_000_000))

    actor.on_timer(timer_event(NOW_NS))

    assert store.get_json_value(keys.REGIME_LATEST)["ratio"] == 0.8824
    ((name, signal),) = actor.ports.published
    assert name == "MacroRegimeSignal"
    assert (signal.vix, signal.vix3m, signal.short_vol_allowed) == (15.0, pytest.approx(17.0), True)
    assert (signal.ts_event, signal.ts_init) == (NOW_NS, NOW_NS + 7)


def test_no_signal_without_both_levels(make_actor):
    actor = make_actor()
    actor.on_timer(timer_event(NOW_NS))
    assert actor.ports.published == []
    assert actor.ports.infos == ["MacroRegimeActor timer tick: vix=None, vix3m=None"]
