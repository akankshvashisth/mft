from unittest import mock

import pytest

pytest.importorskip("nautilus_trader")
pytestmark = pytest.mark.nautilus

import src.actors.surface_metrics_actor as metrics_module  # noqa: E402
from src.common import keys  # noqa: E402
from tests.factories import DAY_NS, NOW_NS, seed_healthy_inputs  # noqa: E402
from tests.fakes import FakeActorPorts  # noqa: E402
from tests.nautilus_objects import bar, iid, timer_event  # noqa: E402

DAILY = "SPY.ARCA-1-DAY-LAST-EXTERNAL"


def rising_daily_bars(n=60):
    first = NOW_NS - (n + 1) * DAY_NS
    return [bar(DAILY, 600.0 + i, first + i * DAY_NS) for i in range(n)]


def catalog_with(bars):
    catalog = mock.Mock(spec=["bars"])
    catalog.bars.return_value = bars
    return catalog


@pytest.fixture
def actor(monkeypatch, healthy_store):
    monkeypatch.setattr(metrics_module, "make_store", lambda url: healthy_store)
    actor = metrics_module.SurfaceRegimeMetricsActor(metrics_module.SurfaceRegimeMetricsActorConfig(underlying_id=iid("SPY.ARCA")))
    actor.catalog_factory = mock.Mock(return_value=catalog_with(rising_daily_bars()))
    actor.ports = FakeActorPorts(now_ns=NOW_NS)
    actor.wall_clock_ns = lambda: NOW_NS
    return actor


def test_warm_started_indicators_feed_the_trend(actor, healthy_store):
    actor.on_start()
    actor.catalog_factory.return_value.bars.assert_called_once_with(bar_types=[DAILY])
    assert actor.ports.bar_subscriptions == [DAILY]
    assert actor.ports.timers == [("surface_metrics_eval", 15, actor.on_timer)]
    assert actor.sma_20.initialized and actor.sma_50.initialized

    actor.on_timer(timer_event(NOW_NS))

    technicals = healthy_store.get_json_value(keys.metrics_key("SPY.ARCA"))["technicals"]
    assert technicals["trend_state"] == "UPTREND"  # Surface spot 700 > SMA20 > SMA50 of a rising series
    assert technicals["sma_20"] == pytest.approx(649.5)
    assert technicals["sma_50"] == pytest.approx(634.5)


def test_minute_and_duplicate_bars_do_not_touch_the_indicators(actor):
    actor.on_start()
    before = float(actor.sma_20.value)
    actor.on_bar(bar("SPY.ARCA-1-MINUTE-LAST-EXTERNAL", 999.0, NOW_NS))
    actor.on_bar(rising_daily_bars()[-1])
    assert float(actor.sma_20.value) == before

    actor.on_bar(bar(DAILY, 700.0, NOW_NS))
    assert float(actor.sma_20.value) != before


def test_warm_start_failure_is_logged(actor):
    actor.catalog_factory = mock.Mock(side_effect=OSError("no catalog"))
    actor.on_start()
    assert actor.ports.warnings == ["SurfaceRegimeMetricsActor indicator warm-start skipped: no catalog"]
    assert not actor.sma_20.initialized


def test_payload_is_written_without_indicators(actor, healthy_store):
    seed_healthy_inputs(healthy_store)
    actor.on_timer(timer_event(NOW_NS))
    assert healthy_store.get_json_value(keys.metrics_key("SPY.ARCA"))["technicals"]["rsi_14"] == 50.0
