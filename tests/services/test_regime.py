import pytest

from src.common import keys
from src.common.market import BarObs, LatestValue, QuoteObs
from src.services.regime import RegimeService, RegimeSettings, regime_state
from tests.factories import NOW_NS, ago

VIX, VIX3M = "^VIX.CBOE", "^VIX3M.CBOE"
SEC = 1_000_000_000


def bar(iid, close, ts_ns):
    return BarObs(iid, close, close, close, close, 0.0, ts_ns)


@pytest.fixture
def service(store, recording_log):
    return RegimeService(RegimeSettings(vix_id=VIX, vix3m_id=VIX3M), store=store, log=recording_log)


def test_no_signal_until_both_levels_are_known(service, fake_valkey, recording_log):
    service.on_quote(QuoteObs(VIX, 14.9, 15.1, ago(1.0)))
    assert service.evaluate(NOW_NS, NOW_NS) is None
    recording_log.info.assert_called_once_with("MacroRegimeActor timer tick: vix=15.0, vix3m=None")
    assert fake_valkey.dbsize() == 0


def test_contango_allows_short_vol_and_publishes(service, store):
    service.on_quote(QuoteObs(VIX, 14.9, 15.1, ago(2.0)))
    service.on_bar(bar(VIX3M, 17.0, ago(30.0)))

    state = service.evaluate(event_ts_ns=NOW_NS, wall_ns=NOW_NS + 5)

    assert state.short_vol_allowed and not state.is_stale
    assert state.ratio == pytest.approx(15.0 / 17.0)
    assert store.get_json_value(keys.REGIME_LATEST) == {
        "vix": 15.0,
        "vix3m": 17.0,
        "ratio": 0.8824,
        "short_vol_allowed": True,
        "is_stale": False,
        "max_age_secs": 30.0,
        "ts_ns": NOW_NS + 5,
        "event_ts_ns": NOW_NS,
    }


def test_backwardation_blocks_short_vol(service):
    service.on_bar(bar(VIX, 18.0, NOW_NS))
    service.on_bar(bar(VIX3M, 17.0, NOW_NS))
    state = service.evaluate(NOW_NS, NOW_NS)
    assert state.ratio > 1.0 and not state.short_vol_allowed


def test_stale_levels_block_short_vol_even_in_contango(service):
    service.on_bar(bar(VIX, 15.0, ago(91.0)))
    service.on_bar(bar(VIX3M, 17.0, NOW_NS))
    state = service.evaluate(NOW_NS, NOW_NS)
    assert state.is_stale and not state.short_vol_allowed
    assert state.max_age_secs == pytest.approx(91.0)


def test_newest_observation_wins_across_quotes_and_bars(service):
    service.on_bar(bar(VIX, 15.0, NOW_NS))
    service.on_quote(QuoteObs(VIX, 19.9, 20.1, ago(5.0)))  # Older: ignored
    assert service.vix.value == 15.0
    service.on_quote(QuoteObs(VIX, 15.9, 16.1, NOW_NS))  # Same timestamp: accepted
    assert service.vix.value == pytest.approx(16.0)


def test_unusable_inputs_are_ignored(service):
    service.on_quote(QuoteObs(VIX, 0.0, 15.0, NOW_NS))
    service.on_quote(QuoteObs(VIX, 15.2, 15.0, NOW_NS))
    service.on_bar(bar(VIX3M, 0.0, NOW_NS))
    service.on_bar(bar("SPY.ARCA", 700.0, NOW_NS))
    assert (service.vix.value, service.vix3m.value) == (None, None)


@pytest.mark.parametrize("age, stale", [(90.0, False), (90.5, True)])
def test_staleness_boundary(age, stale):
    vix = LatestValue(15.0, NOW_NS - int(age * SEC))
    vix3m = LatestValue(17.0, NOW_NS)
    assert regime_state(vix, vix3m, NOW_NS, 90.0, 1.0).is_stale is stale


def test_zero_vix3m_level_gives_no_state():
    assert regime_state(LatestValue(15.0, NOW_NS), LatestValue(0.0, NOW_NS), NOW_NS, 90.0, 1.0) is None
