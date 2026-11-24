import pytest

from src.common import keys
from src.common.market import BarObs
from src.services.surface_metrics import SurfaceMetricsService, SurfaceMetricsSettings, Technicals
from tests.factories import NOW_NS, UNDERLYING, regime_payload, rv_payload, surface_payload


@pytest.fixture
def make_service(store, recording_log):
    def make(surface=None, rv=None, regime=None, **settings):
        if surface is not None:
            store.set_json(keys.surface_key(UNDERLYING), surface)
        store.set_json(keys.rv_key(UNDERLYING), rv or rv_payload(rv_20d=0.13))
        store.set_json(keys.REGIME_LATEST, regime or regime_payload())
        return SurfaceMetricsService(SurfaceMetricsSettings(UNDERLYING, **settings), store=store, log=recording_log)

    return make


def flat_surface(**overrides):
    """Surface without SVI parameters: 25-delta vols come from ATM vol and skew."""
    spec = {"svi_params": None, "atm_iv": 0.16, "skew_25d": 0.02}
    spec.update(overrides)
    return surface_payload(**spec)


def test_no_surface_no_metrics(make_service, fake_valkey):
    service = make_service()
    assert service.evaluate(NOW_NS, NOW_NS) is None
    assert not fake_valkey.exists(keys.metrics_key(UNDERLYING))


def test_metrics_payload_from_atm_and_skew_fallback(make_service, store, recording_log):
    payload = make_service(surface=flat_surface()).evaluate(event_ts_ns=NOW_NS - 9, wall_ns=NOW_NS)

    assert store.get_json_value(keys.metrics_key(UNDERLYING)) == payload
    assert payload["rr_25d"] == pytest.approx(0.02)
    assert payload["bf_25d"] == pytest.approx(0.0, abs=1e-9)
    assert payload["vrp_spread"] == pytest.approx(0.03)
    assert (payload["atm_iv_pct"], payload["rr_25d_pct"]) == (50.0, 50.0)  # Too little history
    assert payload["vix_term_ratio"] == 0.8824
    assert payload["strategy_bias"] == "SELECTIVE_RELATIVE_VALUE_SPREADS"  # BF below 0.5 vol pt
    assert payload["technicals"] == {"sma_20": 700.0, "sma_50": 700.0, "rsi_14": 50.0, "trend_state": "RANGE_OR_TRANSITION"}
    assert (payload["ts_ns"], payload["event_ts_ns"]) == (NOW_NS, NOW_NS - 9)
    assert recording_log.info.call_args_list[0].args[0].startswith(f"SurfaceMetrics {UNDERLYING} ATM=16.00%")


def test_svi_smile_gives_positive_risk_reversal_and_butterfly(make_service):
    payload = make_service(surface=surface_payload()).evaluate(NOW_NS, NOW_NS)
    assert payload["rr_25d"] > 0.0  # Negative SVI rho: puts over calls
    assert payload["bf_25d"] > 0.0


@pytest.mark.parametrize(
    "rv, regime",
    [
        (rv_payload(rv_20d=0.13), regime_payload(ratio=1.05)),  # Backwardation
        (rv_payload(rv_20d=0.20), regime_payload()),  # Negative VRP
    ],
)
def test_defensive_bias(make_service, rv, regime):
    payload = make_service(surface=flat_surface(), rv=rv, regime=regime).evaluate(NOW_NS, NOW_NS)
    assert payload["strategy_bias"] == "DEFENSIVE_OR_LONG_CONVEXITY"


def test_percentiles_start_after_five_observations(make_service, store):
    service = make_service(surface=flat_surface())
    pcts = []
    for atm in (0.10, 0.11, 0.12, 0.13, 0.14, 0.09):
        store.set_json(keys.surface_key(UNDERLYING), flat_surface(atm_iv=atm))
        pcts.append(service.evaluate(NOW_NS, NOW_NS)["atm_iv_pct"])
    assert pcts == [50.0, 50.0, 50.0, 50.0, 100.0, 16.67]


def test_history_length_is_bounded(make_service):
    service = make_service(surface=flat_surface(), history_maxlen=3)
    for _ in range(5):
        service.evaluate(NOW_NS, NOW_NS)
    assert len(service.atm_hist) == len(service.vrp_hist) == 3


def test_technicals_drive_the_trend_and_rsi(make_service):
    payload = make_service(surface=flat_surface()).evaluate(
        NOW_NS, NOW_NS, Technicals(sma_20=690.0, sma_50=680.0, rsi_14=0.55)
    )
    assert payload["technicals"] == {"sma_20": 690.0, "sma_50": 680.0, "rsi_14": 55.0, "trend_state": "UPTREND"}


def _bar(ts, iid=UNDERLYING, aggregation="DAY", step=1):
    return BarObs(iid, 1.0, 1.0, 1.0, 1.0, 0.0, ts, aggregation, step)


def test_only_new_daily_bars_of_the_underlying_feed_indicators(make_service):
    service = make_service()
    assert service.accept_indicator_bar(_bar(100))
    assert not service.accept_indicator_bar(_bar(100))  # Duplicate (warm start overlap)
    assert not service.accept_indicator_bar(_bar(50))
    assert not service.accept_indicator_bar(_bar(200, aggregation="MINUTE"))
    assert not service.accept_indicator_bar(_bar(200, iid="QQQ.NASDAQ"))
    assert not service.accept_indicator_bar(_bar(200, step=2))
    assert service.accept_indicator_bar(_bar(200))
