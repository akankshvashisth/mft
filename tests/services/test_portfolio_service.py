from unittest import mock

import pytest

from src.common import keys
from src.common.market import OptionSpec, QuoteObs
from src.services.portfolio import (
    OPTION,
    OPTION_SPREAD,
    OTHER,
    InstrumentInfo,
    PortfolioService,
    PortfolioSettings,
    PositionView,
)
from tests.factories import DAY_NS, NOW_NS, ago, positions_payload, rv_payload, surface_payload
from tests.fakes import FakeBookMarket

SPY = "SPY.ARCA"
P690, P685, SPREAD = "SPY 261218P00690000.SMART", "SPY 261218P00685000.SMART", "(1)SPY P690_(-1)SPY P685"
EXPIRY_NS = NOW_NS + 24 * DAY_NS  # Same maturity as the factory surface


def option_info(iid, strike):
    return InstrumentInfo(OPTION, 100.0, OptionSpec(iid, SPY, strike, False, EXPIRY_NS))


def quote(bid, ask, ts_ns=None):
    return QuoteObs("x", bid, ask, ago(1.0) if ts_ns is None else ts_ns)


@pytest.fixture
def market():
    return FakeBookMarket(
        instruments={
            SPY: InstrumentInfo(OTHER, 1.0),
            P690: option_info(P690, 690.0),
            P685: option_info(P685, 685.0),
            SPREAD: InstrumentInfo(OPTION_SPREAD, 100.0),
        },
        quotes={SPY: quote(699.9, 700.1), P690: quote(5.4, 5.6), P685: quote(4.1, 4.3)},
        symbols={"SPY": [SPY]},
    )


@pytest.fixture
def make_service(store, market, recording_log):
    store.set_json(keys.surface_key(SPY), surface_payload())
    store.set_json(keys.RV_UNIVERSE, {SPY: rv_payload(rv_20d=0.13)})

    def make(**settings):
        return PortfolioService(PortfolioSettings(benchmark_id=SPY, **settings), store=store, log=recording_log, market=market)

    return make


def row(payload, instrument):
    return next(r for r in payload["positions"] if r["instrument"] == instrument)


def test_empty_book_publishes_flat_risk_and_sleeve(make_service, store, fake_valkey):
    service = make_service()
    with mock.patch.object(fake_valkey, "pipeline", wraps=fake_valkey.pipeline) as pipeline:
        payload = service.evaluate([], 0.0, NOW_NS)

    pipeline.assert_called_once()  # Risk and sleeve are published in one batch
    assert store.get_json_value(keys.PORTFOLIO_RISK) == payload
    assert store.get_json_value(keys.POSITIONS_LATEST) == payload["sleeve"]
    assert (payload["open_positions"], payload["benchmark_spot"], payload["total_pnl_usd"]) == (0, 700.0, 0.0)
    assert set(payload["greeks"].values()) == {0.0}
    assert payload["risk"]["stress_grid_usd"] == [[0.0] * 6 for _ in range(7)]
    assert (payload["risk"]["book_var_99_usd"], payload["risk"]["book_es_99_usd"]) == (0.0, 0.0)
    assert payload["sleeve"] == positions_payload(NOW_NS)


def test_share_position_marks_live_and_stresses_linearly(make_service):
    payload = make_service().evaluate([PositionView(SPY, 100.0, 690.0)], 0.0, NOW_NS)

    position = row(payload, SPY)
    assert (position["mark_px"], position["mark_source"], position["pnl_usd"], position["delta_sh"]) == (
        700.0, "LIVE_MID", 1_000.0, 100.0,
    )
    assert payload["greeks"]["beta_delta_shares"] == 100.0
    grid = payload["risk"]["stress_grid_usd"]
    assert grid[0] == [-7_000.0] * 6 and grid[3] == [0.0] * 6 and grid[6] == [7_000.0] * 6
    assert payload["risk"]["worst_7d_stress_usd"] == -7_000.0
    risk = payload["risk"]
    assert risk["book_es_99_usd"] > 0.0 and risk["book_var_99_usd"] > 0.0
    assert risk["book_1d_linear_es_99_usd"] > 0.0 and risk["book_1d_linear_var_99_usd"] > 0.0
    assert (payload["sleeve"]["equity"], payload["sleeve"]["high_water_mark"]) == (101_000.0, 101_000.0)


@pytest.mark.parametrize(
    "instrument, quotes, source, mark",
    [
        (SPY, {}, "ACTOR_SPOT", 700.0),  # Surface spot when there is no quote
        (SPY, {SPY: quote(699.9, 700.1, ts_ns=ago(301.0))}, "ACTOR_SPOT", 700.0),  # Quote too old
        ("IWM.ARCA", {}, "ENTRY_PRICE", 690.0),
    ],
)
def test_share_mark_fallbacks(make_service, market, instrument, quotes, source, mark):
    market.quotes = quotes
    payload = make_service().evaluate([PositionView(instrument, 10.0, 690.0)], 0.0, NOW_NS)
    assert (row(payload, instrument)["mark_source"], row(payload, instrument)["mark_px"]) == (source, mark)


def test_short_option_is_marked_at_its_live_mid_with_model_greeks(make_service):
    payload = make_service().evaluate([PositionView(P690, -1.0, 5.0)], 0.0, NOW_NS)

    position = row(payload, P690)
    assert (position["mark_px"], position["mark_source"], position["pnl_usd"]) == (5.5, "LIVE_MID", -50.0)
    assert position["delta_sh"] > 0.0 and position["vega_1v"] < 0.0 and position["theta_day"] > 0.0
    assert payload["greeks"]["net_vega_1v_usd"] == position["vega_1v"]


def test_registered_spread_is_decomposed_into_option_legs(make_service, store):
    store.set_json(keys.spread_legs_key(SPREAD), [{"instrument_id": P690, "ratio": 1}, {"instrument_id": P685, "ratio": -1}])
    payload = make_service().evaluate([PositionView(SPREAD, -2.0, 1.2)], 0.0, NOW_NS)

    position = row(payload, SPREAD)
    assert position["mark_px"] == pytest.approx(1.3)  # 5.5 - 4.2
    assert (position["mark_source"], position["pnl_usd"]) == ("LIVE_MID", pytest.approx(-20.0))
    assert position["vega_1v"] < 0.0  # Short the nearer-the-money put
    assert payload["risk"]["worst_7d_stress_usd"] < 0.0


def test_spread_with_one_unquoted_leg_has_mixed_marks(make_service, store, market):
    store.set_json(keys.spread_legs_key(SPREAD), [{"instrument_id": P690, "ratio": 1}, {"instrument_id": P685, "ratio": -1}])
    del market.quotes[P685]
    payload = make_service().evaluate([PositionView(SPREAD, -2.0, 1.2)], 0.0, NOW_NS)
    assert row(payload, SPREAD)["mark_source"] == "MIXED"


@pytest.mark.parametrize(
    "legs",
    [None, "not a list", [{"instrument_id": "UNKNOWN", "ratio": 1}], [{"ratio": 1}], [{"instrument_id": SPY, "ratio": 1}]],
)
def test_spread_without_usable_legs_is_not_modelled(make_service, store, market, legs):
    if legs is not None:
        store.set_json(keys.spread_legs_key(SPREAD), legs)
    market.quotes[SPREAD] = quote(1.25, 1.35)
    payload = make_service().evaluate([PositionView(SPREAD, -2.0, 1.2)], 0.0, NOW_NS)

    position = row(payload, SPREAD)
    assert (position["mark_source"], position["mark_px"], position["pnl_usd"]) == ("UNMODELED_SPREAD", 1.3, pytest.approx(-20.0))
    assert position["delta_sh"] == 0.0


def test_option_on_an_unpriced_underlying(make_service, market):
    market.instruments["XYZ P10"] = InstrumentInfo(OPTION, 100.0, OptionSpec("XYZ P10", "XYZ.ARCA", 10.0, False, EXPIRY_NS))
    payload = make_service().evaluate([PositionView("XYZ P10", 1.0, 0.5)], 0.0, NOW_NS)
    assert (row(payload, "XYZ P10")["mark_source"], row(payload, "XYZ P10")["pnl_usd"]) == ("UNPRICED_UNDERLYING", 0.0)


def test_underlying_spot_falls_back_to_a_live_quote(make_service, store, market):
    store.delete(keys.surface_key(SPY))
    store.delete(keys.RV_UNIVERSE)
    payload = make_service().evaluate([PositionView(P690, -1.0, 5.0)], 0.0, NOW_NS)
    assert row(payload, P690)["mark_source"] == "LIVE_MID"  # Priced: spot 700 came from the SPY quote


def test_realized_pnl_and_restored_sleeve_state(make_service, store):
    store.set_json(keys.POSITIONS_LATEST, positions_payload(realized_total_usd=500.0, high_water_mark=101_000.0))
    service = make_service()
    service.restore()

    payload = service.evaluate([PositionView(SPY, 0.0, 700.0, realized_pnl=25.0)], 75.0, NOW_NS)

    assert payload["realized_pnl_usd"] == 600.0  # Carry 500 + closed 75 + open 25
    sleeve = payload["sleeve"]
    assert (sleeve["equity"], sleeve["high_water_mark"]) == (100_600.0, 101_000.0)
    assert sleeve["drawdown_pct"] == pytest.approx(-0.396, abs=1e-3)


def test_attribution_runs_across_evaluations(make_service, store, market):
    service = make_service()
    position = [PositionView(SPY, 100.0, 690.0)]
    assert service.evaluate(position, 0.0, NOW_NS)["attribution"]["actual_pnl_change_usd"] == 0.0

    store.set_json(keys.surface_key(SPY), surface_payload(spot=707.0))
    market.quotes[SPY] = quote(706.9, 707.1)
    attribution = service.evaluate(position, 0.0, NOW_NS + 60 * 1_000_000_000)["attribution"]
    assert (attribution["actual_pnl_change_usd"], attribution["delta_pnl_usd"]) == (700.0, 700.0)
    assert attribution["since_ts_ns"] == NOW_NS


def test_unmodeled_spread_sets_flag(make_service, store, market):
    """An OPTION_SPREAD with no info['legs'] sets has_unmodeled_positions=True."""
    store.set_json(keys.spread_legs_key("UNKNOWN_SPREAD"), "not valid")  # No usable legs
    market.quotes["UNKNOWN_SPREAD"] = quote(1.25, 1.35)
    market.instruments["UNKNOWN_SPREAD"] = InstrumentInfo(OPTION_SPREAD, 100.0)
    payload = make_service().evaluate([PositionView("UNKNOWN_SPREAD", -2.0, 1.2)], 0.0, NOW_NS)
    assert payload["risk"]["has_unmodeled_positions"] is True
    assert "UNKNOWN_SPREAD" in payload["risk"]["unmodeled_positions"]


def test_registered_spread_does_not_set_unmodeled_flag(make_service, store):
    """A spread with registered legs in Valkey is modeled and not unmodeled."""
    store.set_json(keys.spread_legs_key(SPREAD),
                   [{"instrument_id": P690, "ratio": 1}, {"instrument_id": P685, "ratio": -1}])
    payload = make_service().evaluate([PositionView(SPREAD, -2.0, 1.2)], 0.0, NOW_NS)
    assert payload["risk"]["has_unmodeled_positions"] is False
    assert payload["risk"]["unmodeled_positions"] == []


def test_horizon_var_es_is_nonlinear_14d(make_service, store):
    """book_es_99_usd is the 14-day nonlinear MC ES99 (not the 1-day linear)."""
    store.set_json(keys.spread_legs_key(SPREAD),
                   [{"instrument_id": P690, "ratio": 1}, {"instrument_id": P685, "ratio": -1}])
    payload = make_service().evaluate([PositionView(SPREAD, -2.0, 1.2)], 0.0, NOW_NS)
    risk = payload["risk"]
    # The 14d MC ES99 should be larger than the 1d linear ES99 (longer horizon + non-normal tails)
    assert risk["book_es_99_usd"] > 0.0
    assert risk["book_1d_linear_es_99_usd"] > 0.0
    store.set_json(keys.surface_key("QQQ.NASDAQ"), surface_payload("QQQ.NASDAQ"))
    store.set_text(keys.surface_key("BAD.ARCA"), "not json")
    assert sorted(make_service().surfaces_by_symbol()) == ["QQQ", "SPY"]
