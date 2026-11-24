import math
from unittest import mock

import numpy as np
import pytest
import QuantLib as ql

from src.common import keys
from src.common.market import BarObs, OptionSpec, QuoteObs
from src.quant.black_scholes import bsm_implied_vols
from src.quant.svi import fit_svi_slice, svi_implied_vol
from src.services.surface_metrics import SurfaceMetricsService, SurfaceMetricsSettings
from src.services.vol_surface import (
    OptionSlot,
    VolSurfaceService,
    VolSurfaceSettings,
    aggregate_by_log_moneyness,
    select_slice_options,
)
from tests.factories import DAY_NS, NOW_NS, SVI_PARAMS, ago

SPY = "SPY.ARCA"
SPOT, R, Q_DEFAULT = 700.0, 0.0425, 0.0125
EXPIRY_NS = NOW_NS + 30 * DAY_NS
T = 30 / 365.25
PUT_STRIKES = [650.0 + 5 * i for i in range(11)]   # 650 .. 700
CALL_STRIKES = [705.0 + 5 * i for i in range(10)]  # 705 .. 750


def occ(strike, is_call, root="SPY"):
    return f"{root} 261224{'C' if is_call else 'P'}{int(strike * 1000):08d}.SMART"


def spec(strike, is_call, **overrides):
    fields = dict(
        instrument_id=occ(strike, is_call), underlying=SPY, strike=strike, is_call=is_call,
        expiry_ns=EXPIRY_NS, multiplier=100.0, open_interest=1_000.0,
    )
    fields.update(overrides)
    return OptionSpec(**fields)


SLICE_SPECS = [spec(k, False) for k in PUT_STRIKES] + [spec(k, True) for k in CALL_STRIKES]


def smile_solver(prices, S, K, t, r, flags):
    """Market IVs = the SVI_PARAMS smile around the forward implied by the solver inputs."""
    forward = S[0] * math.exp(r[0] * t[0])
    return svi_implied_vol(np.log(K / forward), t[0], SVI_PARAMS)


def exact_fitter(log_m, ivs, weights, ttm, x0_warm=None):
    return svi_implied_vol(log_m, ttm, SVI_PARAMS), np.array(SVI_PARAMS), 0.0


@pytest.fixture
def make_service(store, recording_log):
    def make(specs=SLICE_SPECS, iv_solver=smile_solver, arb_validator=lambda lm, t, iv: True, svi_fitter=exact_fitter):
        service = VolSurfaceService(
            VolSurfaceSettings(underlying_id=SPY, strike_min=640.0, strike_max=760.0),
            store=store, log=recording_log,
            iv_solver=iv_solver, arb_validator=arb_validator, svi_fitter=svi_fitter,
        )
        service.configure(specs)
        return service

    return make


def quote_everything(service, ts_ns=ago(1.0), skip=()):
    service.on_quote(QuoteObs(SPY, 699.95, 700.05, ts_ns))
    for slot in service.book.slots:
        if slot.instrument_id not in skip:
            service.on_quote(QuoteObs(slot.instrument_id, 1.95, 2.05, ts_ns))


# --- Slice selection -------------------------------------------------------------------------------


def test_select_slice_options_filters_and_orders():
    specs = [
        spec(700.0, False),
        spec(700.0, True),
        spec(690.0, False, open_interest=None),
        spec(695.0, True, multiplier=10.0),  # Adjusted contract
        spec(600.0, False),  # Below the strike range
        spec(700.0, True, instrument_id=occ(700.0, True, root="QQQ")),
    ]
    slots, expiry = select_slice_options(specs, SPY, 640.0, 760.0)
    assert slots == [
        OptionSlot(occ(690.0, False), 690.0, False, 0.0),
        OptionSlot(occ(700.0, True), 700.0, True, 1_000.0),
        OptionSlot(occ(700.0, False), 700.0, False, 1_000.0),
    ]
    assert expiry == EXPIRY_NS
    assert select_slice_options([], SPY, 0.0, 1e9) == ([], None)


def test_configure_keeps_the_configured_slice_when_nothing_is_cached(store):
    settings = VolSurfaceSettings(underlying_id=SPY, option_ids=(occ(690.0, False),), strikes=(690.0,), is_calls=(False,))
    service = VolSurfaceService(settings, store=store)
    assert service.configure([]) == 1
    assert service.book.option_ids == [occ(690.0, False)]
    assert service.current_ttm_years(NOW_NS) == settings.ttm_years


def test_aggregation_weights_shared_strikes():
    grid, ivs, weights, inv = aggregate_by_log_moneyness(
        np.array([0.0, 0.0, 0.05]), np.array([0.20, 0.10, 0.30]), np.array([1.0, 3.0, 2.0])
    )
    np.testing.assert_allclose(grid, [0.0, 0.05])
    np.testing.assert_allclose(ivs, [(0.20 * 1.0 + 0.10 * 3.0) / 4.0, 0.30])
    np.testing.assert_allclose(weights, [4.0, 2.0])
    assert inv.tolist() == [0, 0, 1]


# --- Calibration -----------------------------------------------------------------------------------


def test_calibrated_slice_is_published(make_service, store, recording_log):
    service = make_service()
    quote_everything(service)

    fit = service.evaluate(event_ts_ns=NOW_NS, wall_ns=NOW_NS + 11)

    assert fit is not None and fit.arb_free
    assert (fit.quote_count, fit.expiry_ns) == (21, EXPIRY_NS)
    assert fit.forward == pytest.approx(SPOT * math.exp((R - Q_DEFAULT) * T))
    assert fit.atm_iv == pytest.approx(float(svi_implied_vol(0.0, T, SVI_PARAMS)), abs=0.005)

    payload = store.get_json_value(keys.surface_key(SPY))
    assert payload == fit.payload
    assert (payload["parity_pairs"], payload["implied_q"], payload["ttm_days"]) == (0, Q_DEFAULT, 30.0)
    assert (payload["spot"], payload["spot_age_secs"], payload["quotes"]) == (700.0, 1.0, 21)
    assert payload["svi_params"] == list(SVI_PARAMS)
    assert (payload["ts_ns"], payload["event_ts_ns"]) == (NOW_NS + 11, NOW_NS)
    assert sum(payload["dropped"].values()) == 0
    assert len(payload["log_moneyness"]) == len(payload["ivs"]) == len(payload["raw_ivs"]) == 21

    legs = payload["legs"]
    assert [leg["instrument_id"] for leg in legs] == service.book.option_ids
    assert {leg["right"] for leg in legs[:11]} == {"P"} and {leg["right"] for leg in legs[11:]} == {"C"}
    # Market vols are the model smile: residuals only reflect the 1e-6 log-moneyness grid rounding
    assert all(abs(leg["residual_pts"]) <= 1e-4 and abs(leg["edge_to_half_spread"]) <= 1e-3 for leg in legs)
    assert all(leg["vega"] > 0.0 and leg["open_interest"] == 1_000 for leg in legs)
    assert recording_log.info.call_args.args[0].startswith(f"Surface {SPY} S=700.00")


def test_inversion_uses_the_dividend_discounted_spot_and_the_true_rate(make_service):
    seen = {}

    def recording_solver(prices, S, K, t, r, flags):
        seen.update(S=np.unique(S), r=np.unique(r), flags=list(flags))
        return smile_solver(prices, S, K, t, r, flags)

    service = make_service(iv_solver=recording_solver)
    quote_everything(service)
    service.evaluate(NOW_NS, NOW_NS)

    np.testing.assert_allclose(seen["S"], [SPOT * math.exp(-Q_DEFAULT * T)])
    np.testing.assert_allclose(seen["r"], [R])
    assert seen["flags"] == ["p"] * 11 + ["c"] * 10


def test_parity_pairs_set_the_forward(make_service):
    specs = SLICE_SPECS + [spec(700.0, True)]  # Call and put at 700 with equal mids => F = 700
    service = make_service(specs=specs)
    quote_everything(service)

    fit = service.evaluate(NOW_NS, NOW_NS)

    assert fit.payload["parity_pairs"] == 1
    assert fit.forward == pytest.approx(700.0)
    assert fit.payload["implied_q"] == pytest.approx(R, abs=1e-4)
    assert fit.quote_count == 22
    assert len(fit.payload["log_moneyness"]) == 21  # ATM call and put share one grid point


def test_warm_start_only_after_an_arbitrage_free_fit(make_service):
    starts = []

    def recording_fitter(log_m, ivs, weights, ttm, x0_warm=None):
        starts.append(None if x0_warm is None else list(x0_warm))
        return exact_fitter(log_m, ivs, weights, ttm)

    verdicts = iter([False, True, True])
    service = make_service(svi_fitter=recording_fitter, arb_validator=lambda lm, t, iv: next(verdicts))
    quote_everything(service)
    for _ in range(3):
        service.evaluate(NOW_NS, NOW_NS)

    assert starts == [None, None, list(SVI_PARAMS)]


def test_skew_matches_the_metrics_risk_reversal(make_service, store):
    service = make_service()
    quote_everything(service)
    fit = service.evaluate(NOW_NS, NOW_NS)

    metrics = SurfaceMetricsService(SurfaceMetricsSettings(SPY), store=store).evaluate(NOW_NS, NOW_NS)
    assert fit.skew_25d > 0.0
    assert metrics["rr_25d"] == pytest.approx(fit.skew_25d, abs=1e-4)


# --- Inputs that stop a calibration ----------------------------------------------------------------


def test_missing_or_stale_spot(make_service, recording_log):
    service = make_service()
    assert service.evaluate(NOW_NS, NOW_NS) is None
    assert recording_log.warning.call_args == mock.call("FastVolActor underlying stale/missing (spot=None, age=9999.0s)")

    quote_everything(service, ts_ns=ago(91.0))
    assert service.evaluate(NOW_NS, NOW_NS) is None
    assert recording_log.warning.call_args == mock.call("FastVolActor underlying stale/missing (spot=700.0, age=91.0s)")


def test_too_few_clean_quotes(make_service, recording_log, fake_valkey):
    service = make_service()
    quote_everything(service, skip=service.book.option_ids[4:])
    assert service.evaluate(NOW_NS, NOW_NS) is None
    assert recording_log.warning.call_args.args[0].startswith("FastVolActor insufficient clean quotes (4): ")
    assert not fake_valkey.exists(keys.surface_key(SPY))


def test_stale_option_quotes_are_dropped_and_counted(make_service):
    service = make_service()
    quote_everything(service)
    service.on_quote(QuoteObs(service.book.option_ids[0], 1.95, 2.05, ago(120.0)))

    fit = service.evaluate(NOW_NS, NOW_NS)
    assert fit.quote_count == 20
    assert fit.payload["dropped"]["stale"] == 1


def test_solver_failures_below_five_valid_vols_stop_the_fit(make_service):
    def failing_solver(prices, S, K, t, r, flags):
        ivs = np.full(len(K), np.nan)
        ivs[:4] = 0.2
        return ivs

    service = make_service(iv_solver=failing_solver)
    quote_everything(service)
    assert service.evaluate(NOW_NS, NOW_NS) is None


def test_bars_stand_in_for_quotes(make_service):
    service = make_service()
    option_id = service.book.option_ids[0]
    service.on_bar(BarObs(SPY, 0, 0, 0, 701.0, 0, NOW_NS))
    service.on_bar(BarObs(option_id, 0, 0, 0, 2.0, 0, NOW_NS))
    service.on_bar(BarObs(option_id, 0, 0, 0, 0.0, 0, NOW_NS + 1))  # Ignored

    assert (service.book.spot_mid, service.book.spot_ts_ns) == (701.0, NOW_NS)
    assert service.book.quotes[option_id] == pytest.approx((1.99, 2.01, 2.0, NOW_NS))


def test_quantlib_priced_chain_round_trips_through_inversion_and_fit(make_service):
    """Mids priced off the SVI smile with QuantLib, inverted with QuantLib and calibrated with scipy."""
    service = make_service(iv_solver=bsm_implied_vols, svi_fitter=fit_svi_slice)
    forward = SPOT * math.exp((R - Q_DEFAULT) * T)  # No call/put pair shares a strike: the prior forward
    service.on_quote(QuoteObs(SPY, 699.95, 700.05, ago(1.0)))
    for slot in service.book.slots:
        vol = float(svi_implied_vol(math.log(slot.strike / forward), T, SVI_PARAMS))
        option_type = ql.Option.Call if slot.is_call else ql.Option.Put
        mid = ql.blackFormula(option_type, slot.strike, forward, vol * math.sqrt(T), math.exp(-R * T))
        service.on_quote(QuoteObs(slot.instrument_id, mid - 0.02, mid + 0.02, ago(1.0)))

    fit = service.evaluate(NOW_NS, NOW_NS)

    assert fit is not None and fit.quote_count == 21
    k = np.asarray(fit.payload["log_moneyness"])
    # Payload vols and log-moneyness are rounded to 4 decimals
    np.testing.assert_allclose(fit.payload["raw_ivs"], svi_implied_vol(k, T, SVI_PARAMS), atol=2e-4)
    assert fit.payload["rmse_vol_pts"] < 0.25
    assert fit.atm_iv == pytest.approx(float(svi_implied_vol(0.0, T, SVI_PARAMS)), abs=0.005)
