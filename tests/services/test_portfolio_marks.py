import math

import pytest
import QuantLib as ql

from src.common.market import OptionSpec
from src.quant.black_scholes import bs_greeks
from src.services.portfolio_marks import (
    book_stress_grid,
    linear_stress_leg,
    option_exposure,
    smile_iv,
    stress_leg_pnl,
)
from src.services.portfolio_tracking import AttributionTracker, SleeveTracker
from tests.factories import DAY_NS, NOW_NS, SVI_PARAMS, surface_payload

R = 0.0425
T_SURF = 24 / 365.25
SURFACE = surface_payload()  # spot 700, forward 701, 24 days, implied q 1.25%
PUT_690 = OptionSpec("SPY P690", "SPY.ARCA", 690.0, False, NOW_NS + 24 * DAY_NS)


def quantlib_slice_vol(strike, forward=701.0):
    """The surface's SVI slice as a QuantLib smile section (QuantLib orders the parameters a, b, sigma, rho, m)."""
    a, b, rho, m, sigma = SVI_PARAMS
    return ql.SviSmileSection(T_SURF, forward, [a, b, sigma, rho, m]).volatility(strike)


# --- Smile lookup ----------------------------------------------------------------------------------


def test_same_expiry_uses_the_slice_forward():
    assert smile_iv(SURFACE, 700.0, 0.0125, 690.0, T_SURF, R) == pytest.approx(quantlib_slice_vol(690.0), rel=1e-12)


def test_other_expiries_map_their_own_atm_onto_the_slice_atm():
    ttm = 2 * T_SURF
    own_forward = 700.0 * math.exp((R - 0.0125) * ttm)
    assert smile_iv(SURFACE, 700.0, 0.0125, own_forward, ttm, R) == pytest.approx(quantlib_slice_vol(701.0), rel=1e-12)


@pytest.mark.parametrize(
    "surface, spot, strike",
    [(surface_payload(svi_params=[]), 700.0, 690.0), (surface_payload(ttm_days=0.0), 700.0, 690.0), (SURFACE, 0.0, 690.0)],
)
def test_smile_lookup_needs_a_usable_slice(surface, spot, strike):
    assert smile_iv(surface, spot, 0.0125, strike, T_SURF, R) is None


# --- Exposures -------------------------------------------------------------------------------------


def exposure(qty=-1.0, bench_spot=700.0, surface=SURFACE, live_mid=None, rv=None):
    return option_exposure(
        PUT_690, 100.0, qty, NOW_NS, spot=700.0, surface=surface, rv=rv or {},
        bench_spot=bench_spot, bench_atm_iv=0.2, r=R, live_mid=live_mid,
    )


def test_short_put_exposure_signs_and_scaling():
    exp = exposure()
    g = bs_greeks(700.0, 690.0, T_SURF, R, 0.0125, exp["stress_leg"]["iv"], False)

    assert exp["greeks"]["delta"] == pytest.approx(-100.0 * g["delta"])
    assert exp["greeks"]["delta"] > 0.0 and exp["greeks"]["vega_1v"] < 0.0 and exp["greeks"]["theta_day"] > 0.0
    assert exp["greeks"]["gamma_1pct"] == pytest.approx(-100.0 * g["gamma"] * 7.0)
    assert (exp["mark"], exp["mark_source"]) == (pytest.approx(g["price"]), "MODEL")
    assert exp["stress_leg"]["iv"] == pytest.approx(smile_iv(SURFACE, 700.0, 0.0125, 690.0, T_SURF, R))


def test_beta_weighting_and_live_marks():
    exp = exposure(bench_spot=1_400.0, live_mid=5.5)
    assert exp["greeks"]["beta_delta"] == pytest.approx(0.5 * exp["greeks"]["delta"])
    assert (exp["mark"], exp["mark_source"]) == (5.5, "LIVE_MID")
    assert exposure(bench_spot=0.0)["greeks"]["beta_delta"] == exposure()["greeks"]["delta"]


@pytest.mark.parametrize(
    "surface, rv, iv",
    [
        (surface_payload(svi_params=[], atm_iv=0.21), {}, 0.21),
        (surface_payload(svi_params=[], atm_iv=None), {"rv_20d": 0.17}, 0.17),
        ({}, {}, 0.2),
    ],
)
def test_iv_fallbacks_without_a_smile(surface, rv, iv):
    assert exposure(surface=surface, rv=rv)["stress_leg"]["iv"] == iv


# --- Stress ----------------------------------------------------------------------------------------


def test_linear_leg_stress():
    leg = linear_stress_leg(10.0, 5.0, 100.0)
    assert stress_leg_pnl(leg, 1.1, 0.05, 7 / 365.25, R) == pytest.approx(500.0)


def test_option_leg_has_no_pnl_without_shocks_or_time():
    leg = exposure()["stress_leg"]
    assert stress_leg_pnl(leg, 1.0, 0.0, 0.0, R) == pytest.approx(0.0, abs=1e-9)


def test_book_grid_shape_and_direction_for_a_long_put():
    grid = book_stress_grid([exposure(qty=1.0)["stress_leg"]], R)
    assert (len(grid), len(grid[0])) == (7, 6)
    assert all(grid[0][j] > grid[-1][j] for j in range(6))  # Long put gains when spot falls
    assert all(grid[3][j] < grid[3][j + 1] for j in range(5))  # ... and when vol rises


def test_empty_book_grid_is_zero():
    assert book_stress_grid([], R) == [[0.0] * 6 for _ in range(7)]


# --- Attribution and sleeve ------------------------------------------------------------------------

GREEKS = {"delta_sh": 100.0, "gamma_1pct": 2.0, "vega_1v": -50.0, "theta_day": 10.0}


def test_attribution_explains_pnl_with_baseline_greeks():
    tracker = AttributionTracker()
    first = tracker.update("2026-11-24", ("book",), 700.0, 0.16, 1_000.0, GREEKS, NOW_NS)
    assert first == {
        "actual_pnl_change_usd": 0.0, "delta_pnl_usd": 0.0, "gamma_pnl_usd": 0.0, "vega_pnl_usd": 0.0,
        "theta_pnl_usd": 0.0, "residual_pnl_usd": 0.0, "since_ts_ns": NOW_NS,
    }

    later = tracker.update("2026-11-24", ("book",), 707.0, 0.17, 1_800.0, {**GREEKS, "delta_sh": 0.0}, NOW_NS + DAY_NS // 2)
    assert later["delta_pnl_usd"] == 700.0  # 100 sh x $7, baseline Greeks are kept
    assert later["gamma_pnl_usd"] == 7.0  # 0.5 x 2 x 1% x $7
    assert later["vega_pnl_usd"] == -50.0  # -50 $/pt x 1 pt
    assert later["theta_pnl_usd"] == 5.0  # Half a day
    assert later["actual_pnl_change_usd"] == 800.0
    assert later["residual_pnl_usd"] == pytest.approx(800.0 - (700.0 + 7.0 - 50.0 + 5.0))


@pytest.mark.parametrize("session, book", [("2026-11-25", ("book",)), ("2026-11-24", ("new book",))])
def test_attribution_rebases_on_a_new_day_or_book(session, book):
    tracker = AttributionTracker()
    tracker.update("2026-11-24", ("book",), 700.0, 0.16, 1_000.0, GREEKS, NOW_NS)
    rebased = tracker.update(session, book, 707.0, 0.17, 1_800.0, GREEKS, NOW_NS + 10)
    assert rebased["actual_pnl_change_usd"] == 0.0
    assert rebased["since_ts_ns"] == NOW_NS + 10


def test_attribution_waits_for_a_benchmark_price():
    assert AttributionTracker().update("d", (), 0.0, 0.16, 0.0, GREEKS, NOW_NS)["since_ts_ns"] == 0


def test_sleeve_tracks_high_water_mark_and_drawdown():
    sleeve = SleeveTracker(100_000.0)
    up = sleeve.snapshot(session_realized_usd=0.0, unrealized_pnl_usd=2_000.0, open_positions=1, now_ns=NOW_NS)
    assert (up["equity"], up["high_water_mark"], up["drawdown_pct"]) == (102_000.0, 102_000.0, 0.0)

    down = sleeve.snapshot(500.0, -3_500.0, 1, NOW_NS + 1)
    assert (down["equity"], down["high_water_mark"]) == (97_000.0, 102_000.0)
    assert down["drawdown_pct"] == pytest.approx(-4.902, abs=1e-3)
    assert (down["realized_total_usd"], down["unrealized_pnl_usd"], down["ts_ns"]) == (500.0, -3_500.0, NOW_NS + 1)


def test_sleeve_restores_carry_and_high_water_mark():
    sleeve = SleeveTracker(100_000.0)
    sleeve.restore({"prior_carry_usd": 750.0, "high_water_mark": 101_000.0})
    assert (sleeve.prior_carry_usd, sleeve.high_water_mark) == (750.0, 101_000.0)
    sleeve.restore({})
    assert (sleeve.prior_carry_usd, sleeve.high_water_mark) == (0.0, 0.0)
