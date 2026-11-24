import math

import numpy as np
import pytest
import QuantLib as ql
from scipy.differentiate import derivative

from src.quant.black_scholes import (
    black_vega,
    bs_greeks,
    bsm_implied_vols,
    bsm_prices,
    extrinsic_value,
    intrinsic_value,
)

S, K, R, Q, SIGMA = 700.0, 690.0, 0.0425, 0.0125, 0.18
DAYS = 30
T = DAYS / 365.0  # Actual/365 Fixed year fraction of the QuantLib reference set-up
TODAY = ql.Date(24, ql.November, 2026)


@pytest.fixture
def analytic_option():
    """Reference: QuantLib EuropeanOption on flat curves, priced by AnalyticEuropeanEngine."""

    def make(is_call, q=Q):
        day_count = ql.Actual365Fixed()
        process = ql.BlackScholesMertonProcess(
            ql.QuoteHandle(ql.SimpleQuote(S)),
            ql.YieldTermStructureHandle(ql.FlatForward(TODAY, q, day_count)),
            ql.YieldTermStructureHandle(ql.FlatForward(TODAY, R, day_count)),
            ql.BlackVolTermStructureHandle(ql.BlackConstantVol(TODAY, ql.NullCalendar(), SIGMA, day_count)),
        )
        payoff = ql.PlainVanillaPayoff(ql.Option.Call if is_call else ql.Option.Put, K)
        option = ql.EuropeanOption(payoff, ql.EuropeanExercise(TODAY + DAYS))
        option.setPricingEngine(ql.AnalyticEuropeanEngine(process))
        return option

    with ql.SavedSettings():
        ql.Settings.instance().evaluationDate = TODAY
        yield make


def delta_at(is_call, sigma=SIGMA, ttm=T):
    return bs_greeks(S, K, ttm, R, Q, sigma, is_call)["delta"]


@pytest.mark.parametrize("is_call", [True, False])
def test_price_and_first_order_greeks_match_the_analytic_engine(analytic_option, is_call):
    g = bs_greeks(S, K, T, R, Q, SIGMA, is_call)
    option = analytic_option(is_call)

    assert g["price"] == pytest.approx(option.NPV(), rel=1e-10)
    assert g["delta"] == pytest.approx(option.delta(), rel=1e-10)
    assert g["gamma"] == pytest.approx(option.gamma(), rel=1e-10)
    assert g["vega_1v"] == pytest.approx(option.vega() * 0.01, rel=1e-10)
    assert g["theta_day"] == pytest.approx(option.theta() / 365.25, rel=1e-10)


@pytest.mark.parametrize("is_call", [True, False])
def test_vanna_and_charm_are_numerical_derivatives_of_delta(is_call):
    g = bs_greeks(S, K, T, R, Q, SIGMA, is_call)
    d_sigma = derivative(np.vectorize(lambda sigma: delta_at(is_call, sigma=sigma), otypes=[float]), SIGMA, initial_step=0.05)
    d_ttm = derivative(np.vectorize(lambda ttm: delta_at(is_call, ttm=ttm), otypes=[float]), T, initial_step=T / 2)

    assert g["vanna"] == pytest.approx(0.01 * float(d_sigma.df), rel=1e-6)
    # Charm is a calendar-time derivative: time passing shortens the maturity
    assert g["charm_day"] == pytest.approx(-float(d_ttm.df) / 365.25, rel=1e-6)


def test_put_call_parity_for_price_delta_and_shared_greeks():
    call = bs_greeks(S, K, T, R, Q, SIGMA, True)
    put = bs_greeks(S, K, T, R, Q, SIGMA, False)

    assert call["price"] - put["price"] == pytest.approx(S * math.exp(-Q * T) - K * math.exp(-R * T), abs=1e-9)
    assert call["delta"] - put["delta"] == pytest.approx(math.exp(-Q * T), abs=1e-12)
    assert call["gamma"] == pytest.approx(put["gamma"], rel=1e-12)
    assert call["vega_1v"] == pytest.approx(put["vega_1v"], rel=1e-12)
    assert call["vanna"] == pytest.approx(put["vanna"], rel=1e-12)


@pytest.mark.parametrize(
    "spot, strike, ttm, sigma, is_call, price, delta",
    [
        (700.0, 690.0, 0.0, 0.2, True, 10.0, 1.0),
        (700.0, 690.0, 0.0, 0.2, False, 0.0, 0.0),
        (680.0, 690.0, 0.0, 0.2, False, 10.0, -1.0),
        (680.0, 690.0, 0.5, 0.0, True, 0.0, 0.0),
        (0.0, 690.0, 0.5, 0.2, True, 0.0, 0.0),
        (0.0, 690.0, 0.5, 0.2, False, 690.0, -1.0),
        (700.0, 0.0, 0.5, 0.2, True, 700.0, 1.0),
    ],
)
def test_expired_or_zero_vol_options_are_worth_intrinsic(spot, strike, ttm, sigma, is_call, price, delta):
    g = bs_greeks(spot, strike, ttm, R, Q, sigma, is_call)
    assert g["price"] == pytest.approx(price)
    assert g["delta"] == delta
    assert g["gamma"] == g["vega_1v"] == g["theta_day"] == g["vanna"] == g["charm_day"] == 0.0


def test_intrinsic_and_extrinsic_value():
    assert intrinsic_value(700.0, 690.0, True) == 10.0
    assert intrinsic_value(700.0, 690.0, False) == 0.0
    assert intrinsic_value(680.0, 690.0, False) == 10.0
    assert extrinsic_value(12.5, 700.0, 690.0, True) == pytest.approx(2.5)
    assert extrinsic_value(8.0, 700.0, 690.0, True) == 0.0  # Below intrinsic: floored
    assert extrinsic_value(3.0, 700.0, 690.0, False) == pytest.approx(3.0)


def test_vectorised_prices_match_the_analytic_engine(analytic_option):
    prices = bsm_prices([S, S], [K, K], [T, T], [R, R], [SIGMA, SIGMA], np.array(["c", "p"]), q=Q)
    np.testing.assert_allclose(prices, [analytic_option(True).NPV(), analytic_option(False).NPV()], rtol=1e-10)


def test_vectorised_prices_default_to_no_dividend(analytic_option):
    assert float(bsm_prices(S, K, T, R, SIGMA, "c")) == pytest.approx(analytic_option(True, q=0.0).NPV(), rel=1e-10)


def test_implied_vols_invert_engine_prices(analytic_option):
    prices = [analytic_option(True).NPV(), analytic_option(False).NPV()]
    spot_ex_dividend = S * math.exp(-Q * T)  # The solver convention: dividend-discounted spot, true rate
    vols = bsm_implied_vols(prices, spot_ex_dividend, K, T, R, np.array(["c", "p"]))
    np.testing.assert_allclose(vols, [SIGMA, SIGMA], atol=1e-8)


def test_prices_outside_the_no_arbitrage_bounds_have_no_implied_vol():
    below_intrinsic = S - K * math.exp(-R * T) - 1.0
    above_spot = S + 1.0
    vols = bsm_implied_vols([below_intrinsic, above_spot], S, K, T, R, np.array(["c", "c"]))
    assert np.isnan(vols).all()


def test_black_vega_on_the_carry_forward_is_the_engine_vega(analytic_option):
    forward = S * math.exp((R - Q) * T)
    assert black_vega(forward, K, T, SIGMA, math.exp(-R * T)) == pytest.approx(analytic_option(True).vega(), rel=1e-10)
