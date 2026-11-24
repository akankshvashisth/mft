import math

import pytest
import QuantLib as ql

from src.quant.surface_metrics import (
    BIAS_BULLISH_RR,
    BIAS_DEFENSIVE,
    BIAS_SELECTIVE,
    BIAS_SHORT_VRP,
    classify_strategy_bias,
    classify_trend,
    delta25_log_moneyness,
    smile_wings,
)
from tests.factories import SVI_PARAMS

T = 30 / 365.25


def test_delta25_log_moneyness_and_its_vol_floor():
    z_25 = -ql.InverseCumulativeNormal()(0.25)  # Reference quantile from QuantLib (the implementation uses scipy)
    assert delta25_log_moneyness(0.16, 0.25) == pytest.approx(z_25 * 0.16 * 0.5)
    assert delta25_log_moneyness(0.05, 0.25) == pytest.approx(z_25 * 0.10 * 0.5)


def test_symmetric_smile_has_no_risk_reversal_and_positive_butterfly():
    wings = smile_wings(0.2, T, (0.002, 0.05, 0.0, 0.0, 0.1))
    assert wings.rr_25d == pytest.approx(0.0, abs=1e-12)
    assert wings.bf_25d > 0.0


def test_svi_wings_are_the_smile_at_plus_and_minus_k25():
    wings = smile_wings(0.16, T, SVI_PARAMS)

    k_25 = delta25_log_moneyness(0.16, T)
    a, b, rho, m, sigma = SVI_PARAMS
    section = ql.SviSmileSection(T, 1.0, [a, b, sigma, rho, m])  # Forward 1: strike = exp(log-moneyness)
    assert (wings.iv_25p, wings.iv_25c, wings.iv_atm) == (
        pytest.approx(section.volatility(math.exp(-k_25))),
        pytest.approx(section.volatility(math.exp(k_25))),
        pytest.approx(section.volatility(1.0)),
    )
    assert wings.rr_25d > 0.0  # Negative rho: puts over calls


def test_wings_fall_back_to_atm_and_skew_without_params():
    wings = smile_wings(0.20, T, None, skew_25d=0.04)
    assert (wings.iv_25p, wings.iv_25c, wings.iv_atm) == (pytest.approx(0.22), pytest.approx(0.18), 0.20)
    assert wings.rr_25d == pytest.approx(0.04)
    assert wings.bf_25d == pytest.approx(0.0, abs=1e-12)

    assert smile_wings(0.20, T, [0.1, 0.2], skew_25d=0.5).iv_25c == 0.01  # Malformed params -> fallback, floored


@pytest.mark.parametrize(
    "vix_ratio, vrp, bf, rr_pct, bias",
    [
        (1.00, 0.05, 0.010, 50.0, BIAS_DEFENSIVE),  # Backwardation wins over everything
        (0.90, -0.02, 0.010, 50.0, BIAS_DEFENSIVE),  # Negative VRP
        (0.90, -0.015, 0.010, 50.0, BIAS_SELECTIVE),  # -1.5% is not below the defensive threshold
        (0.90, 0.02, 0.005, 10.0, BIAS_SHORT_VRP),  # Inclusive thresholds; beats the bullish rule
        (0.90, 0.019, 0.010, 20.0, BIAS_BULLISH_RR),
        (0.90, 0.03, 0.004, 21.0, BIAS_SELECTIVE),
    ],
)
def test_strategy_bias_decision_table(vix_ratio, vrp, bf, rr_pct, bias):
    assert classify_strategy_bias(vix_ratio, vrp, bf, rr_pct) == bias


@pytest.mark.parametrize(
    "spot, sma20, sma50, state",
    [
        (110.0, 105.0, 100.0, "UPTREND"),
        (90.0, 95.0, 100.0, "DOWNTREND"),
        (100.0, 100.0, 100.0, "RANGE_OR_TRANSITION"),
        (105.0, 110.0, 100.0, "RANGE_OR_TRANSITION"),
    ],
)
def test_trend_states(spot, sma20, sma50, state):
    assert classify_trend(spot, sma20, sma50) == state
