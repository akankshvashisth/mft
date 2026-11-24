import math

import pytest
import QuantLib as ql

from src.quant.parity import estimate_forward_and_q
from src.quant.quote_hygiene import SliceQuote
from tests.factories import NOW_NS, ago

SPOT, T, R, Q_TRUE, Q_DEFAULT = 700.0, 30 / 365.25, 0.0425, 0.015, 0.0125
VOL = 0.16
FORWARD = SPOT * math.exp((R - Q_TRUE) * T)
PRIOR = SPOT * math.exp((R - Q_DEFAULT) * T)
FILTERS = {"max_age_secs": 90.0, "min_mid": 0.15, "max_rel_spread": 0.25}


def _quote(strike, is_call, mid, ts_ns=NOW_NS, half_spread=0.05):
    right = "C" if is_call else "P"
    return SliceQuote(f"SPY {right}{strike:g}", strike, is_call, mid - half_spread, mid + half_spread, mid, ts_ns)


def _parity_pair(strike, forward=FORWARD, call_shift=0.0, **kwargs):
    """Black-76 call and put mids on `forward` (QuantLib), so C - P = exp(-rT) (F - K) holds."""
    std_dev, discount = VOL * math.sqrt(T), math.exp(-R * T)
    call_mid = ql.blackFormula(ql.Option.Call, strike, forward, std_dev, discount) + call_shift
    put_mid = ql.blackFormula(ql.Option.Put, strike, forward, std_dev, discount)
    return [_quote(strike, True, call_mid, **kwargs), _quote(strike, False, put_mid, **kwargs)]


def _estimate(quotes):
    return estimate_forward_and_q(quotes, SPOT, T, R, Q_DEFAULT, NOW_NS, **FILTERS)


def _chain(strikes=(690.0, 695.0, 700.0, 705.0, 710.0), **kwargs):
    return [q for k in strikes for q in _parity_pair(k, **kwargs)]


def test_parity_consistent_quotes_round_trip_forward_and_dividend_yield():
    forward, q, pairs = _estimate(_chain())
    assert forward == pytest.approx(FORWARD, abs=1e-9)
    assert q == pytest.approx(Q_TRUE, abs=1e-9)
    assert pairs == 5


def test_median_ignores_a_single_mispriced_pair():
    quotes = _chain(strikes=(690.0, 695.0, 705.0, 710.0)) + _parity_pair(700.0, call_shift=5.0)
    forward, q, pairs = _estimate(quotes)
    assert forward == pytest.approx(FORWARD, abs=1e-9)
    assert pairs == 5


def test_implausible_dividend_yield_falls_back_to_prior():
    rich_forward = SPOT * math.exp((R - 0.20) * T)  # Implies q = 20%
    forward, q, pairs = _estimate(_chain(forward=rich_forward))
    assert (forward, q, pairs) == (pytest.approx(PRIOR), Q_DEFAULT, 0)


def test_calls_without_matching_puts_fall_back_to_prior():
    calls_only = [q for q in _chain() if q.is_call]
    assert _estimate(calls_only) == (pytest.approx(PRIOR), Q_DEFAULT, 0)


@pytest.mark.parametrize(
    "pair_kwargs, strikes",
    [
        ({"ts_ns": ago(91.0)}, (700.0,)),  # Stale
        ({"half_spread": 3.0}, (700.0,)),  # Relative spread above 25%
        ({}, (740.0, 660.0)),  # Outside the 2.5% ATM band
    ],
)
def test_filtered_quotes_do_not_count_as_pairs(pair_kwargs, strikes):
    assert _estimate(_chain(strikes=strikes, **pair_kwargs)) == (pytest.approx(PRIOR), Q_DEFAULT, 0)
