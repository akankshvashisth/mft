import math

import numpy as np
import pytest
from src.quant.risk_measures import (
    TailMetrics,
    contracts_for_tail_budget,
    parametric_book_var_es,
    tail_loss_per_lot,
    tail_metrics,
)

Z_99 = 2.3263478744090126
ES_MULT_99 = 2.6652132234682377


def test_tail_metrics_on_a_uniform_ladder():
    m = tail_metrics(np.arange(-49.0, 51.0))  # -49 .. 50

    assert m.mean == pytest.approx(0.5)
    assert m.prob_profit == pytest.approx(0.5)
    assert m.var_95 == pytest.approx(-44.05)  # Linear interpolation at the 5th percentile
    assert m.es_95 == pytest.approx(-47.0)  # Mean of -49 .. -45
    assert m.var_99 == pytest.approx(-48.01)
    assert m.es_99 == pytest.approx(-49.0)  # -49 is the only value at or below -48.01


def test_tail_metrics_ignore_non_finite_observations():
    m = tail_metrics([-20.0, np.nan, 10.0, np.inf, -10.0])
    assert m.mean == pytest.approx(-20 / 3)
    assert m.prob_profit == pytest.approx(1 / 3)
    assert all(math.isfinite(x) for x in (m.var_95, m.es_95, m.var_99, m.es_99))


def test_tail_metrics_are_zero_for_empty_or_non_finite_input():
    for pnl in ([], [np.nan], [np.inf, -np.inf]):
        assert tail_metrics(pnl) == TailMetrics(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)


def test_expected_shortfall_never_exceeds_var():
    pnl = np.random.default_rng(1).standard_t(df=3, size=5_000) * 100.0
    m = tail_metrics(pnl)
    assert m.es_95 <= m.var_95
    assert m.es_99 <= m.var_99
    assert m.var_99 <= m.var_95


@pytest.mark.parametrize("es_99, loss", [(-300.0, 300.0), (-0.2, 1.0), (5.0, 1.0)])
def test_tail_loss_per_lot_is_floored_at_one_dollar(es_99, loss):
    assert tail_loss_per_lot(es_99) == loss


@pytest.mark.parametrize("es_99", [np.nan, np.inf, -np.inf])
def test_tail_loss_per_lot_handles_non_finite_es(es_99):
    assert tail_loss_per_lot(es_99) == 1.0


@pytest.mark.parametrize(
    "net_ev, es_99, budget, lots",
    [
        (10.0, -300.0, 1_000.0, 3),
        (10.0, -0.5, 1_000.0, 1_000),  # Near-riskless P&L: sized off the $1 floor
        (0.0, -300.0, 1_000.0, 0),
        (-5.0, -300.0, 1_000.0, 0),
    ],
)
def test_contracts_for_tail_budget(net_ev, es_99, budget, lots):
    assert contracts_for_tail_budget(net_ev, es_99, budget) == lots


@pytest.mark.parametrize(
    "delta_shares, vega_1v, lin_std_usd",
    [
        (100.0, 0.0, 500.0),  # One-day sigma = 500 x 1% = $5 per share on 100 shares
        (-100.0, -10.0, 500.8),  # Plus 10 x (1% x 100 x 0.08 = 0.08 vol pts) of vega
    ],
)
def test_parametric_var_es_are_normal_multiples_of_the_linear_pnl_std(delta_shares, vega_1v, lin_std_usd):
    var_99, es_99 = parametric_book_var_es(delta_shares, vega_1v, 500.0, 0.01, trading_days=1)
    assert var_99 == pytest.approx(Z_99 * lin_std_usd, abs=0.01)  # Results are rounded to cents
    assert es_99 == pytest.approx(ES_MULT_99 * lin_std_usd, abs=0.01)


def test_parametric_var_es_properties():
    assert parametric_book_var_es(0.0, 0.0, 700.0, 0.15) == (0.0, 0.0)
    var_99, es_99 = parametric_book_var_es(250.0, 40.0, 700.0, 0.15)
    assert es_99 / var_99 == pytest.approx(ES_MULT_99 / Z_99, rel=1e-3)
    assert var_99 > 0.0
    # Daily sigma scales with 1 / sqrt(trading days)
    var_252, _ = parametric_book_var_es(250.0, 0.0, 700.0, 0.15)
    var_63, _ = parametric_book_var_es(250.0, 0.0, 700.0, 0.15, trading_days=63)
    assert var_63 == pytest.approx(2.0 * var_252, rel=1e-3)
    assert math.isfinite(var_63)
    # Lower confidence, smaller multipliers
    var_95, es_95 = parametric_book_var_es(250.0, 40.0, 700.0, 0.15, confidence=0.95)
    assert var_95 < var_99 and es_95 < es_99


@pytest.mark.parametrize(
    "kwargs",
    [
        {"trading_days": 0},
        {"confidence": 0.0},
        {"confidence": 1.0},
        {"rv_annual": -0.1},
        {"rv_annual": np.nan},
        {"vol_move_frac": -0.1},
    ],
)
def test_parametric_var_es_rejects_invalid_numeric_inputs(kwargs):
    with pytest.raises(ValueError):
        rv_annual = kwargs.pop("rv_annual", 0.15)
        parametric_book_var_es(250.0, 40.0, 700.0, rv_annual, **kwargs)
