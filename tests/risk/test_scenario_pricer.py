import numpy as np
import pytest

from src.quant.black_scholes import bsm_prices
from src.risk.scenario_pricer import (
    ScenarioPricer,
    defined_risk_margin_usd,
    friction_usd,
)
from src.scanner.combos import CandidateCombo, CandidateLeg


def bull_put(short_mid=8.4, wing_mid=5.3, **overrides) -> CandidateCombo:
    short = CandidateLeg("SPY P690", 690.0, "P", -1, 1, short_mid - 0.1, short_mid + 0.1, short_mid, 0.17, 0.17, 0.0, 0.0)
    wing = CandidateLeg("SPY P680", 680.0, "P", +1, 1, wing_mid - 0.1, wing_mid + 0.1, wing_mid, 0.18, 0.18, 0.0, 0.0)
    spec = dict(
        name="SPY 690/680P Bull Put", strategy_type="BULL_PUT_SPREAD", underlying="SPY.ARCA",
        spot=700.0, forward=701.0, ttm_days=30.0, risk_free_rate=0.0425, dividend_yield=0.0125,
        legs=[short, wing], net_credit=round(short_mid - wing_mid, 4), max_width=10.0,
    )
    spec.update(overrides)
    return CandidateCombo(**spec)


def pricer(**kwargs) -> ScenarioPricer:
    kwargs.setdefault("n_scenarios", 2_000)
    return ScenarioPricer(price_fn=bsm_prices, **kwargs)


def test_repeated_evaluations_see_identical_scenarios():
    p = pricer()
    combo = bull_put()
    first = p.evaluate_combo(combo, realized_vol_p=0.14)
    assert p.evaluate_combo(combo, realized_vol_p=0.14) == first
    assert pricer().evaluate_combo(combo, realized_vol_p=0.14) == first  # Same seed, same paths


def test_seed_changes_the_scenarios():
    combo = bull_put()
    assert pricer(seed=1).evaluate_combo(combo).net_ev_usd != pricer(seed=2).evaluate_combo(combo).net_ev_usd


def test_friction_is_half_spread_plus_commissions():
    combo = bull_put()  # Both legs 0.20 wide
    assert friction_usd(combo.legs, 0.5, 0.65) == pytest.approx(2 * 0.5 * 0.20 * 100.0 + 2 * 0.65)
    locked = [CandidateLeg("X", 1.0, "P", -1, 1, 2.0, 2.0, 2.0, 0.2, 0.2, 0.0, 0.0)]
    assert friction_usd(locked, 0.5, 0.0) == pytest.approx(0.5)  # Spread floored at one cent
    assert pricer().evaluate_combo(combo).friction_cost_usd == pytest.approx(21.3)


def test_rich_credit_is_approved_and_sized():
    risk = pricer().evaluate_combo(bull_put(short_mid=12.0, wing_mid=3.0), realized_vol_p=0.15)
    assert risk.net_ev_usd > 0.0
    assert risk.approved and risk.recommended_contracts >= 1
    assert risk.es_99_usd <= risk.var_99_usd <= risk.var_95_usd
    assert risk.ev_to_es99_ratio > 0.0


def test_negative_expected_value_is_rejected_with_zero_size():
    risk = pricer().evaluate_combo(bull_put(short_mid=5.0, wing_mid=7.0), realized_vol_p=0.15)
    assert risk.net_ev_usd < 0.0
    assert not risk.approved
    assert risk.recommended_contracts == 0


def test_stress_grid_shape_labels_and_direction():
    risk = pricer().evaluate_combo(bull_put())
    assert risk.spot_shocks_pct == [-10.0, -5.0, -2.0, 0.0, 2.0, 5.0, 10.0]
    assert risk.vol_shocks_pts == [-5.0, 0.0, 5.0, 10.0, 15.0]
    grid = np.asarray(risk.scenario_grid_usd)
    assert grid.shape == (7, 5)
    # A short put spread loses when spot falls and when implied vol rises
    assert np.all(grid[0] < grid[-1])
    assert np.all(np.diff(grid[3]) <= 0.0)


def test_pricing_function_gets_the_true_rate_and_the_dividend_yield():
    seen = []

    def recording(S, K, t, r, sigma, flag, q):
        seen.append((float(np.unique(r)[0]), float(np.unique(q)[0])))
        return bsm_prices(S, K, t, r, sigma, flag, q)

    ScenarioPricer(n_scenarios=100, price_fn=recording).evaluate_combo(bull_put())
    assert set(seen) == {(0.0425, 0.0125)}


def test_margin_estimates():
    combo = bull_put()
    assert defined_risk_margin_usd(combo, [[0.0]]) == pytest.approx((10.0 - 3.1) * 100.0)
    assert defined_risk_margin_usd(bull_put(max_width=1.0, short_mid=8.4, wing_mid=5.3), [[0.0]]) == 100.0
    naked = bull_put(max_width=0.0)
    assert defined_risk_margin_usd(naked, [[-50.0, 10.0], [-900.0, 0.0]]) == 900.0
    assert defined_risk_margin_usd(naked, [[-50.0]]) == 500.0


def test_default_price_function_matches_quantlib():
    pytest.importorskip("fast_vollib")
    from src.risk.scenario_pricer import fast_vollib_price

    S = np.array([700.0, 700.0])
    K = np.array([690.0, 720.0])
    t = np.full(2, 0.08)
    r = np.full(2, 0.0425)
    sigma = np.full(2, 0.18)
    flag = np.array(["p", "c"])
    q = np.full(2, 0.0125)
    np.testing.assert_allclose(fast_vollib_price(S, K, t, r, sigma, flag, q), bsm_prices(S, K, t, r, sigma, flag, q), rtol=1e-6)
