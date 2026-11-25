import numpy as np
import pytest

from src.quant.black_scholes import bsm_prices
from src.risk.scenario_pricer import (
    ScenarioPricer,
    defined_risk_margin_usd,
    friction_usd,
    round_trip_friction_usd,
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
    assert pricer().evaluate_combo(combo).friction_cost_usd == pytest.approx(42.6)  # Round-trip: entry+exit


# --- WP-6: ScenarioPricer hardening ----------------------------------------------------------------


def test_bsm_carry_golden_check():
    """BSM carry: S=100, K=100, T=0.5, r=0.05, q=0.04, sigma=0.30 matches black-scholes.py < 1e-5."""
    from src.quant.black_scholes import bs_greeks

    ref = bs_greeks(100.0, 100.0, 0.5, 0.05, 0.04, 0.30, True)
    from src.risk.scenario_pricer import fast_vollib_price

    p = fast_vollib_price(np.array([100.0]), 100.0, 0.5, 0.05, np.array([0.30]), "c", 0.04)
    assert abs(float(p[0]) - ref["price"]) < 1e-5


def test_round_trip_friction_deducts_more_than_entry_only():
    """Evaluating the same spread at horizon_days=14 (< ttm_days=30) deducts strictly more friction."""
    combo = bull_put()
    # Entry-only friction is half-spread + commissions for 2 legs
    entry_only = friction_usd(combo.legs, 0.5, 0.65)
    # Round-trip is double that (entry + exit at same slippage)
    round_trip = round_trip_friction_usd(combo.legs, 0.5, 0.65)
    assert round_trip == pytest.approx(2 * entry_only)
    assert round_trip > entry_only

    # Actual pricer: horizon < ttm
    p = pricer()
    assert p.evaluate_combo(combo, horizon_days=14.0).friction_cost_usd > entry_only
    # At horizon >= ttm, friction is entry-only
    assert p.evaluate_combo(combo, horizon_days=30.0).friction_cost_usd == pytest.approx(entry_only, rel=0.02)


def test_crash_tail_monotonicity():
    """A bull put spread has es_99_usd worse (more negative = bigger loss) at higher realized vol."""
    combo = bull_put()
    p = pricer(n_scenarios=5_000)
    low_vol = p.evaluate_combo(combo, realized_vol_p=0.10, horizon_days=14.0)
    high_vol = p.evaluate_combo(combo, realized_vol_p=0.25, horizon_days=14.0)
    # Higher RV means worse tail (more negative ES99)
    assert low_vol.es_99_usd < 0.0  # Bull put loses in extreme down moves
    assert high_vol.es_99_usd < low_vol.es_99_usd  # Strictly worse at higher vol
    # |ES99| + friction <= max_width * 100
    assert abs(high_vol.es_99_usd) <= 10.0 * 100.0 + high_vol.friction_cost_usd


def test_crn_determinism():
    """Evaluating candidate A then B vs B then A produces identical results."""
    p = pricer(n_scenarios=2_000, seed=42)
    combo_a = bull_put(short_mid=8.4, wing_mid=5.3, name="A")
    combo_b = bull_put(short_mid=7.4, wing_mid=4.3, name="B")
    # A then B
    res_a_first = p.evaluate_combo(combo_a, realized_vol_p=0.20, horizon_days=14.0)
    res_b_second = p.evaluate_combo(combo_b, realized_vol_p=0.20, horizon_days=14.0)
    # B then A with reset pricer
    p2 = pricer(n_scenarios=2_000, seed=42)
    res_b_first = p2.evaluate_combo(combo_b, realized_vol_p=0.20, horizon_days=14.0)
    res_a_second = p2.evaluate_combo(combo_a, realized_vol_p=0.20, horizon_days=14.0)
    # A evaluated first vs second should match (same CRN)
    assert res_a_first.net_ev_usd == pytest.approx(res_a_second.net_ev_usd, abs=0.01)
    assert res_a_first.es_99_usd == pytest.approx(res_a_second.es_99_usd, abs=0.01)
    assert res_b_first.net_ev_usd == pytest.approx(res_b_second.net_ev_usd, abs=0.01)
    assert res_b_first.es_99_usd == pytest.approx(res_b_second.es_99_usd, abs=0.01)


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
