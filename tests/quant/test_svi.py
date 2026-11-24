import math
import types
from unittest import mock

import numpy as np
import pytest
import QuantLib as ql

import src.quant.svi as svi
from src.quant.svi import (
    SVI_LOWER_BOUNDS,
    SVI_UPPER_BOUNDS,
    fit_svi_slice,
    roger_lee_utilization,
    svi_implied_vol,
    svi_implied_vol_or_none,
    svi_smile_section,
    svi_total_variance,
)
from tests.factories import SVI_PARAMS

T = 30 / 365.25
K_GRID = np.linspace(-0.15, 0.08, 15)
BAD_PARAMS = (0.01, 0.3, 0.5, 0.1, 0.3)  # ~100% vols: far from the market smile


def quantlib_section(params, ttm=T, forward=1.0):
    """Reference smile: QuantLib's SviSmileSection, built directly in its (a, b, sigma, rho, m) order."""
    a, b, rho, m, sigma = params
    return ql.SviSmileSection(ttm, forward, [a, b, sigma, rho, m])


@pytest.mark.parametrize("params", [SVI_PARAMS, (0.002, 0.05, 0.0, 0.0, 0.1), BAD_PARAMS])
def test_total_variance_and_vols_match_quantlib(params):
    section = quantlib_section(params)
    strikes = np.exp(K_GRID)  # Forward 1: strike = exp(log-moneyness)

    np.testing.assert_allclose(svi_total_variance(K_GRID, params), [section.variance(k) for k in strikes], rtol=1e-12)
    np.testing.assert_allclose(svi_implied_vol(K_GRID, T, params), [section.volatility(k) for k in strikes], rtol=1e-12)


def test_smile_section_maps_our_parameter_order_onto_quantlib():
    section = svi_smile_section(SVI_PARAMS, T, forward=700.0)
    vols = [section.volatility(strike) for strike in 700.0 * np.exp(K_GRID)]
    np.testing.assert_allclose(vols, svi_implied_vol(K_GRID, T, SVI_PARAMS), rtol=1e-12)


@pytest.mark.parametrize(
    "params",
    [
        (-0.1, 0.01, 0.0, 0.0, 0.1),  # Negative minimum variance
        (0.001, 3.0, 0.5, 0.0, 0.1),  # b (1 + |rho|) above Roger Lee's bound of 4
        (0.001, 0.05, 1.0, 0.0, 0.1),  # |rho| = 1
    ],
)
def test_smile_section_rejects_inadmissible_parameters(params):
    with pytest.raises(RuntimeError):
        svi_smile_section(params, T)


def test_floor_versus_none_policies_for_negative_variance():
    params = (-0.1, 0.01, 0.0, 0.0, 0.1)  # w(0) = -0.099
    assert float(svi_implied_vol(0.0, 0.5, params)) == pytest.approx(math.sqrt(1e-8 / 0.5))
    assert svi_implied_vol_or_none(0.0, 0.5, params) is None


def test_scalar_helper_agrees_with_vectorised_helper():
    expected = float(svi_implied_vol(0.03, T, SVI_PARAMS))
    assert svi_implied_vol_or_none(0.03, T, list(SVI_PARAMS)) == pytest.approx(expected, rel=1e-12)


@pytest.mark.parametrize("params, ttm", [(None, T), ([0.1, 0.2], T), (SVI_PARAMS, 0.0), (SVI_PARAMS, 1e-7)])
def test_scalar_helper_rejects_malformed_inputs(params, ttm):
    assert svi_implied_vol_or_none(0.0, ttm, params) is None


def test_vectorised_helper_floors_maturity():
    assert np.isfinite(float(svi_implied_vol(0.0, 0.0, SVI_PARAMS)))


def test_roger_lee_utilization():
    assert roger_lee_utilization(SVI_PARAMS) == pytest.approx(0.02 * 1.7 / 2.0)
    assert roger_lee_utilization([0.0, 2.0, -1.0, 0.0, 0.1]) == pytest.approx(2.0)


# --- Calibration control flow (scipy's least_squares replaced by a mock) ---------------------------


@pytest.fixture
def scripted(monkeypatch):
    """Patches least_squares with a mock that returns the given parameter vectors in call order."""

    def install(*param_vectors):
        optimiser = mock.Mock(side_effect=[types.SimpleNamespace(x=np.array(p, dtype=float)) for p in param_vectors])
        monkeypatch.setattr(svi, "least_squares", optimiser)
        return optimiser

    return install


def _market():
    ivs = svi_implied_vol(K_GRID, T, SVI_PARAMS)
    return ivs, np.ones_like(K_GRID)


def test_cold_start_uses_the_documented_initial_guess(scripted):
    optimiser = scripted(SVI_PARAMS)
    ivs, weights = _market()

    iv_fit, params, rmse = fit_svi_slice(K_GRID, ivs, weights, T)

    (call,) = optimiser.call_args_list
    residuals = call.args[0]
    assert np.all(np.isfinite(residuals(np.array(SVI_PARAMS))))
    assert call.kwargs["max_nfev"] == 120
    np.testing.assert_allclose(call.kwargs["x0"][1:], [0.08, -0.55, 0.0, 0.08])
    np.testing.assert_allclose(call.kwargs["bounds"][0], SVI_LOWER_BOUNDS)
    np.testing.assert_allclose(call.kwargs["bounds"][1], SVI_UPPER_BOUNDS)
    np.testing.assert_allclose(params, SVI_PARAMS)
    np.testing.assert_allclose(iv_fit, ivs, atol=1e-12)
    assert rmse < 1e-9


def test_good_warm_start_is_kept_without_a_cold_refit(scripted):
    optimiser = scripted(SVI_PARAMS)
    ivs, weights = _market()

    _, params, _ = fit_svi_slice(K_GRID, ivs, weights, T, x0_warm=np.array(SVI_PARAMS))

    assert [c.kwargs["max_nfev"] for c in optimiser.call_args_list] == [120]
    np.testing.assert_allclose(optimiser.call_args.kwargs["x0"], SVI_PARAMS)
    np.testing.assert_allclose(params, SVI_PARAMS)


def test_poor_warm_fit_falls_back_to_cold_start(scripted):
    optimiser = scripted(BAD_PARAMS, SVI_PARAMS)
    ivs, weights = _market()

    _, params, rmse = fit_svi_slice(K_GRID, ivs, weights, T, x0_warm=np.array(BAD_PARAMS))

    assert [c.kwargs["max_nfev"] for c in optimiser.call_args_list] == [120, 150]
    np.testing.assert_allclose(params, SVI_PARAMS)
    assert rmse < 1e-9


def test_cold_refit_is_discarded_when_it_is_worse(scripted):
    worse = (0.02, 0.5, 0.5, 0.1, 0.5)
    optimiser = scripted(BAD_PARAMS, worse)
    ivs, weights = _market()

    _, params, rmse = fit_svi_slice(K_GRID, ivs, weights, T, x0_warm=np.array(BAD_PARAMS))

    assert optimiser.call_count == 2
    np.testing.assert_allclose(params, BAD_PARAMS)
    assert rmse > 0.005


def test_warm_start_is_clipped_inside_the_bounds(scripted):
    optimiser = scripted(SVI_PARAMS)
    ivs, weights = _market()
    warm = np.array([0.001, 0.05, 1.5, 0.0, 0.1])  # rho beyond its upper bound

    fit_svi_slice(K_GRID, ivs, weights, T, x0_warm=warm)

    assert optimiser.call_args.kwargs["x0"][2] == pytest.approx(SVI_UPPER_BOUNDS[2] - 1e-6)


# --- Real optimiser ----------------------------------------------------------------------------


def test_fit_recovers_a_synthetic_smile_from_a_cold_start():
    ivs, weights = _market()

    iv_fit, params, rmse = fit_svi_slice(K_GRID, ivs, weights, T)

    assert rmse < 0.005  # Half a vol point
    assert float(svi_implied_vol(0.0, T, params)) == pytest.approx(float(svi_implied_vol(0.0, T, SVI_PARAMS)), abs=0.005)
    np.testing.assert_allclose(iv_fit, svi_implied_vol(K_GRID, T, params, w_floor=1e-6), atol=1e-12)
    assert np.all(params >= SVI_LOWER_BOUNDS) and np.all(params <= SVI_UPPER_BOUNDS)
