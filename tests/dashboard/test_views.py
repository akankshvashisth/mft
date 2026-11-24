import numpy as np
import pytest

from src.dashboard.views import (
    WAITING,
    actor_health_rows,
    attribution_rows,
    banner,
    decision_matrix_rows,
    dense_smile,
    forward_comparison_rows,
    risk_transition_rows,
    smile_density,
    svi_parameter_rows,
    universe_rv_rows,
    universe_snapshot_rows,
    vrp_spread,
)
from src.quant.density import density_area
from src.quant.surface_metrics import classify_strategy_bias
from src.quant.svi import svi_implied_vol
from tests.factories import (
    NOW_NS,
    SVI_PARAMS,
    ago,
    metrics_payload,
    regime_payload,
    risk_status_payload,
    rv_payload,
    surface_payload,
)


# --- Banner and KPIs -------------------------------------------------------------------------------


def test_banner_levels():
    assert banner({"risk_status": None}, NOW_NS)[0] == "warning"

    halted = banner({"risk_status": risk_status_payload(halted=True, reasons=["drawdown_breach:-6%"])}, NOW_NS)
    assert halted == ("error", "**SYSTEM HALTED** — Active Reasons: `drawdown_breach:-6%`")

    level, text = banner(
        {
            "risk_status": risk_status_payload(),
            "regime": regime_payload(now_ns=ago(4.0)),
            "surface": surface_payload(now_ns=ago(9.0)),
            "metrics": metrics_payload(),
        },
        NOW_NS,
    )
    assert level == "success"
    assert text == (
        "**SYSTEM ARMED & STREAMING** | Regime Age: `4.0s` | Surface Age: `9.0s` | "
        "Active Bias: **SHORT_VRP_IRON_CONDOR_OR_CREDIT_SPREAD**"
    )


def test_banner_before_metrics_and_surface_exist():
    level, text = banner({"risk_status": risk_status_payload(), "regime": None, "surface": None, "metrics": None}, NOW_NS)
    assert level == "success"
    assert "Surface Age: `Nones`" in text and "**CALIBRATING**" in text


def test_vrp_spread_sources():
    assert vrp_spread({}, {}, {"vrp_spread": 0.031}) == 0.031
    assert vrp_spread({"atm_iv": 0.16}, {"rv_20d": 0.13}, {}) == pytest.approx(0.03)
    assert vrp_spread({"atm_iv": 0.16}, {}, {}) is None


# --- Tables ----------------------------------------------------------------------------------------


def test_actor_rows_wait_for_missing_payloads():
    rows = actor_health_rows({}, {}, {}, {}, {}, NOW_NS)
    assert [row["Primary Output"] for row in rows] == [WAITING] * 5
    assert {row["Age (s)"] for row in rows} == {None}


def test_actor_rows_format_live_payloads():
    rows = actor_health_rows(
        regime_payload(now_ns=ago(2.0)),
        surface_payload(atm_iv=None),
        rv_payload(),
        metrics_payload(),
        {"open_positions": 1, "greeks": {"beta_delta_shares": 12.34}, "ts_ns": ago(1.0)},
        NOW_NS,
    )
    regime, surface, rv, metrics, port = rows
    assert regime["Age (s)"] == 2.0
    assert regime["Primary Output"] == "VIX=15.0 | VIX3M=17.0 | Ratio=0.8824 | ShortVolAllowed=True"
    assert "ATM_IV=N/A" in surface["Primary Output"]
    assert rv["Primary Output"].startswith("RV5=11.00% | RV20=13.00% | Pct=40.0% | Jump=5.0%")
    assert metrics["Primary Output"] == "RR25=+3.00% | BF25=+0.60% | VRP=+3.00% | Bias=SHORT_VRP_IRON_CONDOR_OR_CREDIT_SPREAD"
    assert port["Primary Output"] == "Open=1 | BetaDelta=+12.3sh | Vega=$+0.00/v | Theta=$+0.00/d"


def test_universe_snapshot_sorted_by_percentile():
    rows = universe_snapshot_rows(
        {
            "SPY.ARCA": rv_payload(rv_percentile=40.0),
            "QQQ.NASDAQ": rv_payload(rv_percentile=85.0, rv_5d=0.25),
            "TLT.NASDAQ": rv_payload(rv_percentile=None),
        }
    )
    assert [r["Symbol"] for r in rows] == ["QQQ", "SPY", "TLT"]
    assert rows[0] == {"Symbol": "QQQ", "RV 5d": "25.0%", "RV 20d": "13.0%", "1Y Pct": "85%", "Jump %": "5%"}
    assert rows[2]["1Y Pct"] == "N/A%"


def test_universe_rv_rows_in_percent():
    (row,) = universe_rv_rows({"SPY.ARCA": rv_payload(rv_10d=None)})
    assert row == {
        "Symbol": "SPY.ARCA",
        "Spot / Close": 700.0,
        "RV 5d (%)": 11.0,
        "RV 10d (%)": None,
        "RV 20d (%)": 13.0,
        "RV 60d (%)": 14.0,
        "1Y Percentile": 40.0,
        "Jump Ratio (%)": 5.0,
    }


def test_svi_parameter_rows():
    rows = svi_parameter_rows(SVI_PARAMS)
    assert [r["Value"] for r in rows] == ["0.000800", "0.020000", "-0.700000", "0.010000", "0.060000", "1.7%"]
    assert rows[-1]["Parameter"] == "Roger Lee Bound"


def test_decision_matrix_matches_the_classifier():
    biases = [row["Assigned Bias"] for row in decision_matrix_rows()]
    assert biases == [
        classify_strategy_bias(1.0, 0.0, 0.0, 50.0),
        classify_strategy_bias(0.9, 0.02, 0.005, 50.0),
        classify_strategy_bias(0.9, 0.0, 0.0, 20.0),
        classify_strategy_bias(0.9, 0.0, 0.0, 50.0),
    ]


def test_attribution_rows():
    rows = attribution_rows({"actual_pnl_change_usd": 12.346, "delta_pnl_usd": None})
    assert rows[0] == {"Component": "Actual P&L Change", "USD ($)": "$+12.35"}
    assert rows[1]["USD ($)"] == "$N/A"
    assert rows[2]["USD ($)"] == "$+0.00"  # Missing component
    assert len(rows) == 6


def test_forward_comparison_rows_handle_missing_futures_values():
    rows = forward_comparison_rows({"ratio_source": "UNAVAILABLE"}, surface_payload())
    parity, futures, at_expiry = rows
    assert (parity["Horizon (Days)"], parity["Implied Carry q (%)"]) == (24.0, 1.25)
    assert parity["Notes"] == "Estimated from 1 ATM SPY option pair(s)"
    assert futures["Notes"] == "Model-free CME futures scaled by N/A (UNAVAILABLE)"
    assert at_expiry["Notes"] == "Parity vs. Futures Divergence = N/A (N/A bps)"


def test_risk_transition_rows():
    rows = risk_transition_rows(
        [
            {"event": "STATE_CHANGE", "to_halted": False, "halted": True, "reasons": []},
            {"event": "DAEMON_ERROR", "halted": True, "reasons": ["daemon_error:X:y", "other"]},
        ]
    )
    assert (rows[0]["Halted"], rows[0]["Reasons"]) == (False, "None (Healthy)")
    assert (rows[1]["Halted"], rows[1]["Reasons"]) == (True, "daemon_error:X:y, other")


# --- Derived series --------------------------------------------------------------------------------


def test_dense_smile_spans_the_quoted_moneyness():
    surface = surface_payload()
    k, ivs = dense_smile(surface)
    assert (len(k), k[0], k[-1]) == (120, pytest.approx(-0.06), pytest.approx(0.06))
    np.testing.assert_allclose(ivs, svi_implied_vol(k, 24 / 365.25, SVI_PARAMS) * 100.0)
    assert dense_smile(surface_payload(svi_params=[])) is None
    assert dense_smile(surface_payload(log_moneyness=[])) is None


def test_smile_density_is_normalised():
    strikes, density = smile_density(surface_payload())
    assert density_area(strikes, density) == pytest.approx(1.0)
    assert smile_density(surface_payload(svi_params=None)) is None


def test_smile_density_of_inadmissible_svi_parameters_is_unavailable():
    # a + b sigma sqrt(1 - rho^2) < 0: negative minimum variance, rejected by QuantLib's SviSmileSection
    assert smile_density(surface_payload(svi_params=[-0.1, 0.01, 0.0, 0.0, 0.1])) is None
