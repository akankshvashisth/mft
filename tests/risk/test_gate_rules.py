from datetime import date
from unittest import mock

import pytest

from src.risk.gate_rules import (
    OrderRateLimiter,
    book_es99_usd,
    check_legs,
    daemon_block_reason,
    es_budget_check,
    ev_reasons,
    futures_check,
    metrics_reasons,
    portfolio_reasons,
    regime_reasons,
    surface_reasons,
)
from src.risk.gate_types import CandidateLegSpec, GateConfig
from tests.factories import (
    NOW_NS,
    ago,
    futures_payload,
    metrics_payload,
    portfolio_payload,
    regime_payload,
    risk_status_payload,
    surface_payload,
)

CFG = GateConfig()
AS_OF = date(2026, 11, 24)
EXPIRY = date(2026, 12, 18)
FAR_FUTURE_NS = NOW_NS + 901 * 1_000_000_000


def leg(**overrides) -> CandidateLegSpec:
    spec = dict(
        instrument_id="SPY P690", right="P", side="SELL", strike=690.0, expiry_date=EXPIRY, qty=5,
        bid=2.0, ask=2.2, bid_size=100, ask_size=100, open_interest=10_000,
    )
    spec.update(overrides)
    return CandidateLegSpec(**spec)


# --- Risk daemon -----------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "halted, status, expected",
    [
        ("0", risk_status_payload(), None),
        (" 0\n", risk_status_payload(), None),
        (None, risk_status_payload(), "RISK_DAEMON_HALTED_OR_MISSING"),
        ("1", risk_status_payload(), "RISK_DAEMON_HALTED_OR_MISSING"),
        ("0", {}, "RISK_DAEMON_HALTED_OR_MISSING"),
        ("0", None, "RISK_DAEMON_HALTED_OR_MISSING"),
        ("0", risk_status_payload(now_ns=ago(31.0)), "RISK_DAEMON_STALE(31s)"),
        ("0", risk_status_payload(halted=True), "RISK_DAEMON_HALTED_OR_MISSING"),
        ("0", {"evaluated_at_ns": NOW_NS}, "RISK_DAEMON_HALTED_OR_MISSING"),  # "halted" must be literally false
    ],
)
def test_daemon_block_reason_matrix(halted, status, expected):
    assert daemon_block_reason(halted, status, NOW_NS, CFG.max_risk_status_age_secs) == expected


# --- Snapshot rules --------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "payload",
    [{}, regime_payload(now_ns=ago(121.0)), regime_payload(now_ns=FAR_FUTURE_NS), regime_payload(ts_ns="soon")],
)
def test_missing_stale_or_unusable_regime_blocks(payload):
    assert regime_reasons(payload, NOW_NS, CFG.max_input_age_secs) == ["MACRO_REGIME_MISSING_OR_STALE"]


def test_regime_rules():
    assert regime_reasons(regime_payload(), NOW_NS, 120.0) == []
    blocked = regime_payload(short_vol_allowed=False, ratio=1.05)
    assert regime_reasons(blocked, NOW_NS, 120.0) == ["MACRO_REGIME_BLOCKED(ratio=1.05)"]


@pytest.mark.parametrize(
    "overrides, expected",
    [
        ({}, []),
        ({"arb_free": False}, ["SVI_SURFACE_NOT_ARB_FREE"]),
        ({"rmse_vol_pts": 0.6}, ["SVI_FIT_POOR(rmse=0.6pts>0.5)"]),
        ({"rmse_vol_pts": 0.5}, []),
        ({"rmse_vol_pts": None}, ["SVI_FIT_POOR(rmse=Nonepts>0.5)"]),
        ({"arb_free": False, "rmse_vol_pts": 0.9}, ["SVI_SURFACE_NOT_ARB_FREE", "SVI_FIT_POOR(rmse=0.9pts>0.5)"]),
        ({"ts_ns": ago(121.0)}, ["SVI_SURFACE_MISSING_OR_STALE"]),
    ],
)
def test_surface_rules(overrides, expected):
    assert surface_reasons(surface_payload(**overrides), NOW_NS, CFG) == expected


def test_metrics_rules():
    assert metrics_reasons(metrics_payload(), NOW_NS, 120.0) == []
    defensive = metrics_payload(strategy_bias="DEFENSIVE_OR_LONG_CONVEXITY", vrp_spread=-0.02)
    assert metrics_reasons(defensive, NOW_NS, 120.0) == ["SURFACE_BIAS_DEFENSIVE(VRP=-0.02)"]
    assert metrics_reasons({}, NOW_NS, 120.0) == ["SURFACE_METRICS_MISSING_OR_STALE"]


@pytest.mark.parametrize(
    "payload, expected",
    [
        (futures_payload(), ([], [])),
        (futures_payload(parity_vs_fut_diff_bps=25.0), (["FUTURES_PARITY_DIVERGENCE(25.0bps)"], [])),
        (futures_payload(parity_vs_fut_diff_bps=-25.0), (["FUTURES_PARITY_DIVERGENCE(-25.0bps)"], [])),
        (futures_payload(cross_check_valid=False, parity_vs_fut_diff_bps=99.0), ([], ["FUTURES_CROSS_CHECK_UNAVAILABLE"])),
        (futures_payload(parity_vs_fut_diff_bps=None), ([], ["FUTURES_CROSS_CHECK_UNAVAILABLE"])),
        (futures_payload(now_ns=ago(121.0), parity_vs_fut_diff_bps=99.0), ([], ["FUTURES_CROSS_CHECK_UNAVAILABLE"])),
        ({}, ([], ["FUTURES_CROSS_CHECK_UNAVAILABLE"])),
    ],
)
def test_futures_cross_check_is_advisory_unless_valid(payload, expected):
    assert futures_check(payload, NOW_NS, CFG) == expected


@pytest.mark.parametrize(
    "ev, expected",
    [(0.01, []), (0.0, ["NON_POSITIVE_NET_EV($0.00/lot)"]), (-1.5, ["NON_POSITIVE_NET_EV($-1.50/lot)"])],
)
def test_ev_rule(ev, expected):
    assert ev_reasons(ev) == expected


def test_portfolio_freshness_and_book_es99():
    assert portfolio_reasons(portfolio_payload(), NOW_NS, 120.0) == []
    assert portfolio_reasons({}, NOW_NS, 120.0) == ["PORTFOLIO_RISK_MISSING_OR_STALE"]
    assert book_es99_usd(portfolio_payload(book_es_99_usd=250)) == 250.0
    assert book_es99_usd({}) == 0.0
    assert book_es99_usd({"risk": {"book_es_99_usd": None}}) == 0.0


@pytest.mark.parametrize(
    "payload",
    [
        portfolio_payload(book_es_99_usd=None),
        portfolio_payload(book_es_99_usd=float("nan")),
        portfolio_payload(book_es_99_usd="250"),
        {"risk": "corrupt", "ts_ns": NOW_NS},
        {"ts_ns": NOW_NS},
    ],
)
def test_fresh_but_unreadable_book_es99_blocks(payload):
    # Reading it as 0 would approve trades against an unknown book
    assert portfolio_reasons(payload, NOW_NS, 120.0) == ["PORTFOLIO_RISK_INVALID"]
    assert book_es99_usd(payload) == 0.0


def test_unmodeled_positions_block_entry():
    """has_unmodeled_positions=True in the portfolio payload blocks entry."""
    payload = portfolio_payload(risk={"book_es_99_usd": 100.0, "has_unmodeled_positions": True})
    assert portfolio_reasons(payload, NOW_NS, 120.0) == ["PORTFOLIO_HAS_UNMODELED_POSITIONS"]


# --- Order-rate throttle ---------------------------------------------------------------------------


def test_rate_limiter_counts_submissions_in_a_sliding_window():
    clock = mock.Mock(return_value=1_000.0)
    limiter = OrderRateLimiter(2, clock=clock)

    assert limiter.throttle_reason() is None  # Checking does not consume capacity
    limiter.record()
    assert limiter.throttle_reason() is None
    limiter.record()
    assert limiter.throttle_reason() == "ORDER_RATE_THROTTLE(2/min)"

    clock.return_value = 1_059.0
    assert limiter.throttle_reason() == "ORDER_RATE_THROTTLE(2/min)"
    clock.return_value = 1_060.0  # Both submissions are now a full window old
    assert limiter.throttle_reason() is None
    assert limiter.timestamps == []


# --- Legs ------------------------------------------------------------------------------------------


def test_healthy_leg_passes_uncapped():
    check = check_legs([leg()], AS_OF, CFG)
    assert (check.capped_qty, check.reasons, check.warnings) == (5, [], [])
    assert check.max_expiry == EXPIRY
    assert not check.has_short_call


@pytest.mark.parametrize(
    "overrides, reason",
    [
        ({"multiplier": 10}, "NON_STANDARD_CONTRACT(SPY P690, mult=10)"),
        ({"is_adjusted": True}, "NON_STANDARD_CONTRACT(SPY P690, mult=100)"),
        ({"bid": 0.0}, "INVALID_QUOTE(SPY P690: 0.0/2.2)"),
        ({"bid": 2.2, "ask": 2.2}, "INVALID_QUOTE(SPY P690: 2.2/2.2)"),  # Locked
        ({"bid": 1.0, "ask": 1.5}, "WIDE_MIDCAP_SPREAD(SPY P690: 40.0%>18.0%)"),
        ({"open_interest": None}, "OPEN_INTEREST_UNKNOWN(SPY P690)"),
        ({"open_interest": 50}, "LOW_OPEN_INTEREST(SPY P690: OI=50<100)"),
    ],
)
def test_leg_block_reasons(overrides, reason):
    assert reason in check_legs([leg(**overrides)], AS_OF, CFG).reasons


def test_invalid_quote_skips_the_remaining_leg_checks():
    check = check_legs([leg(bid=0.0, open_interest=None)], AS_OF, CFG)
    assert check.reasons == ["INVALID_QUOTE(SPY P690: 0.0/2.2)"]


def test_unknown_open_interest_can_be_downgraded_to_a_warning():
    check = check_legs([leg(open_interest=None)], AS_OF, GateConfig(allow_unknown_open_interest=True))
    assert check.reasons == []
    assert check.warnings == ["OPEN_INTEREST_UNKNOWN(SPY P690)"]


def test_low_open_interest_also_caps_size():
    check = check_legs([leg(open_interest=50)], AS_OF, CFG)
    assert check.capped_qty == 4  # 8% of 50
    assert check.warnings == ["QTY_CAPPED_BY_OI(SPY P690: 4)"]


def test_open_interest_then_top_of_book_caps():
    check = check_legs([leg(qty=50, open_interest=500, bid_size=100)], AS_OF, CFG)
    assert check.capped_qty == 30
    assert check.warnings == ["QTY_CAPPED_BY_OI(SPY P690: 40)", "QTY_CAPPED_BY_TOB(SPY P690: 30)"]


def test_top_of_book_side_follows_the_leg_side():
    sell = check_legs([leg(qty=10, bid_size=10, ask_size=1_000)], AS_OF, CFG)
    buy = check_legs([leg(qty=10, side="BUY", bid_size=1, ask_size=20)], AS_OF, CFG)
    assert (sell.capped_qty, buy.capped_qty) == (3, 6)


def test_short_call_expiry_and_quantity_are_collected():
    legs = [
        leg(instrument_id="SPY C720", right="C", strike=720.0, extrinsic_value=1.4, qty=5),
        leg(instrument_id="SPY C725", right="C", side="BUY", strike=725.0, qty=3, expiry_date=date(2027, 1, 15)),
    ]
    check = check_legs(legs, AS_OF, CFG)
    assert check.capped_qty == 3
    assert check.max_expiry == date(2027, 1, 15)
    assert (check.has_short_call, check.short_call_strike, check.short_call_extrinsic) == (True, 720.0, 1.4)


def test_tob_size_zero_is_not_converted_and_rejected():
    """bid_size=0 on a SELL leg is not converted to 10 and is rejected by check_legs()."""
    check = check_legs([leg(bid_size=0, ask_size=0)], AS_OF, CFG)
    assert "TOB_SIZE_UNKNOWN(SPY P690)" not in check.reasons  # 0 is known, just low
    # With min_tob_size=4, size 0 should trigger LOW_TOB_SIZE
    has_low_tob = any("LOW_TOB_SIZE" in r for r in check.reasons)
    assert has_low_tob


def test_tob_size_none_rejected_when_unknown_not_allowed():
    """bid_size=None (surface fallback without TOB) is rejected when allow_unknown_tob_size=False."""
    cfg = GateConfig(allow_unknown_tob_size=False)
    check = check_legs([leg(bid_size=None, ask_size=None)], AS_OF, cfg)
    assert any("TOB_SIZE_UNKNOWN" in r for r in check.reasons)


def test_tob_size_none_allowed_as_warning_when_configured():
    """bid_size=None is downgraded to a warning when allow_unknown_tob_size=True."""
    cfg = GateConfig(allow_unknown_tob_size=True)
    check = check_legs([leg(bid_size=None, ask_size=None)], AS_OF, cfg)
    assert check.reasons == []  # No block reasons
    assert any("TOB_SIZE_UNKNOWN" in w for w in check.warnings)


def test_tob_participation_zero_blocks_when_cap_is_zero():
    """tob_size=2 with max_tob_participation=0.25 produces tob_cap=0 and blocks entry."""
    cfg = GateConfig(max_tob_participation=0.25, min_tob_size=1)
    check = check_legs([leg(qty=5, bid_size=2, ask_size=2)], AS_OF, cfg)
    assert check.capped_qty == 0
    assert any("TOB_PARTICIPATION_ZERO" in r for r in check.reasons)


def test_oi_participation_zero_blocks_when_cap_is_zero():
    """open_interest=5 with max_oi_participation=0.10 produces oi_cap=0 and blocks entry."""
    cfg = GateConfig(max_oi_participation=0.10)
    check = check_legs([leg(qty=5, open_interest=5)], AS_OF, cfg)
    assert check.capped_qty == 0
    assert any("OI_PARTICIPATION_ZERO" in r for r in check.reasons)


def test_no_legs_means_nothing_to_trade():
    check = check_legs([], AS_OF, CFG)
    assert (check.capped_qty, check.max_expiry) == (0, AS_OF)


# --- Tail budget -----------------------------------------------------------------------------------


def test_es_budget_caps_lots():
    check = es_budget_check(0.0, 300.0, 5, CFG)  # $1,000 budget
    assert (check.capped_qty, check.reasons) == (3, [])
    assert check.warnings == ["QTY_CAPPED_BY_ES99(3 lots)"]
    assert check.projected_es99_usd == pytest.approx(900.0)


@pytest.mark.parametrize("current", [900.0, 1_000.0, 1_200.0])
def test_es_budget_exhausted(current):
    check = es_budget_check(current, 300.0, 5, CFG)
    assert check.reasons == [f"SLEEVE_ES99_BUDGET_EXHAUSTED(cur=${current:.0f}+$300>$1000)"]


def test_zero_incremental_tail_risk_is_rejected():
    """Zero or negative incremental ES99 is fail-closed with INVALID_INCREMENTAL_ES99."""
    check = es_budget_check(400.0, 0.0, 5, CFG)
    assert (check.capped_qty, check.reasons) == (0, ["INVALID_INCREMENTAL_ES99"])

    check = es_budget_check(400.0, -1.0, 5, CFG)
    assert (check.capped_qty, check.reasons) == (0, ["INVALID_INCREMENTAL_ES99"])

    check = es_budget_check(400.0, float("nan"), 5, CFG)
    assert (check.capped_qty, check.reasons) == (0, ["INVALID_INCREMENTAL_ES99"])
