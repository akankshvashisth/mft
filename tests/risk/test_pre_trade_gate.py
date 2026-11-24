from datetime import date
from unittest import mock

import pytest

import src.risk.pre_trade_gate as gate_module
from src.common import keys
from src.common.store import JsonStore, NullStore
from src.data.event_calendar import EventCheckResult
from src.risk.pre_trade_gate import CandidateLegSpec, GateConfig, GateDecision, PreTradeGate
from tests.factories import (
    NOW_NS,
    UNDERLYING,
    ago,
    futures_payload,
    metrics_payload,
    portfolio_payload,
    regime_payload,
    risk_status_payload,
    seed_healthy_inputs,
    surface_payload,
    utc_ns,
)
from tests.fakes import stub_calendar, valkey_client

EXPIRY = date(2026, 12, 18)
ES_PER_LOT, EV_PER_LOT = 150.0, 12.0


def spread_legs(qty=5, **wing_overrides):
    sold = CandidateLegSpec("SPY P690", "P", "SELL", 690.0, EXPIRY, qty, 2.0, 2.2, 100, 100, open_interest=10_000)
    wing = dict(
        instrument_id="SPY P685", right="P", side="BUY", strike=685.0, expiry_date=EXPIRY, qty=qty,
        bid=1.4, ask=1.6, bid_size=100, ask_size=100, open_interest=10_000,
    )
    wing.update(wing_overrides)
    return [sold, CandidateLegSpec(**wing)]


@pytest.fixture
def calendar():
    return stub_calendar()


@pytest.fixture
def make_gate(healthy_store, calendar, wall_clock, mono_clock):
    def make(store=None, config=None):
        return PreTradeGate(
            config or GateConfig(),
            calendar=calendar,
            store=store if store is not None else healthy_store,
            clock_ns=wall_clock,
            monotonic=mono_clock,
        )

    return make


def evaluate(gate, legs=None, es=ES_PER_LOT, ev=EV_PER_LOT, **kwargs) -> GateDecision:
    return gate.evaluate_combo(UNDERLYING, 700.0, legs or spread_legs(), es, ev, **kwargs)


def test_healthy_inputs_approve_the_requested_size(make_gate, calendar):
    decision = evaluate(make_gate())

    assert decision == GateDecision(
        approved=True, approved_qty=5, reasons=[], warnings=[], projected_book_es99_usd=750.0
    )
    calendar.check_entry.assert_called_once_with(
        UNDERLYING,
        as_of=date(2026, 11, 24),
        expiry_date=EXPIRY,
        has_short_call=False,
        short_call_strike=None,
        spot_price=700.0,
        short_call_extrinsic=None,
    )


def _set(key, payload):
    return lambda store: store.set_json(key, payload)


@pytest.mark.parametrize(
    "mutate, eval_kwargs, reason",
    [
        (lambda s: s.set_text(keys.RISK_HALTED, "1"), {}, "RISK_DAEMON_HALTED_OR_MISSING"),
        (_set(keys.RISK_STATUS, risk_status_payload(now_ns=ago(31.0))), {}, "RISK_DAEMON_STALE(31s)"),
        (_set(keys.REGIME_LATEST, regime_payload(short_vol_allowed=False)), {}, "MACRO_REGIME_BLOCKED(ratio=0.8824)"),
        (_set(keys.surface_key(UNDERLYING), surface_payload(now_ns=ago(121.0))), {}, "SVI_SURFACE_MISSING_OR_STALE"),
        (_set(keys.surface_key(UNDERLYING), surface_payload(rmse_vol_pts=0.8)), {}, "SVI_FIT_POOR(rmse=0.8pts>0.5)"),
        (
            _set(keys.metrics_key(UNDERLYING), metrics_payload(strategy_bias="DEFENSIVE_OR_LONG_CONVEXITY")),
            {},
            "SURFACE_BIAS_DEFENSIVE(VRP=0.03)",
        ),
        (_set(keys.FUTURES_LATEST, futures_payload(parity_vs_fut_diff_bps=30.0)), {}, "FUTURES_PARITY_DIVERGENCE(30.0bps)"),
        (lambda s: None, {"ev": -1.0}, "NON_POSITIVE_NET_EV($-1.00/lot)"),
        (lambda s: None, {"legs": spread_legs(open_interest=None)}, "OPEN_INTEREST_UNKNOWN(SPY P685)"),
        (lambda s: s.delete(keys.PORTFOLIO_RISK), {}, "PORTFOLIO_RISK_MISSING_OR_STALE"),
        (_set(keys.PORTFOLIO_RISK, portfolio_payload(book_es_99_usd=None)), {}, "PORTFOLIO_RISK_INVALID"),
        (
            _set(keys.PORTFOLIO_RISK, portfolio_payload(book_es_99_usd=1_000.0)),
            {},
            "SLEEVE_ES99_BUDGET_EXHAUSTED(cur=$1000+$150>$1000)",
        ),
    ],
)
def test_each_failed_input_blocks_with_its_reason(make_gate, healthy_store, mutate, eval_kwargs, reason):
    mutate(healthy_store)
    decision = evaluate(make_gate(), **eval_kwargs)

    assert not decision.approved
    assert decision.approved_qty == 0
    assert reason in decision.reasons


def test_reasons_keep_the_documented_order(make_gate, healthy_store, calendar):
    healthy_store.set_text(keys.RISK_HALTED, "1")
    healthy_store.set_json(keys.REGIME_LATEST, regime_payload(short_vol_allowed=False))
    healthy_store.delete(keys.PORTFOLIO_RISK)
    calendar.check_entry.return_value = EventCheckResult(False, ["EARNINGS_IN_WINDOW(2026-12-01)"])

    decision = evaluate(make_gate(), legs=spread_legs(open_interest=None), ev=-1.0)

    assert decision.reasons == [
        "RISK_DAEMON_HALTED_OR_MISSING",
        "MACRO_REGIME_BLOCKED(ratio=0.8824)",
        "NON_POSITIVE_NET_EV($-1.00/lot)",
        "OPEN_INTEREST_UNKNOWN(SPY P685)",
        "EARNINGS_IN_WINDOW(2026-12-01)",
        "PORTFOLIO_RISK_MISSING_OR_STALE",
    ]


def test_advisory_inputs_only_warn(make_gate, healthy_store, calendar):
    healthy_store.set_json(keys.FUTURES_LATEST, futures_payload(cross_check_valid=False))
    calendar.check_entry.return_value = EventCheckResult(True, [], "2026-12-04 US Non-Farm Payrolls")

    decision = evaluate(make_gate())

    assert decision.approved and decision.approved_qty == 5
    assert decision.warnings == [
        "FUTURES_CROSS_CHECK_UNAVAILABLE",
        "MACRO_EVENT_NEAR(2026-12-04 US Non-Farm Payrolls)",
    ]


def test_tail_budget_caps_the_approved_size(make_gate, healthy_store):
    healthy_store.set_json(keys.PORTFOLIO_RISK, portfolio_payload(book_es_99_usd=500.0))
    decision = evaluate(make_gate(), es=200.0)

    assert (decision.approved, decision.approved_qty) == (True, 2)
    assert decision.warnings == ["QTY_CAPPED_BY_ES99(2 lots)"]
    assert decision.projected_book_es99_usd == 900.0


def test_short_call_details_reach_the_calendar(make_gate, calendar):
    legs = [
        CandidateLegSpec("SPY C720", "C", "SELL", 720.0, EXPIRY, 5, 3.0, 3.2, 100, 100, 10_000, extrinsic_value=3.1),
        CandidateLegSpec("SPY C725", "C", "BUY", 725.0, EXPIRY, 5, 1.9, 2.1, 100, 100, 10_000),
    ]
    evaluate(make_gate(), legs=legs)
    call = calendar.check_entry.call_args.kwargs
    assert (call["has_short_call"], call["short_call_strike"], call["short_call_extrinsic"]) == (True, 720.0, 3.1)


def test_default_as_of_is_the_new_york_trading_date(make_gate, calendar, wall_clock):
    late_evening_ns = utc_ns(2026, 11, 25, 2, 0, 0)  # 21:00 New York on the 24th
    wall_clock.return_value = late_evening_ns
    store = seed_healthy_inputs(JsonStore(valkey_client()), now_ns=late_evening_ns)

    evaluate(make_gate(store=store))

    assert calendar.check_entry.call_args.kwargs["as_of"] == date(2026, 11, 24)


def test_explicit_as_of_wins(make_gate, calendar):
    evaluate(make_gate(), as_of=date(2026, 12, 1))
    assert calendar.check_entry.call_args.kwargs["as_of"] == date(2026, 12, 1)


def test_without_a_store_the_daemon_is_unreachable(calendar, wall_clock):
    for gate in (
        PreTradeGate(calendar=calendar, store=NullStore(), clock_ns=wall_clock),
        PreTradeGate(calendar=calendar, valkey_url="", clock_ns=wall_clock),
    ):
        assert gate.risk_daemon_block_reason() == "RISK_DAEMON_UNREACHABLE"
        decision = evaluate(gate)
        assert not decision.approved
        assert decision.reasons[0] == "RISK_DAEMON_UNREACHABLE"


def test_evaluations_do_not_consume_the_throttle_but_submissions_do(make_gate, mono_clock):
    gate = make_gate()
    for _ in range(10):
        assert evaluate(gate).approved

    for _ in range(6):
        gate.record_order_submission()
    assert "ORDER_RATE_THROTTLE(6/min)" in evaluate(gate).reasons

    mono_clock.return_value += 60.0
    assert evaluate(gate).approved


def test_default_calendar_is_only_built_when_none_is_injected(monkeypatch, healthy_store):
    default = mock.Mock(return_value=stub_calendar())
    monkeypatch.setattr(gate_module, "default_event_calendar", default)

    PreTradeGate(calendar=stub_calendar(), store=healthy_store)
    default.assert_not_called()
    gate = PreTradeGate(store=healthy_store)
    default.assert_called_once_with()
    assert gate.calendar is default.return_value


def test_wall_clock_is_injected(healthy_store, calendar):
    clock = mock.Mock(return_value=NOW_NS + 121 * 1_000_000_000)  # Every snapshot is now two minutes old
    decision = evaluate(PreTradeGate(calendar=calendar, store=healthy_store, clock_ns=clock))
    assert "MACRO_REGIME_MISSING_OR_STALE" in decision.reasons
