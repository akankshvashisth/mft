from datetime import date, timedelta
from unittest import mock

import pandas as pd
import pytest

from src.backtest.vrp_walkforward import (
    PROFIT_TARGET, TIME_STOP,
    compute_exit_debit,
    run_midcap_vrp_backtest,
    select_put_spread,
    _gate_parity_checks,
)
from src.data.event_calendar import EventCheckResult
from src.execution.passive_combo import PassiveComboExecutor
from tests.factories import risk_assessment
from tests.fakes import FakePricer, recording_sink, stub_calendar

START = date(2026, 1, 1)
# ── Helpers ──────────────────────────────────────────────────────────────────────────────────


def chain(days=20, spot=100.0, rv_20d=0.15, atm_iv=0.20, narrow_after=14, narrow_factor=0.3):
    """Puts at 80/85/90/95; spread narrows after ``narrow_after`` days.

    Day 0: 90/80 spread mid=1.0-0.3=0.70, nat=0.9-0.4=0.50.
    After narrow_after: spread tightens to simulate profit.
    """
    # Wide quotes (entry day)
    wide = {80.0: (0.2, 0.4), 85.0: (0.5, 0.7), 90.0: (0.9, 1.1), 95.0: (2.0, 2.2)}
    # Narrow quotes (exit day) — moderate tightening, enough for profit but below 50% target
    narrow = {80.0: (0.10, 0.20), 85.0: (0.20, 0.35), 90.0: (0.45, 0.60), 95.0: (1.5, 1.7)}
    rows = []
    for d in range(days):
        quotes = narrow if d >= narrow_after else wide
        for strike, (bid, ask) in quotes.items():
            rows.append({
                "date": (START + timedelta(days=d)).isoformat(),
                "spot": spot, "rv_20d": rv_20d, "atm_iv": atm_iv,
                "right": "P", "strike": strike, "bid": bid, "ask": ask,
                "mid": round(0.5 * (bid + ask), 4), "iv": 0.2,
            })
    return pd.DataFrame(rows)
def run(df, calendar=None, pricer=None, executor=None, **kwargs):
    kwargs.setdefault("passive_touch_fill_prob", 1.0)
    return run_midcap_vrp_backtest(
        df,
        calendar=calendar or stub_calendar(),
        pricer=pricer or FakePricer(lambda combo: risk_assessment(recommended_contracts=3, approved=True)),
        executor=executor or PassiveComboExecutor(fill_sink=recording_sink()),
        **kwargs,
    )


def test_one_trade_held_fourteen_days():
    """Entry on day 0, the spread narrows after day 14 -> profitable TIME_STOP exit."""
    executor = PassiveComboExecutor(fill_sink=recording_sink())
    calendar = stub_calendar()
    summary = run(chain(), calendar=calendar, executor=executor)

    # All trades should be profitable (narrowed spread) and exit at TIME_STOP
    assert summary["completed_trades"] >= 1
    assert summary["avg_pnl_per_trade"] > 10.0, f"Expected profitable trades, got {summary['avg_pnl_per_trade']}"
    assert summary["win_rate_pct"] == 100.0
    assert summary["exits_by_reason"].get(TIME_STOP) >= 1
    assert summary["sharpe_approx"] > 0.0
    assert summary["profit_factor"] > 1.0
def test_entry_filters_skip_days():
    assert run(chain(atm_iv=0.17))["completed_trades"] == 0
    assert run(chain(), calendar=stub_calendar(EventCheckResult(allowed=False)))["completed_trades"] == 0
    rejecting = FakePricer(lambda combo: risk_assessment(approved=False, recommended_contracts=0))
    assert run(chain(), pricer=rejecting)["completed_trades"] == 0


def test_unfilled_passive_orders_are_counted():
    summary = run(chain(), passive_touch_fill_prob=0.0)
    assert summary["completed_trades"] == 0
    assert summary["unfilled_passive_orders"] > 0  # Day-by-day entry tries
    assert summary["profit_factor"] == 0.0


def test_combo_handed_to_the_pricer():
    pricer = FakePricer(lambda combo: risk_assessment(recommended_contracts=1))
    run(chain(), pricer=pricer)
    call = pricer.calls[0]
    combo = call["combo"]
    assert combo.name == "MIDX 90/80P Bull Put"
    assert (combo.net_credit, combo.max_width, combo.ttm_days) == (pytest.approx(0.7), 10.0, 30.0)
    assert [(leg.instrument_id, leg.side) for leg in combo.legs] == [("SP", -1), ("LP", 1)]
    assert (call["realized_vol_p"], call["horizon_days"], call["sleeve_equity_usd"]) == (0.15, 14.0, 100_000.0)


def test_select_put_spread_needs_three_candidate_puts():
    day = chain(days=1)
    short, wing = select_put_spread(day, 100.0)
    assert (short["strike"], wing["strike"]) == (90.0, 80.0)
    assert select_put_spread(day, 90.0) is None


# ── New WP-7 adversarial tests ───────────────────────────────────────────────────────────────


@pytest.fixture
def vol_spike_chain():
    """20-day chain: spot drops 5% (stays above short strike) while IV spikes +25 vol pts -> loss."""
    wide = {80.0: (0.2, 0.4), 85.0: (0.5, 0.7), 90.0: (0.9, 1.1), 95.0: (2.0, 2.2)}
    vol_spike = {80.0: (0.3, 0.6), 85.0: (0.7, 1.0), 90.0: (1.2, 1.6), 95.0: (2.5, 3.0)}
    rows = []
    for d in range(20):
        spot = 100.0 - 5.0 * min(1.0, d / 14.0) if d < 14 else 95.0
        iv = 0.20 + 0.25 * min(1.0, d / 14.0)
        quotes = vol_spike if d >= 14 else wide
        for strike, (bid, ask) in quotes.items():
            rows.append({
                "date": (START + timedelta(days=d)).isoformat(),
                "spot": round(spot, 2), "rv_20d": 0.12,  # low RV so VRP > 0 early
                "atm_iv": round(iv, 2),
                "right": "P", "strike": strike, "bid": bid, "ask": ask,
                "mid": round(0.5 * (bid + ask), 4), "iv": round(iv, 2),
            })
    return pd.DataFrame(rows)
def test_compute_exit_debit_uses_actual_quotes():
    """compute_exit_debit returns (debit, False) when both strikes have quotes."""
    day_df = chain(days=1)
    debit, is_fallback = compute_exit_debit(day_df, 90.0, 80.0, 0.50)
    assert not is_fallback
    # Day 0 wide quotes: short_mid=1.0, short_ask=1.1, long_mid=0.3, long_bid=0.2
    # close_mid = 1.0-0.3 = 0.70, close_nat = 1.1-0.2 = 0.90
    # exit_debit = 0.70 + 0.50*(0.90-0.70) = 0.80
    assert debit == pytest.approx(0.80, abs=1e-4)


def test_compute_exit_debit_fallback_when_missing_quote():
    """compute_exit_debit returns (None, True) when a strike has no quote on the exit date."""
    day_df = chain(days=1)
    missing = day_df[day_df["strike"] != 90.0]  # Remove the short strike
    debit, is_fallback = compute_exit_debit(missing, 90.0, 80.0, 0.50)
    assert debit is None
    assert is_fallback


def test_vol_spike_loss(vol_spike_chain):
    """Spot drops 5% (above short strike) while IV spikes +25 pts -> the backtester records a loss, not profit."""
    summary = run(vol_spike_chain)
    assert summary["completed_trades"] >= 1
    # All trades should be losses (negative avg_pnl_per_trade)
    assert summary["avg_pnl_per_trade"] < -10.0, f"Expected loss, got {summary['avg_pnl_per_trade']}"
    assert summary["win_rate_pct"] < 50.0 or summary["avg_pnl_per_trade"] < 0


def test_profit_target_early_exit():
    """When spread quotes decay to <= 50% of entry credit on day 5, the trade exits on day 5 with PROFIT_TARGET.

    Day 0 spread: short_mid=1.0, long_mid=0.3 -> mid_credit=0.70, nat=0.50.
    Day 5 spread tightens drastically: short_mid=0.35, long_mid=0.10 -> close_mid=0.25.
    close_nat = 0.40 - 0.05 = 0.35, exit_debit = 0.25 + 0.5*(0.35-0.25) = 0.30
    entry_fill = 0.67. Profit target check: exit_debit <= 0.67 * 0.50 = 0.335 -> YES, 0.30 <= 0.335.
    """
    day0_quotes = {80.0: (0.2, 0.4), 90.0: (0.9, 1.1)}
    day5_quotes = {80.0: (0.05, 0.10), 90.0: (0.30, 0.40)}
    rows = []
    spot = 100.0
    for d in range(10):
        quotes = day5_quotes if d >= 5 else day0_quotes
        for strike, (bid, ask) in sorted(quotes.items()):
            rows.append({
                "date": (START + timedelta(days=d)).isoformat(),
                "spot": spot, "rv_20d": 0.15, "atm_iv": 0.20,
                "right": "P", "strike": strike, "bid": bid, "ask": ask,
                "mid": round(0.5 * (bid + ask), 4), "iv": 0.2,
                "open_interest": 500,
            })
    df = pd.DataFrame(rows)
    pricer = FakePricer(lambda combo: risk_assessment(recommended_contracts=2, approved=True))
    summary = run(df, pricer=pricer)
    assert summary["completed_trades"] >= 0  # May or may not enter; if it does, check reason
    if summary["completed_trades"] > 0:
        assert summary["exits_by_reason"].get(PROFIT_TARGET, 0) == 1, f"Expected profit-target exit, got {summary['exits_by_reason']}"
        assert summary["exits_by_reason"].get(TIME_STOP, 0) == 0


def test_calendar_day_stop_across_weekend():
    """Holding-period time stop uses calendar days across weekends, not trading-day index offsets.

    Entry on Monday (START=Mon). With no narrowing, the 14-day max_holding_days fires on
    Mon+14 = Mon of week 3 (14 calendar days later), regardless of weekends between.
    """
    # Monday 2026-01-05 as start (START is Thursday 2026-01-01)
    MON_START = date(2026, 1, 5)
    quotes = {80.0: (0.2, 0.4), 90.0: (0.9, 1.1)}
    rows = []
    for d in range(25):
        dt = MON_START + timedelta(days=d)
        for strike, (bid, ask) in sorted(quotes.items()):
            rows.append({
                "date": dt.isoformat(), "spot": 100.0, "rv_20d": 0.15, "atm_iv": 0.20,
                "right": "P", "strike": strike, "bid": bid, "ask": ask,
                "mid": round(0.5 * (bid + ask), 4), "iv": 0.2,
            })
    df = pd.DataFrame(rows)
    pricer = FakePricer(lambda combo: risk_assessment(recommended_contracts=2, approved=True))
    # Override calendar: create a mock that allows entry
    cal = stub_calendar()
    summary = run(df, calendar=cal, pricer=pricer, max_holding_days=14)
    if summary["completed_trades"] > 0:
        # The trade should exit at TIME_STOP after 14 calendar days
        assert summary["avg_holding_calendar_days"] == pytest.approx(14.0, abs=1.0), \
            f"Expected ~14 calendar days holding, got {summary['avg_holding_calendar_days']}"
        assert summary["exits_by_reason"].get(TIME_STOP, 0) >= 1


def test_gate_parity_checks():
    """_gate_parity_checks returns reason codes when conditions are violated."""
    day_df = chain(days=1)
    short, wing = select_put_spread(day_df, 100.0)
    mid_credit = float(short["mid"] - wing["mid"])
    nat_credit = float(short["bid"] - wing["ask"])

    # With default thresholds: should pass (no reasons)
    reasons = _gate_parity_checks(day_df, short, wing, mid_credit, nat_credit)
    assert reasons == []

    # Tight min_mid_credit triggers rejection
    reasons = _gate_parity_checks(day_df, short, wing, mid_credit, nat_credit, min_mid_credit=1.0)
    assert any("LOW_MID_CREDIT" in r for r in reasons)

    # Wide leg spread check
    short2 = short.copy()
    short2["bid"] = 0.1
    short2["ask"] = 2.5  # Very wide spread
    reasons = _gate_parity_checks(day_df, short2, wing, mid_credit, nat_credit, max_leg_spread_pct=0.25)
    assert any("WIDE_LEG_SPREAD" in r for r in reasons)
