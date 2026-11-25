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


def chain(days=20, spot=100.0, rv_20d=0.15, atm_iv=0.20, narrow_after=14):
    """Puts at 80/85/90/95 with leg spreads <= 18%. Spread narrows after ``narrow_after`` days.

    Day 0: 90/80 spread mid=2.00-0.50=1.50, nat=1.90-0.52=1.38.
    After narrow_after: spread tightens (profitable but below 50% profit target).
    """
    # Wide quotes (entry day) — all leg spreads ≤ 18%
    wide = {80.0: (0.48, 0.52), 85.0: (0.95, 1.05), 90.0: (1.90, 2.10), 95.0: (3.00, 3.20)}
    # Narrow quotes (exit day) — moderate tightening
    narrow = {80.0: (0.14, 0.16), 85.0: (0.40, 0.44), 90.0: (0.84, 0.96), 95.0: (2.20, 2.40)}
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
    """Entry on day 0, the spread narrows after day 14 -> profitable TIME_STOP exit (max_holding_days=14)."""
    executor = PassiveComboExecutor(fill_sink=recording_sink())
    calendar = stub_calendar()
    # 16 days so only 1 trade fits (day 0 entry, day 14 exit, chain ends)
    summary = run(chain(days=15), calendar=calendar, executor=executor)

    assert summary["completed_trades"] == 1
    assert summary["avg_pnl_per_trade"] > 10.0, f"Expected profitable trade, got {summary['avg_pnl_per_trade']}"
    assert summary["win_rate_pct"] == 100.0
    assert summary["exits_by_reason"].get(TIME_STOP) == 1
    assert summary["fallback_exit_marks"] == 0
    assert summary["sharpe_approx"] >= 0.0
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
    assert (combo.net_credit, combo.max_width, combo.ttm_days) == (pytest.approx(1.5), 10.0, 30.0)
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
    wide = {80.0: (0.48, 0.52), 85.0: (0.95, 1.05), 90.0: (1.90, 2.10), 95.0: (3.00, 3.20)}
    vol_spike = {80.0: (0.58, 0.66), 85.0: (1.20, 1.35), 90.0: (2.20, 2.50), 95.0: (3.50, 4.00)}
    rows = []
    for d in range(20):
        spot = 100.0 - 5.0 * min(1.0, d / 14.0) if d < 14 else 95.0
        iv = 0.20 + 0.25 * min(1.0, d / 14.0)
        quotes = vol_spike if d >= 14 else wide
        for strike, (bid, ask) in quotes.items():
            rows.append({
                "date": (START + timedelta(days=d)).isoformat(),
                "spot": round(spot, 2), "rv_20d": 0.12,
                "atm_iv": round(iv, 2),
                "right": "P", "strike": strike, "bid": bid, "ask": ask,
                "mid": round(0.5 * (bid + ask), 4), "iv": round(iv, 2),
            })
    return pd.DataFrame(rows)


def test_quote_by_strike_filters_right():
    """_quote_by_strike returns None when only a Call row exists at the same strike (Put row missing)."""
    day_df = chain(days=1)
    # Add a Call row at strike 90 (same strike as one of the Puts)
    call_row = {
        "date": day_df["date"].iloc[0], "spot": 100.0, "rv_20d": 0.15, "atm_iv": 0.20,
        "right": "C", "strike": 90.0, "bid": 1.80, "ask": 2.00,
        "mid": 1.90, "iv": 0.2,
    }
    day_df = pd.concat([day_df, pd.DataFrame([call_row])], ignore_index=True)
    assert len(day_df[(day_df["strike"] == 90.0) & (day_df["right"] == "C")]) == 1  # Call exists
    # _quote_by_strike with default right="P" should still find the Put row
    from src.backtest.vrp_walkforward import _quote_by_strike
    row = _quote_by_strike(day_df, 90.0, right="P")
    assert row is not None
    assert float(row["mid"]) == 2.00  # The Put's mid, not the Call's 1.90
    # Explicit right="C" returns the Call row
    crow = _quote_by_strike(day_df, 90.0, right="C")
    assert crow is not None
    assert float(crow["mid"]) == 1.90
def test_compute_exit_debit_uses_actual_quotes():
    """compute_exit_debit returns (debit, False) when both strikes have quotes."""
    day_df = chain(days=1)
    debit, is_fallback = compute_exit_debit(day_df, 90.0, 80.0, 0.50)
    assert not is_fallback
    # Day 0 wide quotes: short_mid=2.00, short_ask=2.10, long_mid=0.50, long_bid=0.48
    # close_mid = 2.00-0.50 = 1.50, close_nat = 2.10-0.48 = 1.62
    # exit_debit = 1.50 + 0.50*(1.62-1.50) = 1.56
    assert debit == pytest.approx(1.56, abs=1e-4)


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
    assert summary["avg_pnl_per_trade"] < -10.0, f"Expected loss, got {summary['avg_pnl_per_trade']}"
    assert summary["win_rate_pct"] < 50.0 or summary["avg_pnl_per_trade"] < 0
    assert summary["fallback_exit_marks"] == 0  # All quotes present every step


def test_profit_target_early_exit():
    """When spread quotes decay to <= 50% of entry credit on day 5, the trade exits on day 5 with PROFIT_TARGET.

    Day 0: 90/80 spread mid=2.00-0.50=1.50, entry_fill=1.49.
    Day 5: spread tightens drastically: 90/80 close_mid ≈ 0.57-0.21=0.36,
            close_nat ≈ 0.59-0.20=0.39, exit_debit=0.36+0.5*0.03=0.375.
    Profit check: 0.375 <= 1.49*0.50 = 0.745 -> YES.
    """
    day0_quotes = {80.0: (0.48, 0.52), 85.0: (0.95, 1.05), 90.0: (1.90, 2.10)}
    day5_quotes = {80.0: (0.20, 0.22), 85.0: (0.50, 0.54), 90.0: (0.55, 0.59)}
    rows = []
    for d in range(6):
        quotes = day5_quotes if d >= 5 else day0_quotes
        for strike, (bid, ask) in sorted(quotes.items()):
            rows.append({
                "date": (START + timedelta(days=d)).isoformat(),
                "spot": 100.0, "rv_20d": 0.15, "atm_iv": 0.20,
                "right": "P", "strike": strike, "bid": bid, "ask": ask,
                "mid": round(0.5 * (bid + ask), 4), "iv": 0.2,
                "open_interest": 500,
            })
    df = pd.DataFrame(rows)
    pricer = FakePricer(lambda combo: risk_assessment(recommended_contracts=2, approved=True))
    summary = run(df, pricer=pricer, max_concurrent_trades=1)
    assert summary["completed_trades"] >= 1, f"Expected at least 1 trade, got {summary}"
    assert summary["exits_by_reason"].get(PROFIT_TARGET, 0) >= 1, \
        f"Expected profit-target exit, got {summary['exits_by_reason']}"
    assert summary["exits_by_reason"].get(TIME_STOP, 0) == 0
    assert summary["avg_holding_calendar_days"] == pytest.approx(5.0, abs=1.0)
    assert summary["fallback_exit_marks"] == 0


def test_calendar_day_stop_across_weekend():
    """Holding-period time stop uses calendar days across weekends, not trading-day index offsets.

    14 calendar days from entry fires even though only ~10 trading days have passed due to weekends.
    Entry day 0 (Mon 2026-01-05), exit happens on day 14 (Mon 2026-01-19) at TIME_STOP.
    Weekends are skipped so the df has fewer rows than calendar days.
    """
    MON_START = date(2026, 1, 5)
    quotes = {80.0: (0.48, 0.52), 85.0: (0.95, 1.05), 90.0: (1.90, 2.10)}
    rows = []
    for d in range(15):
        dt = MON_START + timedelta(days=d)
        if dt.weekday() >= 5:  # Skip weekends
            continue
        for strike, (bid, ask) in sorted(quotes.items()):
            rows.append({
                "date": dt.isoformat(), "spot": 100.0, "rv_20d": 0.15, "atm_iv": 0.20,
                "right": "P", "strike": strike, "bid": bid, "ask": ask,
                "mid": round(0.5 * (bid + ask), 4), "iv": 0.2,
            })
    df = pd.DataFrame(rows)
    pricer = FakePricer(lambda combo: risk_assessment(recommended_contracts=2, approved=True))
    cal = stub_calendar()
    summary = run(df, calendar=cal, pricer=pricer, max_concurrent_trades=1, max_holding_days=14)
    assert summary["completed_trades"] >= 1, f"Expected at least 1 trade, got {summary}"
    assert summary["avg_holding_calendar_days"] == pytest.approx(14.0, abs=1.0), \
        f"Expected ~14 calendar days holding, got {summary['avg_holding_calendar_days']}"
    assert summary["exits_by_reason"].get(TIME_STOP, 0) >= 1
    assert summary["fallback_exit_marks"] == 0


def test_force_close_with_live_quotes_on_short_chain():
    """5-day chain where a trade is still open on the last day, but live quotes exist -> force-close uses quotes.

    Chain has 5 days (range 0-4). Entry on day 0, trade stays open (5 days < 14 holding limit).
    On day 4 (last day), force-close runs: quotes exist, so no BSM fallback needed.
    """
    quotes = {80.0: (0.48, 0.52), 85.0: (0.95, 1.05), 90.0: (1.90, 2.10)}
    rows = []
    for d in range(5):
        for strike, (bid, ask) in sorted(quotes.items()):
            rows.append({
                "date": (START + timedelta(days=d)).isoformat(),
                "spot": 100.0, "rv_20d": 0.15, "atm_iv": 0.20,
                "right": "P", "strike": strike, "bid": bid, "ask": ask,
                "mid": round(0.5 * (bid + ask), 4), "iv": 0.2,
            })
    df = pd.DataFrame(rows)
    pricer = FakePricer(lambda combo: risk_assessment(recommended_contracts=2, approved=True))
    summary = run(df, pricer=pricer, max_concurrent_trades=1)
    assert summary["completed_trades"] == 1
    assert summary["fallback_exit_marks"] == 0
    assert summary["avg_holding_calendar_days"] == pytest.approx(4.0, abs=0.5)

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
    reasons = _gate_parity_checks(day_df, short, wing, mid_credit, nat_credit, min_mid_credit=2.0)
    assert any("LOW_MID_CREDIT" in r for r in reasons)

    # Wide leg spread check
    short2 = short.copy()
    short2["bid"] = 0.1
    short2["ask"] = 2.5  # Very wide spread
    reasons = _gate_parity_checks(day_df, short2, wing, mid_credit, nat_credit, max_leg_spread_pct=0.25)
    assert any("WIDE_LEG_SPREAD" in r for r in reasons)
