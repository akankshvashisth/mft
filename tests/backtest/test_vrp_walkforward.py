from datetime import date, timedelta
from unittest import mock

import pandas as pd
import pytest

from src.backtest.vrp_walkforward import approx_exit_pnl, run_midcap_vrp_backtest, select_put_spread
from src.data.event_calendar import EventCheckResult
from src.execution.passive_combo import PassiveComboExecutor
from tests.factories import risk_assessment
from tests.fakes import FakePricer, recording_sink, stub_calendar

START = date(2026, 1, 1)


def chain(days=20, spot=100.0, rv_20d=0.15, atm_iv=0.20):
    """Puts at 80/85/90/95; only 80-90 are at or below 94% of spot. The 90/80 spread: mid 0.70, natural 0.50."""
    quotes = {80.0: (0.2, 0.4), 85.0: (0.5, 0.7), 90.0: (0.9, 1.1), 95.0: (2.0, 2.2)}
    rows = []
    for d in range(days):
        for strike, (bid, ask) in quotes.items():
            rows.append(
                {
                    "date": (START + timedelta(days=d)).isoformat(), "spot": spot, "rv_20d": rv_20d, "atm_iv": atm_iv,
                    "right": "P", "strike": strike, "bid": bid, "ask": ask, "mid": round(0.5 * (bid + ask), 4), "iv": 0.2,
                }
            )
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
    executor = PassiveComboExecutor(fill_sink=recording_sink())
    calendar = stub_calendar()
    summary = run(chain(), calendar=calendar, executor=executor)

    # Entry at 0.67 (15% of the 0.20 half-spread below mid); spot unchanged, so the exit costs 35% of the credit
    expected_pnl = ((0.67 - 0.35 * 0.67) * 100.0 - 2.60) * 3
    assert summary["completed_trades"] == 1
    assert summary["avg_pnl_per_trade"] == pytest.approx(round(expected_pnl, 2))
    assert summary["ending_equity"] == pytest.approx(100_000.0 + round(expected_pnl, 2), abs=0.01)
    assert (summary["win_rate_pct"], summary["max_drawdown_pct"], summary["unfilled_passive_orders"]) == (100.0, 0.0, 0)
    assert executor.active_clips == {}  # Preview clips are never registered as working
    assert calendar.check_entry.call_args_list[0] == mock.call("MIDX", as_of=START, expiry_date=START + timedelta(days=30))


def test_entry_filters_skip_days():
    assert run(chain(atm_iv=0.17))["completed_trades"] == 0  # VRP 2% < 2.5%
    assert run(chain(), calendar=stub_calendar(EventCheckResult(allowed=False)))["completed_trades"] == 0
    rejecting = FakePricer(lambda combo: risk_assessment(approved=False, recommended_contracts=0))
    assert run(chain(), pricer=rejecting)["completed_trades"] == 0


def test_unfilled_passive_orders_are_counted():
    summary = run(chain(), passive_touch_fill_prob=0.0)
    assert (summary["completed_trades"], summary["unfilled_passive_orders"]) == (0, 5)


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
    assert select_put_spread(day, 90.0) is None  # Only the 80 put is at or below 84.6


@pytest.mark.parametrize(
    "end_spot, pnl",
    [
        (100.0, ((0.67 - 0.35 * 0.67) * 100.0 - 2.60) * 2),  # Both strikes out of the money
        (85.0, ((0.67 - 5.0) * 100.0 - 2.60) * 2),  # Halfway into the spread
        (70.0, ((0.67 - 10.0) * 100.0 - 2.60) * 2),  # Loss capped at the width
    ],
)
def test_exit_pnl_cases(end_spot, pnl):
    assert approx_exit_pnl(90.0, 80.0, 10.0, 0.67, end_spot, 2) == pytest.approx(pnl)
