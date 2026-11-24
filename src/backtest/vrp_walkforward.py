"""Walk-forward mid-cap VRP simulation on a daily option chain, with passive fill haircuts and friction."""
from __future__ import annotations

import math
from datetime import date, timedelta
from typing import Any

import numpy as np
import pandas as pd

from src.common.constants import DEFAULT_RISK_FREE_RATE
from src.common.timeutil import days_to_years
from src.execution.passive_combo import CREDIT_OPEN, PassiveComboExecutor
from src.risk.calendar import EventCalendarLike, default_event_calendar
from src.scanner.combos import CandidateCombo, CandidateLeg

HOLDING_DAYS = 14
TARGET_DTE_DAYS = 30
MIN_VRP = 0.025
SHORT_STRIKE_MAX_FRAC = 0.94
MIN_MID_CREDIT = 0.25
MIN_NATURAL_CREDIT = 0.05
EXIT_TIME_VALUE_SHARE = 0.35
ROUND_TRIP_COMMISSION_PER_LOT = 2.60
ASSUMED_CARRY = 0.03
ASSUMED_DIVIDEND_YIELD = 0.0125


def select_put_spread(day_df: pd.DataFrame, spot: float) -> tuple[pd.Series, pd.Series] | None:
    """(short, wing) puts: the highest strike at or below 94% of spot, and two strikes further down."""
    puts = day_df[day_df["right"] == "P"].sort_values("strike")
    short_candidates = puts[puts["strike"] <= spot * SHORT_STRIKE_MAX_FRAC]
    if len(short_candidates) < 3:
        return None
    return short_candidates.iloc[-1], short_candidates.iloc[-3]


def approx_exit_pnl(
    short_strike: float,
    long_strike: float,
    width: float,
    entry_fill: float,
    end_spot: float,
    lots: int,
) -> float:
    """Net USD P&L of closing at the horizon: intrinsic loss plus a share of the remaining credit as exit cost.

    The exit debit is capped at the spread width; round-trip commissions are charged per lot.
    """
    intrinsic_loss = max(0.0, short_strike - end_spot) - max(0.0, long_strike - end_spot)
    exit_debit = min(width, intrinsic_loss + EXIT_TIME_VALUE_SHARE * max(0.0, entry_fill - intrinsic_loss))
    return ((entry_fill - exit_debit) * 100.0 - ROUND_TRIP_COMMISSION_PER_LOT) * lots


def put_spread_combo(symbol: str, spot: float, short: pd.Series, wing: pd.Series, mid_credit: float, width: float) -> CandidateCombo:
    def leg(name: str, row: pd.Series, side: int) -> CandidateLeg:
        iv = float(row["iv"])
        return CandidateLeg(name, float(row["strike"]), "P", side, 1, float(row["bid"]), float(row["ask"]), float(row["mid"]), iv, iv, 0.0, 1.0)

    return CandidateCombo(
        name=f"{symbol} {int(short['strike'])}/{int(wing['strike'])}P Bull Put",
        strategy_type="BULL_PUT_SPREAD",
        underlying=symbol,
        spot=spot,
        forward=spot * math.exp(ASSUMED_CARRY * days_to_years(TARGET_DTE_DAYS)),
        ttm_days=float(TARGET_DTE_DAYS),
        risk_free_rate=DEFAULT_RISK_FREE_RATE,
        dividend_yield=ASSUMED_DIVIDEND_YIELD,
        legs=[leg("SP", short, -1), leg("LP", wing, +1)],
        net_credit=mid_credit,
        max_width=width,
    )


def run_midcap_vrp_backtest(
    df: pd.DataFrame,
    sleeve_equity_usd: float = 100_000.0,
    passive_touch_fill_prob: float = 0.50,
    seed: int = 42,
    *,
    symbol: str = "MIDX",
    calendar: EventCalendarLike | None = None,
    pricer: Any = None,
    executor: PassiveComboExecutor | None = None,
) -> dict[str, Any]:
    """Walk-forward mid-cap VRP simulation with passive fill haircuts & realistic friction.

    df holds one row per option per day: date (ISO), spot, rv_20d, atm_iv, right, strike, bid, ask, mid, iv.
    """
    if pricer is None:
        from src.risk.scenario_pricer import ScenarioPricer

        pricer = ScenarioPricer(n_scenarios=5_000, commission_per_leg=0.65, slippage_fraction=0.50)
    executor = executor or PassiveComboExecutor()
    calendar = calendar if calendar is not None else default_event_calendar()
    rng = np.random.default_rng(seed)

    equity = sleeve_equity_usd
    equity_curve = [equity]
    trades: list[dict[str, Any]] = []
    skipped_unfilled = 0

    grouped = list(df.groupby("date", sort=True))
    idx = 0
    while idx < len(grouped) - (HOLDING_DAYS + 1):
        dt_str, day_df = grouped[idx]
        as_of = date.fromisoformat(dt_str)
        spot = float(day_df["spot"].iloc[0])
        rv_20d = float(day_df["rv_20d"].iloc[0])
        vrp = float(day_df["atm_iv"].iloc[0]) - rv_20d

        # Require positive VRP (+2.5 vol pts) and clean event calendar
        ev_chk = calendar.check_entry(symbol, as_of=as_of, expiry_date=as_of + timedelta(days=TARGET_DTE_DAYS))
        if vrp < MIN_VRP or not ev_chk.allowed:
            idx += 1
            continue

        pair = select_put_spread(day_df, spot)
        if pair is None:
            idx += 1
            continue
        short, wing = pair
        width = float(short["strike"] - wing["strike"])
        mid_credit = float(short["mid"] - wing["mid"])
        nat_credit = float(short["bid"] - wing["ask"])
        if mid_credit <= MIN_MID_CREDIT or nat_credit <= MIN_NATURAL_CREDIT:
            idx += 1
            continue

        combo = put_spread_combo(symbol, spot, short, wing, mid_credit, width)
        risk = pricer.evaluate_combo(combo, realized_vol_p=rv_20d, horizon_days=float(HOLDING_DAYS), sleeve_equity_usd=equity)
        if not risk.approved:
            idx += 1
            continue

        # Conservative passive execution simulation: fill probability when touched; the plan is a preview only
        clips = executor.create_parent_plan(
            symbol, combo.name, CREDIT_OPEN, risk.recommended_contracts, spot, mid_credit, nat_credit, register=False
        )
        if rng.random() > passive_touch_fill_prob:
            skipped_unfilled += 1
            idx += 1
            continue

        entry_fill = clips[0].current_limit_price
        lots = risk.recommended_contracts

        # Hold up to 14 days (approximate mid-horizon exit value + exit friction + round-trip commissions)
        exit_idx = min(idx + HOLDING_DAYS, len(grouped) - 1)
        end_spot = float(grouped[exit_idx][1]["spot"].iloc[0])
        net_pnl_usd = approx_exit_pnl(float(short["strike"]), float(wing["strike"]), width, entry_fill, end_spot, lots)

        equity += net_pnl_usd
        equity_curve.append(equity)
        trades.append({"entry_date": dt_str, "lots": lots, "entry_credit": entry_fill, "pnl_usd": round(net_pnl_usd, 2)})
        idx = exit_idx + 1

    curve = np.asarray(equity_curve, dtype=float)
    max_dd_pct = float(np.min(curve / np.maximum.accumulate(curve) - 1.0))
    pnls = np.array([t["pnl_usd"] for t in trades], dtype=float)
    return {
        "starting_equity": sleeve_equity_usd,
        "ending_equity": round(equity, 2),
        "total_return_pct": round(((equity - sleeve_equity_usd) / sleeve_equity_usd) * 100.0, 2),
        "max_drawdown_pct": round(max_dd_pct * 100.0, 2),
        "completed_trades": len(trades),
        "unfilled_passive_orders": skipped_unfilled,
        "win_rate_pct": round(float(np.mean(pnls > 0) * 100.0), 1) if len(pnls) else 0.0,
        "avg_pnl_per_trade": round(float(np.mean(pnls)), 2) if len(pnls) else 0.0,
    }
