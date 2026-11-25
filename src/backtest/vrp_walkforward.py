"""Walk-forward mid-cap VRP simulation on a daily option chain, with quote-marked exits and live-gate parity."""
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

# --- Default constants ---------------------------------------------------------------------------
HOLDING_DAYS = 14
TARGET_DTE_DAYS = 30
MIN_VRP = 0.025
SHORT_STRIKE_MAX_FRAC = 0.94
MIN_MID_CREDIT = 0.25
MIN_NATURAL_CREDIT = 0.05
EXIT_SLIPPAGE_FRAC = 0.50
ROUND_TRIP_COMMISSION_PER_LOT = 2.60
PROFIT_TARGET_FRACTION = 0.50
MIN_EXIT_DTE = 14                             # Matches live VrpSettings
ASSUMED_CARRY = 0.03
ASSUMED_DIVIDEND_YIELD = 0.0125
MAX_LEG_SPREAD_PCT = 0.18                     # Matches live GateConfig
MIN_OPEN_INTEREST = 100

# --- Exit reason codes ---------------------------------------------------------------------------
PROFIT_TARGET = "PROFIT_TARGET"
TIME_STOP = "TIME_STOP"
def select_put_spread(day_df: pd.DataFrame, spot: float) -> tuple[pd.Series, pd.Series] | None:
    """(short, wing) puts: the highest strike at or below 94% of spot, and two strikes further down."""
    puts = day_df[day_df["right"] == "P"].sort_values("strike")
    short_candidates = puts[puts["strike"] <= spot * SHORT_STRIKE_MAX_FRAC]
    if len(short_candidates) < 3:
        return None
    return short_candidates.iloc[-1], short_candidates.iloc[-3]


def _quote_by_strike(
    day_df: pd.DataFrame, strike: float, right: str = "P",
    expiry: str | None = None,
) -> pd.Series | None:
    """Return the option row for a given strike and right (C/P) on a given day, or None if missing.

    When ``expiry`` is provided and the DataFrame has an ``"expiry"`` column, the lookup
    also filters by the exact expiry.  This enables multi-expiry backtesting where the
    same strike may exist across multiple expirations.
    """
    matches = day_df[(day_df["right"] == right) & (day_df["strike"] == strike)]
    if expiry is not None and "expiry" in day_df.columns:
        matches = matches[matches["expiry"] == expiry]
    if len(matches) == 0:
        return None
    return matches.iloc[0]


def compute_exit_debit(
    day_df: pd.DataFrame,
    short_strike: float,
    long_strike: float,
    exit_slippage_frac: float,
    expiry: str | None = None,
) -> tuple[float | None, bool]:
    """Exit debit per share to close a bull put spread using day-$t$ quotes.

    When ``expiry`` is provided and the DataFrame has an ``"expiry"`` column, both leg
    lookups are filtered by the exact expiry.  This ensures multi-expiry chains do not
    cross-pollute quotes across expirations.

    Returns (exit_debit, is_fallback). When either strike lacks a quote on the exit date,
    ``is_fallback = True`` and the function returns ``None`` as the debit.
    """
    short_row = _quote_by_strike(day_df, short_strike, right="P", expiry=expiry)
    long_row = _quote_by_strike(day_df, long_strike, right="P", expiry=expiry)
    if short_row is not None and long_row is not None:
        short_mid = float(short_row["mid"])
        short_ask = float(short_row["ask"])
        long_mid = float(long_row["mid"])
        long_bid = float(long_row["bid"])
        close_mid = short_mid - long_mid
        close_nat = short_ask - long_bid
        exit_debit = close_mid + exit_slippage_frac * (close_nat - close_mid)
        return max(exit_debit, 0.0), False
    return None, True


def bsm_fallback_exit_debit(
    short_strike: float,
    long_strike: float,
    spot: float,
    atm_iv: float,
    ttm_remaining_days: float,
    exit_slippage_frac: float,
    risk_free_rate: float = DEFAULT_RISK_FREE_RATE,
    dividend_yield: float = ASSUMED_DIVIDEND_YIELD,
) -> tuple[float, float]:
    """BSM-repriced exit debit when one or both strikes have no quote on the exit date.

    Returns (exit_debit, net_mid) where net_mid = short_bsm - long_bsm.
    """
    try:
        from src.quant.black_scholes import bsm_prices as price_fn
    except ImportError:
        from src.risk.scenario_pricer import fast_vollib_price as price_fn
    ttm_years = ttm_remaining_days / 365.25
    ttm_years = max(ttm_years, 1e-6)
    short_bsm = float(np.asarray(price_fn(spot, short_strike, ttm_years, risk_free_rate, atm_iv, "p", dividend_yield)).flat[0])
    long_bsm = float(np.asarray(price_fn(spot, long_strike, ttm_years, risk_free_rate, atm_iv, "p", dividend_yield)).flat[0])
    net_mid = short_bsm - long_bsm
    short_spread = max(0.05, 0.10 * abs(short_bsm))
    long_spread = max(0.05, 0.10 * abs(long_bsm))
    penalty = exit_slippage_frac * (short_spread + long_spread) / 2.0
    return max(net_mid + penalty, 0.0), net_mid


def put_spread_combo(
    symbol: str, spot: float, short: pd.Series, wing: pd.Series, mid_credit: float, width: float,
    ttm_days: float = float(TARGET_DTE_DAYS),
) -> CandidateCombo:
    def leg(name: str, row: pd.Series, side: int) -> CandidateLeg:
        iv = float(row["iv"])
        return CandidateLeg(
            name, float(row["strike"]), "P", side, 1,
            float(row["bid"]), float(row["ask"]), float(row["mid"]),
            iv, iv, 0.0, 1.0,
        )

    return CandidateCombo(
        name=f"{symbol} {int(short['strike'])}/{int(wing['strike'])}P Bull Put",
        strategy_type="BULL_PUT_SPREAD",
        underlying=symbol,
        spot=spot,
        forward=spot * math.exp(ASSUMED_CARRY * days_to_years(ttm_days)),
        ttm_days=float(ttm_days),
        risk_free_rate=DEFAULT_RISK_FREE_RATE,
        dividend_yield=ASSUMED_DIVIDEND_YIELD,
        legs=[leg("SP", short, -1), leg("LP", wing, +1)],
        net_credit=mid_credit,
        max_width=width,
    )


def _gate_parity_checks(
    day_df: pd.DataFrame,
    short: pd.Series,
    wing: pd.Series,
    mid_credit: float,
    nat_credit: float,
    *,
    min_mid_credit: float = MIN_MID_CREDIT,
    min_natural_credit: float = MIN_NATURAL_CREDIT,
    max_leg_spread_pct: float = MAX_LEG_SPREAD_PCT,
    min_open_interest: int = MIN_OPEN_INTEREST,
) -> list[str]:
    """Check live-gate entry conditions. Returns list of reason codes (empty = pass)."""
    reasons: list[str] = []
    if mid_credit <= min_mid_credit:
        reasons.append(f"LOW_MID_CREDIT({mid_credit:.2f} <= {min_mid_credit})")
    if nat_credit <= min_natural_credit:
        reasons.append(f"LOW_NATURAL_CREDIT({nat_credit:.2f} <= {min_natural_credit})")
    for label, leg in [("short", short), ("long", wing)]:
        bid = float(leg["bid"])
        ask = float(leg["ask"])
        mid = float(leg["mid"])
        if mid > 0:
            spread_pct = (ask - bid) / mid
            if spread_pct > max_leg_spread_pct:
                reasons.append(f"WIDE_LEG_SPREAD({label}:{spread_pct:.2%} > {max_leg_spread_pct:.0%})")
    if "open_interest" in day_df.columns:
        for label, leg in [("short", short), ("long", wing)]:
            oi_val = float(leg.get("open_interest", 0))
            if oi_val < min_open_interest:
                reasons.append(f"LOW_OPEN_INTEREST({label}:{oi_val} < {min_open_interest})")
    return reasons


def select_entry_slice(
    day_df: pd.DataFrame,
    as_of: date,
    target_dte: int = TARGET_DTE_DAYS,
    min_entry_dte: int = 21,
    max_entry_dte: int = 45,
) -> tuple[pd.DataFrame, date, str | None]:
    """Select the expiry slice for entry on ``as_of``.

    For backward compatibility, when ``"expiry"`` is not in ``day_df.columns`` the function
    returns the whole DataFrame unchanged with a synthetic expiry date ``as_of + target_dte``
    and ``None`` for the ISO string.

    When multi-expiry data *is* present, it finds the expiry whose calendar DTE is in
    ``[min_entry_dte, max_entry_dte]`` and closest to ``target_dte`` (tie-break: longer DTE).
    If none qualifies, returns an empty DataFrame.

    Returns
    -------
    (slice_df, chosen_date, chosen_iso)
        - ``slice_df``: rows of ``day_df`` filtered to the chosen expiry (or all rows).
        - ``chosen_date``: ``date`` of the expiry.
        - ``chosen_iso``: ISO string of the expiry (``None`` for single-expiry data).
    """
    if "expiry" not in day_df.columns:
        chosen_date = as_of + timedelta(days=target_dte)
        return day_df, chosen_date, None

    day_dtes: dict[str, int] = {}
    for exp_str in day_df["expiry"].unique():
        try:
            exp_date = date.fromisoformat(exp_str)
        except (ValueError, TypeError):
            continue
        day_dtes[exp_str] = (exp_date - as_of).days

    # Filter by entry DTE range
    eligible = {k: v for k, v in day_dtes.items() if min_entry_dte <= v <= max_entry_dte}
    if not eligible:
        return pd.DataFrame(columns=day_df.columns), as_of, None  # skip

    # Choose expiry minimising |dte - target_dte|; tie-break: longer DTE
    best_exp = min(eligible, key=lambda k: (abs(eligible[k] - target_dte), -eligible[k]))
    chosen_date = date.fromisoformat(best_exp)
    slice_df = day_df[day_df["expiry"] == best_exp].copy()
    return slice_df, chosen_date, best_exp


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
    exit_slippage_frac: float = EXIT_SLIPPAGE_FRAC,
    profit_target_fraction: float = PROFIT_TARGET_FRACTION,
    min_exit_dte: int = MIN_EXIT_DTE,
    max_holding_days: int = HOLDING_DAYS,
    commission_per_lot: float = ROUND_TRIP_COMMISSION_PER_LOT,
    max_concurrent_trades: int = 1,
    max_leg_spread_pct: float = MAX_LEG_SPREAD_PCT,
    min_open_interest: int = MIN_OPEN_INTEREST,
    use_delta_targeted: bool = False,
    min_vrp: float = MIN_VRP,
    min_tob_size: int = 4,
    max_tob_participation: float = 0.25,
) -> dict[str, Any]:
    """Walk-forward mid-cap VRP simulation with quote-marked exits, live-gate parity,
    multi-expiry entry selection, optional delta-targeted verticals, and TOB size parity.

    Parameters
    ----------
    df : pd.DataFrame
        One row per option per day: date (ISO), spot, rv_20d, atm_iv,
        right, strike, bid, ask, mid, iv.  Optionally ``open_interest``,
        ``expiry`` (ISO), ``bid_size``, ``ask_size``.
    sleeve_equity_usd : float
        Starting equity.
    passive_touch_fill_prob : float
        Probability that a touched passive order fills.
    seed : int
        RNG seed.
    symbol : str
        Underlying symbol for calendar checks.
    calendar : EventCalendarLike or None
        Calendar for entry blackout checks.
    pricer : ScenarioPricer or None
        Pricer for risk assessment at entry.
    executor : PassiveComboExecutor or None
        Executor for passive fill simulation.
    exit_slippage_frac : float
        Fraction of half-spread conceded at exit.
    profit_target_fraction : float
        Exit when exit_debit <= entry_fill * (1 - profit_target_fraction).
    min_exit_dte : int
        Calendar days to expiry below which a TIME_STOP exit fires.
    max_holding_days : int
        Max calendar holding days before a TIME_STOP exit fires.
    commission_per_lot : float
        Round-trip commissions deducted from each trade's P&L.
    max_concurrent_trades : int
        Maximum number of concurrently open trades (default 1).
    max_leg_spread_pct : float
        Maximum allowed leg spread as fraction of mid.
    min_open_interest : int
        Minimum open interest to allow leg entry (applied when column present).
    use_delta_targeted : bool
        If True, use ``delta_targeted_verticals`` from ``src.scanner.verticals``
        instead of adjacent-strike selection.  Requires ``expiry`` and ``iv`` columns.
    min_vrp : float
        Minimum VRP (atm_iv - rv_20d) to attempt an entry.
    min_tob_size : int
        Minimum bid/ask size required on each leg when ``bid_size``/``ask_size`` columns exist.
    max_tob_participation : float
        Caps lots to a fraction of the minimum TOB size among the two legs.

    Returns
    -------
    dict with summary keys including sharpe_approx, profit_factor,
    avg_holding_calendar_days, exits_by_reason, fallback_exit_marks, total_friction_usd.
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
    gate_rejects = 0
    fallback_exit_marks = 0
    exits_by_reason: dict[str, int] = {PROFIT_TARGET: 0, TIME_STOP: 0}
    open_trades: list[dict[str, Any]] = []

    grouped = list(df.groupby("date", sort=True))
    grouped_dates = [date.fromisoformat(dt_str) for dt_str, _ in grouped]

    for idx, (dt_str, day_df) in enumerate(grouped):
        as_of = date.fromisoformat(dt_str)
        spot = float(day_df["spot"].iloc[0])
        rv_20d = float(day_df["rv_20d"].iloc[0])
        atm_iv = float(day_df["atm_iv"].iloc[0])
        vrp = atm_iv - rv_20d

        # --- 1. Check / exit open trades ---------------------------------------------------------
        still_open: list[dict[str, Any]] = []
        for t in open_trades:
            short_strike = t["short_strike"]
            long_strike = t["long_strike"]
            expiry_str = t.get("expiry")  # None for single-expiry chains
            exit_debit, is_fallback = compute_exit_debit(
                day_df, short_strike, long_strike, exit_slippage_frac, expiry=expiry_str,
            )
            if is_fallback:
                days_held = (as_of - t["entry_date"]).days
                expiry_date = t["expiry_date"]
                ttm_remaining = max((expiry_date - as_of).days, 1.0)
                bsm_debit, net_mid = bsm_fallback_exit_debit(
                    short_strike, long_strike, spot, atm_iv,
                    ttm_remaining, exit_slippage_frac,
                )
                exit_debit = bsm_debit
                exit_mid = max(net_mid, 0.0)
                fallback_exit_marks += 1
            else:
                exit_mid = 0.0
            entry_fill = t["entry_fill"]
            lots = t["lots"]

            # 50% profit-target early exit
            if exit_debit <= entry_fill * (1.0 - profit_target_fraction):
                if not is_fallback:
                    srow = _quote_by_strike(day_df, short_strike, right="P", expiry=expiry_str)
                    lrow = _quote_by_strike(day_df, long_strike, right="P", expiry=expiry_str)
                    if srow is not None and lrow is not None:
                        exit_mid = max(float(srow["mid"]) - float(lrow["mid"]), 0.0)
                exit_slippage = max(0.0, exit_debit - exit_mid)
                pnl = (entry_fill - exit_debit) * 100.0 * lots - commission_per_lot * lots
                t.update(pnl_usd=round(pnl, 2), exit_reason=PROFIT_TARGET,
                         exit_debit=exit_debit, exit_date=as_of,
                         holding_calendar_days=(as_of - t["entry_date"]).days,
                         exit_slippage=round(exit_slippage, 4))
                trades.append(t)
                equity += pnl
                equity_curve.append(equity)
                exits_by_reason[PROFIT_TARGET] += 1
                continue

            # Calendar-day time stop
            expiry_date = t["expiry_date"]
            dte = (expiry_date - as_of).days
            days_held = (as_of - t["entry_date"]).days
            if dte <= min_exit_dte or days_held >= max_holding_days:
                if not is_fallback:
                    srow = _quote_by_strike(day_df, short_strike, right="P", expiry=expiry_str)
                    lrow = _quote_by_strike(day_df, long_strike, right="P", expiry=expiry_str)
                    if srow is not None and lrow is not None:
                        exit_mid = max(float(srow["mid"]) - float(lrow["mid"]), 0.0)
                exit_slippage = max(0.0, exit_debit - exit_mid)
                pnl = (entry_fill - exit_debit) * 100.0 * lots - commission_per_lot * lots
                t.update(pnl_usd=round(pnl, 2), exit_reason=TIME_STOP,
                         exit_debit=exit_debit, exit_date=as_of,
                         holding_calendar_days=days_held,
                         exit_slippage=round(exit_slippage, 4))
                trades.append(t)
                equity += pnl
                equity_curve.append(equity)
                exits_by_reason[TIME_STOP] += 1
                continue

            still_open.append(t)
        open_trades = still_open
# --- 2. Try to enter a new trade ---------------------------------------------------------
        if idx >= len(grouped) - 1:
            continue  # Don't enter on the last day (trades would force-close immediately)
        if len(open_trades) >= max_concurrent_trades:
            continue

        # -- Multi-expiry entry selection --
        entry_slice, chosen_expiry_date, chosen_expiry_iso = select_entry_slice(
            day_df, as_of, target_dte=TARGET_DTE_DAYS,
        )
        if entry_slice.empty:
            continue

        entry_ttm_days = float((chosen_expiry_date - as_of).days)

        ev_chk = calendar.check_entry(symbol, as_of=as_of, expiry_date=chosen_expiry_date)
        if vrp < min_vrp or not ev_chk.allowed:
            continue

        if use_delta_targeted:
            # -- Delta-targeted mode --
            from src.scanner.verticals import delta_targeted_verticals

            puts = entry_slice[entry_slice["right"] == "P"].sort_values("strike")
            puts_dicts = []
            for _, row in puts.iterrows():
                d = dict(row)
                d.setdefault("raw_iv", float(row.get("iv", 0.35)))
                d.setdefault("svi_iv", float(row.get("iv", 0.35)))
                d.setdefault("vega", 0.5)
                d.setdefault("edge_to_half_spread", 0.0)
                d.setdefault("open_interest", 0)
                puts_dicts.append(d)

            pairs = delta_targeted_verticals(
                puts_dicts, [],
                spot=spot,
                short_delta_range=(0.15, 0.30),
                min_width_usd=2.50,
                min_width_pct_spot=0.04,
                max_width_pct_spot=0.12,
                min_mid_credit=MIN_MID_CREDIT,
                ttm_days=entry_ttm_days,
            )
            if not pairs:
                continue

            best_pair = None
            best_risk = None
            best_lots = 0
            for pair in pairs:
                short_row = _quote_by_strike(entry_slice, pair.short["strike"], right="P", expiry=chosen_expiry_iso)
                wing_row = _quote_by_strike(entry_slice, pair.long["strike"], right="P", expiry=chosen_expiry_iso)
                if short_row is None or wing_row is None:
                    continue
                width = pair.width
                mid_credit = pair.mid_credit
                nat_credit = pair.nat_credit
                if "bid_size" in entry_slice.columns:
                    sb = int(short_row.get("bid_size", 0))
                    wa = int(wing_row.get("ask_size", 0))
                    if sb < min_tob_size or wa < min_tob_size:
                        continue
                gate_reasons = _gate_parity_checks(
                    entry_slice, short_row, wing_row, mid_credit, nat_credit,
                    max_leg_spread_pct=max_leg_spread_pct, min_open_interest=min_open_interest,
                )
                if gate_reasons:
                    gate_rejects += 1
                    continue
                combo = put_spread_combo(symbol, spot, short_row, wing_row, mid_credit, width, ttm_days=entry_ttm_days)
                risk = pricer.evaluate_combo(
                    combo, realized_vol_p=rv_20d, horizon_days=float(max_holding_days),
                    sleeve_equity_usd=equity,
                )
                if not risk.approved:
                    continue
                if best_pair is None or risk.net_ev_usd > best_risk.net_ev_usd:
                    best_pair = (short_row, wing_row, width, mid_credit, nat_credit, combo)
                    best_risk = risk
                    best_lots = risk.recommended_contracts

            if best_pair is None:
                continue
            short, wing, width, mid_credit, nat_credit, combo = best_pair
            risk = best_risk
            lots = best_lots

        else:
            pair = select_put_spread(entry_slice, spot)
            if pair is None:
                continue
            short, wing = pair
            width = float(short["strike"] - wing["strike"])
            mid_credit = float(short["mid"] - wing["mid"])
            nat_credit = float(short["bid"] - wing["ask"])
            if "bid_size" in entry_slice.columns:
                sb = int(short.get("bid_size", 0))
                wa = int(wing.get("ask_size", 0))
                if sb < min_tob_size or wa < min_tob_size:
                    gate_rejects += 1
                    continue
            gate_reasons = _gate_parity_checks(
                entry_slice, short, wing, mid_credit, nat_credit,
                max_leg_spread_pct=max_leg_spread_pct, min_open_interest=min_open_interest,
            )
            if gate_reasons:
                gate_rejects += 1
                continue
            combo = put_spread_combo(symbol, spot, short, wing, mid_credit, width, ttm_days=entry_ttm_days)
            risk = pricer.evaluate_combo(
                combo, realized_vol_p=rv_20d, horizon_days=float(max_holding_days),
                sleeve_equity_usd=equity,
            )
            if not risk.approved:
                continue
            lot_cap = 999_999
            if "bid_size" in entry_slice.columns:
                sb = int(short.get("bid_size", 0))
                wa = int(wing.get("ask_size", 0))
                lot_cap = min(sb, wa)
            lots = min(risk.recommended_contracts, lot_cap)

        # TOB participation cap
        if "bid_size" in entry_slice.columns:
            sb = int(short.get("bid_size", 0))
            wa = int(wing.get("ask_size", 0))
            capped = min(lots, int(sb * max_tob_participation), int(wa * max_tob_participation))
            if capped < 1:
                continue
            lots = capped

        clips = executor.create_parent_plan(
            symbol, combo.name, CREDIT_OPEN, lots, spot, mid_credit, nat_credit,
            register=False
        )
        if rng.random() > passive_touch_fill_prob:
            skipped_unfilled += 1
            continue

        entry_fill = float(clips[0].current_limit_price)
        open_trades.append({
            "entry_date": as_of, "entry_fill": entry_fill, "lots": lots,
            "short_strike": float(short["strike"]),
            "long_strike": float(wing["strike"]),
            "width": width,
            "expiry_date": chosen_expiry_date,
            "expiry": chosen_expiry_iso,
            "entry_credit": mid_credit,
        })
# --- Force-close trades still open at simulation end --------------------------------------------
    if open_trades and grouped:
        last_group = grouped[-1]
        last_df = last_group[1]
        for t in open_trades:
            expiry_str = t.get("expiry")
            exit_debit, is_fallback = compute_exit_debit(
                last_df, t["short_strike"], t["long_strike"], exit_slippage_frac, expiry=expiry_str,
            )
            days_held = (grouped_dates[-1] - t["entry_date"]).days
            exit_mid = 0.0
            if is_fallback:
                last_spot = float(last_df["spot"].iloc[0])
                last_iv = float(last_df["atm_iv"].iloc[0])
                expiry_date = t["expiry_date"]
                ttm_remaining = max((expiry_date - grouped_dates[-1]).days, 1.0)
                bsm_debit, net_mid = bsm_fallback_exit_debit(
                    t["short_strike"], t["long_strike"], last_spot, last_iv,
                    ttm_remaining, exit_slippage_frac,
                )
                exit_debit = bsm_debit
                exit_mid = max(net_mid, 0.0)
                fallback_exit_marks += 1
            else:
                srow = _quote_by_strike(last_df, t["short_strike"], right="P", expiry=expiry_str)
                lrow = _quote_by_strike(last_df, t["long_strike"], right="P", expiry=expiry_str)
                if srow is not None and lrow is not None:
                    exit_mid = max(float(srow["mid"]) - float(lrow["mid"]), 0.0)
            exit_slippage = max(0.0, exit_debit - exit_mid)
            pnl = (t["entry_fill"] - exit_debit) * 100.0 * t["lots"] - commission_per_lot * t["lots"]
            trades.append({
                **t, "pnl_usd": round(pnl, 2), "exit_reason": TIME_STOP,
                "exit_debit": exit_debit, "exit_date": grouped_dates[-1],
                "holding_calendar_days": days_held,
                "exit_slippage": round(exit_slippage, 4),
            })
            equity += pnl
            equity_curve.append(equity)
            exits_by_reason[TIME_STOP] += 1
# --- Compute summary --------------------------------------------------------------------------
    curve = np.asarray(equity_curve, dtype=float)
    max_dd_pct = float(np.min(curve / np.maximum.accumulate(curve) - 1.0)) if len(curve) > 0 else 0.0
    pnls = np.array([t["pnl_usd"] for t in trades], dtype=float) if trades else np.array([], dtype=float)
    n = len(pnls)

    total_return = equity - sleeve_equity_usd
    total_friction_usd = commission_per_lot * float(np.sum([t["lots"] for t in trades])) if trades else 0.0
    # Entry + exit spread slippage: credit given up stepping inside from mid on entry,
    # plus extra debit paid by not closing at mid on exit.
    for t in trades:
        entry_spread = max(0.0, t.get("entry_credit", 0.0) - t.get("entry_fill", 0.0))
        exit_slip = t.get("exit_slippage", 0.0)
        total_friction_usd += (entry_spread + exit_slip) * 100.0 * t["lots"]
    avg_hold = float(np.mean([t.get("holding_calendar_days", 0) for t in trades])) if trades else 0.0

    if n >= 2 and float(np.std(pnls)) > 0:
        sharpe_approx = float(np.mean(pnls)) / float(np.std(pnls))
    else:
        sharpe_approx = 0.0
    gross_wins = float(np.sum(pnls[pnls > 0])) if np.any(pnls > 0) else 0.0
    gross_losses = abs(float(np.sum(pnls[pnls < 0]))) if np.any(pnls < 0) else 0.0
    profit_factor = (gross_wins / gross_losses) if gross_losses > 0 else (99.9 if gross_wins > 0 else 0.0)

    return {
        "starting_equity": sleeve_equity_usd,
        "ending_equity": round(equity, 2),
        "total_return_pct": round((total_return / sleeve_equity_usd) * 100.0, 2),
        "max_drawdown_pct": round(max_dd_pct * 100.0, 2),
        "completed_trades": n,
        "unfilled_passive_orders": skipped_unfilled,
        "win_rate_pct": round(float(np.mean(pnls > 0) * 100.0), 1) if n else 0.0,
        "avg_pnl_per_trade": round(float(np.mean(pnls)), 2) if n else 0.0,
        # New WP-7 summary keys
        "sharpe_approx": round(sharpe_approx, 4),
        "profit_factor": round(profit_factor, 4),
        "avg_holding_calendar_days": round(avg_hold, 1),
        "exits_by_reason": {k: v for k, v in exits_by_reason.items()},
        "fallback_exit_marks": fallback_exit_marks,
        "total_friction_usd": round(total_friction_usd, 2),
    }
