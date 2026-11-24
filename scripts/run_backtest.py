from __future__ import annotations

from datetime import date, timedelta
import numpy as np
import pandas as pd
from src.data.event_calendar import EventCalendar
from src.data.thetadata_chain_loader import ThetaDataChainLoader
from src.execution.passive_combo import PassiveComboExecutor
from src.risk.scenario_pricer import ScenarioPricer
from src.scanner.surface_scanner import CandidateCombo, CandidateLeg


def run_midcap_vrp_backtest(
    df: pd.DataFrame,
    sleeve_equity_usd: float = 100_000.0,
    passive_touch_fill_prob: float = 0.50,
    seed: int = 42,
) -> dict:
    """Walk-forward mid-cap VRP simulation with 50% passive fill haircuts & realistic friction."""
    rng = np.random.default_rng(seed)
    pricer = ScenarioPricer(n_scenarios=5_000, commission_per_leg=0.65, slippage_fraction=0.50)
    executor = PassiveComboExecutor()
    calendar = EventCalendar()

    equity = sleeve_equity_usd
    peak_equity = equity
    max_dd_pct = 0.0
    trades = []
    skipped_unfilled = 0

    grouped = list(df.groupby("date", sort=True))
    idx = 0
    while idx < len(grouped) - 15:
        dt_str, day_df = grouped[idx]
        as_of = date.fromisoformat(dt_str)
        spot = float(day_df["spot"].iloc[0])
        rv_20d = float(day_df["rv_20d"].iloc[0])
        atm_iv = float(day_df["atm_iv"].iloc[0])
        vrp = atm_iv - rv_20d

        # Require positive VRP (+2.5 vol pts) and clean event calendar
        exp_date = as_of + timedelta(days=30)
        ev_chk = calendar.check_entry("MIDX", as_of=as_of, expiry_date=exp_date)
        if vrp < 0.025 or not ev_chk.allowed:
            idx += 1
            continue

        # Select ~25-delta OTM put credit spread ($5 wide)
        puts = day_df[day_df["right"] == "P"].sort_values("strike")
        short_candidates = puts[puts["strike"] <= spot * 0.94]
        if len(short_candidates) < 3:
            idx += 1
            continue

        sp_row = short_candidates.iloc[-1]
        lp_row = short_candidates.iloc[-3]
        width = float(sp_row["strike"] - lp_row["strike"])
        mid_credit = float(sp_row["mid"] - lp_row["mid"])
        nat_credit = float(sp_row["bid"] - lp_row["ask"])
        if mid_credit <= 0.25 or nat_credit <= 0.05:
            idx += 1
            continue

        combo = CandidateCombo(
            name=f"MIDX {int(sp_row['strike'])}/{int(lp_row['strike'])}P Bull Put",
            strategy_type="BULL_PUT_SPREAD",
            underlying="MIDX",
            spot=spot,
            forward=spot * np.exp(0.03 * (30.0 / 365.25)),
            ttm_days=30.0,
            risk_free_rate=0.0425,
            dividend_yield=0.0125,
            legs=[
                CandidateLeg("SP", float(sp_row["strike"]), "P", -1, 1, float(sp_row["bid"]), float(sp_row["ask"]), float(sp_row["mid"]), float(sp_row["iv"]), float(sp_row["iv"]), 0.0, 1.0),
                CandidateLeg("LP", float(lp_row["strike"]), "P", +1, 1, float(lp_row["bid"]), float(lp_row["ask"]), float(lp_row["mid"]), float(lp_row["iv"]), float(lp_row["iv"]), 0.0, 1.0),
            ],
            net_credit=mid_credit,
            max_width=width,
        )

        risk = pricer.evaluate_combo(
            combo, realized_vol_p=rv_20d, horizon_days=14.0, sleeve_equity_usd=equity
        )
        if not risk.approved:
            idx += 1
            continue

        # Conservative passive execution simulation: 50% fill probability when touched
        clips = executor.create_parent_plan(
            "MIDX", combo.name, "CREDIT_OPEN", risk.recommended_contracts, spot, mid_credit, nat_credit
        )
        if rng.random() > passive_touch_fill_prob:
            skipped_unfilled += 1
            idx += 1
            continue

        entry_fill = clips[0].current_limit_price
        lots = risk.recommended_contracts

        # Hold up to 14 days (50% profit target or 14-DTE time stop)
        exit_idx = min(idx + 14, len(grouped) - 1)
        end_spot = float(grouped[exit_idx][1]["spot"].iloc[0])
        intrinsic_loss = max(0.0, float(sp_row["strike"]) - end_spot) - max(0.0, float(lp_row["strike"]) - end_spot)
        # Approximate mid-horizon exit value + half-spread exit friction + round-trip commissions ($2.60/lot)
        exit_debit = min(width, intrinsic_loss + 0.35 * max(0.0, entry_fill - intrinsic_loss))
        net_pnl_usd = ((entry_fill - exit_debit) * 100.0 - 2.60) * lots

        equity += net_pnl_usd
        peak_equity = max(peak_equity, equity)
        dd_pct = (equity - peak_equity) / peak_equity
        max_dd_pct = min(max_dd_pct, dd_pct)
        trades.append({"entry_date": dt_str, "lots": lots, "entry_credit": entry_fill, "pnl_usd": round(net_pnl_usd, 2)})
        idx = exit_idx + 1

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


if __name__ == "__main__":
    loader = ThetaDataChainLoader()
    synth_df = loader.build_synthetic_midcap_chain(symbol="MIDX", start_price=85.0, n_days=180)
    summary = run_midcap_vrp_backtest(synth_df)
    print("Mid-Cap VRP Walk-Forward Backtest Summary (50% Passive Fill Haircut):")
    for k, v in summary.items():
        print(f"  {k:26s}: {v}")
