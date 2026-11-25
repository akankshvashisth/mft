#!/usr/bin/env python3
"""5-experiment VRP ablation harness.

Usage
-----
    python scripts/run_vrp_experiments.py \\
        --parquet data/backtest/DECK_opra_daily.parquet \\
        --out-json logs/vrp_experiments.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.backtest.vrp_walkforward import run_midcap_vrp_backtest
from src.execution.passive_combo import PassiveComboExecutor
from tests.fakes import FakePricer, recording_sink, stub_calendar
from tests.factories import risk_assessment
from tests.factories import risk_assessment


def _build_pricer() -> Any:
    return FakePricer(lambda combo: risk_assessment(
        combo_name=combo.name, recommended_contracts=3, approved=True,
    ))


def _build_executor() -> Any:
    return PassiveComboExecutor(fill_sink=recording_sink())


def run_ablation_suite(
    df: pd.DataFrame,
    symbol: str,
    *,
    pricer: Any = None,
    calendar: Any = None,
) -> dict[str, list[dict[str, Any]]]:
    """Run 4 single-symbol experiments.

    Parameters
    ----------
    df : pd.DataFrame
        Canonical backtest chain.
    symbol : str
        Symbol name for the calendar.
    pricer : optional
        FakePricer or real pricer.
    calendar : optional
        Event calendar.

    Returns
    -------
    dict[str, list[dict]]
        Keys: ``friction_frontier``, ``exit_rules``, ``vertical_selection``, ``vrp_threshold``.
    """
    pricer = pricer or _build_pricer()
    executor = _build_executor()
    cal = calendar or stub_calendar()

    results: dict[str, list[dict[str, Any]]] = {}

    # 1. Friction Frontier
    results["friction_frontier"] = []
    for fill_prob in [0.25, 0.50, 0.75]:
        for slip_frac in [0.25, 0.50, 0.75, 1.00]:
            s = run_midcap_vrp_backtest(
                df, symbol=symbol, pricer=pricer, executor=executor, calendar=cal,
                passive_touch_fill_prob=fill_prob, exit_slippage_frac=slip_frac,
            )
            results["friction_frontier"].append({
                "passive_touch_fill_prob": fill_prob,
                "exit_slippage_frac": slip_frac,
                **s,
            })

    # 2. Exit Rules (TP vs Time-Stop)
    results["exit_rules"] = []
    for pt_frac in [0.25, 0.50, 0.75, 1.00]:
        for min_dte in [7, 14, 21]:
            s = run_midcap_vrp_backtest(
                df, symbol=symbol, pricer=pricer, executor=executor, calendar=cal,
                profit_target_fraction=pt_frac, min_exit_dte=min_dte,
            )
            results["exit_rules"].append({
                "profit_target_fraction": pt_frac,
                "min_exit_dte": min_dte,
                **s,
            })

    # 3. Vertical Selection: Adjacent vs Delta-Targeted
    results["vertical_selection"] = []
    try:
        s_adj = run_midcap_vrp_backtest(
            df, symbol=symbol, pricer=pricer, executor=executor, calendar=cal,
            use_delta_targeted=False,
        )
        results["vertical_selection"].append({"mode": "adjacent", **s_adj})
    except Exception as e:
        results["vertical_selection"].append({"mode": "adjacent", "error": str(e)})

    try:
        s_dt = run_midcap_vrp_backtest(
            df, symbol=symbol, pricer=pricer, executor=executor, calendar=cal,
            use_delta_targeted=True,
        )
        results["vertical_selection"].append({"mode": "delta_targeted", **s_dt})
    except Exception as e:
        results["vertical_selection"].append({"mode": "delta_targeted", "error": str(e)})

    # 4. VRP Threshold Sweep
    results["vrp_threshold"] = []
    for mv in [0.00, 0.025, 0.05, 0.08]:
        s = run_midcap_vrp_backtest(
            df, symbol=symbol, pricer=pricer, executor=executor, calendar=cal,
            min_vrp=mv,
        )
        results["vrp_threshold"].append({"min_vrp": mv, **s})

    return results


def _print_table(title: str, rows: list[dict[str, Any]], cols: list[str]) -> None:
    print(f"\n### {title}\n")
    header = " | ".join(cols)
    sep = " | ".join(["---"] * len(cols))
    print(f"| {header} |")
    print(f"| {sep} |")
    for r in rows:
        vals = []
        for c in cols:
            v = r.get(c, "")
            if isinstance(v, float):
                vals.append(f"{v:.4f}")
            elif isinstance(v, dict):
                vals.append(str(dict(sorted(v.items()))))
            else:
                vals.append(str(v))
        print(f"| {' | '.join(vals)} |")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description="VRP ablation experiment suite.")
    parser.add_argument("--parquet", nargs="+", required=True,
                        help="Path(s) to one or more canonical backtest Parquet files.")
    parser.add_argument("--out-json", default="logs/vrp_experiments.json",
                        help="Output JSON path (default logs/vrp_experiments.json).")
    args = parser.parse_args()

    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    all_results: dict[str, Any] = {}

    for pq_path in args.parquet:
        path = Path(pq_path)
        if not path.exists():
            print(f"SKIP {pq_path}: file not found", file=sys.stderr)
            continue

        symbol = path.stem.replace("_opra_daily", "").replace("_opra_chain", "")
        df = pd.read_parquet(path)
        print(f"\n{'='*60}")
        print(f"  {symbol} ({len(df)} rows, {df['date'].min()}..{df['date'].max()})")
        print(f"{'='*60}")

        suite = run_ablation_suite(df, symbol=symbol)
        all_results[symbol] = suite

        if suite["friction_frontier"]:
            _print_table(
                f"{symbol} — Friction Frontier",
                suite["friction_frontier"],
                ["passive_touch_fill_prob", "exit_slippage_frac", "completed_trades",
                 "avg_pnl_per_trade", "sharpe_approx", "total_friction_usd"],
            )
        if suite["exit_rules"]:
            _print_table(
                f"{symbol} — Exit Rules",
                suite["exit_rules"],
                ["profit_target_fraction", "min_exit_dte", "completed_trades",
                 "avg_pnl_per_trade", "exits_by_reason", "avg_holding_calendar_days"],
            )
        if suite["vertical_selection"]:
            _print_table(
                f"{symbol} — Vertical Selection",
                suite["vertical_selection"],
                ["mode", "completed_trades", "avg_pnl_per_trade", "sharpe_approx", "profit_factor"],
            )
        if suite["vrp_threshold"]:
            _print_table(
                f"{symbol} — VRP Threshold Sweep",
                suite["vrp_threshold"],
                ["min_vrp", "completed_trades", "avg_pnl_per_trade", "sharpe_approx"],
            )

    # Cross-symbol comparison
    if len(args.parquet) > 1:
        print("\n### Cross-Symbol Comparison (default params)\n")
        comp_rows: list[dict[str, Any]] = []
        for sym, suite in all_results.items():
            base = suite.get("vrp_threshold", [])
            default_row = next(
                (r for r in base if abs(r.get("min_vrp", 0.025) - 0.025) < 1e-4), None
            )
            if default_row is None:
                default_row = suite.get("friction_frontier", [None])[0]
            if default_row:
                comp_rows.append({"symbol": sym, **default_row})
        if comp_rows:
            _print_table(
                "Cross-Symbol Comparison",
                comp_rows,
                ["symbol", "completed_trades", "avg_pnl_per_trade", "sharpe_approx",
                 "profit_factor", "total_friction_usd"],
            )

    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()