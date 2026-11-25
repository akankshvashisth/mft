#!/usr/bin/env python3
"""5-experiment VRP ablation harness."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.backtest.vrp_walkforward import run_midcap_vrp_backtest
from src.execution.passive_combo import PassiveComboExecutor
from src.risk.calendar import default_event_calendar
from src.risk.scenario_pricer import ScenarioPricer


def _build_pricer() -> Any:
    return ScenarioPricer(n_scenarios=5_000, commission_per_leg=0.65, slippage_fraction=0.50)


def _build_executor() -> Any:
    return PassiveComboExecutor(fill_sink=lambda _: None)


def run_ablation_suite(
    df: pd.DataFrame,
    symbol: str,
    *,
    pricer: Any = None,
    calendar: Any = None,
) -> dict[str, list[dict[str, Any]]]:
    """Run 4 single-symbol experiments."""
    pricer = pricer or _build_pricer()
    executor = _build_executor()
    cal = calendar or default_event_calendar()
    results: dict[str, list[dict[str, Any]]] = {}
    # 1. Friction Frontier
    results["friction_frontier"] = []
    for fill_prob in [0.25, 0.50, 0.75]:
        for slip_frac in [0.25, 0.50, 0.75, 1.00]:
            s = run_midcap_vrp_backtest(df, symbol=symbol, pricer=pricer, executor=executor, calendar=cal,
                passive_touch_fill_prob=fill_prob, exit_slippage_frac=slip_frac)
            results["friction_frontier"].append({"passive_touch_fill_prob": fill_prob, "exit_slippage_frac": slip_frac, **s})
    # 2. Exit Rules
    results["exit_rules"] = []
    for pt_frac in [0.25, 0.50, 0.75, 1.00]:
        for min_dte in [7, 14, 21]:
            s = run_midcap_vrp_backtest(df, symbol=symbol, pricer=pricer, executor=executor, calendar=cal,
                profit_target_fraction=pt_frac, min_exit_dte=min_dte)
            results["exit_rules"].append({"profit_target_fraction": pt_frac, "min_exit_dte": min_dte, **s})
    # 3. Vertical Selection
    results["vertical_selection"] = []
    try:
        s = run_midcap_vrp_backtest(df, symbol=symbol, pricer=pricer, executor=executor, calendar=cal, use_delta_targeted=False)
        results["vertical_selection"].append({"mode": "adjacent", **s})
    except Exception as e:
        results["vertical_selection"].append({"mode": "adjacent", "error": str(e)})
    try:
        s = run_midcap_vrp_backtest(df, symbol=symbol, pricer=pricer, executor=executor, calendar=cal, use_delta_targeted=True)
        results["vertical_selection"].append({"mode": "delta_targeted", **s})
    except Exception as e:
        results["vertical_selection"].append({"mode": "delta_targeted", "error": str(e)})
    # 4. VRP Threshold Sweep
    results["vrp_threshold"] = []
    for mv in [0.00, 0.025, 0.05, 0.08]:
        s = run_midcap_vrp_backtest(df, symbol=symbol, pricer=pricer, executor=executor, calendar=cal, min_vrp=mv)
        results["vrp_threshold"].append({"min_vrp": mv, **s})
    return results


def _print_table(title, rows, cols):
    print()
    print(f"### {title}")
    print()
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
        print(f"| {" | ".join(vals)} |")
    print()


def main():
    p = argparse.ArgumentParser(description="VRP ablation experiment suite.")
    p.add_argument("--parquet", nargs="+", required=True)
    p.add_argument("--out-json", default="logs/vrp_experiments.json")
    args = p.parse_args()
    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    all_results: dict[str, Any] = {}
    for pq_path in args.parquet:
        path = Path(pq_path)
        if not path.exists():
            print(f"SKIP {pq_path}", file=sys.stderr)
            continue
        sym = path.stem.replace("_opra_daily", "").replace("_opra_chain", "")
        df = pd.read_parquet(path)
        print()
        print(f"{'=' * 60}")
        print(f"  {sym} ({len(df)} rows, {df['date'].min()}..{df['date'].max()})")
        print(f"{'=' * 60}")
        suite = run_ablation_suite(df, symbol=sym)
        all_results[sym] = suite
        if suite.get("friction_frontier"):
            _print_table(f"{sym} -- Friction Frontier", suite["friction_frontier"],
                ["passive_touch_fill_prob","exit_slippage_frac","completed_trades","avg_pnl_per_trade","sharpe_approx","total_friction_usd"])
        if suite.get("exit_rules"):
            _print_table(f"{sym} -- Exit Rules", suite["exit_rules"],
                ["profit_target_fraction","min_exit_dte","completed_trades","avg_pnl_per_trade","exits_by_reason","avg_holding_calendar_days"])
        if suite.get("vertical_selection"):
            _print_table(f"{sym} -- Vertical Selection", suite["vertical_selection"],
                ["mode","completed_trades","avg_pnl_per_trade","sharpe_approx","profit_factor"])
        if suite.get("vrp_threshold"):
            _print_table(f"{sym} -- VRP Threshold Sweep", suite["vrp_threshold"],
                ["min_vrp","completed_trades","avg_pnl_per_trade","sharpe_approx"])
    if len(args.parquet) > 1:
        print()
        print("### Cross-Symbol Comparison (default params)")
        print()
        comp = []
        for sym, suite in all_results.items():
            base = suite.get("vrp_threshold", [])
            row = next((r for r in base if abs(r.get("min_vrp", 0.025) - 0.025) < 1e-4), None)
            if row is None:
                row = suite.get("friction_frontier", [None])[0]
            if row:
                comp.append({"symbol": sym, **row})
        if comp:
            _print_table("Cross-Symbol Comparison", comp,
                ["symbol","completed_trades","avg_pnl_per_trade","sharpe_approx","profit_factor","total_friction_usd"])
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
