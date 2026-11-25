#!/usr/bin/env python3
"""Cost-safe CLI to download OPRA definition + CBBO-1m + underlying bars from Databento."""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.opra_chain_builder import build_opra_backtest_chain

DEFAULT_RAW_DIR = "data/databento/opra"
DEFAULT_OUT_DIR = "data/backtest"


def _build_cache_path(raw_dir, sym, schema, dataset, start, end):
    return raw_dir / f"{dataset}_{sym}_{schema}_{start}_{end}.dbn.zst"


def _load_dbn(path):
    import databento as db
    return db.DBNStore.from_file(str(path)).to_df()


def _underlying_bar_start(start, buffer_days=45):
    return (pd.Timestamp(start) - pd.Timedelta(days=buffer_days)).strftime("%Y-%m-%d")


def _check_local_thetadata_bars(symbol, start, end):
    import glob
    files = sorted(glob.glob(f"data/thetadata/bars/{symbol}_*.parquet"))
    if not files:
        return False
    try:
        df = pd.read_parquet(files[-1])
        if "date" not in df.columns:
            return False
        df["_d"] = pd.to_datetime(df["date"], errors="coerce")
        bar_start = _underlying_bar_start(start)
        return bool(not df.empty and df["_d"].min() <= pd.Timestamp(bar_start) and df["_d"].max() >= pd.Timestamp(end))
    except Exception:
        return False


def _opra_symbols(raw_list):
    return [f"{s}.OPT" for s in raw_list]


def _parse_snap_time(st):
    parts = st.strip().split(":")
    return int(parts[0]), int(parts[1])


def main():
    p = argparse.ArgumentParser(description="Download OPRA + underlying bars from Databento.")
    p.add_argument("--symbols", nargs="+", required=True)
    p.add_argument("--start", required=True)
    p.add_argument("--end", required=True)
    p.add_argument("--max-cost-usd", type=float, default=20.0)
    p.add_argument("--confirm", action="store_true")
    p.add_argument("--raw-dir", default=DEFAULT_RAW_DIR)
    p.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    p.add_argument("--snap-time", default="15:45")
    args = p.parse_args()

    raw_dir = Path(args.raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    symbols, start, end = args.symbols, args.start, args.end
    snap_hour_et, snap_minute_et = _parse_snap_time(args.snap_time)

    api_key = os.environ.get("DATABENTO_API_KEY", "")
    if not api_key or "YOUR_DATABENTO" in api_key:
        print("ERROR: Set DATABENTO_API_KEY.", file=sys.stderr)
        sys.exit(1)

    import databento as db
    client = db.Historical(key=api_key)

    rows, total_cost = [], 0.0
    opra_syms = _opra_symbols(symbols)

    for sym, psym in zip(symbols, opra_syms):
        dp = _build_cache_path(raw_dir, sym, "definition", "OPRA.PILLAR", start, end)
        dc = dp.exists()
        dcost = 0.0 if dc else float(client.metadata.get_cost(
            dataset="OPRA.PILLAR", symbols=[psym], schema="definition",
            start=start, end=end, stype_in="parent"))
        rows.append({"symbol": sym, "schema": "definition", "cached": dc, "cost_usd": dcost})
        total_cost += dcost

        cp = _build_cache_path(raw_dir, sym, "cbbo-1m", "OPRA.PILLAR", start, end)
        cc = cp.exists()
        ccost = 0.0 if cc else float(client.metadata.get_cost(
            dataset="OPRA.PILLAR", symbols=[psym], schema="cbbo-1m",
            start=start, end=end, stype_in="parent"))
        rows.append({"symbol": sym, "schema": "cbbo-1m", "cached": cc, "cost_usd": ccost})
        total_cost += ccost

        has_local = _check_local_thetadata_bars(sym, start, end)
        bar_start = _underlying_bar_start(start)
        ep = _build_cache_path(raw_dir.parent / "equities", sym, "ohlcv-1d", "EQUS.MINI", bar_start, end)
        ec = ep.exists()
        ecost = 0.0 if (has_local or ec) else float(client.metadata.get_cost(
            dataset="EQUS.MINI", symbols=[sym], schema="ohlcv-1d",
            start=bar_start, end=end, stype_in="raw_symbol"))
        rows.append({"symbol": sym, "schema": "ohlcv-1d (underlying)", "cached": (has_local or ec), "cost_usd": ecost})
        total_cost += ecost

    print(f"{"Symbol":<8} {"Schema":<24} {"Cached":<8} {"Cost USD":>10}")
    print("-" * 52)
    for r in rows:
        print(f"{r["symbol"]:<8} {r["schema"]:<24} {str(r["cached"]):<8} {r["cost_usd"]:>10.4f}")
    print("-" * 52)
    print(f"{"TOTAL":>42} {total_cost:>10.4f}")

    if not args.confirm:
        print("\nDry run. Re-run with --confirm.")
        sys.exit(0)
    if total_cost > args.max_cost_usd:
        print(f"\nERROR: ${total_cost:.4f} exceeds --max-cost-usd=${args.max_cost_usd:.2f}.", file=sys.stderr)
        sys.exit(1)

    for sym, psym in zip(symbols, opra_syms):
        print(f"\n--- {sym} ---")
        dp = _build_cache_path(raw_dir, sym, "definition", "OPRA.PILLAR", start, end)
        if not dp.exists():
            print("  Downloading definition...")
            client.timeseries.get_range(
                dataset="OPRA.PILLAR", symbols=[psym], schema="definition",
                start=start, end=end, stype_in="parent").to_file(str(dp))
        else:
            print(f"  [cached] {dp.name}")
        cp = _build_cache_path(raw_dir, sym, "cbbo-1m", "OPRA.PILLAR", start, end)
        if not cp.exists():
            print("  Downloading cbbo-1m...")
            client.timeseries.get_range(
                dataset="OPRA.PILLAR", symbols=[psym], schema="cbbo-1m",
                start=start, end=end, stype_in="parent").to_file(str(cp))
        else:
            print(f"  [cached] {cp.name}")
        has_local = _check_local_thetadata_bars(sym, start, end)
        bar_start = _underlying_bar_start(start)
        ep = raw_dir.parent / "equities" / f"EQUS.MINI_{sym}_ohlcv-1d_{bar_start}_{end}.dbn.zst"
        (raw_dir.parent / "equities").mkdir(parents=True, exist_ok=True)
        if not has_local and not ep.exists():
            print("  Downloading EQUS.MINI ohlcv-1d...")
            client.timeseries.get_range(
                dataset="EQUS.MINI", symbols=[sym], schema="ohlcv-1d",
                start=bar_start, end=end, stype_in="raw_symbol").to_file(str(ep))

    for sym in symbols:
        print(f"\n  Building chain for {sym}...")
        dp = _build_cache_path(raw_dir, sym, "definition", "OPRA.PILLAR", start, end)
        cp = _build_cache_path(raw_dir, sym, "cbbo-1m", "OPRA.PILLAR", start, end)
        if not dp.exists() or not cp.exists():
            print(f"  SKIP {sym}: missing cache")
            continue
        def_df, cbbo_df = _load_dbn(dp), _load_dbn(cp)
        has_local = _check_local_thetadata_bars(sym, start, end)
        bar_start = _underlying_bar_start(start)
        ep = raw_dir.parent / "equities" / f"EQUS.MINI_{sym}_ohlcv-1d_{bar_start}_{end}.dbn.zst"
        bars_df = None
        if has_local:
            import glob
            files = sorted(glob.glob(f"data/thetadata/bars/{sym}_*.parquet"))
            if files:
                bars_df = pd.read_parquet(files[-1])
                print(f"  Using ThetaData: {files[-1]}")
        elif ep.exists():
            bars_df = _load_dbn(ep)
            print(f"  Using EQUS.MINI: {ep.name}")
        if bars_df is None or bars_df.empty:
            print(f"  SKIP {sym}: no bars")
            continue
        chain = build_opra_backtest_chain(def_df, cbbo_df, bars_df, snap_hour_et=snap_hour_et, snap_minute_et=snap_minute_et)
        if chain.empty:
            print(f"  WARNING: empty chain")
            continue
        op = out_dir / f"{sym}_opra_daily.parquet"
        chain.to_parquet(op, index=False)
        print(f"  Wrote {op} ({len(chain)} rows, {chain["date"].min()}..{chain["date"].max()}, {chain["expiry"].nunique()} expiries)")
    print("\nDone.")


if __name__ == "__main__":
    main()
