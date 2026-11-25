#!/usr/bin/env python3
"""Cost-safe CLI to download OPRA definition + CBBO-1m + underlying bars from Databento,
build the canonical backtest chain, and write ``data/backtest/<SYMBOL>_opra_daily.parquet``.

Usage
-----
    python scripts/fetch_opra_databento.py \\
        --symbols DECK XBI \\
        --start 2026-05-20 --end 2026-11-20 \\
        --confirm

Without ``--confirm`` the script only prints the itemised cost table and exits \$0
without spending any Databento credit.  Pass ``--max-cost-usd 20.0`` (default) to
cap the total spend.

Cached ``.dbn.zst`` files under ``--raw-dir`` are reused at \$0.00 cost.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import pandas as pd

from src.data.opra_chain_builder import build_opra_backtest_chain

DEFAULT_RAW_DIR = "data/databento/opra"
DEFAULT_OUT_DIR = "data/backtest"
DEFAULT_RAW_DIR = "data/databento/opra"
DEFAULT_OUT_DIR = "data/backtest"


def _build_cache_path(
    raw_dir: Path, symbol: str, schema: str, dataset: str, start: str, end: str,
) -> Path:
    return raw_dir / f"{dataset}_{symbol}_{schema}_{start}_{end}.dbn.zst"


def _load_dbn(path: Path) -> pd.DataFrame:
    import databento as db
    store = db.DBNStore.from_file(str(path))
    return store.to_df()


def _underlying_bar_start(start: str, buffer_days: int = 45) -> str:
    dt = pd.Timestamp(start) - pd.Timedelta(days=buffer_days)
    return dt.strftime("%Y-%m-%d")


def _check_local_thetadata_bars(symbol: str, start: str, end: str) -> bool:
    import glob
    pattern = f"data/thetadata/bars/{symbol}_*.parquet"
    return len(sorted(glob.glob(pattern))) > 0
return len(sorted(glob.glob(pattern))) > 0


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download OPRA definition + CBBO-1m + underlying bars from Databento "
                    "and build canonical backtest chain.",
    )
    parser.add_argument("--symbols", nargs="+", required=True,
                        help="One or more root symbols (e.g. DECK XBI)")
    parser.add_argument("--start", required=True, help="ISO start date (e.g. 2026-05-20)")
    parser.add_argument("--end", required=True, help="ISO end date (e.g. 2026-11-20)")
    parser.add_argument("--max-cost-usd", type=float, default=20.0,
                        help="Safety cap on total estimated cost (default 20.0)")
    parser.add_argument("--confirm", action="store_true",
                        help="Actually download. Without this flag the script is a dry-run only.")
    parser.add_argument("--raw-dir", default=DEFAULT_RAW_DIR,
                        help="Directory for raw .dbn.zst cache files (default data/databento/opra)")
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR,
                        help="Directory for output Parquet files (default data/backtest)")
    parser.add_argument("--snap-time", default="15:45",
                        help="Snapshot time in ET (default 15:45)")
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    symbols = args.symbols
    start = args.start
    end = args.end

    api_key = os.environ.get("DATABENTO_API_KEY", "")
    if not api_key or "YOUR_DATABENTO" in api_key:
        print("ERROR: Set DATABENTO_API_KEY in your environment.", file=sys.stderr)
        sys.exit(1)

    import databento as db
    client = db.Historical(key=api_key)

    rows: list[dict] = []
    total_cost = 0.0

    for sym in symbols:
        def_path = _build_cache_path(raw_dir, sym, "definition", "OPRA.PILLAR", start, end)
        def_cached = def_path.exists()
        def_cost = 0.0 if def_cached else float(
            client.metadata.get_cost(
                dataset="OPRA.PILLAR", symbols=[sym], schema="definition",
                start=start, end=end, stype_in="raw_symbol",
            )
        )
        rows.append({"symbol": sym, "schema": "definition", "cached": def_cached, "cost_usd": def_cost})
        total_cost += def_cost

        cbbo_path = _build_cache_path(raw_dir, sym, "cbbo-1m", "OPRA.PILLAR", start, end)
        cbbo_cached = cbbo_path.exists()
        cbbo_cost = 0.0 if cbbo_cached else float(
            client.metadata.get_cost(
                dataset="OPRA.PILLAR", symbols=[sym], schema="cbbo-1m",
                start=start, end=end, stype_in="raw_symbol",
            )
        )
        rows.append({"symbol": sym, "schema": "cbbo-1m", "cached": cbbo_cached, "cost_usd": cbbo_cost})
        total_cost += cbbo_cost

        has_local = _check_local_thetadata_bars(sym, start, end)
        bar_start = _underlying_bar_start(start)
        eq_path = _build_cache_path(
            raw_dir.parent / "equities", sym, "ohlcv-1d", "EQUS.MINI", bar_start, end,
        )
        eq_cached = eq_path.exists()
        if has_local or eq_cached:
            eq_cost = 0.0
        else:
            eq_cost = float(
                client.metadata.get_cost(
                    dataset="EQUS.MINI", symbols=[sym], schema="ohlcv-1d",
                    start=bar_start, end=end, stype_in="raw_symbol",
                )
            )
        rows.append({
            "symbol": sym, "schema": "ohlcv-1d (underlying)", "cached": (has_local or eq_cached),
            "cost_usd": eq_cost,
        })
        total_cost += eq_cost

    print(f"{'Symbol':<8} {'Schema':<24} {'Cached':<8} {'Cost USD':>10}")
    print("-" * 52)
    for r in rows:
        print(f"{r['symbol']:<8} {r['schema']:<24} {str(r['cached']):<8} {r['cost_usd']:>10.4f}")
    print("-" * 52)
    print(f"{'TOTAL ESTIMATED COST USD':>42} {total_cost:>10.4f}")

    if not args.confirm:
        print("\nDry run complete.  Re-run with --confirm to download.")
        sys.exit(0)

sys.exit(1)

    for sym in symbols:
        print(f"\n--- {sym} ---")

        def_path = _build_cache_path(raw_dir, sym, "definition", "OPRA.PILLAR", start, end)
        if not def_path.exists():
            print(f"  Downloading OPRA.PILLAR definition for {sym} ...")
            data = client.timeseries.get_range(
                dataset="OPRA.PILLAR", symbols=[sym], schema="definition",
                start=start, end=end, stype_in="raw_symbol",
            )
            data.to_file(str(def_path))
            print(f"  -> {def_path}")
        else:
            print(f"  [cached] definition {def_path.name}")

        cbbo_path = _build_cache_path(raw_dir, sym, "cbbo-1m", "OPRA.PILLAR", start, end)
        if not cbbo_path.exists():
            print(f"  Downloading OPRA.PILLAR cbbo-1m for {sym} ...")
            data = client.timeseries.get_range(
                dataset="OPRA.PILLAR", symbols=[sym], schema="cbbo-1m",
                start=start, end=end, stype_in="raw_symbol",
            )
            data.to_file(str(cbbo_path))
            print(f"  -> {cbbo_path}")
        else:
            print(f"  [cached] cbbo-1m {cbbo_path.name}")

        has_local = _check_local_thetadata_bars(sym, start, end)
        eq_dir = raw_dir.parent / "equities"
        eq_dir.mkdir(parents=True, exist_ok=True)
        bar_start = _underlying_bar_start(start)
        eq_path = eq_dir / f"EQUS.MINI_{sym}_ohlcv-1d_{bar_start}_{end}.dbn.zst"

        if not has_local and not eq_path.exists():
            print(f"  Downloading EQUS.MINI ohlcv-1d for {sym} ...")
            data = client.timeseries.get_range(
                dataset="EQUS.MINI", symbols=[sym], schema="ohlcv-1d",
                start=bar_start, end=end, stype_in="raw_symbol",
            )
            data.to_file(str(eq_path))
            print(f"  -> {eq_path}")

    for sym in symbols:
        print(f"\n  Building chain for {sym} ...")

        def_path = _build_cache_path(raw_dir, sym, "definition", "OPRA.PILLAR", start, end)
        cbbo_path = _build_cache_path(raw_dir, sym, "cbbo-1m", "OPRA.PILLAR", start, end)

        if not def_path.exists() or not cbbo_path.exists():
            print(f"  SKIP {sym}: missing definition or cbbo-1m cache")
            continue

        def_df = _load_dbn(def_path)
        cbbo_df = _load_dbn(cbbo_path)

        has_local = _check_local_thetadata_bars(sym, start, end)
        bar_start = _underlying_bar_start(start)
        eq_path = raw_dir.parent / "equities" / f"EQUS.MINI_{sym}_ohlcv-1d_{bar_start}_{end}.dbn.zst"

        bars_df: pd.DataFrame | None = None
        if has_local:
            import glob
            pattern = f"data/thetadata/bars/{sym}_*.parquet"
            local_files = sorted(glob.glob(pattern))
            if local_files:
                bars_df = pd.read_parquet(local_files[-1])
                print(f"  Using local ThetaData bars: {local_files[-1]}")
        elif eq_path.exists():
            bars_df = _load_dbn(eq_path)
            print(f"  Using EQUS.MINI bars: {eq_path.name}")
        else:
            print(f"  SKIP {sym}: no underlying bars available")
            continue

        if bars_df is None or bars_df.empty:
            print(f"  SKIP {sym}: empty bars DataFrame")
            continue

        chain = build_opra_backtest_chain(def_df, cbbo_df, bars_df)

        if chain.empty:
            print(f"  WARNING: empty chain for {sym}")
            continue

        out_path = out_dir / f"{sym}_opra_daily.parquet"
        chain.to_parquet(out_path, index=False)
        print(f"  Wrote {out_path}  ({len(chain)} rows, "
              f"{chain['date'].min()}..{chain['date'].max()}, "
              f"{chain['expiry'].nunique()} expiries)")

    print("\nDone.")


if __name__ == "__main__":
    main()
    if total_cost > args.max_cost_usd:
        print(
            f"\nERROR: Total estimated cost ${total_cost:.4f} exceeds "
            f"--max-cost-usd=${args.max_cost_usd:.2f}.  Aborting.",
            file=sys.stderr,
        )
        sys.exit(1)