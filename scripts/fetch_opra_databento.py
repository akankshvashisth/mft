#!/usr/bin/env python3
"""Cost-safe CLI to download OPRA definition + 15-min daily NBBO windows + underlying bars from Databento."""
from __future__ import annotations

import argparse
import glob
import os
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.opra_chain_builder import build_opra_backtest_chain

DEFAULT_RAW_DIR = "data/databento/opra"
DEFAULT_OUT_DIR = "data/backtest"
MAX_RETRIES = 3


def _build_cache_path(raw_dir: Path, sym: str, schema: str, dataset: str, start: str, end: str) -> Path:
    return raw_dir / f"{dataset}_{sym}_{schema}_{start}_{end}.dbn.zst"


def _day_cbbo_path(raw_dir: Path, sym: str, day_iso: str) -> Path:
    d = raw_dir / "daily" / sym
    d.mkdir(parents=True, exist_ok=True)
    return d / f"OPRA.PILLAR_{sym}_cbbo-1m_{day_iso}.dbn.zst"


def _load_dbn(path: Path) -> pd.DataFrame:
    import databento as db
    return db.DBNStore.from_file(str(path)).to_df()


def _underlying_bar_start(start: str, buffer_days: int = 45) -> str:
    return (pd.Timestamp(start) - pd.Timedelta(days=buffer_days)).strftime("%Y-%m-%d")


def _check_local_thetadata_bars(symbol: str, start: str, end: str) -> bool:
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


def _opra_symbols(raw_list: list[str]) -> list[str]:
    return [f"{s}.OPT" for s in raw_list]


def _parse_snap_time(st: str) -> tuple[int, int]:
    parts = st.strip().split(":")
    return int(parts[0]), int(parts[1])


def _day_window_utc(day_iso: str, snap_hour_et: int, snap_minute_et: int, window_minutes: int = 15) -> tuple[str, str]:
    """Convert [snap - window, snap + 1 min] in America/New_York to ISO UTC strings (DST-aware)."""
    end_ny = pd.Timestamp(f"{day_iso} {snap_hour_et:02d}:{snap_minute_et:02d}:00", tz="America/New_York") + pd.Timedelta(minutes=1)
    start_ny = end_ny - pd.Timedelta(minutes=window_minutes + 1)
    return start_ny.tz_convert("UTC").isoformat(), end_ny.tz_convert("UTC").isoformat()


def _trading_days_from_bars(bars_df: pd.DataFrame, start: str, end: str) -> list[str]:
    """Extract sorted ISO trading dates within [start, end) from underlying daily bars."""
    if "date" in bars_df.columns:
        dts = pd.to_datetime(bars_df["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    elif isinstance(bars_df.index, pd.DatetimeIndex):
        idx = bars_df.index
        if idx.tz is None:
            dts = idx.strftime("%Y-%m-%d")
        else:
            dts = idx.tz_convert("UTC").strftime("%Y-%m-%d")
    elif "ts_event" in bars_df.columns:
        dts = pd.to_datetime(bars_df["ts_event"].astype("int64"), unit="ns", utc=True).dt.strftime("%Y-%m-%d")
    else:
        return []
    days = sorted({d for d in dts.dropna() if start <= d < end})
    return days


def _get_cost_with_retry(client, **kwargs) -> float:
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            return float(client.metadata.get_cost(**kwargs))
        except Exception as exc:
            if attempt == MAX_RETRIES:
                raise
            wait = 2.0 * attempt
            print(f"  [retry {attempt}/{MAX_RETRIES}] get_cost failed ({exc}); sleeping {wait:.0f}s...")
            time.sleep(wait)
    return 0.0


def _download_stream_to_file(client, dest: Path, **kwargs) -> None:
    """Stream directly to disk via .part file + atomic rename with retries."""
    tmp = dest.with_suffix(dest.suffix + ".part")
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            if tmp.exists():
                tmp.unlink()
            client.timeseries.get_range(path=str(tmp), **kwargs)
            tmp.rename(dest)
            return
        except Exception as exc:
            if tmp.exists():
                tmp.unlink()
            if attempt == MAX_RETRIES:
                raise
            wait = 3.0 * attempt
            print(f"  [retry {attempt}/{MAX_RETRIES}] stream failed ({exc}); sleeping {wait:.0f}s...")
            time.sleep(wait)


def _load_or_fetch_bars(client, raw_dir: Path, sym: str, start: str, end: str, allow_download: bool) -> tuple[pd.DataFrame | None, float]:
    has_local = _check_local_thetadata_bars(sym, start, end)
    bar_start = _underlying_bar_start(start)
    eq_dir = raw_dir.parent / "equities"
    eq_dir.mkdir(parents=True, exist_ok=True)
    ep = _build_cache_path(eq_dir, sym, "ohlcv-1d", "EQUS.MINI", bar_start, end)

    if has_local:
        files = sorted(glob.glob(f"data/thetadata/bars/{sym}_*.parquet"))
        return pd.read_parquet(files[-1]), 0.0
    if ep.exists():
        return _load_dbn(ep), 0.0
    cost = _get_cost_with_retry(
        client, dataset="EQUS.MINI", symbols=[sym], schema="ohlcv-1d",
        start=bar_start, end=end, stype_in="raw_symbol",
    )
    if allow_download:
        _download_stream_to_file(
            client, ep, dataset="EQUS.MINI", symbols=[sym], schema="ohlcv-1d",
            start=bar_start, end=end, stype_in="raw_symbol",
        )
        return _load_dbn(ep), cost
    return None, cost


def main() -> None:
    p = argparse.ArgumentParser(description="Download OPRA 15-min daily NBBO snapshots + underlying bars from Databento.")
    p.add_argument("--symbols", nargs="+", required=True)
    p.add_argument("--start", required=True)
    p.add_argument("--end", required=True)
    p.add_argument("--max-cost-usd", type=float, default=20.0)
    p.add_argument("--confirm", action="store_true")
    p.add_argument("--raw-dir", default=DEFAULT_RAW_DIR)
    p.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    p.add_argument("--snap-time", default="15:45")
    p.add_argument("--window-minutes", type=int, default=15)
    args = p.parse_args()

    raw_dir = Path(args.raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    symbols, start, end = args.symbols, args.start, args.end
    snap_hour_et, snap_minute_et = _parse_snap_time(args.snap_time)
    win_min = args.window_minutes

    api_key = os.environ.get("DATABENTO_API_KEY", "")
    if not api_key or "YOUR_DATABENTO" in api_key:
        print("ERROR: Set DATABENTO_API_KEY.", file=sys.stderr)
        sys.exit(1)

    import databento as db
    client = db.Historical(key=api_key)

    # Approximate business days for dry-run cost estimate (exact trading days come from bars once downloaded)
    est_days = [d.strftime("%Y-%m-%d") for d in pd.bdate_range(start, end, inclusive="left")]
    sample_day = est_days[len(est_days) // 2] if est_days else start
    s_utc, e_utc = _day_window_utc(sample_day, snap_hour_et, snap_minute_et, win_min)

    rows, total_cost = [], 0.0
    opra_syms = _opra_symbols(symbols)

    for sym, psym in zip(symbols, opra_syms):
        dp = _build_cache_path(raw_dir, sym, "definition", "OPRA.PILLAR", start, end)
        dc = dp.exists()
        dcost = 0.0 if dc else _get_cost_with_retry(
            client, dataset="OPRA.PILLAR", symbols=[psym], schema="definition",
            start=start, end=end, stype_in="parent",
        )
        rows.append({"symbol": sym, "schema": "definition", "cached": str(dc), "cost_usd": dcost})
        total_cost += dcost

        cached_days = sum(1 for d in est_days if _day_cbbo_path(raw_dir, sym, d).exists())
        uncached_days = max(0, len(est_days) - cached_days)
        if uncached_days == 0:
            cbbo_est = 0.0
        else:
            one_day_cost = _get_cost_with_retry(
                client, dataset="OPRA.PILLAR", symbols=[psym], schema="cbbo-1m",
                start=s_utc, end=e_utc, stype_in="parent",
            )
            cbbo_est = one_day_cost * uncached_days
        rows.append({
            "symbol": sym, "schema": f"cbbo-1m ({win_min}m/day x{len(est_days)}d)",
            "cached": f"{cached_days}/{len(est_days)}", "cost_usd": cbbo_est,
        })
        total_cost += cbbo_est

        _, ecost = _load_or_fetch_bars(client, raw_dir, sym, start, end, allow_download=False)
        rows.append({"symbol": sym, "schema": "ohlcv-1d (underlying)", "cached": str(ecost == 0.0), "cost_usd": ecost})
        total_cost += ecost

    print(f"{'Symbol':<8} {'Schema':<28} {'Cached':<10} {'Est. USD':>10}")
    print("-" * 58)
    for r in rows:
        print(f"{r['symbol']:<8} {r['schema']:<28} {r['cached']:<10} {r['cost_usd']:>10.4f}")
    print("-" * 58)
    print(f"{'TOTAL ESTIMATED USD':>48} {total_cost:>10.4f}")

    if not args.confirm:
        print("\nDry run. Re-run with --confirm.")
        sys.exit(0)
    if total_cost > args.max_cost_usd:
        print(f"\nERROR: ${total_cost:.4f} exceeds --max-cost-usd=${args.max_cost_usd:.2f}.", file=sys.stderr)
        sys.exit(1)

    for sym, psym in zip(symbols, opra_syms):
        print(f"\n--- {sym} ---")
        # 1. Underlying bars first -> gives exact NYSE trading days (skips holidays automatically)
        bars_df, _ = _load_or_fetch_bars(client, raw_dir, sym, start, end, allow_download=True)
        if bars_df is None or bars_df.empty:
            print(f"  SKIP {sym}: no underlying bars")
            continue
        trading_days = _trading_days_from_bars(bars_df, start, end)
        print(f"  Trading days in [{start}, {end}): {len(trading_days)}")

        # 2. Option definitions
        dp = _build_cache_path(raw_dir, sym, "definition", "OPRA.PILLAR", start, end)
        if not dp.exists():
            print("  Downloading definition...")
            _download_stream_to_file(
                client, dp, dataset="OPRA.PILLAR", symbols=[psym], schema="definition",
                start=start, end=end, stype_in="parent",
            )
        else:
            print(f"  [cached] {dp.name}")

        # 3. 15-min daily NBBO snapshots (fast, ~1-3 MB per day, resumable)
        cbbo_parts: list[pd.DataFrame] = []
        downloaded = 0
        for i, day_iso in enumerate(trading_days, 1):
            cp = _day_cbbo_path(raw_dir, sym, day_iso)
            if not cp.exists():
                ds, de = _day_window_utc(day_iso, snap_hour_et, snap_minute_et, win_min)
                try:
                    _download_stream_to_file(
                        client, cp, dataset="OPRA.PILLAR", symbols=[psym], schema="cbbo-1m",
                        start=ds, end=de, stype_in="parent",
                    )
                    downloaded += 1
                except Exception as exc:
                    print(f"  [warn] {sym} {day_iso}: skipped ({exc})")
                    continue
            try:
                day_df = _load_dbn(cp)
                if not day_df.empty:
                    cbbo_parts.append(day_df)
            except Exception as exc:
                print(f"  [warn] {sym} {day_iso}: failed reading cache ({exc})")
            if i % 25 == 0 or i == len(trading_days):
                print(f"  Progress: {i}/{len(trading_days)} days ({downloaded} newly downloaded)")

        if not cbbo_parts:
            print(f"  SKIP {sym}: no cbbo-1m rows loaded")
            continue

        print(f"  Building canonical chain for {sym}...")
        def_df = _load_dbn(dp)
        cbbo_df = pd.concat(cbbo_parts, ignore_index=True)
        chain = build_opra_backtest_chain(
            def_df, cbbo_df, bars_df, snap_hour_et=snap_hour_et, snap_minute_et=snap_minute_et,
        )
        if chain.empty:
            print(f"  WARNING: empty chain for {sym}")
            continue

        op = out_dir / f"{sym}_opra_daily.parquet"
        chain.to_parquet(op, index=False)
        vrp_mean = float((chain.groupby("date")["atm_iv"].first() - chain.groupby("date")["rv_20d"].first()).mean())
        print(
            f"  Wrote {op} ({len(chain):,} rows, {chain['date'].min()}..{chain['date'].max()}, "
            f"{chain['expiry'].nunique()} expiries, mean VRP={vrp_mean:+.2%})"
        )
    print("\nDone.")


if __name__ == "__main__":
    main()
