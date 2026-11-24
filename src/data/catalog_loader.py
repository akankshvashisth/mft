import json
import os
from pathlib import Path
import time
import databento as db
import numpy as np
import pandas as pd
import redis
import requests
from nautilus_trader.adapters.databento import DatabentoDataLoader
from nautilus_trader.model import (
    Bar,
    BarType,
    Currency,
    Equity,
    InstrumentId,
    Price,
    Quantity,
    QuoteTick,
)
from nautilus_trader.persistence import ParquetDataCatalog


def compute_yang_zhang_rv(df: pd.DataFrame, window: int = 20, trading_days: int = 252) -> float:
    """Computes annualized Yang-Zhang realized volatility from OHLC bars."""
    if len(df) < window + 1:
        raise ValueError(f"Need at least {window + 1} bars for {window}d RV, got {len(df)}")

    sub = df.tail(window + 1).copy()
    o = np.log(sub["open"].astype(float).values)
    h = np.log(sub["high"].astype(float).values)
    l = np.log(sub["low"].astype(float).values)
    c = np.log(sub["close"].astype(float).values)

    r_o = o[1:] - c[:-1]
    r_c = c[1:] - o[1:]
    u = h[1:] - o[1:]
    d = l[1:] - o[1:]

    n = float(window)
    k = 0.34 / (1.34 + (n + 1.0) / (n - 1.0))

    sigma_o_sq = float(np.var(r_o, ddof=1))
    sigma_c_sq = float(np.var(r_c, ddof=1))
    sigma_rs_sq = float(np.mean(u * (u - r_o) + d * (d - r_o)))

    yz_var = max(sigma_o_sq + k * sigma_c_sq + (1.0 - k) * sigma_rs_sq, 1e-8)
    return float(np.sqrt(yz_var * trading_days))


def compute_rv_series_stats(df: pd.DataFrame, window: int = 20) -> dict:
    """Computes current 20d/60d Yang-Zhang RV and rolling percentile rank."""
    rv_20d = compute_yang_zhang_rv(df, window=window)
    rv_60d = compute_yang_zhang_rv(df, window=min(60, len(df) - 1))

    rolling_rvs = []
    for end_idx in range(window + 1, len(df) + 1):
        rolling_rvs.append(compute_yang_zhang_rv(df.iloc[:end_idx], window=window))
    rv_arr = np.asarray(rolling_rvs, dtype=np.float64)
    pct_rank = float(np.mean(rv_arr <= rv_20d) * 100.0) if len(rv_arr) > 0 else 50.0

    return {
        "rv_20d": round(rv_20d, 4),
        "rv_60d": round(rv_60d, 4),
        "rv_percentile": round(pct_rank, 1),
        "rv_min": round(float(np.min(rv_arr)), 4),
        "rv_max": round(float(np.max(rv_arr)), 4),
    }


def fetch_thetadata_eod_bars(
    symbol: str,
    start_date: str,  # YYYYMMDD
    end_date: str,    # YYYYMMDD
    venue: str = "ARCA",
    catalog_path: str = "catalog",
    raw_cache_dir: str = "data/thetadata/bars",
    valkey_url: str = "redis://127.0.0.1:6379/0",
) -> tuple[pd.DataFrame, dict]:
    """Pulls free EOD stock OHLCV from Theta Terminal v3, caches Parquet, computes RV, and writes to catalog."""
    base_url = os.environ.get("THETADATA_BASE_URL", "http://127.0.0.1:25503/v3")
    url = f"{base_url}/stock/history/eod"
    resp = requests.get(
        url,
        params={"symbol": symbol, "start_date": start_date, "end_date": end_date, "format": "json"},
        timeout=20,
    )
    resp.raise_for_status()
    payload = resp.json()

    if isinstance(payload, dict) and "response" in payload:
        df = pd.DataFrame(payload["response"])
    elif isinstance(payload, list):
        df = pd.DataFrame(payload)
    else:
        raise ValueError(f"Unexpected ThetaData v3 payload keys: {list(payload.keys())}")

    date_col = "date" if "date" in df.columns else ("created" if "created" in df.columns else df.columns[0])
    df["date_str"] = df[date_col].astype(str).str.replace("-", "", regex=False).str[:8]
    df["ts_dt"] = pd.to_datetime(df["date_str"], format="%Y%m%d", utc=True) + pd.Timedelta(hours=21)
    df = df.sort_values("ts_dt").drop_duplicates(subset=["date_str"]).reset_index(drop=True)

    # Save raw portable Parquet backup under data/thetadata/bars/
    cache_dir = Path(raw_cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    df.to_parquet(cache_dir / f"{symbol}_{start_date}_{end_date}.parquet", index=False)

    rv_stats = compute_rv_series_stats(df, window=20)

    # Write native Nautilus Equity + Daily Bars into ParquetDataCatalog
    iid = InstrumentId.from_str(f"{symbol}.{venue}")
    eq = Equity(
        instrument_id=iid,
        raw_symbol=iid.symbol,
        currency=Currency.from_str("USD"),
        price_precision=2,
        price_increment=Price.from_str("0.01"),
        lot_size=Quantity.from_int(1),
        ts_event=int(df["ts_dt"].iloc[0].value),
        ts_init=int(df["ts_dt"].iloc[0].value),
    )
    bar_type = BarType.from_str(f"{iid}-1-DAY-LAST-EXTERNAL")
    bars = []
    for _, r in df.iterrows():
        o_px = round(float(r["open"]), 2)
        h_px = round(float(r["high"]), 2)
        l_px = round(float(r["low"]), 2)
        c_px = round(float(r["close"]), 2)
        h_px = max(o_px, h_px, l_px, c_px)
        l_px = min(o_px, h_px, l_px, c_px)
        bars.append(
            Bar(
                bar_type=bar_type,
                open=Price.from_str(f"{o_px:.2f}"),
                high=Price.from_str(f"{h_px:.2f}"),
                low=Price.from_str(f"{l_px:.2f}"),
                close=Price.from_str(f"{c_px:.2f}"),
                volume=Quantity.from_int(max(int(r.get("volume", 1)), 1)),
                ts_event=int(r["ts_dt"].value),
                ts_init=int(r["ts_dt"].value),
            )
        )

    catalog = ParquetDataCatalog(catalog_path)
    catalog.write_instruments([eq])
    existing_dir = Path(catalog_path) / "data" / "bar" / str(bar_type)
    if existing_dir.exists():
        for pq in existing_dir.glob("*.parquet"):
            pq.unlink()
    catalog.write_bars(bars)

    if valkey_url:
        rdb = redis.from_url(valkey_url)
        rdb.set(
            f"mft:rv:{iid}",
            json.dumps(
                {
                    "underlying": str(iid),
                    **rv_stats,
                    "last_close": round(float(df["close"].iloc[-1]), 2),
                    "as_of_date": df["date_str"].iloc[-1],
                    "bars_loaded": len(bars),
                }
            ),
        )

    return df, rv_stats


def fetch_thetadata_eod_option_chain(
    symbol: str,
    expiration: str,  # YYYYMMDD or YYYY-MM-DD
    start_date: str,  # YYYYMMDD
    end_date: str,    # YYYYMMDD
    out_dir: str = "data/thetadata/options",
) -> Path:
    """Pulls free EOD option chain history for an expiry from Theta Terminal v3 and saves Parquet."""
    base_url = os.environ.get("THETADATA_BASE_URL", "http://127.0.0.1:25503/v3")
    exp_clean = expiration.replace("-", "")
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    parquet_file = out_path / f"{symbol}_exp{exp_clean}_{start_date}_{end_date}.parquet"
    if parquet_file.exists():
        return parquet_file

    url = f"{base_url}/option/history/eod"
    resp = requests.get(
        url,
        params={
            "symbol": symbol,
            "expiration": exp_clean,
            "start_date": start_date,
            "end_date": end_date,
            "format": "json",
        },
        timeout=30,
    )
    resp.raise_for_status()
    payload = resp.json()
    rows = payload["response"] if isinstance(payload, dict) and "response" in payload else payload
    df = pd.DataFrame(rows)
    df.to_parquet(parquet_file, index=False)
    time.sleep(0.25)  # Respect ThetaData Free Tier 1-concurrent-request limit
    return parquet_file


def preview_databento_cost(
    dataset: str,
    symbols: list[str],
    schema: str,
    start: str,
    end: str,
    stype_in: str = "raw_symbol",
) -> float:
    """Free metadata check returning exact USD cost before downloading from Databento."""
    api_key = os.environ.get("DATABENTO_API_KEY")
    if not api_key or "YOUR_DATABENTO" in api_key:
        raise RuntimeError("Set DATABENTO_API_KEY in .env before calling Databento.")
    client = db.Historical(key=api_key)
    return float(
        client.metadata.get_cost(
            dataset=dataset,
            symbols=symbols,
            schema=schema,
            start=start,
            end=end,
            stype_in=stype_in,
        )
    )


def fetch_databento_dbn(
    dataset: str,
    symbols: list[str],
    schema: str,
    start: str,
    end: str,
    out_dir: str = "data/databento",
    max_cost_usd: float = 5.0,
    stype_in: str = "raw_symbol",
) -> Path:
    """Checks free cost estimate first, caches .dbn.zst locally, and never re-downloads existing files."""
    out_path_dir = Path(out_dir)
    out_path_dir.mkdir(parents=True, exist_ok=True)
    safe_sym = "_".join(s.replace(".", "-") for s in symbols)
    dbn_path = out_path_dir / f"{dataset}_{safe_sym}_{schema}_{start[:10]}_{end[:10]}.dbn.zst"

    if dbn_path.exists():
        return dbn_path

    cost = preview_databento_cost(dataset, symbols, schema, start, end, stype_in=stype_in)
    if cost > max_cost_usd:
        raise ValueError(
            f"Estimated Databento cost ${cost:.2f} exceeds max_cost_usd=${max_cost_usd:.2f}."
        )

    api_key = os.environ["DATABENTO_API_KEY"]
    client = db.Historical(key=api_key)
    print(f"Downloading {dbn_path.name} (cost: ${cost:.4f})...")
    data = client.timeseries.get_range(
        dataset=dataset,
        symbols=symbols,
        schema=schema,
        start=start,
        end=end,
        stype_in=stype_in,
    )
    data.to_file(str(dbn_path))
    return dbn_path


def ingest_databento_to_catalog(
    dbn_path: str | Path,
    instrument_id: InstrumentId,
    catalog_path: str = "catalog",
) -> int:
    """Ingests a local Databento .dbn.zst file directly into Nautilus ParquetDataCatalog."""
    pub_json = Path("data/databento/publishers.json")
    loader = DatabentoDataLoader(pub_json) if pub_json.exists() else DatabentoDataLoader()
    fp = Path(dbn_path)
    name = fp.name.lower()

    catalog = ParquetDataCatalog(catalog_path)
    if "ohlcv" in name or "bars" in name:
        records = loader.load_bars(filepath=fp, instrument_id=instrument_id, price_precision=3)
        if records:
            catalog.write_bars(records)
        return len(records)
    elif "bbo" in name or "mbp-1" in name or "quote" in name:
        records = loader.load_quotes(filepath=fp)
        if records:
            catalog.write_quote_ticks(records)
        return len(records)
    else:
        records = loader.load_bars(filepath=fp, instrument_id=instrument_id, price_precision=3)
        if records:
            catalog.write_bars(records)
        return len(records)
