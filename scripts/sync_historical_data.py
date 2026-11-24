import os
import sys
from pathlib import Path
from dotenv import load_dotenv
from nautilus_trader.model import InstrumentId
from src.data.catalog_loader import (
    fetch_databento_dbn,
    fetch_thetadata_eod_bars,
    ingest_databento_to_catalog,
    preview_databento_cost,
)

load_dotenv()

UNIVERSE = [
    ("SPY", "ARCA"),
    ("QQQ", "NASDAQ"),
    ("IWM", "ARCA"),
    ("XBI", "ARCA"),
    ("GDX", "ARCA"),
    ("SMH", "NASDAQ"),
    ("TLT", "NASDAQ"),
    ("GLD", "ARCA"),
    ("VIXY", "ARCA"),
    ("VIXM", "ARCA"),
]


def main() -> None:
    print("=== 1. Syncing 1-Year EOD Universe Bars & Realized Volatility (ThetaData Free -> Catalog + Valkey) ===")
    for sym, venue in UNIVERSE:
        try:
            df, rv_stats = fetch_thetadata_eod_bars(
                symbol=sym,
                start_date="20251123",
                end_date="20261123",
                venue=venue,
            )
            print(
                f"  {sym:5s} ({venue:6s}) | {len(df):3d} bars | Close=${df['close'].iloc[-1]:7.2f} | "
                f"RV20={rv_stats['rv_20d']:6.2%} | RV60={rv_stats['rv_60d']:6.2%} | "
                f"RV_Pct={rv_stats['rv_percentile']:5.1f}%"
            )
        except Exception as exc:
            print(f"  {sym:5s} ERROR: {exc}")

    api_key = os.environ.get("DATABENTO_API_KEY", "")
    if api_key and "YOUR_DATABENTO" not in api_key:
        print("\n=== 2. Syncing High-Resolution Intraday Bars (Databento -> .dbn.zst + Catalog) ===")
        try:
            est_cost = preview_databento_cost(
                dataset="ARCX.PILLAR",
                symbols=["SPY", "QQQ", "IWM"],
                schema="ohlcv-1m",
                start="2026-10-01",
                end="2026-11-22",
            )
            print(f"  Databento 1-min OHLCV estimate (SPY, QQQ, IWM Oct-Nov 2026): ${est_cost:.4f}")
            if est_cost <= 2.50:
                dbn_path = fetch_databento_dbn(
                    dataset="ARCX.PILLAR",
                    symbols=["SPY"],
                    schema="ohlcv-1m",
                    start="2026-10-01",
                    end="2026-11-22",
                    max_cost_usd=2.50,
                )
                n_bars = ingest_databento_to_catalog(dbn_path, InstrumentId.from_str("SPY.ARCA"))
                print(f"  Cached {dbn_path} & ingested {n_bars:,} 1-min bars into catalog/")
        except Exception as exc:
            print(f"  Databento sync skipped/error: {exc}")


if __name__ == "__main__":
    main()
