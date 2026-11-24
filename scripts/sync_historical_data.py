import os
import sys
from pathlib import Path
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.runtime.instruments import SPY_ID, SYNC_UNIVERSE, split_instrument_id  # noqa: E402

load_dotenv()

UNIVERSE = [split_instrument_id(instrument_id) for instrument_id in SYNC_UNIVERSE]


def main() -> None:
    # The catalog loaders are not in the repository yet, so they are imported only when the script runs
    from nautilus_trader.model import InstrumentId
    from src.data.catalog_loader import (
        fetch_databento_dbn,
        fetch_thetadata_eod_bars,
        ingest_databento_to_catalog,
        preview_databento_cost,
    )

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
                n_bars = ingest_databento_to_catalog(dbn_path, InstrumentId.from_str(SPY_ID))
                print(f"  Cached {dbn_path} & ingested {n_bars:,} 1-min bars into catalog/")
        except Exception as exc:
            print(f"  Databento sync skipped/error: {exc}")


if __name__ == "__main__":
    main()
