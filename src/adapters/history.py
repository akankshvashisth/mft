"""Daily OHLC history for warm starts: Nautilus catalog bars first, ThetaData parquet files second."""
from __future__ import annotations

import logging
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.common.constants import NS_PER_DAY
from src.common.log import LogLike
from src.common.market import symbol_of

DEFAULT_THETADATA_DIR = Path("data/thetadata/bars")
_LOG = logging.getLogger(__name__)


def daily_bar_type_str(instrument_id: Any) -> str:
    return f"{instrument_id}-1-DAY-LAST-EXTERNAL"


def load_catalog_daily_bars(catalog: Any, instrument_id: Any) -> list[Any]:
    """Daily bars only: the catalog can also hold 1-minute bars for the same instrument."""
    return list(catalog.bars(bar_types=[daily_bar_type_str(instrument_id)]) or [])


def bars_to_frame(bars: Iterable[Any]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "ts_ns": int(b.ts_event),
                "open": float(b.open),
                "high": float(b.high),
                "low": float(b.low),
                "close": float(b.close),
                "volume": float(b.volume),
            }
            for b in bars
        ]
    )


def thetadata_frame(raw: pd.DataFrame) -> pd.DataFrame:
    """OHLCV frame from a ThetaData bar export; rows are spaced one day apart when it has no timestamps."""
    if "ts_dt" in raw.columns:
        ts_vals = pd.to_datetime(raw["ts_dt"]).dt.as_unit("ns").astype("int64")
    else:
        ts_vals = np.arange(len(raw), dtype=np.int64) * NS_PER_DAY
    return pd.DataFrame(
        {
            "ts_ns": ts_vals,
            "open": raw["open"].astype(float),
            "high": raw["high"].astype(float),
            "low": raw["low"].astype(float),
            "close": raw["close"].astype(float),
            "volume": raw.get("volume", pd.Series(1.0, index=raw.index)).astype(float),
        }
    )


def load_thetadata_parquet(symbol: str, base_dir: str | Path = DEFAULT_THETADATA_DIR) -> pd.DataFrame | None:
    """Latest `<symbol>_*.parquet` export (by file name), or None."""
    matches = sorted(Path(base_dir).glob(f"{symbol}_*.parquet"))
    if not matches:
        return None
    return thetadata_frame(pd.read_parquet(matches[-1]))


def open_catalog(catalog_path: str) -> Any:
    from nautilus_trader.persistence import ParquetDataCatalog  # Lazy: keeps this module Nautilus-free

    return ParquetDataCatalog(catalog_path)


class DailyHistoryLoader:
    """Callable instrument_id -> daily OHLCV frame (or None) for RealizedVolService.warm_start."""

    def __init__(
        self,
        catalog_path: str,
        thetadata_dir: str | Path = DEFAULT_THETADATA_DIR,
        log: LogLike | None = None,
        catalog_factory: Callable[[str], Any] = open_catalog,
    ) -> None:
        self.thetadata_dir = thetadata_dir
        self.catalog = None
        try:
            self.catalog = catalog_factory(catalog_path)
        except Exception as exc:
            (log or _LOG).warning(f"RealizedVolActor could not open catalog: {exc}")

    def __call__(self, instrument_id: str) -> pd.DataFrame | None:
        df = None
        if self.catalog is not None:
            try:
                bars = load_catalog_daily_bars(self.catalog, instrument_id)
                if bars:
                    df = bars_to_frame(bars)
            except Exception:
                df = None
        if df is None or df.empty:
            df = load_thetadata_parquet(symbol_of(instrument_id), self.thetadata_dir)
        return df
