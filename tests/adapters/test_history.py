from types import SimpleNamespace
from unittest import mock

import pandas as pd
import pytest

from src.adapters.history import (
    DailyHistoryLoader,
    bars_to_frame,
    load_thetadata_parquet,
    thetadata_frame,
)
from src.common.constants import NS_PER_DAY


def bar(ts, close):
    return SimpleNamespace(ts_event=ts, open=close - 1, high=close + 1, low=close - 2, close=close, volume=1_000)


def catalog(bars=(), error=None):
    """Nautilus ParquetDataCatalog double: bars() returns `bars` or raises `error`."""
    cat = mock.Mock(spec=["bars"])
    cat.bars.return_value = list(bars)
    cat.bars.side_effect = error
    return cat


def test_bars_to_frame():
    df = bars_to_frame([bar(1, 100.0), bar(2, 101.0)])
    assert list(df.columns) == ["ts_ns", "open", "high", "low", "close", "volume"]
    assert df["close"].tolist() == [100.0, 101.0]
    assert df["ts_ns"].tolist() == [1, 2]


def test_thetadata_frame_with_and_without_timestamps():
    raw = pd.DataFrame({"open": [1, 2], "high": [2, 3], "low": [0.5, 1.5], "close": [1.5, 2.5]})
    df = thetadata_frame(raw)
    assert df["ts_ns"].tolist() == [0, NS_PER_DAY]
    assert df["volume"].tolist() == [1.0, 1.0]

    stamped = thetadata_frame(raw.assign(ts_dt=["2026-11-23", "2026-11-24"], volume=[10, 20]))
    assert stamped["ts_ns"].tolist() == [
        int(pd.Timestamp("2026-11-23").value),
        int(pd.Timestamp("2026-11-24").value),
    ]
    assert stamped["volume"].tolist() == [10.0, 20.0]


def test_loader_prefers_catalog_daily_bars(tmp_path):
    cat = catalog(bars=[bar(1, 100.0)])
    loader = DailyHistoryLoader("catalog", thetadata_dir=tmp_path, catalog_factory=lambda path: cat)

    df = loader("SPY.ARCA")

    cat.bars.assert_called_once_with(bar_types=["SPY.ARCA-1-DAY-LAST-EXTERNAL"])
    assert df["close"].tolist() == [100.0]


@pytest.mark.parametrize("cat", [catalog(), catalog(error=RuntimeError("corrupt"))])
def test_loader_falls_back_to_thetadata_files(tmp_path, cat):
    loader = DailyHistoryLoader("catalog", thetadata_dir=tmp_path, catalog_factory=lambda path: cat)
    assert loader("SPY.ARCA") is None  # No export either


def test_unopenable_catalog_is_logged_and_skipped(tmp_path, recording_log):
    broken = mock.Mock(side_effect=OSError("missing"))

    loader = DailyHistoryLoader("catalog", thetadata_dir=tmp_path, log=recording_log, catalog_factory=broken)
    assert loader.catalog is None
    recording_log.warning.assert_called_once_with("RealizedVolActor could not open catalog: missing")
    assert loader("SPY.ARCA") is None


def test_thetadata_parquet_export_is_used(tmp_path):
    pytest.importorskip("pyarrow")
    raw = pd.DataFrame({"open": [1.0], "high": [2.0], "low": [0.5], "close": [1.5]})
    raw.to_parquet(tmp_path / "SPY_2025.parquet")
    raw.assign(close=[9.0]).to_parquet(tmp_path / "SPY_2026.parquet")

    assert load_thetadata_parquet("SPY", tmp_path)["close"].tolist() == [9.0]  # Latest file name wins
    loader = DailyHistoryLoader("catalog", thetadata_dir=tmp_path, catalog_factory=lambda path: catalog())
    assert loader("SPY.ARCA")["close"].tolist() == [9.0]
    assert load_thetadata_parquet("QQQ", tmp_path) is None
