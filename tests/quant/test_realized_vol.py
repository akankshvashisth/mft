import math
from datetime import date

import numpy as np
import pandas as pd
import pytest

from src.quant.realized_vol import (
    avg_pairwise_corr,
    bipower_jump_ratio,
    daily_log_returns,
    percentile_rank,
    trailing_rolling_rv,
    yang_zhang_rv,
)
from tests.factories import daily_ohlc_frame, utc_ns


def _closes_frame(closes):
    closes = np.asarray(closes, dtype=float)
    return pd.DataFrame({"open": closes, "high": closes, "low": closes, "close": closes})


# --- Yang-Zhang -----------------------------------------------------------------------------------


def test_constant_prices_hit_the_variance_floor():
    assert yang_zhang_rv(_closes_frame([100.0] * 30), 20) == pytest.approx(math.sqrt(1e-10 * 252))


def test_rv_is_invariant_to_price_scale():
    df = daily_ohlc_frame(n=40, seed=11)
    scaled = df.copy()
    scaled[["open", "high", "low", "close"]] *= 10.0
    assert yang_zhang_rv(scaled, 20) == pytest.approx(yang_zhang_rv(df, 20), rel=1e-9)


def test_rv_tracks_the_simulated_volatility():
    df = daily_ohlc_frame(n=260, seed=5, daily_vol=0.01)
    rv = yang_zhang_rv(df, 250)
    assert 0.08 < rv < 0.32  # ~16% annualised input; generous bounds for range noise


def test_short_history_falls_back_to_close_to_close_vol():
    closes = [100.0, 101.0, 99.0]
    rets = np.diff(np.log(closes))
    assert yang_zhang_rv(_closes_frame(closes), 20) == pytest.approx(np.std(rets, ddof=1) * math.sqrt(252))
    assert yang_zhang_rv(_closes_frame([100.0, 101.0]), 20) == 0.0  # One return: no dispersion estimate


# --- Bipower jump share ---------------------------------------------------------------------------


def test_jump_ratio_is_zero_without_movement_or_history():
    assert bipower_jump_ratio(_closes_frame([100.0] * 30), 20) == 0.0
    assert bipower_jump_ratio(_closes_frame([100.0, 110.0] * 5), 20) == 0.0  # Fewer than window + 1 returns


def test_single_isolated_jump_is_all_jump_variance():
    closes = [100.0] * 15 + [110.0] * 10
    assert bipower_jump_ratio(_closes_frame(closes), 20) == pytest.approx(1.0)


def test_steady_diffusion_is_clipped_to_zero_jump_share():
    closes = 100.0 * np.cumprod([1.0] + [1.01, 0.99] * 15)  # Equal-size moves: BV exceeds RV
    assert bipower_jump_ratio(_closes_frame(closes), 20) == 0.0


# --- Daily returns and correlation ---------------------------------------------------------------


def test_daily_returns_use_new_york_dates_and_last_close_per_date():
    df = pd.DataFrame(
        {
            "ts_ns": [
                utc_ns(2026, 11, 23, 21),  # Mon 16:00 NY
                utc_ns(2026, 11, 24, 21),  # Tue 16:00 NY (superseded)
                utc_ns(2026, 11, 25, 2),  # Tue 21:00 NY, already Wednesday in UTC
                utc_ns(2026, 11, 25, 21),  # Wed 16:00 NY
            ],
            "close": [100.0, 999.0, 101.0, 102.0],
        }
    )
    rets = daily_log_returns(df)
    assert list(rets.index) == [date(2026, 11, 24), date(2026, 11, 25)]
    np.testing.assert_allclose(rets.to_numpy(), [math.log(1.01), math.log(102.0 / 101.0)])


def _returns(values, start="2026-01-01"):
    return pd.Series(values, index=pd.date_range(start, periods=len(values), freq="D"))


def test_average_pairwise_correlation_of_linked_series():
    a = np.random.default_rng(3).normal(0.0, 0.01, 30)
    assert avg_pairwise_corr({"A": _returns(a), "B": _returns(2.0 * a + 0.001)}) == pytest.approx(1.0)
    assert avg_pairwise_corr({"A": _returns(a), "C": _returns(-a)}) == pytest.approx(-1.0)
    trio = {"A": _returns(a), "B": _returns(2.0 * a), "C": _returns(-a)}
    assert avg_pairwise_corr(trio) == pytest.approx(-1.0 / 3.0)


def test_correlation_aligns_on_dates_not_positions():
    a = np.random.default_rng(4).normal(0.0, 0.01, 30)
    full = _returns(a)
    late = full.iloc[5:] * 3.0  # Same dates as the last 25 rows of `full`; positional pairing would decorrelate
    assert avg_pairwise_corr({"A": full, "B": late}) == pytest.approx(1.0)


def test_correlation_needs_two_series_and_enough_overlap():
    a = _returns(np.random.default_rng(5).normal(0.0, 0.01, 30))
    assert avg_pairwise_corr({"A": a}) == 0.0
    short = _returns([0.01, -0.02, 0.005, 0.0])
    assert avg_pairwise_corr({"A": short, "B": short * 2.0}) == 0.0  # 4 rows < min_rows


# --- Percentiles and rolling RV ------------------------------------------------------------------


def test_percentile_rank_options():
    assert percentile_rank([1.0, 2.0, 3.0, 4.0], 2.5) == 50.0
    assert percentile_rank([1.0, 2.0, 3.0, 4.0], 4.0) == 100.0
    assert percentile_rank([1.0, 2.0, 3.0, 4.0], 0.0) == 0.0
    assert percentile_rank([], 1.0) == 50.0
    assert percentile_rank([1.0, 2.0, 3.0, 4.0], 2.0, min_count=5, default=42.0) == 42.0
    assert percentile_rank([1.0, 2.0, 3.0], 1.0, decimals=2) == 33.33
    assert percentile_rank([1.0, np.nan, 3.0], 2.0) == 50.0  # NaNs are ignored, not counted
    assert percentile_rank([np.nan, np.nan], 2.0, default=7.0) == 7.0


def test_trailing_rolling_rv_windows():
    df = daily_ohlc_frame(n=30, seed=9)

    values = trailing_rolling_rv(df, window=20, ann=252)
    assert len(values) == 10  # End points 21..30
    assert values[0] == pytest.approx(yang_zhang_rv(df.iloc[:21], 20, 252))
    assert values[-1] == pytest.approx(yang_zhang_rv(df, 20, 252))

    assert len(trailing_rolling_rv(df, window=20, ann=252, lookback=5)) == 5
