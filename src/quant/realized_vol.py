"""Physical (P-measure) realized volatility, jump share, correlation and percentile helpers."""
from __future__ import annotations

from collections.abc import Iterable, Mapping

import numpy as np
import pandas as pd
from scipy.stats import percentileofscore

from src.common.constants import NY_TZ_NAME, TRADING_DAYS_PER_YEAR


def yang_zhang_rv(df: pd.DataFrame, window: int, ann: int = TRADING_DAYS_PER_YEAR) -> float:
    """Annualized Yang-Zhang realized volatility over the last `window` OHLC bars.

    With fewer than window + 1 bars it falls back to close-to-close volatility of what is available.
    """
    if len(df) < window + 1:
        rets = np.log(df["close"] / df["close"].shift(1)).dropna()
        return float(rets.std(ddof=1) * np.sqrt(ann)) if len(rets) > 1 else 0.0

    sub = df.iloc[-(window + 1):].copy()
    prev_close = sub["close"].shift(1)
    log_oc = np.log(sub["open"] / prev_close).iloc[1:]
    log_co = np.log(sub["close"] / sub["open"]).iloc[1:]
    log_ho = np.log(sub["high"] / sub["open"]).iloc[1:]
    log_lo = np.log(sub["low"] / sub["open"]).iloc[1:]
    log_hc = np.log(sub["high"] / sub["close"]).iloc[1:]
    log_lc = np.log(sub["low"] / sub["close"]).iloc[1:]

    rs = (log_ho * log_hc + log_lo * log_lc).mean()
    var_o = log_oc.var(ddof=1)
    var_c = log_co.var(ddof=1)
    k = 0.34 / (1.34 + (window + 1) / max(window - 1, 1))
    yz_var = max(float(var_o + k * var_c + (1.0 - k) * rs), 1e-10)
    return float(np.sqrt(yz_var * ann))


def bipower_jump_ratio(df: pd.DataFrame, window: int = 20) -> float:
    """Fraction of recent variance attributable to discrete jumps: (RV - BV) / RV, clipped to [0, 1]."""
    rets = np.log(df["close"] / df["close"].shift(1)).dropna().to_numpy(dtype=np.float64)
    if len(rets) < window + 1:
        return 0.0
    r = rets[-window:]
    rv = float(np.sum(r ** 2))
    if rv <= 1e-12:
        return 0.0
    # mu_1 = sqrt(2/pi) -> (pi/2) * sum(|r_t| * |r_{t-1}|)
    bv = float((np.pi / 2.0) * np.sum(np.abs(r[1:]) * np.abs(r[:-1])))
    return float(np.clip((rv - bv) / rv, 0.0, 1.0))


def daily_log_returns(df: pd.DataFrame) -> pd.Series:
    """Close-to-close log returns indexed by New York trading date (one row per date, last close wins)."""
    dates = (
        pd.to_datetime(df["ts_ns"].astype("int64"), unit="ns", utc=True)
        .dt.tz_convert(NY_TZ_NAME)
        .dt.date
    )
    closes = pd.Series(df["close"].astype(float).to_numpy(), index=dates.to_numpy())
    closes = closes[~closes.index.duplicated(keep="last")]
    return np.log(closes / closes.shift(1)).dropna()


def avg_pairwise_corr(
    returns_by_symbol: Mapping[str, pd.Series], window: int = 20, min_rows: int = 5
) -> float:
    """Mean off-diagonal correlation of returns aligned on their index (trading date), last `window` rows."""
    if len(returns_by_symbol) < 2:
        return 0.0
    mat = pd.DataFrame(dict(returns_by_symbol)).sort_index().dropna().iloc[-window:]
    if len(mat) < min_rows:
        return 0.0
    corr = mat.corr().to_numpy()
    n = corr.shape[0]
    off_diag = corr[np.triu_indices(n, k=1)]
    finite = off_diag[np.isfinite(off_diag)]
    return float(np.mean(finite)) if len(finite) > 0 else 0.0


def percentile_rank(
    history: Iterable[float],
    value: float,
    *,
    default: float = 50.0,
    min_count: int = 1,
    decimals: int | None = None,
) -> float:
    """Share (in %) of history at or below value (scipy's "weak" percentile of score); NaNs are ignored.

    default when history has fewer than min_count usable points.
    """
    arr = np.asarray(list(history), dtype=np.float64)
    arr = arr[~np.isnan(arr)]
    if arr.size < max(min_count, 1):
        return default
    pct = float(percentileofscore(arr, value, kind="weak"))
    return float(round(pct, decimals)) if decimals is not None else pct


def trailing_rolling_rv(
    df: pd.DataFrame,
    window: int = 20,
    ann: int = TRADING_DAYS_PER_YEAR,
    lookback: int | None = None,
) -> np.ndarray:
    """Yang-Zhang RV(window) at each of the last `lookback` (default `ann`) history end points."""
    n = len(df)
    lookback = ann if lookback is None else lookback
    first_end = max(window + 1, n - lookback + 1)
    return np.asarray(
        [yang_zhang_rv(df.iloc[:end], window, ann) for end in range(first_end, n + 1)],
        dtype=np.float64,
    )
