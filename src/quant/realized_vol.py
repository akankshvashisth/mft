"""Physical (P-measure) realized volatility, jump share, correlation and percentile helpers."""
from __future__ import annotations

from collections.abc import Iterable, Mapping

import numpy as np
import pandas as pd
from scipy.stats import percentileofscore

from src.common.constants import NY_TZ_NAME, TRADING_DAYS_PER_YEAR


def sanitize_ohlc(df: pd.DataFrame, max_abs_log_ret: float = 0.45) -> pd.DataFrame:
    """Drop rows with non-finite/zero prices, OHLC structure violations or outlier log-return spikes.

    Applies three checks in order:
    1. Any of ``open``, ``high``, ``low``, ``close`` is non-finite or ≤ 0.
    2. OHLC consistency: ``high < max(open, close)`` or ``low > min(open, close)``.
    3. Outlier spike: |ln(C_t / last_accepted_close)| > max_abs_log_ret checked *sequentially*
       so a single 10x spike bar does not also drop the valid bar right after it.
    """
    required = ("open", "high", "low", "close")
    if df.empty or any(col not in df for col in required):
        return df.iloc[:0].copy()

    df = df.copy()
    prices = df[list(required)].to_numpy(dtype=np.float64)
    mask = np.isfinite(prices).all(axis=1) & (prices > 0.0).all(axis=1)
    mask &= ~(df["high"].to_numpy(dtype=np.float64) < np.maximum(df["open"].to_numpy(dtype=np.float64), df["close"].to_numpy(dtype=np.float64)))
    mask &= ~(df["low"].to_numpy(dtype=np.float64) > np.minimum(df["open"].to_numpy(dtype=np.float64), df["close"].to_numpy(dtype=np.float64)))

    df = df.loc[mask].copy()
    if len(df) < 2:
        return df

    # Sequential outlier check: each row's close is compared to the *last accepted* close
    closes = df["close"].to_numpy(dtype=np.float64)
    keep = np.ones(len(df), dtype=bool)
    last_accepted = closes[0]
    for i in range(1, len(df)):
        if abs(np.log(closes[i] / last_accepted)) > max_abs_log_ret:
            keep[i] = False  # drop this spike bar
        else:
            last_accepted = closes[i]  # only update on accepted rows
    return df.loc[keep]


def yang_zhang_rv(df: pd.DataFrame, window: int, ann: int = TRADING_DAYS_PER_YEAR) -> float:
    """Annualized Yang-Zhang realized volatility over the last `window` OHLC bars.

    Sanitizes only the active `window + 1` tail slice so an old corrupt bar 200 days
    ago does not zero out today's 20d RV.  With fewer than ``window + 1`` usable bars
    it falls back to close-to-close volatility of what is available.
    """
    if window <= 0 or ann <= 0 or df.empty:
        return 0.0
    required = ("open", "high", "low", "close")
    if any(column not in df for column in required):
        return 0.0

    # Sanitize only the active tail slice
    sub = df.iloc[-(window + 1):].copy()
    sub = sanitize_ohlc(sub)
    if len(sub) < 2:
        return 0.0

    prices = sub[list(required)].to_numpy(dtype=np.float64)
    if prices.shape[0] == 0 or not np.isfinite(prices).all() or (prices <= 0.0).any():
        return 0.0

    if len(sub) < window + 1:
        rets = np.log(sub["close"] / sub["close"].shift(1)).dropna()
        return float(rets.std(ddof=1) * np.sqrt(ann)) if len(rets) > 1 else 0.0

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
    """Fraction of recent variance attributable to discrete jumps: (RV - BV) / RV, clipped to [0, 1].

    Applies the finite-sample correction BV = W/(W-1) · (π/2) · Σ|r_t|·|r_{t-1}|
    so the expected jump share on pure Gaussian returns is ≈ 0 (not biased +5 % at W=20).
    """
    if window <= 0 or df.empty or "close" not in df:
        return 0.0
    closes = df["close"].to_numpy(dtype=np.float64)
    if not np.isfinite(closes).all() or (closes <= 0.0).any():
        return 0.0
    rets = np.log(df["close"] / df["close"].shift(1)).dropna().to_numpy(dtype=np.float64)
    if len(rets) < window + 1:
        return 0.0
    r = rets[-window:]
    rv = float(np.sum(r ** 2))
    if rv <= 1e-12:
        return 0.0
    w = float(len(r))
    # Finite-sample correction: W/(W-1) * (pi/2) * sum(|r_t| * |r_{t-1}|)
    finite_correction = w / max(w - 1.0, 0.5)
    bv = float(finite_correction * (np.pi / 2.0) * np.sum(np.abs(r[1:]) * np.abs(r[:-1])))
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
    """Yang-Zhang RV(window) at each of the last `lookback` (default `ann`) history end points.

    Fully vectorised via pre-computed log components and rolling sums on the
    pre-sanitised DataFrame.  Runs 250 bars in well under 5 ms.
    Returns at most ``lookback`` values.
    """
    n = len(df)
    lookback_val = ann if lookback is None else lookback

    # Pre-sanitize once
    clean = sanitize_ohlc(df)
    kept_idx = len(clean)
    if kept_idx < window + 1:
        return np.array([], dtype=np.float64)

    # Truncate to the last lookback windows
    first_kept = max(window + 1, kept_idx - lookback_val + 1)
    if first_kept > kept_idx:
        return np.array([], dtype=np.float64)

    w = window
    out_len = kept_idx - first_kept + 1
    result = np.empty(out_len, dtype=np.float64)

    # Pre-compute the log components on the full clean DataFrame
    o_arr = clean["open"].to_numpy(dtype=np.float64)
    h_arr = clean["high"].to_numpy(dtype=np.float64)
    l_arr = clean["low"].to_numpy(dtype=np.float64)
    c_arr = clean["close"].to_numpy(dtype=np.float64)

    log_oc = np.log(o_arr[1:] / c_arr[:-1])
    log_co = np.log(c_arr[1:] / o_arr[1:])
    log_ho = np.log(h_arr[1:] / o_arr[1:])
    log_lo = np.log(l_arr[1:] / o_arr[1:])
    log_hc = np.log(h_arr[1:] / c_arr[1:])
    log_lc = np.log(l_arr[1:] / c_arr[1:])

    rs_t = log_ho * log_hc + log_lo * log_lc
    n_comp = kept_idx - 1  # number of usable log returns
    # Use cumulative sums for O(1) per-window evaluation
    cs_oc = np.zeros(n_comp + 1, dtype=np.float64)
    cs_oc2 = np.zeros(n_comp + 1, dtype=np.float64)
    cs_co = np.zeros(n_comp + 1, dtype=np.float64)
    cs_co2 = np.zeros(n_comp + 1, dtype=np.float64)
    cs_rs = np.zeros(n_comp + 1, dtype=np.float64)
    np.cumsum(log_oc, out=cs_oc[1:])
    np.cumsum(log_oc ** 2, out=cs_oc2[1:])
    np.cumsum(log_co, out=cs_co[1:])
    np.cumsum(log_co ** 2, out=cs_co2[1:])
    np.cumsum(rs_t, out=cs_rs[1:])

    k = 0.34 / (1.34 + (w + 1) / max(w - 1, 1))
    for j in range(out_len):
        end = first_kept + j
        s = end - w - 1
        e = end - 1
        cnt = float(w)
        sum_oc = cs_oc[e] - cs_oc[s]
        sum_oc2 = cs_oc2[e] - cs_oc2[s]
        sum_co = cs_co[e] - cs_co[s]
        sum_co2 = cs_co2[e] - cs_co2[s]
        sum_rs = cs_rs[e] - cs_rs[s]

        var_o = max((sum_oc2 - sum_oc ** 2 / cnt) / (cnt - 1.0), 0.0)
        var_c = max((sum_co2 - sum_co ** 2 / cnt) / (cnt - 1.0), 0.0)
        rs = sum_rs / cnt

        yz_var = max(float(var_o + k * var_c + (1.0 - k) * rs), 1e-10)
        result[j] = float(np.sqrt(yz_var * ann))

    return result
