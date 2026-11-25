"""Pure transformation pipeline: Databento OPRA definitions + CBBO-1m -> canonical daily backtest Parquet.

Every function in this module is stateless and offline-safe -- no databento, nautilus_trader,
or redis imports.  Designed to be called by ``scripts/fetch_opra_databento.py`` after the
raw .dbn.zst files have been downloaded and converted to DataFrames.
"""
from __future__ import annotations

from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from src.common.constants import DEFAULT_RISK_FREE_RATE, NY_TZ_NAME

NY_TZ = ZoneInfo(NY_TZ_NAME)

CANONICAL_COLUMNS = [
    "date", "expiry", "dte", "spot", "rv_20d", "atm_iv",
    "right", "strike", "bid", "ask", "mid", "bid_size", "ask_size", "iv",
]


def parse_opra_definitions(def_df: pd.DataFrame) -> pd.DataFrame:
    """Extract and normalise Databento OPRA.PILLAR definition rows."""
    if def_df.empty:
        return pd.DataFrame(columns=["instrument_id", "strike", "expiry", "right", "multiplier"])

    df = def_df.copy()
    if "raw_symbol" not in df.columns and "symbol" in df.columns:
        df = df.rename(columns={"symbol": "raw_symbol"})

    raw_exp = df.get("expiration")
    if raw_exp is None:
        return pd.DataFrame(columns=["instrument_id", "strike", "expiry", "right", "multiplier"])
    try:
        expiry_series = pd.to_datetime(raw_exp, errors="coerce")
    except Exception:
        return pd.DataFrame(columns=["instrument_id", "strike", "expiry", "right", "multiplier"])
    df["expiry"] = expiry_series.dt.strftime("%Y-%m-%d")
    df = df.dropna(subset=["expiry"]).copy()

    strike_col = "strike_price" if "strike_price" in df.columns else "strike"
    df["strike"] = pd.to_numeric(df.get(strike_col, float("nan")), errors="coerce")

    cls_col = "instrument_class" if "instrument_class" in df.columns else "right"
    df["right"] = df.get(cls_col, "").astype(str).str.upper().str[0]

    mult_col = "contract_multiplier" if "contract_multiplier" in df.columns else (
        "unit_of_measure_qty" if "unit_of_measure_qty" in df.columns else None
    )
    if mult_col is not None:
        raw_mult = pd.to_numeric(df[mult_col], errors="coerce").fillna(100).astype("int64")
        # Databento OPRA.PILLAR uses INT32_MAX (2147483647) or 0 as sentinel for standard 100-share contracts
        df["multiplier"] = raw_mult.replace({2147483647: 100, 0: 100}).astype(int)
    else:
        df["multiplier"] = 100

    valid = (
        df["right"].isin(["C", "P"])
        & (df["strike"] > 0)
        & (df["multiplier"] == 100)
        & df["expiry"].notna()
    )
    df = df.loc[valid].copy()
    if df.empty:
        return pd.DataFrame(columns=["instrument_id", "strike", "expiry", "right", "multiplier"])

    if "instrument_id" not in df.columns:
        df["instrument_id"] = (
            df.get("raw_symbol", "") + "_" + df["expiry"] + "_"
            + df["right"] + df["strike"].astype(str)
        )

    df = df.sort_values("instrument_id").drop_duplicates(subset=["instrument_id"], keep="last")
    return df[["instrument_id", "strike", "expiry", "right", "multiplier"]].reset_index(drop=True)


def _to_ny_time(ts_col: pd.Series) -> pd.Series:
    """Convert an event-timestamp column (epoch-ns or datetime) to America/New_York time."""
    if ts_col.dtype == "int64" or ts_col.dtype.kind in ("i", "u"):
        dt = pd.to_datetime(ts_col.astype("int64"), unit="ns", utc=True)
    else:
        dt = pd.to_datetime(ts_col, utc=True)
    return dt.dt.tz_convert(NY_TZ)


def extract_daily_nbbo_snapshot(
    cbbo_df: pd.DataFrame,
    *,
    snap_hour_et: int = 15,
    snap_minute_et: int = 45,
    window_minutes: int = 30,
) -> pd.DataFrame:
    """Extract the last two-sided NBBO quote at or before snap_hour_et:snap_minute_et ET
    for each (date, instrument_id)."""
    if cbbo_df.empty:
        return pd.DataFrame(
            columns=["date", "instrument_id", "bid", "ask", "mid", "bid_size", "ask_size"]
        )

    # Reset index immediately so concatenated daily frames never trigger duplicate-label reindex errors
    df = cbbo_df.copy()
    ts_col = None
    for col in ("ts_recv", "ts_event"):
        if col in df.columns:
            ts_col = col
            break
    if ts_col is None:
        if isinstance(df.index, pd.DatetimeIndex):
            df = df.reset_index()
            ts_col = df.columns[0]
        else:
            return pd.DataFrame(
                columns=["date", "instrument_id", "bid", "ask", "mid", "bid_size", "ask_size"]
            )
    else:
        df = df.reset_index(drop=True)

    ny_time = _to_ny_time(df[ts_col])
    df["_ny_time"] = ny_time
    df["date"] = ny_time.dt.strftime("%Y-%m-%d")

    # Filter using total seconds so 15:45:30 is strictly excluded (even if valid)
    snap_deadline_secs = snap_hour_et * 3600 + snap_minute_et * 60
    snap_start_secs = snap_deadline_secs - window_minutes * 60
    df["_total_secs"] = ny_time.dt.hour * 3600 + ny_time.dt.minute * 60 + ny_time.dt.second
    in_window = (
        df["_ny_time"].notna()
        & (df["_total_secs"] >= snap_start_secs)
        & (df["_total_secs"] <= snap_deadline_secs)
    )
    df = df.loc[in_window].copy()
    if df.empty:
        return pd.DataFrame(
            columns=["date", "instrument_id", "bid", "ask", "mid", "bid_size", "ask_size"]
        )

    bid_col = "bid_px_00" if "bid_px_00" in df.columns else "bid"
    ask_col = "ask_px_00" if "ask_px_00" in df.columns else "ask"
    bid_sz_col = "bid_sz_00" if "bid_sz_00" in df.columns else None
    ask_sz_col = "ask_sz_00" if "ask_sz_00" in df.columns else None

    df["bid_val"] = pd.to_numeric(df[bid_col], errors="coerce")
    df["ask_val"] = pd.to_numeric(df[ask_col], errors="coerce")

    valid = (
        (df["bid_val"] > 0.0)
        & (df["ask_val"] >= df["bid_val"])
        & np.isfinite(df["bid_val"])
        & np.isfinite(df["ask_val"])
    )
    df = df.loc[valid].sort_values("_ny_time").reset_index(drop=True)
    if df.empty:
        return pd.DataFrame(
            columns=["date", "instrument_id", "bid", "ask", "mid", "bid_size", "ask_size"]
        )

    # Take last valid intraday observation per (date, instrument_id) at or before snap_time
    last = df.groupby(["date", "instrument_id"], as_index=False).last()

    result = pd.DataFrame()
    result["date"] = last["date"]
    result["instrument_id"] = last["instrument_id"]
    result["bid"] = last["bid_val"].values
    result["ask"] = last["ask_val"].values
    result["mid"] = 0.5 * (result["bid"] + result["ask"])

    if bid_sz_col and bid_sz_col in df.columns:
        result["bid_size"] = pd.to_numeric(last[bid_sz_col], errors="coerce").fillna(0).astype(int)
    else:
        result["bid_size"] = 0
    if ask_sz_col and ask_sz_col in df.columns:
        result["ask_size"] = pd.to_numeric(last[ask_sz_col], errors="coerce").fillna(0).astype(int)
    else:
        result["ask_size"] = 0
    result["bid_size"] = result["bid_size"].clip(lower=0)
    result["ask_size"] = result["ask_size"].clip(lower=0)
    return result.reset_index(drop=True)


def compute_daily_rv_table(
    bars_df: pd.DataFrame,
    window: int = 20,
) -> pd.DataFrame:
    """Compute rv_20d (YZ RV) and spot (close) for each date, no lookahead.

    Supports a ``"date"`` (ISO string), ``"ts_ns"`` (epoch ns), ``"ts_event"`` (epoch ns)
    column, or a ``DatetimeIndex`` index.
    """
    from src.quant.realized_vol import sanitize_ohlc, yang_zhang_rv

    if bars_df.empty:
        return pd.DataFrame(columns=["date", "spot", "rv_20d"])

    df = bars_df.copy()
    if "date" in df.columns:
        df["_date"] = pd.to_datetime(df["date"], errors="coerce")
    elif "ts_event" in df.columns:
        df["_date"] = _to_ny_time(df["ts_event"])
    elif isinstance(df.index, pd.DatetimeIndex):
        df = df.reset_index()
        df["_date"] = df[df.columns[0]]
        if df["_date"].dt.tz is None:
            df["_date"] = df["_date"].dt.tz_localize("UTC").dt.tz_convert(NY_TZ)
    elif "ts_ns" in df.columns:
        df["_date"] = pd.to_datetime(df["ts_ns"].astype("int64"), unit="ns", utc=True).dt.tz_convert(NY_TZ)
    else:
        return pd.DataFrame(columns=["date", "spot", "rv_20d"])

    df = df.dropna(subset=["_date"]).sort_values("_date").reset_index(drop=True)
    for col in ("open", "high", "low", "close"):
        if col not in df.columns:
            return pd.DataFrame(columns=["date", "spot", "rv_20d"])

    clean = sanitize_ohlc(df)
    if len(clean) < window + 1:
        return pd.DataFrame(columns=["date", "spot", "rv_20d"])

    results: list[dict[str, Any]] = []
    for i in range(window, len(clean)):
        sub = clean.iloc[: i + 1]
        rv = yang_zhang_rv(sub, window=window)
        row_date = clean.iloc[i]["_date"]
        date_str = row_date.strftime("%Y-%m-%d") if hasattr(row_date, "strftime") else str(row_date)
        results.append({"date": date_str, "spot": float(clean.iloc[i]["close"]), "rv_20d": rv})

    out = pd.DataFrame(results)
    if out.empty:
        return pd.DataFrame(columns=["date", "spot", "rv_20d"])
    out = out.loc[out["rv_20d"].notna() & (out["rv_20d"] > 0.0) & np.isfinite(out["rv_20d"])].copy()
    return out.reset_index(drop=True)


def build_opra_backtest_chain(
    def_df: pd.DataFrame,
    cbbo_df: pd.DataFrame,
    bars_df: pd.DataFrame,
    *,
    risk_free_rate: float = DEFAULT_RISK_FREE_RATE,
    dividend_yield: float = 0.0,
    min_dte: int = 1,
    max_dte: int = 90,
    snap_hour_et: int = 15,
    snap_minute_et: int = 45,
) -> pd.DataFrame:
    """Build the canonical daily backtest chain (Section 2 schema)."""
    from src.quant.black_scholes import bsm_implied_vols

    parsed = parse_opra_definitions(def_df)
    if parsed.empty:
        return pd.DataFrame(columns=CANONICAL_COLUMNS)

    snapshot = extract_daily_nbbo_snapshot(cbbo_df, snap_hour_et=snap_hour_et, snap_minute_et=snap_minute_et)
    if snapshot.empty:
        return pd.DataFrame(columns=CANONICAL_COLUMNS)

    rv_table = compute_daily_rv_table(bars_df)
    if rv_table.empty:
        return pd.DataFrame(columns=CANONICAL_COLUMNS)

    merged = snapshot.merge(parsed, on="instrument_id", how="inner")
    if merged.empty:
        return pd.DataFrame(columns=CANONICAL_COLUMNS)

    merged = merged.merge(rv_table, on="date", how="inner", suffixes=("", "_rv"))
    if merged.empty:
        return pd.DataFrame(columns=CANONICAL_COLUMNS)

    merged["expiry_dt"] = pd.to_datetime(merged["expiry"], errors="coerce")
    merged["date_dt"] = pd.to_datetime(merged["date"], errors="coerce")
    merged = merged.dropna(subset=["expiry_dt", "date_dt"]).copy()
    merged["dte"] = (merged["expiry_dt"] - merged["date_dt"]).dt.days
    merged = merged.loc[(merged["dte"] >= min_dte) & (merged["dte"] <= max_dte)].copy()
    if merged.empty:
        return pd.DataFrame(columns=CANONICAL_COLUMNS)

    T = (merged["dte"] / 365.25).clip(lower=1e-6).values
    S = merged["spot"].values * np.exp(-dividend_yield * T)  # dividend-discounted per bsm_implied_vols docs
    K = merged["strike"].values
    r = np.full_like(T, risk_free_rate)
    flags = merged["right"].values
    prices = merged["mid"].values

    ivs = bsm_implied_vols(prices, S, K, T, r, flags)
    merged["iv"] = ivs
    merged = merged.loc[merged["iv"].notna() & (merged["iv"] >= 0.02) & (merged["iv"] <= 4.00)].copy()
    if merged.empty:
        return pd.DataFrame(columns=CANONICAL_COLUMNS)

    # Compute atm_iv per date: median IV of 4 strikes closest to spot,
    # in expiry closest to 30 DTE (preferring [21, 45] DTE range).
    date_atm: dict[str, float] = {}
    for date_str, grp in merged.groupby("date"):
        exp_dtes = grp.groupby("expiry")["dte"].mean()
        target_dte = 30
        in_range = exp_dtes[(exp_dtes >= 21) & (exp_dtes <= 45)]
        if not in_range.empty:
            best_exp = (in_range - target_dte).abs().idxmin()
        elif not exp_dtes.empty:
            best_exp = (exp_dtes - target_dte).abs().idxmin()
        else:
            date_atm[date_str] = float("nan")
            continue
        slice_df = grp[grp["expiry"] == best_exp]
        if slice_df.empty:
            date_atm[date_str] = float("nan")
            continue
        spot_val = float(slice_df["spot"].iloc[0])
        slice_df = slice_df.copy()
        slice_df["_dist"] = abs(slice_df["strike"] - spot_val)
        closest = slice_df.nsmallest(4, "_dist")
        date_atm[date_str] = closest["iv"].median()

    merged["atm_iv"] = merged["date"].map(date_atm)
    merged["atm_iv"] = merged["atm_iv"].fillna(merged.groupby("date")["iv"].transform("median"))

    result = pd.DataFrame()
    result["date"] = merged["date"]
    result["expiry"] = merged["expiry"]
    result["dte"] = merged["dte"].astype(int)
    for col in ("spot", "rv_20d", "atm_iv", "bid", "ask", "mid", "strike", "iv"):
        result[col] = merged[col].astype(float)
    result["right"] = merged["right"]
    result["bid_size"] = merged["bid_size"].astype(int)
    result["ask_size"] = merged["ask_size"].astype(int)

    result = result.sort_values(["date", "expiry", "right", "strike"]).reset_index(drop=True)
    return result[CANONICAL_COLUMNS]