"""P-measure realized volatility, jump share and cross-asset correlation per universe instrument."""
from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from src.common import keys
from src.common.constants import DEFAULT_VALKEY_URL, TRADING_DAYS_PER_YEAR
from src.common.log import LogLike
from src.common.market import BarObs, QuoteObs, symbol_of
from src.common.store import JsonStore, NullStore
from src.common.timeutil import is_regular_session, ny_date
from src.quant.realized_vol import (
    avg_pairwise_corr,
    bipower_jump_ratio,
    daily_log_returns,
    percentile_rank,
    trailing_rolling_rv,
    yang_zhang_rv,
)

OHLCV_COLUMNS = ("ts_ns", "open", "high", "low", "close", "volume")
CORRELATION_EXCLUDED_SYMBOLS = frozenset({"VIXY", "VIXM"})   # Vol ETPs are not equity-risk proxies
MIN_ROWS_FOR_RV = 10

HistoryLoader = Callable[[str], "pd.DataFrame | None"]


@dataclass(frozen=True)
class RealizedVolSettings:
    primary_id: str
    universe_ids: tuple[str, ...] = ()
    catalog_path: str = "catalog"
    interval_secs: int = 15
    trading_days: int = TRADING_DAYS_PER_YEAR
    valkey_url: str = DEFAULT_VALKEY_URL


@dataclass(frozen=True)
class RvSnapshot:
    """Primary instrument values for the RealizedVolSnapshot custom data (unrounded)."""

    instrument_id: str
    spot: float
    rv_5d: float
    rv_10d: float
    rv_20d: float
    rv_60d: float
    rv_percentile: float
    jump_ratio: float
    avg_pair_corr: float


class DailyHistoryBook:
    """Daily OHLC history per instrument plus a provisional regular-session bar built from live prices."""

    def __init__(self, history_cap: int) -> None:
        self.history_cap = history_cap
        self.histories: dict[str, pd.DataFrame] = {}
        # Regular-session OHLC built from live prices: open/high/low/close, session (ISO date), last_ts_ns
        self.live_sessions: dict[str, dict[str, Any]] = {}
        self.last_tick_ns: dict[str, int] = {}
        # Trailing completed-history RV20 distribution, keyed by (rows, last ts) so it rebuilds only on new bars
        self._rv20_cache: dict[str, tuple[tuple[int, int], np.ndarray]] = {}

    def set_history(self, instrument_id: str, df: pd.DataFrame) -> None:
        self.histories[instrument_id] = df.sort_values("ts_ns").drop_duplicates("ts_ns").reset_index(drop=True)

    def append_daily_row(self, instrument_id: str, row: dict[str, float]) -> None:
        prev = self.histories.get(instrument_id)
        new_row = pd.DataFrame([row])
        if prev is None or prev.empty:
            self.histories[instrument_id] = new_row
        else:
            self.histories[instrument_id] = (
                pd.concat([prev, new_row], ignore_index=True)
                .drop_duplicates("ts_ns", keep="last")
                .tail(self.history_cap)
                .reset_index(drop=True)
            )

    def on_daily_bar(self, instrument_id: str, row: dict[str, float]) -> None:
        """A completed daily bar supersedes the provisional live session."""
        self.append_daily_row(instrument_id, row)
        self.live_sessions.pop(instrument_id, None)

    def _roll_live_session(self, instrument_id: str, state: dict[str, Any]) -> None:
        """Moves a finished regular session into the daily history."""
        prev = self.histories.get(instrument_id)
        if prev is not None and not prev.empty and int(prev["ts_ns"].iloc[-1]) >= int(state["last_ts_ns"]):
            return  # History already holds this session
        self.append_daily_row(
            instrument_id,
            {
                "ts_ns": int(state["last_ts_ns"]),
                "open": state["open"],
                "high": state["high"],
                "low": state["low"],
                "close": state["close"],
                "volume": 0.0,
            },
        )

    def update_intraday(self, instrument_id: str, price: float, ts_ns: int) -> None:
        if price <= 0.0:
            return
        # Daily history is regular-session OHLC; extended-hours prints would inflate the live range
        if not is_regular_session(ts_ns):
            return
        self.last_tick_ns[instrument_id] = ts_ns
        session = ny_date(ts_ns).isoformat()
        state = self.live_sessions.get(instrument_id)
        if state is not None and state["session"] != session:
            self._roll_live_session(instrument_id, state)
            state = None
        if state is None:
            self.live_sessions[instrument_id] = {
                "session": session,
                "open": price,
                "high": price,
                "low": price,
                "close": price,
                "last_ts_ns": ts_ns,
            }
        else:
            state["high"] = max(state["high"], price)
            state["low"] = min(state["low"], price)
            state["close"] = price
            state["last_ts_ns"] = max(int(state["last_ts_ns"]), ts_ns)

    def effective_frame(self, instrument_id: str) -> pd.DataFrame | None:
        """Completed history plus the current regular session as a provisional daily bar."""
        df = self.histories.get(instrument_id)
        if df is None or df.empty:
            return None
        live = self.live_sessions.get(instrument_id)
        if live is None or int(live["last_ts_ns"]) <= int(df["ts_ns"].iloc[-1]):
            return df
        live_row = pd.DataFrame(
            [
                {
                    "ts_ns": int(live["last_ts_ns"]),
                    "open": live["open"],
                    "high": live["high"],
                    "low": live["low"],
                    "close": live["close"],
                    "volume": 0.0,
                }
            ]
        )
        return pd.concat([df, live_row], ignore_index=True)

    def historical_rv20(self, instrument_id: str, ann: int) -> np.ndarray:
        """Trailing `ann` rolling 20d RVs over completed history (cached until history changes)."""
        hist = self.histories.get(instrument_id)
        if hist is None or hist.empty:
            return np.empty(0, dtype=np.float64)
        key = (len(hist), int(hist["ts_ns"].iloc[-1]))
        cached = self._rv20_cache.get(instrument_id)
        if cached is not None and cached[0] == key:
            return cached[1]
        values = trailing_rolling_rv(hist, 20, ann)
        self._rv20_cache[instrument_id] = (key, values)
        return values


class RealizedVolService:
    """Writes mft:rv:<id> per instrument and mft:rv:universe; returns the primary instrument's snapshot."""

    def __init__(
        self, settings: RealizedVolSettings, store: JsonStore | None = None, log: LogLike | None = None
    ) -> None:
        self.settings = settings
        self.store = store if store is not None else NullStore()
        self.log = log if log is not None else logging.getLogger(__name__)
        self.universe: tuple[str, ...] = tuple(settings.universe_ids) or (settings.primary_id,)
        self.book = DailyHistoryBook(history_cap=max(300, settings.trading_days + 25))

    def warm_start(self, loader: HistoryLoader) -> int:
        """Loads daily history for every universe instrument; returns how many have history."""
        for iid in self.universe:
            df = loader(iid)
            if df is not None and not df.empty:
                self.book.set_history(iid, df)
        return len(self.book.histories)

    def on_quote(self, quote: QuoteObs) -> None:
        mid = quote.mid
        if mid is not None:
            self.book.update_intraday(quote.instrument_id, mid, quote.ts_ns)

    def on_bar(self, bar: BarObs) -> None:
        if bar.is_daily:
            self.book.on_daily_bar(bar.instrument_id, bar.ohlcv_row())
        else:
            self.book.update_intraday(bar.instrument_id, bar.close, bar.ts_ns)

    def avg_pair_corr(self, window: int = 20) -> float:
        ret_series: dict[str, pd.Series] = {}
        for iid in self.universe:
            sym = symbol_of(iid)
            if sym in CORRELATION_EXCLUDED_SYMBOLS:
                continue
            df = self.book.effective_frame(iid)
            if df is None or len(df) < window + 2:
                continue
            ret_series[sym] = daily_log_returns(df)
        # Aligned on trading date so gaps or differing history lengths can't pair returns from different days
        return avg_pairwise_corr(ret_series, window=window)

    def snapshot(self, instrument_id: str, df: pd.DataFrame, avg_corr: float) -> RvSnapshot:
        ann = self.settings.trading_days
        rv_20d = yang_zhang_rv(df, 20, ann)
        # 1-year (trailing trading_days) percentile of the current 20d RV
        rv_pct = percentile_rank(self.book.historical_rv20(instrument_id, ann), rv_20d)
        return RvSnapshot(
            instrument_id=instrument_id,
            spot=float(df["close"].iloc[-1]),
            rv_5d=yang_zhang_rv(df, 5, ann),
            rv_10d=yang_zhang_rv(df, 10, ann),
            rv_20d=rv_20d,
            rv_60d=yang_zhang_rv(df, 60, ann),
            rv_percentile=rv_pct,
            jump_ratio=bipower_jump_ratio(df, 20),
            avg_pair_corr=avg_corr,
        )

    @staticmethod
    def payload(snap: RvSnapshot, wall_ns: int, event_ts_ns: int) -> dict[str, Any]:
        return {
            "symbol": snap.instrument_id,
            "spot": round(snap.spot, 3),
            "rv_5d": round(snap.rv_5d, 6),
            "rv_10d": round(snap.rv_10d, 6),
            "rv_20d": round(snap.rv_20d, 6),
            "rv_60d": round(snap.rv_60d, 6),
            "rv_percentile": round(snap.rv_percentile, 2),
            "jump_ratio": round(snap.jump_ratio, 4),
            "avg_pair_corr": round(snap.avg_pair_corr, 4),
            "ts_ns": wall_ns,
            "event_ts_ns": event_ts_ns,
        }

    def evaluate(self, event_ts_ns: int, wall_ns: int) -> RvSnapshot | None:
        avg_corr = self.avg_pair_corr(window=20)
        universe_summary: dict[str, dict[str, Any]] = {}
        primary: RvSnapshot | None = None

        for iid in self.universe:
            df = self.book.effective_frame(iid)
            if df is None or len(df) < MIN_ROWS_FOR_RV:
                continue
            snap = self.snapshot(iid, df, avg_corr)
            payload = self.payload(snap, wall_ns, event_ts_ns)
            universe_summary[iid] = payload
            self.store.set_json(keys.rv_key(iid), payload)

            if iid == self.settings.primary_id:
                primary = snap
                self.log.info(
                    f"RealizedVol {iid} S={snap.spot:.2f} RV5={snap.rv_5d:.2%} RV20={snap.rv_20d:.2%} "
                    f"RV60={snap.rv_60d:.2%} Pct={snap.rv_percentile:.1f}% Jump={snap.jump_ratio:.1%} "
                    f"Corr={avg_corr:.2f}"
                )

        if universe_summary:
            self.store.set_json(keys.RV_UNIVERSE, universe_summary)
        return primary
