from pathlib import Path
from datetime import timedelta
import json
import time
import numpy as np
import pandas as pd
import redis
from nautilus_trader.common import DataActor, DataActorConfig
from nautilus_trader.model import (
    Bar,
    BarAggregation,
    BarSpecification,
    BarType,
    CustomData,
    DataType,
    InstrumentId,
    PriceType,
    QuoteTick,
)
from nautilus_trader.persistence import ParquetDataCatalog
from src.models.vol_surface import RealizedVolSnapshot


class RealizedVolActorConfig(DataActorConfig):
    def __init__(
        self,
        primary_id: InstrumentId,
        universe_ids: tuple[InstrumentId, ...] = (),
        catalog_path: str = "catalog",
        interval_secs: int = 15,
        trading_days: int = 252,
        valkey_url: str = "redis://127.0.0.1:6379/0",
    ) -> None:
        super().__init__()
        self.primary_id = primary_id
        self.universe_ids = universe_ids or (primary_id,)
        self.catalog_path = catalog_path
        self.interval_secs = interval_secs
        self.trading_days = trading_days
        self.valkey_url = valkey_url


def _yang_zhang_rv(df: pd.DataFrame, window: int, ann: int = 252) -> float:
    """Computes annualized Yang-Zhang realized volatility over `window` bars."""
    if len(df) < window + 1:
        rets = np.log(df["close"] / df["close"].shift(1)).dropna()
        return float(rets.std(ddof=1) * np.sqrt(ann)) if len(rets) > 1 else 0.0

    sub = df.iloc[-(window + 1) :].copy()
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


def _bipower_jump_ratio(df: pd.DataFrame, window: int = 20) -> float:
    """Estimates fraction of recent variance attributable to discrete jumps via bipower variation."""
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


class RealizedVolActor(DataActor):
    """Stateful P-measure realized volatility, jump, and cross-asset correlation engine."""

    def __init__(self, config: RealizedVolActorConfig) -> None:
        super().__init__(config)
        self.histories: dict[InstrumentId, pd.DataFrame] = {}
        self.live_session_ohlc: dict[InstrumentId, dict[str, float]] = {}
        self.last_tick_ns: dict[InstrumentId, int] = {}
        self.rdb = redis.from_url(config.valkey_url) if config.valkey_url else None

    def _warm_start_from_catalog(self) -> None:
        catalog = None
        try:
            catalog = ParquetDataCatalog(self.config.catalog_path)
        except Exception as exc:
            self.log.warning(f"RealizedVolActor could not open catalog: {exc}")

        for iid in self.config.universe_ids:
            df = None
            if catalog is not None:
                try:
                    bars = catalog.bars(instrument_ids=[iid])
                    if bars:
                        rows = [
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
                        df = pd.DataFrame(rows)
                except Exception:
                    pass

            if df is None or df.empty:
                sym = iid.symbol.value
                matches = sorted(Path("data/thetadata/bars").glob(f"{sym}_*.parquet"))
                if matches:
                    raw = pd.read_parquet(matches[-1])
                    ts_col = "ts_dt" if "ts_dt" in raw.columns else None
                    ts_vals = (
                        pd.to_datetime(raw[ts_col]).astype("int64")
                        if ts_col
                        else np.arange(len(raw), dtype=np.int64) * 86_400_000_000_000
                    )
                    df = pd.DataFrame(
                        {
                            "ts_ns": ts_vals,
                            "open": raw["open"].astype(float),
                            "high": raw["high"].astype(float),
                            "low": raw["low"].astype(float),
                            "close": raw["close"].astype(float),
                            "volume": raw.get("volume", pd.Series(1.0, index=raw.index)).astype(float),
                        }
                    )

            if df is not None and not df.empty:
                self.histories[iid] = (
                    df.sort_values("ts_ns").drop_duplicates("ts_ns").reset_index(drop=True)
                )

    def on_start(self) -> None:
        self._warm_start_from_catalog()

        # Subscribe to live quotes & 1-min bars for primary_id (and any cached universe instruments)
        cached_ids = {inst.id for inst in self.cache.instruments()}
        for iid in self.config.universe_ids:
            if iid == self.config.primary_id or iid in cached_ids:
                self.subscribe_quotes(iid)
                self.subscribe_bars(
                    BarType(iid, BarSpecification(1, BarAggregation.MINUTE, PriceType.LAST))
                )

        self.clock.set_timer(
            "realized_vol_eval",
            timedelta(seconds=self.config.interval_secs),
            callback=self.on_timer,
        )
        self.log.info(
            f"RealizedVolActor warm-started {len(self.histories)}/{len(self.config.universe_ids)} "
            f"universe symbols from {self.config.catalog_path}"
        )

    def _update_intraday_bar(self, iid: InstrumentId, price: float, ts_ns: int) -> None:
        if price <= 0.0:
            return
        self.last_tick_ns[iid] = ts_ns
        state = self.live_session_ohlc.get(iid)
        if state is None:
            self.live_session_ohlc[iid] = {
                "open": price,
                "high": price,
                "low": price,
                "close": price,
            }
        else:
            state["high"] = max(state["high"], price)
            state["low"] = min(state["low"], price)
            state["close"] = price

    def on_quote(self, tick: QuoteTick) -> None:
        bid = float(tick.bid_price)
        ask = float(tick.ask_price)
        if bid <= 0.0 or ask <= 0.0 or ask < bid:
            return
        self._update_intraday_bar(tick.instrument_id, 0.5 * (bid + ask), int(tick.ts_event))

    def on_bar(self, bar: Bar) -> None:
        iid = bar.bar_type.instrument_id
        spec = bar.bar_type.spec
        if spec.aggregation == BarAggregation.DAY:
            row = pd.DataFrame(
                [
                    {
                        "ts_ns": int(bar.ts_event),
                        "open": float(bar.open),
                        "high": float(bar.high),
                        "low": float(bar.low),
                        "close": float(bar.close),
                        "volume": float(bar.volume),
                    }
                ]
            )
            prev = self.histories.get(iid)
            if prev is None:
                self.histories[iid] = row
            else:
                self.histories[iid] = (
                    pd.concat([prev, row], ignore_index=True)
                    .drop_duplicates("ts_ns", keep="last")
                    .tail(300)
                    .reset_index(drop=True)
                )
            self.live_session_ohlc.pop(iid, None)
        else:
            self._update_intraday_bar(iid, float(bar.close), int(bar.ts_event))

    def _effective_df(self, iid: InstrumentId) -> pd.DataFrame | None:
        df = self.histories.get(iid)
        live = self.live_session_ohlc.get(iid)
        if df is None or df.empty:
            return None
        if live is None:
            return df
        # Append current live intraday session bar for real-time RV tracking
        live_row = pd.DataFrame(
            [
                {
                    "ts_ns": int(df["ts_ns"].iloc[-1]) + 86_400_000_000_000,
                    "open": live["open"],
                    "high": live["high"],
                    "low": live["low"],
                    "close": live["close"],
                    "volume": 0.0,
                }
            ]
        )
        return pd.concat([df, live_row], ignore_index=True)

    def _compute_avg_pair_corr(self, window: int = 20) -> float:
        ret_series: dict[str, pd.Series] = {}
        for iid in self.config.universe_ids:
            sym = iid.symbol.value
            if sym in ("VIXY", "VIXM"):
                continue
            df = self._effective_df(iid)
            if df is None or len(df) < window + 2:
                continue
            rets = np.log(df["close"] / df["close"].shift(1)).dropna().iloc[-window:]
            ret_series[sym] = rets.reset_index(drop=True)

        if len(ret_series) < 2:
            return 0.0
        mat = pd.DataFrame(ret_series).dropna()
        if len(mat) < 5:
            return 0.0
        corr = mat.corr().to_numpy()
        n = corr.shape[0]
        off_diag = corr[np.triu_indices(n, k=1)]
        finite = off_diag[np.isfinite(off_diag)]
        return float(np.mean(finite)) if len(finite) > 0 else 0.0

    def on_timer(self, event) -> None:
        avg_corr = self._compute_avg_pair_corr(window=20)
        now_wall_ns = time.time_ns()
        universe_summary: dict[str, dict] = {}

        for iid in self.config.universe_ids:
            df = self._effective_df(iid)
            if df is None or len(df) < 10:
                continue

            spot = float(df["close"].iloc[-1])
            rv_5d = _yang_zhang_rv(df, 5, self.config.trading_days)
            rv_10d = _yang_zhang_rv(df, 10, self.config.trading_days)
            rv_20d = _yang_zhang_rv(df, 20, self.config.trading_days)
            rv_60d = _yang_zhang_rv(df, 60, self.config.trading_days)
            jump_ratio = _bipower_jump_ratio(df, 20)

            # Rolling 20d RV percentile over trailing history
            rolling_20 = [
                _yang_zhang_rv(df.iloc[: idx + 1], 20, self.config.trading_days)
                for idx in range(21, len(df) + 1)
            ]
            if rolling_20:
                rv_pct = float(100.0 * np.mean(np.asarray(rolling_20) <= rv_20d))
            else:
                rv_pct = 50.0

            payload = {
                "symbol": str(iid),
                "spot": round(spot, 3),
                "rv_5d": round(rv_5d, 6),
                "rv_10d": round(rv_10d, 6),
                "rv_20d": round(rv_20d, 6),
                "rv_60d": round(rv_60d, 6),
                "rv_percentile": round(rv_pct, 2),
                "jump_ratio": round(jump_ratio, 4),
                "avg_pair_corr": round(avg_corr, 4),
                "ts_ns": now_wall_ns,
                "event_ts_ns": int(event.ts_event),
            }
            universe_summary[str(iid)] = payload

            if self.rdb is not None:
                self.rdb.set(f"mft:rv:{iid}", json.dumps(payload))

            if iid == self.config.primary_id:
                snap = RealizedVolSnapshot(
                    underlying_id=iid,
                    spot=spot,
                    rv_5d=rv_5d,
                    rv_10d=rv_10d,
                    rv_20d=rv_20d,
                    rv_60d=rv_60d,
                    rv_percentile=rv_pct,
                    jump_ratio=jump_ratio,
                    avg_pair_corr=avg_corr,
                    ts_event=int(event.ts_event),
                    ts_init=now_wall_ns,
                )
                self.publish_data(
                    DataType("RealizedVolSnapshot"),
                    CustomData(DataType("RealizedVolSnapshot"), snap),
                )
                self.log.info(
                    f"RealizedVol {iid} S={spot:.2f} RV5={rv_5d:.2%} RV20={rv_20d:.2%} "
                    f"RV60={rv_60d:.2%} Pct={rv_pct:.1f}% Jump={jump_ratio:.1%} Corr={avg_corr:.2f}"
                )

        if self.rdb is not None and universe_summary:
            self.rdb.set("mft:rv:universe", json.dumps(universe_summary))
