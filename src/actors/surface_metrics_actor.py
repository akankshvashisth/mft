from collections import deque
from datetime import timedelta
import json
import math
import time
import numpy as np
import redis
from nautilus_trader.common import DataActor, DataActorConfig
from nautilus_trader.indicators import RelativeStrengthIndex, SimpleMovingAverage
from nautilus_trader.model import (
    Bar,
    BarAggregation,
    BarSpecification,
    BarType,
    InstrumentId,
    PriceType,
)
from nautilus_trader.persistence import ParquetDataCatalog


class SurfaceRegimeMetricsActorConfig(DataActorConfig):
    def __init__(
        self,
        underlying_id: InstrumentId,
        catalog_path: str = "catalog",
        interval_secs: int = 15,
        history_maxlen: int = 2000,
        valkey_url: str = "redis://127.0.0.1:6379/0",
    ) -> None:
        super().__init__()
        self.underlying_id = underlying_id
        self.catalog_path = catalog_path
        self.interval_secs = interval_secs
        self.history_maxlen = history_maxlen
        self.valkey_url = valkey_url


def _svi_iv(k: float, ttm: float, params: list[float]) -> float:
    """Evaluates annualized implied volatility from raw SVI parameters [a, b, rho, m, sigma]."""
    if len(params) != 5 or ttm <= 1e-6:
        return 0.0
    a, b, rho, m, sig = params
    dk = k - m
    w = max(a + b * (rho * dk + math.sqrt(dk * dk + sig * sig)), 1e-8)
    return math.sqrt(w / ttm)


def _empirical_percentile(history: deque[float], value: float, default: float = 50.0) -> float:
    if len(history) < 5:
        return default
    arr = np.asarray(history, dtype=np.float64)
    return float(round(100.0 * np.mean(arr <= value), 2))


class SurfaceRegimeMetricsActor(DataActor):
    """Computes SVI 25d Risk-Reversal, 25d Butterfly, VRP spread, percentiles, and SMA/RSI regime."""

    def __init__(self, config: SurfaceRegimeMetricsActorConfig) -> None:
        super().__init__(config)
        self.rdb = redis.from_url(config.valkey_url) if config.valkey_url else None
        self.sma_20 = SimpleMovingAverage(20)
        self.sma_50 = SimpleMovingAverage(50)
        self.rsi_14 = RelativeStrengthIndex(14)

        self.atm_hist: deque[float] = deque(maxlen=config.history_maxlen)
        self.rr_hist: deque[float] = deque(maxlen=config.history_maxlen)
        self.bf_hist: deque[float] = deque(maxlen=config.history_maxlen)
        self.vrp_hist: deque[float] = deque(maxlen=config.history_maxlen)

    def _warm_start_indicators(self) -> None:
        try:
            catalog = ParquetDataCatalog(self.config.catalog_path)
            bars = catalog.bars(bar_types=[f"{self.config.underlying_id}-1-DAY-LAST-EXTERNAL"])
            for b in bars[-100:]:
                self.sma_20.handle_bar(b)
                self.sma_50.handle_bar(b)
                self.rsi_14.handle_bar(b)
        except Exception as exc:
            self.log.warning(f"SurfaceRegimeMetricsActor indicator warm-start skipped: {exc}")

    def on_start(self) -> None:
        self._warm_start_indicators()
        self.subscribe_bars(
            BarType(
                self.config.underlying_id,
                BarSpecification(1, BarAggregation.MINUTE, PriceType.LAST),
            )
        )
        self.clock.set_timer(
            "surface_metrics_eval",
            timedelta(seconds=self.config.interval_secs),
            callback=self.on_timer,
        )
        self.log.info(
            f"SurfaceRegimeMetricsActor started for {self.config.underlying_id} "
            f"(SMA20_init={self.sma_20.initialized}, SMA50_init={self.sma_50.initialized})"
        )

    def on_bar(self, bar: Bar) -> None:
        if bar.bar_type.instrument_id == self.config.underlying_id:
            self.sma_20.handle_bar(bar)
            self.sma_50.handle_bar(bar)
            self.rsi_14.handle_bar(bar)

    def _read_json(self, key: str) -> dict:
        if self.rdb is None:
            return {}
        raw = self.rdb.get(key)
        if not raw:
            return {}
        try:
            return json.loads(raw)
        except Exception:
            return {}

    def on_timer(self, event) -> None:
        sym = str(self.config.underlying_id)
        surf = self._read_json(f"mft:surface:{sym}")
        rv = self._read_json(f"mft:rv:{sym}")
        regime = self._read_json("mft:regime:latest")

        if not surf:
            return

        spot = float(surf.get("spot") or 0.0)
        atm_iv = float(surf.get("atm_iv") or 0.0)
        ttm_days = float(surf.get("ttm_days") or 30.0)
        ttm = max(ttm_days / 365.25, 1e-4)
        svi_params = surf.get("svi_params") or []

        # Approximate 25-delta log-moneyness: k_25 ≈ ±0.6745 * atm_iv * sqrt(T)
        k_25 = 0.6745 * max(atm_iv, 0.10) * math.sqrt(ttm)
        if len(svi_params) == 5:
            iv_25p = _svi_iv(-k_25, ttm, svi_params)
            iv_25c = _svi_iv(+k_25, ttm, svi_params)
            iv_atm_fit = _svi_iv(0.0, ttm, svi_params)
        else:
            skew_25d = float(surf.get("skew_25d") or 0.0)
            iv_atm_fit = atm_iv
            iv_25p = atm_iv + 0.5 * skew_25d
            iv_25c = max(atm_iv - 0.5 * skew_25d, 0.01)

        rr_25d = iv_25p - iv_25c
        bf_25d = 0.5 * (iv_25p + iv_25c) - iv_atm_fit

        rv_20d = float(rv.get("rv_20d") or atm_iv)
        vrp_spread = atm_iv - rv_20d

        self.atm_hist.append(atm_iv)
        self.rr_hist.append(rr_25d)
        self.bf_hist.append(bf_25d)
        self.vrp_hist.append(vrp_spread)

        atm_pct = _empirical_percentile(self.atm_hist, atm_iv)
        rr_pct = _empirical_percentile(self.rr_hist, rr_25d)
        bf_pct = _empirical_percentile(self.bf_hist, bf_25d)
        vrp_pct = _empirical_percentile(self.vrp_hist, vrp_spread)

        # Recommend structural strategy bias from surface + VRP regime
        vix_ratio = float(regime.get("ratio") or 1.0)
        if vix_ratio >= 1.0 or vrp_spread < -0.015:
            strategy_bias = "DEFENSIVE_OR_LONG_CONVEXITY"
        elif vrp_spread >= 0.02 and bf_25d >= 0.005:
            strategy_bias = "SHORT_VRP_IRON_CONDOR_OR_CREDIT_SPREAD"
        elif rr_pct <= 20.0:
            strategy_bias = "BULLISH_RISK_REVERSAL_OR_PUT_SPREAD"
        else:
            strategy_bias = "SELECTIVE_RELATIVE_VALUE_SPREADS"

        sma20_val = float(self.sma_20.value) if self.sma_20.initialized else spot
        sma50_val = float(self.sma_50.value) if self.sma_50.initialized else spot
        rsi14_val = float(self.rsi_14.value) * 100.0 if self.rsi_14.initialized else 50.0
        trend_state = (
            "UPTREND" if spot > sma20_val > sma50_val
            else ("DOWNTREND" if spot < sma20_val < sma50_val else "RANGE_OR_TRANSITION")
        )

        payload = {
            "underlying": sym,
            "spot": round(spot, 2),
            "atm_iv": round(atm_iv, 6),
            "atm_iv_pct": atm_pct,
            "rr_25d": round(rr_25d, 6),
            "rr_25d_pct": rr_pct,
            "bf_25d": round(bf_25d, 6),
            "bf_25d_pct": bf_pct,
            "rv_20d": round(rv_20d, 6),
            "vrp_spread": round(vrp_spread, 6),
            "vrp_pct": vrp_pct,
            "vix_term_ratio": round(vix_ratio, 4),
            "strategy_bias": strategy_bias,
            "technicals": {
                "sma_20": round(sma20_val, 2),
                "sma_50": round(sma50_val, 2),
                "rsi_14": round(rsi14_val, 2),
                "trend_state": trend_state,
            },
            "ts_ns": time.time_ns(),
            "event_ts_ns": int(event.ts_event),
        }

        if self.rdb is not None:
            self.rdb.set(f"mft:metrics:{sym}", json.dumps(payload))

        self.log.info(
            f"SurfaceMetrics {sym} ATM={atm_iv:.2%} RR25={rr_25d:+.2%} BF25={bf_25d:+.2%} "
            f"VRP={vrp_spread:+.2%} Trend={trend_state} RSI={rsi14_val:.1f} Bias={strategy_bias}"
        )
