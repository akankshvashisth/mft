"""SVI 25-delta risk reversal / butterfly, VRP spread, their percentiles, strategy bias and trend."""
from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass
from typing import Any

from src.common import keys
from src.common.constants import DEFAULT_VALKEY_URL
from src.common.log import LogLike
from src.common.market import BarObs, LatestValue
from src.common.store import JsonStore, NullStore
from src.common.timeutil import days_to_years
from src.quant.realized_vol import percentile_rank
from src.quant.surface_metrics import classify_strategy_bias, classify_trend, smile_wings

MIN_HISTORY_FOR_PERCENTILE = 5


@dataclass(frozen=True)
class SurfaceMetricsSettings:
    underlying_id: str
    catalog_path: str = "catalog"
    interval_secs: int = 15
    history_maxlen: int = 2000
    valkey_url: str = DEFAULT_VALKEY_URL


@dataclass(frozen=True)
class Technicals:
    """Daily-close indicator values; None while an indicator is still warming up."""

    sma_20: float | None = None
    sma_50: float | None = None
    rsi_14: float | None = None   # 0..1 scale (Nautilus RelativeStrengthIndex)


def _pct(history: deque[float], value: float) -> float:
    return percentile_rank(history, value, default=50.0, min_count=MIN_HISTORY_FOR_PERCENTILE, decimals=2)


class SurfaceMetricsService:
    """Reads the surface, RV and regime snapshots and publishes mft:metrics:<underlying>."""

    def __init__(
        self, settings: SurfaceMetricsSettings, store: JsonStore | None = None, log: LogLike | None = None
    ) -> None:
        self.settings = settings
        self.store = store if store is not None else NullStore()
        self.log = log if log is not None else logging.getLogger(__name__)
        maxlen = settings.history_maxlen
        self.atm_hist: deque[float] = deque(maxlen=maxlen)
        self.rr_hist: deque[float] = deque(maxlen=maxlen)
        self.bf_hist: deque[float] = deque(maxlen=maxlen)
        self.vrp_hist: deque[float] = deque(maxlen=maxlen)
        self._last_indicator_bar = LatestValue()

    def accept_indicator_bar(self, bar: BarObs) -> bool:
        """SMA/RSI are daily-close indicators: only new 1-DAY bars of the underlying may update them.

        Bars at or before the last accepted one are rejected (e.g. a live bar overlapping the warm start).
        """
        if bar.instrument_id != self.settings.underlying_id or not bar.is_daily or bar.step != 1:
            return False
        return self._last_indicator_bar.update(bar.close, bar.ts_ns, strictly_newer=True)

    def evaluate(self, event_ts_ns: int, wall_ns: int, technicals: Technicals = Technicals()) -> dict[str, Any] | None:
        sym = self.settings.underlying_id
        surf = self.store.get_json(keys.surface_key(sym))
        rv = self.store.get_json(keys.rv_key(sym))
        regime = self.store.get_json(keys.REGIME_LATEST)
        if not surf:
            return None

        spot = float(surf.get("spot") or 0.0)
        atm_iv = float(surf.get("atm_iv") or 0.0)
        ttm = max(days_to_years(float(surf.get("ttm_days") or 30.0)), 1e-4)
        svi_params = surf.get("svi_params") or []
        wings = smile_wings(
            atm_iv,
            ttm,
            svi_params if len(svi_params) == 5 else None,
            skew_25d=float(surf.get("skew_25d") or 0.0),
        )
        rr_25d = wings.rr_25d
        bf_25d = wings.bf_25d

        rv_20d = float(rv.get("rv_20d") or atm_iv)
        vrp_spread = atm_iv - rv_20d

        self.atm_hist.append(atm_iv)
        self.rr_hist.append(rr_25d)
        self.bf_hist.append(bf_25d)
        self.vrp_hist.append(vrp_spread)

        atm_pct = _pct(self.atm_hist, atm_iv)
        rr_pct = _pct(self.rr_hist, rr_25d)
        bf_pct = _pct(self.bf_hist, bf_25d)
        vrp_pct = _pct(self.vrp_hist, vrp_spread)

        # Recommend structural strategy bias from surface + VRP regime
        vix_ratio = float(regime.get("ratio") or 1.0)
        strategy_bias = classify_strategy_bias(vix_ratio, vrp_spread, bf_25d, rr_pct)

        sma20_val = technicals.sma_20 if technicals.sma_20 is not None else spot
        sma50_val = technicals.sma_50 if technicals.sma_50 is not None else spot
        rsi14_val = technicals.rsi_14 * 100.0 if technicals.rsi_14 is not None else 50.0
        trend_state = classify_trend(spot, sma20_val, sma50_val)

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
            "ts_ns": wall_ns,
            "event_ts_ns": event_ts_ns,
        }
        self.store.set_json(keys.metrics_key(sym), payload)

        self.log.info(
            f"SurfaceMetrics {sym} ATM={atm_iv:.2%} RR25={rr_25d:+.2%} BF25={bf_25d:+.2%} "
            f"VRP={vrp_spread:+.2%} Trend={trend_state} RSI={rsi14_val:.1f} Bias={strategy_bias}"
        )
        return payload
