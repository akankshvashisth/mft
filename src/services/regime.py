"""VIX / VIX3M term-structure regime with feed-staleness gating."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from src.common import keys
from src.common.constants import DEFAULT_VALKEY_URL, NS_PER_SEC
from src.common.log import LogLike
from src.common.market import BarObs, LatestValue, QuoteObs
from src.common.store import JsonStore, NullStore


@dataclass(frozen=True)
class RegimeSettings:
    vix_id: str
    vix3m_id: str
    backwardation_gate: float = 1.00
    interval_secs: int = 15
    max_quote_age_secs: float = 90.0
    use_quotes: bool = True
    use_bars: bool = True
    valkey_url: str = DEFAULT_VALKEY_URL


@dataclass(frozen=True)
class RegimeState:
    vix: float
    vix3m: float
    ratio: float
    is_stale: bool
    max_age_secs: float
    short_vol_allowed: bool


def regime_state(
    vix: LatestValue,
    vix3m: LatestValue,
    event_ts_ns: int,
    max_quote_age_secs: float,
    backwardation_gate: float,
) -> RegimeState | None:
    """Ratio and short-vol permission; None until both levels are known. Stale feeds never allow short vol."""
    if vix.value is None or vix3m.value is None or vix3m.value <= 0.0:
        return None
    vix_age = (event_ts_ns - vix.ts_ns) / NS_PER_SEC
    vix3m_age = (event_ts_ns - vix3m.ts_ns) / NS_PER_SEC
    max_age = max(vix_age, vix3m_age)
    is_stale = max_age > max_quote_age_secs
    ratio = vix.value / vix3m.value
    return RegimeState(
        vix=vix.value,
        vix3m=vix3m.value,
        ratio=ratio,
        is_stale=is_stale,
        max_age_secs=max_age,
        short_vol_allowed=(not is_stale) and (ratio <= backwardation_gate),
    )


def regime_payload(state: RegimeState, wall_ns: int, event_ts_ns: int) -> dict[str, Any]:
    return {
        "vix": round(state.vix, 2),
        "vix3m": round(state.vix3m, 2),
        "ratio": round(state.ratio, 4),
        "short_vol_allowed": state.short_vol_allowed,
        "is_stale": state.is_stale,
        "max_age_secs": round(state.max_age_secs, 2),
        "ts_ns": wall_ns,
        "event_ts_ns": event_ts_ns,
    }


class RegimeService:
    """Keeps the newest VIX and VIX3M levels (from quotes or bars) and publishes mft:regime:latest.

    The ratio is only meaningful for the indices themselves (^VIX.CBOE / ^VIX3M.CBOE). Cash indices carry
    no two-sided quotes, so levels may also come from 1-minute LAST bars.
    """

    def __init__(self, settings: RegimeSettings, store: JsonStore | None = None, log: LogLike | None = None) -> None:
        self.settings = settings
        self.store = store if store is not None else NullStore()
        self.log = log if log is not None else logging.getLogger(__name__)
        self.vix = LatestValue()
        self.vix3m = LatestValue()

    def update_level(self, instrument_id: str, level: float, ts_ns: int) -> None:
        # Quotes and bars can both feed a level; keep whichever observation is newest
        if instrument_id == self.settings.vix_id:
            self.vix.update(level, ts_ns)
        elif instrument_id == self.settings.vix3m_id:
            self.vix3m.update(level, ts_ns)

    def on_quote(self, quote: QuoteObs) -> None:
        mid = quote.mid
        if mid is not None:
            self.update_level(quote.instrument_id, mid, quote.ts_ns)

    def on_bar(self, bar: BarObs) -> None:
        if bar.close > 0.0:
            self.update_level(bar.instrument_id, bar.close, bar.ts_ns)

    def evaluate(self, event_ts_ns: int, wall_ns: int) -> RegimeState | None:
        s = self.settings
        state = regime_state(self.vix, self.vix3m, event_ts_ns, s.max_quote_age_secs, s.backwardation_gate)
        if state is None:
            self.log.info(f"MacroRegimeActor timer tick: vix={self.vix.value}, vix3m={self.vix3m.value}")
            return None
        self.store.set_json(keys.REGIME_LATEST, regime_payload(state, wall_ns, event_ts_ns))
        self.log.info(
            f"Regime VIX={state.vix:.2f} VIX3M={state.vix3m:.2f} "
            f"ratio={state.ratio:.3f} stale={state.is_stale} short_vol_allowed={state.short_vol_allowed}"
        )
        return state
