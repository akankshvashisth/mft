import time
from datetime import timedelta
import json
import redis
from nautilus_trader.common import DataActor, DataActorConfig
from nautilus_trader.model import CustomData, DataType, InstrumentId, QuoteTick
from src.models.vol_surface import MacroRegimeSignal


class MacroRegimeActorConfig(DataActorConfig):
    def __init__(
        self,
        vix_id: InstrumentId,
        vix3m_id: InstrumentId,
        backwardation_gate: float = 1.00,
        interval_secs: int = 15,
        max_quote_age_secs: float = 90.0,
        valkey_url: str = "redis://127.0.0.1:6379/0",
    ) -> None:
        super().__init__()
        self.vix_id = vix_id
        self.vix3m_id = vix3m_id
        self.backwardation_gate = backwardation_gate
        self.interval_secs = interval_secs
        self.max_quote_age_secs = max_quote_age_secs
        self.valkey_url = valkey_url


class MacroRegimeActor(DataActor):
    """Monitors volatility term structure with feed-staleness gating."""

    def __init__(self, config: MacroRegimeActorConfig) -> None:
        super().__init__(config)
        self.vix_mid: float | None = None
        self.vix3m_mid: float | None = None
        self.vix_ts_ns: int = 0
        self.vix3m_ts_ns: int = 0
        self.rdb = redis.from_url(config.valkey_url) if config.valkey_url else None

    def on_start(self) -> None:
        self.subscribe_quotes(self.config.vix_id)
        self.subscribe_quotes(self.config.vix3m_id)
        self.clock.set_timer(
            "regime_eval",
            timedelta(seconds=self.config.interval_secs),
            callback=self.on_timer,
        )
        self.log.info(
            f"MacroRegimeActor subscribed to {self.config.vix_id} & {self.config.vix3m_id}"
        )

    def on_quote(self, tick: QuoteTick) -> None:
        bid = float(tick.bid_price)
        ask = float(tick.ask_price)
        if bid <= 0.0 or ask <= 0.0 or ask < bid:
            return
        mid = 0.5 * (bid + ask)
        ts_ns = int(tick.ts_event)
        if tick.instrument_id == self.config.vix_id:
            self.vix_mid = mid
            self.vix_ts_ns = ts_ns
        elif tick.instrument_id == self.config.vix3m_id:
            self.vix3m_mid = mid
            self.vix3m_ts_ns = ts_ns

    def on_timer(self, event) -> None:
        if self.vix_mid is None or self.vix3m_mid is None or self.vix3m_mid <= 0.0:
            self.log.info(
                f"MacroRegimeActor timer tick: vix={self.vix_mid}, vix3m={self.vix3m_mid}"
            )
            return

        now_ns = int(event.ts_event)
        vix_age = (now_ns - self.vix_ts_ns) / 1e9
        vix3m_age = (now_ns - self.vix3m_ts_ns) / 1e9
        is_stale = max(vix_age, vix3m_age) > self.config.max_quote_age_secs

        ratio = self.vix_mid / self.vix3m_mid
        allowed = (not is_stale) and (ratio <= self.config.backwardation_gate)

        signal = MacroRegimeSignal(
            vix=self.vix_mid,
            vix3m=self.vix3m_mid,
            ratio=ratio,
            short_vol_allowed=allowed,
            ts_event=now_ns,
            ts_init=self.clock.timestamp_ns(),
        )
        self.publish_data(
            DataType("MacroRegimeSignal"),
            CustomData(DataType("MacroRegimeSignal"), signal),
        )
        if self.rdb is not None:
            self.rdb.set(
                "mft:regime:latest",
                json.dumps(
                    {
                        "vix": round(self.vix_mid, 2),
                        "vix3m": round(self.vix3m_mid, 2),
                        "ratio": round(ratio, 4),
                        "short_vol_allowed": allowed,
                        "is_stale": is_stale,
                        "max_age_secs": round(max(vix_age, vix3m_age), 2),
                        "ts_ns": time.time_ns(),
                        "event_ts_ns": now_ns,
                    }
                ),
            )
        self.log.info(
            f"Regime VIX={self.vix_mid:.2f} VIX3M={self.vix3m_mid:.2f} "
            f"ratio={ratio:.3f} stale={is_stale} short_vol_allowed={allowed}"
        )
