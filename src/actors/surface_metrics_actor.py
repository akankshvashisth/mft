import time

from nautilus_trader.common import DataActor, DataActorConfig
from nautilus_trader.indicators import RelativeStrengthIndex, SimpleMovingAverage
from nautilus_trader.model import Bar, InstrumentId

from src.adapters.extract import plain_settings
from src.adapters.history import load_catalog_daily_bars, open_catalog
from src.adapters.nautilus import bar_obs, daily_bar_type
from src.adapters.ports import ActorPorts, PortsLog
from src.common.constants import DEFAULT_VALKEY_URL
from src.common.store import make_store
from src.services.surface_metrics import SurfaceMetricsService, SurfaceMetricsSettings, Technicals

WARM_START_BARS = 100


class SurfaceRegimeMetricsActorConfig(DataActorConfig):
    def __init__(
        self,
        underlying_id: InstrumentId,
        catalog_path: str = SurfaceMetricsSettings.catalog_path,
        interval_secs: int = SurfaceMetricsSettings.interval_secs,
        history_maxlen: int = SurfaceMetricsSettings.history_maxlen,
        valkey_url: str = DEFAULT_VALKEY_URL,
    ) -> None:
        super().__init__()
        self.underlying_id = underlying_id
        self.catalog_path = catalog_path
        self.interval_secs = interval_secs
        self.history_maxlen = history_maxlen
        self.valkey_url = valkey_url


class SurfaceRegimeMetricsActor(DataActor):
    """SVI 25d risk reversal / butterfly, VRP, percentiles and SMA/RSI regime (logic: SurfaceMetricsService)."""

    def __init__(self, config: SurfaceRegimeMetricsActorConfig) -> None:
        super().__init__(config)
        self.ports = ActorPorts(self)
        self.core = SurfaceMetricsService(
            plain_settings(config, SurfaceMetricsSettings), store=make_store(config.valkey_url), log=PortsLog(self)
        )
        # SMA/RSI are daily-close indicators: only new 1-DAY bars of the underlying may update them
        self.sma_20 = SimpleMovingAverage(20)
        self.sma_50 = SimpleMovingAverage(50)
        self.rsi_14 = RelativeStrengthIndex(14)
        self.catalog_factory = open_catalog
        self.wall_clock_ns = time.time_ns

    def _feed_indicators(self, bar: Bar) -> None:
        if not self.core.accept_indicator_bar(bar_obs(bar)):
            return
        self.sma_20.handle_bar(bar)
        self.sma_50.handle_bar(bar)
        self.rsi_14.handle_bar(bar)

    def _warm_start_indicators(self) -> None:
        s = self.core.settings
        try:
            catalog = self.catalog_factory(s.catalog_path)
            for bar in load_catalog_daily_bars(catalog, s.underlying_id)[-WARM_START_BARS:]:
                self._feed_indicators(bar)
        except Exception as exc:
            self.ports.warning(f"SurfaceRegimeMetricsActor indicator warm-start skipped: {exc}")

    def on_start(self) -> None:
        s = self.core.settings
        self._warm_start_indicators()
        self.ports.subscribe_bars(daily_bar_type(s.underlying_id))
        self.ports.set_timer("surface_metrics_eval", s.interval_secs, self.on_timer)
        self.ports.info(
            f"SurfaceRegimeMetricsActor started for {s.underlying_id} "
            f"(SMA20_init={self.sma_20.initialized}, SMA50_init={self.sma_50.initialized})"
        )

    def on_bar(self, bar: Bar) -> None:
        self._feed_indicators(bar)

    def technicals(self) -> Technicals:
        return Technicals(
            sma_20=float(self.sma_20.value) if self.sma_20.initialized else None,
            sma_50=float(self.sma_50.value) if self.sma_50.initialized else None,
            rsi_14=float(self.rsi_14.value) if self.rsi_14.initialized else None,
        )

    def on_timer(self, event) -> None:
        self.core.evaluate(int(event.ts_event), self.wall_clock_ns(), self.technicals())
