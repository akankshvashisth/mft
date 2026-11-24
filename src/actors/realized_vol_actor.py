import time

from nautilus_trader.common import DataActor, DataActorConfig
from nautilus_trader.model import Bar, InstrumentId, QuoteTick

from src.adapters.extract import plain_settings
from src.adapters.history import DailyHistoryLoader
from src.adapters.nautilus import bar_obs, minute_bar_type, quote_obs
from src.adapters.ports import ActorPorts, PortsLog
from src.common.constants import DEFAULT_VALKEY_URL
from src.common.store import make_store
from src.models.vol_surface import RealizedVolSnapshot
from src.services.realized_vol import RealizedVolService, RealizedVolSettings


class RealizedVolActorConfig(DataActorConfig):
    def __init__(
        self,
        primary_id: InstrumentId,
        universe_ids: tuple[InstrumentId, ...] = (),
        catalog_path: str = RealizedVolSettings.catalog_path,
        interval_secs: int = RealizedVolSettings.interval_secs,
        trading_days: int = RealizedVolSettings.trading_days,
        valkey_url: str = DEFAULT_VALKEY_URL,
    ) -> None:
        super().__init__()
        self.primary_id = primary_id
        self.universe_ids = universe_ids or (primary_id,)
        self.catalog_path = catalog_path
        self.interval_secs = interval_secs
        self.trading_days = trading_days
        self.valkey_url = valkey_url


class RealizedVolActor(DataActor):
    """Stateful P-measure realized volatility, jump, and cross-asset correlation engine (logic: RealizedVolService)."""

    def __init__(self, config: RealizedVolActorConfig) -> None:
        super().__init__(config)
        self.ports = ActorPorts(self)
        self.core = RealizedVolService(
            plain_settings(config, RealizedVolSettings), store=make_store(config.valkey_url), log=PortsLog(self)
        )
        self.history_loader_factory = DailyHistoryLoader
        self.wall_clock_ns = time.time_ns

    def on_start(self) -> None:
        s = self.core.settings
        loaded = self.core.warm_start(self.history_loader_factory(s.catalog_path, log=self.core.log))

        # Live quotes & 1-min bars for the primary instrument (and any cached universe instruments)
        cached_ids = {str(inst.id) for inst in self.ports.instruments()}
        for iid in self.core.universe:
            if iid == s.primary_id or iid in cached_ids:
                self.ports.subscribe_quotes(iid)
                self.ports.subscribe_bars(minute_bar_type(iid))

        self.ports.set_timer("realized_vol_eval", s.interval_secs, self.on_timer)
        self.ports.info(
            f"RealizedVolActor warm-started {loaded}/{len(self.core.universe)} universe symbols from {s.catalog_path}"
        )

    def on_quote(self, tick: QuoteTick) -> None:
        self.core.on_quote(quote_obs(tick))

    def on_bar(self, bar: Bar) -> None:
        self.core.on_bar(bar_obs(bar))

    def on_timer(self, event) -> None:
        event_ts_ns = int(event.ts_event)
        wall_ns = self.wall_clock_ns()
        snap = self.core.evaluate(event_ts_ns=event_ts_ns, wall_ns=wall_ns)
        if snap is None:
            return
        self.ports.publish_custom(
            "RealizedVolSnapshot",
            RealizedVolSnapshot(
                underlying_id=self.config.primary_id,
                spot=snap.spot,
                rv_5d=snap.rv_5d,
                rv_10d=snap.rv_10d,
                rv_20d=snap.rv_20d,
                rv_60d=snap.rv_60d,
                rv_percentile=snap.rv_percentile,
                jump_ratio=snap.jump_ratio,
                avg_pair_corr=snap.avg_pair_corr,
                ts_event=event_ts_ns,
                ts_init=wall_ns,
            ),
        )
