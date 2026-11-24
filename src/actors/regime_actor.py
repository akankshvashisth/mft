import time

from nautilus_trader.common import DataActor, DataActorConfig
from nautilus_trader.model import Bar, InstrumentId, QuoteTick

from src.adapters.extract import plain_settings
from src.adapters.nautilus import bar_obs, minute_bar_type, quote_obs
from src.adapters.ports import ActorPorts, PortsLog
from src.common.constants import DEFAULT_VALKEY_URL
from src.common.store import make_store
from src.models.vol_surface import MacroRegimeSignal
from src.services.regime import RegimeService, RegimeSettings


class MacroRegimeActorConfig(DataActorConfig):
    def __init__(
        self,
        vix_id: InstrumentId,
        vix3m_id: InstrumentId,
        backwardation_gate: float = RegimeSettings.backwardation_gate,
        interval_secs: int = RegimeSettings.interval_secs,
        max_quote_age_secs: float = RegimeSettings.max_quote_age_secs,
        use_quotes: bool = RegimeSettings.use_quotes,
        use_bars: bool = RegimeSettings.use_bars,
        valkey_url: str = DEFAULT_VALKEY_URL,
    ) -> None:
        super().__init__()
        self.vix_id = vix_id
        self.vix3m_id = vix3m_id
        self.backwardation_gate = backwardation_gate
        self.interval_secs = interval_secs
        self.max_quote_age_secs = max_quote_age_secs
        self.use_quotes = use_quotes
        self.use_bars = use_bars
        self.valkey_url = valkey_url


class MacroRegimeActor(DataActor):
    """Monitors the VIX / VIX3M index term structure with feed-staleness gating (logic: RegimeService).

    The ratio is only meaningful for the indices themselves (^VIX.CBOE / ^VIX3M.CBOE). Cash indices
    carry no two-sided quotes, so levels are also taken from 1-minute LAST bars when use_bars is set.
    """

    def __init__(self, config: MacroRegimeActorConfig) -> None:
        super().__init__(config)
        self.ports = ActorPorts(self)
        self.core = RegimeService(
            plain_settings(config, RegimeSettings), store=make_store(config.valkey_url), log=PortsLog(self)
        )
        self.wall_clock_ns = time.time_ns

    def on_start(self) -> None:
        s = self.core.settings
        for iid in (s.vix_id, s.vix3m_id):
            if s.use_quotes:
                self.ports.subscribe_quotes(iid)
            if s.use_bars:
                self.ports.subscribe_bars(minute_bar_type(iid))
        self.ports.set_timer("regime_eval", s.interval_secs, self.on_timer)
        self.ports.info(
            f"MacroRegimeActor subscribed to {s.vix_id} & {s.vix3m_id} (quotes={s.use_quotes}, bars={s.use_bars})"
        )

    def on_quote(self, tick: QuoteTick) -> None:
        self.core.on_quote(quote_obs(tick))

    def on_bar(self, bar: Bar) -> None:
        self.core.on_bar(bar_obs(bar))

    def on_timer(self, event) -> None:
        event_ts_ns = int(event.ts_event)
        state = self.core.evaluate(event_ts_ns=event_ts_ns, wall_ns=self.wall_clock_ns())
        if state is None:
            return
        signal = MacroRegimeSignal(
            vix=state.vix,
            vix3m=state.vix3m,
            ratio=state.ratio,
            short_vol_allowed=state.short_vol_allowed,
            ts_event=event_ts_ns,
            ts_init=self.ports.now_ns(),
        )
        self.ports.publish_custom("MacroRegimeSignal", signal)
