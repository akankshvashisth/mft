import time

from nautilus_trader.common import DataActor, DataActorConfig
from nautilus_trader.model import Bar, InstrumentId, QuoteTick, TradeTick

from src.adapters.extract import plain_settings
from src.adapters.nautilus import bar_obs, minute_bar_type, quote_obs, trade_obs
from src.adapters.ports import ActorPorts, PortsLog
from src.common.constants import DEFAULT_VALKEY_URL
from src.common.store import make_store
from src.services.futures_basis import FuturesBasisService, FuturesSettings


class FuturesContextActorConfig(DataActorConfig):
    def __init__(
        self,
        spy_id: InstrumentId,
        mes_id: InstrumentId,
        spx_id: InstrumentId | None = None,
        mnq_id: InstrumentId | None = None,
        qqq_id: InstrumentId | None = None,
        risk_free_rate: float = FuturesSettings.risk_free_rate,
        interval_secs: int = FuturesSettings.interval_secs,
        max_quote_age_secs: float = FuturesSettings.max_quote_age_secs,
        max_surface_age_secs: float = FuturesSettings.max_surface_age_secs,
        max_ratio_age_secs: float = FuturesSettings.max_ratio_age_secs,
        valkey_url: str = DEFAULT_VALKEY_URL,
    ) -> None:
        super().__init__()
        self.spy_id = spy_id
        self.mes_id = mes_id
        self.spx_id = spx_id
        self.mnq_id = mnq_id
        self.qqq_id = qqq_id
        self.risk_free_rate = risk_free_rate
        self.interval_secs = interval_secs
        self.max_quote_age_secs = max_quote_age_secs
        self.max_surface_age_secs = max_surface_age_secs
        self.max_ratio_age_secs = max_ratio_age_secs
        self.valkey_url = valkey_url


class FuturesContextActor(DataActor):
    """Streams front-month CME futures (MES/MNQ) to benchmark the SVI parity forward & carry.

    MES is converted to SPY units with a live ^SPX.CBOE / SPY ratio taken from time-aligned 1-minute
    bars. Without that ratio the cross-check is published as unavailable rather than anchored to the
    parity forward it is supposed to validate (logic: FuturesBasisService).
    """

    def __init__(self, config: FuturesContextActorConfig) -> None:
        super().__init__(config)
        self.ports = ActorPorts(self)
        self.core = FuturesBasisService(
            plain_settings(config, FuturesSettings), store=make_store(config.valkey_url), log=PortsLog(self)
        )
        self.wall_clock_ns = time.time_ns

    def on_start(self) -> None:
        s = self.core.settings
        subs = []
        for iid in self.core.quote_ids():
            if self.ports.instrument(iid) is not None:
                self.ports.subscribe_quotes(iid)
                subs.append(iid)

        if s.spx_id is not None and self.ports.instrument(s.spx_id) is not None:
            # Cash indices have no two-sided quotes: pair SPX and SPY minute bars for the conversion ratio
            self.ports.subscribe_bars(minute_bar_type(s.spx_id))
            self.ports.subscribe_bars(minute_bar_type(s.spy_id))
            subs.append(f"{s.spx_id} (1-min bars)")
        else:
            self.ports.warning(
                "FuturesContextActor has no SPX index feed; the parity-vs-futures cross-check will be unavailable"
            )

        self.ports.set_timer("futures_context_eval", s.interval_secs, self.on_timer)
        self.ports.info(f"FuturesContextActor subscribed to {', '.join(subs)}")

    def on_quote(self, tick: QuoteTick) -> None:
        self.core.on_quote(quote_obs(tick))

    def on_trade(self, tick: TradeTick) -> None:
        self.core.on_trade(trade_obs(tick))

    def on_bar(self, bar: Bar) -> None:
        self.core.on_bar(bar_obs(bar), wall_ns=self.wall_clock_ns())

    def on_timer(self, event) -> None:
        mes = self.ports.instrument(self.core.settings.mes_id)
        mes_expiry_ns = int(getattr(mes, "expiration_ns", 0) or 0) if mes is not None else 0
        self.core.evaluate(wall_ns=self.wall_clock_ns(), event_ts_ns=int(event.ts_event), mes_expiry_ns=mes_expiry_ns)
