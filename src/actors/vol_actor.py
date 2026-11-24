import time

from nautilus_trader.common import DataActor, DataActorConfig
from nautilus_trader.model import Bar, InstrumentId, OptionContract, QuoteTick

from src.adapters.extract import option_spec_from_instrument, plain_settings
from src.adapters.nautilus import bar_obs, quote_obs
from src.adapters.ports import ActorPorts, PortsLog
from src.common.constants import DEFAULT_VALKEY_URL
from src.common.store import make_store
from src.models.vol_surface import SviSliceParams
from src.services.vol_surface import VolSurfaceService, VolSurfaceSettings


class FastVolActorConfig(DataActorConfig):
    def __init__(
        self,
        underlying_id: InstrumentId,
        option_ids: tuple[InstrumentId, ...] = (),
        strikes: tuple[float, ...] = (),
        is_calls: tuple[bool, ...] = (),
        ttm_years: float = VolSurfaceSettings.ttm_years,
        risk_free_rate: float = VolSurfaceSettings.risk_free_rate,
        dividend_yield: float = VolSurfaceSettings.dividend_yield,
        interval_secs: int = VolSurfaceSettings.interval_secs,
        strike_min: float = VolSurfaceSettings.strike_min,
        strike_max: float = VolSurfaceSettings.strike_max,
        max_rel_spread: float = VolSurfaceSettings.max_rel_spread,
        min_mid_price: float = VolSurfaceSettings.min_mid_price,
        min_open_interest: int = VolSurfaceSettings.min_open_interest,
        atm_overlap_pct: float = VolSurfaceSettings.atm_overlap_pct,
        max_quote_age_secs: float = VolSurfaceSettings.max_quote_age_secs,
        valkey_url: str = DEFAULT_VALKEY_URL,
    ) -> None:
        super().__init__()
        self.underlying_id = underlying_id
        self.option_ids = option_ids
        self.strikes = strikes
        self.is_calls = is_calls
        self.ttm_years = ttm_years
        self.risk_free_rate = risk_free_rate
        self.dividend_yield = dividend_yield
        self.interval_secs = interval_secs
        self.strike_min = strike_min
        self.strike_max = strike_max
        self.max_rel_spread = max_rel_spread
        self.min_mid_price = min_mid_price
        self.min_open_interest = min_open_interest
        self.atm_overlap_pct = atm_overlap_pct
        self.max_quote_age_secs = max_quote_age_secs
        self.valkey_url = valkey_url


class FastVolActor(DataActor):
    """Full-hygiene IV inversion + spread-weighted warm-start SVI surface calibrator (logic: VolSurfaceService)."""

    def __init__(self, config: FastVolActorConfig) -> None:
        super().__init__(config)
        self.ports = ActorPorts(self)
        self.core = VolSurfaceService(
            plain_settings(config, VolSurfaceSettings), store=make_store(config.valkey_url), log=PortsLog(self)
        )
        self.wall_clock_ns = time.time_ns

    def on_start(self) -> None:
        s = self.core.settings
        self.ports.subscribe_quotes(s.underlying_id)
        cached_options = [
            option_spec_from_instrument(inst) for inst in self.ports.instruments() if isinstance(inst, OptionContract)
        ]
        self.core.configure(cached_options)
        for option_id in self.core.book.option_ids:
            self.ports.subscribe_quotes(option_id)
        self.ports.set_timer("fast_vol_fit", s.interval_secs, self.on_timer)
        self.ports.info(
            f"FastVolActor watching {s.underlying_id} + {len(self.core.book.option_ids)} standard 100-mult options"
        )

    def on_quote(self, tick: QuoteTick) -> None:
        self.core.on_quote(quote_obs(tick))

    def on_bar(self, bar: Bar) -> None:
        self.core.on_bar(bar_obs(bar))

    def on_timer(self, event) -> None:
        event_ts_ns = int(event.ts_event)
        fit = self.core.evaluate(event_ts_ns=event_ts_ns, wall_ns=self.wall_clock_ns())
        if fit is None:
            return
        snapshot = SviSliceParams(
            underlying_id=self.config.underlying_id,
            expiry_ns=fit.expiry_ns,
            forward=fit.forward,
            atm_iv=fit.atm_iv,
            skew_25d=fit.skew_25d,
            arb_free=fit.arb_free,
            quote_count=fit.quote_count,
            ts_event=event_ts_ns,
            ts_init=self.ports.now_ns(),
        )
        self.ports.publish_custom("SviSliceParams", snapshot)
