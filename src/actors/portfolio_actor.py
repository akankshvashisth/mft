import time

from nautilus_trader.common import DataActor, DataActorConfig
from nautilus_trader.model import InstrumentId

from src.adapters.book_market import NautilusBookMarket
from src.adapters.extract import plain_settings, position_view
from src.adapters.ports import ActorPorts, PortsLog, PortsProxy
from src.common.constants import DEFAULT_VALKEY_URL
from src.common.store import make_store
from src.services.portfolio import PortfolioService, PortfolioSettings


class PortfolioAnalyticsActorConfig(DataActorConfig):
    def __init__(
        self,
        benchmark_id: InstrumentId,
        allocated_capital_usd: float = PortfolioSettings.allocated_capital_usd,
        risk_free_rate: float = PortfolioSettings.risk_free_rate,
        interval_secs: int = PortfolioSettings.interval_secs,
        max_mark_quote_age_secs: float = PortfolioSettings.max_mark_quote_age_secs,
        valkey_url: str = DEFAULT_VALKEY_URL,
    ) -> None:
        super().__init__()
        self.benchmark_id = benchmark_id
        self.allocated_capital_usd = allocated_capital_usd
        self.risk_free_rate = risk_free_rate
        self.interval_secs = interval_secs
        self.max_mark_quote_age_secs = max_mark_quote_age_secs
        self.valkey_url = valkey_url


class PortfolioAnalyticsActor(DataActor):
    """Real-time beta-weighted book Greeks, Spot x Vol stress matrix, P&L attribution, and sleeve drawdown.

    Logic: PortfolioService, reading instruments and quotes through the actor ports.
    """

    def __init__(self, config: PortfolioAnalyticsActorConfig) -> None:
        super().__init__(config)
        self.ports = ActorPorts(self)
        self.core = PortfolioService(
            plain_settings(config, PortfolioSettings),
            store=make_store(config.valkey_url),
            log=PortsLog(self),
            market=NautilusBookMarket(PortsProxy(self)),
        )
        self.wall_clock_ns = time.time_ns

    def on_start(self) -> None:
        # The in-memory cache only knows this session's realized P&L, so sleeve equity state is carried in Valkey
        self.core.restore()
        s = self.core.settings
        self.ports.set_timer("portfolio_analytics_eval", s.interval_secs, self.on_timer)
        self.ports.info(
            f"PortfolioAnalyticsActor started (benchmark={s.benchmark_id}, "
            f"sleeve=${s.allocated_capital_usd:,.0f}, realized_carry=${self.core.sleeve.prior_carry_usd:,.2f}, "
            f"hwm=${self.core.sleeve.high_water_mark:,.2f})"
        )

    def on_timer(self, event) -> None:
        positions = [position_view(p) for p in self.ports.positions_open()]
        closed_realized_usd = sum(float(p.realized_pnl) for p in self.ports.positions_closed())
        self.core.evaluate(positions, closed_realized_usd, self.wall_clock_ns())
