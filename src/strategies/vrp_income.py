from __future__ import annotations

import time

from nautilus_trader.config import StrategyConfig
from nautilus_trader.model import InstrumentId, QuoteTick
from nautilus_trader.trading import Strategy

from src.adapters.extract import plain_settings
from src.adapters.nautilus import quote_obs
from src.adapters.order_gateway import NautilusMarketView, NautilusOrderGateway
from src.adapters.ports import PortsLog, PortsProxy, StrategyPorts
from src.common.constants import DEFAULT_VALKEY_URL
from src.common.store import make_store
from src.common.timeutil import ny_date
from src.execution.passive_combo import PassiveComboExecutor, PassiveExecutionConfig
from src.risk.pre_trade_gate import GateConfig, PreTradeGate
from src.risk.scenario_pricer import ScenarioPricer
from src.strategies.vrp_engine import VrpEngine, VrpSettings


class VrpIncomeStrategyConfig(StrategyConfig):
    def __init__(
        self,
        underlying_id: InstrumentId,
        option_ids: tuple[InstrumentId, ...],
        paper_routing_enabled: bool = VrpSettings.paper_routing_enabled,
        eval_interval_secs: int = VrpSettings.eval_interval_secs,
        sleeve_equity_usd: float = VrpSettings.sleeve_equity_usd,
        profit_target_fraction: float = VrpSettings.profit_target_fraction,
        min_exit_dte: int = VrpSettings.min_exit_dte,
        valkey_url: str = DEFAULT_VALKEY_URL,
    ) -> None:
        super().__init__()
        self.underlying_id = underlying_id
        self.option_ids = option_ids
        self.paper_routing_enabled = paper_routing_enabled
        self.eval_interval_secs = eval_interval_secs
        self.sleeve_equity_usd = sleeve_equity_usd
        self.profit_target_fraction = profit_target_fraction
        self.min_exit_dte = min_exit_dte
        self.valkey_url = valkey_url


class VrpIncomeStrategy(Strategy):
    """Multi-actor VRP & relative-value strategy with PreTradeGate + Passive OptionSpread execution.

    Spread instruments carry ratio +1 on the leg sold at entry and -1 on the protective wing, so the
    spread price is a positive credit: entries SELL it, exits BUY it back. At most one child clip works
    at a time, and every further clip needs a fresh gate approval against the refreshed book.
    The logic lives in VrpEngine; this shell converts Nautilus events and forwards them.
    """

    def __init__(self, config: VrpIncomeStrategyConfig) -> None:
        super().__init__(config)
        self.ports = StrategyPorts(self)
        store = make_store(config.valkey_url)
        self.vrp_engine = VrpEngine(
            plain_settings(config, VrpSettings),
            gate=PreTradeGate(
                config=GateConfig(sleeve_equity_usd=config.sleeve_equity_usd),
                valkey_url=config.valkey_url,
                store=store,
            ),
            pricer=ScenarioPricer(n_scenarios=5_000, commission_per_leg=0.65, slippage_fraction=0.50),
            executor=PassiveComboExecutor(PassiveExecutionConfig()),
            store=store,
            gateway=NautilusOrderGateway(PortsProxy(self)),
            market=NautilusMarketView(PortsProxy(self)),
            log=PortsLog(self),
        )
        self.wall_clock_ns = time.time_ns

    def on_start(self) -> None:
        s = self.vrp_engine.settings
        self.ports.subscribe_quotes(s.underlying_id)
        restored = self.vrp_engine.start()
        self.ports.set_timer("vrp_strategy_eval", s.eval_interval_secs, self.on_timer)
        self.ports.info(
            f"VrpIncomeStrategy started in {s.mode} mode watching {s.underlying_id} "
            f"+ {len(s.option_ids)} options ({restored} open spreads restored)"
        )

    def on_stop(self) -> None:
        self.vrp_engine.stop()

    def on_quote(self, tick: QuoteTick) -> None:
        self.vrp_engine.on_quote(quote_obs(tick))

    # ------------------------------------------------------------------------------
    # Order lifecycle
    # ------------------------------------------------------------------------------

    def on_order_filled(self, event) -> None:
        commission = getattr(event, "commission", None)
        self.vrp_engine.on_fill(
            str(event.client_order_id),
            float(event.last_qty),
            float(event.last_px),
            float(commission) if commission is not None else None,
        )

    def on_order_canceled(self, event) -> None:
        self.vrp_engine.on_terminal(str(event.client_order_id), "CANCELLED")

    def on_order_expired(self, event) -> None:
        self.vrp_engine.on_terminal(str(event.client_order_id), "EXPIRED")

    def on_order_rejected(self, event) -> None:
        self.vrp_engine.on_terminal(str(event.client_order_id), "REJECTED", str(getattr(event, "reason", "")))

    def on_order_denied(self, event) -> None:
        self.vrp_engine.on_terminal(str(event.client_order_id), "DENIED", str(getattr(event, "reason", "")))

    def on_order_modify_rejected(self, event) -> None:
        self.vrp_engine.on_modify_rejected(str(event.client_order_id))

    def on_timer(self, event) -> None:
        self.vrp_engine.on_timer(
            event_ts_ns=int(event.ts_event),
            wall_ns=self.wall_clock_ns(),
            today=ny_date(self.ports.now_ns()),
        )
