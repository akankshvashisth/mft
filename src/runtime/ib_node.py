"""Interactive Brokers LiveNode configuration shared by run_paper.py and probe_instrument.py."""
from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from nautilus_trader.adapters.interactive_brokers import (
    InteractiveBrokersDataClientConfig,
    InteractiveBrokersDataClientFactory,
    InteractiveBrokersExecutionClientConfig,
    InteractiveBrokersExecutionClientFactory,
    InteractiveBrokersInstrumentProviderConfig,
    MarketDataType,
    SymbologyMethod,
)
from nautilus_trader.config import LiveExecutionEngineConfig, LiveNodeConfig
from nautilus_trader.live import LiveNodeBuilder
from nautilus_trader.model import InstrumentId, TraderId

IB_CLIENT = "IB"
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 4002   # IB Gateway paper trading


def instrument_provider_config(
    load_ids: Iterable[Any], load_contracts: list[dict] | None = None
) -> InteractiveBrokersInstrumentProviderConfig:
    kwargs: dict[str, Any] = {
        "symbology_method": SymbologyMethod.SIMPLIFIED,
        "load_ids": frozenset(i if isinstance(i, InstrumentId) else InstrumentId.from_str(str(i)) for i in load_ids),
    }
    if load_contracts:
        kwargs["load_contracts"] = load_contracts
    return InteractiveBrokersInstrumentProviderConfig(**kwargs)


def data_client_config(
    provider: InteractiveBrokersInstrumentProviderConfig,
    host: str = DEFAULT_HOST,
    port: int = DEFAULT_PORT,
    client_id: int = 1,
    market_data_type: Any = MarketDataType.DELAYED_FROZEN,
) -> InteractiveBrokersDataClientConfig:
    return InteractiveBrokersDataClientConfig(
        host=host,
        port=port,
        client_id=client_id,
        instrument_provider=provider,
        market_data_type=market_data_type,
    )


def exec_client_config(
    provider: InteractiveBrokersInstrumentProviderConfig,
    account_id: str,
    host: str = DEFAULT_HOST,
    port: int = DEFAULT_PORT,
    client_id: int = 2,
) -> InteractiveBrokersExecutionClientConfig:
    return InteractiveBrokersExecutionClientConfig(
        host=host,
        port=port,
        client_id=client_id,
        account_id=account_id,
        instrument_provider=provider,
    )


def live_node_config(
    trader_name: str,
    data_config: InteractiveBrokersDataClientConfig,
    exec_config: InteractiveBrokersExecutionClientConfig | None = None,
) -> LiveNodeConfig:
    kwargs: dict[str, Any] = {"trader_id": TraderId(trader_name), "data_clients": {IB_CLIENT: data_config}}
    if exec_config is not None:
        kwargs["exec_clients"] = {IB_CLIENT: exec_config}
        kwargs["exec_engine"] = LiveExecutionEngineConfig(reconciliation=True)
    return LiveNodeConfig(**kwargs)


def build_live_node(
    trader_name: str,
    data_config: InteractiveBrokersDataClientConfig,
    exec_config: InteractiveBrokersExecutionClientConfig | None = None,
) -> Any:
    """LiveNode with the IB data client (and the IB execution client with reconciliation when given)."""
    factories: dict[str, Any] = {"data_factories": {IB_CLIENT: InteractiveBrokersDataClientFactory()}}
    if exec_config is not None:
        factories["exec_factories"] = {IB_CLIENT: InteractiveBrokersExecutionClientFactory()}
    config = live_node_config(trader_name, data_config, exec_config)
    return LiveNodeBuilder.from_config(trader_name, config, **factories).build()
