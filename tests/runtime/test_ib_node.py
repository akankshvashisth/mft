import pytest

pytest.importorskip("nautilus_trader.adapters.interactive_brokers")
pytestmark = pytest.mark.nautilus

from nautilus_trader.model import InstrumentId, TraderId  # noqa: E402

from src.runtime.ib_node import (  # noqa: E402
    data_client_config,
    exec_client_config,
    instrument_provider_config,
    live_node_config,
)


def test_provider_loads_the_requested_ids():
    provider = instrument_provider_config(["SPY.ARCA", InstrumentId.from_str("^VIX.CBOE")])
    assert provider.load_ids == frozenset({InstrumentId.from_str("SPY.ARCA"), InstrumentId.from_str("^VIX.CBOE")})


def test_client_configs_carry_connection_settings():
    provider = instrument_provider_config(["SPY.ARCA"])
    data = data_client_config(provider, host="10.0.0.5", port=4001, client_id=7)
    execution = exec_client_config(provider, account_id="DU123", client_id=8)

    assert (data.host, data.port, data.client_id) == ("10.0.0.5", 4001, 7)
    assert (execution.host, execution.port, execution.client_id, execution.account_id) == ("127.0.0.1", 4002, 8, "DU123")


def test_node_config_with_and_without_execution():
    provider = instrument_provider_config(["SPY.ARCA"])
    data = data_client_config(provider)

    probe = live_node_config("MFT-PROBE", data)
    assert probe.trader_id == TraderId("MFT-PROBE")
    assert list(probe.data_clients) == ["IB"]
    assert not probe.exec_clients

    paper = live_node_config("MFT-PAPER-01", data, exec_client_config(provider, account_id="DU123"))
    assert list(paper.exec_clients) == ["IB"]
    assert paper.exec_engine.reconciliation is True
