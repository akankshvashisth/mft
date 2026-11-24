import os
import sys
from pathlib import Path
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nautilus_trader.adapters.interactive_brokers import (
    InteractiveBrokersDataClientConfig,
    InteractiveBrokersDataClientFactory,
    InteractiveBrokersExecutionClientConfig,
    InteractiveBrokersExecutionClientFactory,
    InteractiveBrokersInstrumentProviderConfig,
    MarketDataType,
    SymbologyMethod,
)
from nautilus_trader.config import (
    LiveExecutionEngineConfig,
    LiveNodeConfig,
)
from nautilus_trader.live import LiveNodeBuilder
from nautilus_trader.model import (
    InstrumentId,
    TraderId,
)
from src.actors.regime_actor import MacroRegimeActor, MacroRegimeActorConfig
from src.actors.vol_actor import FastVolActor, FastVolActorConfig
from src.actors.realized_vol_actor import RealizedVolActor, RealizedVolActorConfig
from src.actors.surface_metrics_actor import SurfaceRegimeMetricsActor, SurfaceRegimeMetricsActorConfig
from src.actors.portfolio_actor import PortfolioAnalyticsActor, PortfolioAnalyticsActorConfig
from src.actors.futures_actor import FuturesContextActor, FuturesContextActorConfig
from src.strategies.vrp_income import VrpIncomeStrategy, VrpIncomeStrategyConfig

load_dotenv()


def build_spy_otm_slice(
    expiry_yymmdd: str = "261218",
) -> tuple[tuple[InstrumentId, ...], tuple[float, ...], tuple[bool, ...]]:
    """Builds a 15-strike spliced OTM slice (7 OTM puts + 8 OTM calls) centered near SPY ~707.45."""
    put_strikes = [675.0, 680.0, 685.0, 690.0, 695.0, 700.0, 705.0]
    call_strikes = [710.0, 715.0, 720.0, 725.0, 730.0, 735.0, 740.0, 745.0]

    option_ids: list[InstrumentId] = []
    strikes: list[float] = []
    is_calls: list[bool] = []

    for k in put_strikes:
        strike_str = f"{int(round(k * 1000)):08d}"
        option_ids.append(InstrumentId.from_str(f"SPY {expiry_yymmdd}P{strike_str}.SMART"))
        strikes.append(k)
        is_calls.append(False)

    for k in call_strikes:
        strike_str = f"{int(round(k * 1000)):08d}"
        option_ids.append(InstrumentId.from_str(f"SPY {expiry_yymmdd}C{strike_str}.SMART"))
        strikes.append(k)
        is_calls.append(True)

    return tuple(option_ids), tuple(strikes), tuple(is_calls)


def main() -> None:
    spy_id = InstrumentId.from_str("SPY.ARCA")
    vix_id = InstrumentId.from_str("VIXY.ARCA")
    vix3m_id = InstrumentId.from_str("VIXM.ARCA")
    option_ids, strikes, is_calls = build_spy_otm_slice("261218")

    # Step A: test spy_id alone first; once confirmed, add vix_id, vix3m_id, and option_ids one group at a time
    all_ids = frozenset([spy_id, vix_id, vix3m_id])
    print(f"Loading {len(all_ids)} total IBKR instruments (well under 100-line cap)...")

    option_contracts = [
        *(
            {
                "secType": "OPT",
                "symbol": "SPY",
                "exchange": "SMART",
                "lastTradeDateOrContractMonth": "20261218",
                "strike": float(k),
                "right": "P",
            }
            for k in range(685, 725, 5)
        ),
        *(
            {
                "secType": "OPT",
                "symbol": "SPY",
                "exchange": "SMART",
                "lastTradeDateOrContractMonth": "20261218",
                "strike": float(k),
                "right": "C",
            }
            for k in range(720, 760, 5)
        ),
        {
            "secType": "FUT",
            "symbol": "MES",
            "exchange": "CME",
            "currency": "USD",
            "lastTradeDateOrContractMonth": "202612",
        },
        {
            "secType": "FUT",
            "symbol": "MNQ",
            "exchange": "CME",
            "currency": "USD",
            "lastTradeDateOrContractMonth": "202612",
        },
    ]

    provider_cfg = InteractiveBrokersInstrumentProviderConfig(
        symbology_method=SymbologyMethod.SIMPLIFIED,
        load_ids=all_ids,
        load_contracts=option_contracts,
    )

    data_cfg = InteractiveBrokersDataClientConfig(
        host="127.0.0.1",
        port=4002,
        client_id=1,
        instrument_provider=provider_cfg,
        market_data_type=MarketDataType.DELAYED_FROZEN,
    )

    exec_cfg = InteractiveBrokersExecutionClientConfig(
        host="127.0.0.1",
        port=4002,
        client_id=2,
        account_id=os.environ["IB_ACCOUNT_ID"],
        instrument_provider=provider_cfg,
    )

    config = LiveNodeConfig(
        trader_id=TraderId("MFT-PAPER-01"),
        data_clients={"IB": data_cfg},
        exec_clients={"IB": exec_cfg},
        exec_engine=LiveExecutionEngineConfig(reconciliation=True),
    )

    node = LiveNodeBuilder.from_config(
        "MFT-PAPER-01",
        config,
        data_factories={"IB": InteractiveBrokersDataClientFactory()},
        exec_factories={"IB": InteractiveBrokersExecutionClientFactory()},
    ).build()

    node.add_actor(
        MacroRegimeActor(
            MacroRegimeActorConfig(
                vix_id=vix_id,
                vix3m_id=vix3m_id,
                backwardation_gate=1.00,
                interval_secs=15,
            )
        )
    )
    node.add_actor(
        FastVolActor(
            FastVolActorConfig(
                underlying_id=spy_id,
                ttm_years=25.0 / 365.25,
                interval_secs=15,
            )
        )
    )

    try:
        node.add_actor(
            RealizedVolActor(
                RealizedVolActorConfig(
                    primary_id=spy_id,
                    universe_ids=(
                        InstrumentId.from_str("SPY.ARCA"),
                        InstrumentId.from_str("QQQ.NASDAQ"),
                        InstrumentId.from_str("IWM.ARCA"),
                        InstrumentId.from_str("XBI.ARCA"),
                        InstrumentId.from_str("GDX.ARCA"),
                        InstrumentId.from_str("SMH.NASDAQ"),
                        InstrumentId.from_str("TLT.NASDAQ"),
                        InstrumentId.from_str("GLD.ARCA"),
                    ),
                    catalog_path="catalog",
                    interval_secs=15,
                )
            )
        )
        node.add_actor(
            SurfaceRegimeMetricsActor(
                SurfaceRegimeMetricsActorConfig(
                    underlying_id=spy_id,
                    catalog_path="catalog",
                    interval_secs=15,
                )
            )
        )
        node.add_actor(
            FuturesContextActor(
                FuturesContextActorConfig(
                    spy_id=spy_id,
                    mes_id=InstrumentId.from_str("MESZ26.CME"),
                    mnq_id=InstrumentId.from_str("MNQZ26.CME"),
                    interval_secs=15,
                )
            )
        )
        node.add_actor(
            PortfolioAnalyticsActor(
                PortfolioAnalyticsActorConfig(
                    benchmark_id=spy_id,
                    allocated_capital_usd=100_000.0,
                    interval_secs=15,
                )
            )
        )
        node.add_strategy(
            VrpIncomeStrategy(
                VrpIncomeStrategyConfig(
                    underlying_id=spy_id,
                    option_ids=option_ids,
                    paper_routing_enabled=False,
                    eval_interval_secs=15,
                )
            )
        )
        node.run()
    finally:
        node.dispose()


if __name__ == "__main__":
    main()
