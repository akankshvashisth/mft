import os
import sys
from pathlib import Path
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nautilus_trader.model import InstrumentId  # noqa: E402

from src.actors.futures_actor import FuturesContextActor, FuturesContextActorConfig  # noqa: E402
from src.actors.portfolio_actor import PortfolioAnalyticsActor, PortfolioAnalyticsActorConfig  # noqa: E402
from src.actors.realized_vol_actor import RealizedVolActor, RealizedVolActorConfig  # noqa: E402
from src.actors.regime_actor import MacroRegimeActor, MacroRegimeActorConfig  # noqa: E402
from src.actors.surface_metrics_actor import SurfaceRegimeMetricsActor, SurfaceRegimeMetricsActorConfig  # noqa: E402
from src.actors.vol_actor import FastVolActor, FastVolActorConfig  # noqa: E402
from src.common.timeutil import days_to_years  # noqa: E402
from src.runtime.ib_node import (  # noqa: E402
    build_live_node,
    data_client_config,
    exec_client_config,
    instrument_provider_config,
)
from src.runtime.instruments import (  # noqa: E402
    MES_ID,
    MNQ_ID,
    RV_UNIVERSE,
    SPX_ID,
    SPY_DEC26_SLICE,
    SPY_ID,
    VIX3M_ID,
    VIX_ID,
    ib_future_contract,
    ib_option_contracts,
    slice_option_ids,
)
from src.strategies.vrp_income import VrpIncomeStrategy, VrpIncomeStrategyConfig  # noqa: E402

load_dotenv()


def main() -> None:
    spy_id = InstrumentId.from_str(SPY_ID)
    # Cash indices (IB simplified symbology). The term-structure gate needs VIX / VIX3M levels themselves;
    # the VIXY / VIXM ETF price ratio carries no contango/backwardation information.
    vix_id = InstrumentId.from_str(VIX_ID)
    vix3m_id = InstrumentId.from_str(VIX3M_ID)
    spx_id = InstrumentId.from_str(SPX_ID)
    # One slice definition: the contracts loaded below, the strategy's option ids and the vol actor's strikes
    option_ids = tuple(InstrumentId.from_str(option_id) for option_id in slice_option_ids(SPY_DEC26_SLICE))

    # Step A: test spy_id alone first; once confirmed, add the index ids and option_ids one group at a time
    all_ids = frozenset([spy_id, vix_id, vix3m_id, spx_id])
    print(f"Loading {len(all_ids)} total IBKR instruments (well under 100-line cap)...")

    provider_cfg = instrument_provider_config(
        all_ids,
        [*ib_option_contracts(SPY_DEC26_SLICE), ib_future_contract("MES"), ib_future_contract("MNQ")],
    )
    node = build_live_node(
        "MFT-PAPER-01",
        data_client_config(provider_cfg, client_id=1),
        exec_client_config(provider_cfg, account_id=os.environ["IB_ACCOUNT_ID"], client_id=2),
    )

    node.add_actor(
        MacroRegimeActor(
            MacroRegimeActorConfig(
                vix_id=vix_id,
                vix3m_id=vix3m_id,
                backwardation_gate=1.00,
                interval_secs=15,
                # Indices publish levels, not two-sided quotes
                use_quotes=False,
                use_bars=True,
            )
        )
    )
    node.add_actor(
        FastVolActor(
            FastVolActorConfig(
                underlying_id=spy_id,
                ttm_years=days_to_years(25.0),
                interval_secs=15,
                strike_min=SPY_DEC26_SLICE.strike_min,
                strike_max=SPY_DEC26_SLICE.strike_max,
            )
        )
    )

    try:
        node.add_actor(
            RealizedVolActor(
                RealizedVolActorConfig(
                    primary_id=spy_id,
                    universe_ids=tuple(InstrumentId.from_str(iid) for iid in RV_UNIVERSE),
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
                    mes_id=InstrumentId.from_str(MES_ID),
                    spx_id=spx_id,
                    mnq_id=InstrumentId.from_str(MNQ_ID),
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
