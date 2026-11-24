import argparse
import sys
from nautilus_trader.adapters.interactive_brokers import (
    InteractiveBrokersDataClientConfig,
    InteractiveBrokersDataClientFactory,
    InteractiveBrokersInstrumentProviderConfig,
    MarketDataType,
    SymbologyMethod,
)
from nautilus_trader.config import LiveNodeConfig
from nautilus_trader.live import LiveNodeBuilder
from nautilus_trader.model import InstrumentId, TraderId


def probe_symbol(
    symbol: str,
    host: str = "127.0.0.1",
    port: int = 4002,
    client_id: int = 9,
) -> bool:
    iid = InstrumentId.from_str(symbol)
    trader_name = "MFT-PROBE"
    cfg = LiveNodeConfig(
        trader_id=TraderId(trader_name),
        data_clients={
            "IB": InteractiveBrokersDataClientConfig(
                host=host,
                port=port,
                client_id=client_id,
                market_data_type=MarketDataType.DELAYED_FROZEN,
                instrument_provider=InteractiveBrokersInstrumentProviderConfig(
                    symbology_method=SymbologyMethod.SIMPLIFIED,
                    load_ids=frozenset([iid]),
                ),
            )
        },
    )
    node = LiveNodeBuilder.from_config(
        trader_name,
        cfg,
        data_factories={"IB": InteractiveBrokersDataClientFactory()},
    ).build()
    try:
        node.run()
        return True
    except KeyboardInterrupt:
        print(f"\n[STOPPED] {symbol} (reached running loop; press Ctrl+C to test next or exit)")
        return True
    except Exception as exc:
        print(f"\n[FAILED] {symbol} -> {type(exc).__name__}: {exc}")
        return False
    finally:
        node.dispose()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Probe IBKR contract resolution for one or more InstrumentIds."
    )
    parser.add_argument(
        "symbols",
        nargs="+",
        help='InstrumentId strings, e.g. "VIX.CBOE" "SPY 261218P00690000.SMART"',
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=4002)
    parser.add_argument("--client-id", type=int, default=9)
    args = parser.parse_args()

    results: dict[str, bool] = {}
    for sym in args.symbols:
        print(f"\n--- Probing {sym} (client_id={args.client_id}) ---")
        results[sym] = probe_symbol(sym, host=args.host, port=args.port, client_id=args.client_id)

    print("\n=== Probe Summary ===")
    for sym, ok in results.items():
        status = "OK / REACHED LOOP" if ok else "FAILED"
        print(f"  {sym:30s} -> {status}")


if __name__ == "__main__":
    main()
