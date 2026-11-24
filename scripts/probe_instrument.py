import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.runtime.ib_node import (  # noqa: E402
    DEFAULT_HOST,
    DEFAULT_PORT,
    build_live_node,
    data_client_config,
    instrument_provider_config,
)


def probe_symbol(
    symbol: str,
    host: str = DEFAULT_HOST,
    port: int = DEFAULT_PORT,
    client_id: int = 9,
) -> bool:
    node = build_live_node(
        "MFT-PROBE",
        data_client_config(instrument_provider_config([symbol]), host=host, port=port, client_id=client_id),
    )
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
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
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
