from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.backtest.vrp_walkforward import run_midcap_vrp_backtest  # noqa: E402,F401  (re-exported for callers)
from src.data.event_calendar import EventCalendar  # noqa: E402


def main() -> None:
    # The chain loader is not in the repository yet, so it is imported only when the script runs
    from src.data.thetadata_chain_loader import ThetaDataChainLoader

    loader = ThetaDataChainLoader()
    synth_df = loader.build_synthetic_midcap_chain(symbol="MIDX", start_price=85.0, n_days=180)
    summary = run_midcap_vrp_backtest(synth_df, calendar=EventCalendar())
    print("Mid-Cap VRP Walk-Forward Backtest Summary (50% Passive Fill Haircut):")
    for k, v in summary.items():
        print(f"  {k:26s}: {v}")


if __name__ == "__main__":
    main()
