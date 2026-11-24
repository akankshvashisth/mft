"""Instrument definitions shared by run_paper.py, the scripts and the strategy configuration.

One option-slice definition feeds the IBKR contracts that get loaded, the strategy's option_ids and the
vol actor's strike range, so they can't drift apart.
"""
from __future__ import annotations

from dataclasses import dataclass

SPY_ID = "SPY.ARCA"
VIX_ID = "^VIX.CBOE"
VIX3M_ID = "^VIX3M.CBOE"
SPX_ID = "^SPX.CBOE"
MES_ID = "MESZ26.CME"
MNQ_ID = "MNQZ26.CME"

# Realized-vol / correlation universe, and the wider set synced for history (adds the VIX ETPs)
RV_UNIVERSE: tuple[str, ...] = (
    "SPY.ARCA",
    "QQQ.NASDAQ",
    "IWM.ARCA",
    "XBI.ARCA",
    "GDX.ARCA",
    "SMH.NASDAQ",
    "TLT.NASDAQ",
    "GLD.ARCA",
)
SYNC_UNIVERSE: tuple[str, ...] = RV_UNIVERSE + ("VIXY.ARCA", "VIXM.ARCA")


def split_instrument_id(instrument_id: str) -> tuple[str, str]:
    """"SPY.ARCA" -> ("SPY", "ARCA")."""
    symbol, _, venue = instrument_id.rpartition(".")
    return symbol, venue


def occ_option_id(root: str, expiry_yymmdd: str, is_call: bool, strike: float, venue: str = "SMART") -> str:
    """IBKR simplified-symbology option id, e.g. "SPY 261218P00690000.SMART"."""
    return f"{root} {expiry_yymmdd}{'C' if is_call else 'P'}{int(round(strike * 1000)):08d}.{venue}"


def strike_ladder(low: int, high: int, step: int = 5) -> tuple[float, ...]:
    """Strikes from low to high inclusive."""
    return tuple(float(k) for k in range(low, high + 1, step))


@dataclass(frozen=True)
class OptionSliceSpec:
    root: str
    expiry: str                       # YYYYMMDD
    put_strikes: tuple[float, ...]
    call_strikes: tuple[float, ...]
    exchange: str = "SMART"

    @property
    def expiry_yymmdd(self) -> str:
        return self.expiry[2:]

    @property
    def strike_min(self) -> float:
        return min(self.put_strikes + self.call_strikes)

    @property
    def strike_max(self) -> float:
        return max(self.put_strikes + self.call_strikes)

    def options(self) -> list[tuple[str, float, bool]]:
        """(instrument id, strike, is_call): puts first, then calls."""
        legs = [(k, False) for k in self.put_strikes] + [(k, True) for k in self.call_strikes]
        return [(occ_option_id(self.root, self.expiry_yymmdd, c, k, self.exchange), k, c) for k, c in legs]


# The December 2026 SPY slice actually loaded from IBKR: OTM puts 685-720 and OTM calls 720-755
SPY_DEC26_SLICE = OptionSliceSpec(
    root="SPY",
    expiry="20261218",
    put_strikes=strike_ladder(685, 720),
    call_strikes=strike_ladder(720, 755),
)


def slice_option_ids(spec: OptionSliceSpec) -> tuple[str, ...]:
    return tuple(option_id for option_id, _, _ in spec.options())


def ib_option_contracts(spec: OptionSliceSpec) -> list[dict]:
    return [
        {
            "secType": "OPT",
            "symbol": spec.root,
            "exchange": spec.exchange,
            "lastTradeDateOrContractMonth": spec.expiry,
            "strike": float(strike),
            "right": "C" if is_call else "P",
        }
        for _, strike, is_call in spec.options()
    ]


def ib_future_contract(symbol: str, contract_month: str = "202612", exchange: str = "CME", currency: str = "USD") -> dict:
    return {
        "secType": "FUT",
        "symbol": symbol,
        "exchange": exchange,
        "currency": currency,
        "lastTradeDateOrContractMonth": contract_month,
    }
