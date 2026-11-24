"""Plain market observations and instrument descriptions used by the Nautilus-free layer."""
from __future__ import annotations

from dataclasses import dataclass

from src.common.constants import NS_PER_SEC
from src.common.quotes import two_sided_mid


@dataclass(frozen=True)
class QuoteObs:
    instrument_id: str
    bid: float
    ask: float
    ts_ns: int
    bid_size: float | None = None
    ask_size: float | None = None

    @property
    def mid(self) -> float | None:
        return two_sided_mid(self.bid, self.ask)


@dataclass(frozen=True)
class TradeObs:
    instrument_id: str
    price: float
    ts_ns: int


@dataclass(frozen=True)
class BarObs:
    instrument_id: str
    open: float
    high: float
    low: float
    close: float
    volume: float
    ts_ns: int
    aggregation: str = "MINUTE"   # "MINUTE", "DAY", ...
    step: int = 1

    @property
    def is_daily(self) -> bool:
        return self.aggregation == "DAY"

    def ohlcv_row(self) -> dict[str, float]:
        return {
            "ts_ns": int(self.ts_ns),
            "open": float(self.open),
            "high": float(self.high),
            "low": float(self.low),
            "close": float(self.close),
            "volume": float(self.volume),
        }


@dataclass(frozen=True)
class OptionSpec:
    instrument_id: str
    underlying: str
    strike: float
    is_call: bool
    expiry_ns: int = 0
    multiplier: float = 100.0
    open_interest: float | None = None


def symbol_of(instrument_id: object) -> str:
    """Symbol part of an instrument id: "SPY.ARCA" -> "SPY"; ids without a venue are returned as-is."""
    text = str(instrument_id)
    return text.rsplit(".", 1)[0] if "." in text else text


@dataclass
class LatestValue:
    """Most recent observation of a price or level, by event timestamp."""

    value: float | None = None
    ts_ns: int = 0

    def update(self, value: float, ts_ns: int, *, strictly_newer: bool = False) -> bool:
        """Stores value unless ts_ns is older (or, with strictly_newer, not newer) than the current one."""
        if ts_ns < self.ts_ns or (strictly_newer and ts_ns <= self.ts_ns):
            return False
        self.value = value
        self.ts_ns = ts_ns
        return True

    def fresh(self, now_ns: int, max_age_secs: float) -> float | None:
        """Current value if it is at most max_age_secs old, else None."""
        if self.value is None or (now_ns - self.ts_ns) / NS_PER_SEC > max_age_secs:
            return None
        return self.value
