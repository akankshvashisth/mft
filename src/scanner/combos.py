"""Candidate option combos shared by the scanner, the pricer, the strategy and the backtest."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field


@dataclass(frozen=True)
class CandidateLeg:
    instrument_id: str
    strike: float
    right: str          # "C" or "P"
    side: int           # +1 = BUY, -1 = SELL
    qty: int            # Contract multiplier per combo unit (usually 1)
    bid: float
    ask: float
    mid: float
    raw_iv: float
    svi_iv: float
    residual_pts: float
    edge_to_half_spread: float


@dataclass
class CandidateCombo:
    name: str
    strategy_type: str  # "BULL_PUT_SPREAD", "BEAR_CALL_SPREAD", "IRON_CONDOR"
    underlying: str
    spot: float
    forward: float
    ttm_days: float
    risk_free_rate: float
    dividend_yield: float
    legs: list[CandidateLeg] = field(default_factory=list)
    net_credit: float = 0.0         # Positive = credit received ($/share)
    max_width: float = 0.0          # Strike width ($/share) for defined-risk spreads
    theo_edge_per_share: float = 0.0
    avg_edge_to_half_spread: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)
