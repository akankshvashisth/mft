"""Value types of the pre-trade gate (re-exported by src.risk.pre_trade_gate)."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date


@dataclass(frozen=True)
class CandidateLegSpec:
    instrument_id: str
    right: str               # "P" or "C"
    side: str                # "SELL" or "BUY"
    strike: float
    expiry_date: date
    qty: int
    bid: float
    ask: float
    bid_size: int | None = None
    ask_size: int | None = None
    open_interest: int | None = None   # None = unknown; never substitute a placeholder
    multiplier: int = 100
    is_adjusted: bool = False
    extrinsic_value: float | None = None   # Time value (mid - intrinsic); None = unknown


@dataclass(frozen=True)
class GateConfig:
    sleeve_equity_usd: float = 100_000.0
    max_book_es99_pct: float = 0.01          # 1.0% ($1,000 on $100k sleeve)
    max_sector_es99_pct: float = 0.25        # Max 25% of tail budget in any single sector
    max_oi_participation: float = 0.08       # Max 8% of strike Open Interest per leg
    max_tob_participation: float = 0.30      # Max 30% of displayed top-of-book size
    max_leg_spread_pct: float = 0.18         # Max 18% bid-ask width for mid-cap legs
    min_open_interest: int = 100             # Minimum strike OI floor
    allow_unknown_open_interest: bool = False  # Fail closed until an OI source is wired
    min_tob_size: int = 4                    # Minimum top-of-book size for entry
    allow_unknown_tob_size: bool = False     # Fail closed when TOB size is unknown
    max_order_rate_per_min: int = 6          # Anti-loop throttle (counts actual submissions)
    max_parity_fut_diff_bps: float = 20.0    # Max forward divergence vs CME futures
    max_svi_rmse_vol_pts: float = 0.50       # Max SVI calibration RMSE before the surface is unusable
    max_input_age_secs: float = 120.0        # Max age of actor snapshots read from Valkey
    max_risk_status_age_secs: float = 30.0   # Max age of the risk daemon's last evaluation


@dataclass
class GateDecision:
    approved: bool
    approved_qty: int
    reasons: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    projected_book_es99_usd: float = 0.0
