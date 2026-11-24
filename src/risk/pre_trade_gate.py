from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
import json
import time
import redis
from src.data.event_calendar import EventCalendar


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
    bid_size: int = 10
    ask_size: int = 10
    open_interest: int = 500
    multiplier: int = 100
    is_adjusted: bool = False
    extrinsic_value: float = 1.0


@dataclass(frozen=True)
class GateConfig:
    sleeve_equity_usd: float = 100_000.0
    max_book_es99_pct: float = 0.01          # 1.0% ($1,000 on $100k sleeve)
    max_sector_es99_pct: float = 0.25        # Max 25% of tail budget in any single sector
    max_oi_participation: float = 0.08       # Max 8% of strike Open Interest per leg
    max_tob_participation: float = 0.30      # Max 30% of displayed top-of-book size
    max_leg_spread_pct: float = 0.18         # Max 18% bid-ask width for mid-cap legs
    min_open_interest: int = 100             # Minimum strike OI floor
    max_order_rate_per_min: int = 6          # Anti-loop throttle
    max_parity_fut_diff_bps: float = 20.0    # Max forward divergence vs CME futures


@dataclass
class GateDecision:
    approved: bool
    approved_qty: int
    reasons: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    projected_book_es99_usd: float = 0.0


class PreTradeGate:
    """Unified mid-cap transaction boundary before any NautilusTrader order submission."""

    def __init__(
        self,
        config: GateConfig | None = None,
        calendar: EventCalendar | None = None,
        valkey_url: str = "redis://127.0.0.1:6379/0",
    ) -> None:
        self.cfg = config or GateConfig()
        self.calendar = calendar or EventCalendar()
        self.rdb = redis.from_url(valkey_url) if valkey_url else None
        self._recent_order_ts: list[float] = []

    def _get_json(self, key: str) -> dict:
        if self.rdb is None:
            return {}
        raw = self.rdb.get(key)
        if not raw:
            return {}
        try:
            return json.loads(raw)
        except Exception:
            return {}

    def evaluate_combo(
        self,
        underlying: str,
        spot_price: float,
        legs: list[CandidateLegSpec],
        incremental_es99_per_lot_usd: float,
        net_ev_per_lot_usd: float,
        as_of: date | None = None,
    ) -> GateDecision:
        as_of_date = as_of or date.today()
        reasons: list[str] = []
        warnings: list[str] = []

        # 1. Fail-closed Out-of-Process Risk Daemon check
        if self.rdb is not None:
            halted_raw = self.rdb.get("mft:risk:halted")
            if halted_raw is None or halted_raw.decode("utf-8").strip() != "0":
                reasons.append("RISK_DAEMON_HALTED_OR_MISSING")

        # 2. Macro Regime & Surface Hygiene checks
        regime = self._get_json("mft:regime:latest")
        if not regime or not regime.get("short_vol_allowed", False):
            reasons.append(f"MACRO_REGIME_BLOCKED(ratio={regime.get('ratio')})")

        surf = self._get_json(f"mft:surface:{underlying}")
        if not surf or not surf.get("arb_free", False):
            reasons.append("SVI_SURFACE_NOT_ARB_FREE_OR_MISSING")

        metrics = self._get_json(f"mft:metrics:{underlying}")
        if metrics.get("strategy_bias") == "DEFENSIVE_OR_LONG_CONVEXITY":
            reasons.append(f"SURFACE_BIAS_DEFENSIVE(VRP={metrics.get('vrp_spread')})")

        fut = self._get_json("mft:futures:latest")
        if fut and abs(float(fut.get("parity_vs_fut_diff_bps", 0.0))) > self.cfg.max_parity_fut_diff_bps:
            reasons.append(f"FUTURES_PARITY_DIVERGENCE({fut.get('parity_vs_fut_diff_bps')}bps)")

        # 3. Positive Net Expected Value check (after fees & half-spread slippage)
        if net_ev_per_lot_usd <= 0.0:
            reasons.append(f"NON_POSITIVE_NET_EV(${net_ev_per_lot_usd:.2f}/lot)")

        # 4. Order-rate loop protection
        now = time.monotonic()
        self._recent_order_ts = [t for t in self._recent_order_ts if now - t < 60.0]
        if len(self._recent_order_ts) >= self.cfg.max_order_rate_per_min:
            reasons.append(f"ORDER_RATE_THROTTLE({len(self._recent_order_ts)}/min)")

        # 5. Mid-cap Contract Hygiene, Liquidity & Participation Caps
        requested_qty = min(leg.qty for leg in legs) if legs else 0
        capped_qty = requested_qty
        has_short_call = False
        short_call_strike = None
        short_call_ext = None
        max_expiry = as_of_date

        for leg in legs:
            if leg.expiry_date > max_expiry:
                max_expiry = leg.expiry_date
            if leg.multiplier != 100 or leg.is_adjusted:
                reasons.append(f"NON_STANDARD_CONTRACT({leg.instrument_id}, mult={leg.multiplier})")
            if leg.bid <= 0.0 or leg.ask <= leg.bid:
                reasons.append(f"INVALID_QUOTE({leg.instrument_id}: {leg.bid}/{leg.ask})")
                continue

            mid = 0.5 * (leg.bid + leg.ask)
            spread_pct = (leg.ask - leg.bid) / max(mid, 0.01)
            if spread_pct > self.cfg.max_leg_spread_pct:
                reasons.append(f"WIDE_MIDCAP_SPREAD({leg.instrument_id}: {spread_pct:.1%}>{self.cfg.max_leg_spread_pct:.1%})")
            if leg.open_interest < self.cfg.min_open_interest:
                reasons.append(f"LOW_OPEN_INTEREST({leg.instrument_id}: OI={leg.open_interest}<{self.cfg.min_open_interest})")

            oi_cap = max(1, int(leg.open_interest * self.cfg.max_oi_participation))
            tob_size = leg.bid_size if leg.side == "SELL" else leg.ask_size
            tob_cap = max(1, int(tob_size * self.cfg.max_tob_participation))
            if oi_cap < capped_qty:
                capped_qty = oi_cap
                warnings.append(f"QTY_CAPPED_BY_OI({leg.instrument_id}: {oi_cap})")
            if tob_cap < capped_qty:
                capped_qty = tob_cap
                warnings.append(f"QTY_CAPPED_BY_TOB({leg.instrument_id}: {tob_cap})")

            if leg.right == "C" and leg.side == "SELL":
                has_short_call = True
                short_call_strike = leg.strike
                short_call_ext = leg.extrinsic_value

        # 6. Corporate Event, Earnings & Ex-Dividend Gate
        ev_res = self.calendar.check_entry(
            symbol=underlying,
            as_of=as_of_date,
            expiry_date=max_expiry,
            has_short_call=has_short_call,
            short_call_strike=short_call_strike,
            spot_price=spot_price,
            short_call_extrinsic=short_call_ext,
        )
        if not ev_res.allowed:
            reasons.extend(ev_res.reasons)
        if ev_res.macro_event_within_2d:
            warnings.append(f"MACRO_EVENT_NEAR({ev_res.macro_event_within_2d})")

        # 7. Marginal Book ES_99 Tail Budget Check
        port = self._get_json("mft:portfolio:risk")
        current_es99 = float((port.get("risk") or {}).get("book_es_99_usd", 0.0))
        max_es99_usd = self.cfg.sleeve_equity_usd * self.cfg.max_book_es99_pct
        remaining_es99 = max(0.0, max_es99_usd - current_es99)

        if incremental_es99_per_lot_usd > 0.0:
            es_qty_cap = int(remaining_es99 // incremental_es99_per_lot_usd)
            if es_qty_cap <= 0:
                reasons.append(
                    f"SLEEVE_ES99_BUDGET_EXHAUSTED(cur=${current_es99:.0f}+${incremental_es99_per_lot_usd:.0f}>${max_es99_usd:.0f})"
                )
            elif es_qty_cap < capped_qty:
                capped_qty = es_qty_cap
                warnings.append(f"QTY_CAPPED_BY_ES99({es_qty_cap} lots)")

        projected_es99 = current_es99 + max(0, capped_qty) * max(0.0, incremental_es99_per_lot_usd)
        approved = len(reasons) == 0 and capped_qty >= 1
        if approved:
            self._recent_order_ts.append(now)

        return GateDecision(
            approved=approved,
            approved_qty=capped_qty if approved else 0,
            reasons=reasons,
            warnings=warnings,
            projected_book_es99_usd=round(projected_es99, 2),
        )
