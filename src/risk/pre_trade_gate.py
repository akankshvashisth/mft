from __future__ import annotations

import time
from collections.abc import Callable
from datetime import date

from src.common import keys
from src.common.constants import DEFAULT_VALKEY_URL
from src.common.store import JsonStore, make_store
from src.common.timeutil import ny_date
from src.risk.calendar import EventCalendarLike, default_event_calendar
from src.risk.gate_rules import (
    OrderRateLimiter,
    book_es99_usd,
    check_legs,
    daemon_block_reason,
    es_budget_check,
    ev_reasons,
    futures_check,
    metrics_reasons,
    portfolio_reasons,
    regime_reasons,
    surface_reasons,
)
from src.risk.gate_types import CandidateLegSpec, GateConfig, GateDecision

__all__ = ["CandidateLegSpec", "GateConfig", "GateDecision", "PreTradeGate"]


class PreTradeGate:
    """Unified mid-cap transaction boundary before any NautilusTrader order submission.

    Reads the actor snapshots and the risk daemon state from the store and composes the pure rules in
    src.risk.gate_rules. The store, the event calendar and both clocks are injectable for tests.
    """

    def __init__(
        self,
        config: GateConfig | None = None,
        calendar: EventCalendarLike | None = None,
        valkey_url: str = DEFAULT_VALKEY_URL,
        *,
        store: JsonStore | None = None,
        clock_ns: Callable[[], int] = time.time_ns,
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        self.cfg = config or GateConfig()
        self.calendar = calendar if calendar is not None else default_event_calendar()
        self.store = store if store is not None else make_store(valkey_url)
        self.clock_ns = clock_ns
        self.rate_limiter = OrderRateLimiter(self.cfg.max_order_rate_per_min, clock=monotonic)

    def risk_daemon_block_reason(self, now_ns: int | None = None) -> str | None:
        """Fail-closed daemon check: requires halted == "0" and a recent, non-halted evaluation."""
        if not self.store.connected:
            return "RISK_DAEMON_UNREACHABLE"
        now_ns = now_ns if now_ns is not None else self.clock_ns()
        return daemon_block_reason(
            self.store.get_text(keys.RISK_HALTED),
            self.store.get_json(keys.RISK_STATUS),
            now_ns,
            self.cfg.max_risk_status_age_secs,
        )

    def record_order_submission(self) -> None:
        """Counts an order actually sent to the venue against the anti-loop throttle."""
        self.rate_limiter.record()

    def evaluate_combo(
        self,
        underlying: str,
        spot_price: float,
        legs: list[CandidateLegSpec],
        incremental_es99_per_lot_usd: float,
        net_ev_per_lot_usd: float,
        as_of: date | None = None,
    ) -> GateDecision:
        now_ns = self.clock_ns()
        as_of_date = as_of or ny_date(now_ns)
        cfg = self.cfg
        reasons: list[str] = []
        warnings: list[str] = []

        # 1. Fail-closed Out-of-Process Risk Daemon check (flag + fresh heartbeat)
        daemon_reason = self.risk_daemon_block_reason(now_ns)
        if daemon_reason is not None:
            reasons.append(daemon_reason)

        # 2. Macro Regime & Surface Hygiene checks (stale snapshots count as missing)
        reasons += regime_reasons(self.store.get_json(keys.REGIME_LATEST), now_ns, cfg.max_input_age_secs)
        reasons += surface_reasons(self.store.get_json(keys.surface_key(underlying)), now_ns, cfg)
        reasons += metrics_reasons(self.store.get_json(keys.metrics_key(underlying)), now_ns, cfg.max_input_age_secs)
        fut_reasons, fut_warnings = futures_check(self.store.get_json(keys.FUTURES_LATEST), now_ns, cfg)
        reasons += fut_reasons
        warnings += fut_warnings

        # 3. Positive Net Expected Value check (after fees & half-spread slippage)
        reasons += ev_reasons(net_ev_per_lot_usd)

        # 4. Order-rate loop protection (submissions are recorded by the caller when it routes)
        throttle_reason = self.rate_limiter.throttle_reason()
        if throttle_reason is not None:
            reasons.append(throttle_reason)

        # 5. Mid-cap Contract Hygiene, Liquidity & Participation Caps
        leg_check = check_legs(legs, as_of_date, cfg)
        reasons += leg_check.reasons
        warnings += leg_check.warnings

        # 6. Corporate Event, Earnings & Ex-Dividend Gate
        ev_res = self.calendar.check_entry(
            symbol=underlying,
            as_of=as_of_date,
            expiry_date=leg_check.max_expiry,
            has_short_call=leg_check.has_short_call,
            short_call_strike=leg_check.short_call_strike,
            spot_price=spot_price,
            short_call_extrinsic=leg_check.short_call_extrinsic,
        )
        if not ev_res.allowed:
            reasons.extend(ev_res.reasons)
        if ev_res.macro_event_within_2d:
            warnings.append(f"MACRO_EVENT_NEAR({ev_res.macro_event_within_2d})")

        # 7. Marginal Book ES_99 Tail Budget Check (a missing/stale book must not read as zero risk)
        port = self.store.get_json(keys.PORTFOLIO_RISK)
        reasons += portfolio_reasons(port, now_ns, cfg.max_input_age_secs)
        budget = es_budget_check(book_es99_usd(port), incremental_es99_per_lot_usd, leg_check.capped_qty, cfg)
        reasons += budget.reasons
        warnings += budget.warnings

        approved = len(reasons) == 0 and budget.capped_qty >= 1
        return GateDecision(
            approved=approved,
            approved_qty=budget.capped_qty if approved else 0,
            reasons=reasons,
            warnings=warnings,
            projected_book_es99_usd=round(budget.projected_es99_usd, 2),
        )
