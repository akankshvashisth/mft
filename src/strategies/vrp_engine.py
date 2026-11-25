"""Nautilus-free VRP income engine: candidates -> pricer -> gate -> ranking -> one passive clip, plus exits.

Venue access goes through two small protocols (OrderGateway, MarketView) so the engine can be driven by
fakes in tests and by the Nautilus adapters (src.adapters.order_gateway) in production.
"""
from __future__ import annotations

import json
import logging
import math
import time
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from datetime import date
from typing import Any, Protocol

from src.common import keys
from src.common.constants import DEFAULT_VALKEY_URL, SECS_PER_DAY
from src.common.log import LogLike
from src.common.market import OptionSpec, QuoteObs, symbol_of
from src.common.quotes import combo_mid_natural
from src.common.store import JsonStore
from src.execution.passive_combo import CREDIT_OPEN, DEBIT_CLOSE, ChildClipOrder, PassiveComboExecutor
from src.risk.gate_rules import snapshot_is_fresh
from src.strategies.vrp_core import (
    ENTRY,
    EXIT,
    SpreadBook,
    apply_dte_window,
    build_candidates,
    candidate_row,
    clip_preview_rows,
    entry_block_reason,
    exit_confirmed,
    exit_reason,
    exit_signal,
    leg_mismatches,
    per_lot_risk,
    rank_key,
)

HISTORY_MAX_LEN = 200
RISK_CHECK_INTERVAL_SECS = 1.0
MAX_CANDIDATE_ROWS = 8


@dataclass(frozen=True)
class VrpSettings:
    underlying_id: str
    option_ids: tuple[str, ...] = ()
    paper_routing_enabled: bool = False
    eval_interval_secs: int = 15
    sleeve_equity_usd: float = 100_000.0
    profit_target_fraction: float = 0.50
    min_exit_dte: int = 14
    valkey_url: str = DEFAULT_VALKEY_URL

    @property
    def symbol(self) -> str:
        return symbol_of(self.underlying_id)

    @property
    def mode(self) -> str:
        return "LIVE_PAPER_ROUTING" if self.paper_routing_enabled else "SHADOW_DRY_RUN"


@dataclass(frozen=True)
class OrderView:
    is_open: bool
    is_closed: bool
    is_pending_update: bool = False
    is_pending_cancel: bool = False
    status: str = ""


@dataclass(frozen=True)
class PreparedOrder:
    order_key: str      # Client order id
    spread_id: str      # Spread instrument id
    handle: Any = None  # Gateway-specific order object


class OrderGateway(Protocol):
    def prepare(self, legs: list[dict], side: str, lots: int, limit_price: float) -> PreparedOrder:
        """Builds (or reuses) the spread instrument and a DAY limit order for it; nothing is sent yet."""
        ...

    def submit(self, order: PreparedOrder) -> None: ...

    def modify_price(self, order_key: str, price: float) -> bool:
        """False when the order can't be repriced (e.g. its instrument is unknown)."""
        ...

    def cancel(self, order_key: str) -> None: ...

    def order_view(self, order_key: str) -> OrderView | None: ...


class MarketView(Protocol):
    def cache_quote(self, instrument_id: str) -> QuoteObs | None: ...

    def option_spec(self, instrument_id: str) -> OptionSpec | None: ...

    def net_position_qty(self, instrument_id: str) -> float: ...

    def ensure_subscribed(self, instrument_id: str) -> None:
        """Subscribes to an option's quotes once, if the instrument is known."""
        ...


@dataclass
class WorkingOrder:
    """One in-flight passive child clip (an OptionSpread DAY limit order)."""

    order_key: str
    clip_id: str
    parent_id: str
    purpose: str            # ENTRY or EXIT
    side: str               # CREDIT_OPEN or DEBIT_CLOSE
    spread_key: str
    legs: list[dict]
    combo: Any
    expiry: str
    filled_lots: float = 0.0
    fill_notional: float = 0.0
    commission_usd: float = 0.0


@dataclass
class EntryOutcome:
    action: str = "STAND_DOWN"
    clips_planned: list[dict] = field(default_factory=list)
    routed_order_ids: list[str] = field(default_factory=list)


class VrpEngine:
    """At most one child clip works at a time, and every further clip needs a fresh gate approval."""

    def __init__(
        self,
        settings: VrpSettings,
        *,
        gate: Any,
        pricer: Any,
        executor: PassiveComboExecutor,
        store: JsonStore,
        gateway: OrderGateway,
        market: MarketView,
        log: LogLike | None = None,
        clock_ns: Callable[[], int] = time.time_ns,
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        self.settings = settings
        self.gate = gate
        self.pricer = pricer
        self.executor = executor
        self.store = store
        self.gateway = gateway
        self.market = market
        self.log = log if log is not None else logging.getLogger(__name__)
        self.clock_ns = clock_ns
        self.monotonic = monotonic

        self.quotes: dict[str, tuple[float, float, int, int]] = {}
        self.spot_mid: float = 0.0
        self.book = SpreadBook()
        self.working: dict[str, WorkingOrder] = {}
        self.book_key = keys.vrp_book_key(settings.underlying_id)
        self._registered_spreads: set[str] = set()
        self._last_fill_wall_ns: int = 0
        self._risk_check_mono: float = float("-inf")
        self._risk_block_reason: str | None = None
        self._rv_block_reason: str | None = None

    # --- Lifecycle -------------------------------------------------------------------------------

    def start(self) -> int:
        """Subscribes to the configured options, restores the persisted book; returns open spreads."""
        for option_id in self.settings.option_ids:
            self.market.ensure_subscribed(str(option_id))
        self.book = SpreadBook.from_json(self.store.get_json(self.book_key))
        for trade in self.book.trades.values():
            self._ensure_leg_subscriptions(trade["legs"])
        return len(self.book)

    def stop(self) -> None:
        """Nothing keeps working unattended after shutdown (DAY orders would otherwise live until the close)."""
        for key in list(self.working):
            view = self.gateway.order_view(key)
            if view is not None and view.is_open and not view.is_pending_cancel:
                self.gateway.cancel(key)
        self.persist_book()

    def persist_book(self) -> None:
        self.store.set_json(self.book_key, self.book.to_json())

    def _ensure_leg_subscriptions(self, legs: list[dict]) -> None:
        for leg in legs:
            self.market.ensure_subscribed(str(leg["instrument_id"]))

    # --- Market data -----------------------------------------------------------------------------

    def on_quote(self, quote: QuoteObs) -> None:
        if quote.mid is None:
            return
        if quote.instrument_id == str(self.settings.underlying_id):
            self.spot_mid = quote.mid
            # Adverse-move cancels and dwell-timer limit steps react to every underlying tick
            self.manage_working_orders()
        else:
            self.quotes[quote.instrument_id] = (
                quote.bid,
                quote.ask,
                int(quote.bid_size or 10),
                int(quote.ask_size or 10),
            )

    def leg_quote(self, instrument_id: str) -> tuple[float, float] | None:
        own = self.quotes.get(instrument_id)
        if own is not None:
            return own[0], own[1]
        cached = self.market.cache_quote(instrument_id)
        if cached is None or cached.mid is None:
            return None
        return cached.bid, cached.ask

    def combo_prices(self, legs: list[dict], side: str) -> tuple[float, float] | None:
        """(mid, natural) in the spread's own convention: sum(ratio x leg price), positive for credit spreads."""
        return combo_mid_natural(legs, self.leg_quote, selling=(side == CREDIT_OPEN))

    # --- Order lifecycle -------------------------------------------------------------------------

    def on_fill(self, order_key: str, last_qty: float, last_px: float, commission: float | None = None) -> None:
        wo = self.working.get(order_key)
        if wo is None:
            return
        lots = float(last_qty)
        px = float(last_px)
        wo.filled_lots += lots
        wo.fill_notional += px * lots
        if commission is not None:
            wo.commission_usd += abs(float(commission))

        if wo.purpose == ENTRY:
            self.book.apply_entry_fill(wo.spread_key, wo.combo, wo.legs, wo.expiry, px, lots, self.clock_ns())
        else:
            self.book.apply_exit_fill(wo.spread_key, lots)
        self._last_fill_wall_ns = self.clock_ns()
        self.persist_book()

        view = self.gateway.order_view(order_key)
        if view is None or view.is_closed:
            self.finalize(order_key, "FILLED")

    def on_terminal(self, order_key: str, status: str, reason: str = "") -> None:
        """Cancelled, expired, rejected or denied: the clip stops working."""
        if status in ("REJECTED", "DENIED") and order_key in self.working:
            self.log.warning(f"OptionSpread order {status.lower()} {order_key}: {reason}")
        self.finalize(order_key, status)

    def on_modify_rejected(self, order_key: str) -> None:
        wo = self.working.get(order_key)
        view = self.gateway.order_view(order_key)
        if wo is None or view is None or not view.is_open or view.is_pending_cancel:
            return
        # The executor already stepped its limit; cancel rather than work a price it no longer tracks
        self.executor.mark_clip_status(wo.clip_id, "CANCELLED_MODIFY_REJECTED")
        self.gateway.cancel(order_key)

    def finalize(self, order_key: str, status: str) -> None:
        wo = self.working.pop(order_key, None)
        if wo is None:
            return
        if wo.filled_lots > 0.0:
            self.executor.record_fill(
                wo.clip_id,
                wo.fill_notional / wo.filled_lots,
                wo.commission_usd,
                filled_lots=int(round(wo.filled_lots)),
            )
        else:
            self.executor.mark_clip_status(wo.clip_id, status)
        self.executor.discard_clip(wo.clip_id)

    def _register_spread_legs(self, spread_id: str, legs: list[dict]) -> None:
        if spread_id in self._registered_spreads:
            return
        # Lets PortfolioAnalyticsActor decompose spread-level positions into option legs
        self.store.set_json(
            keys.spread_legs_key(spread_id),
            [{"instrument_id": str(leg["instrument_id"]), "ratio": int(leg["ratio"])} for leg in legs],
        )
        self._registered_spreads.add(spread_id)

    def route_clip(self, clip: ChildClipOrder, purpose: str, legs: list[dict], combo: Any, expiry: str) -> str | None:
        """Submits one passive child clip as an atomic OptionSpread LIMIT order (DAY)."""
        order_key: str | None = None
        try:
            prepared = self.gateway.prepare(legs, clip.side, clip.lots, clip.current_limit_price)
            order_key = prepared.order_key
            self._register_spread_legs(prepared.spread_id, legs)
            # Registered before submission: denials (and simulated fills) can arrive synchronously
            self.working[order_key] = WorkingOrder(
                order_key=order_key,
                clip_id=clip.clip_id,
                parent_id=clip.parent_order_id,
                purpose=purpose,
                side=clip.side,
                spread_key=prepared.spread_id,
                legs=legs,
                combo=combo,
                expiry=expiry,
            )
            self.gateway.submit(prepared)
        except Exception as exc:
            self.log.error(f"Failed to submit {purpose} OptionSpread order ({clip.combo_label}): {exc}")
            if order_key is not None:
                self.working.pop(order_key, None)
            self.executor.mark_clip_status(clip.clip_id, "SUBMIT_FAILED")
            self.executor.discard_clip(clip.clip_id)
            return None

        if purpose == ENTRY:
            self.gate.record_order_submission()
        self._ensure_leg_subscriptions(legs)
        return order_key

    def cached_risk_block_reason(self) -> str | None:
        # Underlying ticks can be very frequent; re-read the daemon state at most once per second
        now = self.monotonic()
        if now - self._risk_check_mono >= RISK_CHECK_INTERVAL_SECS:
            self._risk_check_mono = now
            self._risk_block_reason = self.gate.risk_daemon_block_reason()
        return self._risk_block_reason

    def _cancel(self, order_key: str, wo: WorkingOrder, clip_status: str) -> None:
        self.executor.mark_clip_status(wo.clip_id, clip_status)
        self.gateway.cancel(order_key)

    def manage_working_orders(self) -> None:
        if not self.working:
            return
        risk_block = self.cached_risk_block_reason()
        underlying = str(self.settings.underlying_id)
        for key, wo in list(self.working.items()):
            view = self.gateway.order_view(key)
            if view is None or view.is_closed:
                self.log.warning(
                    f"Order {key} closed without a tracked terminal event "
                    f"({view.status if view is not None else 'missing from cache'})"
                )
                self.finalize(key, "CLOSED_UNTRACKED")
                continue
            if not view.is_open or view.is_pending_update or view.is_pending_cancel:
                continue
            if wo.purpose == ENTRY and risk_block is not None:
                self.log.warning(f"Cancelling working entry {key} ({wo.combo}): {risk_block}")
                self._cancel(key, wo, "CANCELLED_RISK")
                continue
            if self.spot_mid <= 0.0:
                continue
            prices = self.combo_prices(wo.legs, wo.side)
            if prices is None:
                continue
            mid, natural = prices
            if mid <= 0.0:
                self.log.warning(f"Cancelling {key} ({wo.combo}): spread mid {mid:.2f} is no longer positive")
                self._cancel(key, wo, "CANCELLED_INVALID_MID")
                continue

            actions = self.executor.on_market_tick(
                underlying=underlying,
                current_spot=self.spot_mid,
                new_mid_price=mid,
                new_natural_price=natural,
                parent_order_id=wo.parent_id,
            )
            for act in actions:
                if act["action"] == "MODIFY_LIMIT" and self.gateway.modify_price(key, act["new_limit"]):
                    continue
                self.log.warning(f"PassiveCombo {act['action']} {act['clip_id']}: {act.get('reason', 'NO_INSTRUMENT')}")
                self._cancel(key, wo, "CANCELLED")

    # --- Exits -----------------------------------------------------------------------------------

    def evaluate_exits(self, today: date) -> list[dict[str, Any]]:
        """Profit-target and DTE time-stop exits for spreads opened by this strategy."""
        signals: list[dict[str, Any]] = []
        if not len(self.book):
            return signals
        s = self.settings
        working_spreads = {wo.spread_key for wo in self.working.values()}
        net_qty = self.market.net_position_qty
        mismatches = leg_mismatches(self.book.expected_leg_positions(), net_qty) if s.paper_routing_enabled else set()
        book_changed = False

        for key, trade in list(self.book.trades.items()):
            open_lots = float(trade.get("open_lots") or 0.0)
            dte = (date.fromisoformat(str(trade["expiry"])) - today).days
            if open_lots <= 0.0 or dte < 0:
                # Expired spreads settle at the venue; nothing left to work here
                self.log.warning(f"Dropping {trade.get('combo')} from book (open_lots={open_lots}, dte={dte})")
                del self.book.trades[key]
                book_changed = True
                continue

            prices = self.combo_prices(trade["legs"], DEBIT_CLOSE)
            entry_credit = float(trade["entry_credit"])
            reason = exit_reason(
                dte, prices[0] if prices is not None else None, entry_credit, s.profit_target_fraction, s.min_exit_dte
            )
            if reason is None:
                continue

            signal = exit_signal(key, trade, reason, open_lots, entry_credit, prices)
            lots = int(min(open_lots, self.executor.cfg.max_child_clip_lots))
            if not s.paper_routing_enabled:
                pass
            elif key in working_spreads:
                signal["status"] = "ORDER_WORKING_ON_SPREAD"
            elif prices is None or self.spot_mid <= 0.0:
                signal["status"] = "NO_LIVE_QUOTES"
            elif lots < 1:
                signal["status"] = "NOTHING_TO_CLOSE"
            elif not exit_confirmed(key, trade, mismatches, net_qty):
                # Buying back a spread we may not hold would open the opposite position
                signal["status"] = "POSITION_NOT_CONFIRMED_IN_CACHE"
            else:
                clip = self.executor.create_parent_plan(
                    underlying=str(s.underlying_id),
                    combo_label=str(trade.get("combo")),
                    side=DEBIT_CLOSE,
                    total_lots=lots,
                    spot_price=self.spot_mid,
                    net_mid_price=prices[0],
                    net_natural_price=prices[1],
                )[0]
                order_key = self.route_clip(clip, EXIT, trade["legs"], trade.get("combo"), trade["expiry"])
                signal["status"] = "EXIT_ORDER_ROUTED" if order_key else "ORDER_SUBMIT_FAILED"
                signal["order_id"] = order_key
                self.log.info(f"VRP exit {trade.get('combo')} {lots} lot(s): {reason} -> {signal['status']}")
            signals.append(signal)

        if book_changed:
            self.persist_book()
        return signals

    # --- Entries ---------------------------------------------------------------------------------

    def candidates(self, surface: dict[str, Any], today: date) -> list[dict[str, Any]]:
        s = self.settings
        return build_candidates(
            surface,
            spot_mid=self.spot_mid,
            today=today,
            symbol=s.symbol,
            underlying=str(s.underlying_id),
            strategy_quote=self.quotes.get,
            cache_quote=self.market.cache_quote,
            option_spec=self.market.option_spec,
        )

    def evaluate_candidate(self, cand: dict[str, Any], rv_20d: float, spot: float, today: date):
        s = self.settings
        combo = cand["combo"]
        risk = self.pricer.evaluate_combo(
            combo,
            realized_vol_p=rv_20d,
            horizon_days=min(14.0, combo.ttm_days),
            sleeve_equity_usd=s.sleeve_equity_usd,
        )
        # ScenarioPricer already reports per 1-lot figures; the floor mirrors its own sizing rule
        per_lot_es99, per_lot_ev, requested_qty = per_lot_risk(risk)
        decision = self.gate.evaluate_combo(
            underlying=str(s.underlying_id),
            spot_price=spot,
            legs=[replace(leg, qty=requested_qty) for leg in cand["gate_legs"]],
            incremental_es99_per_lot_usd=per_lot_es99,
            net_ev_per_lot_usd=per_lot_ev,
            as_of=today,
        )
        dte = (cand["expiry"] - today).days
        apply_dte_window(decision, dte, s.min_exit_dte)
        return candidate_row(cand, risk, decision, dte, per_lot_es99, per_lot_ev), cand, decision

    def current_entry_block_reason(self) -> str | None:
        if self._rv_block_reason is not None:
            return self._rv_block_reason
        portfolio_ts_ns = 0
        if self._last_fill_wall_ns:
            portfolio_ts_ns = int(self.store.get_json(keys.PORTFOLIO_RISK).get("ts_ns") or 0)
        return entry_block_reason(bool(self.working), self.spot_mid, self._last_fill_wall_ns, portfolio_ts_ns)

    def _enter(self, best, spot: float, block_reason: str | None) -> EntryOutcome:
        outcome = EntryOutcome()
        if best is None:
            return outcome
        s = self.settings
        row, cand, decision = best
        preview = self.executor.create_parent_plan(
            underlying=str(s.underlying_id),
            combo_label=row["combo"],
            side=CREDIT_OPEN,
            total_lots=decision.approved_qty,
            spot_price=spot,
            net_mid_price=cand["mid_credit"],
            net_natural_price=cand["nat_credit"],
            register=False,
        )
        outcome.clips_planned = clip_preview_rows(preview)
        if not s.paper_routing_enabled:
            outcome.action = "SHADOW_SIGNAL_READY"
        elif block_reason is not None:
            outcome.action = f"ENTRY_DEFERRED({block_reason})"
        else:
            # One clip per approval; the next clip is re-gated once the book reflects this fill
            clip = self.executor.create_parent_plan(
                underlying=str(s.underlying_id),
                combo_label=row["combo"],
                side=CREDIT_OPEN,
                total_lots=min(decision.approved_qty, self.executor.cfg.max_child_clip_lots),
                spot_price=self.spot_mid,
                net_mid_price=cand["mid_credit"],
                net_natural_price=cand["nat_credit"],
            )[0]
            order_key = self.route_clip(clip, ENTRY, cand["legs"], row["combo"], cand["expiry"].isoformat())
            if order_key is None:
                outcome.action = "ORDER_SUBMIT_FAILED"
            else:
                outcome.action = "LIVE_PAPER_ORDER_ROUTED"
                outcome.routed_order_ids.append(order_key)
        return outcome

    def working_rows(self) -> list[dict[str, Any]]:
        rows = []
        for key, wo in self.working.items():
            clip = self.executor.active_clips.get(wo.clip_id)
            rows.append(
                {
                    "client_order_id": key,
                    "purpose": wo.purpose,
                    "combo": wo.combo,
                    "lots": clip.lots if clip is not None else None,
                    "limit": clip.current_limit_price if clip is not None else None,
                    "step": clip.step_count if clip is not None else None,
                    "filled_lots": wo.filled_lots,
                }
            )
        return rows

    def on_timer(self, event_ts_ns: int, wall_ns: int, today: date) -> dict[str, Any]:
        """One evaluation cycle; `today` is the New York trading date. Returns the published payload."""
        s = self.settings
        underlying = str(s.underlying_id)
        surf = self.store.get_json(keys.surface_key(underlying))
        rv_snap = self.store.get_json(keys.rv_key(underlying))
        rv_20d: float | None = None
        self._rv_block_reason = None

        # Validate RV freshness and finiteness — fail closed
        if snapshot_is_fresh(rv_snap, wall_ns, SECS_PER_DAY):
            raw_rv = rv_snap.get("rv_20d") if isinstance(rv_snap, dict) else None
            if isinstance(raw_rv, (int, float)) and math.isfinite(float(raw_rv)) and float(raw_rv) > 0.0:
                rv_20d = float(raw_rv)
            else:
                self._rv_block_reason = "RV_MISSING_OR_INVALID"
        else:
            self._rv_block_reason = "RV_MISSING_OR_INVALID"

        spot = self.spot_mid or float(surf.get("spot") or 0.0)

        self.manage_working_orders()
        exit_signals = self.evaluate_exits(today)

        if rv_20d is not None:
            evaluated = [self.evaluate_candidate(cand, rv_20d, spot, today) for cand in self.candidates(surf, today)]
            evaluated.sort(key=lambda e: rank_key(e[0]), reverse=True)
            rows = [e[0] for e in evaluated]
            top_row = rows[0] if rows else None
            best = evaluated[0] if evaluated and evaluated[0][0]["gate_approved"] else None
        else:
            evaluated = []
            rows = []
            top_row = None
            best = None

        block_reason = self.current_entry_block_reason()
        outcome = self._enter(best, spot, block_reason)

        payload = {
            "mode": s.mode,
            "action": outcome.action,
            "entry_block_reason": block_reason,
            "underlying": underlying,
            "spot": round(spot, 2),
            "rv_20d": round(rv_20d, 4) if rv_20d is not None else None,
            "candidates_scanned": len(rows),
            "top_candidate": top_row,
            "all_candidates": rows[:MAX_CANDIDATE_ROWS],
            "passive_clips": outcome.clips_planned,
            "routed_order_ids": outcome.routed_order_ids,
            "working_orders": self.working_rows(),
            "open_book": self.book.rows(),
            "exit_signals": exit_signals,
            "ts_ns": wall_ns,
            "event_ts_ns": event_ts_ns,
        }

        raw_json = json.dumps(payload)
        self.store.set_text(keys.VRP_LATEST, raw_json)
        self.store.push_capped(keys.VRP_HISTORY, raw_json, HISTORY_MAX_LEN)

        if top_row:
            self.log.info(
                f"VRP[{s.mode}] {outcome.action} | Top={top_row['combo']} "
                f"Mid=${top_row['mid_credit']:.2f} EV=${top_row['net_ev_per_lot']:+.2f}/lot "
                f"ES99=${top_row['es99_per_lot']:.0f} x{top_row['approved_qty']} | Reasons={top_row['gate_reasons']}"
            )
        return payload
