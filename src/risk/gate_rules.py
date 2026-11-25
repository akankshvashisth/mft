"""Pure pre-trade rules composed by PreTradeGate.

Every rule takes plain payloads and returns its block reasons (and warnings) as strings; the gate keeps
their order stable because the strategy and dashboard display them.
"""
from __future__ import annotations

import math
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date
from typing import Any

from src.common.constants import NS_PER_SEC
from src.common.freshness import WALL_TS_FIELDS, is_fresh
from src.common.numeric import finite_or_none, is_number
from src.common.quotes import is_tradable_quote
from src.quant.surface_metrics import BIAS_DEFENSIVE
from src.risk.gate_types import CandidateLegSpec, GateConfig

Payload = Mapping[str, Any]


def snapshot_is_fresh(payload: Payload | None, now_ns: int, max_age_secs: float) -> bool:
    """Actor snapshots are judged by their wall-clock "ts_ns"; missing, invalid or far-future stamps are stale."""
    return is_fresh(payload, now_ns, max_age_secs, WALL_TS_FIELDS)


def daemon_block_reason(
    halted_text: str | None,
    status: Payload | None,
    now_ns: int,
    max_status_age_secs: float,
) -> str | None:
    """Fail-closed daemon check: requires halted == "0" and a recent, non-halted evaluation.

    Rejects far-future ``evaluated_at_ns`` (age_secs < -5.0) as ``RISK_DAEMON_INVALID_TIMESTAMP``.
    """
    if halted_text is None or halted_text.strip() != "0":
        return "RISK_DAEMON_HALTED_OR_MISSING"
    status = status or {}
    evaluated_ns = status.get("evaluated_at_ns")
    if not is_number(evaluated_ns):
        return "RISK_DAEMON_HALTED_OR_MISSING"
    age_secs = (now_ns - float(evaluated_ns)) / NS_PER_SEC
    if age_secs < -5.0:
        return "RISK_DAEMON_INVALID_TIMESTAMP"
    if age_secs > max_status_age_secs:
        return f"RISK_DAEMON_STALE({age_secs:.0f}s)"
    if status.get("halted") is not False:
        return "RISK_DAEMON_HALTED_OR_MISSING"
    return None


def regime_reasons(regime: Payload, now_ns: int, max_age_secs: float) -> list[str]:
    if not snapshot_is_fresh(regime, now_ns, max_age_secs):
        return ["MACRO_REGIME_MISSING_OR_STALE"]
    if not regime.get("short_vol_allowed", False):
        return [f"MACRO_REGIME_BLOCKED(ratio={regime.get('ratio')})"]
    return []


def surface_reasons(surface: Payload, now_ns: int, cfg: GateConfig) -> list[str]:
    if not snapshot_is_fresh(surface, now_ns, cfg.max_input_age_secs):
        return ["SVI_SURFACE_MISSING_OR_STALE"]
    reasons: list[str] = []
    if not surface.get("arb_free", False):
        reasons.append("SVI_SURFACE_NOT_ARB_FREE")
    rmse_pts = surface.get("rmse_vol_pts")
    if not is_number(rmse_pts) or rmse_pts > cfg.max_svi_rmse_vol_pts:
        reasons.append(f"SVI_FIT_POOR(rmse={rmse_pts}pts>{cfg.max_svi_rmse_vol_pts})")
    return reasons


def metrics_reasons(metrics: Payload, now_ns: int, max_age_secs: float) -> list[str]:
    if not snapshot_is_fresh(metrics, now_ns, max_age_secs):
        return ["SURFACE_METRICS_MISSING_OR_STALE"]
    if metrics.get("strategy_bias") == BIAS_DEFENSIVE:
        return [f"SURFACE_BIAS_DEFENSIVE(VRP={metrics.get('vrp_spread')})"]
    return []


def futures_check(futures: Payload, now_ns: int, cfg: GateConfig, is_index_etf: bool = True) -> tuple[list[str], list[str]]:
    """(reasons, warnings). Futures context is advisory: only a valid, fresh cross-check can block.

    When ``is_index_etf`` is ``False`` (e.g. single-name mid-caps without listed futures),
    the check returns ``([], [])`` cleanly without spamming ``FUTURES_CROSS_CHECK_UNAVAILABLE``.
    """
    if not is_index_etf:
        return [], []
    diff_bps = futures.get("parity_vs_fut_diff_bps")
    if snapshot_is_fresh(futures, now_ns, cfg.max_input_age_secs) and futures.get("cross_check_valid") and is_number(diff_bps):
        if abs(float(diff_bps)) > cfg.max_parity_fut_diff_bps:
            return [f"FUTURES_PARITY_DIVERGENCE({diff_bps}bps)"], []
        return [], []
    return [], ["FUTURES_CROSS_CHECK_UNAVAILABLE"]


def ev_reasons(net_ev_per_lot_usd: float) -> list[str]:
    """Positive net expected value is required (after fees and half-spread slippage)."""
    if net_ev_per_lot_usd <= 0.0:
        return [f"NON_POSITIVE_NET_EV(${net_ev_per_lot_usd:.2f}/lot)"]
    return []


def portfolio_reasons(portfolio: Payload, now_ns: int, max_age_secs: float) -> list[str]:
    """A missing, stale or unreadable book must not read as zero tail risk."""
    if not snapshot_is_fresh(portfolio, now_ns, max_age_secs):
        return ["PORTFOLIO_RISK_MISSING_OR_STALE"]
    if _book_es99_or_none(portfolio) is None:
        return ["PORTFOLIO_RISK_INVALID"]
    risk = portfolio.get("risk")
    if isinstance(risk, dict) and risk.get("has_unmodeled_positions"):
        return ["PORTFOLIO_HAS_UNMODELED_POSITIONS"]
    return []


def _book_es99_or_none(portfolio: Payload) -> float | None:
    risk = portfolio.get("risk")
    return finite_or_none(risk.get("book_es_99_usd")) if isinstance(risk, Mapping) else None


def book_es99_usd(portfolio: Payload) -> float:
    """Book ES99 for the budget arithmetic; 0 when unreadable, a case portfolio_reasons already blocks."""
    value = _book_es99_or_none(portfolio)
    return value if value is not None else 0.0


class OrderRateLimiter:
    """Anti-loop throttle over a sliding window of actual order submissions."""

    def __init__(
        self,
        max_per_window: int,
        clock: Callable[[], float] = time.monotonic,
        window_secs: float = 60.0,
    ) -> None:
        self.max_per_window = max_per_window
        self.clock = clock
        self.window_secs = window_secs
        self.timestamps: list[float] = []

    def record(self) -> None:
        self.timestamps.append(self.clock())

    def throttle_reason(self) -> str | None:
        now = self.clock()
        self.timestamps = [t for t in self.timestamps if now - t < self.window_secs]
        if len(self.timestamps) >= self.max_per_window:
            return f"ORDER_RATE_THROTTLE({len(self.timestamps)}/min)"
        return None


@dataclass
class LegCheck:
    capped_qty: int
    max_expiry: date
    reasons: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    has_short_call: bool = False
    short_call_strike: float | None = None
    short_call_extrinsic: float | None = None


def check_legs(legs: Sequence[CandidateLegSpec], as_of: date, cfg: GateConfig) -> LegCheck:
    """Contract hygiene, liquidity and participation caps per leg; also finds the short call and last expiry."""
    out = LegCheck(capped_qty=min(leg.qty for leg in legs) if legs else 0, max_expiry=as_of)

    for leg in legs:
        if leg.expiry_date > out.max_expiry:
            out.max_expiry = leg.expiry_date
        if leg.multiplier != 100 or leg.is_adjusted:
            out.reasons.append(f"NON_STANDARD_CONTRACT({leg.instrument_id}, mult={leg.multiplier})")
        if not is_tradable_quote(leg.bid, leg.ask):
            out.reasons.append(f"INVALID_QUOTE({leg.instrument_id}: {leg.bid}/{leg.ask})")
            continue

        mid = 0.5 * (leg.bid + leg.ask)
        spread_pct = (leg.ask - leg.bid) / max(mid, 0.01)
        if spread_pct > cfg.max_leg_spread_pct:
            out.reasons.append(
                f"WIDE_MIDCAP_SPREAD({leg.instrument_id}: {spread_pct:.1%}>{cfg.max_leg_spread_pct:.1%})"
            )

        if leg.open_interest is None:
            if cfg.allow_unknown_open_interest:
                out.warnings.append(f"OPEN_INTEREST_UNKNOWN({leg.instrument_id})")
            else:
                out.reasons.append(f"OPEN_INTEREST_UNKNOWN({leg.instrument_id})")
        else:
            if leg.open_interest < cfg.min_open_interest:
                out.reasons.append(
                    f"LOW_OPEN_INTEREST({leg.instrument_id}: OI={leg.open_interest}<{cfg.min_open_interest})"
                )
            oi_cap = int(leg.open_interest * cfg.max_oi_participation)
            if oi_cap < 1:
                out.reasons.append(f"OI_PARTICIPATION_ZERO({leg.instrument_id})")
                out.capped_qty = 0
            elif oi_cap < out.capped_qty:
                out.capped_qty = oi_cap
                out.warnings.append(f"QTY_CAPPED_BY_OI({leg.instrument_id}: {oi_cap})")

        tob_size = leg.bid_size if leg.side == "SELL" else leg.ask_size
        if tob_size is None:
            if cfg.allow_unknown_tob_size:
                out.warnings.append(f"TOB_SIZE_UNKNOWN({leg.instrument_id})")
            else:
                out.reasons.append(f"TOB_SIZE_UNKNOWN({leg.instrument_id})")
        else:
            if tob_size < cfg.min_tob_size:
                out.reasons.append(
                    f"LOW_TOB_SIZE({leg.instrument_id}: size={tob_size}<{cfg.min_tob_size})"
                )
            tob_cap = int(tob_size * cfg.max_tob_participation)
            if tob_cap < 1:
                out.reasons.append(f"TOB_PARTICIPATION_ZERO({leg.instrument_id})")
                out.capped_qty = 0
            elif tob_cap < out.capped_qty:
                out.capped_qty = tob_cap
                out.warnings.append(f"QTY_CAPPED_BY_TOB({leg.instrument_id}: {tob_cap})")

        if leg.right == "C" and leg.side == "SELL":
            out.has_short_call = True
            out.short_call_strike = leg.strike
            out.short_call_extrinsic = leg.extrinsic_value

    return out


@dataclass(frozen=True)
class EsBudgetCheck:
    capped_qty: int
    reasons: list[str]
    warnings: list[str]
    projected_es99_usd: float


def es_budget_check(
    current_es99_usd: float,
    incremental_es99_per_lot_usd: float,
    capped_qty: int,
    cfg: GateConfig,
) -> EsBudgetCheck:
    """Caps lots so the book's ES99 stays inside the sleeve tail budget."""
    reasons: list[str] = []
    warnings: list[str] = []
    max_es99_usd = cfg.sleeve_equity_usd * cfg.max_book_es99_pct
    remaining_es99 = max(0.0, max_es99_usd - current_es99_usd)

    # Fail-closed: invalid (zero, negative, non-finite) incremental ES99 blocks entry
    if not math.isfinite(incremental_es99_per_lot_usd) or incremental_es99_per_lot_usd <= 0.0:
        reasons.append("INVALID_INCREMENTAL_ES99")
        return EsBudgetCheck(capped_qty=0, reasons=reasons, warnings=warnings, projected_es99_usd=current_es99_usd)

    es_qty_cap = int(remaining_es99 // incremental_es99_per_lot_usd)
    if es_qty_cap <= 0:
        reasons.append(
            f"SLEEVE_ES99_BUDGET_EXHAUSTED(cur=${current_es99_usd:.0f}+${incremental_es99_per_lot_usd:.0f}"
            f">${max_es99_usd:.0f})"
        )
    elif es_qty_cap < capped_qty:
            capped_qty = es_qty_cap
            warnings.append(f"QTY_CAPPED_BY_ES99({es_qty_cap} lots)")

    projected = current_es99_usd + max(0, capped_qty) * max(0.0, incremental_es99_per_lot_usd)
    return EsBudgetCheck(capped_qty=capped_qty, reasons=reasons, warnings=warnings, projected_es99_usd=projected)
