"""Two-tier mid-cap universe rotator: select an IBKR 100-line streaming subset from 100-300+ candidates.

Pure module — zero broker or cache imports.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

MIN_STREAMING_LINES = 4  # Fixed lines reserved even for rank-and-file symbols


@dataclass(frozen=True)
class UniverseCandidate:
    """One mid-cap symbol the scanner could stream quotes for.

    ``has_open_position`` is always True for symbols with an open spread — these are never dropped.
    ``earnings_within_window``, ``median_rel_spread``, and ``min_oi`` are used for filtering.
    ``vrp_score`` and ``liquidity_score`` drive the composite ranking.
    ``sector`` enables the sector-share cap.
    """
    symbol: str
    atm_iv: float = 0.20
    rv_20d: float = 0.15
    vrp_ratio: float = 0.05          # (atm_iv - rv_20d) / rv_20d or similar
    rr25_pct: float = 0.03           # Risk-reversal at 25 delta (models edge)
    median_rel_spread: float = 0.12  # Typical leg spread / mid
    min_oi: int = 500
    sector: str = ""
    has_open_position: bool = False
    earnings_within_window: bool = False


@dataclass
class RotationPlan:
    """Result of ``select_active_streaming_universe``."""
    active_symbols: list[str] = field(default_factory=list)
    total_lines: int = 0
    dropped: list[str] = field(default_factory=list)
    fixed_symbols: list[str] = field(default_factory=list)


def _composite_score(cand: UniverseCandidate, sector_weight: float = 0.10) -> float:
    """Composite VRP + liquidity ranking score (higher = better)."""
    vrp_score = cand.vrp_ratio * 100.0  # Scale to %
    liq_score = max(0.0, 1.0 - cand.median_rel_spread) * 10.0
    return vrp_score + liq_score


def select_active_streaming_universe(
    candidates: list[UniverseCandidate],
    *,
    max_lines: int = 90,
    lines_per_symbol: int = 8,
    max_sector_share: float = 0.30,
    fixed_lines: int = MIN_STREAMING_LINES,
    max_rel_spread: float = 0.18,
    min_oi: int = 100,
) -> RotationPlan:
    """Select a streaming universe from ``candidates`` that fits inside IBKR's 100-line cap.

    Always reserves lines for **all open positions first** (never drop quotes for an open spread).
    Filters out candidates with earnings inside the window, ``median_rel_spread > max_rel_spread``,
    or ``min_oi < min_oi``.  Ranks remaining candidates by composite VRP + liquidity score while
    enforcing ``max_sector_share`` cap.  Guarantees ``total_lines <= max_lines``.
    """
    plan = RotationPlan()

    # 1. Always keep open positions
    fixed: list[UniverseCandidate] = [c for c in candidates if c.has_open_position]
    plan.fixed_symbols = [c.symbol for c in fixed]
    remaining_budget = max_lines - len(fixed) * lines_per_symbol
    if remaining_budget <= 0:
        plan.active_symbols = plan.fixed_symbols
        plan.total_lines = len(plan.fixed_symbols) * lines_per_symbol
        plan.dropped = [c.symbol for c in candidates if not c.has_open_position]
        return plan

    # 2. Filter out candidates with earnings, wide spreads, or low OI
    filtered = [
        c for c in candidates
        if not c.has_open_position
        and not c.earnings_within_window
        and c.median_rel_spread <= max_rel_spread
        and c.min_oi >= min_oi
    ]
    plan.dropped = [c.symbol for c in candidates if not c.has_open_position
                and (c.earnings_within_window or c.median_rel_spread > max_rel_spread or c.min_oi < min_oi)]

    # 3. Rank by composite score, enforce sector cap
    ranked = sorted(filtered, key=_composite_score, reverse=True)
    sector_counts: dict[str, int] = {}
    selected: list[UniverseCandidate] = []
    for cand in ranked:
        lines_needed = lines_per_symbol if cand.symbol not in plan.fixed_symbols else 0
        if lines_needed > remaining_budget:
            continue
        if cand.sector:
            current_sector = sector_counts.get(cand.sector, 0)
            max_sector_lines = int(max_lines * max_sector_share)
            if (current_sector + 1) * lines_per_symbol > max_sector_lines:
                plan.dropped.append(cand.symbol)
                continue
            sector_counts[cand.sector] = current_sector + 1
        selected.append(cand)
        remaining_budget -= lines_needed

    # Track sector-capped and budget-excluded symbols as dropped
    all_selected_symbols = set(plan.fixed_symbols) | {c.symbol for c in selected}
    sector_budget_dropped = [
        c.symbol for c in candidates
        if c.symbol not in all_selected_symbols and c.symbol not in plan.dropped
    ]
    plan.dropped.extend(sector_budget_dropped)

    plan.active_symbols = plan.fixed_symbols + [c.symbol for c in selected]
    plan.total_lines = len(plan.active_symbols) * lines_per_symbol
    return plan