import pytest

from src.scanner.universe_rotator import (
    UniverseCandidate,
    select_active_streaming_universe,
)


def _make(symbol, sector="", has_open=False, earnings=False, spread=0.12, oi=500):
    return UniverseCandidate(
        symbol=symbol, sector=sector, has_open_position=has_open,
        earnings_within_window=earnings, median_rel_spread=spread, min_oi=oi,
        vrp_ratio=0.05,
    )


def test_open_positions_are_always_retained():
    """3 open positions are included even when 200 candidates compete."""
    candidates = [_make(f"S{i}", has_open=(i < 3)) for i in range(200)]
    plan = select_active_streaming_universe(candidates)

    for sym in [c.symbol for c in candidates[:3]]:
        assert sym in plan.active_symbols, f"Open position {sym} was dropped"
    assert plan.total_lines <= 90


def test_earnings_blackout_excludes():
    """Candidates with earnings inside the window are dropped."""
    candidates = [_make("A"), _make("B", earnings=True)]
    plan = select_active_streaming_universe(candidates)
    assert "B" not in plan.active_symbols
    assert "B" in plan.dropped


def test_wide_spread_filter():
    """Candidates with median_rel_spread > max_rel_spread are dropped."""
    candidates = [_make("A", spread=0.12), _make("B", spread=0.25)]
    plan = select_active_streaming_universe(candidates, max_rel_spread=0.18)
    assert "B" not in plan.active_symbols
    assert "A" in plan.active_symbols


def test_low_oi_filter():
    """Candidates with min_oi < min_oi are dropped."""
    candidates = [_make("A", oi=500), _make("B", oi=50)]
    plan = select_active_streaming_universe(candidates, min_oi=100)
    assert "B" not in plan.active_symbols


def test_sector_cap_enforced():
    """Sector share cap of 30% on 90 lines = max 3 symbols per sector (3 lines each, 8/symbol = 24 lines out of 27 sector limit)."""
    # Actually with max_lines=90 and lines_per_symbol=8, 30% = 27 lines = 3.375 symbols → 3 max
    candidates = [_make(f"S{i}", sector="TECH") for i in range(20)]
    plan = select_active_streaming_universe(candidates, max_lines=90, lines_per_symbol=8, max_sector_share=0.30)
    tech_count = sum(1 for c in plan.active_symbols if "TECH" in [cand.sector for cand in []]  # approximate check
                     or len(plan.active_symbols) <= 3)
    # With only one sector and 30% cap on 90 lines = 27 lines → 3 symbols max
    assert len(plan.active_symbols) <= 3, f"Sector cap exceeded: {len(plan.active_symbols)} symbols"


def test_total_lines_strictly_under_limit():
    """total_lines never exceeds max_lines (90 by default)."""
    candidates = [_make(f"S{i}") for i in range(100)]
    plan = select_active_streaming_universe(candidates)
    assert plan.total_lines <= 90, f"Total lines {plan.total_lines} > 90"
    assert plan.total_lines >= 0


def test_no_open_positions_and_many_candidates():
    """With no open positions, the rotator still selects a ranked subset within line budget."""
    candidates = [_make(f"S{i}") for i in range(50)]
    plan = select_active_streaming_universe(candidates)
    assert len(plan.active_symbols) <= 11  # 11 * 8 = 88 <= 90
    assert plan.total_lines > 0
    assert len(plan.dropped) > 0