import pytest

from src.scanner.verticals import (
    BEAR_CALL_SPREAD,
    BULL_PUT_SPREAD,
    adjacent_verticals,
    candidate_leg,
    leg_vol_edge,
    mean_edge_to_half_spread,
    split_by_right,
    vertical_combo,
    vertical_name,
)
from tests.factories import surface_leg

PUTS = [
    surface_leg("P680", 680.0, "P", 1.0, 1.2),
    surface_leg("P685", 685.0, "P", 1.4, 1.6, raw_iv=0.18, edge_to_half_spread=0.4),
    surface_leg("P690", 690.0, "P", 2.0, 2.2, edge_to_half_spread=0.2),
    surface_leg("P710", 710.0, "P", 11.0, 11.4),  # 20 wide from 690: too wide
]
CALLS = [
    surface_leg("C720", 720.0, "C", 2.0, 2.2),
    surface_leg("C725", 725.0, "C", 1.3, 1.5),
    surface_leg("C730", 730.0, "C", 1.25, 1.35),  # Only 0.10 below C725's mid
]


def _summary(pairs):
    return [(p.kind, p.short["strike"], p.long["strike"]) for p in pairs]


def test_split_by_right_sorts_by_strike():
    puts, calls = split_by_right(list(reversed(PUTS + CALLS)))
    assert [r["strike"] for r in puts] == [680.0, 685.0, 690.0, 710.0]
    assert [r["strike"] for r in calls] == [720.0, 725.0, 730.0]


def test_adjacent_verticals_with_width_and_credit_filters():
    pairs = adjacent_verticals(PUTS, CALLS, min_mid_credit=0.15)
    assert _summary(pairs) == [
        (BULL_PUT_SPREAD, 685.0, 680.0),
        (BULL_PUT_SPREAD, 690.0, 685.0),
        (BEAR_CALL_SPREAD, 720.0, 725.0),
    ]
    first = pairs[0]
    assert (first.width, first.mid_credit, first.nat_credit) == (5.0, pytest.approx(0.4), pytest.approx(0.2))


def test_credit_threshold_is_exclusive_and_configurable():
    assert len(adjacent_verticals(PUTS, CALLS, min_mid_credit=0.0)) == 4
    assert _summary(adjacent_verticals(PUTS, CALLS, min_mid_credit=0.5)) == [
        (BULL_PUT_SPREAD, 690.0, 685.0),
        (BEAR_CALL_SPREAD, 720.0, 725.0),
    ]
    assert adjacent_verticals(PUTS, CALLS, max_width=4.0) == []


def test_duplicate_strikes_never_pair():
    twins = [surface_leg("A", 690.0, "P", 1.0, 1.2), surface_leg("B", 690.0, "P", 2.0, 2.2)]
    assert adjacent_verticals(twins, []) == []


def test_names():
    bull, _, bear = adjacent_verticals(PUTS, CALLS, min_mid_credit=0.15)
    assert vertical_name("SPY", bull) == "SPY 685/680P Bull Put"
    assert vertical_name("SPY", bear) == "SPY 720/725C Bear Call"


def test_candidate_leg_copies_the_row():
    leg = candidate_leg(PUTS[1], -1)
    assert (leg.instrument_id, leg.strike, leg.right, leg.side, leg.qty) == ("P685", 685.0, "P", -1, 1)
    assert (leg.bid, leg.ask, leg.mid, leg.raw_iv, leg.svi_iv) == (1.4, 1.6, 1.5, 0.18, 0.16)


def test_edge_helpers():
    assert leg_vol_edge(PUTS[1]) == pytest.approx(0.02 * 0.5)
    assert leg_vol_edge(PUTS[0]) == 0.0
    assert mean_edge_to_half_spread(PUTS[1], PUTS[2]) == pytest.approx(0.3)


def test_vertical_combo_fields():
    pair = adjacent_verticals(PUTS, CALLS, min_mid_credit=0.15)[0]
    combo = vertical_combo(
        pair, name_symbol="SPY", underlying="SPY.ARCA", spot=700.0, forward=701.0,
        ttm_days=24.0, risk_free_rate=0.0425, dividend_yield=0.0125,
    )
    assert combo.name == "SPY 685/680P Bull Put"
    assert combo.strategy_type == BULL_PUT_SPREAD
    assert [(leg.instrument_id, leg.side) for leg in combo.legs] == [("P685", -1), ("P680", 1)]
    assert (combo.net_credit, combo.max_width) == (0.4, 5.0)
    assert combo.theo_edge_per_share == pytest.approx(0.01)
    assert combo.avg_edge_to_half_spread == pytest.approx(0.2)
def test_delta_targeted_verticals_filters_by_width_and_delta():
    """$1-spaced adjacent puts on a $40 mid-cap skip $1/2-wide pairs; need >= $2.50."""
    from src.scanner.verticals import delta_targeted_verticals

    puts = [
        surface_leg("P36", 36.0, "P", 0.1, 0.3),
        surface_leg("P37", 37.0, "P", 0.2, 0.4),
        surface_leg("P38", 38.0, "P", 0.3, 0.5),
        surface_leg("P40", 40.0, "P", 0.5, 0.7),  # ATM — not OTM
        surface_leg("P41", 41.0, "P", 0.6, 0.8),  # ITM — not OTM
    ]
    # OTM puts at P36-P38. Widths: (37,36)=1, (38,37)=1, (38,36)=2. All < $2.50.
    pairs = delta_targeted_verticals(puts, [], spot=40.0, min_width_usd=2.50,
                                     short_delta_range=(0.15, 0.30),
                                     min_width_pct_spot=0.04, max_width_pct_spot=0.12)
    assert len(pairs) == 0, f"Expected no pairs (all widths < $2.50), got {len(pairs)}"

    # With adjacent_verticals and only OTM put (P36-P38), $1-wide pairs still appear
    from src.scanner.verticals import adjacent_verticals
    adjacent = adjacent_verticals(puts[:3], [], min_mid_credit=0.0)
    assert len(adjacent) > 0  # $1-wide pairs exist


def test_delta_targeted_otm_puts_only():
    """ITM puts (strike > spot) are never selected by delta_targeted_verticals."""
    from src.scanner.verticals import delta_targeted_verticals

    # P34 ($34, OTM) and P37 ($37, OTM) at spot=40
    # Add P43 ($43, ITM — strike > 40) to verify it's excluded as short leg
    puts = [
        surface_leg("P34", 34.0, "P", 0.6, 0.8),
        surface_leg("P37", 37.0, "P", 0.7, 0.9),
        surface_leg("P43", 43.0, "P", 0.9, 1.1),
    ]
    pairs = delta_targeted_verticals(puts, [], spot=40.0, min_width_usd=2.50,
                                     short_delta_range=(0.0, 1.0),  # Wide delta range for testing
                                     min_width_pct_spot=0.0, max_width_pct_spot=0.30)
    # P43 has strike=43 > spot=40 → rejected as short leg. P34' short=37 or 34 are OTM → kept
    short_strikes = [p.short["strike"] for p in pairs]
    assert 43.0 not in short_strikes, "ITM put P43 should not appear as short leg"
    assert len(pairs) >= 1, "OTM pairs should be selected"
