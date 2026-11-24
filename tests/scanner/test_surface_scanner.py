import pytest

from src.scanner.surface_scanner import SurfaceScanner
from tests.factories import regime_payload, surface_leg, surface_payload


def _legs(p685_raw_iv=0.16):
    return [
        surface_leg("P680", 680.0, "P", 1.0, 1.2),
        surface_leg("P685", 685.0, "P", 1.4, 1.6, raw_iv=p685_raw_iv),
        surface_leg("P690", 690.0, "P", 2.0, 2.2),
        surface_leg("P710", 710.0, "P", 11.0, 11.4),
        surface_leg("C720", 720.0, "C", 2.0, 2.2),
        surface_leg("C725", 725.0, "C", 1.3, 1.5),
        surface_leg("C730", 730.0, "C", 1.25, 1.35),
    ]


def scan(legs=None, regime=None, **kwargs):
    surface = surface_payload(legs=legs if legs is not None else _legs())
    return SurfaceScanner().scan_surface(surface, regime=regime, **kwargs)


def test_without_model_edge_candidates_rank_by_credit():
    combos = scan()
    assert [c.name for c in combos] == [
        "SPY.ARCA 685/690P-720/725C IC",
        "SPY.ARCA 720/725C Bear Call",
        "SPY.ARCA 690/685P Bull Put",
        "SPY.ARCA 685/680P Bull Put",
    ]


def test_iron_condor_uses_the_nearest_out_of_the_money_wings():
    condor = scan()[0]
    assert condor.strategy_type == "IRON_CONDOR"
    assert [(leg.instrument_id, leg.side) for leg in condor.legs] == [
        ("P685", 1), ("P690", -1), ("C720", -1), ("C725", 1),
    ]
    assert (condor.net_credit, condor.max_width) == (pytest.approx(1.3), 5.0)
    assert (condor.risk_free_rate, condor.dividend_yield, condor.spot) == (0.0425, 0.0125, 700.0)


def test_model_edge_reorders_candidates():
    combos = scan(legs=_legs(p685_raw_iv=0.40))  # Selling the rich 685 put earns 0.12/share of edge
    assert [c.name for c in combos] == [
        "SPY.ARCA 685/680P Bull Put",
        "SPY.ARCA 720/725C Bear Call",
        "SPY.ARCA 685/690P-720/725C IC",
        "SPY.ARCA 690/685P Bull Put",
    ]
    assert combos[0].theo_edge_per_share == pytest.approx(0.12)


def test_candidate_limit():
    assert len(scan(max_candidates=2)) == 2


def test_blocked_inputs_produce_nothing():
    assert SurfaceScanner().scan_surface(surface_payload(arb_free=False, legs=_legs())) == []
    assert scan(regime=regime_payload(short_vol_allowed=False)) == []
    assert len(scan(regime=regime_payload())) == 4


def test_condor_needs_two_wings_on_each_side():
    no_calls = [leg for leg in _legs() if leg["right"] == "P"]
    assert [c.strategy_type for c in scan(legs=no_calls)] == ["BULL_PUT_SPREAD", "BULL_PUT_SPREAD"]
