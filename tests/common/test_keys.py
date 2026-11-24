from src.common import keys


def test_key_builders_match_the_published_key_names():
    assert keys.surface_key("SPY.ARCA") == "mft:surface:SPY.ARCA"
    assert keys.rv_key("SPY.ARCA") == "mft:rv:SPY.ARCA"
    assert keys.metrics_key("SPY.ARCA") == "mft:metrics:SPY.ARCA"
    assert keys.vrp_book_key("SPY.ARCA") == "mft:strategy:vrp:book:SPY.ARCA"
    assert keys.spread_legs_key("SPREAD-1") == "mft:spread_legs:SPREAD-1"


def test_fixed_key_names_are_stable():
    assert (
        keys.REGIME_LATEST,
        keys.FUTURES_LATEST,
        keys.RV_UNIVERSE,
        keys.PORTFOLIO_RISK,
        keys.POSITIONS_LATEST,
        keys.RISK_HALTED,
        keys.RISK_STATUS,
        keys.RISK_HISTORY,
        keys.RISK_TRANSITIONS,
        keys.RISK_MANUAL_HALT,
        keys.VRP_LATEST,
        keys.VRP_HISTORY,
        keys.SURFACE_PATTERN,
    ) == (
        "mft:regime:latest",
        "mft:futures:latest",
        "mft:rv:universe",
        "mft:portfolio:risk",
        "mft:positions:latest",
        "mft:risk:halted",
        "mft:risk:status",
        "mft:risk:history",
        "mft:risk:transitions",
        "mft:risk:manual_halt",
        "mft:strategy:vrp:latest",
        "mft:strategy:vrp:history",
        "mft:surface:*",
    )


def test_instrument_from_surface_key():
    assert keys.instrument_from_surface_key("mft:surface:SPY.ARCA") == "SPY.ARCA"
    assert keys.instrument_from_surface_key("other") == "other"
