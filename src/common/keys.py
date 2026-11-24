"""Valkey key names shared by every producer and consumer."""

REGIME_LATEST = "mft:regime:latest"
FUTURES_LATEST = "mft:futures:latest"
RV_UNIVERSE = "mft:rv:universe"
PORTFOLIO_RISK = "mft:portfolio:risk"
POSITIONS_LATEST = "mft:positions:latest"
RISK_HALTED = "mft:risk:halted"
RISK_STATUS = "mft:risk:status"
RISK_HISTORY = "mft:risk:history"
RISK_TRANSITIONS = "mft:risk:transitions"
RISK_MANUAL_HALT = "mft:risk:manual_halt"
VRP_LATEST = "mft:strategy:vrp:latest"
VRP_HISTORY = "mft:strategy:vrp:history"
VERIFY_VOL_PIPELINE = "mft:verify:vol_pipeline"

SURFACE_PREFIX = "mft:surface:"
SURFACE_PATTERN = SURFACE_PREFIX + "*"


def surface_key(instrument_id: object) -> str:
    return f"{SURFACE_PREFIX}{instrument_id}"


def rv_key(instrument_id: object) -> str:
    return f"mft:rv:{instrument_id}"


def metrics_key(instrument_id: object) -> str:
    return f"mft:metrics:{instrument_id}"


def vrp_book_key(underlying: object) -> str:
    return f"mft:strategy:vrp:book:{underlying}"


def spread_legs_key(spread_id: object) -> str:
    return f"mft:spread_legs:{spread_id}"


def instrument_from_surface_key(key: str) -> str:
    """"mft:surface:SPY.ARCA" -> "SPY.ARCA"; any other key is returned unchanged."""
    return key[len(SURFACE_PREFIX):] if key.startswith(SURFACE_PREFIX) else key
