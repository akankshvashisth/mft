"""Live dashboard state read from Valkey through a JsonStore."""
from __future__ import annotations

from typing import Any

from src.common import keys
from src.common.flags import is_truthy_flag
from src.common.freshness import DISPLAY_TS_FIELDS, payload_age_secs
from src.common.store import JsonStore

DEFAULT_SURFACE_KEY = keys.surface_key("SPY.ARCA")
RISK_HISTORY_COUNT = 200
RISK_TRANSITIONS_COUNT = 30


def surface_keys(store: JsonStore) -> list[str]:
    """Published surface keys, found with SCAN (KEYS would block Valkey)."""
    return store.list_keys(keys.SURFACE_PATTERN)


def load_live_state(store: JsonStore, surface_key: str) -> dict[str, Any]:
    """Every snapshot the pages show; missing or invalid payloads are None (lists are empty)."""
    underlying = keys.instrument_from_surface_key(surface_key)
    return {
        "regime": store.get_json_or_none(keys.REGIME_LATEST),
        "futures": store.get_json_or_none(keys.FUTURES_LATEST),
        "vrp_strat": store.get_json_or_none(keys.VRP_LATEST),
        "surface": store.get_json_or_none(surface_key),
        "rv_primary": store.get_json_or_none(keys.rv_key(underlying)),
        "rv_universe": store.get_json_or_none(keys.RV_UNIVERSE),
        "metrics": store.get_json_or_none(keys.metrics_key(underlying)),
        "portfolio": store.get_json_or_none(keys.PORTFOLIO_RISK),
        "risk_status": store.get_json_or_none(keys.RISK_STATUS),
        "risk_history": store.get_json_list(keys.RISK_HISTORY, RISK_HISTORY_COUNT),
        "risk_transitions": store.get_json_list(keys.RISK_TRANSITIONS, RISK_TRANSITIONS_COUNT),
    }


def age_seconds(payload: dict[str, Any] | None, now_ns: int) -> float | None:
    """Displayed payload age; None when the payload has no usable timestamp (missing or far in the future)."""
    age = payload_age_secs(payload, now_ns, DISPLAY_TS_FIELDS)
    return round(age, 1) if age is not None else None


def manual_halt_engaged(store: JsonStore) -> bool:
    return is_truthy_flag(store.get_text(keys.RISK_MANUAL_HALT))


def set_manual_halt(store: JsonStore, engaged: bool) -> None:
    """Engages ("1") or releases (deletes) the manual kill switch read by the risk daemon."""
    if engaged:
        store.set_text(keys.RISK_MANUAL_HALT, "1")
    else:
        store.delete(keys.RISK_MANUAL_HALT)
