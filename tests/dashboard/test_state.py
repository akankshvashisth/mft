from unittest import mock

import pytest

from src.common import keys
from src.dashboard.state import (
    age_seconds,
    load_live_state,
    manual_halt_engaged,
    set_manual_halt,
    surface_keys,
)
from tests.factories import NOW_NS, UNDERLYING, ago, risk_status_payload, surface_payload

SEC = 1_000_000_000


def test_surface_keys_use_scan_not_keys(store, fake_valkey):
    store.set_json(keys.surface_key("SPY.ARCA"), surface_payload())
    store.set_json(keys.surface_key("QQQ.NASDAQ"), surface_payload("QQQ.NASDAQ"))
    store.set_json(keys.REGIME_LATEST, {})

    with mock.patch.object(fake_valkey, "keys", side_effect=AssertionError("KEYS blocks Valkey")):
        assert surface_keys(store) == ["mft:surface:QQQ.NASDAQ", "mft:surface:SPY.ARCA"]


def test_live_state_reads_every_snapshot(healthy_store):
    healthy_store.push_capped(keys.RISK_HISTORY, risk_status_payload(), 10)
    healthy_store.push_capped(keys.RISK_TRANSITIONS, {"event": "STATE_CHANGE"}, 10)

    state = load_live_state(healthy_store, keys.surface_key(UNDERLYING))

    assert state["surface"]["underlying"] == UNDERLYING
    assert state["rv_primary"]["symbol"] == UNDERLYING
    assert state["metrics"]["underlying"] == UNDERLYING
    assert state["regime"]["ratio"] == 0.8824
    assert state["risk_status"]["halted"] is False
    assert len(state["risk_history"]) == 1 and state["risk_transitions"] == [{"event": "STATE_CHANGE"}]
    assert state["futures"]["cross_check_valid"] is True
    assert state["portfolio"]["risk"] == {"book_es_99_usd": 0.0}
    assert state["vrp_strat"] is None and state["rv_universe"] is None


def test_live_state_of_an_empty_store(store):
    state = load_live_state(store, keys.surface_key(UNDERLYING))
    assert state["surface"] is None and state["risk_history"] == [] and state["risk_transitions"] == []


@pytest.mark.parametrize(
    "payload, age",
    [
        ({"ts_ns": ago(30.04)}, 30.0),
        ({"event_ts_ns": ago(12.0)}, 12.0),
        ({"ts_event": ago(5.0)}, 5.0),
        ({"ts_ns": NOW_NS + 60 * SEC}, 0.0),  # Small clock skew
        ({"ts_ns": NOW_NS + 3_600 * SEC}, None),  # Far future: unusable
        ({"ts_ns": 12345}, None),
        ({}, None),
        (None, None),
    ],
)
def test_age_seconds(payload, age):
    assert age_seconds(payload, NOW_NS) == age


@pytest.mark.parametrize("raw, engaged", [("1", True), ("yes", True), (" ON ", True), ("0", False), (None, False)])
def test_manual_halt_flag(store, raw, engaged):
    if raw is not None:
        store.set_text(keys.RISK_MANUAL_HALT, raw)
    assert manual_halt_engaged(store) is engaged


def test_manual_halt_engage_and_release(store, fake_valkey):
    set_manual_halt(store, True)
    assert fake_valkey.get(keys.RISK_MANUAL_HALT) == b"1"
    set_manual_halt(store, False)
    assert not fake_valkey.exists(keys.RISK_MANUAL_HALT)
