import json
from unittest import mock

import pytest

from src.common import keys
from src.common.store import JsonStore, NullStore
from src.risk.daemon_rules import RiskDaemonConfig
from src.risk.risk_daemon import OutOfProcessRiskDaemon, print_summary
from tests.factories import NOW_NS, ago, regime_payload
from tests.fakes import recording_sink, valkey_client, written

SEC = 1_000_000_000
TTL_MS = 10_000


@pytest.fixture
def make_daemon(healthy_store, mem_sink, wall_clock):
    def make(config=None, store=None, sink=None, sleep=lambda secs: None):
        return OutOfProcessRiskDaemon(
            config or RiskDaemonConfig(),
            store=store if store is not None else healthy_store,
            sink=sink if sink is not None else mem_sink,
            clock_ns=wall_clock,
            sleep=sleep,
        )

    return make


def events(sink):
    return [record["event"] for record in written(sink)]


def test_healthy_evaluation_publishes_state_with_ttl(make_daemon, healthy_store, fake_valkey, mem_sink):
    with mock.patch.object(fake_valkey, "pipeline", wraps=fake_valkey.pipeline) as pipeline:
        ev = make_daemon().evaluate_once()

    assert not ev.halted
    pipeline.assert_called_once()  # One batched publish
    assert fake_valkey.get(keys.RISK_HALTED) == b"0"
    assert 0 < fake_valkey.pttl(keys.RISK_HALTED) <= TTL_MS
    assert 0 < fake_valkey.pttl(keys.RISK_STATUS) <= TTL_MS
    status = healthy_store.get_json_or_none(keys.RISK_STATUS)
    assert (status["halted"], status["evaluated_at_ns"], status["reason_codes"]) == (False, NOW_NS, [])
    assert fake_valkey.llen(keys.RISK_HISTORY) == 1

    (transition,) = healthy_store.get_json_list(keys.RISK_TRANSITIONS, 10)
    assert (transition["event"], transition["from_halted"], transition["to_halted"]) == ("STATE_CHANGE", None, False)
    assert events(mem_sink) == ["STATE_CHANGE"]


def test_history_is_capped_newest_first(make_daemon, healthy_store, wall_clock):
    daemon = make_daemon(RiskDaemonConfig(history_max_len=3))
    for _ in range(5):
        daemon.evaluate_once()
        wall_clock.return_value += SEC

    history = healthy_store.get_json_list(keys.RISK_HISTORY, 10)
    assert [h["evaluated_at_ns"] for h in history] == [NOW_NS + 4 * SEC, NOW_NS + 3 * SEC, NOW_NS + 2 * SEC]


def test_transitions_are_logged_only_when_reason_codes_change(make_daemon, healthy_store, fake_valkey, wall_clock):
    daemon = make_daemon()
    for _ in range(3):
        daemon.evaluate_once()
    assert fake_valkey.llen(keys.RISK_TRANSITIONS) == 1

    healthy_store.set_text(keys.RISK_MANUAL_HALT, "1")
    daemon.evaluate_once()
    daemon.evaluate_once()
    assert fake_valkey.llen(keys.RISK_TRANSITIONS) == 2
    assert fake_valkey.get(keys.RISK_HALTED) == b"1"

    healthy_store.delete(keys.RISK_MANUAL_HALT)
    healthy_store.set_json(keys.REGIME_LATEST, regime_payload(now_ns=ago(121.0)))
    daemon.evaluate_once()  # New code: stale_regime_signal
    wall_clock.return_value += 5 * SEC
    daemon.evaluate_once()  # Older still, same code
    assert fake_valkey.llen(keys.RISK_TRANSITIONS) == 3


def test_heartbeats_go_to_the_sink_between_transitions(make_daemon, mem_sink, wall_clock):
    daemon = make_daemon(RiskDaemonConfig(heartbeat_log_interval_sec=60.0))
    daemon.evaluate_once()
    wall_clock.return_value += 30 * SEC
    daemon.evaluate_once()
    wall_clock.return_value += 30 * SEC
    daemon.evaluate_once()

    assert events(mem_sink) == ["STATE_CHANGE", "HEARTBEAT"]


def test_unusable_timestamp_halts_end_to_end(make_daemon, healthy_store, fake_valkey):
    healthy_store.set_json(keys.REGIME_LATEST, regime_payload(ts_ns=NOW_NS + 3_600 * SEC))
    ev = make_daemon().evaluate_once()

    assert ev.reason_codes == [f"invalid_timestamp:{keys.REGIME_LATEST}"]
    assert fake_valkey.get(keys.RISK_HALTED) == b"1"


def test_evaluation_error_publishes_a_halt_and_recovery_is_a_transition(make_daemon, healthy_store, fake_valkey, mem_sink):
    daemon = make_daemon()
    daemon.read_inputs = mock.Mock(side_effect=RuntimeError("boom"))

    daemon.run_forever(max_iterations=1, install_signal_handlers=False)

    assert fake_valkey.get(keys.RISK_HALTED) == b"1"
    assert 0 < fake_valkey.pttl(keys.RISK_HALTED) <= TTL_MS
    status = healthy_store.get_json_or_none(keys.RISK_STATUS)
    assert status["reason_codes"] == ["daemon_error"]
    assert status["reasons"] == ["daemon_error:RuntimeError:boom"]
    assert healthy_store.get_json_list(keys.RISK_TRANSITIONS, 1)[0]["event"] == "DAEMON_ERROR"

    del daemon.read_inputs  # Back to the real reads
    daemon.evaluate_once()
    latest = healthy_store.get_json_list(keys.RISK_TRANSITIONS, 1)[0]
    assert (latest["event"], latest["from_halted"], latest["to_halted"]) == ("STATE_CHANGE", True, False)
    assert events(mem_sink) == ["DAEMON_ERROR", "STATE_CHANGE"]


def test_daemon_survives_an_unreachable_store_and_a_failing_disk(make_daemon):
    sleep = mock.Mock()
    daemon = make_daemon(
        store=JsonStore(valkey_client(connected=False)), sink=recording_sink(OSError("disk unavailable")), sleep=sleep
    )

    daemon.run_forever(max_iterations=3, install_signal_handlers=False)

    assert sleep.call_args_list == [mock.call(2.0)] * 3


def test_error_events_reach_the_sink_when_the_store_is_down(make_daemon, mem_sink):
    daemon = make_daemon(store=JsonStore(valkey_client(connected=False)))
    daemon.run_forever(max_iterations=2, install_signal_handlers=False)

    assert events(mem_sink) == ["DAEMON_ERROR", "DAEMON_ERROR"]
    assert written(mem_sink)[0]["reasons"][0].startswith("daemon_error:ConnectionError:")


def test_stop_ends_the_loop(make_daemon):
    holder = {}
    daemon = make_daemon(sleep=lambda secs: holder["daemon"].stop())
    holder["daemon"] = daemon

    daemon.run_forever(install_signal_handlers=False)  # Would loop forever without stop()


def test_constructor_does_not_touch_the_filesystem(tmp_path, healthy_store):
    log_path = tmp_path / "logs" / "risk_history.jsonl"
    daemon = OutOfProcessRiskDaemon(RiskDaemonConfig(jsonl_path=str(log_path)), store=healthy_store)
    assert not log_path.parent.exists()

    daemon.evaluate_once(now_ns=NOW_NS)
    lines = log_path.read_text(encoding="utf-8").splitlines()
    assert [json.loads(line)["event"] for line in lines] == ["STATE_CHANGE"]


def test_print_summary(make_daemon, healthy_store):
    make_daemon().evaluate_once()
    echo = mock.Mock()
    print_summary(healthy_store, echo=echo)

    lines = [call.args[0] for call in echo.call_args_list]
    assert lines[0] == "=== Current Risk Status ==="
    assert json.loads(lines[1])["halted"] is False
    assert lines[2] == "\n=== Rolling History Count: 1 samples ==="
    assert len(lines) == 5  # Header lines + one transition
    assert "halted=False" in lines[4]


def test_print_summary_without_valkey():
    echo = mock.Mock()
    print_summary(NullStore(), echo=echo)
    assert echo.call_args_list[:3] == [
        mock.call("=== Current Risk Status ==="),
        mock.call("null"),
        mock.call("\n=== Rolling History Count: 0 samples ==="),
    ]
