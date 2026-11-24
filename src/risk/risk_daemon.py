from __future__ import annotations

import argparse
import json
import os
import signal
import time
from collections.abc import Callable
from dataclasses import asdict
from typing import Any

from src.common import keys
from src.common.constants import DEFAULT_VALKEY_URL
from src.common.jsonl import JsonlSink, RecordSink
from src.common.store import JsonStore, make_store
from src.common.timeutil import iso_utc
from src.risk.daemon_rules import (
    TRANSITION,
    HEARTBEAT,
    RiskDaemonConfig,
    RiskEvaluation,
    RiskInputs,
    TransitionTracker,
    evaluate_risk,
    state_ttl_ms,
)

__all__ = ["OutOfProcessRiskDaemon", "RiskDaemonConfig", "RiskEvaluation", "main", "print_summary"]


class OutOfProcessRiskDaemon:
    """Independent OS process monitoring Valkey state, enforcing fail-closed halts, and recording history.

    Reads go through read_inputs(), the decision is the pure evaluate_risk(), and results are written with
    TTLs in one batch. The store, the JSONL sink, the wall clock and the sleep function are injectable.
    """

    def __init__(
        self,
        config: RiskDaemonConfig | None = None,
        *,
        store: JsonStore | None = None,
        sink: RecordSink | None = None,
        clock_ns: Callable[[], int] = time.time_ns,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.config = config or RiskDaemonConfig()
        self.store = store if store is not None else make_store(self.config.valkey_url)
        # The log directory is created on first write, not here
        self.sink = sink if sink is not None else JsonlSink(self.config.jsonl_path)
        self.clock_ns = clock_ns
        self.sleep = sleep
        self.transitions = TransitionTracker(self.config.heartbeat_log_interval_sec)
        self._running = True
        self._ttl_ms = state_ttl_ms(self.config)

    def stop(self, *_: Any) -> None:
        self._running = False

    def read_inputs(self) -> RiskInputs:
        store = self.store
        return RiskInputs(
            manual_halt_raw=store.get_text(keys.RISK_MANUAL_HALT),
            regime=store.get_json_or_none(keys.REGIME_LATEST),
            positions=store.get_json_or_none(keys.POSITIONS_LATEST),
            surfaces={key: store.get_json_or_none(key) for key in self.config.surface_keys},
            rvs={key: store.get_json_or_none(key) for key in self.config.rv_keys},
        )

    def evaluate_once(self, now_ns: int | None = None) -> RiskEvaluation:
        now_ns = now_ns if now_ns is not None else self.clock_ns()
        evaluation = evaluate_risk(self.read_inputs(), self.config, now_ns)
        self._publish(evaluation, now_ns)
        return evaluation

    def _publish(self, evaluation: RiskEvaluation, now_ns: int) -> None:
        payload = asdict(evaluation)
        payload_json = json.dumps(payload)

        batch = self.store.batch()
        batch.set_text(keys.RISK_HALTED, "1" if evaluation.halted else "0", ttl_ms=self._ttl_ms)
        batch.set_text(keys.RISK_STATUS, payload_json, ttl_ms=self._ttl_ms)
        batch.push_capped(keys.RISK_HISTORY, payload_json, self.config.history_max_len)

        kind, from_halted = self.transitions.classify(evaluation.halted, evaluation.reason_codes, now_ns)
        if kind == TRANSITION:
            transition_event = {
                "event": "STATE_CHANGE",
                "from_halted": from_halted,
                "to_halted": evaluation.halted,
                **payload,
            }
            batch.push_capped(keys.RISK_TRANSITIONS, transition_event, self.config.transitions_max_len)
            self.sink.write(transition_event)
        elif kind == HEARTBEAT:
            self.sink.write({"event": "HEARTBEAT", **payload})

        batch.execute()

    def record_error(self, exc: BaseException) -> dict[str, Any]:
        """Publishes a halted status for a failed evaluation; never raises."""
        now_ns = self.clock_ns()
        err_payload = {
            "event": "DAEMON_ERROR",
            "halted": True,
            "reasons": [f"daemon_error:{type(exc).__name__}:{exc}"],
            "reason_codes": ["daemon_error"],
            "evaluated_at_ns": now_ns,
            "evaluated_at_utc": iso_utc(now_ns),
        }
        # Ensures the recovery evaluation is logged as a transition
        self.transitions.mark_error()
        try:
            self.store.set_text(keys.RISK_HALTED, "1", ttl_ms=self._ttl_ms)
            self.store.set_json(keys.RISK_STATUS, err_payload, ttl_ms=self._ttl_ms)
            self.store.push_capped(keys.RISK_TRANSITIONS, err_payload, self.config.transitions_max_len)
        except Exception:
            pass  # Valkey unreachable: the TTL'd keys expire, which the gate treats as halted
        try:
            self.sink.write(err_payload)
        except Exception:
            pass  # Disk trouble must not stop the daemon from publishing halts
        return err_payload

    def run_forever(self, max_iterations: int | None = None, install_signal_handlers: bool = True) -> None:
        if install_signal_handlers:
            signal.signal(signal.SIGINT, self.stop)
            signal.signal(signal.SIGTERM, self.stop)
        iterations = 0
        while self._running and (max_iterations is None or iterations < max_iterations):
            try:
                self.evaluate_once()
            except Exception as exc:
                self.record_error(exc)
            iterations += 1
            self.sleep(self.config.poll_interval_sec)


def print_summary(target: str | JsonStore, echo: Callable[[str], None] = print) -> None:
    """Prints the current status and recent transitions from a Valkey URL or an existing store."""
    store = make_store(target) if isinstance(target, str) else target
    status = store.get_json_or_none(keys.RISK_STATUS)
    hist_len = store.list_len(keys.RISK_HISTORY)

    echo("=== Current Risk Status ===")
    echo(json.dumps(status, indent=2))
    echo(f"\n=== Rolling History Count: {hist_len} samples ===")
    echo("=== Recent State Transitions (newest first) ===")
    for t in store.get_json_list(keys.RISK_TRANSITIONS, 15):
        echo(
            f"  {t.get('evaluated_at_utc')} | halted={t.get('to_halted', t.get('halted'))} | "
            f"vix_ratio={t.get('vix_ratio')} | regime_age={t.get('regime_age_sec')}s | "
            f"reasons={t.get('reasons')}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Out-of-process fail-closed risk daemon")
    parser.add_argument("--summary", action="store_true", help="Print current status & recent transitions and exit")
    args = parser.parse_args()

    valkey_url = os.environ.get("VALKEY_URL", DEFAULT_VALKEY_URL)
    if args.summary:
        print_summary(valkey_url)
        return

    cfg = RiskDaemonConfig(
        valkey_url=valkey_url,
        poll_interval_sec=float(os.environ.get("RISK_POLL_INTERVAL_SEC", "2.0")),
        max_drawdown_pct=float(os.environ.get("RISK_MAX_DRAWDOWN_PCT", "-5.0")),
        vix_ratio_halt_threshold=float(os.environ.get("RISK_VIX_RATIO_THRESHOLD", "1.0")),
        state_ttl_sec=float(os.environ.get("RISK_STATE_TTL_SEC", "10.0")),
    )
    daemon = OutOfProcessRiskDaemon(cfg)
    daemon.run_forever()


if __name__ == "__main__":
    main()
