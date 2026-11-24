from __future__ import annotations

import argparse
import json
import os
import signal
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import redis


@dataclass
class RiskDaemonConfig:
    valkey_url: str = "redis://127.0.0.1:6379/0"
    poll_interval_sec: float = 2.0
    max_regime_age_sec: float = 120.0
    max_surface_age_sec: float = 120.0
    max_rv_age_sec: float = 86400.0
    max_drawdown_pct: float = -5.0
    vix_ratio_halt_threshold: float = 1.0
    surface_keys: tuple[str, ...] = ("mft:surface:SPY.ARCA",)
    rv_keys: tuple[str, ...] = ("mft:rv:SPY.ARCA",)
    history_max_len: int = 2000
    transitions_max_len: int = 500
    jsonl_path: str = "logs/risk_history.jsonl"
    heartbeat_log_interval_sec: float = 60.0


@dataclass
class RiskEvaluation:
    halted: bool
    reasons: list[str]
    evaluated_at_ns: int
    evaluated_at_utc: str
    sleeve_drawdown_pct: float | None
    vix_ratio: float | None
    regime_age_sec: float | None
    surface_ages_sec: dict[str, float | None]


def _safe_json_load(raw: str | bytes | None) -> dict[str, Any] | None:
    if raw is None:
        return None
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8")
    try:
        data = json.loads(raw)
        return data if isinstance(data, dict) else None
    except json.JSONDecodeError:
        return None


def _extract_age_sec(payload: dict[str, Any], now_ns: int) -> float | None:
    for key in ("ts_ns", "ts_event", "ts_init", "updated_at_ns"):
        val = payload.get(key)
        if isinstance(val, (int, float)) and val > 1e15:
            delta_sec = (now_ns - float(val)) / 1e9
            if delta_sec < 0.0 and delta_sec >= -900.0:
                return 0.0
            return delta_sec
    return None


def _iso_utc(now_ns: int) -> str:
    return datetime.fromtimestamp(now_ns / 1e9, tz=timezone.utc).isoformat()


class OutOfProcessRiskDaemon:
    """Independent OS process monitoring Valkey state, enforcing fail-closed halts, and recording history."""

    def __init__(self, config: RiskDaemonConfig | None = None) -> None:
        self.config = config or RiskDaemonConfig()
        self.rdb = redis.from_url(self.config.valkey_url)
        self._running = True
        self._prev_halted: bool | None = None
        self._prev_reasons: list[str] | None = None
        self._last_disk_write_ns: int = 0

        Path(self.config.jsonl_path).parent.mkdir(parents=True, exist_ok=True)

    def stop(self, *_: Any) -> None:
        self._running = False

    def _append_jsonl(self, record: dict[str, Any]) -> None:
        with open(self.config.jsonl_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

    def evaluate_once(self, now_ns: int | None = None) -> RiskEvaluation:
        now_ns = now_ns if now_ns is not None else time.time_ns()
        reasons: list[str] = []

        # 1. Explicit manual kill-switch check (never auto-cleared by daemon)
        manual_halt = self.rdb.get("mft:risk:manual_halt")
        if manual_halt and manual_halt.decode("utf-8").strip().lower() in {"1", "true", "yes", "on"}:
            reasons.append("manual_kill_switch_active")

        # 2. Regime & VIX/VIX3M backwardation check
        regime = _safe_json_load(self.rdb.get("mft:regime:latest"))
        vix_ratio: float | None = None
        regime_age_sec: float | None = None
        if regime is None:
            reasons.append("missing_regime_signal")
        else:
            age = _extract_age_sec(regime, now_ns)
            if age is not None:
                regime_age_sec = round(age, 2)
                if age > self.config.max_regime_age_sec:
                    reasons.append(f"stale_regime_signal:{age:.1f}s")

            raw_ratio = regime.get("vix_ratio")
            if raw_ratio is None:
                raw_ratio = regime.get("ratio")
            if raw_ratio is None:
                vixy = regime.get("vixy_price")
                vixm = regime.get("vixm_price")
                if isinstance(vixy, (int, float)) and isinstance(vixm, (int, float)) and vixm > 0:
                    raw_ratio = float(vixy) / float(vixm)
            if isinstance(raw_ratio, (int, float)):
                vix_ratio = float(raw_ratio)
                if vix_ratio > self.config.vix_ratio_halt_threshold:
                    reasons.append(f"vix_backwardation:{vix_ratio:.4f}>{self.config.vix_ratio_halt_threshold}")

        # 3. Sleeve drawdown check from positions / equity snapshot
        sleeve_dd_pct: float | None = None
        pos_snap = _safe_json_load(self.rdb.get("mft:positions:latest"))
        if pos_snap is not None:
            raw_dd = pos_snap.get("drawdown_pct")
            if isinstance(raw_dd, (int, float)):
                sleeve_dd_pct = float(raw_dd)
            else:
                equity = pos_snap.get("equity")
                hwm = pos_snap.get("high_water_mark")
                if isinstance(equity, (int, float)) and isinstance(hwm, (int, float)) and hwm > 0:
                    sleeve_dd_pct = ((float(equity) - float(hwm)) / float(hwm)) * 100.0
            if sleeve_dd_pct is not None and sleeve_dd_pct <= self.config.max_drawdown_pct:
                reasons.append(f"drawdown_breach:{sleeve_dd_pct:.2f}%<={self.config.max_drawdown_pct:.2f}%")

        # 4. Surface freshness check
        surface_ages_sec: dict[str, float | None] = {}
        for surf_key in self.config.surface_keys:
            surf = _safe_json_load(self.rdb.get(surf_key))
            if surf is None:
                surface_ages_sec[surf_key] = None
                reasons.append(f"missing_surface:{surf_key}")
            else:
                age = _extract_age_sec(surf, now_ns)
                surface_ages_sec[surf_key] = round(age, 2) if age is not None else None
                if age is not None and age > self.config.max_surface_age_sec:
                    reasons.append(f"stale_surface:{surf_key}:{age:.1f}s")

        # 5. Realized volatility snapshot check
        for rv_key in self.config.rv_keys:
            rv = _safe_json_load(self.rdb.get(rv_key))
            if rv is None:
                reasons.append(f"missing_rv:{rv_key}")
            else:
                age = _extract_age_sec(rv, now_ns)
                if age is not None and age > self.config.max_rv_age_sec:
                    reasons.append(f"stale_rv:{rv_key}:{age:.0f}s")

        halted = len(reasons) > 0
        evaluation = RiskEvaluation(
            halted=halted,
            reasons=reasons,
            evaluated_at_ns=now_ns,
            evaluated_at_utc=_iso_utc(now_ns),
            sleeve_drawdown_pct=round(sleeve_dd_pct, 4) if sleeve_dd_pct is not None else None,
            vix_ratio=round(vix_ratio, 4) if vix_ratio is not None else None,
            regime_age_sec=regime_age_sec,
            surface_ages_sec=surface_ages_sec,
        )

        payload_dict = asdict(evaluation)
        payload_json = json.dumps(payload_dict)

        pipe = self.rdb.pipeline()
        pipe.set("mft:risk:halted", "1" if halted else "0")
        pipe.set("mft:risk:status", payload_json)
        pipe.lpush("mft:risk:history", payload_json)
        pipe.ltrim("mft:risk:history", 0, self.config.history_max_len - 1)

        state_changed = (self._prev_halted is None) or (halted != self._prev_halted) or (reasons != self._prev_reasons)
        if state_changed:
            transition_event = {
                "event": "STATE_CHANGE",
                "from_halted": self._prev_halted,
                "to_halted": halted,
                **payload_dict,
            }
            pipe.lpush("mft:risk:transitions", json.dumps(transition_event))
            pipe.ltrim("mft:risk:transitions", 0, self.config.transitions_max_len - 1)
            self._append_jsonl(transition_event)
            self._last_disk_write_ns = now_ns
        elif (now_ns - self._last_disk_write_ns) / 1e9 >= self.config.heartbeat_log_interval_sec:
            self._append_jsonl({"event": "HEARTBEAT", **payload_dict})
            self._last_disk_write_ns = now_ns

        pipe.execute()
        self._prev_halted = halted
        self._prev_reasons = list(reasons)
        return evaluation

    def run_forever(self) -> None:
        signal.signal(signal.SIGINT, self.stop)
        signal.signal(signal.SIGTERM, self.stop)
        while self._running:
            try:
                self.evaluate_once()
            except Exception as exc:
                now_ns = time.time_ns()
                err_payload = {
                    "event": "DAEMON_ERROR",
                    "halted": True,
                    "reasons": [f"daemon_error:{type(exc).__name__}:{exc}"],
                    "evaluated_at_ns": now_ns,
                    "evaluated_at_utc": _iso_utc(now_ns),
                }
                self.rdb.set("mft:risk:halted", "1")
                self.rdb.set("mft:risk:status", json.dumps(err_payload))
                self.rdb.lpush("mft:risk:transitions", json.dumps(err_payload))
                self._append_jsonl(err_payload)
            time.sleep(self.config.poll_interval_sec)


def print_summary(valkey_url: str) -> None:
    rdb = redis.from_url(valkey_url)
    status = _safe_json_load(rdb.get("mft:risk:status"))
    hist_len = rdb.llen("mft:risk:history")
    trans_raw = rdb.lrange("mft:risk:transitions", 0, 14)

    print("=== Current Risk Status ===")
    print(json.dumps(status, indent=2))
    print(f"\n=== Rolling History Count: {hist_len} samples ===")
    print("=== Recent State Transitions (newest first) ===")
    for item in trans_raw:
        t = _safe_json_load(item)
        if t:
            print(
                f"  {t.get('evaluated_at_utc')} | halted={t.get('to_halted', t.get('halted'))} | "
                f"vix_ratio={t.get('vix_ratio')} | regime_age={t.get('regime_age_sec')}s | "
                f"reasons={t.get('reasons')}"
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="Out-of-process fail-closed risk daemon")
    parser.add_argument("--summary", action="store_true", help="Print current status & recent transitions and exit")
    args = parser.parse_args()

    valkey_url = os.environ.get("VALKEY_URL", "redis://127.0.0.1:6379/0")
    if args.summary:
        print_summary(valkey_url)
        return

    cfg = RiskDaemonConfig(
        valkey_url=valkey_url,
        poll_interval_sec=float(os.environ.get("RISK_POLL_INTERVAL_SEC", "2.0")),
        max_drawdown_pct=float(os.environ.get("RISK_MAX_DRAWDOWN_PCT", "-5.0")),
        vix_ratio_halt_threshold=float(os.environ.get("RISK_VIX_RATIO_THRESHOLD", "1.0")),
    )
    daemon = OutOfProcessRiskDaemon(cfg)
    daemon.run_forever()


if __name__ == "__main__":
    main()
