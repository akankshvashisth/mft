"""Pure risk-daemon rules: halt evaluation from snapshot payloads and transition/heartbeat tracking."""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from src.common import keys
from src.common.constants import DEFAULT_VALKEY_URL, NS_PER_SEC, SECS_PER_DAY
from src.common.flags import is_truthy_flag
from src.common.freshness import DAEMON_TS_FIELDS, payload_age_secs
from src.common.numeric import is_number
from src.common.timeutil import iso_utc

Payload = dict[str, Any]


@dataclass
class RiskDaemonConfig:
    valkey_url: str = DEFAULT_VALKEY_URL
    poll_interval_sec: float = 2.0
    max_regime_age_sec: float = 120.0
    max_surface_age_sec: float = 120.0
    max_rv_age_sec: float = SECS_PER_DAY
    max_positions_age_sec: float = 120.0
    require_positions_snapshot: bool = True
    max_drawdown_pct: float = -5.0
    vix_ratio_halt_threshold: float = 1.0
    surface_keys: tuple[str, ...] = (keys.surface_key("SPY.ARCA"),)
    rv_keys: tuple[str, ...] = (keys.rv_key("SPY.ARCA"),)
    history_max_len: int = 2000
    transitions_max_len: int = 500
    jsonl_path: str = "logs/risk_history.jsonl"
    heartbeat_log_interval_sec: float = 60.0
    # mft:risk:halted / mft:risk:status expire unless refreshed, so a dead daemon reads as "halted"
    state_ttl_sec: float = 10.0


@dataclass
class RiskEvaluation:
    halted: bool
    reasons: list[str]
    reason_codes: list[str]
    evaluated_at_ns: int
    evaluated_at_utc: str
    sleeve_drawdown_pct: float | None
    vix_ratio: float | None
    regime_age_sec: float | None
    surface_ages_sec: dict[str, float | None]


@dataclass(frozen=True)
class RiskInputs:
    """Everything one evaluation reads from Valkey (None = key missing or not a JSON object)."""

    manual_halt_raw: str | None = None
    regime: Payload | None = None
    positions: Payload | None = None
    surfaces: Mapping[str, Payload | None] = field(default_factory=dict)
    rvs: Mapping[str, Payload | None] = field(default_factory=dict)


def state_ttl_ms(cfg: RiskDaemonConfig) -> int:
    """TTL of the halted/status keys: at least three polls, so one slow poll never expires them."""
    return int(max(cfg.state_ttl_sec, 3.0 * cfg.poll_interval_sec) * 1000)


def sleeve_drawdown_pct(snapshot: Payload) -> float | None:
    """Published drawdown, else (equity - high-water mark) / high-water mark in percent."""
    raw_dd = snapshot.get("drawdown_pct")
    if is_number(raw_dd):
        return float(raw_dd)
    equity = snapshot.get("equity")
    hwm = snapshot.get("high_water_mark")
    if is_number(equity) and is_number(hwm) and hwm > 0:
        return (float(equity) - float(hwm)) / float(hwm) * 100.0
    return None


def evaluate_risk(inputs: RiskInputs, cfg: RiskDaemonConfig, now_ns: int) -> RiskEvaluation:
    """Fail-closed halt decision; any reason halts.

    Each reason is "<code>" or "<code>:<detail>". Codes are stable (no ages or values) so transitions are
    detected on what failed, not on how stale it is.
    """
    reasons: list[str] = []
    codes: list[str] = []

    def flag(code: str, detail: str | None = None) -> None:
        codes.append(code)
        reasons.append(f"{code}:{detail}" if detail is not None else code)

    def age_of(key: str, payload: Payload) -> float | None:
        # A payload without a usable timestamp (none, or far in the future) can't prove it is fresh
        age = payload_age_secs(payload, now_ns, DAEMON_TS_FIELDS)
        if age is None:
            flag(f"invalid_timestamp:{key}")
        return age

    # 1. Explicit manual kill-switch check (never auto-cleared by daemon)
    if is_truthy_flag(inputs.manual_halt_raw):
        flag("manual_kill_switch_active")

    # 2. Regime & VIX/VIX3M backwardation check
    regime = inputs.regime
    vix_ratio: float | None = None
    regime_age_sec: float | None = None
    if regime is None:
        flag("missing_regime_signal")
    else:
        age = age_of(keys.REGIME_LATEST, regime)
        if age is not None:
            regime_age_sec = round(age, 2)
            if age > cfg.max_regime_age_sec:
                flag("stale_regime_signal", f"{age:.1f}s")
        # The actor republishes on a timer, so a fresh payload can still carry stale index levels
        if regime.get("is_stale") is True:
            flag("stale_regime_feed", f"{regime.get('max_age_secs')}s")

        raw_ratio = regime.get("vix_ratio")
        if raw_ratio is None:
            raw_ratio = regime.get("ratio")
        if is_number(raw_ratio):
            vix_ratio = float(raw_ratio)
            if vix_ratio > cfg.vix_ratio_halt_threshold:
                flag("vix_backwardation", f"{vix_ratio:.4f}>{cfg.vix_ratio_halt_threshold}")

    # 3. Sleeve drawdown check from positions / equity snapshot
    sleeve_dd_pct: float | None = None
    positions = inputs.positions
    if positions is None:
        if cfg.require_positions_snapshot:
            flag("missing_positions_snapshot")
    else:
        age = age_of(keys.POSITIONS_LATEST, positions)
        if age is not None and age > cfg.max_positions_age_sec:
            flag("stale_positions_snapshot", f"{age:.1f}s")
        sleeve_dd_pct = sleeve_drawdown_pct(positions)
        if sleeve_dd_pct is not None and sleeve_dd_pct <= cfg.max_drawdown_pct:
            flag("drawdown_breach", f"{sleeve_dd_pct:.2f}%<={cfg.max_drawdown_pct:.2f}%")

    # 4. Surface freshness check
    surface_ages_sec: dict[str, float | None] = {}
    for surf_key in cfg.surface_keys:
        surf = inputs.surfaces.get(surf_key)
        if surf is None:
            surface_ages_sec[surf_key] = None
            flag(f"missing_surface:{surf_key}")
            continue
        age = age_of(surf_key, surf)
        surface_ages_sec[surf_key] = round(age, 2) if age is not None else None
        if age is not None and age > cfg.max_surface_age_sec:
            flag(f"stale_surface:{surf_key}", f"{age:.1f}s")

    # 5. Realized volatility snapshot check
    for rv_key in cfg.rv_keys:
        rv = inputs.rvs.get(rv_key)
        if rv is None:
            flag(f"missing_rv:{rv_key}")
            continue
        age = age_of(rv_key, rv)
        if age is not None and age > cfg.max_rv_age_sec:
            flag(f"stale_rv:{rv_key}", f"{age:.0f}s")

    return RiskEvaluation(
        halted=len(reasons) > 0,
        reasons=reasons,
        reason_codes=codes,
        evaluated_at_ns=now_ns,
        evaluated_at_utc=iso_utc(now_ns),
        sleeve_drawdown_pct=round(sleeve_dd_pct, 4) if sleeve_dd_pct is not None else None,
        vix_ratio=round(vix_ratio, 4) if vix_ratio is not None else None,
        regime_age_sec=regime_age_sec,
        surface_ages_sec=surface_ages_sec,
    )


TRANSITION = "TRANSITION"
HEARTBEAT = "HEARTBEAT"


class TransitionTracker:
    """Classifies each evaluation as a state transition, a periodic heartbeat, or neither."""

    def __init__(self, heartbeat_interval_secs: float) -> None:
        self.heartbeat_interval_secs = heartbeat_interval_secs
        self.prev_halted: bool | None = None
        self.prev_codes: list[str] | None = None
        self.last_disk_write_ns: int = 0

    def classify(self, halted: bool, codes: list[str], now_ns: int) -> tuple[str | None, bool | None]:
        """(TRANSITION | HEARTBEAT | None, previous halted state) and remembers this evaluation."""
        from_halted = self.prev_halted
        changed = self.prev_halted is None or halted != self.prev_halted or codes != self.prev_codes
        kind: str | None = None
        if changed:
            kind = TRANSITION
            self.last_disk_write_ns = now_ns
        elif (now_ns - self.last_disk_write_ns) / NS_PER_SEC >= self.heartbeat_interval_secs:
            kind = HEARTBEAT
            self.last_disk_write_ns = now_ns
        self.prev_halted = halted
        self.prev_codes = list(codes)
        return kind, from_halted

    def mark_error(self) -> None:
        """After a daemon error the recovery evaluation must be logged as a transition."""
        self.prev_halted = True
        self.prev_codes = ["daemon_error"]
