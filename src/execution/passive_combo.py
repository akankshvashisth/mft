from __future__ import annotations

import math
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.common.jsonl import JsonlSink, RecordSink
from src.common.timeutil import iso_utc

CREDIT_OPEN = "CREDIT_OPEN"
DEBIT_CLOSE = "DEBIT_CLOSE"


@dataclass(frozen=True)
class PassiveExecutionConfig:
    max_child_clip_lots: int = 8            # Slice mid-cap orders into 5-10 lot clips to hide size
    initial_mid_offset_frac: float = 0.15   # Start 15% of half-spread inside natural from mid
    step_increment_frac: float = 0.25       # Step 25% of half-spread per dwell cycle
    max_slippage_frac: float = 0.65         # Hard cap: never cross past 65% of half-spread
    dwell_seconds: float = 20.0             # Seconds to rest passive order before stepping
    adverse_spot_move_pct: float = 0.0025   # Cancel/reprice if underlying moves >= 25 bps against us
    slippage_log_path: str = "logs/execution_slippage.jsonl"
    tick_size: float = 0.01                 # Default minimum price increment


def round_to_tick(price: float, tick_size: float, side: str) -> float:
    """Round price defensively to the nearest tick so the limit never crosses the spread.

    For CREDIT_OPEN (selling), round up so the limit is at or above the tick grid.
    For DEBIT_CLOSE (buying), round down so the limit is at or below the tick grid.
    Otherwise, round to nearest.

    Uses a tiny epsilon (1e-9) to avoid floating-point truncation at exact tick boundaries.
    """
    eps = 1e-9
    units = price / tick_size
    if side == CREDIT_OPEN:
        return math.ceil(units - eps) * tick_size
    if side == DEBIT_CLOSE:
        return math.floor(units + eps) * tick_size
    return round(units) * tick_size


@dataclass
class ChildClipOrder:
    clip_id: str
    parent_order_id: str
    underlying: str
    combo_label: str
    side: str                   # "CREDIT_OPEN" or "DEBIT_CLOSE"
    lots: int
    ref_spot: float
    net_mid_price: float        # Positive = credit received, or debit paid
    net_natural_price: float    # Immediate fill price (worse than mid)
    current_limit_price: float
    arrival_mid_price: float | None = None    # Immutable arrival price for shortfall measurement
    arrival_natural_price: float | None = None
    step_count: int = 0
    last_step_ts: float = field(default_factory=time.monotonic)
    status: str = "WORKING"     # "WORKING", "FILLED", "CANCELLED_ADVERSE", "CANCELLED_CAP"
    fill_price: float | None = None


def stepped_limit(
    side: str, mid_px: float, nat_px: float, step_idx: int, cfg: PassiveExecutionConfig
) -> float | None:
    """Passive limit after step_idx dwell steps, conceding toward natural; None once the slippage cap was used."""
    half_spread = abs(mid_px - nat_px)
    frac = min(cfg.initial_mid_offset_frac + step_idx * cfg.step_increment_frac, cfg.max_slippage_frac)
    if step_idx > 0 and (
        cfg.initial_mid_offset_frac + (step_idx - 1) * cfg.step_increment_frac >= cfg.max_slippage_frac
    ):
        return None  # Reached hard slippage cap

    if side == CREDIT_OPEN:
        # Selling spread for credit: mid >= natural; stepping down concedes credit
        return round(max(nat_px, mid_px - frac * half_spread), 2)
    # Buying spread back to close: mid <= natural; stepping up pays more debit
    return round(min(nat_px, mid_px + frac * half_spread), 2)


def slippage_record(
    clip_id: str,
    clip: ChildClipOrder | None,
    fill_price: float,
    commission_usd: float,
    filled_lots: int | None,
    ts_utc: str,
) -> dict[str, Any]:
    """Post-trade audit record; positive slippage_usd = cost paid vs. the modeled mid."""
    if clip is not None:
        mid_ref = clip.arrival_mid_price if clip.arrival_mid_price is not None else clip.net_mid_price
        nat_ref = clip.arrival_natural_price if clip.arrival_natural_price is not None else clip.net_natural_price
        lots = clip.lots if filled_lots is None else filled_lots
        side = clip.side
        underlying = clip.underlying
        combo_label = clip.combo_label
    else:
        mid_ref = fill_price
        nat_ref = fill_price
        lots = 1 if filled_lots is None else filled_lots
        side = "UNKNOWN"
        underlying = "UNKNOWN"
        combo_label = clip_id

    slippage_per_share = mid_ref - fill_price if side == CREDIT_OPEN else fill_price - mid_ref
    return {
        "ts_utc": ts_utc,
        "clip_id": clip_id,
        "underlying": underlying,
        "combo_label": combo_label,
        "side": side,
        "lots": lots,
        "modeled_mid": round(mid_ref, 4),
        "natural_px": round(nat_ref, 4),
        "actual_fill": round(fill_price, 4),
        "slippage_usd": round(slippage_per_share * 100.0 * lots, 2),
        "commission_usd": round(commission_usd, 2),
    }


def _default_parent_id() -> str:
    return uuid.uuid4().hex[:10].upper()


class PassiveComboExecutor:
    """Manages mid-cap clip slicing, passive limit stepping, adverse-tick cancels & slippage logs.

    The monotonic clock, parent-id factory, wall clock and fill sink are injectable; the default sink
    appends JSON lines to cfg.slippage_log_path and creates its directory on first write.
    """

    def __init__(
        self,
        config: PassiveExecutionConfig | None = None,
        *,
        clock: Callable[[], float] = time.monotonic,
        id_factory: Callable[[], str] = _default_parent_id,
        fill_sink: RecordSink | None = None,
        wall_clock_ns: Callable[[], int] = time.time_ns,
    ) -> None:
        self.cfg = config or PassiveExecutionConfig()
        self.active_clips: dict[str, ChildClipOrder] = {}
        self.log_path = Path(self.cfg.slippage_log_path)
        self.clock = clock
        self.id_factory = id_factory
        self.fill_sink: RecordSink = fill_sink if fill_sink is not None else JsonlSink(self.log_path)
        self.wall_clock_ns = wall_clock_ns

    def _compute_stepped_limit(
        self, side: str, mid_px: float, nat_px: float, step_idx: int
    ) -> float | None:
        return stepped_limit(side, mid_px, nat_px, step_idx, self.cfg)

    def create_parent_plan(
        self,
        underlying: str,
        combo_label: str,
        side: str,
        total_lots: int,
        spot_price: float,
        net_mid_price: float,
        net_natural_price: float,
        register: bool = True,
    ) -> list[ChildClipOrder]:
        """Slices total_lots into mid-cap-safe child clips priced passively near mid.

        register=False builds a preview plan that is never tracked as working (e.g. shadow mode).
        """
        parent_id = f"CMB-{self.id_factory()}"
        remaining = max(0, int(total_lots))
        clips: list[ChildClipOrder] = []
        init_limit = stepped_limit(side, net_mid_price, net_natural_price, 0, self.cfg)
        assert init_limit is not None
        now = self.clock()

        while remaining > 0:
            clip_lots = min(remaining, self.cfg.max_child_clip_lots)
            remaining -= clip_lots
            clip = ChildClipOrder(
                clip_id=f"{parent_id}-{len(clips) + 1}",
                parent_order_id=parent_id,
                underlying=underlying,
                combo_label=combo_label,
                side=side,
                lots=clip_lots,
                ref_spot=spot_price,
                net_mid_price=round(net_mid_price, 4),
                net_natural_price=round(net_natural_price, 4),
                arrival_mid_price=round(net_mid_price, 4),
                arrival_natural_price=round(net_natural_price, 4),
                current_limit_price=init_limit,
                last_step_ts=now,
            )
            if register:
                self.active_clips[clip.clip_id] = clip
            clips.append(clip)

        return clips

    def mark_clip_status(self, clip_id: str, status: str) -> None:
        """Moves a still-working clip to a terminal status (venue cancel/reject or strategy-side cancel)."""
        clip = self.active_clips.get(clip_id)
        if clip is not None and clip.status == "WORKING":
            clip.status = status

    def discard_clip(self, clip_id: str) -> None:
        """Forgets a finished clip so working-clip scans stay bounded."""
        self.active_clips.pop(clip_id, None)

    def on_market_tick(
        self,
        underlying: str,
        current_spot: float,
        new_mid_price: float,
        new_natural_price: float,
        now_mono: float | None = None,
        parent_order_id: str | None = None,
        rv_20d: float | None = None,
    ) -> list[dict]:
        """Evaluates working clips for adverse spot moves or dwell-timer limit stepping.

        Mid/natural prices belong to a single combo, so pass parent_order_id whenever clips of different
        combos can be working at the same time.

        When ``rv_20d`` is provided the adverse-spot threshold is at least
        ``max(cfg.adverse_spot_move_pct, 0.15 * rv_20d / sqrt(252))`` (one-sigma daily).
        """
        now = now_mono if now_mono is not None else self.clock()
        actions: list[dict] = []

        for clip in list(self.active_clips.values()):
            if clip.underlying != underlying or clip.status != "WORKING":
                continue
            if parent_order_id is not None and clip.parent_order_id != parent_order_id:
                continue

            spot_move_pct = abs(current_spot - clip.ref_spot) / max(clip.ref_spot, 1e-6)
            adverse_threshold = self.cfg.adverse_spot_move_pct
            if rv_20d is not None and rv_20d > 0.0:
                daily_sigma = rv_20d / math.sqrt(252.0)
                adverse_threshold = max(adverse_threshold, 0.15 * daily_sigma)
            if spot_move_pct >= adverse_threshold:
                clip.status = "CANCELLED_ADVERSE"
                actions.append(
                    {
                        "action": "CANCEL_REPLACE",
                        "clip_id": clip.clip_id,
                        "reason": f"ADVERSE_SPOT_MOVE({spot_move_pct:.2%})",
                        "old_limit": clip.current_limit_price,
                    }
                )
                continue

            if (now - clip.last_step_ts) >= self.cfg.dwell_seconds:
                next_step = clip.step_count + 1
                next_limit = stepped_limit(clip.side, new_mid_price, new_natural_price, next_step, self.cfg)
                if next_limit is None:
                    clip.status = "CANCELLED_CAP"
                    actions.append(
                        {
                            "action": "CANCEL_FINAL",
                            "clip_id": clip.clip_id,
                            "reason": "MAX_SLIPPAGE_CAP_REACHED",
                            "last_limit": clip.current_limit_price,
                        }
                    )
                else:
                    clip.step_count = next_step
                    clip.last_step_ts = now
                    clip.net_mid_price = round(new_mid_price, 4)
                    clip.net_natural_price = round(new_natural_price, 4)
                    clip.current_limit_price = next_limit
                    actions.append(
                        {
                            "action": "MODIFY_LIMIT",
                            "clip_id": clip.clip_id,
                            "step": next_step,
                            "new_limit": next_limit,
                        }
                    )

        return actions

    def record_fill(
        self,
        clip_id: str,
        fill_price: float,
        commission_usd: float,
        filled_lots: int | None = None,
    ) -> dict:
        """Logs modeled mid vs. actual execution slippage for post-trade audit.

        filled_lots overrides the planned clip size for partially filled clips.
        """
        clip = self.active_clips.get(clip_id)
        if clip is not None:
            clip.status = "FILLED"
            clip.fill_price = fill_price
        record = slippage_record(
            clip_id, clip, fill_price, commission_usd, filled_lots, iso_utc(self.wall_clock_ns())
        )
        self.fill_sink.write(record)
        return record
