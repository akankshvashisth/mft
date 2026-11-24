from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import json
from pathlib import Path
import time
import uuid


@dataclass(frozen=True)
class PassiveExecutionConfig:
    max_child_clip_lots: int = 8            # Slice mid-cap orders into 5-10 lot clips to hide size
    initial_mid_offset_frac: float = 0.15   # Start 15% of half-spread inside natural from mid
    step_increment_frac: float = 0.25       # Step 25% of half-spread per dwell cycle
    max_slippage_frac: float = 0.65         # Hard cap: never cross past 65% of half-spread
    dwell_seconds: float = 20.0             # Seconds to rest passive order before stepping
    adverse_spot_move_pct: float = 0.0025   # Cancel/reprice if underlying moves >= 25 bps against us
    slippage_log_path: str = "logs/execution_slippage.jsonl"


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
    step_count: int = 0
    last_step_ts: float = field(default_factory=time.monotonic)
    status: str = "WORKING"     # "WORKING", "FILLED", "CANCELLED_ADVERSE", "CANCELLED_CAP"
    fill_price: float | None = None


class PassiveComboExecutor:
    """Manages mid-cap clip slicing, passive limit stepping, adverse-tick cancels & slippage logs."""

    def __init__(self, config: PassiveExecutionConfig | None = None) -> None:
        self.cfg = config or PassiveExecutionConfig()
        self.active_clips: dict[str, ChildClipOrder] = {}
        self.log_path = Path(self.cfg.slippage_log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

    def _compute_stepped_limit(
        self, side: str, mid_px: float, nat_px: float, step_idx: int
    ) -> float | None:
        half_spread = abs(mid_px - nat_px)
        frac = min(
            self.cfg.initial_mid_offset_frac + step_idx * self.cfg.step_increment_frac,
            self.cfg.max_slippage_frac,
        )
        if step_idx > 0 and (
            self.cfg.initial_mid_offset_frac + (step_idx - 1) * self.cfg.step_increment_frac
            >= self.cfg.max_slippage_frac
        ):
            return None  # Reached hard slippage cap

        if side == "CREDIT_OPEN":
            # Selling spread for credit: mid >= natural; stepping down concedes credit
            return round(max(nat_px, mid_px - frac * half_spread), 2)
        # Buying spread back to close: mid <= natural; stepping up pays more debit
        return round(min(nat_px, mid_px + frac * half_spread), 2)

    def create_parent_plan(
        self,
        underlying: str,
        combo_label: str,
        side: str,
        total_lots: int,
        spot_price: float,
        net_mid_price: float,
        net_natural_price: float,
    ) -> list[ChildClipOrder]:
        """Slices total_lots into mid-cap-safe child clips priced passively near mid."""
        parent_id = f"CMB-{uuid.uuid4().hex[:10].upper()}"
        remaining = max(0, int(total_lots))
        clips: list[ChildClipOrder] = []

        while remaining > 0:
            clip_lots = min(remaining, self.cfg.max_child_clip_lots)
            remaining -= clip_lots
            init_limit = self._compute_stepped_limit(side, net_mid_price, net_natural_price, 0)
            assert init_limit is not None
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
                current_limit_price=init_limit,
            )
            self.active_clips[clip.clip_id] = clip
            clips.append(clip)

        return clips

    def on_market_tick(
        self,
        underlying: str,
        current_spot: float,
        new_mid_price: float,
        new_natural_price: float,
        now_mono: float | None = None,
    ) -> list[dict]:
        """Evaluates working clips for adverse spot moves or dwell-timer limit stepping."""
        now = now_mono if now_mono is not None else time.monotonic()
        actions: list[dict] = []

        for clip in list(self.active_clips.values()):
            if clip.underlying != underlying or clip.status != "WORKING":
                continue

            spot_move_pct = abs(current_spot - clip.ref_spot) / max(clip.ref_spot, 1e-6)
            if spot_move_pct >= self.cfg.adverse_spot_move_pct:
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
                next_limit = self._compute_stepped_limit(
                    clip.side, new_mid_price, new_natural_price, next_step
                )
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

    def record_fill(self, clip_id: str, fill_price: float, commission_usd: float) -> dict:
        """Logs modeled mid vs. actual execution slippage to JSONL for post-trade audit."""
        clip = self.active_clips.get(clip_id)
        if clip is not None:
            clip.status = "FILLED"
            clip.fill_price = fill_price
            mid_ref = clip.net_mid_price
            nat_ref = clip.net_natural_price
            lots = clip.lots
            side = clip.side
            underlying = clip.underlying
            combo_label = clip.combo_label
        else:
            mid_ref = fill_price
            nat_ref = fill_price
            lots = 1
            side = "UNKNOWN"
            underlying = "UNKNOWN"
            combo_label = clip_id

        # Positive slippage_usd = cost paid vs. theoretical mid
        if side == "CREDIT_OPEN":
            slippage_per_share = mid_ref - fill_price
        else:
            slippage_per_share = fill_price - mid_ref

        record = {
            "ts_utc": datetime.now(timezone.utc).isoformat(),
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
        with self.log_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record) + "\n")
        return record
