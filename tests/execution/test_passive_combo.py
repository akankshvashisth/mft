import json
import re
from unittest import mock

import pytest

from src.execution.passive_combo import (
    CREDIT_OPEN,
    DEBIT_CLOSE,
    PassiveComboExecutor,
    PassiveExecutionConfig,
    round_to_tick,
    slippage_record,
    stepped_limit,
)
from tests.factories import NOW_NS
from tests.fakes import recording_sink, written

CFG = PassiveExecutionConfig()


@pytest.mark.parametrize(
    "side, mid, natural, limits",
    [
        ("CREDIT_OPEN", 1.00, 0.80, [0.97, 0.92, 0.87, None]),
        ("DEBIT_CLOSE", 1.00, 1.20, [1.03, 1.08, 1.13, None]),
    ],
)
def test_limits_step_toward_natural_until_the_cap(side, mid, natural, limits):
    got = [stepped_limit(side, mid, natural, step, CFG) for step in range(4)]
    for g, l in zip(got, limits):
        if l is None:
            assert g is None
        else:
            assert g == pytest.approx(l)


def test_limit_never_crosses_natural():
    aggressive = PassiveExecutionConfig(step_increment_frac=1.0, max_slippage_frac=2.0)
    assert stepped_limit("CREDIT_OPEN", 1.00, 0.80, 1, aggressive) == 0.80
    assert stepped_limit("DEBIT_CLOSE", 1.00, 1.20, 1, aggressive) == 1.20


@pytest.fixture
def clock():
    return mock.Mock(return_value=1_000.0)


@pytest.fixture
def executor(clock, mem_sink, wall_clock):
    ids = iter(["AAA", "BBB", "CCC"])
    return PassiveComboExecutor(clock=clock, id_factory=lambda: next(ids), fill_sink=mem_sink, wall_clock_ns=wall_clock)


def plan(executor, lots=20, side="CREDIT_OPEN", mid=1.00, natural=0.80, register=True, underlying="SPY.ARCA"):
    return executor.create_parent_plan(underlying, "SPY 690/685P Bull Put", side, lots, 700.0, mid, natural, register=register)


def test_parent_order_is_sliced_into_registered_clips(executor):
    clips = plan(executor, lots=20)

    assert [c.lots for c in clips] == [8, 8, 4]
    assert [c.clip_id for c in clips] == ["CMB-AAA-1", "CMB-AAA-2", "CMB-AAA-3"]
    assert {c.parent_order_id for c in clips} == {"CMB-AAA"}
    assert all(c.current_limit_price == 0.97 and c.last_step_ts == 1_000.0 for c in clips)
    assert list(executor.active_clips) == ["CMB-AAA-1", "CMB-AAA-2", "CMB-AAA-3"]


def test_previews_and_empty_plans_track_nothing(executor):
    assert len(plan(executor, lots=3, register=False)) == 1
    assert plan(executor, lots=0) == []
    assert executor.active_clips == {}


def test_adverse_spot_move_cancels(executor):
    (clip,) = plan(executor, lots=5)
    assert executor.on_market_tick("SPY.ARCA", 701.0, 1.0, 0.8, now_mono=1_001.0) == []  # 14 bps

    actions = executor.on_market_tick("SPY.ARCA", 702.0, 1.0, 0.8, now_mono=1_001.0)  # 29 bps
    assert actions == [
        {"action": "CANCEL_REPLACE", "clip_id": clip.clip_id, "reason": "ADVERSE_SPOT_MOVE(0.29%)", "old_limit": 0.97}
    ]
    assert clip.status == "CANCELLED_ADVERSE"
    assert executor.on_market_tick("SPY.ARCA", 702.0, 1.0, 0.8, now_mono=1_100.0) == []


def test_dwell_steps_then_final_cancel_at_the_cap(executor):
    (clip,) = plan(executor, lots=5)

    assert executor.on_market_tick("SPY.ARCA", 700.0, 1.0, 0.8, now_mono=1_019.0) == []
    step1 = executor.on_market_tick("SPY.ARCA", 700.0, 1.0, 0.8, now_mono=1_020.0)
    assert step1 == [{"action": "MODIFY_LIMIT", "clip_id": clip.clip_id, "step": 1, "new_limit": 0.92}]
    assert (clip.step_count, clip.last_step_ts, clip.current_limit_price) == (1, 1_020.0, 0.92)

    assert executor.on_market_tick("SPY.ARCA", 700.0, 1.0, 0.8, now_mono=1_030.0) == []
    assert executor.on_market_tick("SPY.ARCA", 700.0, 1.1, 0.9, now_mono=1_040.0)[0]["new_limit"] == 0.97
    assert (clip.net_mid_price, clip.net_natural_price) == (1.1, 0.9)  # Re-anchored on the new market

    final = executor.on_market_tick("SPY.ARCA", 700.0, 1.1, 0.9, now_mono=1_060.0)
    assert final == [
        {"action": "CANCEL_FINAL", "clip_id": clip.clip_id, "reason": "MAX_SLIPPAGE_CAP_REACHED", "last_limit": 0.97}
    ]
    assert clip.status == "CANCELLED_CAP"


def test_default_clock_drives_dwell(executor, clock):
    plan(executor, lots=1)
    clock.return_value += 25.0
    assert executor.on_market_tick("SPY.ARCA", 700.0, 1.0, 0.8)[0]["action"] == "MODIFY_LIMIT"


def test_ticks_only_touch_the_matching_parent_and_underlying(executor):
    (a,) = plan(executor, lots=1)
    (b,) = plan(executor, lots=1)
    (other,) = plan(executor, lots=1, underlying="QQQ.NASDAQ")

    actions = executor.on_market_tick("SPY.ARCA", 700.0, 1.0, 0.8, now_mono=1_030.0, parent_order_id=a.parent_order_id)
    assert [act["clip_id"] for act in actions] == [a.clip_id]
    assert (b.step_count, other.step_count) == (0, 0)


def test_status_helpers(executor):
    first, second = plan(executor, lots=16)
    executor.mark_clip_status(first.clip_id, "CANCELLED")
    executor.mark_clip_status(first.clip_id, "SUBMIT_FAILED")  # Terminal status is kept
    assert first.status == "CANCELLED"

    executor.discard_clip(first.clip_id)
    executor.discard_clip("unknown")
    assert list(executor.active_clips) == [second.clip_id]


def test_fill_records_slippage_against_the_modeled_mid(executor, mem_sink):
    (clip,) = plan(executor, lots=8)
    record = executor.record_fill(clip.clip_id, 0.95, 5.2)

    assert record == {
        "ts_utc": "2026-11-24T15:00:00+00:00",
        "clip_id": clip.clip_id,
        "underlying": "SPY.ARCA",
        "combo_label": "SPY 690/685P Bull Put",
        "side": "CREDIT_OPEN",
        "lots": 8,
        "modeled_mid": 1.0,
        "natural_px": 0.8,
        "actual_fill": 0.95,
        "slippage_usd": 40.0,
        "commission_usd": 5.2,
    }
    assert written(mem_sink) == [record]
    assert (clip.status, clip.fill_price) == ("FILLED", 0.95)


def test_partial_fills_and_closing_debits():
    clip = PassiveComboExecutor(fill_sink=recording_sink()).create_parent_plan("SPY", "x", "DEBIT_CLOSE", 8, 700.0, 1.0, 1.2)[0]
    record = slippage_record(clip.clip_id, clip, 1.05, 0.0, 3, "t")
    assert (record["lots"], record["slippage_usd"]) == (3, 15.0)  # Paid 0.05 over mid on 3 lots


def test_unknown_clip_fill_is_still_recorded(executor, mem_sink):
    record = executor.record_fill("ghost", 1.25, 1.3)
    assert (record["side"], record["lots"], record["slippage_usd"], record["combo_label"]) == ("UNKNOWN", 1, 0.0, "ghost")
    mem_sink.write.assert_called_once_with(record)


def test_default_sink_writes_lazily(tmp_path):
    path = tmp_path / "logs" / "execution_slippage.jsonl"
    executor = PassiveComboExecutor(PassiveExecutionConfig(slippage_log_path=str(path)), wall_clock_ns=lambda: NOW_NS)
    assert not path.parent.exists()

    executor.record_fill("ghost", 1.0, 0.0)
    assert json.loads(path.read_text(encoding="utf-8"))["clip_id"] == "ghost"


def test_default_parent_ids_are_random_hex():
    clip = PassiveComboExecutor(fill_sink=recording_sink()).create_parent_plan("SPY", "x", "CREDIT_OPEN", 1, 700.0, 1.0, 0.8)[0]
    assert re.fullmatch(r"CMB-[0-9A-F]{10}", clip.parent_order_id)


# --- Arrival mid & tick sizing --------------------------------------------------------------------


def test_arrival_mid_unchanged_after_limit_steps(executor, clock):
    """After 2 limit steps with changing new_mid_price, clip.arrival_mid_price stays at creation mid."""
    (clip,) = plan(executor, lots=5, mid=1.00, natural=0.80)
    assert clip.arrival_mid_price == 1.00
    assert clip.arrival_natural_price == 0.80

    # Step 1
    clock.return_value += 20.0
    executor.on_market_tick("SPY.ARCA", 700.0, 0.95, 0.75, now_mono=clock())
    assert clip.arrival_mid_price == 1.00  # Unchanged from creation
    assert clip.arrival_natural_price == 0.80

    # Step 2
    clock.return_value += 20.0
    executor.on_market_tick("SPY.ARCA", 700.0, 0.90, 0.70, now_mono=clock())
    assert clip.arrival_mid_price == 1.00  # Still unchanged


def test_arrival_mid_used_in_slippage_record(executor):
    """record_fill logs shortfall vs arrival_mid_price, not current net_mid_price."""
    (clip,) = plan(executor, lots=8, mid=1.00, natural=0.80)
    clip.net_mid_price = 0.90  # Step advanced
    record = executor.record_fill(clip.clip_id, 0.95, 5.2)
    # Shortfall against arrival 1.00 (not current 0.90): 1.00 - 0.95 = 0.05 * 100 * 8 = 40
    assert record["slippage_usd"] == 40.0
    assert record["modeled_mid"] == 1.0


def test_round_to_tick():
    """round_to_tick with tick_size=0.05 rounds 1.23 -> 1.25 for CREDIT_OPEN and 1.20 for DEBIT_CLOSE."""
    assert round_to_tick(1.23, 0.05, CREDIT_OPEN) == 1.25
    assert round_to_tick(1.23, 0.05, DEBIT_CLOSE) == pytest.approx(1.20)
    assert round_to_tick(1.20, 0.05, CREDIT_OPEN) == pytest.approx(1.20)  # Exact tick, but float
    assert round_to_tick(1.20, 0.05, DEBIT_CLOSE) == pytest.approx(1.20)
    assert round_to_tick(1.249, 0.05, CREDIT_OPEN) == 1.25
    assert round_to_tick(1.249, 0.05, DEBIT_CLOSE) == pytest.approx(1.20)


def test_tick_size_produces_aligned_limits():
    """PassiveExecutionConfig(tick_size=0.05) produces $0.05-aligned limits on create_parent_plan."""
    cfg = PassiveExecutionConfig(tick_size=0.05, initial_mid_offset_frac=0.20, max_child_clip_lots=8)
    clock = mock.Mock(return_value=1_000.0)
    executor = PassiveComboExecutor(cfg, clock=clock, fill_sink=recording_sink())
    clips = executor.create_parent_plan("SPY.ARCA", "x", "CREDIT_OPEN", 5, 700.0, 1.03, 0.80)
    # mid=1.03, natural=0.80, half_spread=0.23, offset=0.20*0.23=0.046
    # stepped: max(0.80, 1.03 - 0.046) = 0.984 -> round_to_tick(0.984, 0.05, CREDIT_OPEN) = ceil(0.984/0.05)*0.05 = 1.00
    assert clips[0].current_limit_price == pytest.approx(1.00)
    # Advance clock to trigger dwell step
    clock.return_value += 21.0
    executor.on_market_tick("SPY.ARCA", 700.0, 1.03, 0.80, now_mono=clock())
    # After step 1: frac = 0.20 + 0.25 = 0.45, half_spread = 0.23
    # stepped: max(0.80, 1.03 - 0.45*0.23) = max(0.80, 0.927) = 0.927 -> round_to_tick = ceil(0.927/0.05)*0.05 = 0.95
    assert clips[0].current_limit_price == pytest.approx(0.95)
