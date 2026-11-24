import pytest

from src.common import keys
from src.risk.daemon_rules import (
    HEARTBEAT,
    TRANSITION,
    RiskDaemonConfig,
    RiskInputs,
    TransitionTracker,
    evaluate_risk,
    sleeve_drawdown_pct,
    state_ttl_ms,
)
from tests.factories import (
    NOW_NS,
    UNDERLYING,
    ago,
    positions_payload,
    regime_payload,
    rv_payload,
    surface_payload,
)

CFG = RiskDaemonConfig()
SURF_KEY = keys.surface_key(UNDERLYING)
RV_KEY = keys.rv_key(UNDERLYING)
SEC = 1_000_000_000


def inputs(**overrides) -> RiskInputs:
    spec = dict(
        manual_halt_raw=None,
        regime=regime_payload(),
        positions=positions_payload(),
        surfaces={SURF_KEY: surface_payload()},
        rvs={RV_KEY: rv_payload()},
    )
    spec.update(overrides)
    return RiskInputs(**spec)


def test_healthy_inputs_do_not_halt():
    ev = evaluate_risk(inputs(), CFG, NOW_NS)

    assert (ev.halted, ev.reasons, ev.reason_codes) == (False, [], [])
    assert ev.vix_ratio == 0.8824
    assert ev.regime_age_sec == 0.0
    assert ev.surface_ages_sec == {SURF_KEY: 0.0}
    assert ev.sleeve_drawdown_pct == 0.0
    assert ev.evaluated_at_ns == NOW_NS
    assert ev.evaluated_at_utc == "2026-11-24T15:00:00+00:00"


@pytest.mark.parametrize(
    "overrides, code, reason",
    [
        ({"manual_halt_raw": "1"}, "manual_kill_switch_active", "manual_kill_switch_active"),
        ({"manual_halt_raw": " Yes "}, "manual_kill_switch_active", "manual_kill_switch_active"),
        ({"regime": None}, "missing_regime_signal", "missing_regime_signal"),
        ({"regime": regime_payload(now_ns=ago(121.0))}, "stale_regime_signal", "stale_regime_signal:121.0s"),
        ({"regime": regime_payload(is_stale=True, max_age_secs=95.5)}, "stale_regime_feed", "stale_regime_feed:95.5s"),
        ({"regime": regime_payload(ratio=1.05)}, "vix_backwardation", "vix_backwardation:1.0500>1.0"),
        ({"regime": regime_payload(ratio=0.9, vix_ratio=1.2)}, "vix_backwardation", "vix_backwardation:1.2000>1.0"),
        ({"positions": None}, "missing_positions_snapshot", "missing_positions_snapshot"),
        (
            {"positions": positions_payload(now_ns=ago(121.0))},
            "stale_positions_snapshot",
            "stale_positions_snapshot:121.0s",
        ),
        ({"positions": positions_payload(drawdown_pct=-5.0)}, "drawdown_breach", "drawdown_breach:-5.00%<=-5.00%"),
        (
            {"positions": positions_payload(drawdown_pct=None, equity=94_000.0, high_water_mark=100_000.0)},
            "drawdown_breach",
            "drawdown_breach:-6.00%<=-5.00%",
        ),
        ({"surfaces": {}}, f"missing_surface:{SURF_KEY}", f"missing_surface:{SURF_KEY}"),
        (
            {"surfaces": {SURF_KEY: surface_payload(now_ns=ago(121.0))}},
            f"stale_surface:{SURF_KEY}",
            f"stale_surface:{SURF_KEY}:121.0s",
        ),
        ({"rvs": {}}, f"missing_rv:{RV_KEY}", f"missing_rv:{RV_KEY}"),
        (
            {"rvs": {RV_KEY: rv_payload(now_ns=ago(2 * 86_400.0))}},
            f"stale_rv:{RV_KEY}",
            f"stale_rv:{RV_KEY}:172800s",
        ),
    ],
)
def test_each_failed_check_halts_with_its_reason(overrides, code, reason):
    ev = evaluate_risk(inputs(**overrides), CFG, NOW_NS)
    assert ev.halted
    assert ev.reason_codes == [code]
    assert ev.reasons == [reason]


def test_manual_flag_must_be_truthy():
    assert not evaluate_risk(inputs(manual_halt_raw="0"), CFG, NOW_NS).halted


def test_reason_codes_are_stable_while_details_change():
    first = evaluate_risk(inputs(regime=regime_payload(now_ns=ago(121.0))), CFG, NOW_NS)
    later = evaluate_risk(inputs(regime=regime_payload(now_ns=ago(121.0))), CFG, NOW_NS + 30 * SEC)
    assert first.reason_codes == later.reason_codes == ["stale_regime_signal"]
    assert first.reasons != later.reasons


@pytest.mark.parametrize("ts_ns", [None, "recent", NOW_NS + 901 * SEC])
@pytest.mark.parametrize(
    "build, key",
    [
        (lambda ts: {"regime": regime_payload(ts_ns=ts)}, keys.REGIME_LATEST),
        (lambda ts: {"positions": positions_payload(ts_ns=ts)}, keys.POSITIONS_LATEST),
        (lambda ts: {"surfaces": {SURF_KEY: surface_payload(ts_ns=ts)}}, SURF_KEY),
        (lambda ts: {"rvs": {RV_KEY: rv_payload(ts_ns=ts)}}, RV_KEY),
    ],
)
def test_unusable_timestamps_fail_closed(build, key, ts_ns):
    ev = evaluate_risk(inputs(**build(ts_ns)), CFG, NOW_NS)
    assert ev.halted
    assert ev.reason_codes == [f"invalid_timestamp:{key}"]


def test_invalid_timestamps_report_no_age():
    ev = evaluate_risk(
        inputs(regime=regime_payload(ts_ns=None), surfaces={SURF_KEY: surface_payload(ts_ns=None)}), CFG, NOW_NS
    )
    assert ev.regime_age_sec is None
    assert ev.surface_ages_sec == {SURF_KEY: None}


def test_small_clock_skew_and_alternative_timestamp_fields_are_accepted():
    skewed = evaluate_risk(inputs(regime=regime_payload(ts_ns=NOW_NS + 899 * SEC)), CFG, NOW_NS)
    assert not skewed.halted and skewed.regime_age_sec == 0.0

    via_ts_event = evaluate_risk(inputs(regime=regime_payload(ts_ns=None, ts_event=ago(10.0))), CFG, NOW_NS)
    assert not via_ts_event.halted and via_ts_event.regime_age_sec == 10.0


def test_positions_snapshot_can_be_optional():
    cfg = RiskDaemonConfig(require_positions_snapshot=False)
    assert not evaluate_risk(inputs(positions=None), cfg, NOW_NS).halted


def test_reasons_follow_check_order():
    ev = evaluate_risk(
        inputs(manual_halt_raw="on", regime=None, positions=positions_payload(drawdown_pct=-7.5), rvs={}),
        CFG,
        NOW_NS,
    )
    assert ev.reason_codes == [
        "manual_kill_switch_active",
        "missing_regime_signal",
        "drawdown_breach",
        f"missing_rv:{RV_KEY}",
    ]


def test_sleeve_drawdown_sources():
    assert sleeve_drawdown_pct({"drawdown_pct": -2.5, "equity": 1.0, "high_water_mark": 2.0}) == -2.5
    assert sleeve_drawdown_pct({"equity": 99_000.0, "high_water_mark": 100_000.0}) == pytest.approx(-1.0)
    assert sleeve_drawdown_pct({"equity": 99_000.0, "high_water_mark": 0.0}) is None
    assert sleeve_drawdown_pct({}) is None


def test_state_ttl_covers_at_least_three_polls():
    assert state_ttl_ms(RiskDaemonConfig()) == 10_000
    assert state_ttl_ms(RiskDaemonConfig(poll_interval_sec=5.0)) == 15_000


def test_transition_tracker():
    tracker = TransitionTracker(heartbeat_interval_secs=60.0)
    t0 = NOW_NS

    assert tracker.classify(False, [], t0) == (TRANSITION, None)
    assert tracker.classify(False, [], t0 + 1 * SEC) == (None, False)
    assert tracker.classify(False, [], t0 + 60 * SEC) == (HEARTBEAT, False)
    assert tracker.classify(False, [], t0 + 61 * SEC) == (None, False)
    assert tracker.classify(True, ["drawdown_breach"], t0 + 62 * SEC) == (TRANSITION, False)
    assert tracker.classify(True, ["drawdown_breach", "missing_regime_signal"], t0 + 63 * SEC) == (TRANSITION, True)
    assert tracker.classify(True, ["drawdown_breach", "missing_regime_signal"], t0 + 64 * SEC) == (None, True)


def test_recovery_after_an_error_is_a_transition():
    tracker = TransitionTracker(heartbeat_interval_secs=60.0)
    tracker.classify(False, [], NOW_NS)
    tracker.mark_error()
    assert tracker.classify(False, [], NOW_NS + SEC) == (TRANSITION, True)
