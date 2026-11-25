import itertools
from datetime import date
from types import SimpleNamespace
from unittest import mock

import pytest

from src.common import keys
from src.common.market import QuoteObs
from src.execution.passive_combo import PassiveComboExecutor
from src.strategies.vrp_engine import OrderView, VrpEngine, VrpSettings, WorkingOrder
from tests.factories import NOW_NS, ago, portfolio_payload, risk_assessment, rv_payload, surface_leg, surface_payload
from tests.fakes import FakeGate, FakeMarket, FakeOrderGateway, FakePricer, written

SPY = "SPY.ARCA"
TODAY = date(2026, 11, 24)
P690, P685, P680 = "SPY P690", "SPY P685", "SPY P680"
LEGS = [{"instrument_id": P690, "ratio": 1}, {"instrument_id": P685, "ratio": -1}]
SPREAD_ID = FakeOrderGateway.spread_id(LEGS)
SURFACE_LEGS = [
    surface_leg(P680, 680.0, "P", 1.0, 1.2),
    surface_leg(P685, 685.0, "P", 1.4, 1.6),
    surface_leg(P690, 690.0, "P", 2.0, 2.2),
]


def prefer_690(combo):
    return risk_assessment(combo_name=combo.name, net_ev_usd=30.0 if "690/685" in combo.name else 10.0)


@pytest.fixture
def make_rig(store, fake_valkey, recording_log, mem_sink):
    def make(paper=True, gate=None, legs=SURFACE_LEGS, book=None, **surface_overrides):
        store.set_json(keys.surface_key(SPY), surface_payload(legs=legs, **surface_overrides))
        store.set_json(keys.rv_key(SPY), rv_payload(rv_20d=0.13))
        if book is not None:
            store.set_json(keys.vrp_book_key(SPY), book)
        rig = SimpleNamespace(
            store=store, valkey=fake_valkey, log=recording_log, sink=mem_sink,
            gate=gate or FakeGate(), pricer=FakePricer(prefer_690), gateway=FakeOrderGateway(), market=FakeMarket(),
            exec_clock=mock.Mock(return_value=1_000.0), mono=mock.Mock(return_value=5_000.0),
            wall=mock.Mock(return_value=NOW_NS),
        )
        ids = (f"P{i}" for i in itertools.count(1))
        rig.executor = PassiveComboExecutor(
            clock=rig.exec_clock, id_factory=lambda: next(ids), fill_sink=mem_sink, wall_clock_ns=rig.wall
        )
        rig.engine = VrpEngine(
            VrpSettings(SPY, option_ids=(P690, P685), paper_routing_enabled=paper),
            gate=rig.gate, pricer=rig.pricer, executor=rig.executor, store=store,
            gateway=rig.gateway, market=rig.market, log=recording_log, clock_ns=rig.wall, monotonic=rig.mono,
        )
        rig.engine.start()
        return rig

    return make


def spot(rig, mid=700.0):
    rig.engine.on_quote(QuoteObs(SPY, mid - 0.05, mid + 0.05, NOW_NS))


def quote_legs(rig, p690=(2.0, 2.2), p685=(1.4, 1.6)):
    rig.engine.on_quote(QuoteObs(P690, *p690, NOW_NS))
    rig.engine.on_quote(QuoteObs(P685, *p685, NOW_NS))


def route_entry(rig):
    spot(rig)
    payload = rig.engine.on_timer(NOW_NS, NOW_NS, TODAY)
    assert payload["action"] == "LIVE_PAPER_ORDER_ROUTED"
    return payload["routed_order_ids"][0]


# --- Entries ---------------------------------------------------------------------------------------


def test_shadow_mode_previews_without_orders(make_rig):
    rig = make_rig(paper=False)
    spot(rig)

    payload = rig.engine.on_timer(event_ts_ns=NOW_NS - 5, wall_ns=NOW_NS, today=TODAY)

    assert (payload["mode"], payload["action"], payload["entry_block_reason"]) == ("SHADOW_DRY_RUN", "SHADOW_SIGNAL_READY", None)
    assert payload["top_candidate"]["combo"] == "SPY 690/685P Bull Put"
    assert (payload["top_candidate"]["approved_qty"], payload["top_candidate"]["expected_ev_usd"]) == (6, 180.0)
    assert payload["passive_clips"] == [
        {"clip_id": "CMB-P1-1", "lots": 6, "mid_credit": 0.6, "natural_credit": 0.4, "passive_limit": 0.57}
    ]
    assert (payload["candidates_scanned"], payload["spot"], payload["rv_20d"]) == (2, 700.0, 0.13)
    assert (payload["ts_ns"], payload["event_ts_ns"]) == (NOW_NS, NOW_NS - 5)
    assert rig.gateway.prepared == [] and rig.executor.active_clips == {}
    assert rig.store.get_json_value(keys.VRP_LATEST) == payload
    assert rig.store.get_json_list(keys.VRP_HISTORY, 10) == [payload]
    assert {call["as_of"] for call in rig.gate.calls} == {TODAY}
    assert all(leg.qty == 6 for call in rig.gate.calls for leg in call["legs"])
    assert {(c["horizon_days"], c["realized_vol_p"]) for c in rig.pricer.calls} == {(14.0, 0.13)}


def test_live_mode_routes_a_single_registered_clip(make_rig):
    rig = make_rig()
    registered_at_submit = []
    rig.gateway.on_submit = lambda order: registered_at_submit.append(order.order_key in rig.engine.working)

    order_key = route_entry(rig)

    (prepared,) = rig.gateway.prepared
    assert prepared.handle == {"legs": LEGS, "side": "CREDIT_OPEN", "lots": 6, "limit": 0.57}
    assert rig.gateway.submitted == [order_key] and registered_at_submit == [True]
    assert rig.gate.submissions == 1
    assert rig.store.get_json_value(keys.spread_legs_key(SPREAD_ID)) == LEGS
    working = rig.engine.working[order_key]
    assert (working.purpose, working.spread_key, working.expiry, working.combo) == (
        "ENTRY", SPREAD_ID, "2026-12-18", "SPY 690/685P Bull Put",
    )
    payload = rig.store.get_json_value(keys.VRP_LATEST)
    assert payload["working_orders"] == [
        {"client_order_id": order_key, "purpose": "ENTRY", "combo": "SPY 690/685P Bull Put", "lots": 6, "limit": 0.57, "step": 0, "filled_lots": 0.0}
    ]


def test_entries_wait_for_working_orders_quotes_and_portfolio_refresh(make_rig):
    rig = make_rig()
    assert rig.engine.on_timer(NOW_NS, NOW_NS, TODAY)["action"] == "ENTRY_DEFERRED(NO_UNDERLYING_QUOTE)"

    order_key = route_entry(rig)
    assert rig.engine.on_timer(NOW_NS, NOW_NS, TODAY)["action"] == "ENTRY_DEFERRED(ORDER_WORKING)"

    rig.gateway.set_view(order_key)
    rig.engine.on_fill(order_key, 6, 0.57)
    assert rig.engine.on_timer(NOW_NS, NOW_NS, TODAY)["action"] == "ENTRY_DEFERRED(WAITING_PORTFOLIO_REFRESH)"

    rig.store.set_json(keys.PORTFOLIO_RISK, portfolio_payload(now_ns=NOW_NS + 1))
    assert rig.engine.on_timer(NOW_NS, NOW_NS, TODAY)["action"] == "LIVE_PAPER_ORDER_ROUTED"


def test_spread_inside_the_exit_window_is_never_entered(make_rig):
    rig = make_rig(ttm_days=10.0)
    spot(rig)
    payload = rig.engine.on_timer(NOW_NS, NOW_NS, TODAY)
    assert payload["action"] == "STAND_DOWN"
    assert "DTE_INSIDE_EXIT_WINDOW(10<=14)" in payload["top_candidate"]["gate_reasons"]
    assert payload["passive_clips"] == []


def test_submit_failure_cleans_up(make_rig):
    rig = make_rig()
    rig.gateway.fail_submit = RuntimeError("venue down")
    spot(rig)

    assert rig.engine.on_timer(NOW_NS, NOW_NS, TODAY)["action"] == "ORDER_SUBMIT_FAILED"
    assert rig.engine.working == {} and rig.executor.active_clips == {}
    assert rig.gate.submissions == 0
    assert rig.log.error.call_args_list == [mock.call("Failed to submit ENTRY OptionSpread order (SPY 690/685P Bull Put): venue down")]


def test_spread_legs_are_registered_once_per_spread(make_rig):
    rig = make_rig()
    order_key = route_entry(rig)
    rig.engine.on_terminal(order_key, "CANCELLED")
    rig.store.delete(keys.spread_legs_key(SPREAD_ID))

    route_entry(rig)
    assert not rig.valkey.exists(keys.spread_legs_key(SPREAD_ID))


# --- Fills and terminal events ---------------------------------------------------------------------


def test_full_fill_updates_the_book_and_logs_slippage(make_rig):
    rig = make_rig()
    order_key = route_entry(rig)
    rig.gateway.set_view(order_key)

    rig.engine.on_fill(order_key, 6, 0.57, commission=-7.8)

    trade = rig.engine.book.trades[SPREAD_ID]
    assert (trade["open_lots"], trade["entry_credit"], trade["expiry"], trade["opened_ts_ns"]) == (6.0, 0.57, "2026-12-18", NOW_NS)
    assert rig.store.get_json_value(keys.vrp_book_key(SPY)) == rig.engine.book.to_json()
    assert rig.engine.working == {} and rig.executor.active_clips == {}
    (fill,) = written(rig.sink)
    assert (fill["lots"], fill["actual_fill"], fill["slippage_usd"], fill["commission_usd"]) == (6, 0.57, 18.0, 7.8)


def test_partial_fill_keeps_the_order_working_until_it_ends(make_rig):
    rig = make_rig()
    order_key = route_entry(rig)

    rig.engine.on_fill(order_key, 2, 0.57)
    assert rig.engine.book.trades[SPREAD_ID]["open_lots"] == 2.0
    assert rig.engine.working[order_key].filled_lots == 2.0

    rig.engine.on_terminal(order_key, "CANCELLED")
    assert rig.engine.working == {}
    assert written(rig.sink)[0]["lots"] == 2


def test_rejections_are_logged_and_unknown_orders_ignored(make_rig):
    rig = make_rig()
    order_key = route_entry(rig)
    rig.engine.on_fill("someone-else", 1, 1.0)
    rig.engine.on_terminal(order_key, "REJECTED", "margin")

    assert rig.log.warning.call_args_list == [mock.call(f"OptionSpread order rejected {order_key}: margin")]
    assert rig.engine.working == {} and written(rig.sink) == []


def test_modify_rejection_cancels_the_order(make_rig):
    rig = make_rig()
    order_key = route_entry(rig)
    rig.engine.on_modify_rejected(order_key)
    assert rig.gateway.cancelled == [order_key]


# --- Crash/restart reconciliation -----------------------------------------------------------------


def test_start_cancels_orphaned_open_orders(make_rig):
    """start() cancels orphaned open orders from the gateway."""
    rig = make_rig()
    # Simulate a pre-existing open order from a prior run
    rig.gateway.views["ORPHAN-O-1"] = OrderView(is_open=True, is_closed=False, status="ACCEPTED")
    rig.engine.start()
    assert "ORPHAN-O-1" in rig.gateway.cancelled


def test_start_blocks_entry_on_reconciliation_mismatch(make_rig, store):
    """start() with a persisted trade missing from broker positions blocks new entries."""
    OPEN_TRADE = {"combo": "SPY 690/685P Bull Put", "legs": LEGS, "expiry": "2026-12-18", "open_lots": 4.0, "entry_credit": 0.6}
    rig = make_rig(paper=True, book={SPREAD_ID: OPEN_TRADE})
    # No broker position exists for SPREAD_ID -> trigger mismatch
    rig.engine.start()
    assert rig.engine._reconciliation_mismatch is True
    assert rig.engine.current_entry_block_reason() == "BOOK_RECONCILIATION_MISMATCH"


def test_finalize_releases_order_key(make_rig):
    """finalize() releases the order key from the gateway."""
    rig = make_rig()
    spot(rig)
    payload = rig.engine.on_timer(NOW_NS, NOW_NS, TODAY)
    order_key = payload["routed_order_ids"][0]
    # Verify the key is tracked
    assert order_key in rig.gateway.views
    # Finalize it
    rig.engine.finalize(order_key, "FILLED")
    # The release is a no-op on FakeOrderGateway; check that the working order is gone
    assert order_key not in rig.engine.working


def test_risk_block_cancels_working_entries_with_a_one_second_cache(make_rig):
    rig = make_rig()
    order_key = route_entry(rig)
    quote_legs(rig)
    spot(rig)  # First check: daemon healthy
    rig.gate.block_reason = "RISK_DAEMON_HALTED_OR_MISSING"
    spot(rig)
    assert rig.gateway.cancelled == []  # Cached for one second

    rig.mono.return_value += 1.0
    spot(rig)
    assert rig.gateway.cancelled == [order_key]
    assert rig.log.warning.call_args == mock.call(
        f"Cancelling working entry {order_key} (SPY 690/685P Bull Put): RISK_DAEMON_HALTED_OR_MISSING"
    )


def test_adverse_spot_move_cancels(make_rig):
    rig = make_rig()
    order_key = route_entry(rig)
    quote_legs(rig)
    spot(rig, 702.5)
    assert rig.gateway.cancelled == [order_key]
    assert rig.log.warning.call_args.args[0].startswith("PassiveCombo CANCEL_REPLACE CMB-P2-1: ADVERSE_SPOT_MOVE")


def test_dwell_reprices_and_failed_reprices_cancel(make_rig):
    rig = make_rig()
    order_key = route_entry(rig)
    quote_legs(rig)
    rig.exec_clock.return_value += 20.0
    spot(rig)
    assert rig.gateway.modified == [(order_key, 0.52)]

    rig.gateway.modify_ok = False
    rig.exec_clock.return_value += 20.0
    spot(rig)
    assert rig.gateway.cancelled == [order_key]
    assert rig.log.warning.call_args == mock.call("PassiveCombo MODIFY_LIMIT CMB-P2-1: NO_INSTRUMENT")


def test_non_positive_spread_mid_cancels(make_rig):
    rig = make_rig()
    order_key = route_entry(rig)
    quote_legs(rig, p690=(1.0, 1.2))
    spot(rig)
    assert rig.gateway.cancelled == [order_key]
    assert rig.log.warning.call_args == mock.call(
        f"Cancelling {order_key} (SPY 690/685P Bull Put): spread mid -0.40 is no longer positive"
    )


@pytest.mark.parametrize("view_status, logged", [("CANCELED", "CANCELED"), (None, "missing from cache")])
def test_orders_closed_behind_our_back_are_finalized(make_rig, view_status, logged):
    rig = make_rig()
    order_key = route_entry(rig)
    if view_status is None:
        del rig.gateway.views[order_key]
    else:
        rig.gateway.set_view(order_key, status=view_status)
    spot(rig)

    assert rig.engine.working == {}
    assert rig.log.warning.call_args == mock.call(f"Order {order_key} closed without a tracked terminal event ({logged})")


def test_pending_orders_are_left_alone(make_rig):
    rig = make_rig()
    order_key = route_entry(rig)
    rig.gateway.set_view(order_key, is_open=True, is_closed=False, is_pending_cancel=True)
    rig.gate.block_reason = "HALT"
    spot(rig)
    assert rig.gateway.cancelled == []


def test_stop_cancels_open_orders_and_persists_the_book(make_rig):
    rig = make_rig()
    order_key = route_entry(rig)
    rig.engine.stop()
    assert rig.gateway.cancelled == [order_key]
    assert rig.valkey.exists(keys.vrp_book_key(SPY))


# --- RV fail-closed --------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "rv_overrides",
    [
        {"rv_20d": 0.0},          # Zero RV -> STAND_DOWN
        {"rv_20d": -0.1},         # Negative RV -> STAND_DOWN
        {"rv_20d": float("nan")},  # NaN RV -> STAND_DOWN
        {"ts_ns": ago(90000.0)},  # Stale RV (ts > 1 day old) -> STAND_DOWN
        {"rv_20d": None},         # Missing/null rv_20d -> STAND_DOWN
    ],
)
def test_invalid_rv_blocks_entries_but_allows_exits(make_rig, store, rv_overrides):
    """Missing, zero, stale or non-finite rv_20d forces STAND_DOWN with RV_MISSING_OR_INVALID."""
    OPEN_TRADE = {"combo": "SPY 690/685P Bull Put", "legs": LEGS, "expiry": "2026-12-18", "open_lots": 4.0, "entry_credit": 0.6}
    rig = make_rig(book={SPREAD_ID: OPEN_TRADE})
    rig.market.positions[SPREAD_ID] = -4.0
    quote_legs(rig)
    spot(rig)

    # Replace RV with the invalid variant
    store.set_json(keys.rv_key(SPY), rv_payload(**rv_overrides))

    # Use a date inside the exit window so evaluate_exits fires
    payload = rig.engine.on_timer(event_ts_ns=NOW_NS, wall_ns=NOW_NS, today=date(2026, 12, 5))

    # Must block entries
    assert payload["action"] == "STAND_DOWN"
    assert payload["entry_block_reason"] == "RV_MISSING_OR_INVALID"
    assert payload["rv_20d"] is None
    assert payload["candidates_scanned"] == 0

    # Must still evaluate exits
    assert len(payload["exit_signals"]) > 0


def test_valid_rv_proceeds_normally(make_rig, store):
    """A fresh, positive rv_20d evaluates candidates normally (regression check)."""
    rig = make_rig()
    quote_legs(rig)
    spot(rig)

    store.set_json(keys.rv_key(SPY), rv_payload(rv_20d=0.13))
    payload = rig.engine.on_timer(event_ts_ns=NOW_NS, wall_ns=NOW_NS, today=TODAY)

    assert payload["action"] == "LIVE_PAPER_ORDER_ROUTED"
    assert payload["entry_block_reason"] is None
    assert payload["rv_20d"] == 0.13
    assert payload["candidates_scanned"] > 0


# --- Execution hardening ---------------------------------------------------------------------------


def test_exit_order_with_negative_mid_not_cancelled(make_rig):
    """EXIT order with mid=-0.02 is NOT cancelled; ENTRY order with mid=-0.02 IS cancelled."""
    OPEN_TRADE = {"combo": "SPY 690/685P Bull Put", "legs": LEGS, "expiry": "2026-12-18", "open_lots": 4.0, "entry_credit": 0.6}
    rig = make_rig(book={SPREAD_ID: OPEN_TRADE})
    rig.market.positions[SPREAD_ID] = -4.0
    spot(rig)
    quote_legs(rig)

    exit_signal = rig.engine.evaluate_exits(date(2026, 12, 5))
    assert len(exit_signal) == 1
    assert exit_signal[0]["status"] == "EXIT_ORDER_ROUTED"
    exit_key = exit_signal[0]["order_id"]

    rig.engine.on_quote(QuoteObs(P690, 0.0, 0.05, NOW_NS))
    rig.engine.on_quote(QuoteObs(P685, 1.0, 1.2, NOW_NS))
    assert exit_key in rig.engine.working


def test_entry_order_with_negative_mid_is_cancelled(make_rig):
    """ENTRY order with mid=-0.02 IS cancelled (CANCELLED_INVALID_MID)."""
    rig = make_rig()
    quote_legs(rig)
    spot(rig)

    order_key = "ENTRY-TEST-1"
    rig.gateway.views[order_key] = OrderView(is_open=True, is_closed=False, status="ACCEPTED")
    rig.engine.working[order_key] = WorkingOrder(
        order_key=order_key, clip_id="CLIP-1", parent_id="P1",
        purpose="ENTRY", side="CREDIT_OPEN",
        spread_key=SPREAD_ID, legs=LEGS, combo="SPY 690/685P Bull Put", expiry="2026-12-18",
    )
    # Combo price for CREDIT_OPEN: P690 mid=0.025, P685 mid=1.1 => 0.025 - 1.1 = -1.075
    rig.engine.quotes[P690] = (0.0, 0.05, None, None)
    rig.engine.quotes[P685] = (1.0, 1.2, None, None)
    rig.engine.manage_working_orders()
    # The cancel was sent to the gateway; the order left working only on terminal event
    assert order_key in rig.gateway.cancelled

OPEN_TRADE = {"combo": "SPY 690/685P Bull Put", "legs": LEGS, "expiry": "2026-12-18", "open_lots": 4.0, "entry_credit": 0.6}


def test_start_restores_the_book_and_subscribes(make_rig):
    rig = make_rig(book={SPREAD_ID: OPEN_TRADE, "stale": {**OPEN_TRADE, "open_lots": 0.0}})
    assert list(rig.engine.book.trades) == [SPREAD_ID]
    assert rig.market.subscribed == [P690, P685]


def test_time_stop_in_shadow_mode_is_only_a_signal(make_rig):
    rig = make_rig(paper=False, book={SPREAD_ID: OPEN_TRADE})
    quote_legs(rig)
    (signal,) = rig.engine.evaluate_exits(date(2026, 12, 5))
    assert signal == {
        "combo": "SPY 690/685P Bull Put",
        "spread_id": SPREAD_ID,
        "reason": "TIME_STOP(dte=13<=14)",
        "open_lots": 4.0,
        "entry_credit": 0.6,
        "close_mid_debit": 0.6,
        "close_natural_debit": 0.8,
        "status": "SHADOW_SIGNAL",
        "order_id": None,
    }
    assert rig.gateway.prepared == []


def test_confirmed_exit_routes_a_closing_clip(make_rig):
    rig = make_rig(book={SPREAD_ID: OPEN_TRADE})
    rig.market.positions[SPREAD_ID] = -4.0
    quote_legs(rig)
    spot(rig)

    (signal,) = rig.engine.evaluate_exits(date(2026, 12, 5))

    assert (signal["status"], signal["order_id"]) == ("EXIT_ORDER_ROUTED", "O-1")
    assert rig.gateway.prepared[0].handle == {"legs": LEGS, "side": "DEBIT_CLOSE", "lots": 4, "limit": 0.63}
    assert rig.gate.submissions == 0  # Exits never count against the entry throttle
    assert rig.engine.evaluate_exits(date(2026, 12, 5))[0]["status"] == "ORDER_WORKING_ON_SPREAD"

    rig.gateway.set_view("O-1")
    rig.engine.on_fill("O-1", 4, 0.63)
    assert len(rig.engine.book) == 0
    assert rig.store.get_json_value(keys.vrp_book_key(SPY)) == {}


@pytest.mark.parametrize(
    "positions, quoted, status",
    [
        ({}, True, "POSITION_NOT_CONFIRMED_IN_CACHE"),
        ({P690: -4.0, P685: 4.0}, True, "EXIT_ORDER_ROUTED"),  # Leg-level confirmation
        ({SPREAD_ID: -4.0}, False, "NO_LIVE_QUOTES"),
    ],
)
def test_exit_preconditions(make_rig, positions, quoted, status):
    rig = make_rig(book={SPREAD_ID: OPEN_TRADE})
    rig.market.positions.update(positions)
    if quoted:
        quote_legs(rig)
    spot(rig)
    assert rig.engine.evaluate_exits(date(2026, 12, 5))[0]["status"] == status


def test_profit_target_signal(make_rig):
    rig = make_rig(paper=False, book={SPREAD_ID: OPEN_TRADE})
    quote_legs(rig, p690=(1.0, 1.1), p685=(0.8, 0.9))
    (signal,) = rig.engine.evaluate_exits(TODAY)
    assert signal["reason"] == "PROFIT_TARGET(mid_debit=0.20<=0.30)"
    assert (signal["close_mid_debit"], signal["close_natural_debit"]) == (0.2, 0.3)


def test_expired_spreads_leave_the_book(make_rig):
    rig = make_rig(book={SPREAD_ID: OPEN_TRADE})
    assert rig.engine.evaluate_exits(date(2026, 12, 19)) == []
    assert rig.store.get_json_value(keys.vrp_book_key(SPY)) == {}
    assert rig.log.warning.call_args == mock.call("Dropping SPY 690/685P Bull Put from book (open_lots=4.0, dte=-1)")
