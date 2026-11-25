import json
from datetime import date

import pytest

from src.common.market import OptionSpec, QuoteObs
from src.risk.gate_types import GateDecision
from src.strategies.vrp_core import (
    SpreadBook,
    apply_dte_window,
    build_candidates,
    candidate_row,
    enrich_surface_legs,
    entry_block_reason,
    exit_confirmed,
    exit_reason,
    gate_leg,
    leg_mismatches,
    normalize_right,
    per_lot_risk,
    rank_key,
)
from tests.factories import NOW_NS, risk_assessment, surface_leg, surface_payload, utc_ns

TODAY = date(2026, 11, 24)
P690, P685, P680 = "SPY P690", "SPY P685", "SPY P680"
LEGS = [{"instrument_id": P690, "ratio": 1}, {"instrument_id": P685, "ratio": -1}]


def no_quote(_):
    return None


# --- Spread book -----------------------------------------------------------------------------------


def test_entry_fills_average_the_credit_and_exit_fills_close():
    book = SpreadBook()
    book.apply_entry_fill("S1", "SPY 690/685P Bull Put", LEGS, "2026-12-18", 0.60, 2.0, NOW_NS)
    trade = book.apply_entry_fill("S1", "ignored", LEGS, "ignored", 0.54, 4.0, NOW_NS + 1)

    assert (trade["open_lots"], trade["entry_credit"]) == (6.0, pytest.approx(0.56))
    assert (trade["combo"], trade["opened_ts_ns"]) == ("SPY 690/685P Bull Put", NOW_NS)

    book.apply_exit_fill("S1", 2.0)
    assert book.trades["S1"]["open_lots"] == 4.0
    book.apply_exit_fill("S1", 4.0)
    assert "S1" not in book
    book.apply_exit_fill("unknown", 1.0)


def test_json_round_trip_keeps_only_open_well_formed_trades():
    good = {"combo": "c", "legs": LEGS, "expiry": "2026-12-18", "open_lots": 3.0, "entry_credit": 0.6}
    raw = {
        "good": good,
        "closed": {**good, "open_lots": 0.0},
        "no_legs": {**good, "legs": []},
        "no_expiry": {**good, "expiry": None},
        "garbage_lots": {**good, "open_lots": "three"},
        "not_a_trade": [1, 2],
    }
    book = SpreadBook.from_json(raw)
    assert list(book.trades) == ["good"]
    assert SpreadBook.from_json(json.loads(json.dumps(book.to_json()))).trades == book.trades


def test_expected_leg_positions_net_across_trades():
    book = SpreadBook(
        {
            "A": {"legs": LEGS, "open_lots": 4.0},
            "B": {"legs": [{"instrument_id": P685, "ratio": 1}, {"instrument_id": P680, "ratio": -1}], "open_lots": 2.0},
        }
    )
    assert book.expected_leg_positions() == {P690: -4.0, P685: 2.0, P680: 2.0}


def test_book_rows():
    book = SpreadBook({"S1": {"combo": "c", "legs": LEGS, "expiry": "2026-12-18", "open_lots": 2.0, "entry_credit": 0.123456}})
    assert book.rows() == [{"spread_id": "S1", "combo": "c", "open_lots": 2.0, "entry_credit": 0.1235, "expiry": "2026-12-18"}]


# --- Exit rules ------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "dte, mid_debit, reason",
    [
        (14, 0.50, "TIME_STOP(dte=14<=14)"),
        (14, 0.10, "TIME_STOP(dte=14<=14)"),  # Time stop wins
        (15, 0.30, "PROFIT_TARGET(mid_debit=0.30<=0.30)"),
        (15, 0.31, None),
        (15, None, None),
    ],
)
def test_exit_reason(dte, mid_debit, reason):
    assert exit_reason(dte, mid_debit, entry_credit=0.60, profit_target_fraction=0.5, min_exit_dte=14) == reason


def test_exit_confirmation_by_spread_or_leg_positions():
    trade = {"legs": LEGS, "open_lots": 4.0}
    expected = SpreadBook({"S1": trade}).expected_leg_positions()

    spread_short = {"S1": -4.0}.get
    assert exit_confirmed("S1", trade, set(), lambda iid: spread_short(iid, 0.0))

    legs_held = {P690: -4.0, P685: 4.0}
    assert leg_mismatches(expected, lambda iid: legs_held.get(iid, 0.0)) == set()
    assert exit_confirmed("S1", trade, set(), lambda iid: legs_held.get(iid, 0.0))

    nothing_held = leg_mismatches(expected, lambda iid: 0.0)
    assert nothing_held == {P690, P685}
    assert not exit_confirmed("S1", trade, nothing_held, lambda iid: 0.0)


# --- Entry rules -----------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "working, spot, last_fill, portfolio_ts, reason",
    [
        (True, 700.0, 0, 0, "ORDER_WORKING"),
        (False, 0.0, 0, 0, "NO_UNDERLYING_QUOTE"),
        (False, 700.0, 100, 100, "WAITING_PORTFOLIO_REFRESH"),
        (False, 700.0, 100, 101, None),
        (False, 700.0, 0, 0, None),
    ],
)
def test_entry_block_reason(working, spot, last_fill, portfolio_ts, reason):
    assert entry_block_reason(working, spot, last_fill, portfolio_ts) == reason


def test_dte_window_rejects_spreads_the_time_stop_would_close():
    decision = apply_dte_window(GateDecision(True, 5), 14, 14)
    assert (decision.approved, decision.approved_qty, decision.reasons) == (False, 0, ["DTE_INSIDE_EXIT_WINDOW(14<=14)"])
    assert apply_dte_window(GateDecision(True, 5), 15, 14).approved


@pytest.mark.parametrize("raw, right", [("p", "P"), ("PUT", "P"), ("Call", "C"), ("c", "C"), (None, ""), ("x", "X")])
def test_normalize_right(raw, right):
    assert normalize_right(raw) == right


# --- Leg enrichment --------------------------------------------------------------------------------


def enrich(raw_legs, strategy=no_quote, cache=no_quote, specs=no_quote):
    return enrich_surface_legs(raw_legs, 0.20, date(2026, 12, 18), strategy, cache, specs)


def test_quote_precedence_strategy_then_cache_then_surface():
    raw = [surface_leg("A", 690.0, "P", 2.0, 2.2), surface_leg("B", 685.0, "P", 1.4, 1.6), surface_leg("C", 680.0, "P", 1.0, 1.2)]
    strategy = {"A": (2.05, 2.15, 30, 40)}.get
    cache = {"A": QuoteObs("A", 1.0, 3.0, 0), "B": QuoteObs("B", 1.45, 1.55, 0, bid_size=7.0, ask_size=None)}.get

    a, b, c = enrich(raw, strategy, cache)

    assert (a["bid"], a["ask"], a["bid_size"], a["ask_size"]) == (2.05, 2.15, 30, 40)
    assert (b["bid"], b["ask"], b["bid_size"], b["ask_size"]) == (1.45, 1.55, 7, None)
    assert (c["bid"], c["ask"], c["bid_size"], c["ask_size"]) == (1.0, 1.2, None, None)
    assert c["mid"] == pytest.approx(1.1)


def test_unusable_rows_are_skipped():
    raw = [
        surface_leg("", 690.0, "P", 2.0, 2.2),
        surface_leg("NO_BID", 690.0, "P", 0.0, 2.2),
        surface_leg("CROSSED", 690.0, "P", 2.3, 2.2),
    ]
    assert enrich(raw) == []


def test_open_interest_and_expiry_sources():
    spec = OptionSpec("B", "SPY.ARCA", 685.0, False, expiry_ns=utc_ns(2027, 1, 15, 21), open_interest=321.0)
    raw = [
        surface_leg("A", 690.0, "P", 2.0, 2.2, open_interest=1_500),
        surface_leg("B", 685.0, "P", 1.4, 1.6, open_interest=None),
        surface_leg("C", 680.0, "P", 1.0, 1.2, open_interest=0),
        surface_leg("D", 675.0, "P", 0.8, 1.0, open_interest="n/a"),
    ]
    a, b, c, d = enrich(raw, specs={"B": spec}.get)

    assert (a["open_interest"], b["open_interest"], c["open_interest"], d["open_interest"]) == (1_500, 321, None, None)
    assert (a["expiry"], b["expiry"]) == (date(2026, 12, 18), date(2027, 1, 15))


def test_missing_vols_default_to_atm():
    (row,) = enrich([surface_leg("A", 690.0, "put", 2.0, 2.2, raw_iv=None, svi_iv=0.0, vega=None)])
    assert (row["right"], row["raw_iv"], row["svi_iv"], row["vega"]) == ("P", 0.20, 0.20, 0.0)


def test_gate_leg_carries_the_real_extrinsic_value():
    (call,) = enrich([surface_leg("C690", 690.0, "C", 12.4, 12.6)])
    (put,) = enrich([surface_leg("P690", 690.0, "P", 2.0, 2.2)])

    short_call = gate_leg(call, "SELL", 700.0)
    assert (short_call.right, short_call.side, short_call.qty, short_call.open_interest) == ("C", "SELL", 1, 1_000)
    assert short_call.extrinsic_value == pytest.approx(12.5 - 10.0)
    assert gate_leg(put, "BUY", 700.0).extrinsic_value == pytest.approx(2.1)  # Out of the money: all time value


# --- Candidates ------------------------------------------------------------------------------------


def candidates(legs, spot_mid=0.0, **surface_overrides):
    surface = surface_payload(legs=legs, **surface_overrides)
    return build_candidates(
        surface, spot_mid=spot_mid, today=TODAY, symbol="SPY", underlying="SPY.ARCA",
        strategy_quote=no_quote, cache_quote=no_quote, option_spec=no_quote,
    )


PUT_LEGS = [
    surface_leg(P680, 680.0, "P", 1.0, 1.2),
    surface_leg(P685, 685.0, "P", 1.4, 1.6),
    surface_leg(P690, 690.0, "P", 2.0, 2.2),
]


def test_candidates_are_adjacent_credit_spreads():
    cands = candidates(PUT_LEGS)

    assert [c["combo"].name for c in cands] == ["SPY 685/680P Bull Put", "SPY 690/685P Bull Put"]
    best = cands[1]
    assert best["legs"] == [{"instrument_id": P690, "ratio": 1}, {"instrument_id": P685, "ratio": -1}]
    assert (best["mid_credit"], best["nat_credit"]) == (pytest.approx(0.6), pytest.approx(0.4))
    assert best["expiry"] == date(2026, 12, 18)  # NY "today" + 24 surface days
    assert [(g.instrument_id, g.side) for g in best["gate_legs"]] == [(P690, "SELL"), (P685, "BUY")]
    combo = best["combo"]
    assert (combo.underlying, combo.spot, combo.forward, combo.dividend_yield, combo.risk_free_rate) == (
        "SPY.ARCA", 700.0, 701.0, 0.0125, 0.0425,
    )


def test_live_underlying_mid_overrides_the_surface_spot():
    assert candidates(PUT_LEGS, spot_mid=705.0)[0]["combo"].spot == 705.0


def test_no_spot_or_no_legs_means_no_candidates():
    assert candidates(PUT_LEGS, spot=0.0) == []
    assert candidates([]) == []


def test_credit_at_or_below_five_cents_is_skipped():
    legs = [surface_leg(P685, 685.0, "P", 1.40, 1.50), surface_leg(P690, 690.0, "P", 1.44, 1.54)]
    assert candidates(legs) == []  # 0.04 credit
    legs[1] = surface_leg(P690, 690.0, "P", 1.46, 1.56)
    assert len(candidates(legs)) == 1  # 0.06 credit


# --- Rows and ranking ------------------------------------------------------------------------------


def test_per_lot_risk_floors_tail_loss_and_requests_at_least_one_lot():
    assert per_lot_risk(risk_assessment(es_99_usd=-150.0, net_ev_usd=12.0, recommended_contracts=6)) == (150.0, 12.0, 6)
    assert per_lot_risk(risk_assessment(es_99_usd=0.4, recommended_contracts=0))[::2] == (1.0, 1)


def test_candidate_row_and_ranking():
    cand = candidates(PUT_LEGS)[1]
    approved = candidate_row(cand, risk_assessment(prob_profit=0.71), GateDecision(True, 3, [], ["w"]), 24, 150.0, 12.0)
    assert approved == {
        "combo": "SPY 690/685P Bull Put",
        "mid_credit": 0.6,
        "nat_credit": 0.4,
        "dte": 24,
        "net_ev_per_lot": 12.0,
        "es99_per_lot": 150.0,
        "ev_to_es99": 0.08,
        "pop_pct": 71.0,
        "recommended_qty": 6,
        "gate_approved": True,
        "approved_qty": 3,
        "expected_ev_usd": 36.0,
        "gate_reasons": [],
        "gate_warnings": ["w"],
    }
    rejected_richer = {**approved, "gate_approved": False, "expected_ev_usd": 500.0}
    bigger = {**approved, "expected_ev_usd": 48.0}
    ranked = sorted([rejected_richer, approved, bigger], key=rank_key, reverse=True)
    assert ranked == [bigger, approved, rejected_richer]
