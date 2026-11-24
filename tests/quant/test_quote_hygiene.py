import numpy as np
import pytest

from src.quant.quote_hygiene import (
    SliceFilter,
    SliceQuote,
    clean_quote_mask,
    is_stale,
    select_slice_quotes,
)
from tests.factories import NOW_NS, ago

FORWARD, DISC = 700.0, 0.99


def _q(name, strike, is_call, bid, ask, ts_ns=NOW_NS, oi=0.0):
    return SliceQuote(name, strike, is_call, bid, ask, 0.5 * (bid + ask), ts_ns, oi)


def _select(quotes, **filter_kwargs):
    return select_slice_quotes(quotes, FORWARD, DISC, NOW_NS, SliceFilter(**filter_kwargs))


def test_otm_quotes_are_kept_with_their_spreads():
    keep_put = _q("P690", 690.0, False, 4.9, 5.1)
    keep_call = _q("C710", 710.0, True, 3.0, 3.2)
    sel = _select([keep_put, keep_call])
    assert sel.quotes == [keep_put, keep_call]
    assert sel.spreads == pytest.approx([0.2, 0.2])
    assert sum(sel.dropped.values()) == 0


@pytest.mark.parametrize(
    "quote, reason",
    [
        (_q("stale", 690.0, False, 4.9, 5.1, ts_ns=ago(91.0)), "stale"),
        (_q("cheap", 650.0, False, 0.08, 0.12), "spread"),  # Mid 0.10 < 0.15
        (_q("wide", 690.0, False, 4.0, 6.0), "spread"),  # 40% of mid
        (_q("itm_call", 690.0, True, 14.0, 14.2), "itm"),  # 690 < 700 * (1 - 0.6%)
        (_q("itm_put", 710.0, False, 13.0, 13.2), "itm"),  # 710 > 700 * (1 + 0.6%)
        (_q("below_intrinsic", 697.0, True, 2.85, 2.95), "intrinsic"),  # 0.99 * 3 = 2.97 > 2.90
    ],
)
def test_each_drop_reason_is_counted(quote, reason):
    sel = _select([quote])
    assert sel.quotes == []
    assert sel.dropped[reason] == 1
    assert sum(sel.dropped.values()) == 1


def test_drop_counter_keys_are_stable():
    assert list(_select([]).dropped) == ["stale", "spread", "intrinsic", "itm", "oi"]


def test_open_interest_gate_applies_only_to_known_open_interest():
    thin = _q("thin", 690.0, False, 4.9, 5.1, oi=50.0)
    unknown = _q("unknown", 685.0, False, 3.9, 4.1, oi=0.0)
    deep = _q("deep", 680.0, False, 2.9, 3.1, oi=500.0)

    sel = _select([thin, unknown, deep], min_open_interest=100)

    assert [q.instrument_id for q in sel.quotes] == ["unknown", "deep"]
    assert sel.dropped["oi"] == 1


def test_locked_quotes_get_a_minimum_spread():
    sel = _select([_q("locked", 690.0, False, 5.0, 5.0)])
    assert sel.spreads == [0.01]


def test_staleness_check_can_be_disabled():
    assert is_stale(ago(91.0), NOW_NS, 90.0)
    assert not is_stale(ago(90.0), NOW_NS, 90.0)
    assert not is_stale(ago(10_000.0), NOW_NS, 0.0)


# --- Whole-chain mask ----------------------------------------------------------------------------

S, T, R = 700.0, 0.1, 0.04


def _mask(K, prices, flags, **kwargs):
    n = len(K)
    return clean_quote_mask(
        np.full(n, S), np.asarray(K, dtype=float), np.full(n, T), np.full(n, R),
        np.asarray(prices, dtype=float), np.asarray(flags), **kwargs
    )


def test_mask_keeps_sane_quotes_and_rejects_bad_ones():
    # intrinsic of the 650 call = 700 - 650 * exp(-0.004) ~= 52.59
    K = [650.0, 650.0, 690.0, 690.0, 720.0, 1200.0, 680.0]
    prices = [55.0, 49.0, np.nan, 5e-4, 6.0, 0.5, 750.0]
    flags = ["c", "c", "p", "p", "c", "c", "p"]
    np.testing.assert_array_equal(
        _mask(K, prices, flags),
        [True, False, False, False, True, False, False],
    )


def test_mask_rejects_zero_bids_and_wide_spreads_when_quotes_are_given():
    K = [690.0, 690.0, 690.0]
    prices = [5.0, 5.0, 5.0]
    bids = [4.9, 0.0, 3.0]
    asks = [5.1, 10.0, 7.0]  # Spreads: 0.2, 10, 4 -> relative 4%, 200%, 80%
    flags = ["p", "p", "p"]

    np.testing.assert_array_equal(_mask(K, prices, flags, bids=bids), [True, False, True])
    np.testing.assert_array_equal(
        _mask(K, prices, flags, bids=bids, asks=asks, max_rel_spread=0.25), [True, False, False]
    )
